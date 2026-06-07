"""
Copilot Studio 风格的多Agent编排 API (v3)

基于新的 OrchestratorAgent + LangGraph 编排工作流：
- classify → plan → delegate → verify → respond
- SSE 流式响应，显示完整 Agent 调用过程
- Human-in-the-Loop 支持
- 结构化日志与遥测

相比 v1 (手动编排) 和 v2 (线性LangGraph) 的改进：
- 真正的多Agent编排（Orchestrator-Subagent模式）
- Single Response Principle
- 完整的控制层（AI/Hybrid/Deterministic）
- Agent声明式注册与路由
"""

from __future__ import annotations

import logging
from typing import Optional
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from agents.orchestrator import orchestrator
from agents.graph_workflow import orchestration_runner
from agents.orchestrator.agent_registry import agent_registry
from agents.orchestrator.telemetry import telemetry
from agents.orchestrator.governance import audit_trail, AuditEvent, AuditEventType

logger = logging.getLogger("api.chat_handler_v3")

# 创建 v3 路由
router = APIRouter(prefix="/api/v3", tags=["Copilot Studio Orchestration"])


# ============================================================
# 请求/响应模型
# ============================================================

class ChatRequest(BaseModel):
    """聊天请求"""
    user_input: str = Field(..., description="用户输入文本")
    thread_id: str = Field(default="default", description="会话线程ID")
    user_id: str = Field(default="default_user", description="用户ID")
    debug_mode: bool = Field(default=False, description="是否开启调试模式")


class HumanReviewRequest(BaseModel):
    """人工审核决策请求"""
    thread_id: str = Field(..., description="会话线程ID")
    decision: str = Field(..., description="决策: approved | rejected")
    comment: str = Field(default="", description="审核备注")


class AgentInfo(BaseModel):
    """Agent信息"""
    agent_id: str
    name: str
    description: str
    capabilities: list[str]
    knowledge_domains: list[str]


class OrchestrationReport(BaseModel):
    """编排报告"""
    trace_id: str
    agents_called: list[str]
    total_duration_ms: float
    success: bool


# ============================================================
# API 端点
# ============================================================

@router.post("/chat", summary="多Agent编排聊天（SSE流式）")
async def chat_stream(request: ChatRequest):
    """
    Copilot Studio 风格的多Agent编排入口

    编排流程：
    1. classify - 意图分类
    2. plan - 任务规划
    3. delegate - 委派子Agent
    4. verify - 验证结果
    5. respond - 合成响应

    响应格式: Server-Sent Events (SSE)
    """
    logger.info(f"[API v3] 收到请求: thread={request.thread_id}, "
                f"input={request.user_input[:50]}...")

    # 开始遥测
    telemetry.start_orchestration(request.thread_id)

    # 审计事件：编排开始
    audit_trail.record(AuditEvent(
        event_id="",
        event_type=AuditEventType.ORCHESTRATION_START,
        trace_id=request.thread_id,
        agent_id="orchestrator",
        details={"user_input": request.user_input[:200]},
        user_id=request.user_id,
        session_id=request.thread_id,
    ))

    async def event_generator():
        """SSE 事件生成器"""
        try:
            # 使用编排工作流运行器处理
            async for token in orchestration_runner.run_stream(
                request.user_input,
                request.thread_id,
            ):
                # SSE 格式
                yield f"data: {token}\n\n"

            # 发送完成信号
            yield "data: [DONE]\n\n"

            # 审计事件：编排完成
            audit_trail.record(AuditEvent(
                event_id="",
                event_type=AuditEventType.ORCHESTRATION_END,
                trace_id=request.thread_id,
                agent_id="orchestrator",
                details={"status": "completed"},
                user_id=request.user_id,
                session_id=request.thread_id,
            ))

        except Exception as e:
            logger.error(f"[API v3] 编排失败: {e}")
            yield f"data: [ERROR] {str(e)}\n\n"
            yield "data: [DONE]\n\n"

            # 审计事件：编排失败
            audit_trail.record(AuditEvent(
                event_id="",
                event_type=AuditEventType.AGENT_FAILED,
                trace_id=request.thread_id,
                agent_id="orchestrator",
                details={"error": str(e)},
                user_id=request.user_id,
                session_id=request.thread_id,
            ))

        finally:
            # 结束遥测
            telemetry.end_orchestration(
                trace_id=request.thread_id,
                success=True,
            )

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@router.post("/chat/sync", summary="多Agent编排聊天（同步）")
async def chat_sync(request: ChatRequest) -> dict:
    """
    同步版本的编排聊天

    返回完整结果，适用于非流式场景。
    """
    logger.info(f"[API v3 sync] 收到请求: thread={request.thread_id}")

    try:
        result = await orchestration_runner.run(
            request.user_input,
            request.thread_id,
        )

        return {
            "success": True,
            "thread_id": request.thread_id,
            "intent": result.get("intent", ""),
            "urgency": result.get("urgency", ""),
            "response": result.get("final_response", ""),
            "agents_called": list(result.get("agent_results", {}).keys()),
        }

    except Exception as e:
        logger.error(f"[API v3 sync] 处理失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/chat/human-review", summary="提交人工审核决策")
async def submit_human_review(review: HumanReviewRequest) -> dict:
    """
    Human-in-the-Loop: 提交人工审核决策

    在编排流程中，高风险操作会暂停并等待此接口的调用。
    """
    logger.info(
        f"[API v3] 人工审核: thread={review.thread_id}, "
        f"decision={review.decision}"
    )

    valid_decisions = {"approved", "rejected"}
    if review.decision not in valid_decisions:
        raise HTTPException(
            status_code=400,
            detail=f"无效决策，支持: {valid_decisions}",
        )

    try:
        result = orchestration_runner.resume_with_decision(
            review.thread_id,
            review.decision,
        )

        # 审计事件
        audit_trail.record(AuditEvent(
            event_id="",
            event_type=AuditEventType.HUMAN_REVIEW_COMPLETED,
            trace_id=review.thread_id,
            agent_id="orchestrator",
            details={
                "decision": review.decision,
                "comment": review.comment,
            },
            session_id=review.thread_id,
        ))

        return {
            "success": True,
            "decision": review.decision,
            "message": f"审核已完成: {review.decision}",
        }

    except Exception as e:
        logger.error(f"[API v3] 人工审核处理失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/agents", summary="列出所有已注册的Agent")
async def list_agents() -> list[AgentInfo]:
    """返回所有已注册的专业Agent信息"""
    agents = agent_registry.list_agents()
    return [
        AgentInfo(
            agent_id=a["agent_id"],
            name=a["name"],
            description=a["description"],
            capabilities=a["capabilities"],
            knowledge_domains=a["knowledge_domains"],
        )
        for a in agents
        if a["agent_id"] != "orchestrator"
    ]


@router.get("/report", summary="获取编排系统运行报告")
async def get_orchestration_report() -> dict:
    """获取遥测和监控报告"""
    return telemetry.get_report()


@router.get("/trace/{thread_id}", summary="获取指定会话的审计追踪")
async def get_audit_trace(thread_id: str) -> dict:
    """获取指定thread的完整审计追踪"""
    summary = audit_trail.get_trace_summary(thread_id)
    return summary


@router.post("/reset/{thread_id}", summary="重置会话")
async def reset_conversation(thread_id: str) -> dict:
    """重置指定会话的编排状态"""
    orchestration_runner.reset(thread_id)
    # 清理相关上下文
    from agents.a2a.context_manager import context_manager
    from agents.a2a.message_bus import message_bus

    context_manager.clear_context(thread_id)
    message_bus.clear_trace(thread_id)
    audit_trail.clear_trace(thread_id)

    return {"success": True, "message": f"会话 {thread_id} 已重置"}

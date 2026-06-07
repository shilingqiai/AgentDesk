"""
Copilot Studio 多Agent编排工作流 — Hub & Spoke 三级路由 (v3)

架构：
    route → fast_track → END      (80% 知识查询 → EnterpriseRAGAgent)
          → action_track → END    (15% 工单派发 → TicketDispatchSubAgent)
          → complex_track → END   (5%  复合指令 → TaskPlanner)
          → clarification → END   (AI不确定 → 反问用户)

v3 改进：
    - 废除"伪 Agent"：it_consultant / hr_consultant / facilities 合并为 EnterpriseRAGAgent
    - 废除关键词硬匹配：Router 纯语义路由，不确定时返回 clarify 反问用户
    - fast_track 不再按 agent_id 分发，统一委派 EnterpriseRAGAgent (FAISS 跨领域检索)

流式输出：
    [THINKING] → 前端显示"思考中..."
    [ROUTE]    → 更新侧边栏路由轨道
    [CLARIFY]  → AI反问用户（需在聊天框上方显示反问内容）
    [STREAM]   → 逐字流式输出到对话气泡
    [DONE]     → 完成标记
"""

from __future__ import annotations

import asyncio
import logging
from typing import TypedDict, Literal, Annotated, Optional, AsyncGenerator
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph.message import add_messages
from langchain_core.messages import HumanMessage, AIMessage

logger = logging.getLogger("graph_workflow")


# ============================================================
# TicketState
# ============================================================

class TicketState(TypedDict):
    """工单编排状态"""
    messages: Annotated[list, add_messages]
    track: str
    agent_id: str
    intent: str
    urgency: str
    confidence: float
    plan: list[dict]
    current_step: int
    agent_results: dict
    needs_human_review: bool
    human_decision: Optional[str]
    final_response: str
    resolved: bool
    thread_id: str


def create_initial_state(user_input: str, thread_id: str = "default") -> TicketState:
    return TicketState(
        messages=[HumanMessage(content=user_input)],
        track="", agent_id="", intent="", urgency="medium",
        confidence=0.0, plan=[], current_step=0, agent_results={},
        needs_human_review=False, human_decision=None,
        final_response="", resolved=False, thread_id=thread_id,
    )


# ============================================================
# 辅助函数
# ============================================================

def _build_conversation_context(messages: list, max_turns: int = 5) -> str:
    """构建多轮对话上下文（最近 N 轮）"""
    if len(messages) <= 1:
        return ""
    recent = messages[-(max_turns * 2):]
    lines = []
    for msg in recent[:-1]:
        role = "用户" if isinstance(msg, HumanMessage) else "助手"
        content = msg.content[:300] if hasattr(msg, 'content') else str(msg)[:300]
        lines.append(f"{role}: {content}")
    return "\n".join(lines) if lines else ""


def _get_user_text(state: TicketState) -> str:
    """获取当前用户输入"""
    return state["messages"][-1].content


# ============================================================
# 工作流节点
# ============================================================

async def route_node(state: TicketState) -> TicketState:
    """语义路由 — LLM 判定轨道，低置信度 → clarify"""
    from agents.orchestrator.router import Router
    from agents.orchestrator.agent_registry import agent_registry

    router = Router()
    user_text = _get_user_text(state)
    agent_descriptions = agent_registry.get_routing_descriptions()
    conversation_history = _build_conversation_context(state["messages"])

    result = await router.route(user_text, agent_descriptions, conversation_history)

    state["track"] = result.track
    state["agent_id"] = result.agent_id
    state["intent"] = result.category
    state["urgency"] = result.urgency
    state["confidence"] = result.confidence
    state["resolved"] = False

    # 低置信度 → 强制反问
    if result.track != "clarify" and result.confidence < 0.7:
        logger.info(f"[Route] 置信度 {result.confidence:.0%} < 70%，强制转为 clarify")
        state["track"] = "clarify"
        state["agent_id"] = ""

    # 三层控制模型集成
    from agents.orchestrator.control_layers import control_manager

    action_type = "query"
    if result.track == "action":
        action_type = "create_ticket"
    elif result.track == "complex":
        action_type = "multi_step"

    control_decision = control_manager.evaluate(
        intent=result.category, urgency=result.urgency,
        action_type=action_type, confidence=result.confidence,
    )
    state["needs_human_review"] = control_decision.needs_human_review

    logger.info(f"[Route] track={state['track']}, confidence={result.confidence:.0%}, "
                f"reason={result.reason[:60]}")
    return state


async def fast_track_node(state: TicketState) -> TicketState:
    """
    极速通道 (80%)：统一委派 EnterpriseRAGAgent

    FAISS 向量检索自动跨领域匹配（IT+HR+行政），
    不再按 agent_id 分发到不同的伪 Agent。
    """
    from agents.orchestrator.agent_registry import agent_registry
    from agents.a2a.protocol import AgentMessage as AM
    from agents.a2a.message_bus import message_bus

    user_text = _get_user_text(state)
    conversation_history = _build_conversation_context(state["messages"])

    agent_instance = agent_registry.get_agent("enterprise_rag")
    if agent_instance is None:
        logger.error("[FastTrack] EnterpriseRAGAgent 未注册！")
        state["final_response"] = "系统初始化未完成，请稍后重试。"
        state["resolved"] = True
        return state

    delegation = AM.create_delegation(
        from_agent="orchestrator", to_agent="enterprise_rag",
        payload={
            "user_input": user_text,
            "task": "企业知识库问答",
            "intent_category": "fast",
            "conversation_history": conversation_history,
            "urgency": state.get("urgency", "low"),
        },
        trace_id=state.get("thread_id", ""),
    )

    try:
        result = await agent_instance.execute(delegation)
        message_bus.record(result)
        state["agent_results"]["enterprise_rag"] = {
            "success": result.success,
            "payload": result.payload,
            "error": result.error,
        }

        if result.success and result.payload.get("direct_response"):
            state["final_response"] = result.payload["direct_response"]
        elif result.error:
            state["final_response"] = (
                f"咨询处理异常：{result.error}\n\n请稍后重试或联系人工服务。"
            )
        else:
            state["final_response"] = "处理完成，如有疑问请联系人工服务。"
    except Exception as e:
        logger.error(f"[FastTrack] EnterpriseRAGAgent 执行失败: {e}")
        state["final_response"] = "抱歉，处理您的请求时出现了问题。请稍后重试。"

    state["resolved"] = True
    source_count = len(result.payload.get("sources", [])) if result.success else 0
    logger.info(f"[FastTrack] EnterpriseRAG 检索 {source_count} 篇，"
                f"响应 {len(state['final_response'])} 字")
    return state


async def action_track_node(state: TicketState) -> TicketState:
    """动作通道 (15%)：委派 TicketDispatchSubAgent 创建工单"""
    from agents.orchestrator.agent_registry import agent_registry
    from agents.a2a.protocol import AgentMessage as AM
    from agents.a2a.message_bus import message_bus

    user_text = _get_user_text(state)

    agent_instance = agent_registry.get_agent("ticket_dispatch")
    if agent_instance is None:
        logger.warning("[ActionTrack] TicketDispatch 未注册，降级为 fast")
        return await fast_track_node(state)

    delegation = AM.create_delegation(
        from_agent="orchestrator", to_agent="ticket_dispatch",
        payload={
            "user_input": user_text,
            "task": "提取参数并创建工单",
            "intent_category": "action",
            "urgency": state.get("urgency", "medium"),
        },
        trace_id=state.get("thread_id", ""),
    )

    try:
        result = await agent_instance.execute(delegation)
        message_bus.record(result)
        state["agent_results"]["ticket_dispatch"] = {
            "success": result.success, "payload": result.payload, "error": result.error,
        }

        if result.success and result.payload.get("direct_response"):
            state["final_response"] = result.payload["direct_response"]
        elif result.error:
            state["final_response"] = (
                f"操作未能完成：{result.error}\n\n"
                "请稍后重试，或拨打服务台热线获取人工支持。"
            )
        else:
            state["final_response"] = "操作已完成，如需查看详情请稍后查询。"
    except Exception as e:
        logger.error(f"[ActionTrack] 执行失败: {e}")
        state["final_response"] = "抱歉，执行操作时出现了问题。请稍后重试或联系服务台。"

    state["resolved"] = True
    return state


async def complex_track_node(state: TicketState) -> TicketState:
    """复杂通道 (5%)：Plan → 多Agent委派 → Synthesize"""
    from agents.orchestrator.task_planner import TaskPlanner
    from agents.orchestrator.agent_registry import agent_registry
    from agents.orchestrator.response_synthesizer import ResponseSynthesizer
    from agents.a2a.protocol import AgentMessage as AM
    from agents.a2a.message_bus import message_bus
    from agents.orchestrator.router import RouteResult
    from config.model_provider import create_chat_model

    user_text = _get_user_text(state)
    conversation_history = _build_conversation_context(state["messages"])
    llm = create_chat_model(model_type="main", temperature=0)

    intent_result = RouteResult(track="complex", reason=user_text[:100])
    planner = TaskPlanner(llm, agent_registry)

    plan_input = user_text
    if conversation_history:
        plan_input = f"对话历史:\n{conversation_history}\n\n当前输入: {user_text}"

    plan = await planner.plan(intent_result, plan_input)

    state["plan"] = [
        {"agent_id": s.agent_id, "task": s.task, "params": s.params, "depends_on": s.depends_on}
        for s in plan.steps
    ]
    state["needs_human_review"] = state.get("needs_human_review", False) or plan.needs_human_review

    agent_results = {}
    for step in plan.steps:
        agent_id = step.agent_id
        if not agent_id:
            continue
        agent_instance = agent_registry.get_agent(agent_id)
        if agent_instance is None:
            logger.warning(f"[ComplexTrack] Agent '{agent_id}' 未注册，跳过")
            continue

        delegation = AM.create_delegation(
            from_agent="orchestrator", to_agent=agent_id,
            payload={
                "user_input": user_text, "task": step.task,
                "params": step.params, "intent_category": "complex",
                "conversation_history": conversation_history,
            },
            trace_id=state.get("thread_id", ""),
        )
        try:
            result = await agent_instance.execute(delegation)
            message_bus.record(result)
            agent_results[agent_id] = {
                "success": result.success, "payload": result.payload, "error": result.error,
            }
        except Exception as e:
            logger.error(f"[ComplexTrack] Agent '{agent_id}' 失败: {e}")
            agent_results[agent_id] = {"success": False, "payload": {}, "error": str(e)}

    state["agent_results"] = agent_results

    # 合成
    agent_msgs = {}
    for aid, rdict in agent_results.items():
        agent_msgs[aid] = AM(
            from_agent=aid, to_agent="orchestrator",
            payload=rdict.get("payload", {}),
            success=rdict.get("success", False),
            error=rdict.get("error"),
        )

    synthesizer = ResponseSynthesizer(llm)
    response = await synthesizer.synthesize(agent_msgs, user_text)
    state["final_response"] = response
    state["resolved"] = True
    return state


async def clarification_node(state: TicketState) -> TicketState:
    """
    反问节点：AI 不确定用户意图时，主动反问澄清

    触发条件：
    - Router 返回 track="clarify"
    - Router 返回 confidence < 0.7（route_node 强制转为 clarify）
    - LLM JSON 解析失败
    """
    user_text = _get_user_text(state)
    confidence = state.get("confidence", 0)

    if confidence < 0.3:
        # 完全无法理解 → 通用引导
        state["final_response"] = (
            "抱歉，我不太确定您想做什么。\n\n"
            "您可以这样对我说：\n"
            "- 🔍 **查询知识**：'VPN怎么排查？''请假流程是什么？''食堂在哪？'\n"
            "- 📋 **提交工单**：'帮我提交一个网络故障工单'\n"
            "- 🏢 **行政服务**：'会议室怎么预定？''快递怎么寄？'\n\n"
            "请描述您的具体需求，我会尽力帮您解决。"
        )
    else:
        # 有一定理解但不确信 → 引导式反问
        state["final_response"] = (
            f"我不太确定您的具体需求，想跟您确认一下：\n\n"
            f"您是想要：\n"
            f"1. **查询相关信息**（如政策、流程、故障排查方法）？\n"
            f"2. **提交一个工单**（让工程师或HR处理）？\n\n"
            f"请告诉我具体内容，我会帮您处理。"
        )

    state["resolved"] = True
    logger.info(f"[Clarify] 反问用户 (confidence={confidence:.0%}, input=\"{user_text[:50]}\")")
    return state


async def respond_node(state: TicketState) -> TicketState:
    """最终响应节点：记录消息 + 人工审核拦截"""
    if not state.get("final_response"):
        state["final_response"] = "处理完成，如有疑问请咨询服务台。"

    if state.get("needs_human_review") and state.get("track") == "action":
        state["final_response"] = (
            "⚠️ 此操作需要人工审核确认。\n\n"
            f"{state['final_response']}\n\n"
            "---\n💡 工单已创建但需要管理员审核后才会派发。"
        )

    state["messages"].append(AIMessage(content=state["final_response"]))
    state["resolved"] = True
    return state


# ============================================================
# 路由函数
# ============================================================

def route_after_route(state: TicketState) -> Literal[
    "fast_track", "action_track", "complex_track", "clarification",
]:
    track = state.get("track", "clarification")
    if track == "fast":       return "fast_track"
    elif track == "action":   return "action_track"
    elif track == "complex":  return "complex_track"
    else:                     return "clarification"


# ============================================================
# 构建工作流图
# ============================================================

def build_orchestration_workflow() -> StateGraph:
    workflow = StateGraph(TicketState)

    workflow.add_node("route", route_node)
    workflow.add_node("fast_track", fast_track_node)
    workflow.add_node("action_track", action_track_node)
    workflow.add_node("complex_track", complex_track_node)
    workflow.add_node("clarification", clarification_node)
    workflow.add_node("respond", respond_node)

    workflow.set_entry_point("route")

    workflow.add_conditional_edges("route", route_after_route, {
        "fast_track": "fast_track",
        "action_track": "action_track",
        "complex_track": "complex_track",
        "clarification": "clarification",
    })

    workflow.add_edge("fast_track", "respond")
    workflow.add_edge("action_track", "respond")
    workflow.add_edge("complex_track", "respond")
    workflow.add_edge("clarification", "respond")
    workflow.add_edge("respond", END)

    return workflow


# ============================================================
# 工作流运行器 — 带流式输出
# ============================================================

STREAM_CHUNK_SIZE = 3
STREAM_DELAY = 0.025


class OrchestrationWorkflowRunner:
    """Hub & Spoke 编排工作流运行器 (v3)"""

    def __init__(self):
        self._ensure_agents_loaded()
        self.workflow = build_orchestration_workflow()
        self.checkpointer = MemorySaver()
        self.app = self.workflow.compile(checkpointer=self.checkpointer)

    @staticmethod
    def _ensure_agents_loaded():
        """确保子Agent模块已导入并注册"""
        try:
            import agents.sub_agents.enterprise_rag     # noqa: F401
            import agents.sub_agents.ticket_dispatch    # noqa: F401
        except ImportError as e:
            logger.warning(f"Agent 模块加载警告: {e}")

    async def run(self, user_input: str, thread_id: str = "default") -> TicketState:
        initial_state = create_initial_state(user_input, thread_id)
        config = {"configurable": {"thread_id": thread_id}}
        return await self.app.ainvoke(initial_state, config)

    async def run_stream(
        self, user_input: str, thread_id: str = "default",
    ) -> AsyncGenerator[str, None]:
        """
        流式运行编排工作流。

        输出令牌：
          [THINKING] <文字>            — 更新"思考中..."文字
          [ROUTE] <轨道描述>           — 路由判定结果
          [CLARIFY] <反问文字>         — AI 反问用户澄清意图
          [FAST]/[ACTION]/[COMPLEX]    — 轨道入口
          [STREAM]<文字片段>           — 流式回答片段
          [DONE]                       — 完成
        """
        initial_state = create_initial_state(user_input, thread_id)
        config = {"configurable": {"thread_id": thread_id}}

        yield "[THINKING] 🔍 正在分析您的问题...\n"

        async for event in self.app.astream(initial_state, config):
            for node_name, node_state in event.items():

                # --- 路由结果 ---
                if node_name == "route":
                    track = node_state.get("track", "")
                    confidence = node_state.get("confidence", 0)

                    if track == "clarify":
                        yield "[ROUTE] 🤔 AI 需要确认您的意图\n"
                        yield "[THINKING] 💭 正在组织追问...\n"
                    else:
                        track_names = {
                            "fast": "🔍 极速通道 · 企业知识库问答",
                            "action": "⚡ 动作通道 · 工单派发",
                            "complex": "🧩 复杂通道 · 多步骤编排",
                        }
                        yield f"[ROUTE] {track_names.get(track, track)}\n"

                        track_thinking = {
                            "fast": "📚 向量检索知识库，生成回答...",
                            "action": "⚡ 分析工单需求，提取参数...",
                            "complex": "🧩 分析复合指令，制定计划...",
                        }
                        yield f"[THINKING] {track_thinking.get(track, '处理中...')}\n"

                    if node_state.get("needs_human_review"):
                        yield "[THINKING] ⚠️ 此操作可能需要人工审核...\n"

                # --- 轨道入口 ---
                if node_name == "fast_track":
                    yield "[FAST] 📚 企业知识库检索 · EnterpriseRAG\n"
                    yield "[THINKING] ✍️ 正在基于检索结果生成回答...\n"
                elif node_name == "action_track":
                    yield "[ACTION] ⚡ 动作通道 · 工单派发\n"
                    yield "[THINKING] 📝 正在创建工单...\n"
                elif node_name == "complex_track":
                    yield "[COMPLEX] 🧩 复杂通道 · 多步骤编排\n"
                    yield "[THINKING] 🔗 委派多个Agent协作处理...\n"
                elif node_name == "clarification":
                    yield "[CLARIFY] 🤔 AI 需要确认您的意图\n"
                    yield "[THINKING] 💬 正在生成反问引导...\n"

                # --- 流式输出 ---
                if node_name in ("fast_track", "action_track", "complex_track", "clarification"):
                    resp = node_state.get("final_response", "")
                    if resp:
                        for i in range(0, len(resp), STREAM_CHUNK_SIZE):
                            chunk = resp[i:i + STREAM_CHUNK_SIZE]
                            yield f"[STREAM]{chunk}\n"
                            await asyncio.sleep(STREAM_DELAY)

        yield "[DONE]\n"

    def get_state(self, thread_id: str = "default") -> Optional[TicketState]:
        config = {"configurable": {"thread_id": thread_id}}
        return self.app.get_state(config)

    def reset(self, thread_id: str = "default"):
        config = {"configurable": {"thread_id": thread_id}}
        self.app.update_state(config, None)


# 全局实例
orchestration_runner = OrchestrationWorkflowRunner()
workflow_runner = orchestration_runner

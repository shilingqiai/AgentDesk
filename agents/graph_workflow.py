"""
Copilot Studio 多Agent编排工作流 — Hub & Spoke 三级路由 (v4)

架构：
    phase=initial:
        route → fast_track → END      (80% 知识查询 → EnterpriseRAGAgent)
              → action_track → END    (15% 工单派发 → TicketDispatchSubAgent)
              → complex_track → END   (5%  复合指令 → TaskPlanner)
              → clarification → END   (AI不确定 → 反问用户)

    phase=self_help_provided:
        route → re_evaluate → {
            escalation → action_track → END (升级为工单)
            follow_up  → fast_track(带上下文) → END (追问细节)
            new_topic  → route (清状态重路由)
            confirm    → END (清状态结束)
        }

v4 改进：
    - 对话阶段追踪：initial / self_help_provided，防止 RAG 死循环
    - re_evaluate_node: LLM 语义判断用户对上一轮方案的态度（升级/追问/换话题/确认）
    - RAG 去重：follow_up 时注入 last_rag_topic 上下文，不重复相同方案
    - 状态清理：new_topic / confirm / escalation 出口自动清 self-help 上下文

流式输出：
    [THINKING] → 前端显示"思考中..."
    [ROUTE]    → 更新侧边栏路由轨道
    [CLARIFY]  → AI反问用户（需在聊天框上方显示反问内容）
    [STREAM]   → 逐字流式输出到对话气泡
    [DONE]     → 完成标记
"""

from __future__ import annotations

import asyncio
import json
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
    pending_card_type: str    # "" = 无锁, "admin"/"leave"/"expense"/"it_fault" = 卡片锁定中
    re_route: bool            # True = action_track 处理完回 Router 重路由
    # v4: 对话阶段追踪
    conversation_phase: str   # "initial" | "self_help_provided"
    last_rag_topic: str       # 上一轮 RAG 回答的主题（如 "VPN故障排查"）
    last_rag_summary: str     # 上一轮 RAG 回答的核心内容（100字摘要）
    # v5: 用户身份
    user_name: str            # 当前用户姓名（如 "张三"）
    role: str                 # 角色: "employee" | "admin"
    # v6: 并行节点结果隔离（防 Race Condition）
    parallel_rag_result: dict     # RAG 查询结果（complex track 并行）
    parallel_tool_result: dict    # ToolAgent 查询结果（complex track 并行）


def create_initial_state(user_input: str, thread_id: str = "default",
                        user_name: str = "", role: str = "employee") -> TicketState:
    return TicketState(
        messages=[HumanMessage(content=user_input)],
        track="", agent_id="", intent="", urgency="medium",
        confidence=0.0, plan=[], current_step=0, agent_results={},
        needs_human_review=False, human_decision=None,
        final_response="", resolved=False, thread_id=thread_id,
        pending_card_type="", re_route=False,
        conversation_phase="initial", last_rag_topic="", last_rag_summary="",
        user_name=user_name, role=role,
        parallel_rag_result={}, parallel_tool_result={},
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


def _reset_self_help_state(state: TicketState) -> None:
    """清空 self-help 追踪状态（防幽灵上下文）"""
    state["conversation_phase"] = "initial"
    state["last_rag_topic"] = ""
    state["last_rag_summary"] = ""


def _generate_rag_topic(user_input: str, response: str) -> str:
    """从用户输入和 RAG 回答中提取简要主题（规则兜底，不调 LLM）"""
    # 取用户输入的前 20 字作为主题标签
    topic = user_input[:20].replace("\n", " ").strip()
    return topic if topic else "企业服务咨询"


# ============================================================
# 工作流节点
# ============================================================

async def route_node(state: TicketState) -> TicketState:
    """
    语义路由 — LLM 判定轨道，低置信度 → clarify。

    短路规则（按优先级）：
    1. 卡片锁 pending_card → action_track
    2. self_help_provided 阶段 → re_evaluate_node（不调 Router）
    3. 正常 Router 判定
    """
    from agents.orchestrator.router import Router
    from agents.orchestrator.agent_registry import agent_registry

    # ── 短路 1: 卡片锁 ──
    pending = state.get("pending_card_type", "")
    if pending:
        logger.info(f"[Route] 卡片锁 pending_card={pending}，短路 Router → action_create")
        state["track"] = "action_create"
        state["agent_id"] = "ticket_dispatch"
        state["confidence"] = 1.0
        state["intent"] = pending
        state["resolved"] = False
        return state

    # ── 短路 2: self_help_provided 阶段 → re_evaluate ──
    if state.get("conversation_phase") == "self_help_provided":
        logger.info(
            f"[Route] phase=self_help_provided, topic={state.get('last_rag_topic')}, "
            f"短路 Router → re_evaluate"
        )
        state["track"] = "re_evaluate"
        state["resolved"] = False
        return state

    # ── 正常 Router 判定 ──
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
    if result.track == "action_query":
        action_type = "query"
    elif result.track == "action_create":
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


async def re_evaluate_node(state: TicketState) -> TicketState:
    """
    重新评估节点 (v4)：self_help_provided 阶段，用户回复后判断意图。

    用 LLM 语义理解用户对上一轮方案的反馈：
      - escalation → 方案无效，需要工单/人工（强制 action_track）
      - follow_up → 追问细节，仍走 RAG（带 last_rag_topic 防重复）
      - new_topic → 换话题（清状态，回 route_node 重路由）
      - confirm → 问题解决（清状态，结束）
    """
    from config.model_provider import create_chat_model
    import json as _json

    user_text = _get_user_text(state)
    topic = state.get("last_rag_topic", "企业服务")
    summary = state.get("last_rag_summary", "")[:150]

    system_prompt = (
        "你是对话意图分类器。上一轮助手针对「{topic}」提供了方案，用户刚回复了一句话。\n"
        "请判断用户意图（返回 JSON）：\n\n"
        "**escalation** — 方案无效，需升级为工单/人工\n"
        "  - 直接否定：'还是不行''没用''试了但没解决''按照做了但没用'\n"
        "  - 间接否定：'这些我都知道''检查过了都正常''版本已经最新'\n"
        "  - 沮丧表达：'搞不定''放弃了''帮我找人'\n"
        "  - 嫌弃方案：'太麻烦了，有别的办法吗'（若知识库可能无替代方案→escalation）\n\n"
        "**follow_up** — 围绕同一问题追问，未否定方案\n"
        "  - 要细节：'第二步怎么操作''客户端在哪下载'\n"
        "  - 扩展询问：'还有其他排查方法吗''有没有简单点的办法'\n"
        "  关键：不表达'方案无效'，只是要更多信息\n\n"
        "**new_topic** — 完全切换话题，不涉及对上一轮方案的评价\n"
        "  - '帮我查请假政策''食堂怎么走''报销流程'\n"
        "  - '算了先不管了，帮我查报销' ← 主动放弃当前问题\n\n"
        "**confirm** — 问题已解决\n"
        "  - '好了''可以了''解决了谢谢''原来是我密码错了'\n\n"
        "注意：一旦判定 new_topic 或 confirm，说明当前上下文已结束，JSON 中不要引用旧方案。"
    ).replace("{topic}", topic)

    try:
        llm = create_chat_model(model_type="main", temperature=0)
        response = await llm.ainvoke([
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": (
                f"上一轮方案摘要：{summary}\n\n"
                f"用户回复：\"{user_text}\"\n\n"
                f"返回 JSON：{{\"intent\":\"escalation|follow_up|new_topic|confirm\",\"reason\":\"...\"}}"
            )},
        ])

        # 解析
        text = response.content.strip()
        try:
            data = _json.loads(text)
        except _json.JSONDecodeError:
            import re
            m = re.search(r'\{[\s\S]*\}', text)
            data = _json.loads(m.group(0)) if m else {}

        intent = data.get("intent", "escalation")
        reason = data.get("reason", "")

        if intent not in ("escalation", "follow_up", "new_topic", "confirm"):
            intent = "escalation"  # 兜底：不确定时升级

        logger.info(
            f"[ReEvaluate] intent={intent} reason={reason[:60]} "
            f"input={user_text[:50]}"
        )

        state["agent_results"]["re_evaluate"] = {
            "intent": intent, "reason": reason,
        }

        if intent == "escalation":
            # 强制走 action_create 创建工单
            state["track"] = "action_create"
            state["agent_id"] = "ticket_dispatch"
            state["intent"] = "ticket_request"
            state["urgency"] = "high"
            state["confidence"] = 0.9
            state["resolved"] = False
            # 不清 state — action_track 执行成功后清

        elif intent == "follow_up":
            # 仍走 fast_track，但带上下文防重复
            state["track"] = "fast"
            state["agent_id"] = "enterprise_rag"
            state["intent"] = "knowledge_query"
            state["confidence"] = 0.85
            state["resolved"] = False
            # follow_up 时不清 state — fast_track 需要 last_rag_topic

        elif intent == "new_topic":
            # 清状态，走 route 重路由
            _reset_self_help_state(state)
            state["track"] = ""
            state["agent_id"] = ""
            state["resolved"] = False
            state["re_route"] = True

        elif intent == "confirm":
            # 清状态，直接结束
            _reset_self_help_state(state)
            state["final_response"] = "很高兴能帮到您！还有其他问题随时问我。"
            state["resolved"] = True

    except Exception as e:
        logger.error(f"[ReEvaluate] LLM 调用失败: {e}，兜底为 escalation")
        state["track"] = "action_create"
        state["agent_id"] = "ticket_dispatch"
        state["intent"] = "ticket_request"
        state["urgency"] = "high"
        state["confidence"] = 0.7
        state["resolved"] = False

    return state


async def fast_track_node(state: TicketState) -> TicketState:
    """
    极速通道 (80%)：EnterpriseRAG — 真流式 token 输出 + 状态持久化

    v4 流式：通过 LangGraph StreamWriter 发射每个 token，
    run_stream 端以 stream_mode=["values","custom"] 接收并转为 [STREAM] 标签。
    """
    from agents.orchestrator.agent_registry import agent_registry
    from agents.a2a.message_bus import message_bus
    from langgraph.config import get_stream_writer

    user_text = _get_user_text(state)
    conversation_history = _build_conversation_context(state["messages"])
    phase = state.get("conversation_phase", "initial")

    agent_instance = agent_registry.get_agent("enterprise_rag")
    if agent_instance is None:
        logger.error("[FastTrack] EnterpriseRAGAgent 未注册！")
        state["final_response"] = "系统初始化未完成，请稍后重试。"
        state["resolved"] = True
        return state

    try:
        await agent_instance._ensure_initialized()

        # 检索
        docs = await agent_instance.knowledge_service.search(user_text, top_k=5)

        writer = get_stream_writer()
        full_response = ""

        if not docs:
            full_response = (
                "抱歉，我在知识库中没有找到相关的信息。\n\n"
                "建议提交工单让工程师协助处理。"
            )
            writer(full_response)
        else:
            # v4 follow_up 上下文注入
            if phase == "self_help_provided":
                topic = state.get("last_rag_topic", "")
                conversation_history = (
                    f"用户对上一轮「{topic}」方案提出了追问。\n{conversation_history}"
                )

            # 真流式 token 发射
            async for token in agent_instance._synthesize_stream(
                user_text, docs, conversation_history,
            ):
                full_response += token
                writer(token)

        # 流式完成后设置状态
        state["final_response"] = full_response.strip()
        source_list = [{"category": d.get("category", ""), "score": d.get("score", 0)}
                       for d in docs] if docs else []
        state["agent_results"]["enterprise_rag"] = {
            "success": True,
            "payload": {"direct_response": full_response.strip(), "sources": source_list},
            "error": None,
        }
        state["conversation_phase"] = "self_help_provided"
        state["last_rag_topic"] = _generate_rag_topic(user_text, full_response)
        state["last_rag_summary"] = full_response[:150]
        state["resolved"] = True
        logger.info(
            f"[FastTrack] phase → self_help_provided, "
            f"topic={state['last_rag_topic']}, "
            f"tokens={len(full_response)}"
        )

    except Exception as e:
        logger.error(f"[FastTrack] 执行失败: {e}")
        state["final_response"] = "抱歉，处理您的请求时出现了问题。请稍后重试。"
        state["resolved"] = True

    return state


async def action_track_node(state: TicketState) -> TicketState:
    """动作通道 (15%)：委派 TicketDispatchSubAgent。卡片锁期间做意图分类。"""
    from agents.orchestrator.agent_registry import agent_registry
    from agents.a2a.protocol import AgentMessage as AM
    from agents.a2a.message_bus import message_bus

    user_text = _get_user_text(state)
    pending = state.get("pending_card_type", "")

    agent_instance = agent_registry.get_agent("ticket_dispatch")
    if agent_instance is None:
        logger.warning("[ActionTrack] TicketDispatch 未注册，降级为 fast")
        state["pending_card_type"] = ""
        state["re_route"] = False
        return await fast_track_node(state)

    # ================================================================
    # 卡片锁模式：用户回复已存在的卡片 → LLM 分类意图
    # ================================================================
    if pending:
        prev_card = state.get("agent_results", {}).get("ticket_dispatch", {}).get("card", {})

        try:
            intent = await agent_instance.classify_card_response(
                user_text=user_text, card=prev_card, ticket_type=pending,
            )
        except Exception as e:
            logger.error(f"[ActionTrack] 意图分类失败: {e}，fallback → confirm")
            intent = "confirm"

        logger.info(
            f"[ActionTrack] 卡片锁 pending={pending} intent={intent} "
            f"input={user_text[:50]}"
        )

        if intent == "confirm":
            try:
                result_text = await agent_instance.execute_card(prev_card, user_text)
                state["final_response"] = result_text
            except Exception as e:
                logger.error(f"[ActionTrack] 卡片执行失败: {e}")
                state["final_response"] = f"操作失败：{e}\n请稍后重试。"
            state["pending_card_type"] = ""
            state["re_route"] = False

        elif intent == "modify":
            try:
                new_card = await agent_instance.rebuild_card(
                    prev_card, user_text, pending,
                )
                import json as _json
                desc = new_card.get("description", "")
                state["final_response"] = (
                    "📋 **已根据您的要求更新：**\n\n"
                    + desc
                    + "\n[CARD]" + _json.dumps(new_card, ensure_ascii=False)
                )
                state["agent_results"]["ticket_dispatch"] = {
                    "success": True, "payload": {}, "error": None,
                    "card": new_card,
                }
                # pending_card_type 保持，不下锁
            except Exception as e:
                logger.error(f"[ActionTrack] 卡片重建失败: {e}")
                state["final_response"] = f"无法更新卡片：{e}"
                state["pending_card_type"] = ""
            state["re_route"] = False

        elif intent == "cancel":
            state["final_response"] = "好的，已取消。还有其他需要帮您的吗？"
            state["pending_card_type"] = ""
            state["re_route"] = False

        elif intent == "new_topic":
            state["pending_card_type"] = ""
            state["re_route"] = True
            # 清 self-help 状态 — 用户主动放弃当前话题
            _reset_self_help_state(state)
            # 不设 final_response — 后续重路由的节点会填

        state["resolved"] = True
        return state

    # ================================================================
    # 非锁模式：action_query → ToolAgent | action_create → TicketDispatch
    # ================================================================

    track = state.get("track", "action_create")

    # ── action_query: 纯数据查询 → ToolAgent ──
    if track == "action_query":
        from agents.sub_agents.tool_agent import ToolAgent

        tool_agent = ToolAgent()
        tool_msg = AM.create_delegation(
            from_agent="orchestrator", to_agent="tool_agent",
            payload={
                "user_input": user_text,
                "task": f"查询用户 {state.get('user_name', '')} 的请求数据",
                "intent_category": "action_query",
                "user_name": state.get("user_name", ""),
            },
            trace_id=state.get("thread_id", ""),
        )

        try:
            result = await tool_agent.execute(tool_msg)
            message_bus.record(result)
            state["agent_results"]["tool_agent"] = {
                "success": result.success, "payload": result.payload, "error": result.error,
            }

            if result.success and result.payload.get("direct_response"):
                state["final_response"] = result.payload["direct_response"]
            elif result.error:
                state["final_response"] = (
                    f"查询未能完成：{result.error}\n\n"
                    "请稍后重试，或拨打服务台热线获取人工支持。"
                )
            else:
                state["final_response"] = "查询完成，如需进一步操作请告诉我。"
        except Exception as e:
            logger.error(f"[ActionTrack:query] ToolAgent 执行失败: {e}")
            state["final_response"] = "抱歉，查询数据时出现了问题。请稍后重试或联系服务台。"

        state["resolved"] = True
        state["re_route"] = False
        return state

    # ── action_create: 创建工单 / escalation 升级 → TicketDispatch ──

    # v4: 如果是 escalation 路径（re_evaluate → action），生成升级上下文
    escalation_context = ""
    if state.get("conversation_phase") == "self_help_provided":
        topic = state.get("last_rag_topic", "")
        summary = state.get("last_rag_summary", "")[:100]
        escalation_context = (
            f"用户之前咨询「{topic}」，排障方案未能解决问题，现升级为工单。"
            f"上一轮方案摘要：{summary}"
        )

    delegation = AM.create_delegation(
        from_agent="orchestrator", to_agent="ticket_dispatch",
        payload={
            "user_input": (
                f"{escalation_context}\n用户原始输入：{user_text}"
                if escalation_context else user_text
            ),
            "task": "提取参数并创建工单",
            "intent_category": "action",
            "urgency": state.get("urgency", "medium"),
            "user_id": state.get("user_name", ""),
            "user_name": state.get("user_name", ""),
            "role": state.get("role", "employee"),
        },
        trace_id=state.get("thread_id", ""),
    )

    try:
        result = await agent_instance.execute(delegation)
        message_bus.record(result)
        state["agent_results"]["ticket_dispatch"] = {
            "success": result.success, "payload": result.payload, "error": result.error,
        }

        if result.success and result.payload.get("return_card"):
            # 确认卡片模式
            import json as _json
            card = result.payload.get("card", {})
            ticket_type = result.payload.get("ticket_type", "")
            state["final_response"] = (
                "📋 **请确认以下信息**\n\n"
                + card.get("description", "")
                + "\n[CARD]" + _json.dumps(card, ensure_ascii=False)
            )
            state["agent_results"]["ticket_dispatch"] = {
                "success": True, "payload": result.payload, "error": None,
                "card": card,
            }
            # 设置卡片锁：下一轮文本输入将短路 Router
            state["pending_card_type"] = ticket_type
            logger.info(f"[ActionTrack] 设置卡片锁 pending_card_type={ticket_type}")
            # 清 self-help 状态 — 升级已完成
            _reset_self_help_state(state)
        elif result.success and result.payload.get("direct_response"):
            state["final_response"] = result.payload["direct_response"]
            # 清 self-help 状态 — 工单已创建
            _reset_self_help_state(state)
        elif result.error:
            state["final_response"] = (
                f"操作未能完成：{result.error}\n\n"
                "请稍后重试，或拨打服务台热线获取人工支持。"
            )
        else:
            state["final_response"] = "操作已完成，如需查看详情请稍后查询。"
            _reset_self_help_state(state)
    except Exception as e:
        logger.error(f"[ActionTrack] 执行失败: {e}")
        state["final_response"] = "抱歉，执行操作时出现了问题。请稍后重试或联系服务台。"

    state["resolved"] = True
    state["re_route"] = False
    return state


async def complex_track_node(state: TicketState) -> TicketState:
    """
    复杂通道 (5%)：Plan → 多Agent委派 → Synthesize

    v6: 对请假/报销场景实现并行编排（RAG ∥ ToolAgent → TicketDispatch），
    其他场景走串行 TaskPlanner 路径。
    并行结果通过 Python 变量传递（不写 LangGraph State），避免 Race Condition。
    """
    from agents.orchestrator.task_planner import TaskPlanner
    from agents.orchestrator.agent_registry import agent_registry
    from agents.orchestrator.response_synthesizer import ResponseSynthesizer
    from agents.a2a.protocol import AgentMessage as AM
    from agents.a2a.message_bus import message_bus
    from agents.orchestrator.router import RouteResult
    from config.model_provider import create_chat_model

    user_text = _get_user_text(state)
    conversation_history = _build_conversation_context(state["messages"])
    user_name = state.get("user_name", "")
    llm = create_chat_model(model_type="main", temperature=0)

    # ================================================================
    # v6: 请假/报销 → 并行 DAG 模式
    # ================================================================
    # v6.1 正则匹配 "请X天假" "休假" 等模式（防"请4天假"被"请假"子串遗漏）
    import re as _re
    _leave_pattern = _re.compile(r'请.*?假|休.*?假|年假|病假|事假|调休|婚假|产假|天假')
    is_leave = bool(_leave_pattern.search(user_text))
    expense_keywords = ["报销", "差旅", "费用"]
    is_expense = any(kw in user_text for kw in expense_keywords)

    if is_leave or is_expense:
        logger.info(
            f"[ComplexTrack:v6] 检测到 {'请假' if is_leave else '报销'}请求，"
            f"启用并行 DAG 模式"
        )

        try:
            # ── Step 1+2: 并行 RAG查政策 + ToolAgent查余额 ──
            rag_agent = agent_registry.get_agent("enterprise_rag")
            tool_agent = agent_registry.get_agent("tool_agent")
            ticket_agent = agent_registry.get_agent("ticket_dispatch")

            if not rag_agent or not tool_agent or not ticket_agent:
                logger.error("[ComplexTrack:v6] Agent 未就绪，降级为串行")
                state["final_response"] = "系统初始化未完成，请稍后重试。"
                state["resolved"] = True
                return state

            await rag_agent._ensure_initialized()

            async def _rag_policy():
                """并行任务 A: 查询政策"""
                docs = await rag_agent.knowledge_service.search(user_text, top_k=3)
                if docs:
                    doc_context = rag_agent._build_doc_context(docs)
                    policy_text = await rag_agent._synthesize(
                        user_text, docs, conversation_history,
                    )
                    return {
                        "success": True,
                        "policy_text": policy_text,
                        "sources": [
                            {"category": d.get("category", ""), "score": d.get("score", 0)}
                            for d in docs
                        ],
                    }
                return {"success": True, "policy_text": "", "sources": []}

            async def _tool_balance():
                """并行任务 B: 查询员工余额"""
                from agents.sub_agents.tool_agent import ToolAgent
                ta = ToolAgent()
                msg = AM.create_delegation(
                    from_agent="orchestrator", to_agent="tool_agent",
                    payload={
                        "user_input": f"查询员工 {user_name} 的{'年假' if is_leave else '相关'}余额",
                        "task": f"查询 {user_name} 的余额数据",
                    },
                )
                result = await ta.execute(msg)
                if result.success:
                    return {
                        "success": True,
                        "tool_result": result.payload.get("tool_result", {}),
                        "direct_response": result.payload.get("direct_response", ""),
                    }
                return {"success": False, "error": result.error}

            # 并行执行（Python 变量传递，不写 LangGraph State）
            policy_result, balance_result = await asyncio.gather(
                _rag_policy(), _tool_balance(),
            )

            logger.info(
                f"[ComplexTrack:v6] 并行完成 — "
                f"policy={policy_result['success']}, balance={balance_result['success']}"
            )

            # 存入 state（并行完成后安全写入）
            state["parallel_rag_result"] = policy_result
            state["parallel_tool_result"] = balance_result

            # ── Step 3: TicketDispatch 合规检查 + 生成卡片 ──
            policy_text = policy_result.get("policy_text", "")
            balance_data = balance_result.get("tool_result", {})
            balance_text = balance_result.get("direct_response", "")

            compliance_input = (
                f"{user_text}\n\n"
                f"[系统预查询结果]\n"
                f"政策信息：{policy_text[:500] if policy_text else '未查到相关政策'}\n"
                f"员工余额：{balance_text if balance_text else '未查到余额数据'}\n"
                f"余额数据(JSON)：{json.dumps(balance_data, ensure_ascii=False) if balance_data else '无'}"
            )

            ticket_msg = AM.create_delegation(
                from_agent="orchestrator", to_agent="ticket_dispatch",
                payload={
                    "user_input": compliance_input,
                    "task": "合规检查并生成确认卡片",
                    "intent_category": "complex",
                    "urgency": state.get("urgency", "medium"),
                    "user_id": user_name,
                    "user_name": user_name,
                    "role": state.get("role", "employee"),
                    "pre_checked": True,
                    "policy_result": policy_result,
                    "balance_result": balance_result,
                },
                trace_id=state.get("thread_id", ""),
            )

            result = await ticket_agent.execute(ticket_msg)
            message_bus.record(result)

            state["agent_results"] = {
                "enterprise_rag": {"success": policy_result["success"],
                                   "payload": policy_result, "error": None},
                "tool_agent": {"success": balance_result["success"],
                               "payload": balance_result, "error": None},
                "ticket_dispatch": {"success": result.success,
                                    "payload": result.payload, "error": result.error},
            }

            if result.success and result.payload.get("return_card"):
                import json as _json
                card = result.payload.get("card", {})
                ticket_type = result.payload.get("ticket_type", "leave" if is_leave else "expense")
                state["final_response"] = (
                    "📋 **请确认以下信息**\n\n"
                    + card.get("description", "")
                    + "\n[CARD]" + _json.dumps(card, ensure_ascii=False)
                )
                state["agent_results"]["ticket_dispatch"]["card"] = card
                state["pending_card_type"] = ticket_type
                logger.info(f"[ComplexTrack:v6] 设置卡片锁 pending_card_type={ticket_type}")
            elif result.success and result.payload.get("direct_response"):
                state["final_response"] = result.payload["direct_response"]
            elif result.error:
                state["final_response"] = (
                    f"合规检查未能完成：{result.error}\n\n"
                    "请稍后重试，或拨打服务台热线获取人工支持。"
                )
            else:
                state["final_response"] = "申请已提交，如需查看详情请稍后查询。"

        except Exception as e:
            logger.error(f"[ComplexTrack:v6] 并行 DAG 执行失败: {e}，降级为串行")
            # 降级：直接走 TicketDispatch
            ticket_agent = agent_registry.get_agent("ticket_dispatch")
            if ticket_agent:
                try:
                    fallback_msg = AM.create_delegation(
                        from_agent="orchestrator", to_agent="ticket_dispatch",
                        payload={
                            "user_input": user_text,
                            "task": "提取参数并创建工单",
                            "intent_category": "complex",
                            "urgency": state.get("urgency", "medium"),
                            "user_id": user_name,
                            "user_name": user_name,
                            "role": state.get("role", "employee"),
                        },
                        trace_id=state.get("thread_id", ""),
                    )
                    result = await ticket_agent.execute(fallback_msg)
                    state["agent_results"]["ticket_dispatch"] = {
                        "success": result.success, "payload": result.payload, "error": result.error,
                    }
                    state["final_response"] = (
                        result.payload.get("direct_response", "申请已提交。")
                        if result.success
                        else f"操作失败：{result.error}"
                    )
                except Exception as e2:
                    state["final_response"] = f"申请处理失败：{e2}"

        state["resolved"] = True
        return state

    # ================================================================
    # 非请假/报销场景：串行 TaskPlanner 路径（原有逻辑）
    # ================================================================

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

    # v6.1: 检测 TicketDispatch 是否返回了确认卡片
    td_result = agent_results.get("ticket_dispatch", {})
    if td_result.get("success") and td_result.get("payload", {}).get("return_card"):
        import json as _json
        card = td_result["payload"].get("card", {})
        ticket_type = td_result["payload"].get("ticket_type", "")
        state["final_response"] = (
            "📋 **请确认以下信息**\n\n"
            + card.get("description", "")
            + "\n[CARD]" + _json.dumps(card, ensure_ascii=False)
        )
        state["agent_results"]["ticket_dispatch"]["card"] = card
        state["pending_card_type"] = ticket_type
        logger.info(f"[ComplexTrack:serial] 设置卡片锁 pending_card_type={ticket_type}")
    else:
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

    v4: self_help_provided 阶段的反问带上下文，不再问通用的"查还是办"。
    """
    user_text = _get_user_text(state)
    confidence = state.get("confidence", 0)
    topic = state.get("last_rag_topic", "")

    # v4: 有上下文时反问更精准
    if topic and state.get("conversation_phase") == "self_help_provided":
        state["final_response"] = (
            f"关于「{topic}」的方案似乎没有解决您的问题。您是想要：\n\n"
            f"1. 我再提供其他思路？\n"
            f"2. 直接提交工单让工程师处理？\n\n"
            f"请告诉我您的想法。"
        )
    elif confidence < 0.3:
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

    if state.get("needs_human_review") and state.get("track") == "action_create":
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
    "re_evaluate", "fast_track", "action_track", "complex_track", "clarification",
]:
    """路由分发 — v6 增加 action_query / action_create 分支"""
    track = state.get("track", "clarification")
    if track == "re_evaluate":     return "re_evaluate"
    if track == "fast":            return "fast_track"
    elif track == "action_query":  return "action_track"
    elif track == "action_create": return "action_track"
    elif track == "complex":       return "complex_track"
    else:                          return "clarification"


def after_re_evaluate(state: TicketState) -> Literal[
    "action_track", "fast_track", "route", "respond",
]:
    """
    re_evaluate 后的分发 (v4):
      - escalation → action_track
      - follow_up → fast_track（带上下文）
      - new_topic → route（清状态重路由）
      - confirm → respond（清状态结束）
    """
    intent = state.get("agent_results", {}).get("re_evaluate", {}).get("intent", "escalation")
    if intent == "escalation":
        return "action_track"
    elif intent == "follow_up":
        return "fast_track"
    elif intent == "new_topic":
        return "route"
    else:  # confirm
        return "respond"


def after_action_track(state: TicketState) -> Literal["route", "respond"]:
    """卡片锁 / new_topic 时重路由，其他情况正常结束"""
    if state.get("re_route"):
        logger.info("[Graph] action_track → route (re_route)")
        return "route"
    return "respond"


# ============================================================
# 构建工作流图
# ============================================================

def build_orchestration_workflow() -> StateGraph:
    workflow = StateGraph(TicketState)

    workflow.add_node("route", route_node)
    workflow.add_node("re_evaluate", re_evaluate_node)
    workflow.add_node("fast_track", fast_track_node)
    workflow.add_node("action_track", action_track_node)
    workflow.add_node("complex_track", complex_track_node)
    workflow.add_node("clarification", clarification_node)
    workflow.add_node("respond", respond_node)

    workflow.set_entry_point("route")

    # route → 五路分发
    workflow.add_conditional_edges("route", route_after_route, {
        "re_evaluate": "re_evaluate",
        "fast_track": "fast_track",
        "action_track": "action_track",
        "complex_track": "complex_track",
        "clarification": "clarification",
    })

    # re_evaluate → 四路分发
    workflow.add_conditional_edges("re_evaluate", after_re_evaluate, {
        "action_track": "action_track",
        "fast_track": "fast_track",
        "route": "route",
        "respond": "respond",
    })

    workflow.add_edge("fast_track", "respond")
    # action_track 用条件边：new_topic → 回 route 重路由，其余 → respond
    workflow.add_conditional_edges("action_track", after_action_track, {
        "route": "route",
        "respond": "respond",
    })
    workflow.add_edge("complex_track", "respond")
    workflow.add_edge("clarification", "respond")
    workflow.add_edge("respond", END)

    return workflow


# ============================================================
# 流式输出辅助
# ============================================================

def _yield_stream_event(node_name: str, node_state: dict):
    """将图节点事件转为流式标签 yield（生成器辅助函数）"""
    # respond 节点只负责持久化，不产出行内输出
    if node_name == "respond":
        return

    if node_name == "re_evaluate":
        pass  # 静默执行
    elif node_name == "fast_track":
        yield "[FAST] 📚 企业知识库检索 · EnterpriseRAG\n"
    elif node_name == "action_track":
        track = node_state.get("track", "action_create")
        if track == "action_query":
            yield "[ACTION_QUERY] 🔍 数据查询 · ToolAgent\n"
        else:
            yield "[ACTION] ⚡ 动作通道 · 工单派发\n"
    elif node_name == "complex_track":
        yield "[COMPLEX] 🧩 复杂通道 · 多步骤编排\n"
    elif node_name == "clarification":
        yield "[CLARIFY] 🤔 AI 需要确认您的意图\n"
    elif node_name == "route":
        track = node_state.get("track", "")
        if track and track not in ("action_create", "action_query", "re_evaluate"):
            track_names = {
                "fast": "🔍 极速通道 · 企业知识库问答",
                "action_query": "🔍 个人数据查询 · ToolAgent",
                "action_create": "⚡ 动作通道 · 工单派发",
                "complex": "🧩 复杂通道 · 多步骤编排",
            }
            yield f"[ROUTE] {track_names.get(track, track)}\n"

    # 文本 / 卡片输出
    resp = node_state.get("final_response", "")
    if not resp:
        return
    if "[CARD]" in resp:
        parts = resp.split("[CARD]", 1)
        text_part = parts[0].strip()
        card_part = parts[1].strip() if len(parts) > 1 else ""
        if text_part:
            yield f"[STREAM]{text_part}\n"
        if card_part:
            yield f"[CARD]{card_part}\n"
    else:
        yield f"[STREAM]{resp}\n"


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
        self.checkpointer = self._create_checkpointer()
        self.app = self.workflow.compile(checkpointer=self.checkpointer)

    @staticmethod
    def _create_checkpointer():
        """
        MemorySaver：会话检查点内存存储。

        生产环境可替换为持久化方案：
          from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
          import aiosqlite
          conn = await aiosqlite.connect("data/checkpoints.db")
          return AsyncSqliteSaver(conn)
        """
        logger.info("使用 MemorySaver（生产可替换为 AsyncSqliteSaver 持久化）")
        return MemorySaver()

    @staticmethod
    def _ensure_agents_loaded():
        """确保子Agent模块已导入并注册"""
        try:
            import agents.sub_agents.enterprise_rag     # noqa: F401
            import agents.sub_agents.ticket_dispatch    # noqa: F401
            import agents.sub_agents.tool_agent         # noqa: F401
            import agents.tools.builtin_tools           # noqa: F401  工具注册
        except ImportError as e:
            logger.warning(f"Agent 模块加载警告: {e}")

    async def run(self, user_input: str, thread_id: str = "default") -> TicketState:
        initial_state = create_initial_state(user_input, thread_id)
        config = {"configurable": {"thread_id": thread_id}}
        return await self.app.ainvoke(initial_state, config)

    async def run_stream(
        self, user_input: str, thread_id: str = "default",
        user_name: str = "", role: str = "employee",
    ) -> AsyncGenerator[str, None]:
        """
        流式运行编排工作流。

        v3 改进：fast_track 使用真流式（LLM token 级推送），
        其他轨道因执行速度快保留伪流式。

        v3.1 卡片锁：pending_card 期间短路 Router，Agent 分类意图，
        换话题时图内重路由。

        输出令牌：
          [THINKING] <文字>            — 更新"思考中..."文字
          [ROUTE] <轨道描述>           — 路由判定结果
          [CLARIFY] <反问文字>         — AI 反问用户澄清意图
          [FAST]/[ACTION]/[COMPLEX]    — 轨道入口
          [STREAM]<文字片段>           — 流式回答片段（fast_track 为真流式）
          [CARD]<JSON>                 — 确认卡片
          [DONE]                       — 完成
        """
        initial_state = create_initial_state(user_input, thread_id,
                                            user_name=user_name, role=role)
        config = {"configurable": {"thread_id": thread_id}}

        # ── 从 checkpointer 恢复 v4/v5 对话阶段状态 + 身份 ──
        # messages 由 LangGraph add_messages reducer 自动合并，无需手动处理。
        # 但 conversation_phase 等普通字段会被 initial_state 覆盖，需手动恢复。
        prev = self.app.get_state(config)
        pending_card = ""
        if prev and prev.values:
            pending_card = prev.values.get("pending_card_type", "")

            # v4: 继承 self_help_provided 阶段状态
            if prev.values.get("conversation_phase") == "self_help_provided":
                initial_state["conversation_phase"] = "self_help_provided"
                initial_state["last_rag_topic"] = prev.values.get("last_rag_topic", "")
                initial_state["last_rag_summary"] = prev.values.get("last_rag_summary", "")
                logger.info(
                    f"[Stream] 恢复 self_help_provided, "
                    f"topic={initial_state['last_rag_topic']}"
                )
            # 继承卡片数据
            if pending_card:
                prev_agent_results = prev.values.get("agent_results", {})
                if prev_agent_results:
                    initial_state["agent_results"] = prev_agent_results

            # v5: 继承用户身份（新请求未带身份时用上一次的）
            if not initial_state["user_name"] and prev.values.get("user_name"):
                initial_state["user_name"] = prev.values["user_name"]
                initial_state["role"] = prev.values.get("role", "employee")

        if pending_card:
            # 跳过 Router，直送 action_track 让 Agent 分类意图
            initial_state["pending_card_type"] = pending_card
            initial_state["track"] = "action_create"

            yield f"[ROUTE] 📋 处理您对「{pending_card}」卡片的回复\n"
            yield "[THINKING] 🔍 正在理解您的意图...\n"

            async for event in self.app.astream(initial_state, config):
                for node_name, node_state in event.items():
                    if node_name == "action_track":
                        yield "[ACTION] ⚡ 动作通道 · 处理卡片回复\n"
                        if node_state.get("re_route"):
                            yield "[ROUTE] 🔄 切换话题，重新分析...\n"
                            yield "[THINKING] 🔍 正在路由到对应轨道...\n"
                    for _y in _yield_stream_event(node_name, node_state):
                        yield _y

            yield "[DONE]\n"
            return

        # ================================================================
        # 非锁模式：graph 全权接管
        #
        # stream_mode=["updates","custom"]:
        #   "updates" → {node_name: changed_state} → _yield_stream_event 翻译标签
        #   "custom"  → StreamWriter token → 直接转为 [STREAM]
        #
        # checkpointer 自动合并 history / 持久化 state。
        # ================================================================
        yield "[THINKING] 🔍 正在分析您的问题...\n"

        async for event in self.app.astream(
            initial_state, config, stream_mode=["updates", "custom"],
        ):
            if isinstance(event, tuple) and len(event) == 2:
                mode, data = event
                if mode == "custom":
                    # StreamWriter 发射的 token → [STREAM]
                    yield f"[STREAM]{data}\n"
                    continue
                elif mode == "updates":
                    # {node_name: changed_state} → _yield_stream_event
                    for node_name, node_state in data.items():
                        for _y in _yield_stream_event(node_name, node_state):
                            yield _y
            else:
                # 兼容 fallback
                for node_name, node_state in event.items():
                    for _y in _yield_stream_event(node_name, node_state):
                        yield _y

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

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
from langgraph.types import Command
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
    # v7: 动态Agent跨turn状态
    dynamic_agent_messages: list  # 序列化的消息历史（跨turn恢复）
    dynamic_agent_proposals: dict # 上一轮的提议（确认后执行）
    # v9: 真正的中断控制（每次迭代一个工具调用，create_ticket 后立即 interrupt）
    dynamic_iteration: int        # 当前 ReAct 迭代次数
    dynamic_interrupt_card: dict  # 当前触发中断的单张卡片
    dynamic_pending_tool: dict    # 等待确认的工具信息 {name, args, tool_call_id}
    # v11: 上一轮轨道类型 — 供 re_evaluate 正确分发 follow_up
    last_track_type: str          # "fast" | "dynamic" | ""


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
        dynamic_agent_messages=[], dynamic_agent_proposals={},
        dynamic_iteration=0, dynamic_interrupt_card={}, dynamic_pending_tool={},
        last_track_type="",
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
    state["last_track_type"] = ""


def _generate_rag_topic(user_input: str, response: str) -> str:
    """从用户输入和 RAG 回答中提取简要主题（规则兜底，不调 LLM）"""
    # 取用户输入的前 20 字作为主题标签
    topic = user_input[:20].replace("\n", " ").strip()
    return topic if topic else "企业服务咨询"


def _detect_topic_from_history(messages: list) -> str:
    """
    从最近几轮对话中提取话题标签（纯规则匹配，零延迟）。

    仅用于给 LLM 提供话题提示，不参与路由决策。
    """
    from langchain_core.messages import HumanMessage

    # 取最近 2 轮用户消息
    user_msgs = [m for m in messages[-8:]
                 if isinstance(m, HumanMessage) or (
                     isinstance(m, dict) and m.get("role") == "user"
                 )]
    recent_text = " ".join([
        (m.content if hasattr(m, "content") else m.get("content", ""))[:80]
        for m in user_msgs[-2:]
    ])

    topic_keywords = {
        "请假/休假": ["请假", "年假", "病假", "事假", "调休", "休假", "休", "假期"],
        "入职/设备领用": ["入职", "设备", "电脑", "笔记本", "显示器", "领用", "采购", "资产"],
        "IT/故障报修": ["VPN", "网络", "故障", "报修", "连不上", "打不开", "电脑坏", "打印机"],
        "报销": ["报销", "发票", "差旅", "费用"],
        "会议室": ["会议室", "预定", "开会", "预约"],
        "查政策/知识": ["政策", "流程", "怎么", "如何", "规定", "查询", "在哪"],
    }

    for topic, keywords in topic_keywords.items():
        if any(kw in recent_text for kw in keywords):
            return topic

    return "通用咨询"


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

    # ── 短路 1: dynamic 确认卡片 (v8: LangGraph interrupt, v9: dynamic_interrupt) ──
    pending = state.get("pending_card_type", "")
    if pending and pending.startswith("dynamic_"):
        logger.info(f"[Route] dynamic 卡片锁 pending={pending}，短路 Router → dynamic")
        state["track"] = "dynamic"
        state["agent_id"] = "dynamic_action"
        state["confidence"] = 1.0
        state["intent"] = "dynamic_resume"
        state["resolved"] = False
        return state

    # ── 短路 2: 旧卡片锁 ──
    if pending:
        logger.info(f"[Route] 卡片锁 pending_card={pending}，短路 Router → action_create")
        state["track"] = "action_create"
        state["agent_id"] = "ticket_dispatch"
        state["confidence"] = 1.0
        state["intent"] = pending
        state["resolved"] = False
        return state

    # ── 短路 3: self_help_provided 阶段 → re_evaluate ──
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
        "你是对话意图分类器。上一轮助手针对「{topic}」提供了方案/完成了操作，用户刚回复了一句话。\n"
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
        "  - '算了先不管了，帮我查报销' ← 主动放弃当前问题\n"
        "  - '帮我准备入职设备''预定会议室''VPN怎么连' ← 全新领域\n"
        "  重要: 即使上一轮操作已成功完成(如工单已创建)，用户的新消息如果与上一轮话题\n"
        "  完全无关(如从请假切换到设备领用)，也必须判定为 new_topic。\n"
        "  例: 上一轮已成功提交请假 → 用户说'帮我给新同事准备入职设备' → new_topic\n\n"
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
            # v11: 根据上一轮轨道决定 follow_up 目标
            last_track = state.get("last_track_type", "")
            if last_track == "dynamic":
                # 上一轮是 dynamic_action (ReAct slot-filling) → 继续走 dynamic
                state["track"] = "dynamic"
                state["agent_id"] = "dynamic_action"
                state["intent"] = "dynamic_resume"
                state["confidence"] = 0.85
                state["resolved"] = False
                logger.info("[ReEvaluate] follow_up → dynamic (ReAct slot-filling)")
            else:
                # 上一轮是 fast_track (RAG) → 继续走 RAG，带上下文防重复
                state["track"] = "fast"
                state["agent_id"] = "enterprise_rag"
                state["intent"] = "knowledge_query"
                state["confidence"] = 0.85
                state["resolved"] = False
            # follow_up 时不清 state — 下游需要 last_rag_topic

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
        state["last_track_type"] = "fast"
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


async def dynamic_action_node(state: TicketState) -> TicketState:
    """
    动态动作节点 (v10) — 真正的中途中断 ReAct

    每次图节点执行 = 一次 LLM 调用 + 工具执行。
    如果 create_ticket 返回 proposed → 设置 pending_card 后返回，
    由独立的 dynamic_interrupt_node 调用 interrupt() 冻结图。

    自循环边: dynamic_action → dynamic_action（继续迭代）
    中断边:   dynamic_action → dynamic_interrupt → dynamic_action（确认后继续）
    终结边:   dynamic_action → respond（LLM 完成）
    """
    from agents.sub_agents.dynamic_action_agent import DynamicActionAgent
    from langgraph.config import get_stream_writer

    writer = get_stream_writer()
    agent = DynamicActionAgent()
    agent._proposals = {}  # v10: 每次迭代都初始化，供 _tool_create_ticket 存储卡片
    user_text = _get_user_text(state)

    # ================================================================
    # GATE A: 初始化或继续 ReAct
    # ================================================================
    iteration = state.get("dynamic_iteration", 0)
    messages = state.get("dynamic_agent_messages", [])
    is_first = (iteration == 0 and len(messages) == 0)

    if is_first:
        await agent._ensure_inventory_seeded()
        agent._last_user_input = user_text
        agent._last_user_name = state.get("user_name", "")
        agent._last_user_role = state.get("role", "employee")
        agent._last_trace_id = state.get("thread_id", "")

        messages = [
            {"role": "system", "content": agent._build_system_prompt(state.get("user_name", ""))},
        ]

        # v11: 结构化 XML 注入 — 隔离历史与当前，明确话题边界
        conv_hist = _build_conversation_context(state["messages"])
        if conv_hist:
            _last_topic = _detect_topic_from_history(state["messages"])
            history_block = (
                f"<conversation_history>\n"
                f"Previous topic: {_last_topic}\n"
                f"Note: This is for context only. The current request may be on a different topic.\n"
                f"{conv_hist}\n"
                f"</conversation_history>"
            )
            messages.append({"role": "user", "content": history_block})

        messages.append({
            "role": "user",
            "content": (
                f"<current_request>\n{user_text}\n</current_request>\n\n"
                "Process the <current_request> above. "
                "Only use <conversation_history> if it is clearly the SAME topic "
                "(e.g., slot filling where the user is providing missing details). "
                "If <current_request> is a completely different topic, ignore the history."
            ),
        })

        logger.info(
            f"[DynamicAction:v11] FIRST execution — iter={iteration}, "
            f"topic={_detect_topic_from_history(state['messages']) if conv_hist else 'N/A'}"
        )

    if iteration >= agent.MAX_REACT_ITERATIONS:
        logger.warning(f"[DynamicAction:v9] 达到最大迭代 {iteration}，强制总结")
        messages.append({
            "role": "user",
            "content": "已达到最大操作步数。请基于已有结果给用户简洁总结。",
        })
        final = await agent.llm.ainvoke(messages)
        state["final_response"] = final.content.strip() or "处理超时，请重新提交。"
        state["agent_results"]["dynamic_action"] = {
            "success": True,
            "payload": {"direct_response": state["final_response"]},
            "error": None,
        }
        state["resolved"] = True
        state["dynamic_agent_messages"] = []
        state["dynamic_iteration"] = 0
        return state

    logger.info(f"[DynamicAction:v9] 迭代 {iteration + 1}/{agent.MAX_REACT_ITERATIONS}")

    # ================================================================
    # GATE C: 一次 LLM 调用
    # ================================================================
    thought_event = json.dumps({
        "event": "thought",
        "text": f"正在分析第 {iteration + 1} 步...",
    }, ensure_ascii=False)
    writer(f"[REACT]{thought_event}")

    try:
        response = await agent.llm_with_tools.ainvoke(messages)
    except Exception as e:
        logger.error(f"[DynamicAction:v9] LLM 调用失败: {e}")
        state["final_response"] = "抱歉，处理请求时出现问题。请稍后重试。"
        state["resolved"] = True
        state["dynamic_agent_messages"] = []
        return state

    # ── 无 tool_calls → 最终回答 ──
    if not response.tool_calls:
        final_answer = response.content.strip() if response.content else "处理完成。"
        logger.info(f"[DynamicAction:v9] FINAL — iter {iteration + 1}, 无 tool_calls")

        final_event = json.dumps({
            "event": "final",
            "text": final_answer,
            "iterations": iteration + 1,
        }, ensure_ascii=False)
        writer(f"[REACT]{final_event}")

        state["final_response"] = final_answer
        state["agent_results"]["dynamic_action"] = {
            "success": True,
            "payload": {"direct_response": final_answer},
            "error": None,
        }
        # v11: 如果 final answer 是反问（含问号），标记为持续对话
        # 下一轮会走 re_evaluate_node 判断 follow_up 还是 new_topic
        if "?" in final_answer or "？" in final_answer:
            state["conversation_phase"] = "self_help_provided"
            state["last_rag_topic"] = _generate_rag_topic(user_text, final_answer)
            state["last_rag_summary"] = final_answer[:150]
            state["last_track_type"] = "dynamic"
            logger.info(
                f"[DynamicAction:v11] 反问结尾 → phase=self_help_provided, "
                f"topic={state['last_rag_topic']}, last_track=dynamic"
            )
        state["resolved"] = True
        state["dynamic_agent_messages"] = []
        state["dynamic_iteration"] = 0
        return state

    # ================================================================
    # GATE B: 执行工具 — create_ticket 提议时设置 pending 返回
    # ================================================================
    thought = response.content[:200] if response.content else ""

    for i, tool_call in enumerate(response.tool_calls):
        tool_name = tool_call.get("name", "")
        tool_args = tool_call.get("args", {})
        if isinstance(tool_args, str):
            tool_args = json.loads(tool_args)

        tool_call_id = f"call_{iteration}_{tool_name}_{i}"

        # 推送: 工具调用
        tool_call_event = json.dumps({
            "event": "tool_call",
            "text": agent._describe_tool_call(tool_name, tool_args),
            "tool": tool_name,
            "args": tool_args,
        }, ensure_ascii=False)
        writer(f"[REACT]{tool_call_event}")

        # 执行工具
        observation = await agent._execute_tool(tool_name, tool_args)

        # 推送: 工具结果
        obs_summary = agent._summarize_observation(tool_name, observation)
        tool_result_event = json.dumps({
            "event": "tool_result",
            "text": obs_summary,
            "tool": tool_name,
        }, ensure_ascii=False)
        writer(f"[REACT]{tool_result_event}")

        # 追加 assistant 消息（含 tool_call）
        messages.append({
            "role": "assistant",
            "content": thought,
            "tool_calls": [{
                "id": tool_call_id,
                "type": "function",
                "function": {
                    "name": tool_name,
                    "arguments": json.dumps(tool_args, ensure_ascii=False),
                },
            }],
        })

        # ★★★ create_ticket 提议 → 设置 pending, 由独立节点调用 interrupt() ★★★
        if tool_name == "create_ticket":
            try:
                data = json.loads(observation) if isinstance(observation, str) else observation
            except (json.JSONDecodeError, TypeError):
                data = {}

            if data.get("status") == "proposed" and not agent._execution_mode:
                # 不追加 tool result — 等用户确认后再追加
                proposals = getattr(agent, '_proposals', {})
                card = {}
                if proposals:
                    last_key = list(proposals.keys())[-1]
                    card = proposals[last_key].get("card", {})

                state["dynamic_agent_messages"] = messages
                state["dynamic_interrupt_card"] = card
                state["dynamic_pending_tool"] = {
                    "name": tool_name,
                    "args": tool_args,
                    "tool_call_id": tool_call_id,
                }
                state["pending_card_type"] = "dynamic_interrupt"
                state["resolved"] = False
                state["dynamic_iteration"] = iteration  # 还没完成这一轮

                # 推送卡片到前端
                if card:
                    writer(f"[CARD]{json.dumps(card, ensure_ascii=False)}")

                logger.info(
                    f"[DynamicAction:v10] 🛑 PROPOSED — "
                    f"card={card.get('title', '?')[:40]}, "
                    f"iter={iteration}, tool_call_id={tool_call_id}"
                )
                # ★ 不在这里 interrupt() — 由 dynamic_interrupt_node 统一处理
                return state

        # 非 create_ticket / 非 proposed → 正常追加 tool result
        messages.append({
            "role": "tool",
            "tool_call_id": tool_call_id,
            "content": observation if isinstance(observation, str)
            else json.dumps(observation, ensure_ascii=False),
        })

    # ── 无中断 → 保存消息, 自循环 ──
    state["dynamic_agent_messages"] = messages
    state["dynamic_iteration"] = iteration + 1
    state["resolved"] = False

    logger.info(
        f"[DynamicAction:v10] 继续 — iter {iteration + 1}, "
        f"{len(messages)} msgs saved"
    )
    return state


async def dynamic_interrupt_node(state: TicketState) -> TicketState:
    """
    中断节点 (v10) — 单一 interrupt() 调用点，处理用户确认/取消/修改。

    dynamic_action_node 设置 pending_card_type="dynamic_interrupt" 后进入此节点。
    首次执行: interrupt() 冻结图，等待用户操作。
    Replay:   interrupt() 返回决策 → 执行确认/取消/修改 → 返回 state。
    """
    from agents.sub_agents.dynamic_action_agent import DynamicActionAgent
    from langgraph.types import interrupt

    card = state.get("dynamic_interrupt_card", {})
    pending_tool = state.get("dynamic_pending_tool", {})
    messages = state.get("dynamic_agent_messages", [])
    user_text = _get_user_text(state)

    logger.info(
        f"[DynamicInterrupt:v10] 等待用户确认: "
        f"card={card.get('title', '?')[:40]}"
    )

    # ★ 唯一的 interrupt() 调用点 — replay 时正确返回决策
    decision = interrupt({
        "type": "dynamic_interrupt",
        "card": card,
        "message": f"请确认: {card.get('title', '')}",
    })

    action = decision.get("action", "confirm")
    logger.info(
        f"[DynamicInterrupt:v10] RESUME decision={action}, "
        f"iter={state.get('dynamic_iteration', 0)}"
    )

    # ── 执行用户决定 ──
    agent = DynamicActionAgent()
    agent._proposals = {}
    await agent._ensure_inventory_seeded()

    if action == "confirm":
        agent._execution_mode = True
        agent._last_user_input = user_text
        agent._last_user_name = state.get("user_name", "")
        agent._last_user_role = state.get("role", "employee")
        agent._last_trace_id = state.get("thread_id", "")

        result_json = await agent._tool_create_ticket(pending_tool.get("args", {}))
        result = json.loads(result_json) if isinstance(result_json, str) else {}
        agent._execution_mode = False

        if result.get("executed"):
            ticket_no = result.get("ticket_number", "?")
            observation = json.dumps(result, ensure_ascii=False)
            logger.info(f"[DynamicInterrupt:v10] TicketDispatch 落库成功: {ticket_no}")
        else:
            observation = json.dumps(
                {"executed": False, "error": result.get("error", "执行失败")},
                ensure_ascii=False,
            )
    elif action == "cancel":
        observation = json.dumps(
            {"executed": False, "message": "用户取消"},
            ensure_ascii=False,
        )
    else:  # modify
        feedback = decision.get("feedback", "")
        observation = json.dumps(
            {"executed": False, "message": f"用户要求修改: {feedback}"},
            ensure_ascii=False,
        )

    # ★ 把 tool result 追加到对话历史 — LLM 在下一轮看到实际结果
    messages.append({
        "role": "tool",
        "tool_call_id": pending_tool.get(
            "tool_call_id",
            f"call_resume_{state.get('dynamic_iteration', 0)}",
        ),
        "content": observation,
    })

    state["dynamic_agent_messages"] = messages
    state["pending_card_type"] = ""
    state["dynamic_interrupt_card"] = {}
    state["dynamic_pending_tool"] = {}
    state["dynamic_iteration"] = state.get("dynamic_iteration", 0) + 1
    state["resolved"] = False

    logger.info(
        f"[DynamicInterrupt:v10] 完成 — "
        f"{len(messages)} msgs, iter={state['dynamic_iteration']}, "
        f"返回 dynamic_action 继续推理"
    )
    return state


def after_dynamic_action(state: TicketState) -> str:
    """
    dynamic_action 之后的路由:
      - resolved=True → respond (LLM 给出最终回答)
      - pending_card="dynamic_interrupt" → dynamic_interrupt (中断确认)
      - 其他 pending → respond (旧卡片锁)
      - 其他 → dynamic_action (自循环，继续迭代)
    """
    if state.get("resolved"):
        return "respond"
    pending = state.get("pending_card_type", "")
    if pending == "dynamic_interrupt":
        return "dynamic_interrupt"  # v10: 独立中断节点
    if pending:
        return "respond"
    return "dynamic_action"


def after_dynamic_interrupt(state: TicketState) -> str:
    """
    dynamic_interrupt 之后的路由:
      - resolved=True → respond
      - 其他 → dynamic_action (继续 ReAct)
    """
    if state.get("resolved"):
        return "respond"
    return "dynamic_action"


def _serialize_messages(messages: list) -> list:
    """将 LangChain 消息列表序列化为可存储的字典列表"""
    if not messages:
        return []
    result = []
    for m in messages:
        entry = {"role": m.get("role", "") if isinstance(m, dict) else getattr(m, "role", "system")}
        content = m.get("content", "") if isinstance(m, dict) else getattr(m, "content", "")
        entry["content"] = str(content)[:2000]
        if isinstance(m, dict) and m.get("tool_calls"):
            entry["tool_calls"] = m["tool_calls"]
        elif hasattr(m, "tool_calls") and m.tool_calls:
            entry["tool_calls"] = m.tool_calls
        if isinstance(m, dict) and m.get("tool_call_id"):
            entry["tool_call_id"] = m["tool_call_id"]
        elif hasattr(m, "tool_call_id") and m.tool_call_id:
            entry["tool_call_id"] = m.tool_call_id
        result.append(entry)
    return result


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
    "re_evaluate", "fast_track", "dynamic_action", "action_track", "complex_track", "clarification",
]:
    """
    路由分发 (v7: 新增 dynamic_action)

    v7 变化:
      - action_query / action_create / complex 三轨合一为 dynamic
      - 保留旧轨道作为 fallback（渐进迁移）
    """
    track = state.get("track", "clarification")
    if track == "re_evaluate":     return "re_evaluate"
    if track == "fast":            return "fast_track"
    elif track == "dynamic":       return "dynamic_action"
    elif track == "action_query":  return "action_track"   # 向后兼容
    elif track == "action_create": return "action_track"   # 向后兼容
    elif track == "complex":       return "complex_track"  # 向后兼容
    else:                          return "clarification"


def after_re_evaluate(state: TicketState) -> Literal[
    "action_track", "dynamic_action", "fast_track", "route", "respond",
]:
    """
    re_evaluate 后的分发 (v11):
      - escalation → action_track
      - follow_up → dynamic_action（若上一轮是 ReAct slot-filling）
                   → fast_track（若上一轮是 RAG 知识查询）
      - new_topic → route（清状态重路由）
      - confirm → respond（清状态结束）
    """
    intent = state.get("agent_results", {}).get("re_evaluate", {}).get("intent", "escalation")
    if intent == "escalation":
        return "action_track"
    elif intent == "follow_up":
        # v11: 根据上一轮轨道正确分发
        if state.get("last_track_type") == "dynamic":
            return "dynamic_action"
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

async def _classify_dynamic_response(
    user_text: str,
    prev_values: dict,
) -> dict:
    """
    v8: 分类用户对动态卡片的回复意图 (AskUserQuestion 风格)

    当图被 interrupt() 冻结后, 用户通过文本回复(而非按钮),
    需要 LLM 判断意图: confirm / modify / cancel / new_topic

    Returns: {"action": "...", "feedback": "..."}
    """
    from config.model_provider import create_chat_model

    # 获取卡片信息作为分类上下文
    # v9: dynamic_interrupt 单卡片优先 → v8: cards 数组兜底
    single_card = prev_values.get("dynamic_interrupt_card", {})
    if single_card:
        cards = [single_card]
    else:
        agent_results = prev_values.get("agent_results", {})
        dynamic_result = agent_results.get("dynamic_action", {})
        cards = dynamic_result.get("cards", [])
    card_descriptions = "\n".join(
        f"- {c.get('title', '')}: {c.get('description', '')[:200]}"
        for c in cards
    ) if cards else "(no card info)"

    system_prompt = (
        "你是企业服务台的意图分类器。用户看到了一张或多张确认卡片, 然后回复了一句话。\n"
        "请调用 classify_intent 函数判断用户的意图。\n\n"
        "分类标准:\n"
        "- confirm: 用户确认/同意卡片内容, 要求执行操作。\n"
        "  例: '好的''行''确认''可以''没问题''就这样''yes''ok''confirm'\n"
        "- modify: 用户想修改卡片的某个参数。\n"
        "  例: '把显示器改成LG的''数量改成2台''不要耳机''加一个鼠标'\n"
        "  关键: 仍然围绕卡片内容, 但要求调整\n"
        "- cancel: 用户想取消/放弃。\n"
        "  例: '算了''取消''不要了''不用了''cancel'\n"
        "- new_topic: 用户完全换了话题, 与当前卡片无关。\n"
        "  例: '帮我查下请假政策''会议室怎么预定''VPN怎么连'\n"
        "核心判断: 用户的话是否仍然围绕这张卡片?\n"
        "围绕卡片=confirm/modify/cancel, 完全不相关=new_topic。"
    )

    classify_tool = {
        "type": "function",
        "function": {
            "name": "classify_intent",
            "description": "分类用户对确认卡片的回复意图",
            "parameters": {
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": ["confirm", "modify", "cancel", "new_topic"],
                        "description": "用户意图分类",
                    },
                    "feedback": {
                        "type": "string",
                        "description": "如果是modify, 简述用户要求的修改内容; 其他情况为空字符串",
                    },
                },
                "required": ["action"],
            },
        },
    }

    try:
        llm = create_chat_model(model_type="main", temperature=0)
        llm_with_tool = llm.bind_tools([classify_tool], tool_choice="auto")

        response = await llm_with_tool.ainvoke([
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": (
                f"卡片内容:\n{card_descriptions}\n\n"
                f"用户回复: \"{user_text}\"\n\n"
                f"请调用 classify_intent 函数分类用户意图。"
            )},
        ])

        if response.tool_calls:
            args = response.tool_calls[0].get("args", {})
            if isinstance(args, str):
                import json as _json
                args = _json.loads(args)
            return {
                "action": args.get("action", "confirm"),
                "feedback": args.get("feedback", ""),
            }
        else:
            # Fallback: 关键词规则
            text_lower = user_text.lower().strip()
            cancel_words = ["取消", "算了", "不要", "不用", "cancel", "放弃", "别"]
            confirm_words = ["确认", "好的", "行", "可以", "是", "yes", "ok", "对", "好"]

            if any(w in text_lower for w in cancel_words):
                return {"action": "cancel", "feedback": ""}
            elif any(w in text_lower for w in confirm_words):
                return {"action": "confirm", "feedback": ""}
            elif len(user_text) < 30 and "?" not in user_text and "？" not in user_text:
                return {"action": "confirm", "feedback": ""}
            else:
                return {"action": "modify", "feedback": user_text}

    except Exception as e:
        logger.error(f"[ClassifyDynamic] 分类失败: {e}，兜底为 confirm")
        return {"action": "confirm", "feedback": ""}


def build_orchestration_workflow() -> StateGraph:
    workflow = StateGraph(TicketState)

    workflow.add_node("route", route_node)
    workflow.add_node("re_evaluate", re_evaluate_node)
    workflow.add_node("fast_track", fast_track_node)
    workflow.add_node("dynamic_action", dynamic_action_node)   # v10: ReAct 自由编排
    workflow.add_node("dynamic_interrupt", dynamic_interrupt_node)  # v10: 中断确认
    workflow.add_node("action_track", action_track_node)
    workflow.add_node("complex_track", complex_track_node)
    workflow.add_node("clarification", clarification_node)
    workflow.add_node("respond", respond_node)

    workflow.set_entry_point("route")

    # route → 六路分发 (v7: 新增 dynamic_action)
    workflow.add_conditional_edges("route", route_after_route, {
        "re_evaluate": "re_evaluate",
        "fast_track": "fast_track",
        "dynamic_action": "dynamic_action",
        "action_track": "action_track",
        "complex_track": "complex_track",
        "clarification": "clarification",
    })

    # re_evaluate → 五路分发 (v11: 新增 dynamic_action 用于 ReAct slot-filling)
    workflow.add_conditional_edges("re_evaluate", after_re_evaluate, {
        "action_track": "action_track",
        "dynamic_action": "dynamic_action",
        "fast_track": "fast_track",
        "route": "route",
        "respond": "respond",
    })

    workflow.add_edge("fast_track", "respond")
    workflow.add_conditional_edges("dynamic_action", after_dynamic_action, {
        "dynamic_action": "dynamic_action",       # v10: self-loop 继续迭代
        "dynamic_interrupt": "dynamic_interrupt",  # v10: 中断确认
        "respond": "respond",
    })
    workflow.add_conditional_edges("dynamic_interrupt", after_dynamic_interrupt, {
        "dynamic_action": "dynamic_action",  # v10: 确认后继续推理
        "respond": "respond",
    })
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
    # 安全保护: 非 dict 跳过（如 interrupt 产生的 tuple）
    if not isinstance(node_state, dict):
        return

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
    elif node_name == "dynamic_action":
        yield "[DYNAMIC] 🧠 动态编排 · ReAct 循环\n"
    elif node_name == "dynamic_interrupt":
        yield "[INTERRUPT_CARD] 📋 等待确认\n"
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
            import agents.sub_agents.dynamic_action_agent  # noqa: F401  v7: ReAct自由编排
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
                initial_state["last_track_type"] = prev.values.get("last_track_type", "")
                logger.info(
                    f"[Stream] 恢复 self_help_provided, "
                    f"topic={initial_state['last_rag_topic']}, "
                    f"last_track={initial_state['last_track_type']}"
                )
            # 继承卡片数据
            if pending_card:
                prev_agent_results = prev.values.get("agent_results", {})
                if prev_agent_results:
                    initial_state["agent_results"] = prev_agent_results
                # ★ v7: 恢复 dynamic agent 跨turn状态 (防坑1: 避免Turn2失忆)
                if pending_card.startswith("dynamic_"):
                    initial_state["dynamic_agent_messages"] = prev.values.get(
                        "dynamic_agent_messages", [],
                    )
                    initial_state["dynamic_agent_proposals"] = prev.values.get(
                        "dynamic_agent_proposals", {},
                    )
                    logger.info(
                        f"[Stream] 恢复 dynamic agent 状态: "
                        f"{len(initial_state['dynamic_agent_messages'])} msgs, "
                        f"{len(initial_state['dynamic_agent_proposals'])} proposals"
                    )

            # v5: 继承用户身份（新请求未带身份时用上一次的）
            if not initial_state["user_name"] and prev.values.get("user_name"):
                initial_state["user_name"] = prev.values["user_name"]
                initial_state["role"] = prev.values.get("role", "employee")

        if pending_card:
            # ── v8/v9: dynamic_* → 图已冻结, 需分类意图后 Command(resume) ──
            if pending_card in ("dynamic_confirm:interrupt", "dynamic_interrupt"):
                # 用户通过文本回复(而非按钮) — 需要 LLM 分类意图
                decision = await _classify_dynamic_response(
                    user_input, prev.values,
                )
                logger.info(
                    f"[Stream:v8] dynamic_confirm:interrupt → "
                    f"classified as {decision.get('action')}"
                )

                if decision.get("action") == "new_topic":
                    # 用户换了话题 → 先用 cancel 清除 interrupt, 再走正常流式路径
                    await self.app.ainvoke(
                        Command(resume={"action": "cancel"}), config,
                    )
                    yield f"[ROUTE] 🔄 Switching topic...\n"
                    yield "[THINKING] 🔍 Analyzing new request...\n"
                    initial_state["pending_card_type"] = ""
                    # Fall through to normal streaming below
                else:
                    # confirm / modify / cancel → 恢复冻结的图
                    yield f"[ROUTE] 📋 Processing ({decision.get('action')})...\n"
                    yield "[THINKING] 🔍 Resuming agent...\n"

                    async for event in self.app.astream(
                        Command(resume=decision), config,
                        stream_mode=["updates", "custom"],
                    ):
                        if isinstance(event, tuple) and len(event) == 2:
                            mode, data = event
                            if mode == "custom":
                                if isinstance(data, str) and (
                                    data.startswith("[REACT]") or data.startswith("[CARD]")
                                ):
                                    yield f"{data}\n"
                                else:
                                    yield f"[STREAM]{data}\n"
                                continue
                            elif mode == "updates":
                                for node_name, node_state in data.items():
                                    for _y in _yield_stream_event(node_name, node_state):
                                        yield _y
                        else:
                            for node_name, node_state in event.items():
                                for _y in _yield_stream_event(node_name, node_state):
                                    yield _y

                    # 检查是否 resume 后又产生了新 interrupt (modify → 新卡片)
                    resumed_state = self.app.get_state(config)
                    if resumed_state and resumed_state.interrupts:
                        yield "[INTERRUPT]\n"
                    else:
                        yield "[DONE]\n"
                    return

            # ── v8: dynamic_confirm:proposals (向后兼容) → 走正常流式通道 ──
            elif pending_card.startswith("dynamic_confirm:"):
                initial_state["pending_card_type"] = pending_card
                yield f"[ROUTE] 📋 Processing your confirmation...\n"
                yield "[THINKING] 🔍 Resuming agent with your approval...\n"
                # ★ 不 return — 落入下方 stream_mode=["updates","custom"] 正常流式路径
                # route_node 会检测 dynamic_confirm:* 并短路到 dynamic_action (resume模式)

            # ── 旧卡片锁: 直送 action_track ──
            else:
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
                    # StreamWriter 发射的数据 — 根据前缀决定标签
                    # [REACT] → 思维链标签（前端渲染为可折叠面板）
                    # [CARD]  → 卡片标签
                    # 其他    → [STREAM] 流式文本
                    if isinstance(data, str) and (
                        data.startswith("[REACT]") or data.startswith("[CARD]")
                    ):
                        yield f"{data}\n"
                    else:
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

        # v8: 检查 graph 是否被 interrupt() 冻结
        final_state = self.app.get_state(config)
        if final_state and final_state.interrupts:
            yield "[INTERRUPT]\n"
            logger.info(
                f"[Stream:v8] Graph interrupted — {len(final_state.interrupts)} interrupt(s) pending"
            )
        else:
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

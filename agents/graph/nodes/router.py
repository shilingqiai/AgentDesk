"""
路由节点 — route_node + re_evaluate_node

Hub & Spoke 架构的入口判定节点。
从 agents.graph_workflow 拆分出来。
依赖: state.py
"""

from __future__ import annotations

import json as _json
import logging
import re

from agents.graph.state import (
    TicketState, _get_user_text, _build_conversation_context,
    _maybe_compress_history, _reset_self_help_state,
)

logger = logging.getLogger("graph.nodes.router")


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

    # ── 短路 2: 旧卡片锁（complex_track 产生的 leave/expense/admin 卡片）──
    if pending:
        logger.info(f"[Route] 卡片锁 pending_card={pending}，短路 Router → card_lock")
        state["track"] = "card_lock"
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

    # v12: 对话历史压缩 — 超过阈值触发 LLM 递进摘要
    summary = state.get("conversation_summary", "")
    summary = await _maybe_compress_history(state["messages"], summary)
    state["conversation_summary"] = summary

    agent_descriptions = agent_registry.get_routing_descriptions()
    conversation_history = _build_conversation_context(
        state["messages"], summary=summary,
    )

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
    if result.track == "complex":
        action_type = "multi_step"
    elif result.track == "dynamic":
        action_type = "create_ticket"  # ReAct 可能建工单，需风险控制

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
            # 升级走 ReAct，让 DynamicActionAgent 自主处理（含建工单）
            state["track"] = "dynamic"
            state["agent_id"] = "dynamic_action"
            state["intent"] = "dynamic_action"
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
        logger.error(f"[ReEvaluate] LLM 调用失败: {e}，兜底为 escalation → dynamic")
        state["track"] = "dynamic"
        state["agent_id"] = "dynamic_action"
        state["intent"] = "dynamic_action"
        state["urgency"] = "high"
        state["confidence"] = 0.7
        state["resolved"] = False

    return state

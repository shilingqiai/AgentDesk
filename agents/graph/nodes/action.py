"""
动作通道节点 — 卡片响应处理器（confirm/modify/cancel/new_topic）

从 agents.graph_workflow 拆分出来。
依赖: state.py, nodes/fast.py
"""

from __future__ import annotations

import logging
from agents.graph.state import TicketState, _get_user_text, _reset_self_help_state
from agents.graph.nodes.fast import fast_track_node

logger = logging.getLogger("graph.nodes.action")


async def action_track_node(state: TicketState) -> TicketState:
    """卡片响应处理器：仅处理 pending_card 锁期间的 confirm/modify/cancel/new_topic 意图分类。"""
    from agents.orchestrator.agent_registry import agent_registry
    from agents.a2a.protocol import AgentMessage as AM

    user_text = _get_user_text(state)
    user_name = state.get("user_name", "")
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
                result_text = await agent_instance.execute_card(
                    prev_card, user_text, user_name,
                )
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
    # 防御性 fallback：非锁状态下不应到达此节点
    # v8: action_query / action_create 分支已移除，合并到 dynamic (ReAct)
    # ================================================================
    logger.warning(
        f"[ActionTrack] 收到非锁请求 track={state.get('track')}，"
        f"降级为 fast_track（RAG）"
    )
    state["pending_card_type"] = ""
    state["re_route"] = False
    return await fast_track_node(state)

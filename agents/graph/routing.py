"""
工作流路由函数 — route_after_* 条件边分发

从 agents.graph_workflow 拆分出来。
依赖: state.py (TicketState type)
"""

from __future__ import annotations

import logging
from typing import Literal
from agents.graph.state import TicketState, _budget_exhausted, _build_budget_degradation_response, _sh_get

logger = logging.getLogger("graph.routing")


def route_after_route(state: TicketState) -> Literal[
    "re_evaluate", "fast_track", "dynamic_action", "action_track", "complex_track", "clarification",
]:
    """
    路由分发 (v8: 4轨 + card_lock)

    v8 变化:
      - action_query / action_create 移除，合并到 dynamic (ReAct 自由编排)
      - card_lock 仅用于旧卡片锁绕行，送达 action_track（纯卡片响应处理器）
      - complex 保留（请假/报销固定 DAG）
    """
    track = state.get("track", "clarification")
    if track == "re_evaluate":     return "re_evaluate"
    if track == "fast":            return "fast_track"
    elif track == "dynamic":       return "dynamic_action"
    elif track == "complex":       return "complex_track"
    elif track == "card_lock":     return "action_track"   # 卡片锁响应
    else:                          return "clarification"


def after_re_evaluate(state: TicketState) -> Literal[
    "action_track", "dynamic_action", "fast_track", "route", "respond",
]:
    """
    re_evaluate 后的分发:
      - escalation → dynamic_action（升级走 ReAct 自主建工单）
      - follow_up → dynamic_action（若上一轮是 ReAct slot-filling）
                   → fast_track（若上一轮是 RAG 知识查询）
      - new_topic → route（清状态重路由）
      - confirm → respond（清状态结束）
    """
    intent = state.get("agent_results", {}).get("re_evaluate", {}).get("intent", "escalation")
    if intent == "escalation":
        return "dynamic_action"
    elif intent == "follow_up":
        if _sh_get(state, "track") == "dynamic":
            return "dynamic_action"
        return "fast_track"
    elif intent == "new_topic":
        return "route"
    else:  # confirm
        return "respond"


def after_budget_gate(state: TicketState) -> Literal[
    "re_evaluate", "fast_track", "dynamic_action", "action_track",
    "complex_track", "clarification", "respond",
]:
    """
    预算门控分发 — Token 预算耗尽时短路到降级响应。

    面试要点：这就是 Loop/Cost Engineering 的落地 — 不是无限制烧 Token，
    而是设置硬性预算上限，归零后强制降级到确定性路径。
    """
    from agents.graph.state import _budget_exhausted

    if _budget_exhausted(state):
        logger.warning(
            f"[BudgetGate] 预算耗尽: "
            f"remaining={state.get('token_budget_remaining', 0)}/"
            f"{state.get('token_budget_total', 10000)}, 短路到降级响应"
        )
        state["final_response"] = _build_budget_degradation_response(state)
        return "respond"

    # 预算充足 — 正常路由
    return route_after_route(state)


def after_action_track(state: TicketState) -> Literal["route", "respond"]:
    """卡片锁 / new_topic 时重路由，其他情况正常结束"""
    if state.get("re_route"):
        logger.info("[Graph] action_track → route (re_route)")
        return "route"
    return "respond"

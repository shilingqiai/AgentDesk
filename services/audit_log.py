"""
审计日志 — JSONL 结构化打点

记录所有关键决策点: 路由决策 / Agent 调用 / 工具执行 / 卡片确认 / 审批操作 / SLA 超时 / 预算耗尽 / 异常
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime
from typing import Literal

AuditAction = Literal[
    "route_decision", "agent_invoke", "tool_call",
    "card_confirm", "ticket_create", "approval_decide",
    "sla_breach", "budget_exhausted", "error",
]

# 专用 logger — 写入 logs/audit.jsonl
_audit_logger = logging.getLogger("audit")
os.makedirs("logs", exist_ok=True)

# 确保有 file handler
if not any(isinstance(h, logging.FileHandler) for h in _audit_logger.handlers):
    _handler = logging.FileHandler("logs/audit.jsonl", encoding="utf-8")
    _handler.setFormatter(logging.Formatter("%(message)s"))
    _audit_logger.addHandler(_handler)
    _audit_logger.setLevel(logging.INFO)
    _audit_logger.propagate = False


def audit(
    action: AuditAction,
    user: str = "",
    trace_id: str = "",
    detail: dict | None = None,
):
    """
    记录审计日志。

    Args:
        action: 操作类型 (route_decision / agent_invoke / tool_call / ...)
        user: 当前用户
        trace_id: 追踪 ID
        detail: 额外详情
    """
    entry = {
        "timestamp": datetime.now().isoformat(),
        "action": action,
        "user": user,
        "trace_id": trace_id,
        "detail": detail or {},
    }
    _audit_logger.info(json.dumps(entry, ensure_ascii=False))


def audit_route(user: str, trace_id: str, track: str, confidence: float, reason: str):
    """路由决策审计"""
    audit("route_decision", user=user, trace_id=trace_id, detail={
        "track": track, "confidence": round(confidence, 3), "reason": reason[:120],
    })


def audit_agent(agent_id: str, user: str, trace_id: str, success: bool, duration_ms: float = 0):
    """Agent 调用审计"""
    audit("agent_invoke", user=user, trace_id=trace_id, detail={
        "agent_id": agent_id, "success": success, "duration_ms": round(duration_ms, 1),
    })


def audit_tool(tool_name: str, user: str, trace_id: str, success: bool, error: str = ""):
    """工具调用审计"""
    audit("tool_call", user=user, trace_id=trace_id, detail={
        "tool": tool_name, "success": success, "error": error[:100],
    })


def audit_approval(step_id: int, decision: str, user: str, trace_id: str, comment: str = ""):
    """审批操作审计"""
    audit("approval_decide", user=user, trace_id=trace_id, detail={
        "step_id": step_id, "decision": decision, "comment": comment[:120],
    })

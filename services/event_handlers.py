"""
事件处理器 — EventBus 订阅者

三个独立 handler:
- NotificationHandler: 审批状态变更 → 通知（写 DB + 日志）
- AuditHandler: 所有事件 → AuditLog 表（审计追踪）
- DashboardHandler: 工单创建/状态变更 → Dashboard 数据更新

Handler 签名: async def handler(event_type: str, **data) -> None
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone

logger = logging.getLogger("event.handlers")


# ═══════════════════════════════════════════════════════
# NotificationHandler
# ═══════════════════════════════════════════════════════

class NotificationHandler:
    """审批状态变更 → 通知下一环节人 / 通知申请人"""

    @staticmethod
    async def on_ticket_created(event_type: str, **data):
        ticket_number = data.get("ticket_number", "?")
        requester = data.get("requester_name", "")
        ticket_type = data.get("ticket_type", "")
        logger.info(
            f"[Notification] 新工单: {ticket_number} "
            f"类型={ticket_type} 申请人={requester}"
        )

    @staticmethod
    async def on_status_changed(event_type: str, **data):
        ticket_number = data.get("ticket_number", "?")
        from_status = data.get("from_status", "")
        to_status = data.get("to_status", "")
        logger.info(
            f"[Notification] 工单状态变更: {ticket_number} "
            f"{from_status} → {to_status}"
        )

    @staticmethod
    async def on_approval_step_approved(event_type: str, **data):
        next_approver = data.get("next_approver", "")
        ticket_number = data.get("ticket_number", "?")
        if next_approver:
            logger.info(
                f"[Notification] 通知下一审批人: {next_approver} "
                f"工单={ticket_number}"
            )

    @staticmethod
    async def on_approval_completed(event_type: str, **data):
        ticket_number = data.get("ticket_number", "?")
        requester = data.get("requester_name", "")
        logger.info(
            f"[Notification] 审批完成: {ticket_number} "
            f"通知申请人 {requester}"
        )

    @staticmethod
    async def on_approval_rejected(event_type: str, **data):
        ticket_number = data.get("ticket_number", "?")
        reason = data.get("reason", "")
        logger.info(
            f"[Notification] 审批驳回: {ticket_number} "
            f"原因={reason[:60]}"
        )

    @staticmethod
    async def on_sla_breached(event_type: str, **data):
        """SLA 超时通知"""
        ticket_number = data.get("ticket_number", "?")
        rule_label = data.get("rule_label", "")
        action = data.get("action", "")
        detail = data.get("detail", "")
        elapsed_h = data.get("elapsed_h", 0)
        approver = data.get("approver", "")
        logger.warning(
            f"[Notification] SLA 超时: {ticket_number} "
            f"规则={rule_label} 动作={action} "
            f"耗时={elapsed_h:.1f}h"
            + (f" 审批人={approver}" if approver else "")
            + (f" — {detail[:60]}" if detail else "")
        )


# ═══════════════════════════════════════════════════════
# AuditHandler
# ═══════════════════════════════════════════════════════

class AuditHandler:
    """所有事件 → AuditLog 表（审计追踪）"""

    @staticmethod
    async def _write_audit(event_type: str, **data):
        """写入审计日志"""
        try:
            from db.db_router import DatabaseRouter
            from db.models import AuditLog

            db_router = DatabaseRouter()
            session = db_router.session_manager.Session()
            try:
                log_entry = AuditLog(
                    event=event_type,
                    ticket_id=data.get("ticket_id"),
                    ticket_number=data.get("ticket_number", ""),
                    operator=data.get("operator", "system"),
                    data={
                        k: str(v)[:200] for k, v in data.items()
                        if k not in ("ticket_id", "ticket_number", "operator")
                    },
                )
                session.add(log_entry)
                session.commit()
            finally:
                session.close()
        except Exception as e:
            logger.warning(f"[Audit] 写入审计日志失败: {e}")

    @classmethod
    async def on_any_event(cls, event_type: str, **data):
        """通用审计 handler — 订阅所有事件类型"""
        await cls._write_audit(event_type, **data)
        logger.debug(f"[Audit] 事件记录: {event_type}")


# ═══════════════════════════════════════════════════════
# DashboardHandler
# ═══════════════════════════════════════════════════════

class DashboardHandler:
    """工单创建/状态变更 → Dashboard 数据更新"""

    # 内存计数器（原型阶段，生产需换 Redis/DB）
    _counters: dict[str, int] = {
        "tickets_created_today": 0,
        "tickets_approved_today": 0,
        "tickets_rejected_today": 0,
        "sla_breached_today": 0,
    }
    _recent_events: list[dict] = []  # 最近 20 条事件
    _max_recent = 20

    @classmethod
    async def on_ticket_created(cls, event_type: str, **data):
        cls._counters["tickets_created_today"] += 1
        cls._add_recent({
            "event": "ticket.created",
            "ticket": data.get("ticket_number", ""),
            "time": datetime.now(timezone.utc).strftime("%H:%M"),
            "detail": f"{data.get('requester_name', '')} 提交 {data.get('ticket_type', '')} 工单",
        })

    @classmethod
    async def on_status_changed(cls, event_type: str, **data):
        to_status = data.get("to_status", "")
        if to_status == "approved":
            cls._counters["tickets_approved_today"] += 1
        elif to_status == "rejected":
            cls._counters["tickets_rejected_today"] += 1
        cls._add_recent({
            "event": "ticket.status_changed",
            "ticket": data.get("ticket_number", ""),
            "time": datetime.now(timezone.utc).strftime("%H:%M"),
            "detail": f"{data.get('from_status', '')} → {to_status}",
        })

    @classmethod
    def _add_recent(cls, entry: dict):
        cls._recent_events.insert(0, entry)
        if len(cls._recent_events) > cls._max_recent:
            cls._recent_events = cls._recent_events[:cls._max_recent]

    @classmethod
    def get_stats(cls) -> dict:
        """获取 Dashboard 统计数据"""
        return {
            "counters": dict(cls._counters),
            "recent_events": list(cls._recent_events[:10]),
        }

    @classmethod
    async def on_sla_breached(cls, event_type: str, **data):
        """SLA 超时 — 更新 Dashboard 计数"""
        cls._counters["sla_breached_today"] += 1
        cls._add_recent({
            "event": "sla.breached",
            "ticket": data.get("ticket_number", ""),
            "time": datetime.now(timezone.utc).strftime("%H:%M"),
            "detail": f"{data.get('rule_label', '')} — {data.get('action', '')}",
        })

"""
SLA 计时引擎 — 工单时效管理

设计:
- 实时计算而非定时轮询（零额外依赖）
- 阶梯升级: 50%时间通知处理人 → 75%通知主管 → 100%紧急升级
- 仅计算工作时间 (9:00-22:00)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta

logger = logging.getLogger("sla.engine")

# ── SLA 配置 ──
SLA_HOURS = {"P0": 2, "P1": 2, "P2": 4, "P3": 8}
BUSINESS_START = 9   # 9:00
BUSINESS_END = 22     # 22:00
ESCALATION_LEVELS = [
    (0.5, "通知处理人"),
    (0.75, "通知主管"),
    (1.0, "紧急升级 — 通知部门负责人"),
]


@dataclass
class SLAStatus:
    ticket_id: int
    ticket_number: str
    priority: str
    created_at: datetime
    sla_hours: int
    deadline: datetime
    elapsed_hours: float
    remaining_hours: float
    escalation_level: int    # 0=正常, 1=通知处理人, 2=通知主管, 3=紧急
    is_breached: bool


def _add_business_hours(start: datetime, hours: int) -> datetime:
    """在起始时间上累加工作小时数"""
    remaining = hours
    current = start
    while remaining > 0:
        current += timedelta(hours=1)
        if BUSINESS_START <= current.hour < BUSINESS_END:
            remaining -= 1
    return current


class SLAEngine:
    """SLA 计时与升级引擎"""

    @staticmethod
    def get_deadline(priority: str, created_at: datetime) -> datetime:
        sla_h = SLA_HOURS.get(priority.upper(), 8)
        return _add_business_hours(created_at, sla_h)

    @staticmethod
    def check_status(ticket: dict) -> SLAStatus:
        """实时计算单个工单的 SLA 状态"""
        created = ticket.get("created_at")
        if isinstance(created, str):
            created = datetime.fromisoformat(created.replace("Z", "+00:00"))
        priority = ticket.get("priority", "P3")
        sla_hours = SLA_HOURS.get(priority.upper(), 8)
        deadline = SLAEngine.get_deadline(priority, created)
        now = datetime.utcnow()

        elapsed = (now - created).total_seconds() / 3600
        remaining = max(0, sla_hours - elapsed)
        is_breached = now > deadline

        ratio = elapsed / sla_hours if sla_hours > 0 else 1.0
        level = 0
        for threshold, _label in ESCALATION_LEVELS:
            if ratio >= threshold:
                level += 1

        return SLAStatus(
            ticket_id=ticket.get("id", 0),
            ticket_number=ticket.get("ticket_number", ""),
            priority=priority,
            created_at=created,
            sla_hours=sla_hours,
            deadline=deadline,
            elapsed_hours=round(elapsed, 1),
            remaining_hours=round(remaining, 1),
            escalation_level=min(level, 3),
            is_breached=is_breached,
        )

    @staticmethod
    def get_breached(db_session) -> list[SLAStatus]:
        """查询所有超时的未关闭工单"""
        from db.models import Ticket

        open_tickets = db_session.query(Ticket).filter(
            Ticket.status.in_(["created", "assigned", "processing"]),
            Ticket.is_active == 1,
        ).all()

        breached = []
        for t in open_tickets:
            ticket_dict = {
                "id": t.id, "ticket_number": t.ticket_number,
                "priority": t.priority, "created_at": t.created_at,
            }
            status = SLAEngine.check_status(ticket_dict)
            if status.is_breached:
                breached.append(status)

        return breached

    @staticmethod
    def format_escalation_message(status: SLAStatus) -> str:
        """生成升级通知文本"""
        if status.escalation_level == 0:
            return ""
        level_label = ESCALATION_LEVELS[min(status.escalation_level, 3) - 1][1]
        return (
            f"⏰ SLA 预警: 工单 {status.ticket_number} "
            f"已过 {status.elapsed_hours:.1f}h/{status.sla_hours}h, "
            f"触发: {level_label}"
        )

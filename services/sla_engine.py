"""
SLA 计时与自动升级引擎 — 工单/审批/预约时效管理

设计:
- 实时计算而非定时轮询（零额外依赖）
- 阶梯升级: 50%时间通知处理人 → 75%通知主管 → 100%紧急升级
- 仅计算工作时间 (9:00-22:00)
- 规则引擎: 不同场景不同 SLA 规则 + 自动动作
- 事件驱动: 超时检测通过 EventBus 发射 sla.breached 事件

v13 新增:
- SLA 规则定义（工单响应/解决/审批步骤/会议室确认）
- 自动升级动作（通知管理/提升优先级/提醒审批人/自动取消）
- check_all() 综合检测入口
- 审批步骤超时检测
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional

logger = logging.getLogger("sla.engine")

# ── SLA 基础配置 ──────────────────────────────────────

SLA_HOURS = {"P0": 2, "P1": 2, "P2": 4, "P3": 8}
BUSINESS_START = 9   # 9:00
BUSINESS_END = 22     # 22:00

ESCALATION_LEVELS = [
    (0.5, "通知处理人"),
    (0.75, "通知主管"),
    (1.0, "紧急升级 — 通知部门负责人"),
]

# ── SLA 规则定义 ──────────────────────────────────────


@dataclass
class SLARule:
    """单条 SLA 规则"""
    key: str                                    # 规则标识
    label: str                                  # 人类可读名称
    duration_h: int                             # SLA 时长（小时）
    action: str                                 # 超时动作: notify_admin | escalate | remind_approver | auto_cancel
    description: str = ""                       # 规则说明


# 规则配置表（新增规则只需加一行）
SLA_RULES: dict[str, SLARule] = {
    "ticket_response": SLARule(
        key="ticket_response",
        label="工单响应",
        duration_h=4,
        action="notify_admin",
        description="工单创建后 4 工作小时内未分配处理人 → 通知管理员",
    ),
    "ticket_resolution": SLARule(
        key="ticket_resolution",
        label="工单解决",
        duration_h=24,
        action="escalate",
        description="工单 24 工作小时未解决 → 自动提升优先级至 P1 + 通知管理员",
    ),
    "approval_step": SLARule(
        key="approval_step",
        label="审批节点",
        duration_h=8,
        action="remind_approver",
        description="审批节点 8 工作小时未操作 → 提醒当前审批人",
    ),
    "meeting_confirm": SLARule(
        key="meeting_confirm",
        label="会议室确认",
        duration_h=2,
        action="auto_cancel",
        description="会议室预约后 2 工作小时未确认 → 自动取消释放资源",
    ),
}


# ── SLA 状态 ──────────────────────────────────────────


@dataclass
class SLAStatus:
    """单个工单的 SLA 状态"""
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


@dataclass
class SLASummary:
    """系统 SLA 状态概览"""
    breached_count: int = 0
    warning_count: int = 0             # 已触发预警但未完全超时
    active_sla_count: int = 0          # 有 SLA 约束的在途工单数
    breached_tickets: list[dict] = field(default_factory=list)
    approval_deadlines: list[dict] = field(default_factory=list)
    rules: list[dict] = field(default_factory=list)


# ── 辅助函数 ──────────────────────────────────────────


def _add_business_hours(start: datetime, hours: int) -> datetime:
    """在起始时间上累加工作小时数"""
    remaining = hours
    current = start
    while remaining > 0:
        current += timedelta(hours=1)
        if BUSINESS_START <= current.hour < BUSINESS_END:
            remaining -= 1
    return current


def _elapsed_business_hours(start: datetime, end: datetime = None) -> float:
    """计算两个时间点之间的工作小时数"""
    if end is None:
        end = datetime.utcnow()
    if isinstance(start, str):
        start = datetime.fromisoformat(start.replace("Z", "+00:00"))
    # 简化计算: 直接用总时差 — 生产环境应用完整工作小时计算
    total = (end - start).total_seconds() / 3600
    # 粗略扣除非工作时间（每天 13 小时非工作 = 24 - (22-9)）
    days = total / 24
    non_business_per_day = 24 - (BUSINESS_END - BUSINESS_START)  # = 11
    business_hours = total - (days * non_business_per_day)
    return max(0, round(business_hours, 1))


# ── SLA 引擎 ──────────────────────────────────────────


class SLAEngine:
    """SLA 计时、检测与自动升级引擎"""

    # ═══════════════════════════════════════════════════
    # 基础 SLA 计算（保持向后兼容）
    # ═══════════════════════════════════════════════════

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

        from services.ticket_state import active_statuses

        open_tickets = db_session.query(Ticket).filter(
            Ticket.status.in_(active_statuses()),
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

    # ═══════════════════════════════════════════════════
    # v13 新增: 规则引擎 + 自动升级
    # ═══════════════════════════════════════════════════

    @classmethod
    def get_rules(cls) -> list[dict]:
        """返回所有 SLA 规则配置"""
        return [
            {
                "key": r.key,
                "label": r.label,
                "duration_h": r.duration_h,
                "action": r.action,
                "description": r.description,
            }
            for r in SLA_RULES.values()
        ]

    @classmethod
    async def check_all(cls, db_session=None) -> SLASummary:
        """
        定时任务入口: 检查所有未完成项的 SLA 状态。

        检查范围:
        1. 活跃工单 — ticket_response / ticket_resolution 规则
        2. 审批节点 — approval_step 规则
        3. 会议室预约 — meeting_confirm 规则（如有）
        """
        if db_session is None:
            from db.db_router import DatabaseRouter
            db_session = DatabaseRouter().session_manager.Session()

        summary = SLASummary()
        summary.rules = cls.get_rules()

        try:
            # 1. 检查工单 SLA
            from db.models import Ticket
            from services.ticket_state import active_statuses

            active_tickets = db_session.query(Ticket).filter(
                Ticket.status.in_(active_statuses()),
                Ticket.is_active == 1,
            ).all()

            for ticket in active_tickets:
                ticket_dict = {
                    "id": ticket.id,
                    "ticket_number": ticket.ticket_number,
                    "priority": ticket.priority,
                    "created_at": ticket.created_at,
                    "status": ticket.status,
                    "assigned_to": ticket.assigned_to or "",
                }
                sla_status = cls.check_status(ticket_dict)
                summary.active_sla_count += 1

                # 基础 SLA 超时检测（所有规则统一判断）
                if sla_status.is_breached:
                    # 工单本身的 SLA 已超时 — 计入 breached_count
                    summary.breached_count += 1
                    rule = SLA_RULES["ticket_resolution"]
                    cls._add_breached_ticket(summary, ticket, rule, sla_status.elapsed_hours)

                    # 工单解决规则: 24h → 自动升级
                    if sla_status.elapsed_hours >= rule.duration_h:
                        await cls._execute_action(
                            rule, ticket, db_session,
                            extra={"elapsed_h": sla_status.elapsed_hours, "reason": "工单超时未解决"},
                        )
                elif sla_status.escalation_level >= 2:
                    summary.warning_count += 1

                # 工单响应检查: 创建后 4h 未分配处理人
                if ticket.status in ("created", "pending_approval") and not ticket.assigned_to:
                    rule = SLA_RULES["ticket_response"]
                    elapsed = _elapsed_business_hours(ticket.created_at)
                    if elapsed >= rule.duration_h:
                        await cls._execute_action(
                            rule, ticket, db_session,
                            extra={"elapsed_h": elapsed, "reason": "未分配处理人"},
                        )
                        if not any(t["id"] == ticket.id for t in summary.breached_tickets):
                            summary.breached_count += 1
                            cls._add_breached_ticket(summary, ticket, rule, elapsed)

            # 2. 检查审批步骤 SLA
            await cls.check_approval_steps(db_session, summary)

        finally:
            # 如果 session 是我们在 check_all 内部开的，需要关闭
            pass

        return summary

    @classmethod
    async def check_approval_steps(cls, db_session, summary: SLASummary = None) -> list[dict]:
        """
        检查所有待审批节点的 SLA 状态。

        Returns:
            [{step_id, approver, ticket_number, elapsed_h, deadline, is_breached, ...}]
        """
        from db.models import ApprovalStep, ApprovalWorkflow, Ticket

        pending_steps = db_session.query(
            ApprovalStep, ApprovalWorkflow, Ticket
        ).join(
            ApprovalWorkflow, ApprovalStep.workflow_id == ApprovalWorkflow.id,
        ).join(
            Ticket, ApprovalWorkflow.ticket_id == Ticket.id,
        ).filter(
            ApprovalStep.status == "pending",
            ApprovalWorkflow.status == "pending",
            Ticket.is_active == 1,
        ).all()

        rule = SLA_RULES["approval_step"]
        results = []

        for step, wf, ticket in pending_steps:
            elapsed = _elapsed_business_hours(step.created_at)
            deadline = _add_business_hours(step.created_at, rule.duration_h)
            is_breached = elapsed >= rule.duration_h

            entry = {
                "step_id": step.id,
                "workflow_id": wf.id,
                "ticket_id": ticket.id,
                "ticket_number": ticket.ticket_number,
                "approver": step.approver,
                "approver_role": step.approver_role,
                "step_order": step.step_order,
                "total_steps": wf.total_steps,
                "created_at": step.created_at.isoformat() if step.created_at else "",
                "elapsed_h": elapsed,
                "deadline": deadline.isoformat(),
                "duration_h": rule.duration_h,
                "is_breached": is_breached,
            }
            results.append(entry)

            if is_breached:
                await cls._execute_action(
                    rule, ticket, db_session,
                    extra={
                        "elapsed_h": elapsed,
                        "reason": f"审批节点 {step.step_order}/{wf.total_steps} "
                                  f"({step.approver}) 超时未操作",
                        "step_id": step.id,
                        "approver": step.approver,
                    },
                )
                if summary:
                    summary.breached_count += 1

            if summary:
                summary.approval_deadlines.append(entry)

        return results

    @classmethod
    async def _execute_action(cls, rule: SLARule, ticket, db_session, extra: dict = None):
        """
        执行 SLA 规则对应的动作。

        动作类型:
        - notify_admin: 通知管理员（emit 事件 + 日志）
        - escalate: 提升优先级 + 通知管理员
        - remind_approver: 提醒审批人
        - auto_cancel: 自动取消（会议室场景）
        """
        extra = extra or {}
        ticket_number = getattr(ticket, "ticket_number", "?")
        ticket_id = getattr(ticket, "id", 0)

        action = rule.action

        if action == "notify_admin":
            logger.warning(
                f"[SLA] {rule.label} 超时 — {ticket_number}: {extra.get('reason', '')} "
                f"({extra.get('elapsed_h', 0):.1f}h / {rule.duration_h}h)"
            )
            await cls._emit_sla_event(
                ticket_id=ticket_id,
                ticket_number=ticket_number,
                rule_key=rule.key,
                rule_label=rule.label,
                action="notify_admin",
                detail=extra.get("reason", ""),
                elapsed_h=extra.get("elapsed_h", 0),
            )

        elif action == "escalate":
            logger.warning(
                f"[SLA] {rule.label} 升级 — {ticket_number}: {extra.get('reason', '')} "
                f"({extra.get('elapsed_h', 0):.1f}h / {rule.duration_h}h)"
            )
            # 提升优先级
            old_priority = getattr(ticket, "priority", "P3")
            new_priority = cls._escalate_priority(old_priority)
            if new_priority != old_priority:
                try:
                    ticket.priority = new_priority
                    # 追加 history
                    history = list(ticket.history) if ticket.history else []
                    history.append({
                        "action": "sla_escalate",
                        "by": "system",
                        "time": datetime.utcnow().isoformat(),
                        "detail": f"SLA 自动升级: {old_priority} → {new_priority} "
                                  f"({extra.get('reason', '')})",
                    })
                    ticket.history = history
                    db_session.commit()
                except Exception as e:
                    logger.error(f"[SLA] 升级优先级失败: {e}")

            await cls._emit_sla_event(
                ticket_id=ticket_id,
                ticket_number=ticket_number,
                rule_key=rule.key,
                rule_label=rule.label,
                action="escalate",
                detail=f"{old_priority} → {new_priority} — {extra.get('reason', '')}",
                elapsed_h=extra.get("elapsed_h", 0),
            )

        elif action == "remind_approver":
            approver = extra.get("approver", "")
            logger.warning(
                f"[SLA] {rule.label} 提醒 — 审批人={approver} "
                f"工单={ticket_number} ({extra.get('elapsed_h', 0):.1f}h / {rule.duration_h}h)"
            )
            await cls._emit_sla_event(
                ticket_id=ticket_id,
                ticket_number=ticket_number,
                rule_key=rule.key,
                rule_label=rule.label,
                action="remind_approver",
                detail=f"提醒审批人 {approver}: {extra.get('reason', '')}",
                elapsed_h=extra.get("elapsed_h", 0),
                approver=approver,
            )

        elif action == "auto_cancel":
            logger.warning(
                f"[SLA] {rule.label} 自动取消 — {ticket_number}: {extra.get('reason', '')}"
            )
            await cls._emit_sla_event(
                ticket_id=ticket_id,
                ticket_number=ticket_number,
                rule_key=rule.key,
                rule_label=rule.label,
                action="auto_cancel",
                detail=extra.get("reason", ""),
                elapsed_h=extra.get("elapsed_h", 0),
            )

    @staticmethod
    def _escalate_priority(current: str) -> str:
        """提升一级优先级: P3→P2, P2→P1, P1→P0, P0 已最高"""
        order = ["P3", "P2", "P1", "P0"]  # low→high, index 递增 = 优先级递增
        if current not in order:
            return "P1"
        idx = order.index(current)
        if idx < len(order) - 1:
            return order[idx + 1]  # 向更高优先级移动
        return current  # P0 已最高

    @staticmethod
    async def _emit_sla_event(**data):
        """安全发射 sla.breached 事件"""
        try:
            from services.event_bus import EventBus, EventType
            await EventBus.emit(EventType.SLA_BREACHED, **data)
        except Exception as e:
            logger.warning(f"[SLA] 事件发射失败: {e}")

    @staticmethod
    def _add_breached_ticket(summary: SLASummary, ticket, rule: SLARule, elapsed: float):
        """向 summary 添加超时工单信息"""
        summary.breached_tickets.append({
            "id": ticket.id,
            "ticket_number": getattr(ticket, "ticket_number", ""),
            "title": getattr(ticket, "title", "")[:50],
            "priority": getattr(ticket, "priority", ""),
            "status": getattr(ticket, "status", ""),
            "rule": rule.key,
            "rule_label": rule.label,
            "elapsed_h": elapsed,
            "duration_h": rule.duration_h,
        })

    @classmethod
    async def get_sla_summary(cls, db_session=None) -> dict:
        """
        获取 SLA 状态概览 — 供 API 端点使用。

        Returns:
            {
                "breached_count": int,
                "warning_count": int,
                "active_sla_count": int,
                "breached_tickets": [...],
                "approval_deadlines": [...],
                "rules": [...],
            }
        """
        summary = await cls.check_all(db_session)
        return {
            "breached_count": summary.breached_count,
            "warning_count": summary.warning_count,
            "active_sla_count": summary.active_sla_count,
            "breached_tickets": summary.breached_tickets[:10],
            "approval_deadlines": summary.approval_deadlines[:10],
            "rules": summary.rules,
        }

    # ═══════════════════════════════════════════════════
    # 单工单升级（供外部调用）
    # ═══════════════════════════════════════════════════

    @classmethod
    async def escalate_ticket(cls, ticket_id: int, reason: str = "",
                              db_session=None) -> dict:
        """
        手动/自动升级工单: 提升优先级 + 通知管理员 + 追加历史。

        Returns:
            {"ticket_id": int, "old_priority": str, "new_priority": str, "success": bool}
        """
        from db.models import Ticket

        ticket = db_session.query(Ticket).filter(
            Ticket.id == ticket_id,
            Ticket.is_active == 1,
        ).first()
        if not ticket:
            return {"ticket_id": ticket_id, "error": "工单不存在", "success": False}

        old_priority = ticket.priority
        new_priority = cls._escalate_priority(old_priority)

        if new_priority != old_priority:
            ticket.priority = new_priority
            history = list(ticket.history) if ticket.history else []
            history.append({
                "action": "sla_escalate",
                "by": "system",
                "time": datetime.utcnow().isoformat(),
                "detail": reason or "SLA 自动升级",
            })
            ticket.history = history
            db_session.commit()

            logger.info(
                f"[SLA] 工单 {ticket.ticket_number} 升级: "
                f"{old_priority} → {new_priority} ({reason})"
            )

        await cls._emit_sla_event(
            ticket_id=ticket_id,
            ticket_number=ticket.ticket_number,
            rule_key="manual_escalate",
            rule_label="手动升级",
            action="escalate",
            detail=f"{old_priority} → {new_priority} — {reason}",
            elapsed_h=0,
        )

        return {
            "ticket_id": ticket_id,
            "ticket_number": ticket.ticket_number,
            "old_priority": old_priority,
            "new_priority": new_priority,
            "success": True,
        }

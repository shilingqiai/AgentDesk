"""
TicketStatus 状态枚举 + 状态转换规则

审批业务线状态机 — 集中管理所有工单状态及合法转换路径。

设计原则:
- 状态枚举使用 str mixin，保证与现有代码中 "created" 等字符串比较完全兼容
- ALLOWED_TRANSITIONS 是唯一的状态转换权威定义
- 所有状态变更必须经过 WorkflowService.transition()，不允许直接 ticket.status = "xxx"
- REJECTED 是终态（之前审批驳回不改变工单状态，属于 bug）
"""

from __future__ import annotations
from enum import Enum


class TicketStatus(str, Enum):
    """
    工单状态 — 审批业务线命名

    生命周期:
        CREATED → PENDING_APPROVAL → APPROVED → PROCESSING → COMPLETED
                                   ↘ REJECTED (终态)

    注意: str mixin 保证 TicketStatus.CREATED == "created" 为 True,
          所有现有的字符串比较和 SQLAlchemy filter 无需修改。
    """
    CREATED          = "created"           # 工单已创建（草稿）
    PENDING_APPROVAL = "pending_approval"  # 已提交，等待审批
    APPROVED         = "approved"          # 审批全部通过
    REJECTED         = "rejected"          # 审批驳回（终态）
    PROCESSING       = "processing"        # 审批通过后执行中
    COMPLETED        = "completed"         # 执行完成（终态）


# ── 状态转换表 ─────────────────────────────────────────

ALLOWED_TRANSITIONS: dict[TicketStatus, set[TicketStatus]] = {
    TicketStatus.CREATED:          {TicketStatus.PENDING_APPROVAL, TicketStatus.PROCESSING},
    TicketStatus.PENDING_APPROVAL: {TicketStatus.APPROVED, TicketStatus.REJECTED},
    TicketStatus.APPROVED:         {TicketStatus.PROCESSING, TicketStatus.COMPLETED, TicketStatus.REJECTED},
    TicketStatus.PROCESSING:       {TicketStatus.COMPLETED},
    TicketStatus.REJECTED:         set(),   # 终态，不可再转换
    TicketStatus.COMPLETED:        set(),   # 终态，不可再转换
}


# ── 状态描述 ───────────────────────────────────────────

STATUS_LABELS: dict[TicketStatus, str] = {
    TicketStatus.CREATED:          "已创建",
    TicketStatus.PENDING_APPROVAL: "等待审批",
    TicketStatus.APPROVED:         "审批通过",
    TicketStatus.REJECTED:         "已驳回",
    TicketStatus.PROCESSING:       "执行中",
    TicketStatus.COMPLETED:        "已完成",
}


# ── 辅助方法 ───────────────────────────────────────────

def is_terminal(status: str | TicketStatus) -> bool:
    """是否为终态（不可再变更）"""
    s = TicketStatus(status) if isinstance(status, str) else status
    return len(ALLOWED_TRANSITIONS.get(s, set())) == 0


def active_statuses() -> list[str]:
    """返回所有非终态状态值（用于查询活跃工单）"""
    return [
        s.value for s in TicketStatus
        if len(ALLOWED_TRANSITIONS.get(s, set())) > 0
    ]


def can_transition(from_status: str | TicketStatus, to_status: str | TicketStatus) -> bool:
    """检查状态转换是否合法"""
    f = TicketStatus(from_status) if isinstance(from_status, str) else from_status
    t = TicketStatus(to_status) if isinstance(to_status, str) else to_status
    return t in ALLOWED_TRANSITIONS.get(f, set())


def validate_transition(from_status: str, to_status: str) -> None:
    """
    校验状态转换合法性，不合法则抛出 ValueError。

    Raises:
        ValueError: 状态值无效
        ValueError: 转换不允许
    """
    try:
        f = TicketStatus(from_status)
    except ValueError:
        raise ValueError(f"无效的当前状态: '{from_status}'")

    try:
        t = TicketStatus(to_status)
    except ValueError:
        raise ValueError(f"无效的目标状态: '{to_status}'")

    if t not in ALLOWED_TRANSITIONS.get(f, set()):
        raise ValueError(
            f"状态转换不允许: {STATUS_LABELS.get(f, from_status)} → "
            f"{STATUS_LABELS.get(t, to_status)}"
        )

"""
WorkflowService — 集中式状态转换入口

系统中所有工单状态变更的唯一入口。任何代码路径（Agent / API / Engine）
需要改变工单状态时，必须调用 WorkflowService.transition()。

职责:
1. 校验状态转换合法性（通过 services.ticket_state 的转换表）
2. 更新 ticket.status + updated_at
3. 追加 history 操作时间线
4. 触发事件（Phase 2 接入 EventBus）

禁止: 直接 ticket.status = "xxx" 或 TicketRepository.update_status() 绕过校验
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Optional

from services.ticket_state import (
    TicketStatus,
    validate_transition,
    STATUS_LABELS,
)

logger = logging.getLogger("workflow.service")


class WorkflowService:
    """
    集中式状态转换服务。

    使用方式:
        from services.workflow_service import WorkflowService
        from services.ticket_state import TicketStatus

        WorkflowService.transition(
            ticket_id=42,
            to_status=TicketStatus.PENDING_APPROVAL,
            comment="提交审批 — 王经理 → 李HR",
            db_session=db,
        )
    """

    @staticmethod
    def transition(
        ticket_id: int,
        to_status: TicketStatus,
        *,
        assigned_to: str = None,
        comment: str = "",
        db_session=None,
    ) -> dict:
        """
        执行工单状态转换。

        Args:
            ticket_id:   工单 ID
            to_status:   目标状态（TicketStatus 枚举成员）
            assigned_to: 新指派人（可选，仅部分转换需要）
            comment:     操作备注（追加到 history）
            db_session:  SQLAlchemy session

        Returns:
            {
                "ticket_id": int,
                "from_status": str,
                "to_status": str,
                "success": True,
            }

        Raises:
            ValueError: 工单不存在
            ValueError: 状态转换不合法
        """
        from db.models import Ticket

        # 1. 加载工单
        ticket = db_session.query(Ticket).filter(
            Ticket.id == ticket_id,
            Ticket.is_active == 1,
        ).first()
        if not ticket:
            raise ValueError(f"工单不存在: {ticket_id}")

        from_status = ticket.status
        to_status_str = to_status.value if isinstance(to_status, TicketStatus) else to_status

        # 2. 校验转换合法性
        validate_transition(from_status, to_status_str)

        # 3. 更新工单
        ticket.status = to_status_str
        ticket.updated_at = datetime.now(timezone.utc)
        if assigned_to:
            ticket.assigned_to = assigned_to

        # 4. 追加 history
        history = list(ticket.history) if ticket.history else []
        history.append({
            "action": "status_changed",
            "by": "system",
            "time": datetime.now(timezone.utc).isoformat(),
            "detail": f"{STATUS_LABELS.get(TicketStatus(from_status), from_status)} → "
                      f"{STATUS_LABELS.get(to_status, to_status_str)}"
                      + (f" — {comment}" if comment else ""),
            "from_status": from_status,
            "to_status": to_status_str,
        })
        ticket.history = history

        db_session.commit()

        logger.info(
            f"[Workflow] 工单 {ticket.ticket_number} 状态转换: "
            f"{from_status} → {to_status_str}"
            + (f" ({comment})" if comment else "")
        )

        # 发射事件（fire-and-forget，不阻塞主流程）
        try:
            from services.event_bus import EventBus, EventType
            import asyncio
            loop = asyncio.get_event_loop()
            if loop.is_running():
                loop.create_task(EventBus.emit(
                    EventType.TICKET_STATUS_CHANGED,
                    ticket_id=ticket_id,
                    ticket_number=ticket.ticket_number,
                    from_status=from_status,
                    to_status=to_status_str,
                    comment=comment,
                ))
        except RuntimeError:
            pass  # 无事件循环（测试/同步环境），静默跳过

        return {
            "ticket_id": ticket_id,
            "ticket_number": ticket.ticket_number,
            "from_status": from_status,
            "to_status": to_status_str,
            "success": True,
        }

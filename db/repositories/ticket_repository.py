"""
工单数据访问层 — TicketRepository

提供工单的全生命周期 CRUD 操作：
- 创建工单（自动生成 ticket_number）
- 查询工单（按ID/编号/用户/类型/状态）
- 更新状态
- 统计分析
"""

from __future__ import annotations

import uuid
from typing import Optional, Any
from datetime import datetime

from ..base.session_manager import SessionManager
from ..models import Ticket


class TicketRepository:
    """
    工单数据仓库

    复用 SessionManager，提供统一的数据库访问。
    所有方法自动处理事务提交和回滚。
    """

    def __init__(self, session_manager: SessionManager):
        self.session_manager = session_manager

    # ============================================================
    # 工单号生成
    # ============================================================

    @staticmethod
    def _generate_ticket_number() -> str:
        """生成唯一工单号: TK-20260608-A1B2C3"""
        date_part = datetime.now().strftime("%Y%m%d")
        rand_part = uuid.uuid4().hex[:6].upper()
        return f"TK-{date_part}-{rand_part}"

    # ============================================================
    # CRUD
    # ============================================================

    def add_ticket(
        self,
        ticket_type: str = "it_fault",
        title: str = "",
        description: str = "",
        category: str = "其他",
        priority: str = "P2",
        status: str = "created",
        requester_id: str = "",
        requester_name: str = "",
        assigned_to: str = "",
        trace_id: str = "",
        payload: dict = None,
    ) -> dict:
        """
        创建工单

        Returns:
            ticket dict (包含生成的 id 和 ticket_number)
        """
        with self.session_manager.session_scope() as session:
            ticket = Ticket(
                ticket_number=self._generate_ticket_number(),
                ticket_type=ticket_type,
                title=title,
                description=description,
                category=category,
                priority=priority,
                status=status,
                requester_id=requester_id,
                requester_name=requester_name,
                assigned_to=assigned_to,
                trace_id=trace_id,
                payload=payload or {},
            )
            session.add(ticket)
            session.flush()
            return self._ticket_to_dict(ticket)

    def get_ticket(self, ticket_id: int) -> Optional[dict]:
        """按主键 ID 查询"""
        with self.session_manager.session_scope() as session:
            ticket = session.query(Ticket).filter(
                Ticket.id == ticket_id,
                Ticket.is_active == 1,
            ).first()
            return self._ticket_to_dict(ticket) if ticket else None

    def get_by_number(self, ticket_number: str) -> Optional[dict]:
        """按工单号查询"""
        with self.session_manager.session_scope() as session:
            ticket = session.query(Ticket).filter(
                Ticket.ticket_number == ticket_number,
                Ticket.is_active == 1,
            ).first()
            return self._ticket_to_dict(ticket) if ticket else None

    def list_tickets(
        self,
        ticket_type: str = None,
        status: str = None,
        priority: str = None,
        requester_id: str = None,
        limit: int = 50,
        offset: int = 0,
    ) -> list[dict]:
        """
        列表查询，支持多条件筛选

        Args:
            ticket_type: 筛选类型 (it_fault/leave/expense/admin)
            status: 筛选状态
            priority: 筛选优先级
            requester_id: 筛选申请人
            limit: 每页数量
            offset: 偏移量
        """
        with self.session_manager.session_scope() as session:
            query = session.query(Ticket).filter(Ticket.is_active == 1)

            if ticket_type:
                query = query.filter(Ticket.ticket_type == ticket_type)
            if status:
                query = query.filter(Ticket.status == status)
            if priority:
                query = query.filter(Ticket.priority == priority)
            if requester_id:
                query = query.filter(Ticket.requester_id == requester_id)

            query = query.order_by(Ticket.created_at.desc())
            tickets = query.offset(offset).limit(limit).all()
            return [self._ticket_to_dict(t) for t in tickets]

    def get_by_user(self, requester_id: str, limit: int = 20) -> list[dict]:
        """按申请人查询"""
        return self.list_tickets(requester_id=requester_id, limit=limit)

    def update_ticket(
        self,
        ticket_id: int,
        **updates,
    ) -> bool:
        """
        更新工单字段

        可更新字段: status, priority, assigned_to, title, description,
                     category, payload
        """
        allowed_fields = {
            "status", "priority", "assigned_to", "title",
            "description", "category", "payload", "ticket_type",
        }
        filtered = {k: v for k, v in updates.items() if k in allowed_fields}
        if not filtered:
            return False

        with self.session_manager.session_scope() as session:
            ticket = session.query(Ticket).filter(
                Ticket.id == ticket_id,
                Ticket.is_active == 1,
            ).first()
            if not ticket:
                return False

            for key, value in filtered.items():
                setattr(ticket, key, value)
            ticket.updated_at = datetime.utcnow()
            return True

    def update_status(
        self, ticket_id: int, new_status: str, assigned_to: str = None,
    ) -> bool:
        """更新工单状态（便捷方法）"""
        updates = {"status": new_status}
        if assigned_to:
            updates["assigned_to"] = assigned_to
        return self.update_ticket(ticket_id, **updates)

    def delete_ticket(self, ticket_id: int, soft_delete: bool = True) -> bool:
        """删除工单（默认软删除）"""
        with self.session_manager.session_scope() as session:
            ticket = session.query(Ticket).filter(
                Ticket.id == ticket_id,
            ).first()
            if not ticket:
                return False

            if soft_delete:
                ticket.is_active = 0
                ticket.updated_at = datetime.utcnow()
            else:
                session.delete(ticket)
            return True

    def get_ticket_count(self, ticket_type: str = None) -> int:
        """获取工单总数"""
        with self.session_manager.session_scope() as session:
            query = session.query(Ticket).filter(Ticket.is_active == 1)
            if ticket_type:
                query = query.filter(Ticket.ticket_type == ticket_type)
            return query.count()

    def get_stats(self) -> dict[str, Any]:
        """
        获取工单统计

        Returns:
            {
                "total": 总工单数,
                "by_type": {type: count},
                "by_status": {status: count},
                "by_priority": {priority: count},
                "today": 今日新增,
            }
        """
        with self.session_manager.session_scope() as session:
            from sqlalchemy import func

            base = session.query(Ticket).filter(Ticket.is_active == 1)

            total = base.count()

            # 按类型
            by_type_rows = (
                session.query(Ticket.ticket_type, func.count(Ticket.id))
                .filter(Ticket.is_active == 1)
                .group_by(Ticket.ticket_type).all()
            )
            by_type = {row[0]: row[1] for row in by_type_rows}

            # 按状态
            by_status_rows = (
                session.query(Ticket.status, func.count(Ticket.id))
                .filter(Ticket.is_active == 1)
                .group_by(Ticket.status).all()
            )
            by_status = {row[0]: row[1] for row in by_status_rows}

            # 按优先级
            by_priority_rows = (
                session.query(Ticket.priority, func.count(Ticket.id))
                .filter(Ticket.is_active == 1)
                .group_by(Ticket.priority).all()
            )
            by_priority = {row[0]: row[1] for row in by_priority_rows}

            # 今日新增
            today_start = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
            today_count = base.filter(Ticket.created_at >= today_start).count()

            return {
                "total": total,
                "by_type": by_type,
                "by_status": by_status,
                "by_priority": by_priority,
                "today": today_count,
            }

    # ============================================================
    # 辅助
    # ============================================================

    @staticmethod
    def _ticket_to_dict(ticket: Ticket) -> dict:
        """Ticket ORM → dict"""
        return {
            "id": ticket.id,
            "ticket_number": ticket.ticket_number,
            "ticket_type": ticket.ticket_type,
            "title": ticket.title,
            "description": ticket.description,
            "category": ticket.category,
            "priority": ticket.priority,
            "status": ticket.status,
            "requester_id": ticket.requester_id,
            "requester_name": ticket.requester_name,
            "assigned_to": ticket.assigned_to,
            "trace_id": ticket.trace_id,
            "payload": ticket.payload,
            "created_at": ticket.created_at.isoformat() if ticket.created_at else None,
            "updated_at": ticket.updated_at.isoformat() if ticket.updated_at else None,
            "is_active": bool(ticket.is_active),
        }

"""
TicketService — 唯一工单写入入口 (Single Write Entry)

所有工单创建路径（Chat Complex / Chat Dynamic / Card Confirm / REST API）
必须经过此 Service，保证数据契约一致。

核心职责:
1. 统一 requester_id / requester_name 赋值（来自 AgentContext）
2. 自动计算 current_approver + approver_chain
3. 初始化 history 操作时间线
4. 工单落库后自动触发 ApprovalWorkflow
5. 禁止硬编码 fallback 身份（"web_user" / "" / None）
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Optional

from services.agent_context import AgentContext
from services.ticket_state import TicketStatus
from services.workflow_service import WorkflowService
from services.event_bus import EventBus, EventType

logger = logging.getLogger("ticket.service")


class TicketService:
    """
    工单写入服务 — 系统中唯一的工单创建入口。

    使用方式:
        from services.ticket_service import TicketService
        from services.agent_context import AgentContext

        ctx = AgentContext(user_name="张三", role="employee")
        ticket = TicketService.create_ticket(ctx, {
            "ticket_type": "leave",
            "title": "申请4天年假",
            "description": "计划下周休年假",
            "category": "请假",
            "priority": "P2",
            "payload": {"total_days": 4, "start_date": "2026-06-22", "end_date": "2026-06-25"},
        })
    """

    # ── 审批链规则（与 services/approval_engine.py 同步）─────────

    _APPROVAL_CHAINS: dict[str, list[str]] = {
        "leave": ["王经理", "李HR"],
        "expense": ["王经理", "赵财务"],
        "purchase": ["王经理", "赵财务"],
    }

    @classmethod
    def get_approver_chain(cls, ticket_type: str, payload: dict = None) -> list[str]:
        """
        根据工单类型确定审批链（确定性规则，不经过 LLM）。

        Returns:
            审批人姓名列表，如 ["王经理", "李HR"]
            不需要审批的类型返回空列表。
        """
        if ticket_type in ("leave",):
            return cls._APPROVAL_CHAINS.get("leave", [])
        if ticket_type in ("expense",):
            return cls._APPROVAL_CHAINS.get("expense", [])
        if ticket_type in ("admin",):
            # 行政类工单：检查是否涉及采购
            svc = (payload or {}).get("service_type", "")
            if "采购" in svc or "procurement" in svc.lower():
                return cls._APPROVAL_CHAINS.get("purchase", [])
        if ticket_type in ("it_fault",):
            # IT 故障报修：无需审批，工单创建后自动进入处理阶段
            return []
        # 其他未知类型不需要审批
        return []

    @classmethod
    def get_current_approver(cls, ticket_type: str, payload: dict = None) -> Optional[str]:
        """获取审批链第一人（工单创建时的当前审批人）"""
        chain = cls.get_approver_chain(ticket_type, payload)
        return chain[0] if chain else None

    @classmethod
    def create_ticket(
        cls,
        context: AgentContext,
        params: dict,
        *,
        db_router=None,
    ) -> dict:
        """
        创建工单 — 系统唯一写入入口。

        Args:
            context: AgentContext 身份上下文（必填，已验证）
            params: 工单参数字典:
                - ticket_type (str):          it_fault | leave | expense | admin
                - title (str):                工单标题
                - description (str):          工单描述
                - category (str):             分类标签
                - priority (str):             P0/P1/P2/P3
                - payload (dict):             扩展字段（天数/金额/日期等）
                - trace_id (str):             调用链追踪 ID
                - primary_agent (str):        创建来源 agent name
                - assigned_to (str):          指派给（可选）
        Returns:
            ticket dict (包含 id, ticket_number, current_approver 等)
        Raises:
            ValueError: 身份无效时抛出
        """
        from db.db_router import DatabaseRouter

        # 1. 验证身份
        context.validate(allow_anonymous=False)

        ticket_type = params.get("ticket_type", "it_fault")
        payload = params.get("payload") or {}

        # 2. 计算审批链
        approver_chain = cls.get_approver_chain(ticket_type, payload)
        current_approver = approver_chain[0] if approver_chain else ""

        # 3. 初始化 history
        history = [{
            "action": "created",
            "by": context.user_name,
            "time": datetime.now(timezone.utc).isoformat(),
            "detail": f"工单创建 — {params.get('title', '')}",
        }]

        # 4. 写入 DB
        db = db_router or DatabaseRouter()
        try:
            ticket = db.ticket.add_ticket(
                ticket_type=ticket_type,
                title=params.get("title", ""),
                description=params.get("description", ""),
                category=params.get("category", "其他"),
                priority=params.get("priority", "P2"),
                requester_id=context.user_id,
                requester_name=context.user_name,
                assigned_to=params.get("assigned_to", ""),
                trace_id=params.get("trace_id", ""),
                current_approver=current_approver,
                approver_chain=approver_chain,
                history=history,
                payload=payload,
            )

            logger.info(
                f"[TicketService] 工单已创建: {ticket['ticket_number']} "
                f"type={ticket_type} requester={context.user_name} "
                f"chain={' → '.join(approver_chain) if approver_chain else '无审批'}"
            )

            # ── 事件: 工单创建 (fire-and-forget) ──
            import asyncio
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    loop.create_task(EventBus.emit(
                        EventType.TICKET_CREATED,
                        ticket_id=ticket["id"],
                        ticket_number=ticket["ticket_number"],
                        ticket_type=ticket_type,
                        requester_name=context.user_name,
                        has_approval=bool(approver_chain),
                        operator=context.user_name,
                    ))
            except RuntimeError:
                pass  # 无事件循环（测试/同步环境），静默跳过

            # 5. 触发审批流
            if approver_chain and ticket_type in ("leave", "expense"):
                cls._create_approval_workflow(
                    ticket_id=ticket["id"],
                    ticket_type=ticket_type,
                    payload=payload,
                    db=db,
                )
                # 状态转换: CREATED → PENDING_APPROVAL（工单进入审批流程）
                db_session2 = db.session_manager.Session()
                try:
                    WorkflowService.transition(
                        ticket_id=ticket["id"],
                        to_status=TicketStatus.PENDING_APPROVAL,
                        comment=f"提交审批 — {' → '.join(approver_chain)}",
                        db_session=db_session2,
                    )
                    ticket["status"] = TicketStatus.PENDING_APPROVAL.value
                finally:
                    db_session2.close()
            elif not approver_chain:
                # 无需审批的类型 (IT故障等): CREATED → PROCESSING (自动进入处理阶段)
                db_session2 = db.session_manager.Session()
                try:
                    WorkflowService.transition(
                        ticket_id=ticket["id"],
                        to_status=TicketStatus.PROCESSING,
                        comment="无需审批，自动进入处理阶段",
                        db_session=db_session2,
                    )
                    ticket["status"] = TicketStatus.PROCESSING.value
                finally:
                    db_session2.close()

            return ticket

        finally:
            if db_router is None:  # 只在未传入 db_router 时关闭
                db.close()

    @classmethod
    def _create_approval_workflow(
        cls,
        ticket_id: int,
        ticket_type: str,
        payload: dict,
        db,
    ):
        """创建审批流（与 services/approval_engine.py 桥接）"""
        try:
            from services.approval_engine import ApprovalEngine

            workflow_type = None
            amount = 0

            if ticket_type == "leave":
                workflow_type = "leave"
                amount = int(payload.get("total_days", 0) if isinstance(payload, dict) else 0)
            elif ticket_type in ("expense", "admin"):
                svc = (payload or {}).get("service_type", "")
                if ticket_type == "expense" or "采购" in svc or "procurement" in svc.lower():
                    workflow_type = "purchase"
                    amount = int(payload.get("amount", 0) if isinstance(payload, dict) else 0)

            if workflow_type is None:
                return

            db_session = db.session_manager.Session()
            try:
                ApprovalEngine.create_workflow(
                    ticket_id=ticket_id,
                    workflow_type=workflow_type,
                    amount=amount,
                    db_session=db_session,
                )
                logger.info(
                    f"[TicketService] 审批流已创建: ticket_id={ticket_id} "
                    f"type={workflow_type} amount={amount}"
                )
            finally:
                db_session.close()
        except Exception as e:
            logger.warning(f"[TicketService] 审批流创建失败 (非致命): {e}")

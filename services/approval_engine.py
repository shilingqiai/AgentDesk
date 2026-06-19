"""
审批流引擎 — 确定性审批链，不经过 LLM

设计原则:
- 审批链规则 100% 确定性，保证合规可审计
- 串行审批: 当前节点通过后自动推进到下一节点
- 任一节点驳回 → 整个审批流标记为 rejected
- 最后节点通过 → 审批流标记为 approved → 工单自动执行
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Optional

logger = logging.getLogger("approval.engine")

# ── 审批链规则（确定性，可配置）──────────────────────────

ApprovalChain = list[tuple[int | float, list[str]]]

APPROVAL_CHAINS: dict[str, ApprovalChain] = {
    "leave": [
        (99, ["department_manager", "hr"]),   # 固定: 王经理 → 李HR
    ],
    "purchase": [
        (99999, ["department_manager", "finance"]),  # 固定: 王经理 → 赵财务
    ],
}

# 默认审批人映射（原型阶段，生产需对接组织架构系统）
DEFAULT_APPROVERS: dict[str, str] = {
    "department_manager": "王经理",
    "hr": "李HR",
    "finance": "赵财务",
}


def build_approval_chain(workflow_type: str, amount: int | float = 0) -> list[str]:
    """
    根据类型和金额确定审批链。

    Args:
        workflow_type: leave | expense | procurement
        amount: 天数（请假）或金额（报销/采购）

    Returns:
        审批角色列表，如 ["department_manager", "hr"]
    """
    chains = APPROVAL_CHAINS.get(workflow_type)
    if not chains:
        return ["department_manager"]

    for threshold, roles in chains:
        if amount <= threshold:
            return roles
    return chains[-1][1]


def resolve_approver(role: str) -> str:
    """将审批角色解析为具体审批人姓名"""
    return DEFAULT_APPROVERS.get(role, role)


class ApprovalEngine:
    """审批流引擎"""

    @staticmethod
    def create_workflow(
        ticket_id: int,
        workflow_type: str,
        amount: int | float = 0,
        db_session=None,
    ) -> dict:
        """
        为工单创建审批流。

        Returns:
            {"workflow_id": int, "steps": [...], "status": "pending"}
        """
        from db.models import ApprovalWorkflow, ApprovalStep

        chain = build_approval_chain(workflow_type, amount)

        workflow = ApprovalWorkflow(
            ticket_id=ticket_id,
            workflow_type=workflow_type,
            current_step=0,
            total_steps=len(chain),
            status="pending",
        )
        db_session.add(workflow)
        db_session.flush()  # 获取 workflow.id

        steps = []
        for i, role in enumerate(chain, 1):
            step = ApprovalStep(
                workflow_id=workflow.id,
                step_order=i,
                approver=resolve_approver(role),
                approver_role=role,
                status="pending",
            )
            db_session.add(step)
            steps.append({
                "step_order": i,
                "approver": step.approver,
                "approver_role": role,
            })

        db_session.commit()

        logger.info(
            f"[Approval] 审批流已创建: workflow_id={workflow.id}, "
            f"ticket_id={ticket_id}, type={workflow_type}, "
            f"chain={' → '.join(chain)}"
        )
        return {
            "workflow_id": workflow.id,
            "ticket_id": ticket_id,
            "workflow_type": workflow_type,
            "total_steps": len(chain),
            "steps": steps,
            "status": "pending",
        }

    @staticmethod
    def approve_step(workflow_id: int, step_order: int, comment: str = "",
                     db_session=None) -> dict:
        """
        审批通过当前节点 → 推进到下一步或完成。

        Returns:
            {"workflow_status": "pending"|"approved", "next_step": int|None}
        """
        from db.models import ApprovalWorkflow, ApprovalStep

        workflow = db_session.query(ApprovalWorkflow).filter(
            ApprovalWorkflow.id == workflow_id,
        ).first()
        if not workflow:
            raise ValueError(f"审批流不存在: {workflow_id}")
        if workflow.status != "pending":
            raise ValueError(f"审批流已结束: {workflow.status}")

        step = db_session.query(ApprovalStep).filter(
            ApprovalStep.workflow_id == workflow_id,
            ApprovalStep.step_order == step_order,
        ).first()
        if not step:
            raise ValueError(f"审批节点不存在: wf={workflow_id} step={step_order}")
        if step.status != "pending":
            raise ValueError(f"该节点已处理: {step.status}")

        step.status = "approved"
        step.comment = comment
        step.decided_at = datetime.now(timezone.utc)

        # 判断是否到达最终节点
        if step_order >= workflow.total_steps:
            workflow.status = "approved"
            workflow.updated_at = datetime.now(timezone.utc)
            db_session.commit()

            # 更新关联工单状态为 processing + 追加 history
            from db.models import Ticket
            ticket = db_session.query(Ticket).filter(
                Ticket.id == workflow.ticket_id,
            ).first()
            if ticket:
                if ticket.status == "created":
                    ticket.status = "processing"
                ticket.current_approver = ""
                _append_ticket_history(ticket, "approved", step.approver,
                                       f"全部审批通过 (共 {workflow.total_steps} 步)")

            db_session.commit()
            logger.info(f"[Approval] 审批流 {workflow_id} 全部通过 → approved")
            return {"workflow_status": "approved", "next_step": None}

        # 推进到下一步
        workflow.current_step = step_order
        workflow.updated_at = datetime.now(timezone.utc)
        db_session.commit()

        next_step = step_order + 1
        next_approver = db_session.query(ApprovalStep).filter(
            ApprovalStep.workflow_id == workflow_id,
            ApprovalStep.step_order == next_step,
        ).first()

        # 更新工单 current_approver
        if next_approver:
            from db.models import Ticket
            ticket = db_session.query(Ticket).filter(
                Ticket.id == workflow.ticket_id,
            ).first()
            if ticket:
                ticket.current_approver = next_approver.approver
                _append_ticket_history(ticket, "step_approved", step.approver,
                                       f"步骤 {step_order}/{workflow.total_steps} 通过 → {next_approver.approver}")
            db_session.commit()

        logger.info(
            f"[Approval] 审批流 {workflow_id} step {step_order} 通过, "
            f"→ step {next_step} ({next_approver.approver if next_approver else '完成'})"
        )
        return {
            "workflow_status": "pending",
            "next_step": next_step,
            "next_approver": next_approver.approver if next_approver else None,
            "next_approver_role": next_approver.approver_role if next_approver else None,
        }

    @staticmethod
    def reject_step(workflow_id: int, step_order: int, comment: str = "",
                    db_session=None) -> dict:
        """
        驳回当前节点 → 整个审批流标记为 rejected。
        """
        from db.models import ApprovalWorkflow, ApprovalStep

        workflow = db_session.query(ApprovalWorkflow).filter(
            ApprovalWorkflow.id == workflow_id,
        ).first()
        if not workflow:
            raise ValueError(f"审批流不存在: {workflow_id}")

        step = db_session.query(ApprovalStep).filter(
            ApprovalStep.workflow_id == workflow_id,
            ApprovalStep.step_order == step_order,
        ).first()
        if not step:
            raise ValueError(f"审批节点不存在")

        step.status = "rejected"
        step.comment = comment
        step.decided_at = datetime.now(timezone.utc)

        workflow.status = "rejected"
        workflow.updated_at = datetime.now(timezone.utc)
        db_session.commit()

        # 更新工单 current_approver + history
        from db.models import Ticket
        ticket = db_session.query(Ticket).filter(
            Ticket.id == workflow.ticket_id,
        ).first()
        if ticket:
            ticket.current_approver = ""
            _append_ticket_history(ticket, "rejected", step.approver,
                                   comment or f"步骤 {step_order}/{workflow.total_steps} 被驳回")
        db_session.commit()

        logger.info(
            f"[Approval] 审批流 {workflow_id} step {step_order} 被驳回: {comment[:60]}"
        )
        return {"workflow_status": "rejected", "reason": comment}

    @staticmethod
    def get_pending_approvals(approver: str, db_session=None) -> list[dict]:
        """
        获取指定审批人的待审批列表（含可见但不可操作项）。

        企业 OA 核心设计：visibility ≠ actionability。
        - 链上所有审批人从工单创建起就能看到
        - 但只有 current_step 对应的审批人能操作

        返回每个 item 包含:
        - actionable: bool — 当前是否可审批
        - current_approver: str — 当前持有审批权的人（用于"等待 XX 审批"展示）
        """
        from db.models import ApprovalWorkflow, ApprovalStep, Ticket

        steps = db_session.query(ApprovalStep, ApprovalWorkflow, Ticket).join(
            ApprovalWorkflow, ApprovalStep.workflow_id == ApprovalWorkflow.id,
        ).join(
            Ticket, ApprovalWorkflow.ticket_id == Ticket.id,
        ).filter(
            ApprovalStep.approver == approver,
            ApprovalStep.status == "pending",
            ApprovalWorkflow.status == "pending",
        ).order_by(ApprovalStep.created_at.desc()).all()

        results = []
        for step, wf, ticket in steps:
            # 当前可操作 = 该步骤是审批流的"当前步骤"
            is_actionable = (step.step_order == wf.current_step + 1)

            # 查找当前持有审批权的人（用于"等待 XX 审批"展示）
            current_holder = ""
            if not is_actionable:
                current_step_order = wf.current_step + 1
                current_step = db_session.query(ApprovalStep).filter(
                    ApprovalStep.workflow_id == wf.id,
                    ApprovalStep.step_order == current_step_order,
                    ApprovalStep.status == "pending",
                ).first()
                if current_step:
                    current_holder = current_step.approver

            results.append({
                "step_id": step.id,
                "workflow_id": wf.id,
                "ticket_id": ticket.id,
                "ticket_number": ticket.ticket_number,
                "ticket_type": ticket.ticket_type,
                "title": ticket.title,
                "requester": ticket.requester_name,
                "workflow_type": wf.workflow_type,
                "step_order": step.step_order,
                "total_steps": wf.total_steps,
                "actionable": is_actionable,
                "current_approver": current_holder,
                "created_at": step.created_at.isoformat() if step.created_at else "",
            })
        return results

    @staticmethod
    def get_workflow_status(ticket_id: int, db_session=None) -> Optional[dict]:
        """查询工单的审批流状态"""
        from db.models import ApprovalWorkflow, ApprovalStep

        wf = db_session.query(ApprovalWorkflow).filter(
            ApprovalWorkflow.ticket_id == ticket_id,
        ).order_by(ApprovalWorkflow.created_at.desc()).first()

        if not wf:
            return None

        steps = db_session.query(ApprovalStep).filter(
            ApprovalStep.workflow_id == wf.id,
        ).order_by(ApprovalStep.step_order).all()

        return {
            "workflow_id": wf.id,
            "status": wf.status,
            "current_step": wf.current_step,
            "total_steps": wf.total_steps,
            "steps": [{
                "step_order": s.step_order,
                "approver": s.approver,
                "approver_role": s.approver_role,
                "status": s.status,
                "comment": s.comment,
                "decided_at": s.decided_at.isoformat() if s.decided_at else None,
            } for s in steps],
        }


def _append_ticket_history(ticket, action: str, by: str, detail: str = ""):
    """向工单的 history 列追加操作记录（原地修改 ticket 对象）"""
    from datetime import datetime, timezone

    history = list(ticket.history) if ticket.history else []
    history.append({
        "action": action,
        "by": by,
        "time": datetime.now(timezone.utc).isoformat(),
        "detail": detail,
    })
    ticket.history = history

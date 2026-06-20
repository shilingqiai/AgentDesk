"""
审批流 API — 审批人查看、通过、驳回
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

router = APIRouter(prefix="/api/approvals", tags=["审批流"])


class ApproveRequest(BaseModel):
    step_id: int
    comment: str = ""
    approver_name: str = Field(default="", description="操作人姓名（用于权限校验）")


class RejectRequest(BaseModel):
    step_id: int
    comment: str = ""
    approver_name: str = Field(default="", description="操作人姓名（用于权限校验）")


def _get_db():
    """获取数据库 session"""
    from db.db_router import DatabaseRouter
    db_router = DatabaseRouter()
    return db_router.session_manager.Session()


@router.get("/pending")
async def list_pending(approver: str = Query(..., description="审批人姓名")):
    """审批人查看待审批列表"""
    from services.approval_engine import ApprovalEngine

    db = _get_db()
    try:
        items = ApprovalEngine.get_pending_approvals(approver, db_session=db)
        return {"success": True, "count": len(items), "items": items}
    finally:
        db.close()


@router.post("/approve")
async def approve(req: ApproveRequest):
    """通过审批节点"""
    from services.approval_engine import ApprovalEngine
    from db.models import ApprovalStep

    db = _get_db()
    try:
        step = db.query(ApprovalStep).filter(ApprovalStep.id == req.step_id).first()
        if not step:
            raise HTTPException(status_code=404, detail="审批节点不存在")

        result = ApprovalEngine.approve_step(
            workflow_id=step.workflow_id,
            step_order=step.step_order,
            comment=req.comment,
            approver_name=req.approver_name,
            db_session=db,
        )
        return {"success": True, **result}
    except PermissionError as e:
        raise HTTPException(status_code=403, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    finally:
        db.close()


@router.post("/reject")
async def reject(req: RejectRequest):
    """驳回审批节点"""
    from services.approval_engine import ApprovalEngine
    from db.models import ApprovalStep

    db = _get_db()
    try:
        step = db.query(ApprovalStep).filter(ApprovalStep.id == req.step_id).first()
        if not step:
            raise HTTPException(status_code=404, detail="审批节点不存在")

        result = ApprovalEngine.reject_step(
            workflow_id=step.workflow_id,
            step_order=step.step_order,
            comment=req.comment,
            approver_name=req.approver_name,
            db_session=db,
        )
        return {"success": True, **result}
    except PermissionError as e:
        raise HTTPException(status_code=403, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    finally:
        db.close()


@router.get("/status/{ticket_id}")
async def get_workflow_status(ticket_id: int):
    """查询工单的审批流状态"""
    from services.approval_engine import ApprovalEngine

    db = _get_db()
    try:
        status = ApprovalEngine.get_workflow_status(ticket_id, db_session=db)
        if status is None:
            return {"success": True, "workflow": None, "message": "该工单无审批流"}
        return {"success": True, "workflow": status}
    finally:
        db.close()


@router.get("/chains")
async def list_approval_chains():
    """
    获取所有审批链配置及实时状态。

    Admin 面板使用此接口渲染审批链流程图。
    返回每种审批类型的审批链规则 + 当前在途工单数。
    """
    from services.approval_engine import APPROVAL_CHAINS, DEFAULT_APPROVERS
    from db.models import ApprovalWorkflow

    db = _get_db()
    try:
        chains = {}
        for wf_type, thresholds in APPROVAL_CHAINS.items():
            # 取第一个（也是唯一一个）阈值的角色列表
            roles = thresholds[0][1] if thresholds else []
            steps = []
            for role in roles:
                steps.append({
                    "role": role,
                    "name": DEFAULT_APPROVERS.get(role, role),
                })

            # 统计该类型审批流在途数量
            active_count = db.query(ApprovalWorkflow).filter(
                ApprovalWorkflow.workflow_type == wf_type,
                ApprovalWorkflow.status == "pending",
            ).count()

            chains[wf_type] = {
                "name": _chain_name(wf_type),
                "steps": steps,
                "active_count": active_count,
            }

        # 无审批类型（IT 故障等）
        chains["it_fault"] = {
            "name": "IT 故障报修",
            "steps": [],
            "active_count": 0,
        }

        return {"success": True, "chains": chains}
    finally:
        db.close()


def _chain_name(wf_type: str) -> str:
    """审批类型 → 中文名称"""
    names = {
        "leave": "请假审批",
        "purchase": "采购 / 报销审批",
    }
    return names.get(wf_type, wf_type)

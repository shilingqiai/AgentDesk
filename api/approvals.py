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


class RejectRequest(BaseModel):
    step_id: int
    comment: str = ""


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
            db_session=db,
        )
        return {"success": True, **result}
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
            db_session=db,
        )
        return {"success": True, **result}
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

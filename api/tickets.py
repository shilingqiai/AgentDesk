"""
工单管理 API

提供多类型工单的 CRUD 和统计端点。
支持 IT故障 / 请假 / 报销 / 行政服务 四类工单。
"""

from __future__ import annotations

from typing import Optional
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field
from db.db_router import DatabaseRouter

router = APIRouter(prefix="/api/tickets", tags=["工单管理"])


# ============================================================
# 请求/响应模型
# ============================================================

class TicketCreateRequest(BaseModel):
    """创建工单请求"""
    model_config = {"extra": "allow"}

    user_input: str = Field(..., description="用户输入文本")
    ticket_type: Optional[str] = Field(default=None, description="手动指定类型")
    priority: Optional[str] = Field(default="P2", description="优先级 P0/P1/P2/P3")
    user_id: Optional[str] = Field(default="", description="申请人ID")
    skip_card: Optional[bool] = Field(default=False, description="卡片确认时跳过卡片逻辑直接创建")
    title: Optional[str] = Field(default=None)
    description: Optional[str] = Field(default=None)
    category: Optional[str] = Field(default=None)
    payload: Optional[dict] = Field(default=None)


class TicketStatusUpdate(BaseModel):
    """状态更新请求"""
    status: str = Field(..., description="新状态: created/assigned/processing/resolved/closed")
    assigned_to: Optional[str] = Field(default=None, description="指派人")


# ============================================================
# 端点
# ============================================================

@router.get("/", summary="获取工单列表")
async def list_tickets(
    ticket_type: Optional[str] = Query(default=None, description="类型筛选"),
    status: Optional[str] = Query(default=None, description="状态筛选"),
    priority: Optional[str] = Query(default=None, description="优先级筛选"),
    limit: int = Query(default=50, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
):
    """获取工单列表，支持多条件筛选"""
    try:
        db = DatabaseRouter()
        tickets = db.ticket.list_tickets(
            ticket_type=ticket_type,
            status=status,
            priority=priority,
            limit=limit,
            offset=offset,
        )
        total = db.ticket.get_ticket_count(ticket_type=ticket_type)
        return {
            "status": "success",
            "data": tickets,
            "total": total,
            "limit": limit,
            "offset": offset,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取工单列表失败: {str(e)}")


@router.get("/stats", summary="获取工单统计")
async def get_ticket_stats():
    """获取工单的多维度统计"""
    try:
        db = DatabaseRouter()
        stats = db.ticket.get_stats()
        return {
            "status": "success",
            "data": stats,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取统计失败: {str(e)}")


@router.get("/{ticket_id}", summary="获取工单详情")
async def get_ticket(ticket_id: int):
    """按 ID 获取单条工单"""
    try:
        db = DatabaseRouter()
        ticket = db.ticket.get_ticket(ticket_id)
        if not ticket:
            raise HTTPException(status_code=404, detail="工单不存在")
        return {
            "status": "success",
            "data": ticket,
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取工单失败: {str(e)}")


@router.post("/", summary="创建工单")
async def create_ticket(request: TicketCreateRequest):
    """创建工单 — 卡片确认时 skip_card=True 直接写库"""
    try:
        # 卡片确认模式：直接创建工单，不走 LLM 卡片循环
        if request.skip_card and request.ticket_type:
            from datetime import datetime as dt
            db = DatabaseRouter()

            payload = (request.payload or {}).copy()
            extra_fields = request.model_extra or {}
            for k, v in extra_fields.items():
                if v is not None and v != "" and k not in payload:
                    payload[k] = v

            type_category = {
                "it_fault": "IT故障", "leave": "请假",
                "expense": "报销", "admin": "行政服务",
            }.get(request.ticket_type, "其他")

            ticket = db.ticket.add_ticket(
                ticket_type=request.ticket_type,
                title=request.title or request.user_input[:30],
                description=request.description or request.user_input,
                category=request.category or type_category,
                priority=request.priority or "P2",
                requester_id=request.user_id or "",
                trace_id=f"card_{dt.utcnow().strftime('%Y%m%d%H%M%S')}",
                payload=payload,
            )

            return {
                "status": "success",
                "data": {
                    "ticket_id": ticket["id"],
                    "ticket_number": ticket["ticket_number"],
                    "ticket_type": ticket["ticket_type"],
                    "status": ticket["status"],
                    "priority": ticket["priority"],
                    "response": f"工单 {ticket['ticket_number']} 已创建成功！",
                },
            }

        # LLM 模式
        from agents.sub_agents.ticket_dispatch import TicketDispatchSubAgent
        from agents.a2a.protocol import AgentMessage

        agent = TicketDispatchSubAgent()
        message = AgentMessage.create_delegation(
            from_agent="api",
            to_agent="ticket_dispatch",
            payload={
                "user_input": request.user_input,
                "task": "创建工单",
                "urgency": "medium",
                "user_id": request.user_id or "",
            },
        )
        result = await agent.execute(message)

        if result.success and not result.payload.get("return_card"):
            return {
                "status": "success",
                "data": {
                    "ticket_id": result.payload.get("ticket_id"),
                    "ticket_number": result.payload.get("ticket_number"),
                    "ticket_type": result.payload.get("ticket_type"),
                    "status": result.payload.get("status"),
                    "priority": result.payload.get("priority"),
                    "response": result.payload.get("direct_response"),
                },
            }
        elif result.success and result.payload.get("return_card"):
            return {
                "status": "success",
                "data": {"card": result.payload.get("card")},
            }
        else:
            raise HTTPException(status_code=500, detail=result.error or "工单创建失败")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"创建工单失败: {str(e)}")


@router.put("/{ticket_id}/status", summary="更新工单状态")
async def update_ticket_status(ticket_id: int, update: TicketStatusUpdate):
    """更新工单状态"""
    try:
        db = DatabaseRouter()
        success = db.ticket.update_status(
            ticket_id=ticket_id,
            new_status=update.status,
            assigned_to=update.assigned_to,
        )
        if not success:
            raise HTTPException(status_code=404, detail="工单不存在")
        return {
            "status": "success",
            "message": f"工单状态更新为: {update.status}",
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"更新失败: {str(e)}")


@router.delete("/{ticket_id}", summary="删除工单（软删除）")
async def delete_ticket(ticket_id: int):
    """软删除工单"""
    try:
        db = DatabaseRouter()
        success = db.ticket.delete_ticket(ticket_id, soft_delete=True)
        if not success:
            raise HTTPException(status_code=404, detail="工单不存在")
        return {
            "status": "success",
            "message": "工单已删除",
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"删除失败: {str(e)}")

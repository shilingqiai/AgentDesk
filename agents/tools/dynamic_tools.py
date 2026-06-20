"""
Dynamic Action Agent 工具集 — 统一注册到 ToolRegistry

将 DynamicActionAgent 中原内联定义的 7 个工具迁移至 ToolRegistry，
消除双轨制。6 个工具具有独立 handler，create_ticket 由 Agent 内联处理
（因为需要 A2A 委派和提案/执行上下文）。

分类:
- ops: 运营操作 (check_inventory, search_meeting_rooms)
- knowledge: 知识检索 (search_knowledge_base, web_search)
- ticket: 工单相关 (check_ticket_status, check_leave_balance)

使用方式:
    from agents.tools import tool_registry
    from agents.tools import dynamic_tools  # 导入即注册

    schemas = tool_registry.get_tools_as_openai_schemas()
"""

from __future__ import annotations

import json
import logging
from agents.tools import tool_registry

logger = logging.getLogger("agent.tools.dynamic")


# ═══════════════════════════════════════════════════════
# 1. check_inventory — 库存查询
# ═══════════════════════════════════════════════════════

@tool_registry.register(
    name="check_inventory",
    description=(
        "查询办公物品库存数量。参数 keyword: 物品关键词，支持模糊搜索"
        "（如'笔记本''显示器''ThinkPad''键鼠'）。"
        "返回匹配物品的库存数量、最低阈值、单价、供应商信息。"
        "库存数 ≤ 最低阈值时为预警状态，需考虑采购。"
    ),
    parameters={
        "keyword": {"type": "string", "description": "物品关键词，如'笔记本''显示器''ThinkPad''键鼠套装'", "required": True},
    },
    category="ops",
)
async def check_inventory(keyword: str) -> dict:
    """查询库存 — SQLite inventory_items 表，支持模糊搜索"""
    keyword = keyword.strip()
    if not keyword:
        return {"found": False, "error": "请提供物品关键词"}

    try:
        from db.db_router import DatabaseRouter
        from db.models import InventoryItem

        db = DatabaseRouter()
        session = db.session_manager.Session()
        try:
            # 1. 精确匹配
            exact = session.query(InventoryItem).filter(
                InventoryItem.item_name == keyword,
                InventoryItem.is_active == 1,
            ).first()
            if exact:
                return {
                    "found": True, "match_type": "exact",
                    "item_name": exact.item_name, "category": exact.category,
                    "stock": exact.stock, "min_threshold": exact.min_threshold,
                    "available": exact.stock > 0,
                    "low_stock": 0 < exact.stock <= exact.min_threshold,
                    "unit_price": exact.unit_price,
                    "supplier": exact.supplier,
                }

            # 2. 模糊搜索
            fuzzy = session.query(InventoryItem).filter(
                InventoryItem.item_name.like(f"%{keyword}%"),
                InventoryItem.is_active == 1,
            ).all()

            if fuzzy:
                results = []
                for item in fuzzy:
                    results.append({
                        "item_name": item.item_name, "category": item.category,
                        "stock": item.stock, "min_threshold": item.min_threshold,
                        "available": item.stock > 0,
                        "low_stock": 0 < item.stock <= item.min_threshold,
                        "unit_price": item.unit_price,
                        "supplier": item.supplier,
                    })
                return {
                    "found": True, "match_type": "fuzzy",
                    "keyword": keyword, "count": len(results),
                    "items": results,
                }

            # 3. 无匹配
            return {
                "found": False, "keyword": keyword,
                "message": f"库存中未找到与'{keyword}'匹配的物品。请尝试不同的关键词，或提交采购申请。",
            }
        finally:
            session.close()
    except Exception as e:
        logger.error(f"[dynamic_tools] 库存查询失败: {e}")
        return {"found": False, "error": str(e)}


# ═══════════════════════════════════════════════════════
# 2. check_leave_balance — 假期余额（别名）
# ═══════════════════════════════════════════════════════

@tool_registry.register(
    name="check_leave_balance",
    description="查询员工的假期余额。参数 user_name: 员工姓名。返回年假/病假等各类假期余额。",
    parameters={
        "user_name": {"type": "string", "description": "员工姓名", "required": True},
    },
    category="ticket",
)
async def check_leave_balance(user_name: str) -> dict:
    """查询假期余额 — 委托给 builtin leave_balance_query"""
    from agents.tools import tool_registry as tr
    result = await tr.invoke("leave_balance_query", employee_name=user_name)
    if result.success and result.data:
        data = result.data
        return {
            "user_name": user_name,
            "annual_leave_total": data.get("annual_leave_total", 15),
            "annual_leave_used": data.get("annual_leave_used", 0),
            "annual_leave_remaining": data.get("annual_leave_remaining", 15),
            "sick_leave_remaining": data.get("sick_leave_remaining", 5),
        }
    return {"user_name": user_name, "error": "查询失败"}


# ═══════════════════════════════════════════════════════
# 3. check_ticket_status — 工单状态查询（别名）
# ═══════════════════════════════════════════════════════

@tool_registry.register(
    name="check_ticket_status",
    description="查询工单的处理状态。参数 ticket_number: 工单号。返回工单状态、处理人、进度。",
    parameters={
        "ticket_number": {"type": "string", "description": "工单号", "required": True},
    },
    category="ticket",
)
async def check_ticket_status(ticket_number: str) -> dict:
    """查询工单状态 — 委托给 builtin ticket_status"""
    from agents.tools import tool_registry as tr
    result = await tr.invoke("ticket_status", ticket_number=ticket_number)
    if result.success and result.data:
        return result.data
    return {"found": False, "ticket_number": ticket_number, "message": "未找到该工单"}


# ═══════════════════════════════════════════════════════
# 4. search_knowledge_base — 知识库检索
# ═══════════════════════════════════════════════════════

@tool_registry.register(
    name="search_knowledge_base",
    description="搜索企业知识库。参数 query: 搜索问题。返回相关文档内容和来源。"
                "适用于查询政策、流程、故障排查方法等。",
    parameters={
        "query": {"type": "string", "description": "搜索查询", "required": True},
    },
    category="knowledge",
)
async def search_knowledge_base(query: str) -> dict:
    """搜索企业知识库"""
    try:
        from services.knowledge_service import KnowledgeService
        ks = KnowledgeService()
        await ks.initialize()
        docs = await ks.search(query, top_k=3)
        if not docs:
            return {"found": False, "query": query, "message": "未找到相关知识"}
        results = []
        for d in docs:
            results.append({
                "category": d.get("category", ""),
                "score": d.get("score", 0),
                "content": d.get("content", "")[:300],
            })
        return {"found": True, "query": query, "count": len(docs), "results": results}
    except Exception as e:
        return {"error": f"知识库搜索失败: {e}"}


# ═══════════════════════════════════════════════════════
# 5. search_meeting_rooms — 会议室搜索
# ═══════════════════════════════════════════════════════

@tool_registry.register(
    name="search_meeting_rooms",
    description="搜索可用会议室。参数: date(日期YYYY-MM-DD), start_time(HH:MM), end_time(HH:MM), "
                "capacity(可选,人数需求)。返回可用会议室列表及设备信息。",
    parameters={
        "date": {"type": "string", "description": "日期 YYYY-MM-DD", "required": True},
        "start_time": {"type": "string", "description": "开始时间 HH:MM", "required": True},
        "end_time": {"type": "string", "description": "结束时间 HH:MM", "required": True},
        "capacity": {"type": "integer", "description": "人数需求（可选）", "required": False},
    },
    category="ops",
)
async def search_meeting_rooms(date: str, start_time: str, end_time: str, capacity: int = 0) -> dict:
    """搜索可用会议室"""
    try:
        from db.db_router import DatabaseRouter
        from db.models import MeetingRoom, MeetingRoomBooking

        db = DatabaseRouter()
        session = db.session_manager.Session()
        try:
            rooms_q = session.query(MeetingRoom).filter(
                MeetingRoom.is_active == 1, MeetingRoom.status == "available"
            )
            if capacity > 0:
                rooms_q = rooms_q.filter(MeetingRoom.capacity >= capacity)
            rooms = rooms_q.all()

            available = []
            for room in rooms:
                conflict = session.query(MeetingRoomBooking).filter(
                    MeetingRoomBooking.room_id == room.id,
                    MeetingRoomBooking.date == date,
                    MeetingRoomBooking.status == "confirmed",
                    MeetingRoomBooking.is_active == 1,
                    MeetingRoomBooking.start_time < end_time,
                    MeetingRoomBooking.end_time > start_time,
                ).first()
                available.append({
                    "room_id": room.id, "name": room.name,
                    "capacity": room.capacity, "location": room.location,
                    "available": not bool(conflict),
                    "conflict": conflict.title if conflict else None,
                    "amenities": room.amenities if hasattr(room, 'amenities') else [],
                })
            return {"date": date, "time": f"{start_time}-{end_time}", "rooms": available}
        finally:
            session.close()
    except Exception as e:
        return {"error": f"会议室查询失败: {e}"}


# ═══════════════════════════════════════════════════════
# 6. create_ticket — 工单创建（schema 注册，handler 由 Agent 内联）
# ═══════════════════════════════════════════════════════

@tool_registry.register(
    name="create_ticket",
    description="创建工单（请假/报销/IT故障/行政等）。会先生成确认卡片请用户核对，确认后才落库。",
    category="ticket",
    parameters={
        "ticket_type": {
            "type": "string",
            "description": "工单类型: leave(请假) / expense(报销) / it_fault(IT故障) / admin(行政) / purchase(采购)",
            "required": True,
            "enum": ["leave", "expense", "it_fault", "admin", "purchase"],
        },
        "title": {
            "type": "string",
            "description": "工单标题，一句话概括",
            "required": True,
        },
        "description": {
            "type": "string",
            "description": "详细描述（请假原因/故障现象/报销事由等）",
            "required": True,
        },
        "priority": {
            "type": "string",
            "description": "优先级: P0(紧急) / P1(高) / P2(中) / P3(低)",
            "required": False,
            "enum": ["P0", "P1", "P2", "P3"],
        },
        "extra": {
            "type": "object",
            "description": "扩展字段: 请假日期/报销金额/设备型号等",
            "required": False,
        },
    },
)
async def create_ticket_stub(**kwargs) -> dict:
    """
    Stub — 实际执行由 DynamicActionAgent._execute_tool() 内联路由到
    _tool_create_ticket（需要 A2A 委派 + 提案/执行上下文）。
    此注册仅用于提供 OpenAI Function Calling 参数 schema。
    """
    return {"status": "error", "message": "create_ticket 应通过 Agent 内联处理，请勿直接调用此 stub"}

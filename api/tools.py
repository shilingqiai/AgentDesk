"""
工具发现 API

提供 Agent Tool Registry 的工具清单查询，供 Admin 面板和开发者使用。
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query
from typing import Optional

router = APIRouter(prefix="/api/tools", tags=["工具注册中心"])


@router.get("/", summary="获取已注册工具列表")
async def list_tools(category: Optional[str] = Query(default=None, description="按分类筛选: ops|knowledge|ticket|external|internal")):
    """
    返回 ToolRegistry 中所有已注册工具的摘要。

    可选按 category 筛选:
    - ops: 运营操作 (check_inventory, search_meeting_rooms)
    - knowledge: 知识检索 (search_knowledge_base, web_search)
    - ticket: 工单相关 (check_ticket_status, check_leave_balance)
    - external: 外部服务 (web_search)
    - internal: 内部工具 (ticket_status, leave_balance_query)
    """
    try:
        # 确保所有工具已注册
        import agents.tools.builtin_tools as _bt  # noqa: F401
        import agents.tools.dynamic_tools as _dt  # noqa: F401
        from agents.tools import tool_registry

        tools = tool_registry.list_tools(category=category)
        result = []
        for t in tools:
            result.append({
                "name": t.name,
                "description": t.description[:120],
                "category": t.category,
                "parameters": {
                    k: {
                        "type": v.get("type", "string"),
                        "description": v.get("description", ""),
                        "required": v.get("required", False),
                    }
                    for k, v in t.parameters.items()
                },
            })

        return {
            "success": True,
            "count": len(result),
            "tools": result,
            "categories": list(set(t.category for t in tools)),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取工具列表失败: {str(e)}")

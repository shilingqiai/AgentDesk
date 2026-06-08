"""
示例工具集 — 展示 Agent Tool Use 模式

包含：
- weather_query: 天气查询（模拟 MCP 外部服务调用）
- date_calculator: 日期计算（演示本地工具）
- ticket_status: 工单状态查询（演示内部工具）

这些工具可通过 tool_registry 被 Agent 发现和调用。
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta
from agents.tools import tool_registry

logger = logging.getLogger("agent.tools.builtin")


# ============================================================
# 天气查询工具（模拟外部 API）
# ============================================================

# 模拟天气数据
_MOCK_WEATHER = {
    "北京": {"temp": 28, "weather": "晴", "humidity": 45, "wind": "南风 3级"},
    "上海": {"temp": 25, "weather": "多云转小雨", "humidity": 72, "wind": "东风 4级"},
    "深圳": {"temp": 31, "weather": "雷阵雨", "humidity": 85, "wind": "西南风 3级"},
    "杭州": {"temp": 26, "weather": "阴", "humidity": 65, "wind": "北风 2级"},
    "成都": {"temp": 23, "weather": "小雨", "humidity": 78, "wind": "无持续风向"},
    "广州": {"temp": 30, "weather": "多云", "humidity": 70, "wind": "南风 3级"},
    "武汉": {"temp": 27, "weather": "晴转多云", "humidity": 55, "wind": "东北风 2级"},
    "南京": {"temp": 24, "weather": "阴转小雨", "humidity": 68, "wind": "东风 3级"},
}


@tool_registry.register(
    name="weather_query",
    description="查询指定城市的实时天气信息（温度、天气状况、湿度、风力）",
    parameters={
        "city": {"type": "string", "description": "城市名称，如'北京'、'上海'", "required": True},
    },
    category="external",
)
async def weather_query(city: str) -> dict:
    """查询城市天气（模拟外部 API 调用）"""
    city = city.strip()

    # 模糊匹配
    for known_city in _MOCK_WEATHER:
        if known_city in city or city in known_city:
            data = _MOCK_WEATHER[known_city]
            return {
                "city": known_city,
                "query_time": datetime.now().strftime("%Y-%m-%d %H:%M"),
                **data,
            }

    # 未知城市 → 模拟返回
    return {
        "city": city,
        "query_time": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "temp": 25,
        "weather": "晴",
        "humidity": 50,
        "wind": "微风",
        "note": f"（{city} 天气数据为模拟值）",
    }


# ============================================================
# 日期计算工具（本地工具）
# ============================================================

@tool_registry.register(
    name="date_calculator",
    description="计算未来/过去的日期，如'3天后'、'下周五'、'2周后'",
    parameters={
        "expression": {"type": "string", "description": "日期表达式，如'+3d'（3天后）、'-1w'（1周前）、'next_friday'", "required": True},
    },
    category="internal",
)
async def date_calculator(expression: str) -> dict:
    """日期计算"""
    today = datetime.now()

    expr = expression.strip().lower()

    # "next_friday" / "next_monday" 等
    weekday_map = {
        "monday": 0, "tuesday": 1, "wednesday": 2, "thursday": 3,
        "friday": 4, "saturday": 5, "sunday": 6,
        "周一": 0, "周二": 1, "周三": 2, "周四": 3,
        "周五": 4, "周六": 5, "周日": 6,
    }

    if expr.startswith("next_"):
        target_wd_name = expr[5:]
        if target_wd_name in weekday_map:
            target_wd = weekday_map[target_wd_name]
            days_ahead = target_wd - today.weekday()
            if days_ahead <= 0:
                days_ahead += 7
            target = today + timedelta(days=days_ahead)
            return {"date": target.strftime("%Y-%m-%d"), "weekday": target_wd_name}

    # "+Nd" / "-Nd" / "+Nw" 等
    import re
    match = re.match(r'([+-])(\d+)([dw])', expr)
    if match:
        sign = 1 if match.group(1) == "+" else -1
        num = int(match.group(2))
        unit = match.group(3)

        if unit == "d":
            target = today + timedelta(days=sign * num)
        elif unit == "w":
            target = today + timedelta(weeks=sign * num)
        else:
            target = today

        wd_names = ["周一", "周二", "周三", "周四", "周五", "周六", "周日"]
        return {"date": target.strftime("%Y-%m-%d"), "weekday": wd_names[target.weekday()]}

    return {"error": f"无法解析表达式: {expression}", "date": today.strftime("%Y-%m-%d")}


# ============================================================
# 工单状态查询工具（内部工具）
# ============================================================

@tool_registry.register(
    name="ticket_status",
    description="根据工单编号查询工单状态",
    parameters={
        "ticket_number": {"type": "string", "description": "工单编号，如TK-20260608-001", "required": True},
    },
    category="internal",
)
async def ticket_status(ticket_number: str) -> dict:
    """查询工单状态"""
    try:
        from db.db_router import DatabaseRouter
        db = DatabaseRouter()
        tickets = db.ticket.list_tickets(limit=100)
        for ticket in tickets:
            if ticket.get("ticket_number") == ticket_number:
                return {
                    "found": True,
                    "ticket_number": ticket["ticket_number"],
                    "title": ticket.get("title", ""),
                    "status": ticket.get("status", ""),
                    "priority": ticket.get("priority", ""),
                    "created_at": ticket.get("created_at", ""),
                }
        return {"found": False, "ticket_number": ticket_number, "message": "未找到该工单"}
    except Exception as e:
        return {"found": False, "error": str(e)}


# ============================================================
# 自动注册
# ============================================================

def _register_all():
    """确保所有工具已导入并注册"""
    pass  # 模块加载时 @tool_registry.register 装饰器已自动注册

_register_all()

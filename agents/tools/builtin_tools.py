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

# ============================================================
# 请假余额查询工具（模拟 HR 系统 API）
# ============================================================

_MOCK_LEAVE_BALANCES = {
    "张三": {"annual_total": 15, "annual_used": 2, "sick_remaining": 5},
    "李四": {"annual_total": 10, "annual_used": 9, "sick_remaining": 3},
    "王五": {"annual_total": 20, "annual_used": 18, "sick_remaining": 2},
    "赵六": {"annual_total": 15, "annual_used": 0, "sick_remaining": 5},
}


@tool_registry.register(
    name="leave_balance_query",
    description="查询员工年假和病假余额（模拟 HR 系统 API）",
    parameters={
        "employee_name": {"type": "string", "description": "员工姓名，如'张三'", "required": True},
    },
    category="internal",
)
async def leave_balance_query(employee_name: str) -> dict:
    """查询员工假期余额（模拟 HR 系统调用）"""
    employee_name = employee_name.strip()

    # 精确匹配或模糊匹配
    data = _MOCK_LEAVE_BALANCES.get(employee_name)
    if not data:
        for name in _MOCK_LEAVE_BALANCES:
            if employee_name in name or name in employee_name:
                data = _MOCK_LEAVE_BALANCES[name]
                employee_name = name
                break

    if not data:
        return {
            "employee_name": employee_name,
            "annual_leave_total": 10,
            "annual_leave_used": 3,
            "annual_leave_remaining": 7,
            "sick_leave_remaining": 5,
            "note": f"（{employee_name} 余额数据为默认模拟值，请确认）",
        }

    return {
        "employee_name": employee_name,
        "annual_leave_total": data["annual_total"],
        "annual_leave_used": data["annual_used"],
        "annual_leave_remaining": data["annual_total"] - data["annual_used"],
        "sick_leave_remaining": data["sick_remaining"],
        "query_time": datetime.now().strftime("%Y-%m-%d %H:%M"),
    }


# ============================================================
# 网页搜索工具 (v7 新增 — 供 DynamicActionAgent ReAct 使用)
# ============================================================

@tool_registry.register(
    name="web_search",
    description=(
        "在网上搜索信息，获取市场资讯、产品推荐、价格对比等外部数据。"
        "适用于：产品型号推荐、市场价格查询、技术方案对比、供应商信息搜索。"
        "返回搜索结果摘要和来源链接。"
    ),
    parameters={
        "query": {"type": "string", "description": "搜索查询词，如'2026年前端开发推荐显示器型号'", "required": True},
    },
    category="external",
)
async def web_search(query: str) -> dict:
    """
    网页搜索 — 优先使用内置 WebSearch 工具，fallback 为模拟数据。

    生产环境下可接入 SerpAPI / Bing Search API / Tavily 等。
    当前为演示版，对特定查询词返回预设结果，保证演示效果稳定。
    """
    import re as _re

    query_lower = query.lower()

    # ── 显示器推荐 ──
    if any(kw in query_lower for kw in ("显示器", "monitor", "屏幕")):
        return {
            "query": query,
            "results": [
                {
                    "title": "2026年前端开发者最佳显示器推荐",
                    "snippet": (
                        "Dell U2723QE 4K IPS Black — 27英寸4K分辨率，色彩准确度ΔE<2，"
                        "USB-C 90W供电，非常适合前端开发的色彩还原需求。"
                        "参考价 ¥4,299。"
                    ),
                    "url": "https://www.zhihu.com/question/display-for-developers",
                },
                {
                    "title": "LG 27UP850N 评测：性价比之选",
                    "snippet": (
                        "LG 27UP850N 4K 27英寸，支持硬件校色，色域覆盖DCI-P3 95%，"
                        "Type-C 96W反向充电。参考价 ¥3,499，性价比极高。"
                    ),
                    "url": "https://post.smzdm.com/p/lg-27up850n-review/",
                },
                {
                    "title": "小米Redmi 4K显示器27英寸 — 入门首选",
                    "snippet": (
                        "小米Redmi 27英寸4K显示器，ΔE<1出厂校色，Type-C 65W，"
                        "Pantone认证。参考价 ¥1,699，适合预算有限场景。"
                    ),
                    "url": "https://www.mi.com/redmi-monitor-4k",
                },
            ],
            "source": "web_search",
        }

    # ── 键盘/外设推荐 ──
    if any(kw in query_lower for kw in ("键盘", "keyboard", "机械键盘", "键鼠", "鼠标", "外设")):
        return {
            "query": query,
            "results": [
                {
                    "title": "程序员机械键盘横评 2026",
                    "snippet": (
                        "Filco Majestouch 2 茶轴 — 经典之作，手感稳定，办公打字首选。参考价 ¥1,199。"
                        "Leopold FC900RBT 红轴 — 无线双模，PBT键帽，做工精致。参考价 ¥999。"
                        "HHKB Professional Hybrid Type-S — 静电容，极致手感，适合Vim用户。参考价 ¥2,199。"
                    ),
                    "url": "https://www.chiphell.com/thread-keyboard-2026",
                },
                {
                    "title": "2026年人体工学办公鼠标推荐",
                    "snippet": (
                        "罗技 MX Master 3S — 8K DPI，电磁滚轮，跨设备Flow。参考价 ¥699。"
                        "罗技 MX Vertical — 人体工学垂直握持，缓解手腕疲劳。参考价 ¥599。"
                    ),
                    "url": "https://post.smzdm.com/p/mouse-recommend-2026",
                },
            ],
            "source": "web_search",
        }

    # ── 笔记本推荐 ──
    if any(kw in query_lower for kw in ("笔记本", "laptop", "电脑", "macbook", "thinkpad")):
        return {
            "query": query,
            "results": [
                {
                    "title": "2026年程序员笔记本选购指南",
                    "snippet": (
                        "MacBook Pro 16\" M4 Pro — 适合前端/iOS开发，续航优秀，屏幕极佳。参考价 ¥19,999起。"
                        "ThinkPad X1 Carbon Gen 12 — Windows旗舰商务本，键盘手感最佳，企业标配首选。参考价 ¥14,999。"
                        "MacBook Pro 14\" M4 — 轻薄便携，性能足够大多数开发场景。参考价 ¥14,999起。"
                    ),
                    "url": "https://www.zhihu.com/question/dev-laptop-2026",
                },
            ],
            "source": "web_search",
        }

    # ── 通用搜索 fallback ──
    return {
        "query": query,
        "results": [
            {
                "title": f"搜索结果: {query}",
                "snippet": (
                    f"关于「{query}」的搜索：建议参考相关专业论坛(ZH/ChipHell)、"
                    f"产品评测网站(SMZDM)和企业供应商目录。详细参数请查阅对应品牌官网。"
                ),
                "url": f"https://www.google.com/search?q={query.replace(' ', '+')}",
            },
        ],
        "source": "web_search",
        "note": "当前为模拟数据。生产环境请接入 SerpAPI 或 Tavily Search API。",
    }


def _register_all():
    """确保所有工具已导入并注册"""
    pass  # 模块加载时 @tool_registry.register 装饰器已自动注册

_register_all()

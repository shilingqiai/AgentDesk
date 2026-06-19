"""
企业工具集 — Agent Tool Use 模式

包含：
- ticket_status: 工单状态查询
- leave_balance_query: 假期余额查询（模拟 HR 系统）
- web_search: 网页搜索（模拟外部搜索，演示 MCP 外部服务调用）

这些工具可通过 tool_registry 被 Agent 发现和调用。
"""

from __future__ import annotations

import logging
from datetime import datetime
from agents.tools import tool_registry

logger = logging.getLogger("agent.tools.builtin")


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

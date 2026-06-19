"""
DynamicActionAgent — ReAct 循环驱动的自由编排 Agent

思路来自 Anthropic 的 Tool Use 最佳实践和 LangGraph ReAct 模式。

核心改进 (v8: 旧轨道已移除)：
  旧(v6): Router → action_query → ToolAgent(单次调用) → 返回
  旧(v6): Router → action_create → TicketDispatch(创建工单) → 返回
  complex保留: 请假/报销固定DAG（查政策+查余额→合规检查→确认卡片）

  新: Router → dynamic → DynamicActionAgent(ReAct 循环):
        Thought → Act → Observation → Thought → Act → ... → Final Answer

  自由度的涌现:
    - LLM 通过 tool schema 理解每个工具的能力
    - 运行时自主决定调用序列和条件分支
    - 不需要任何代码预设"先查后建"、"先查政策再查余额"等路径
    - 新增工具只需注册到 tool_registry，Agent 自动获得该能力

  例子:
    用户:"帮我查鼠标库存，有的话建个领用工单"
    → Thought: 先查库存
    → Act: check_inventory("鼠标") → Observation: {stock: 12}
    → Thought: 库存>0，可以建单
    → Act: create_ticket("领用", "鼠标") → Observation: {ticket_id: "REQ-001"}
    → Final: "鼠标还有12个库存，已创建领用工单 REQ-001"

    用户:"帮我查下我的年假余额，如果够5天就帮我请下周一到周五的年假"
    → Thought: 先查余额
    → Act: check_leave_balance("张三") → Observation: {remaining: 8}
    → Thought: 8天>5天，余额充足。解析日期: 下周一到周五
    → Act: check_policy("年假") → Observation: {max_consecutive: 5}
    → Thought: 合规。创建请假申请
    → Act: create_ticket("leave", ...) → Observation: {ticket_id: "LEAVE-003"}
    → Final: "您有8天年假余额，5天连续休假符合政策，已提交申请 LEAVE-003"

YOU ARE A SUB-AGENT. DO NOT REPLY TO USER DIRECTLY.
"""

from __future__ import annotations

import json
import logging
from typing import AsyncGenerator, Literal

from pydantic import BaseModel, Field

from agents.base_sub_agent import BaseSubAgent
from agents.a2a.protocol import AgentMessage
from agents.orchestrator.agent_declaration import agent_declaration
from agents.orchestrator.agent_registry import agent_registry
from config.model_provider import create_chat_model

logger = logging.getLogger("agent.dynamic_action")


# ============================================================
# ReAct 循环状态
# ============================================================

class ReActState(BaseModel):
    """单次 ReAct 迭代的状态快照"""
    thought: str = ""              # LLM 的思考过程
    tool_name: str = ""            # 准备调用的工具名
    tool_args: dict = {}           # 工具参数
    observation: str = ""          # 工具返回结果
    is_final: bool = False         # 是否到达最终答案
    final_answer: str = ""         # 最终回答


# ============================================================
# 工具定义 — 统一注册到 tool_registry
# ============================================================

class ToolDefinition(BaseModel):
    """工具定义"""
    name: str
    description: str
    parameters: dict = Field(default_factory=dict)  # JSON Schema


DYNAMIC_ACTION_TOOLS = [
    ToolDefinition(
        name="check_inventory",
        description=(
            "查询办公物品库存数量。参数 keyword: 物品关键词，支持模糊搜索"
            "（如'笔记本''显示器''ThinkPad''键鼠'）。"
            "返回匹配物品的库存数量、最低阈值、单价、供应商信息。"
            "库存数 ≤ 最低阈值时为预警状态，需考虑采购。"
        ),
        parameters={
            "type": "object",
            "properties": {
                "keyword": {"type": "string", "description": "物品关键词，如'笔记本''显示器''ThinkPad''键鼠套装'"},
            },
            "required": ["keyword"],
        },
    ),
    ToolDefinition(
        name="web_search",
        description=(
            "在网上搜索信息，获取市场资讯、产品推荐、价格对比等外部数据。"
            "适用于：产品型号推荐、市场价格查询、技术方案对比、供应商信息搜索。"
            "返回搜索结果摘要和来源链接。"
        ),
        parameters={
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "搜索查询词，如'2026年前端开发推荐显示器型号'"},
            },
            "required": ["query"],
        },
    ),
    ToolDefinition(
        name="check_leave_balance",
        description="查询员工的假期余额。参数 user_name: 员工姓名。返回年假/病假等各类假期余额。",
        parameters={
            "type": "object",
            "properties": {
                "user_name": {"type": "string", "description": "员工姓名"},
            },
            "required": ["user_name"],
        },
    ),
    ToolDefinition(
        name="check_ticket_status",
        description="查询工单的处理状态。参数 ticket_number: 工单号。返回工单状态、处理人、进度。",
        parameters={
            "type": "object",
            "properties": {
                "ticket_number": {"type": "string", "description": "工单号"},
            },
            "required": ["ticket_number"],
        },
    ),
    ToolDefinition(
        name="search_knowledge_base",
        description="搜索企业知识库。参数 query: 搜索问题。返回相关文档内容和来源。"
                    "适用于查询政策、流程、故障排查方法等。",
        parameters={
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "搜索查询"},
            },
            "required": ["query"],
        },
    ),
    ToolDefinition(
        name="search_meeting_rooms",
        description="搜索可用会议室。参数: date(日期YYYY-MM-DD), start_time(HH:MM), end_time(HH:MM), "
                    "capacity(可选,人数需求)。返回可用会议室列表及设备信息。",
        parameters={
            "type": "object",
            "properties": {
                "date": {"type": "string", "description": "日期 YYYY-MM-DD"},
                "start_time": {"type": "string", "description": "开始时间 HH:MM"},
                "end_time": {"type": "string", "description": "结束时间 HH:MM"},
                "capacity": {"type": "integer", "description": "人数需求（可选）"},
            },
            "required": ["date", "start_time", "end_time"],
        },
    ),
    ToolDefinition(
        name="create_ticket",
        description=(
            "【提议阶段】提出一个工单创建建议。此工具不会立即创建工单，"
            "而是将提议记录下来。在所有信息收集完成后，系统会将提议合并为确认卡片，"
            "等待用户确认后才执行。\n"
            "同类工单会自动合并（如多个领用物品→一个领用单，多个采购物品→一个采购单）。\n\n"
            "参数说明:\n"
            "- ticket_type: it_fault(IT故障) | leave(请假) | expense(报销) | admin(行政)\n"
            "- title: 工单标题\n"
            "- description: 详细描述\n"
            "- priority: P0-P3（默认P2）\n"
            "- extra: 扩展字段，按 ticket_type 填写:\n"
            "  * leave: {\"leave_type\":\"病假|年假|事假|调休\", \"start_date\":\"YYYY-MM-DD\", "
            "\"end_date\":\"YYYY-MM-DD\", \"total_days\":N}\n"
            "    务必从对话历史中提取天数、日期、类型，不要遗漏！\n"
            "    例: 用户说\"请4天病假明天开始\" → total_days=4, leave_type=病假, start_date=明天\n"
            "  * expense: {\"expense_type\":\"差旅|办公|餐费|其他\", \"amount\":N, \"description\":\"...\"}\n"
            "  * admin: {\"service_type\":\"资产领用|asset_requisition|采购申请|procurement|...\"}\n"
            "  * it_fault: {\"category\":\"硬件|软件|网络|账号\", \"urgency\":\"high|medium|low\"}\n"
            "返回: 提议的工单摘要（状态=proposed，等待用户确认）"
        ),
        parameters={
            "type": "object",
            "properties": {
                "ticket_type": {
                    "type": "string",
                    "enum": ["it_fault", "leave", "expense", "admin"],
                },
                "title": {"type": "string"},
                "description": {"type": "string"},
                "priority": {
                    "type": "string",
                    "enum": ["P0", "P1", "P2", "P3"],
                },
                "extra": {"type": "object"},
            },
            "required": ["ticket_type", "title", "description"],
        },
    ),
]


# ============================================================
# DynamicActionAgent
# ============================================================

@agent_declaration(
    agent_id="dynamic_action",
    name="动态动作编排Agent",
    description=(
        "基于 ReAct 循环的自由编排Agent。拥有全部工具能力，"
        "能自主决定调用哪些工具、以什么顺序调用、根据中间结果做条件判断。"
        "替代旧的 action_query/action_create 轨道。complex 保留用于请假/报销固定 DAG。"
    ),
    capabilities=[
        "react_loop", "dynamic_orchestration", "tool_use",
        "inventory_query", "leave_balance", "ticket_creation",
        "knowledge_search", "meeting_room_booking",
    ],
    knowledge_domains=[
        "it_support", "hr_policy", "admin_service",
        "ticket_management", "inventory_management",
    ],
    priority=1,
)
class DynamicActionAgent(BaseSubAgent):
    """
    ReAct 循环驱动的动态编排 Agent

    核心流程 (LangGraph ReAct 模式):
      ┌─────────────┐    有 tool_calls     ┌─────────────┐
      │ agent_node  │ ─────────────────→   │ tools_node  │
      │ (LLM决策)   │                      │ (执行工具)   │
      └─────────────┘ ←────────────────── └─────────────┘
             │  无 tool_calls (最终回答)        │
             ↓                                  │
           END                           (循环回 agent_node)

    关键: 没有任何代码预设"先查库存再建工单"这类路径。
    LLM 通过 tool schema 理解工具，运行时自主编排。
    """

    agent_id = "dynamic_action"
    MAX_REACT_ITERATIONS = 15  # 防止无限循环 (qwen3.7-max 支持并行调用)

    def __init__(self):
        super().__init__()
        # 低温度确保工具调用格式正确
        self.llm = create_chat_model(temperature=0)
        # bind_tools — 这是自由度涌现的关键
        self._tool_schemas = self._build_tool_schemas()
        self.llm_with_tools = self.llm.bind_tools(
            self._tool_schemas, tool_choice="auto"
        )
        self._knowledge_service = None  # 懒加载
        self._inventory_seeded = False
        self._execution_mode = False  # True = 确认后实际创建工单
        self._db_router = None  # 懒加载（共享 DB 连接）
        self._db_session = None  # 共享 session factory
        # ★ v9: A2A 委派上下文 — 供 _tool_create_ticket 构造委派消息
        self._last_user_input = ""
        self._last_user_name = ""
        self._last_user_role = "employee"
        self._last_trace_id = ""
        # ★ v9: 提案缓存 — 每次图迭代时通过 graph_workflow 重置
        self._proposals = {}

    @property
    def db_router(self):
        """懒加载 DatabaseRouter（共享连接，避免每个方法独立创建 engine）"""
        if self._db_router is None:
            from db.db_router import DatabaseRouter
            self._db_router = DatabaseRouter()
        return self._db_router

    def _get_session(self):
        """获取共享数据库 session（通过 DatabaseRouter 的 SessionManager）"""
        return self.db_router.session_manager.Session()

    # ================================================================
    # 库存种子数据
    # ================================================================

    INVENTORY_SEED_DATA = [
        # 电子设备
        ("笔记本(ThinkPad X1 Carbon Gen12)", "电子设备", 8, 2, 14999, "联想(Lenovo)",
         "14英寸/32GB/1TB/Win11 Pro"),
        ("笔记本(MacBook Pro 16\" M4 Pro)", "电子设备", 5, 2, 19999, "苹果(Apple)",
         "16英寸/36GB/512GB/macOS"),
        ("笔记本(MacBook Pro 14\" M4)", "电子设备", 4, 2, 14999, "苹果(Apple)",
         "14英寸/24GB/512GB/macOS"),
        ("笔记本(MacBook Air 15\" M4)", "电子设备", 3, 1, 10999, "苹果(Apple)",
         "15英寸/16GB/256GB/macOS"),
        # 显示器
        ("显示器(Dell U2723QE 4K)", "电子设备", 2, 3, 4299, "戴尔(Dell)",
         "27英寸/4K/IPS Black/USB-C 90W"),
        ("显示器(LG 27UP850N 4K)", "电子设备", 0, 3, 3499, "LG",
         "27英寸/4K/DCI-P3 95%/Type-C 96W"),
        ("显示器(Apple Studio Display 5K)", "电子设备", 1, 1, 11499, "苹果(Apple)",
         "27英寸/5K/P3广色域/雷雳3"),
        ("显示器(Dell U3224KB 6K)", "电子设备", 1, 1, 21499, "戴尔(Dell)",
         "32英寸/6K/IPS Black/雷雳4"),
        # 外设
        ("键鼠套装(罗技 MX Keys+Master 3S)", "外设", 15, 5, 1399, "罗技(Logitech)",
         "无线/多设备/USB-C充电"),
        ("键鼠套装(罗技 MX Keys Mini+Anywhere 3S)", "外设", 10, 3, 1199, "罗技(Logitech)",
         "便携/无线/多设备"),
        ("机械键盘(Filco Majestouch 2 茶轴)", "外设", 3, 2, 1199, "Filco",
         "有线/全键/PBT键帽"),
        ("机械键盘(Leopold FC900RBT 红轴)", "外设", 2, 2, 999, "Leopold",
         "无线双模/PBT键帽/98键"),
        ("鼠标(罗技 MX Master 3S)", "外设", 8, 3, 699, "罗技(Logitech)",
         "8K DPI/电磁滚轮/USB-C"),
        ("鼠标(Apple Magic Trackpad)", "外设", 2, 1, 899, "苹果(Apple)",
         "Force Touch/无线/USB-C"),
        # 耳机
        ("耳机(Sony WH-1000XM5)", "外设", 6, 3, 2499, "索尼(Sony)",
         "降噪/无线/30小时续航"),
        ("耳机(Apple AirPods Pro 3)", "外设", 4, 2, 1899, "苹果(Apple)",
         "降噪/无线/USB-C"),
        # 办公家具
        ("人体工学椅(Herman Miller Aeron)", "办公家具", 1, 2, 8900, "Herman Miller",
         "网面/前倾/可调腰部支撑"),
        ("人体工学椅(Steelcase Leap V2)", "办公家具", 3, 2, 6500, "Steelcase",
         "LiveBack技术/4D扶手"),
        # 其他
        ("USB-C Hub(CalDigit TS4)", "外设", 5, 2, 2699, "CalDigit",
         "18端口/98W充电/2.5GbE"),
        ("4K HDMI线(2m)", "耗材", 50, 10, 49, "绿联(Ugreen)",
         "HDMI 2.1/48Gbps"),
        ("数位板(Wacom Intuos Pro M)", "外设", 2, 1, 2999, "Wacom",
         "8192级压感/蓝牙/触控环"),
    ]

    async def _ensure_inventory_seeded(self) -> None:
        """确保库存表有种子数据（首次启动时填充）"""
        if self._inventory_seeded:
            return

        try:
            from db.models import InventoryItem

            db = self._get_session()
            try:
                count = db.query(InventoryItem).filter(
                    InventoryItem.is_active == 1,
                ).count()

                if count == 0:
                    logger.info("[DynamicAction] 库存表为空，填充种子数据...")
                    for item_name, category, stock, threshold, price, supplier, desc in self.INVENTORY_SEED_DATA:
                        db.add(InventoryItem(
                            item_name=item_name, category=category,
                            stock=stock, min_threshold=threshold,
                            unit_price=price, supplier=supplier,
                            description=desc,
                        ))
                    db.commit()
                    logger.info(f"[DynamicAction] 已填充 {len(self.INVENTORY_SEED_DATA)} 条库存记录")
                else:
                    logger.info(f"[DynamicAction] 库存表已有 {count} 条记录，跳过种子数据")

                self._inventory_seeded = True
            finally:
                db.close()
        except Exception as e:
            logger.warning(f"[DynamicAction] 库存种子数据填充失败: {e}")

    # ================================================================
    # 工具 Schema 构建
    # ================================================================

    def _build_tool_schemas(self) -> list[dict]:
        """将 ToolDefinition 转为 OpenAI Function Calling 格式"""
        schemas = []
        for tool in DYNAMIC_ACTION_TOOLS:
            schemas.append({
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": {
                        "type": "object",
                        "properties": tool.parameters.get("properties", {}),
                        "required": tool.parameters.get("required", []),
                    },
                },
            })
        return schemas

    def _build_system_prompt(self, user_name: str = "") -> str:
        from datetime import datetime
        today = datetime.now().strftime("%Y-%m-%d")
        weekday = ["周一","周二","周三","周四","周五","周六","周日"][datetime.now().weekday()]
        user_context = f"Current user: {user_name}." if user_name else ""

        # v14: 用户记忆注入
        memory_text = ""
        if user_name:
            try:
                from services.user_memory import user_memory_store
                memory_text = user_memory_store.inject_memory_context(user_name)
            except Exception:
                pass

        return (
            "You are the enterprise AI service desk Dynamic Action Agent.\n\n"
            f"{user_context}\n"
            f"Today is {today} ({weekday}). Use this for date calculations.\n\n"
            + (f"## User Memory\n{memory_text}\n\n" if memory_text else "")
            + "## CRITICAL RULES\n\n"
            "### Rule 0: PARALLEL TOOL CALLS (MOST IMPORTANT)\n"
            "ALWAYS call multiple independent tools in ONE response to save iterations.\n"
            "Example: search_knowledge_base + check_inventory(laptop) + check_inventory(monitor)\n"
            "  + check_inventory(keyboard) ALL at once in a single LLM response.\n"
            "Each tool call in parallel counts as ONE iteration. You have 15 iterations max.\n"
            "⚠️ EXCEPTION: NEVER call create_ticket in parallel with other tools.\n"
            "  create_ticket triggers user confirmation → all other results would be stale.\n"
            "  Always call create_ticket ALONE in its own response.\n\n"
            "### Rule 0.5: TOPIC-AWARE CONTEXT MERGING\n"
            "Conversation history is provided for context. However:\n\n"
            "**Step 1 — Topic Check (ALWAYS DO THIS FIRST)**:\n"
            "  - Is the latest user message about the SAME topic as the conversation history?\n"
            "  - Same topic → use history to fill missing details (slot filling)\n"
            "  - DIFFERENT topic → IGNORE the history entirely, process as a fresh request\n\n"
            "**Step 2 — Same-Topic Merging** (only when Step 1 confirms same topic):\n"
            "  - Example: User said '我要请4天假' then '病假 明天开始'\n"
            "    → Merge: total_days=4 + leave_type=病假 + start=明天\n"
            "  - Example: User said '前端入职' then '要MacBook和显示器'\n"
            "    → Merge: 前端入职 + MacBook + 显示器\n\n"
            "**Step 3 — Different Topic** (when Step 1 detects topic switch):\n"
            "  - Example: History is about 请假 → user now says '帮我准备前端入职设备'\n"
            "    → This is about EQUIPMENT, not leave. Process as a fresh equipment request.\n"
            "    → Do NOT reference, merge, or continue the leave topic.\n"
            "  - Example: History is about 网络故障 → user now says '帮我查年假政策'\n"
            "    → This is about POLICY. Process as a fresh policy query.\n\n"
            "**Key principle**: The latest user message ALWAYS takes precedence.\n"
            "History ONLY supplements when it's clearly the same topic.\n"
            "When in doubt, treat as new topic.\n\n"
            "### Rule 1: Two-Phase Execution\n"
            "**Phase 1 - Gather Info**: Collect ALL necessary info in ONE PARALLEL batch.\n"
            "  - search_knowledge_base + ALL check_inventory calls in ONE response\n"
            "  - web_search only when items are out of stock or low\n"
            "  - DO NOT propose tickets until ALL info is gathered.\n\n"
            "**Phase 2 - Propose Tickets ONE AT A TIME**: Only after ALL info collected.\n"
            "  - Call create_ticket ONE AT A TIME (one per LLM response).\n"
            "  - Each create_ticket will interrupt for user confirmation immediately.\n"
            "  - After user confirms → you will see the result → call the next one.\n"
            "  - MERGE related items: all in-stock items in ONE requisition ticket,\n"
            "    all out-of-stock items in ONE procurement ticket.\n"
            "  - create_ticket is a PROPOSAL that triggers confirmation. NOT auto-executed.\n\n"
            "### Rule 2: Merge Same-Type Items\n"
            "All requisition items -> 1 merged ticket. All procurement items -> 1 merged ticket.\n\n"
            "### Rule 3: Do NOT execute. Only propose.\n"
            "create_ticket stores proposals. Cards shown to user for confirmation.\n\n"
            "## Available Tools\n"
            "- search_knowledge_base(query): search enterprise knowledge base\n"
            "- check_inventory(keyword): fuzzy search inventory. Call MULTIPLE in parallel\n"
            "- web_search(query): search external web for recommendations/prices\n"
            "- create_ticket(ticket_type=admin, title, description, extra): propose a ticket\n"
            "  service_type=asset_requisition OR procurement\n"
            "- check_leave_balance, search_meeting_rooms: other tools\n\n"
            "IMPORTANT: Use PARALLEL tool calls whenever possible. One LLM response = up to N tool calls."
        )

    # ================================================================
    # 工具执行器
    # ================================================================

    async def _execute_tool(self, name: str, args: dict) -> str:
        """
        执行工具并返回字符串结果（含分层错误处理 + 退避重试）。

        面试要点：Tool Calling 失败处理分四层 —
          - ValidationError  → 不重试，直接返回修正提示
          - TransientError   → 指数退避重试 3 次
          - DependencyError  → 重试 1 次后降级
          - UnknownError     → 重试 1 次后升级
        """
        from agents.tools.error_classifier import (
            classify_error, ToolErrorType,
            get_retry_config, format_tool_error_response,
        )
        import asyncio as _asyncio

        # ── 工具路由映射 ──
        _tool_map = {
            "check_inventory": self._tool_check_inventory,
            "web_search": self._tool_web_search,
            "check_leave_balance": self._tool_check_leave_balance,
            "check_ticket_status": self._tool_check_ticket_status,
            "search_knowledge_base": self._tool_search_knowledge,
            "search_meeting_rooms": self._tool_search_meeting_rooms,
            "create_ticket": self._tool_create_ticket,
        }

        tool_fn = _tool_map.get(name)
        if tool_fn is None:
            return json.dumps({"error": True, "error_type": "validation",
                               "message": f"未知工具: {name}",
                               "suggestion": "检查可用工具列表"}, ensure_ascii=False)

        # ── 分层重试循环 ──
        last_error = None
        last_error_type = ToolErrorType.UNKNOWN

        try:
            return await tool_fn(args)
        except Exception as e:
            last_error_type, _ = classify_error(name, e)
            config = get_retry_config(last_error_type)
            last_error = e

            if config["max_retries"] == 0:
                # Validation 错误 — 不重试
                logger.warning(
                    f"[DynamicAction] 工具 {name} 参数错误(不重试): {e}"
                )
                return format_tool_error_response(
                    name, last_error_type, str(e)[:100], retries_used=0,
                )

            # ── 退避重试 ──
            for attempt in range(1, config["max_retries"] + 1):
                delay = config["backoff_base"] * (2 ** (attempt - 1))
                logger.info(
                    f"[DynamicAction] 工具 {name} {last_error_type.value}错误，"
                    f"退避 {delay:.1f}s 后重试 ({attempt}/{config['max_retries']})"
                )
                await _asyncio.sleep(delay)
                try:
                    return await tool_fn(args)
                except Exception as retry_e:
                    last_error = retry_e
                    # 重试中也可能遇到不同类型的错误
                    retry_type, _ = classify_error(name, retry_e)
                    if retry_type == ToolErrorType.VALIDATION:
                        break  # 变成参数错误就不再重试

            # 重试耗尽
            logger.error(
                f"[DynamicAction] 工具 {name} {last_error_type.value}错误，"
                f"重试 {config['max_retries']} 次后仍失败: {last_error}"
            )
            return format_tool_error_response(
                name, last_error_type, str(last_error)[:100],
                retries_used=config["max_retries"],
                fallback_available=(
                    last_error_type == ToolErrorType.DEPENDENCY
                ),
            )

    async def _tool_check_inventory(self, args: dict) -> str:
        """
        查询库存 — 走 SQLite inventory_items 表，支持模糊搜索。

        比 mock dict 更可靠: SQL LIKE 匹配 + LLM 理解用户意图挑选关键词，
        双重保障确保搜索准确。
        """
        keyword = args.get("keyword", "").strip()
        if not keyword:
            return json.dumps({"found": False, "error": "请提供物品关键词"}, ensure_ascii=False)

        try:
            from db.models import InventoryItem

            db = self._get_session()
            try:
                # 1. 精确匹配
                exact = db.query(InventoryItem).filter(
                    InventoryItem.item_name == keyword,
                    InventoryItem.is_active == 1,
                ).first()
                if exact:
                    return json.dumps({
                        "found": True, "match_type": "exact",
                        "item_name": exact.item_name, "category": exact.category,
                        "stock": exact.stock, "min_threshold": exact.min_threshold,
                        "available": exact.stock > 0,
                        "low_stock": 0 < exact.stock <= exact.min_threshold,
                        "unit_price": exact.unit_price,
                        "supplier": exact.supplier,
                    }, ensure_ascii=False)

                # 2. 模糊搜索 — LLM 传的关键词可能不完全匹配 item_name
                fuzzy = db.query(InventoryItem).filter(
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
                    return json.dumps({
                        "found": True, "match_type": "fuzzy",
                        "keyword": keyword, "count": len(results),
                        "items": results,
                    }, ensure_ascii=False)

                # 3. 无匹配
                return json.dumps({
                    "found": False, "keyword": keyword,
                    "message": f"库存中未找到与'{keyword}'匹配的物品。请尝试不同的关键词，或提交采购申请。",
                }, ensure_ascii=False)

            finally:
                db.close()
        except Exception as e:
            logger.error(f"[DynamicAction] 库存查询失败: {e}")
            return json.dumps({"found": False, "error": str(e)}, ensure_ascii=False)

    async def _tool_web_search(self, args: dict) -> str:
        """网页搜索 — 调用 builtin_tools 中的 web_search 工具"""
        query = args.get("query", "")
        try:
            from agents.tools import tool_registry
            result = await tool_registry.invoke("web_search", query=query)
            if result.success and result.data:
                return json.dumps(result.data, ensure_ascii=False)
            return json.dumps({"error": "搜索失败", "detail": result.error}, ensure_ascii=False)
        except Exception as e:
            logger.error(f"[DynamicAction] web_search 失败: {e}")
            return json.dumps({"error": f"搜索异常: {e}"}, ensure_ascii=False)

    async def _tool_check_leave_balance(self, args: dict) -> str:
        """查询假期余额"""
        user_name = args.get("user_name", "")
        # 模拟数据（实际接入HR系统）
        return json.dumps({
            "user_name": user_name,
            "annual_leave_total": 15,
            "annual_leave_used": 3,
            "annual_leave_remaining": 12,
            "sick_leave_remaining": 5,
            "personal_leave_remaining": 2,
            "message": f"{user_name} 年假剩余 12 天，病假剩余 5 天",
        }, ensure_ascii=False)

    async def _tool_check_ticket_status(self, args: dict) -> str:
        """查询工单状态"""
        ticket_number = args.get("ticket_number", "")
        try:
            tickets = self.db_router.ticket.list_tickets(limit=100)
            for t in tickets:
                if t.get("ticket_number") == ticket_number:
                    return json.dumps({
                        "found": True, "ticket_number": ticket_number,
                        "title": t.get("title", ""), "status": t.get("status", ""),
                        "priority": t.get("priority", ""), "created_at": t.get("created_at", ""),
                    }, ensure_ascii=False)
            return json.dumps({"found": False, "ticket_number": ticket_number, "message": "未找到该工单"}, ensure_ascii=False)
        except Exception as e:
            return json.dumps({"error": f"查询失败: {e}"}, ensure_ascii=False)

    async def _tool_search_knowledge(self, args: dict) -> str:
        """搜索知识库"""
        query = args.get("query", "")
        try:
            if self._knowledge_service is None:
                from services.knowledge_service import KnowledgeService
                self._knowledge_service = KnowledgeService()
                await self._knowledge_service.initialize()
            docs = await self._knowledge_service.search(query, top_k=3)
            if not docs:
                return json.dumps({"found": False, "query": query, "message": "未找到相关知识"}, ensure_ascii=False)
            results = []
            for d in docs:
                results.append({"category": d.get("category", ""), "score": d.get("score", 0), "content": d.get("content", "")[:300]})
            return json.dumps({"found": True, "query": query, "count": len(docs), "results": results}, ensure_ascii=False)
        except Exception as e:
            return json.dumps({"error": f"知识库搜索失败: {e}"}, ensure_ascii=False)

    async def _tool_search_meeting_rooms(self, args: dict) -> str:
        """搜索会议室"""
        date = args.get("date", "")
        start_time = args.get("start_time", "")
        end_time = args.get("end_time", "")
        capacity = args.get("capacity", 0)
        try:
            from db.models import MeetingRoom, MeetingRoomBooking

            db = self._get_session()
            try:
                # 所有活跃房间
                rooms_q = db.query(MeetingRoom).filter(
                    MeetingRoom.is_active == 1, MeetingRoom.status == "available"
                )
                if capacity > 0:
                    rooms_q = rooms_q.filter(MeetingRoom.capacity >= capacity)
                rooms = rooms_q.all()

                available = []
                for room in rooms:
                    conflict = db.query(MeetingRoomBooking).filter(
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
                return json.dumps({"date": date, "time": f"{start_time}-{end_time}", "rooms": available}, ensure_ascii=False)
            finally:
                db.close()
        except Exception as e:
            return json.dumps({"error": f"会议室查询失败: {e}"}, ensure_ascii=False)

    async def _tool_create_ticket(self, args: dict) -> str:
        """
        工单工具 — 通过 A2A 委派给 TicketDispatch，不自己写 SQL。

        提议模式 (self._execution_mode=False):
          TicketDispatch 做合规检查 + 生成专业卡片 → 存入 _proposals。
          同类工单自动合并去重。

        执行模式 (self._execution_mode=True):
          TicketDispatch 落库到 DB → 返回工单号。
        """
        from agents.a2a.protocol import AgentMessage as AM

        ticket_type = args.get("ticket_type", "admin")
        title = args.get("title", "")
        description = args.get("description", "")
        extra = args.get("extra", {})
        service_type = extra.get("service_type", "")

        # ── 获取 TicketDispatch Agent ──
        td_agent = agent_registry.get_agent("ticket_dispatch")
        if td_agent is None:
            logger.error("[DynamicAction] TicketDispatch Agent 不可用")
            if self._execution_mode:
                return json.dumps(
                    {"executed": False, "error": "工单服务不可用"},
                    ensure_ascii=False,
                )
            return json.dumps(
                {"status": "error", "message": "工单服务不可用"},
                ensure_ascii=False,
            )

        # ── 身份防御：_last_user_name 为空时拒绝委托 ──
        if not self._last_user_name or self._last_user_name == "web_user":
            self.logger.error(
                "[DynamicAction] _last_user_name 未设置，无法委托 TicketDispatch!"
            )
            return json.dumps(
                {"status": "error",
                 "message": "用户身份获取失败，请刷新页面重试。"},
                ensure_ascii=False,
            )

        # ── 构造 A2A 委派消息 ──
        pre_extracted = {
            "ticket_type": ticket_type,
            "title": title,
            "description": description,
            "category": service_type or "其他",
            "priority": args.get("priority", "P2"),
            "extra": extra,
        }

        delegation = AM.create_delegation(
            from_agent=self.agent_id,
            to_agent="ticket_dispatch",
            payload={
                "user_input": self._last_user_input,
                "task": (
                    "执行已确认的工单创建" if self._execution_mode
                    else "提取参数、合规检查、生成确认卡片（不落库）"
                ),
                "intent_category": "action",
                "urgency": args.get("priority", "P2"),
                "user_id": self._last_user_name,
                "user_name": self._last_user_name,
                "role": self._last_user_role,
                "confirmed": self._execution_mode,
                "pre_extracted": pre_extracted,
            },
            trace_id=self._last_trace_id,
        )

        try:
            result = await td_agent.execute(delegation)

            # ── 执行模式：TicketDispatch 已落库 ──
            if self._execution_mode:
                if result.success and result.payload.get("executed"):
                    ticket_number = result.payload.get("ticket_number", "")
                    logger.info(
                        f"[DynamicAction] TicketDispatch 落库成功: {ticket_number}"
                    )
                    return json.dumps({
                        "executed": True,
                        "ticket_number": ticket_number,
                        "ticket_id": result.payload.get("ticket_id", ""),
                        "ticket_type": ticket_type,
                        "status": result.payload.get("status", "created"),
                        "title": title,
                        "message": result.payload.get("direct_response", "工单已创建"),
                    }, ensure_ascii=False)
                else:
                    error = result.error or result.payload.get("direct_response", "未知错误")
                    logger.error(f"[DynamicAction] TicketDispatch 落库失败: {error}")
                    return json.dumps(
                        {"executed": False, "error": str(error)[:200]},
                        ensure_ascii=False,
                    )

            # ── 提议模式：TicketDispatch 返回专业卡片 ──
            if result.success and result.payload.get("return_card"):
                card = result.payload.get("card", {})
                result_ticket_type = result.payload.get("ticket_type", ticket_type)

                # ★ 用 TicketDispatch 生成的完整卡片代替自建简化版
                key = f"{result_ticket_type}:{service_type}"
                self._proposals[key] = {
                    "ticket_type": result_ticket_type,
                    # 用 LLM 原始业务标题（非卡片展示标题 "📦 设备领用"），确保 confirm 阶段落库正确
                    "title": title,
                    "description": description,
                    "priority": args.get("priority", "P2"),
                    "extra": extra,
                    "service_type": service_type,
                    "card": card,                     # ★ TicketDispatch 完整卡片
                    "source": "ticket_dispatch",      # ★ 标记来源
                }

                logger.info(
                    f"[DynamicAction] TicketDispatch 卡片已存储: "
                    f"key={key}, card_title={card.get('title', '?')[:40]}"
                )
                return json.dumps({
                    "status": "proposed",
                    "message": f"合规检查通过，{title} 等待确认",
                    "merged_count": len(self._proposals),
                    "ticket_type": result_ticket_type,
                    "service_type": service_type,
                    "title": title,
                }, ensure_ascii=False)

            # TicketDispatch 直接完成了（无需确认的场景，如 it_fault 已解决）
            if result.success and result.payload.get("direct_response"):
                return json.dumps({
                    "executed": True,
                    "message": result.payload.get("direct_response", "操作完成"),
                }, ensure_ascii=False)

            # 其他失败情况
            return json.dumps({
                "status": "error",
                "message": result.error or "工单创建失败",
            }, ensure_ascii=False)

        except Exception as e:
            logger.error(f"[DynamicAction] TicketDispatch A2A 委派异常: {e}")
            if self._execution_mode:
                return json.dumps(
                    {"executed": False, "error": str(e)},
                    ensure_ascii=False,
                )
            return json.dumps(
                {"status": "error", "message": f"工单服务异常: {e}"},
                ensure_ascii=False,
            )

    @staticmethod
    def _proposals_to_cards(proposals: dict) -> list[dict]:
        """
        将提议列表合并为确认卡片。

        v9: 如果 proposal 来自 TicketDispatch (source="ticket_dispatch")
        且有完整 card 字段，直接使用其专业卡片（含合规检查结果、fields、alerts）。
        否则走旧版简化卡片兜底。
        """
        cards = []
        for key, p in proposals.items():
            # ★ v9: TicketDispatch 专业卡片 — 直接转发
            if p.get("source") == "ticket_dispatch" and p.get("card"):
                card = p["card"]
                # 确保走聊天中断通道（非直接 REST API 调用）
                card["confirm_action"] = "chat"
                card["ticket_type"] = p.get("ticket_type", card.get("ticket_type", "admin"))
                card["priority"] = p.get("priority", card.get("priority", "P2"))
                cards.append(card)
                continue

            # ── 兜底：旧版简化卡片（TicketDispatch 不可用时触发）──
            ticket_type = p["ticket_type"]
            service_type = p.get("service_type", "")
            svc = service_type.lower()
            if svc in ("asset_requisition", "资产领用", "领用"):
                emoji = "\U0001F4E6"
                card_title = "Equipment Requisition"
                confirm_text = "Confirm Requisition"
            elif svc in ("procurement", "采购申请", "采购"):
                emoji = "\U0001F6D2"
                card_title = "Procurement Request"
                confirm_text = "Confirm Procurement"
            elif ticket_type == "it_fault":
                emoji = "🔧"
                card_title = "IT工单确认"
                confirm_text = "确认创建"
            elif ticket_type == "leave":
                emoji = "🏖️"
                card_title = "请假申请确认"
                confirm_text = "确认请假"
            else:
                emoji = "📋"
                card_title = "工单确认"
                confirm_text = "确认创建"

            dispatch_service_type = service_type
            if svc in ("asset_requisition", "领用"):
                dispatch_service_type = "资产领用"
            elif svc in ("procurement", "采购"):
                dispatch_service_type = "采购申请"

            cards.append({
                "type": "confirm",
                "title": f"{emoji} {card_title}",
                "description": p["description"],
                "confirm_text": confirm_text,
                "confirm_action": "chat",
                "confirm_message": "确认创建以上工单",
                "success_message": f"{card_title} submitted!",
                "ticket_type": ticket_type,
                "service_type": dispatch_service_type,
                "priority": p["priority"],
                "fallback_url": "/tickets",
                "fallback_text": "View Tickets",
            })
        return cards

    # ================================================================
    # ReAct 循环核心（自由编排的关键）
    # ================================================================

    async def execute_confirm(
        self, message: AgentMessage, saved_messages: list, proposals: dict,
    ) -> AgentMessage:
        """
        确认后执行 — 恢复 ReAct 对话状态，实际创建工单。

        Args:
            message: 原始委派消息
            saved_messages: 上一轮的 LLM 对话历史
            proposals: 上一轮的提议字典
        """
        user_input = message.payload.get("user_input", "")
        user_name = message.payload.get("user_name", "")

        # ★ v9: 恢复上下文供 _tool_create_ticket 构造 A2A 委派消息
        # (resume 时 agent 是新实例，需要重新设置)
        self._last_user_input = user_input
        self._last_user_name = user_name
        self._last_user_role = message.payload.get("role", "employee")
        self._last_trace_id = message.trace_id

        self._execution_mode = True  # 实际创建工单
        self._proposals = proposals

        self.logger.info(
            f"[DynamicAction:confirm] 执行确认模式 — {len(proposals)} 个提议"
        )

        try:
            # 恢复对话历史
            messages = list(saved_messages)
            # 注入确认指令
            messages.append({
                "role": "user",
                "content": (
                    "USER CONFIRMED all proposals. Now EXECUTE them: "
                    "call create_ticket for each proposal with the SAME parameters "
                    "as before. Create the actual tickets in the database.\n"
                    f"Proposals to execute: {json.dumps(list(proposals.values()), ensure_ascii=False)}"
                ),
            })

            react_trace = []
            for iteration in range(5):  # 确认执行最多 5 轮
                response = await self.llm_with_tools.ainvoke(messages)
                if not response.tool_calls:
                    final_answer = response.content.strip() if response.content else "Done."
                    self._execution_mode = False
                    return AgentMessage.create_response(
                        from_agent=self.agent_id,
                        to_agent=message.from_agent,
                        payload={
                            "direct_response": final_answer,
                            "react_trace": [t.model_dump() for t in react_trace],
                            "executed": True,
                            "summary": f"Executed {len(proposals)} tickets",
                        },
                        original_message=message,
                        success=True,
                    )

                for tc in response.tool_calls:
                    tool_name = tc.get("name", "")
                    tool_args = tc.get("args", {})
                    if isinstance(tool_args, str):
                        tool_args = json.loads(tool_args)
                    observation = await self._execute_tool(tool_name, tool_args)
                    react_trace.append(ReActState(
                        thought=response.content[:200] if response.content else "",
                        tool_name=tool_name,
                        tool_args=tool_args,
                        observation=str(observation)[:500],
                    ))
                    messages.append({
                        "role": "assistant",
                        "content": response.content or "",
                        "tool_calls": [{
                            "id": f"exec_{iteration}_{tool_name}",
                            "type": "function",
                            "function": {"name": tool_name, "arguments": json.dumps(tool_args, ensure_ascii=False)},
                        }],
                    })
                    messages.append({
                        "role": "tool",
                        "tool_call_id": f"exec_{iteration}_{tool_name}",
                        "content": str(observation),
                    })

            self._execution_mode = False
            # 达到上限，强制总结
            final = await self.llm.ainvoke(messages)
            return AgentMessage.create_response(
                from_agent=self.agent_id,
                to_agent=message.from_agent,
                payload={
                    "direct_response": final.content.strip() or "Tickets created.",
                    "executed": True,
                },
                original_message=message,
                success=True,
            )

        except Exception as e:
            self._execution_mode = False
            logger.error(f"[DynamicAction:confirm] Failed: {e}")
            return self.create_error_response(message, str(e))

    async def execute(self, message: AgentMessage) -> AgentMessage:
        """
        ReAct 循环入口。

        这是自由编排的核心 — 没有任何硬编码的流程。
        LLM 通过 tool schema 自主决定:
        1. 调用哪些工具
        2. 以什么顺序调用
        3. 如何根据中间结果做条件判断
        4. 何时停止并给出最终答案
        """
        user_input = message.payload.get("user_input", "")
        user_name = message.payload.get("user_name", "")
        conversation_history = message.payload.get("conversation_history", "")

        # ★ v9: 保存上下文供 _tool_create_ticket 构造 A2A 委派消息
        self._last_user_input = user_input
        self._last_user_name = user_name
        self._last_user_role = message.payload.get("role", "employee")
        self._last_trace_id = message.trace_id

        # 确保库存种子数据已填充
        await self._ensure_inventory_seeded()
        # 初始化提议收集器
        self._proposals = {}

        self.logger.info(
            f"[DynamicAction] ReAct 循环开始 "
            f"(trace={message.trace_id[:8]}...): \"{user_input[:80]}\""
        )

        try:
            # 构建消息历史（包含 system prompt + 对话上下文）
            messages = [{"role": "system", "content": self._build_system_prompt(user_name)}]

            if conversation_history:
                messages.append({"role": "user", "content": f"对话历史:\n{conversation_history}"})

            messages.append({"role": "user", "content": user_input})

            # ReAct 循环
            react_trace: list[ReActState] = []

            for iteration in range(self.MAX_REACT_ITERATIONS):
                self.logger.info(f"[DynamicAction] ReAct 迭代 {iteration + 1}/{self.MAX_REACT_ITERATIONS}")

                # ── 调用 LLM（带 tool schemas）──
                response = await self.llm_with_tools.ainvoke(messages)

                # ── 没有 tool_calls → 最终答案 ──
                if not response.tool_calls:
                    final_answer = response.content.strip() if response.content else "处理完成。"
                    self.logger.info(
                        f"[DynamicAction] ReAct 完成，"
                        f"迭代 {iteration + 1} 次，"
                        f"调用了 {len(react_trace)} 个工具"
                    )
                    for i, step in enumerate(react_trace):
                        self.logger.info(
                            f"  Step {i+1}: {step.tool_name}({json.dumps(step.tool_args, ensure_ascii=False)[:60]})"
                        )
                    # ── 生成确认卡片 ──
                    cards = self._proposals_to_cards(self._proposals) if self._proposals else []

                    return AgentMessage.create_response(
                        from_agent=self.agent_id,
                        to_agent=message.from_agent,
                        payload={
                            "direct_response": final_answer,
                            "react_trace": [t.model_dump() for t in react_trace],
                            "iterations": iteration + 1,
                            "summary": f"ReAct 完成: {len(react_trace)} 个工具调用, {iteration + 1} 次迭代",
                            "return_card": len(cards) > 0,
                            "cards": cards,
                        },
                        original_message=message,
                        success=True,
                    )

                # ── 有 tool_calls → 执行工具并继续循环 ──
                # 注意：一次 LLM 调用可能返回多个 tool_calls
                # （如同时查库存和查政策 — 真正的并行编排）
                for tool_call in response.tool_calls:
                    tool_name = tool_call.get("name", "")
                    tool_args = tool_call.get("args", {})
                    if isinstance(tool_args, str):
                        tool_args = json.loads(tool_args)

                    thought = response.content[:200] if response.content else f"决定调用 {tool_name}"

                    self.logger.info(
                        f"[DynamicAction] → {tool_name}({json.dumps(tool_args, ensure_ascii=False)[:80]})"
                    )

                    # 执行工具
                    observation = await self._execute_tool(tool_name, tool_args)

                    # 记录 trace
                    react_trace.append(ReActState(
                        thought=thought,
                        tool_name=tool_name,
                        tool_args=tool_args,
                        observation=observation[:500],
                    ))

                    # 将工具调用和结果追加到消息历史
                    # 这是 LangGraph ReAct 模式的关键 — LLM 能看到自己的历史操作
                    messages.append({
                        "role": "assistant",
                        "content": thought,
                        "tool_calls": [{
                            "id": f"call_{iteration}_{tool_name}",
                            "type": "function",
                            "function": {"name": tool_name, "arguments": json.dumps(tool_args, ensure_ascii=False)},
                        }],
                    })
                    messages.append({
                        "role": "tool",
                        "tool_call_id": f"call_{iteration}_{tool_name}",
                        "content": observation,
                    })

            # ── 达到最大迭代次数 → 强制总结 ──
            self.logger.warning(
                f"[DynamicAction] 达到最大迭代 {self.MAX_REACT_ITERATIONS}，"
                f"强制 LLM 总结"
            )
            messages.append({
                "role": "user",
                "content": "你已达到最大操作步数限制。请基于已有的工具调用结果，"
                           "给用户一个简洁的总结，说明已完成什么、还需要什么。",
            })
            final_response = await self.llm.ainvoke(messages)
            final_answer = final_response.content.strip() if final_response.content else "处理超时，请重新提交请求。"

            return AgentMessage.create_response(
                from_agent=self.agent_id,
                to_agent=message.from_agent,
                payload={
                    "direct_response": final_answer,
                    "react_trace": [t.model_dump() for t in react_trace],
                    "iterations": self.MAX_REACT_ITERATIONS,
                    "summary": f"ReAct 达到最大迭代 ({self.MAX_REACT_ITERATIONS})，强制总结",
                },
                original_message=message,
                success=True,
            )

        except Exception as e:
            self.logger.error(f"[DynamicAction] ReAct 循环异常: {e}")
            return self.create_error_response(message, str(e))

    # ================================================================
    # 流式 ReAct 循环（思维链实时可见）
    # ================================================================

    async def execute_with_stream(
        self, message: AgentMessage,
    ) -> AsyncGenerator[dict, AgentMessage | None]:
        """
        流式 ReAct 循环 — 每一步思考/行动/观察都实时推送。

        与 execute() 的核心区别：
          execute() → 收集所有 trace → 最终吐出一个 AgentMessage
          本方法 → 每步都 yield 事件 → 前端实时看到思维链 → 最后 yield 最终 AgentMessage

        Yield 格式:
          {"event": "thought",  "text": "分析需求: 用户需要先查库存再建单"}
          {"event": "tool_call",  "text": "调用 check_inventory", "tool": "...", "args": {...}}
          {"event": "tool_result", "text": "鼠标库存: 12个", "tool": "...", "data": {...}}
          {"event": "thought",  "text": "库存>0，继续建单"}
          {"event": "tool_call",  "text": "创建领用工单", "tool": "create_ticket", "args": {...}}
          {"event": "tool_result", "text": "工单 REQ-001 已创建", "data": {...}}
          {"event": "final", "text": "鼠标有12个库存，已创建工单 REQ-001"}
          {"event": "card", "card": {...}}    # 可选：需要用户确认时
          (generator 结束 → 调用方用最后一个 yield 收集 AgentMessage)
        """
        user_input = message.payload.get("user_input", "")
        user_name = message.payload.get("user_name", "")
        conversation_history = message.payload.get("conversation_history", "")

        # ★ v9: 保存上下文供 _tool_create_ticket 构造 A2A 委派消息
        self._last_user_input = user_input
        self._last_user_name = user_name
        self._last_user_role = message.payload.get("role", "employee")
        self._last_trace_id = message.trace_id

        # 确保库存种子数据已填充
        await self._ensure_inventory_seeded()
        # 初始化提议收集器
        self._proposals = {}

        self.logger.info(
            f"[DynamicAction:stream] ReAct 流式循环开始 "
            f"(trace={message.trace_id[:8]}...): \"{user_input[:80]}\""
        )

        try:
            messages = [{"role": "system", "content": self._build_system_prompt(user_name)}]
            if conversation_history:
                messages.append({"role": "user", "content": f"对话历史:\n{conversation_history}"})
            messages.append({"role": "user", "content": user_input})

            react_trace: list[ReActState] = []

            for iteration in range(self.MAX_REACT_ITERATIONS):
                # ── 第一步: yield 思考事件 ──
                yield {
                    "event": "thought",
                    "text": f"正在分析第 {iteration + 1} 步...",
                }

                # ── LLM 决策 ──
                response = await self.llm_with_tools.ainvoke(messages)

                # ── 无 tool_calls → 最终回答 ──
                if not response.tool_calls:
                    final_answer = response.content.strip() if response.content else "处理完成。"

                    # 推送最终思考
                    thought_text = response.content[:200] if response.content else "已得出最终答案"
                    yield {
                        "event": "thought",
                        "text": thought_text,
                    }

                    # 推送最终答案
                    yield {
                        "event": "final",
                        "text": final_answer,
                        "iterations": iteration + 1,
                        "tool_count": len(react_trace),
                    }

                    # 推送确认卡片（如果有提议）
                    if self._proposals:
                        cards = self._proposals_to_cards(self._proposals)
                        for card in cards:
                            yield {"event": "card", "card": card}

                    # 保存消息历史供跨turn恢复
                    self._last_messages = messages

                    self.logger.info(
                        f"[DynamicAction:stream] ReAct 完成 — "
                        f"{iteration + 1} iterations, {len(react_trace)} tools, "
                        f"{len(self._proposals)} proposals, "
                        f"{len(messages)} messages saved"
                    )
                    return  # generator ends, caller collects result

                # ── 有 tool_calls → 逐工具流式执行 ──
                for tool_call in response.tool_calls:
                    tool_name = tool_call.get("name", "")
                    tool_args = tool_call.get("args", {})
                    if isinstance(tool_args, str):
                        tool_args = json.loads(tool_args)

                    thought = response.content[:200] if response.content else f"决定调用 {tool_name}"

                    # 推送: 工具调用事件
                    yield {
                        "event": "tool_call",
                        "text": self._describe_tool_call(tool_name, tool_args),
                        "tool": tool_name,
                        "args": tool_args,
                    }

                    # 执行工具
                    observation = await self._execute_tool(tool_name, tool_args)

                    # 推送: 工具返回事件
                    obs_summary = self._summarize_observation(tool_name, observation)
                    yield {
                        "event": "tool_result",
                        "text": obs_summary,
                        "tool": tool_name,
                        "data": observation[:500] if isinstance(observation, str) else observation,
                    }

                    # 记录 trace
                    react_trace.append(ReActState(
                        thought=thought,
                        tool_name=tool_name,
                        tool_args=tool_args,
                        observation=observation[:500] if isinstance(observation, str) else str(observation)[:500],
                    ))

                    # 追加到消息历史（LLM 看到自己的历史才能做条件推理）
                    messages.append({
                        "role": "assistant",
                        "content": thought,
                        "tool_calls": [{
                            "id": f"call_{iteration}_{tool_name}",
                            "type": "function",
                            "function": {"name": tool_name, "arguments": json.dumps(tool_args, ensure_ascii=False)},
                        }],
                    })
                    messages.append({
                        "role": "tool",
                        "tool_call_id": f"call_{iteration}_{tool_name}",
                        "content": observation if isinstance(observation, str) else json.dumps(observation, ensure_ascii=False),
                    })

            # ── 达到最大迭代 → 强制总结 ──
            self.logger.warning(f"[DynamicAction:stream] 达到最大迭代 {self.MAX_REACT_ITERATIONS}")
            yield {
                "event": "thought",
                "text": f"已达到最大步数 ({self.MAX_REACT_ITERATIONS})，正在总结...",
            }
            messages.append({
                "role": "user",
                "content": "你已达到最大操作步数。请基于已有结果给用户一个简洁的总结。",
            })
            final_response = await self.llm.ainvoke(messages)
            final_answer = final_response.content.strip() if final_response.content else "处理超时，请重新提交请求。"

            yield {
                "event": "final",
                "text": final_answer,
                "iterations": self.MAX_REACT_ITERATIONS,
                "tool_count": len(react_trace),
                "truncated": True,
            }

        except Exception as e:
            self.logger.error(f"[DynamicAction:stream] 异常: {e}")
            yield {
                "event": "final",
                "text": f"处理过程中出现问题：{e}\n请稍后重试或联系人工服务。",
                "error": str(e),
            }

    @staticmethod
    def _describe_tool_call(tool_name: str, args: dict) -> str:
        """将工具调用转为人类可读的描述文字"""
        descriptions = {
            "check_inventory": lambda a: f"查询库存: {a.get('keyword', a.get('item', '?'))}",
            "check_leave_balance": lambda a: f"查询 {a.get('user_name', '?')} 的假期余额",
            "check_ticket_status": lambda a: f"查询工单 {a.get('ticket_number', '?')} 的状态",
            "search_knowledge_base": lambda a: f"搜索知识库: {a.get('query', '?')[:50]}",
            "search_meeting_rooms": lambda a: f"搜索 {a.get('date', '?')} {a.get('start_time', '?')} 可用会议室",
            "web_search": lambda a: f"网上搜索: {a.get('query', '?')[:50]}",
            "create_ticket": lambda a: f"创建{a.get('ticket_type', '?')}工单: {a.get('title', '?')[:40]}",
        }
        if tool_name in descriptions:
            return descriptions[tool_name](args)
        return f"调用 {tool_name}"

    @staticmethod
    def _summarize_observation(tool_name: str, raw: str) -> str:
        """将工具返回的 raw JSON 转为人类可读的一句话总结"""
        try:
            data = json.loads(raw) if isinstance(raw, str) else raw
        except (json.JSONDecodeError, TypeError):
            return raw[:200] if isinstance(raw, str) else str(raw)[:200]

        error = data.get("error", "")
        if error:
            return f"❌ {tool_name} 失败: {error[:100]}"

        if tool_name == "check_inventory":
            items = data.get("items", [])
            if not items:
                return f"⚠️ 未找到匹配'{data.get('keyword', '?')}'的物品"
            parts = []
            for item in items:
                name = item.get("item_name", "?")
                stock = item.get("stock", 0)
                low = "⚠️低库存" if item.get("low_stock") else ""
                price = f"¥{item.get('unit_price', '?')}" if item.get('unit_price') else ""
                if stock > 0:
                    parts.append(f"{name}: {stock}台 {price} {low}".strip())
                else:
                    parts.append(f"{name}: 缺货 {price}".strip())
            return " | ".join(parts[:5])

        if tool_name == "web_search":
            results = data.get("results", [])
            if not results:
                return "未找到相关搜索结果"
            top = results[0]
            return f"🌐 {top.get('title', '')}: {top.get('snippet', '')[:120]}..."

        elif tool_name == "check_leave_balance":
            remaining = data.get("annual_leave_remaining", "?")
            user = data.get("user_name", "")
            return f"✅ {user} 剩余年假: {remaining} 天"

        elif tool_name == "check_ticket_status":
            if data.get("found"):
                return f"✅ 工单 {data.get('ticket_number')}: {data.get('status')}"
            return "⚠️ 未找到该工单"

        elif tool_name == "search_knowledge_base":
            count = data.get("count", 0)
            return f"✅ 找到 {count} 篇相关文档" if count > 0 else "⚠️ 未找到相关文档"

        elif tool_name == "search_meeting_rooms":
            rooms = data.get("rooms", [])
            available = [r for r in rooms if r.get("available")]
            return f"✅ {data.get('date')} {data.get('time')}: {len(available)}/{len(rooms)} 间可用"

        elif tool_name == "create_ticket":
            if data.get("executed"):
                return f"✅ 工单已创建: {data.get('ticket_number', '?')} ({data.get('status', '?')})"
            if data.get("status") == "proposed":
                return f"📋 已提议: {data.get('title', '?')}（等待确认）"
            if data.get("success"):
                return f"✅ 工单已创建: {data.get('ticket_number', '?')} ({data.get('status', '?')})"
            return f"❌ 工单创建失败: {data.get('error', data.get('message', 'unknown'))[:80]}"

        return raw[:200] if isinstance(raw, str) else json.dumps(data, ensure_ascii=False)[:200]

    async def execute_stream(self, message: AgentMessage) -> AsyncGenerator[str, None]:
        """向后兼容的流式执行（旧接口）"""
        yield "[DynamicAction] 🧠 正在分析用户需求..."
        yield "[DynamicAction] 🔧 正在动态编排工具调用..."
        yield "[DynamicAction] ✅ 编排完成，返回结果"


# ============================================================
# 自动注册
# ============================================================

def _register():
    agent_registry.register(
        DynamicActionAgent.__agent_declaration__,
        DynamicActionAgent,
    )

_register()

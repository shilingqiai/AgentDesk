"""
工单派发子Agent — 处理多类型工单创建与派发请求（v3.2 bind_tools 版）

负责：
- 识别工单类型（IT故障/请假/报销/行政服务）
- 从用户输入中提取对应参数
- 通过 TicketRepository 持久化工单
- 返回工单状态给编排器

支持的工单类型：
    it_fault  — IT故障报修（网络/硬件/系统）
    leave     — 请假申请（年假/病假/事假）
    expense   — 报销申请（差旅/办公/餐费）
    admin     — 行政服务（会议室/快递/资产领用）

DashScope 兼容：_extract_params / classify_card_response 优先使用
bind_tools + tool_choice="auto" 触发原生 Function Calling，
prompt→JSON 作为 fallback。

YOU ARE A SUB-AGENT. DO NOT REPLY TO USER DIRECTLY.
MUST return structured findings to the Orchestrator.
"""

from __future__ import annotations

import json
import logging
import uuid
from typing import AsyncGenerator, Literal
from datetime import datetime

from pydantic import BaseModel, Field

from agents.base_sub_agent import BaseSubAgent
from agents.a2a.protocol import AgentMessage
from agents.orchestrator.agent_declaration import agent_declaration
from agents.orchestrator.agent_registry import agent_registry
from config.model_provider import create_chat_model

logger = logging.getLogger("agent.ticket_dispatch")


# ============================================================
# Pydantic 结构化输出 Schema（v3.2 bind_tools 版）
# ============================================================

class TicketExtra(BaseModel):
    """工单扩展字段"""
    leave_type: str = Field(default="", description="请假类型：年假|病假|事假|调休|婚假|产假（仅 leave）")
    start_date: str = Field(default="", description="开始日期 YYYY-MM-DD（仅 leave）")
    end_date: str = Field(default="", description="结束日期 YYYY-MM-DD（仅 leave）")
    total_days: int = Field(default=0, description="请假天数（仅 leave）")
    expense_type: str = Field(default="", description="报销类型（仅 expense）")
    amount: float = Field(default=0.0, description="报销金额（仅 expense）")
    has_invoice: bool = Field(default=False, description="是否有发票（仅 expense）")
    service_type: str = Field(default="", description="服务类型：会议室|快递|资产|访客（仅 admin）")
    time_slot: str = Field(default="", description="时间段如 14:00-16:00（仅 admin）")
    suggested_engineer_skill: str = Field(default="", description="建议工程师技能（仅 it_fault）")
    affected_users: int = Field(default=1, description="受影响用户数（仅 it_fault）")


class TicketParams(BaseModel):
    """工单参数提取的结构化输出"""
    ticket_type: Literal["it_fault", "leave", "expense", "admin"] = Field(
        description="工单类型"
    )
    title: str = Field(description="工单标题，简洁明了，不超过20字")
    description: str = Field(description="工单详细描述")
    category: str = Field(description="具体分类，从可用分类中选")
    priority: Literal["P0", "P1", "P2", "P3"] = Field(
        default="P2", description="优先级"
    )
    extra: TicketExtra = Field(default_factory=TicketExtra, description="扩展字段")


class CardIntent(BaseModel):
    """卡片回复意图分类"""
    intent: Literal["confirm", "modify", "cancel", "new_topic"] = Field(
        description="用户意图：confirm=确认执行, modify=修改参数, cancel=取消放弃, new_topic=换话题"
    )
    reason: str = Field(default="", description="分类理由，一句话")


# ============================================================
# OpenAI Function Calling 工具定义
# ============================================================

TICKET_PARAMS_TOOL = {
    "type": "function",
    "function": {
        "name": "extract_ticket_params",
        "description": "从用户输入中提取工单参数：类型、标题、描述、分类、优先级及扩展字段",
        "parameters": {
            "type": "object",
            "properties": {
                "ticket_type": {
                    "type": "string",
                    "enum": ["it_fault", "leave", "expense", "admin"],
                    "description": "工单类型",
                },
                "title": {
                    "type": "string",
                    "description": "工单标题，简洁明了，不超过20字",
                },
                "description": {
                    "type": "string",
                    "description": "工单详细描述",
                },
                "category": {
                    "type": "string",
                    "description": "具体分类，从可用分类中选",
                },
                "priority": {
                    "type": "string",
                    "enum": ["P0", "P1", "P2", "P3"],
                    "description": "优先级。P0=系统宕机/核心业务中断，P1=影响效率但可绕过，P2=一般故障/申请，P3=咨询/非紧急",
                },
                "extra": {
                    "type": "object",
                    "description": "扩展字段，根据工单类型填写对应子字段",
                    "properties": {
                        "leave_type": {
                            "type": "string",
                            "description": "请假类型：年假|病假|事假|调休|婚假|产假（仅 leave）",
                        },
                        "start_date": {
                            "type": "string",
                            "description": "开始日期 YYYY-MM-DD（仅 leave）",
                        },
                        "end_date": {
                            "type": "string",
                            "description": "结束日期 YYYY-MM-DD（仅 leave）",
                        },
                        "total_days": {
                            "type": "integer",
                            "description": "请假天数（仅 leave）",
                        },
                        "expense_type": {
                            "type": "string",
                            "description": "报销类型（仅 expense）",
                        },
                        "amount": {
                            "type": "number",
                            "description": "报销金额（仅 expense）",
                        },
                        "has_invoice": {
                            "type": "boolean",
                            "description": "是否有发票（仅 expense）",
                        },
                        "service_type": {
                            "type": "string",
                            "description": "服务类型：会议室|快递|资产|访客（仅 admin）",
                        },
                        "time_slot": {
                            "type": "string",
                            "description": "时间段如 14:00-16:00（仅 admin）",
                        },
                        "suggested_engineer_skill": {
                            "type": "string",
                            "description": "建议工程师技能（仅 it_fault）",
                        },
                        "affected_users": {
                            "type": "integer",
                            "description": "受影响用户数（仅 it_fault）",
                        },
                    },
                },
            },
            "required": ["ticket_type", "title", "description", "category"],
        },
    },
}

CARD_INTENT_TOOL = {
    "type": "function",
    "function": {
        "name": "classify_intent",
        "description": "分类用户对确认卡片的回复意图",
        "parameters": {
            "type": "object",
            "properties": {
                "intent": {
                    "type": "string",
                    "enum": ["confirm", "modify", "cancel", "new_topic"],
                    "description": (
                        "confirm=确认/同意卡片内容要求执行；"
                        "modify=想修改卡片参数（改时间/换房间/改金额）；"
                        "cancel=想取消/放弃/不要这张卡片；"
                        "new_topic=完全换了话题，问的是和卡片不相关的事"
                    ),
                },
                "reason": {
                    "type": "string",
                    "description": "分类理由，一句话",
                },
            },
            "required": ["intent", "reason"],
        },
    },
}


# ============================================================
# 工单类型定义
# ============================================================

TICKET_TYPE_CONFIG = {
    "it_fault": {
        "label": "IT故障报修",
        "emoji": "🔧",
        "category_options": ["网络故障", "系统运维", "账号管理", "硬件故障", "软件问题", "安全事件", "其他"],
        "response_prefix": "已为您创建IT故障工单",
    },
    "leave": {
        "label": "请假申请",
        "emoji": "🏖️",
        "category_options": ["年假", "病假", "事假", "婚假", "产假", "调休", "其他"],
        "response_prefix": "已为您提交请假申请",
        "extra_fields": ["leave_type", "start_date", "end_date", "total_days", "reason"],
    },
    "expense": {
        "label": "报销申请",
        "emoji": "💰",
        "category_options": ["差旅费", "办公用品", "餐费", "交通费", "培训费", "其他"],
        "response_prefix": "已为您提交报销申请",
        "extra_fields": ["expense_type", "amount", "has_invoice", "description"],
    },
    "admin": {
        "label": "行政服务",
        "emoji": "🏢",
        "category_options": ["会议室预定", "快递寄送", "资产领用", "采购申请", "访客登记", "办公环境", "其他"],
        "response_prefix": "已为您创建行政服务请求",
        "extra_fields": ["service_type", "time_slot", "location", "description"],
    },
}


@agent_declaration(
    agent_id="ticket_dispatch",
    name="工单派发Agent",
    description=(
        "负责创建、查询和派发多类型工单。支持：IT故障报修、请假申请、报销申请、行政服务请求。"
        "当用户需要提交工单、请假、报销、预定会议室等操作时调用此Agent。"
        "从用户输入中提取工单参数，创建工单记录并返回状态。"
    ),
    capabilities=[
        "ticket_creation",
        "ticket_query",
        "parameter_extraction",
        "status_tracking",
        "leave_application",
        "expense_claim",
        "admin_service",
    ],
    knowledge_domains=[
        "ticket_management",
        "dispatch_workflow",
        "sla_enforcement",
        "leave_management",
        "expense_claim",
        "admin_service",
    ],
    priority=2,
)
class TicketDispatchSubAgent(BaseSubAgent):
    """
    工单派发子Agent（v2 — DB持久化 + 多类型支持）

    职责：
    1. 从用户输入中使用LLM识别工单类型并提取参数
    2. 通过 TicketRepository 持久化工单记录
    3. 返回结构化工单状态给编排器
    """

    agent_id = "ticket_dispatch"

    def __init__(self):
        super().__init__()
        base_llm = create_chat_model(temperature=0.1)
        self.llm = base_llm
        # bind_tools 版本 — 触发原生 Function Calling
        self.llm_extract = base_llm.bind_tools(
            [TICKET_PARAMS_TOOL], tool_choice="auto"
        )
        self.llm_classify = base_llm.bind_tools(
            [CARD_INTENT_TOOL], tool_choice="auto"
        )
        self._db_router = None

    @property
    def db_router(self):
        """懒加载 DatabaseRouter"""
        if self._db_router is None:
            from db.db_router import DatabaseRouter
            self._db_router = DatabaseRouter()
        return self._db_router

    async def execute(self, message: AgentMessage) -> AgentMessage:
        """
        执行工单派发任务

        编排器委派的消息格式：
            payload.user_input: 用户原始输入
            payload.task: 任务描述
            payload.intent_category: 意图类别
            payload.urgency: 紧急程度
            payload.conversation_history: 对话历史（可选）

        返回格式：
            payload.ticket_id: 工单ID
            payload.ticket_number: 工单号
            payload.ticket_type: 工单类型
            payload.direct_response: 可展示给用户的工单状态消息
            payload.status: 工单状态
        """
        user_input = message.payload.get("user_input", "")
        task = message.payload.get("task", "")
        urgency = message.payload.get("urgency", "medium")
        conversation_history = message.payload.get("conversation_history", "")

        self.logger.info(
            f"[TicketDispatch] 处理工单请求 (trace={message.trace_id[:8]}...): "
            f"task=\"{task[:50]}\""
        )

        try:
            # ── v6: pre_checked 路径（complex track 已完成 RAG + ToolAgent 并行查询）──
            if message.payload.get("pre_checked"):
                return await self._execute_compliance_check(message)

            # 1. 使用 LLM 提取工单参数（含类型识别）
            ticket_params = await self._extract_params(
                user_input, urgency, conversation_history,
            )

            ticket_type = ticket_params.get("ticket_type", "it_fault")

            # 1.5 判断是否应该返回确认卡片
            if self._should_return_card(ticket_type, ticket_params):
                card = await self._build_card(ticket_type, ticket_params, user_input)
                return AgentMessage.create_response(
                    from_agent=self.agent_id,
                    to_agent=message.from_agent,
                    payload={
                        "direct_response": "",
                        "return_card": True,
                        "card": card,
                        "ticket_type": ticket_type,
                        "summary": f"[{TICKET_TYPE_CONFIG[ticket_type]['label']}] 等待用户确认",
                    },
                    original_message=message,
                    success=True,
                )

            # 2. 构建 payload（扩展字段）
            extra_payload = self._build_extra_payload(ticket_type, ticket_params)

            # 2.5 身份信息（从委派消息中提取）
            requester_id = message.payload.get("user_id", "") or message.payload.get("user_name", "")
            requester_name = message.payload.get("user_name", "") or requester_id

            # 3. 创建工单（写入 DB）
            ticket = self.db_router.ticket.add_ticket(
                ticket_type=ticket_type,
                title=ticket_params.get("title", user_input[:30]),
                description=ticket_params.get("description", user_input),
                category=ticket_params.get("category", "其他"),
                priority=ticket_params.get("priority", "P2"),
                requester_id=requester_id,
                requester_name=requester_name,
                trace_id=message.trace_id,
                payload=extra_payload,
            )

            # 4. 生成用户响应
            response = self._build_response(ticket, ticket_type)

            return AgentMessage.create_response(
                from_agent=self.agent_id,
                to_agent=message.from_agent,
                payload={
                    "direct_response": response,
                    "ticket_id": ticket["id"],
                    "ticket_number": ticket["ticket_number"],
                    "ticket_type": ticket_type,
                    "ticket_summary": ticket["title"],
                    "status": ticket["status"],
                    "priority": ticket["priority"],
                    "summary": (
                        f"[{TICKET_TYPE_CONFIG[ticket_type]['label']}] "
                        f"工单 {ticket['ticket_number']} 已创建"
                    ),
                    "needs_escalation": ticket["priority"] in ("P0", "P1"),
                },
                original_message=message,
                success=True,
            )

        except Exception as e:
            self.logger.error(f"工单派发失败: {e}")
            return self.create_error_response(message, str(e))

    async def _execute_compliance_check(self, message: AgentMessage) -> AgentMessage:
        """
        v6: 合规检查路径（complex track 已完成 RAG + ToolAgent 并行查询）。

        输入 payload 包含：
          - user_input: 用户原始请求
          - policy_result: RAG 政策查询结果
          - balance_result: ToolAgent 余额查询结果
          - user_name / role: 用户身份

        流程：
          1. LLM 合规检查（政策 vs 余额 vs 用户请求）
          2. 生成确认卡片（含余额信息 + 合规结论）
        """
        user_input = message.payload.get("user_input", "")
        policy_result = message.payload.get("policy_result", {})
        balance_result = message.payload.get("balance_result", {})
        user_name = message.payload.get("user_name", "")

        policy_text = policy_result.get("policy_text", "")
        balance_data = balance_result.get("tool_result", {})
        balance_text = balance_result.get("direct_response", "")

        self.logger.info(
            f"[TicketDispatch:compliance] 合规检查 — "
            f"user={user_name}, policy_len={len(policy_text)}, balance_keys={list(balance_data.keys()) if balance_data else 'none'}"
        )

        # 1. 提取请求参数（轻量级，不需要重新做 RAG）
        try:
            params = await self._extract_params(user_input)
        except Exception:
            params = self._fallback_extract(user_input, "medium")

        ticket_type = params.get("ticket_type", "leave")
        extra = params.get("extra", {})

        # 2. LLM 合规检查
        from config.model_provider import create_chat_model
        check_llm = create_chat_model(temperature=0)

        leave_type = extra.get("leave_type", "年假")
        start_date = extra.get("start_date", "")
        end_date = extra.get("end_date", "")
        total_days = extra.get("total_days", 0)

        check_prompt = (
            "你是一个请假合规检查器。根据政策、余额和用户请求，进行合规检查。\n\n"
            f"## 年假政策\n{policy_text[:800] if policy_text else '未查到相关政策'}\n\n"
            f"## 员工余额\n{balance_text if balance_text else '未查到余额数据'}\n"
            f"余额 JSON: {json.dumps(balance_data, ensure_ascii=False) if balance_data else '无'}\n\n"
            f"## 用户请求\n{user_input}\n\n"
            f"## 已提取参数\n"
            f"- 请假类型: {leave_type}\n"
            f"- 开始日期: {start_date or '未指定'}\n"
            f"- 结束日期: {end_date or '未指定'}\n"
            f"- 天数: {total_days or '未指定'}\n\n"
            "## 检查项\n"
            "1. 请假天数是否 ≤ 最长连续休假天数\n"
            "2. 剩余年假是否 ≥ 请假天数\n"
            "3. 日期是否在封账期内\n"
            "4. 是否触发审批要求\n\n"
            "返回 JSON（不要 markdown 包裹）：\n"
            '{"passed": true/false, '
            '"checks": [{"name":"...", "passed":true/false, "detail":"..."}], '
            '"warnings": ["..."], '
            '"compliance_summary": "一句话总结合规检查结果"}'
        )

        try:
            response = await check_llm.ainvoke([{"role": "user", "content": check_prompt}])
            check_data = self._parse_json(response.content)
        except Exception as e:
            self.logger.warning(f"[compliance] 合规检查 LLM 失败: {e}，默认通过")
            check_data = {
                "passed": True,
                "checks": [{"name": "合规检查", "passed": True, "detail": "自动通过（检查服务暂不可用）"}],
                "warnings": [],
                "compliance_summary": "合规检查自动通过",
            }

        passed = check_data.get("passed", False)
        checks = check_data.get("checks", [])
        warnings = check_data.get("warnings", [])
        compliance_summary = check_data.get("compliance_summary", "")

        self.logger.info(
            f"[compliance] 检查完成: passed={passed}, checks={len(checks)}, warnings={len(warnings)}"
        )

        # 3. 构建增强卡片（含余额 + 合规信息）
        annual_remaining = balance_data.get("annual_leave_remaining", "?")
        annual_total = balance_data.get("annual_leave_total", "?")
        annual_used = balance_data.get("annual_leave_used", "?")

        # 描述文本
        desc_parts = ["📋 **请假合规预检查**\n"]
        desc_parts.append(f"👤 **申请人**：{user_name}")
        desc_parts.append(f"🏖️ **请假类型**：{leave_type}")

        if total_days:
            desc_parts.append(f"📅 **天数**：{total_days} 天")
        if start_date:
            desc_parts.append(f"📅 **开始日期**：{start_date}")
        if end_date:
            desc_parts.append(f"📅 **结束日期**：{end_date}")

        desc_parts.append(f"\n💰 **假期余额**：")
        desc_parts.append(f"- 年假总额：{annual_total} 天")
        desc_parts.append(f"- 已用：{annual_used} 天")
        desc_parts.append(f"- 剩余：**{annual_remaining} 天**")

        if total_days and isinstance(annual_remaining, (int, float)) and annual_remaining != "?":
            after = annual_remaining - total_days
            desc_parts.append(f"- 请假后剩余：**{after} 天**")

        desc_parts.append(f"\n🔍 **合规检查结果**：{'✅ 通过' if passed else '❌ 未通过'}")
        for check in checks:
            icon = "✅" if check.get("passed") else "❌"
            desc_parts.append(f"  {icon} {check.get('name', '')}: {check.get('detail', '')}")
        if warnings:
            for w in warnings:
                desc_parts.append(f"  ⚠️ {w}")

        if compliance_summary:
            desc_parts.append(f"\n💡 {compliance_summary}")

        # 4. 构建卡片
        from datetime import datetime as dt, timedelta

        # 日期默认值
        today = dt.now()
        default_start = start_date or today.strftime("%Y-%m-%d")
        default_end = end_date or (today + timedelta(days=total_days - 1 if total_days else 0)).strftime("%Y-%m-%d")

        card = {
            "type": "confirm",
            "title": "🏖️ 请假申请（合规预检查）",
            "description": "\n".join(desc_parts),
            "fields": [
                {
                    "key": "leave_type", "label": "请假类型", "type": "select",
                    "options": [
                        {"value": "年假", "label": "年假"},
                        {"value": "病假", "label": "病假"},
                        {"value": "事假", "label": "事假"},
                        {"value": "调休", "label": "调休"},
                        {"value": "婚假", "label": "婚假"},
                    ],
                    "value": leave_type,
                    "required": True,
                },
                {
                    "key": "start_date", "label": "开始日期", "type": "date",
                    "value": default_start, "required": True,
                },
                {
                    "key": "end_date", "label": "结束日期", "type": "date",
                    "value": default_end, "required": True,
                },
                {
                    "key": "total_days", "label": "天数", "type": "number",
                    "value": str(total_days) if total_days else "",
                    "min": 1, "max": 30,
                    "required": True,
                },
            ],
            "confirm_text": "提交请假申请",
            "action": "/api/tickets/",
            "method": "POST",
            "body_template": {
                "user_input": user_input,
                "ticket_type": "leave",
                "priority": params.get("priority", "P2"),
            },
            "success_message": "请假申请已提交！可在工单管理页面查看进度。",
            "fallback_url": "/tickets",
            "fallback_text": "查看工单",
            # 附加合规信息（前端可展示）
            "compliance": {
                "passed": passed,
                "checks": checks,
                "warnings": warnings,
                "annual_remaining": annual_remaining,
            },
        }

        # 如果合规未通过，添加警告 alerts
        if not passed:
            card["alerts"] = [
                {"type": "warning", "message": f"⚠️ 合规检查未通过，请检查后重新提交。{compliance_summary}"}
            ]

        return AgentMessage.create_response(
            from_agent=self.agent_id,
            to_agent=message.from_agent,
            payload={
                "direct_response": "",
                "return_card": True,
                "card": card,
                "ticket_type": ticket_type,
                "summary": f"[请假申请·合规预检查] {'✅通过' if passed else '❌未通过'} | 剩余年假: {annual_remaining}天",
                "compliance_result": check_data,
            },
            original_message=message,
            success=True,
        )

    @staticmethod
    def _should_return_card(ticket_type: str, params: dict) -> bool:
        """
        判断是否应该返回确认卡片而非直接创建工单。

        规则：
        - admin 类型（会议室预定等）→ 始终返回卡片
        - leave 类型 → 始终返回卡片（需用户确认日期/类型）
        - expense 类型 → 始终返回卡片（需用户确认金额/类型）
        - it_fault 类型 → 返回 RAG-first 卡片（先提供解决方案）
        """
        if ticket_type in ("admin", "leave", "expense", "it_fault"):
            return True
        return False

    # ============================================================
    # 时间表达式解析
    # ============================================================

    @staticmethod
    def _parse_time_expression(user_input: str, extra: dict) -> tuple:
        """
        从用户输入中智能解析日期和时间段。

        支持：
        - "明天早上" → (2026-06-09, 09:00, 10:30, False)
        - "今天下午" → (2026-06-08, 14:00, 16:00, False)
        - "下午会议两小时" → (2026-06-08, 14:00, 16:00, True)
        - "下午2点到4点" → (today, 14:00, 16:00, True)

        Returns:
            (date_str, start_time, end_time, is_explicit_duration)
        """
        from datetime import datetime, timedelta

        today = datetime.now()
        date_str = today.strftime("%Y-%m-%d")
        start_time = "14:00"
        end_time = "16:00"
        is_explicit_duration = False

        text = user_input

        # 1. 解析日期
        if "明天" in text:
            target = today + timedelta(days=1)
            date_str = target.strftime("%Y-%m-%d")
        elif "后天" in text:
            target = today + timedelta(days=2)
            date_str = target.strftime("%Y-%m-%d")
        elif "今天" in text:
            date_str = today.strftime("%Y-%m-%d")
        elif "大后天" in text:
            target = today + timedelta(days=3)
            date_str = target.strftime("%Y-%m-%d")

        # 星期几解析
        weekdays = {"周一": 0, "周二": 1, "周三": 2, "周四": 3,
                     "周五": 4, "周六": 5, "周日": 6,
                     "星期一": 0, "星期二": 1, "星期三": 2, "星期四": 3,
                     "星期五": 4, "星期六": 5, "星期日": 6}
        for name, wd in weekdays.items():
            if name in text:
                current_wd = today.weekday()
                days_ahead = wd - current_wd
                if days_ahead <= 0:
                    days_ahead += 7
                target = today + timedelta(days=days_ahead)
                date_str = target.strftime("%Y-%m-%d")
                break

        # 如果 extra 中有 start_date，优先使用
        if extra.get("start_date"):
            date_str = extra["start_date"]

        # ---- 0. 显式时长检测（优先于时段匹配） ----
        # "两小时" / "2小时" / "一个半小时" / "30分钟" / "一个钟"
        import re

        # 中文数字映射
        CN_NUM = {
            "半": 0.5, "一": 1, "二": 2, "两": 2, "三": 3, "四": 4,
            "五": 5, "六": 6, "七": 7, "八": 8, "九": 9, "十": 10,
        }

        dur_minutes = 0

        # "X个半小时" → X * 90min
        half_hour_match = re.search(
            r'(\d+(?:\.\d+)?|[一二两三四五六七八九十半])个半小[时钟]', text
        )
        if half_hour_match:
            num_str = half_hour_match.group(1)
            try:
                n = float(num_str)
            except ValueError:
                n = CN_NUM.get(num_str, 1)
            dur_minutes = int(n * 90)
            is_explicit_duration = True

        # "X小时" / "X个小时" / "X个钟"
        hour_match = re.search(
            r'(\d+(?:\.\d+)?|[一二两三四五六七八九十半])个?(?:小|钟)(?:时|头)?', text
        )
        if hour_match and not is_explicit_duration:
            num_str = hour_match.group(1)
            try:
                n = float(num_str)
            except ValueError:
                n = CN_NUM.get(num_str, 1)
            dur_minutes = int(n * 60)
            is_explicit_duration = True

        # "X分钟"
        minute_match = re.search(r'(\d+)\s*分钟', text)
        if minute_match and not is_explicit_duration:
            dur_minutes = int(minute_match.group(1))
            is_explicit_duration = True

        # "半小时"
        if re.search(r'半小时', text) and not is_explicit_duration:
            dur_minutes = 30
            is_explicit_duration = True

        # 2. 解析时间段
        # 精确时间: "下午3点", "14:00", "2点到4点"
        time_range = re.search(
            r'(\d{1,2})[点:：](\d{0,2})?\s*[到至\-~]\s*(\d{1,2})[点:：](\d{0,2})?',
            text
        )
        if time_range:
            h1 = int(time_range.group(1))
            h2 = int(time_range.group(3))
            # 处理下午
            if "下午" in text and h1 < 12 and h1 >= 1:
                h1 += 12
            if "下午" in text and h2 < 12 and h2 >= 1:
                h2 += 12
            start_time = f"{h1:02d}:00"
            end_time = f"{h2:02d}:00"
            # 如果有"X点到Y点"，这是显式的
            is_explicit_duration = True
            # 但如果有独立时长声明，用声明时长覆盖
            if dur_minutes > 0 and not time_range:
                pass  # 用 dur_minutes
            return date_str, start_time, end_time, is_explicit_duration

        # "早上/上午/中午/下午/晚上" 关键时段
        if any(t in text for t in ("早上", "上午", "早晨")):
            start_time = "09:00"
            end_time = "11:00"
        elif "中午" in text:
            start_time = "12:00"
            end_time = "13:30"
        elif any(t in text for t in ("下午", "午后")):
            # 检查是否有具体时间，如 "下午3点"
            hour_match = re.search(r'(\d{1,2})点', text)
            if hour_match:
                h = int(hour_match.group(1))
                if h < 9 and h >= 1:  # 下午1-8点 → 13-20点
                    h += 12
                h = max(8, min(20, h))
                start_time = f"{h:02d}:00"
                end_h = min(h + 2, 20)
                end_time = f"{end_h:02d}:00"
            else:
                start_time = "14:00"
                end_time = "17:00"  # 下午默认 3 小时跨度
        elif "晚上" in text:
            start_time = "18:00"
            end_time = "20:00"

        # 3. 应用显式时长
        if dur_minutes > 0:
            from datetime import datetime as dt
            st = dt.strptime(start_time, "%H:%M")
            et = st + timedelta(minutes=dur_minutes)
            max_et = dt.strptime("20:00", "%H:%M")
            if et <= max_et:
                end_time = et.strftime("%H:%M")
            else:
                end_time = "20:00"

        # extra 中的 time_slot 优先级最高
        if extra.get("time_slot"):
            slot = extra["time_slot"]
            if "-" in slot:
                parts = slot.split("-")
                if len(parts) == 2:
                    start_time = parts[0].strip()
                    end_time = parts[1].strip()
                    is_explicit_duration = True

        return date_str, start_time, end_time, is_explicit_duration

    # ============================================================
    # 请假日期范围解析
    # ============================================================

    @staticmethod
    def _parse_date_range(
        user_input: str, extra: dict = None,
    ) -> dict:
        """
        从用户输入中解析请假日期范围。

        支持：
        - "下周2" / "下周二" → 下周那一天的日期
        - "到下周二" / "到下周五" → 今天到下周五
        - "下周一到周三" → 下周一到下周三
        - "明天到后天" → 明天到后天
        - "请假3天" / "3天" → 今天+2天
        - "这周五" → 本周五

        Returns:
            {"start_date": "YYYY-MM-DD" or "", "end_date": "", "total_days": 0}
        """
        from datetime import datetime, timedelta
        import re

        result = {"start_date": "", "end_date": "", "total_days": 0}
        # 使用 date() 避免 datetime 时间部分干扰天数计算
        today = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)

        weekday_names = {
            "周一": 0, "周二": 1, "周三": 2, "周四": 3,
            "周五": 4, "周六": 5, "周日": 6,
            "星期一": 0, "星期二": 1, "星期三": 2, "星期四": 3,
            "星期五": 4, "星期六": 5, "星期日": 6,
            "周1": 0, "周2": 1, "周3": 2, "周4": 3,
            "周5": 4, "周6": 5, "周日": 6,  # 周日特殊
        }

        def _weekday_to_date(name: str, base: str = "this") -> str:
            """将星期名称转为具体日期。base: 'this' | 'next'"""
            if name in weekday_names:
                target_wd = weekday_names[name]
                current_wd = today.weekday()
                days_ahead = target_wd - current_wd
                if base == "next":
                    days_ahead += 7
                elif base == "this" and days_ahead <= 0:
                    days_ahead += 7  # 这周的已过，推到下周
                target = today + timedelta(days=days_ahead)
                return target.strftime("%Y-%m-%d")
            return ""

        text = user_input

        # 1. "下周一 到 下周三" 范围模式（下周X到下周Y）
        range_pattern = r'(下?周[一二三四五六日1-6]|下?星期[一二三四五六日1-6]|明天|后天|今天)\s*[到至\-~]\s*(下?周[一二三四五六日1-6]|下?星期[一二三四五六日1-6]|明天|后天|今天)'
        range_match = re.search(range_pattern, text)
        if range_match:
            left = range_match.group(1)
            right = range_match.group(2)

            def _parse_single(w: str, inherit_base: str = None) -> str:
                if w in ("今天",):
                    return today.strftime("%Y-%m-%d")
                if w in ("明天",):
                    return (today + timedelta(days=1)).strftime("%Y-%m-%d")
                if w in ("后天",):
                    return (today + timedelta(days=2)).strftime("%Y-%m-%d")
                base = "next" if w.startswith("下") else ("this" if w.startswith("这") else None)
                # 如果当前词没有显式前缀，继承第一个词的前缀
                if base is None and inherit_base:
                    base = inherit_base
                elif base is None:
                    base = "this"
                name = w.lstrip("下").lstrip("这")
                if name.startswith("星期"):
                    name = "周" + name[2:]
                return _weekday_to_date(name, base)

            left_base = "next" if left.startswith("下") else ("this" if left.startswith("这") else None)
            start_d = _parse_single(left, left_base)
            end_d = _parse_single(right, left_base)  # 右侧继承左侧前缀
            if start_d and end_d:
                result["start_date"] = start_d
                result["end_date"] = end_d
                sd = datetime.strptime(start_d, "%Y-%m-%d")
                ed = datetime.strptime(end_d, "%Y-%m-%d")
                result["total_days"] = (ed - sd).days + 1
                return result

        # 2. "到下周X" / "到下周一" 单端范围（今天到那天）
        to_pattern = r'到\s*(下?周[一二三四五六日1-6]|下?星期[一二三四五六日1-6]|明天|后天)'
        to_match = re.search(to_pattern, text)
        if to_match:
            w = to_match.group(1)
            if w in ("明天",):
                end_date = (today + timedelta(days=1)).strftime("%Y-%m-%d")
            elif w in ("后天",):
                end_date = (today + timedelta(days=2)).strftime("%Y-%m-%d")
            else:
                base = "next" if w.startswith("下") else "this"
                name = w.lstrip("下")
                if name.startswith("星期"):
                    name = "周" + name[2:]
                end_date = _weekday_to_date(name, base)
            if end_date:
                result["start_date"] = today.strftime("%Y-%m-%d")
                result["end_date"] = end_date
                sd = today
                ed = datetime.strptime(end_date, "%Y-%m-%d")
                result["total_days"] = (ed - sd).days + 1
                return result

        # 3. 单独的 "下周X" / "下周一"（只有开始日期）
        single_pattern = r'(下?周[一二三四五六日1-6]|下?星期[一二三四五六日1-6])'
        single_match = re.search(single_pattern, text)
        if single_match:
            w = single_match.group(1)
            base = "next" if w.startswith("下") else "this"
            name = w.lstrip("下")
            if name.startswith("星期"):
                name = "周" + name[2:]
            date_val = _weekday_to_date(name, base)
            if date_val:
                result["start_date"] = date_val
                # 默认请假1天，但尝试从上下文推断天数
                dur_match = re.search(r'(\d+|[一二两三四五])天', text)
                if dur_match:
                    cn = {"一": 1, "二": 2, "两": 2, "三": 3, "四": 4, "五": 5}
                    try:
                        days = int(dur_match.group(1))
                    except ValueError:
                        days = cn.get(dur_match.group(1), 1)
                    result["total_days"] = days
                    sd = datetime.strptime(date_val, "%Y-%m-%d")
                    ed = sd + timedelta(days=days - 1)
                    result["end_date"] = ed.strftime("%Y-%m-%d")
                else:
                    result["total_days"] = 1
                    result["end_date"] = date_val
                return result

        # 4. "N天" 模式（没有明确日期时）
        dur_match = re.search(r'(\d+|[一二两三四五六七八九十])\s*天', text)
        if dur_match:
            cn_num = {"一": 1, "二": 2, "两": 2, "三": 3, "四": 4, "五": 5,
                       "六": 6, "七": 7, "八": 8, "九": 9, "十": 10}
            try:
                days = int(dur_match.group(1))
            except ValueError:
                days = cn_num.get(dur_match.group(1), 1)
            # 起始日期优先用明天
            start_d = today + timedelta(days=1)
            result["start_date"] = start_d.strftime("%Y-%m-%d")
            end_d = start_d + timedelta(days=days - 1)
            result["end_date"] = end_d.strftime("%Y-%m-%d")
            result["total_days"] = days
            return result

        # 5. "明天" / "后天"（无范围）
        if "明天" in text:
            d = today + timedelta(days=1)
            result["start_date"] = d.strftime("%Y-%m-%d")
            result["end_date"] = d.strftime("%Y-%m-%d")
            result["total_days"] = 1
            return result
        if "后天" in text:
            d = today + timedelta(days=2)
            result["start_date"] = d.strftime("%Y-%m-%d")
            result["end_date"] = d.strftime("%Y-%m-%d")
            result["total_days"] = 1
            return result

        return result

    # ============================================================
    # 用户偏好记忆
    # ============================================================

    @staticmethod
    def _get_user_preferences(user_id: str = "") -> dict:
        """
        从历史预定记录推断用户偏好。

        Returns:
            {
                "preferred_room_id": int or None,
                "avg_duration_minutes": int (default 60),
                "preferred_start_time": "09:00" or None,
                "recent_titles": ["周会", "项目评审", ...],
                "total_bookings": int,
            }
        """
        prefs = {
            "preferred_room_id": None,
            "avg_duration_minutes": 60,
            "preferred_start_time": None,
            "recent_titles": [],
            "total_bookings": 0,
        }

        try:
            from db.models import MeetingRoomBooking
            from sqlalchemy import create_engine, func
            from sqlalchemy.orm import sessionmaker
            import os

            db_path = os.path.join("data", "ticket_dispatch.db")
            engine = create_engine(
                f"sqlite:///{db_path}", connect_args={"check_same_thread": False}
            )
            SessionLocal = sessionmaker(bind=engine)
            db = SessionLocal()
            try:
                query = db.query(MeetingRoomBooking).filter(
                    MeetingRoomBooking.is_active == 1,
                )
                if user_id:
                    query = query.filter(MeetingRoomBooking.booked_by == user_id)

                bookings = query.order_by(
                    MeetingRoomBooking.created_at.desc()
                ).limit(10).all()

                if not bookings:
                    return prefs

                prefs["total_bookings"] = len(bookings)

                # 最常用的会议室
                room_counts = {}
                for b in bookings:
                    room_counts[b.room_id] = room_counts.get(b.room_id, 0) + 1
                prefs["preferred_room_id"] = max(room_counts, key=room_counts.get)

                # 平均时长
                durations = []
                start_times = []
                for b in bookings:
                    try:
                        from datetime import datetime
                        s = datetime.strptime(b.start_time, "%H:%M")
                        e = datetime.strptime(b.end_time, "%H:%M")
                        diff = (e - s).total_seconds() / 60
                        if diff > 0:
                            durations.append(diff)
                        start_times.append(b.start_time)
                    except Exception:
                        pass

                if durations:
                    avg_dur = int(sum(durations) / len(durations))
                    # 取整到 30 分钟
                    avg_dur = round(avg_dur / 30) * 30
                    prefs["avg_duration_minutes"] = max(30, avg_dur)

                # 最常用的开始时间
                if start_times:
                    from collections import Counter
                    prefs["preferred_start_time"] = Counter(start_times).most_common(1)[0][0]

                # 最近的会议主题
                prefs["recent_titles"] = [
                    b.title for b in bookings[:5] if b.title
                ]

            finally:
                db.close()
        except Exception:
            pass

        return prefs

    # ============================================================
    # 冲突检测 + 替代方案
    # ============================================================

    @staticmethod
    def _check_availability_with_alternatives(
        date_str: str,
        start_time: str,
        end_time: str,
        preferred_room_id: int = None,
    ) -> dict:
        """
        检查会议室可用性，如果冲突则提供替代方案。

        Returns:
            {
                "available": bool,
                "conflict": None or {"room_name": "...", "booking_title": "..."},
                "available_rooms": [{"id": 1, "name": "A101", "available": True}, ...],
                "alternative_slots": [{"start": "10:00", "end": "11:30"}, ...],
            }
        """
        result = {
            "available": True,
            "conflict": None,
            "available_rooms": [],
            "alternative_slots": [],
        }

        try:
            from db.models import MeetingRoom, MeetingRoomBooking
            from sqlalchemy import create_engine
            from sqlalchemy.orm import sessionmaker
            from datetime import datetime, timedelta
            import os

            db_path = os.path.join("data", "ticket_dispatch.db")
            engine = create_engine(
                f"sqlite:///{db_path}", connect_args={"check_same_thread": False}
            )
            SessionLocal = sessionmaker(bind=engine)
            db = SessionLocal()
            try:
                # 1. 检查首选房间是否可用
                if preferred_room_id:
                    conflict = db.query(MeetingRoomBooking).filter(
                        MeetingRoomBooking.room_id == preferred_room_id,
                        MeetingRoomBooking.date == date_str,
                        MeetingRoomBooking.status == "confirmed",
                        MeetingRoomBooking.is_active == 1,
                        MeetingRoomBooking.start_time < end_time,
                        MeetingRoomBooking.end_time > start_time,
                    ).first()

                    if conflict:
                        room = db.query(MeetingRoom).filter(
                            MeetingRoom.id == preferred_room_id
                        ).first()
                        result["available"] = False
                        result["conflict"] = {
                            "room_name": room.name if room else str(preferred_room_id),
                            "booking_title": conflict.title,
                            "time": f"{conflict.start_time}-{conflict.end_time}",
                        }

                # 2. 查找同时段可用的所有房间
                all_rooms = db.query(MeetingRoom).filter(
                    MeetingRoom.is_active == 1,
                    MeetingRoom.status == "available",
                ).all()

                for room in all_rooms:
                    has_conflict = db.query(MeetingRoomBooking).filter(
                        MeetingRoomBooking.room_id == room.id,
                        MeetingRoomBooking.date == date_str,
                        MeetingRoomBooking.status == "confirmed",
                        MeetingRoomBooking.is_active == 1,
                        MeetingRoomBooking.start_time < end_time,
                        MeetingRoomBooking.end_time > start_time,
                    ).first()

                    entry = {
                        "id": room.id,
                        "name": room.name,
                        "capacity": room.capacity,
                        "location": room.location,
                        "available": not bool(has_conflict),
                    }
                    if has_conflict:
                        entry["conflict_with"] = has_conflict.title
                    result["available_rooms"].append(entry)

                # 按可用优先 + 首选房间优先排序
                result["available_rooms"].sort(
                    key=lambda r: (
                        not r["available"],
                        0 if r["id"] == preferred_room_id else 1,
                    )
                )

                # 3. 如果首选不可用，为该房间找邻近时段
                if result["conflict"] and preferred_room_id:
                    # 在该日期已有的预定
                    existing = db.query(MeetingRoomBooking).filter(
                        MeetingRoomBooking.room_id == preferred_room_id,
                        MeetingRoomBooking.date == date_str,
                        MeetingRoomBooking.status == "confirmed",
                        MeetingRoomBooking.is_active == 1,
                    ).order_by(MeetingRoomBooking.start_time).all()

                    # 构建占用时段列表
                    busy_slots = [
                        (b.start_time, b.end_time) for b in existing
                    ]

                    # 在 08:00-20:00 范围内找空闲窗口
                    day_start = datetime.strptime("08:00", "%H:%M")
                    day_end = datetime.strptime("20:00", "%H:%M")
                    req_duration = (
                        datetime.strptime(end_time, "%H:%M") -
                        datetime.strptime(start_time, "%H:%M")
                    )

                    cursor = day_start
                    while cursor + req_duration <= day_end:
                        slot_start = cursor.strftime("%H:%M")
                        slot_end = (cursor + req_duration).strftime("%H:%M")

                        # 检查是否与占用冲突
                        free = True
                        for bs, be in busy_slots:
                            if slot_start < be and slot_end > bs:
                                free = False
                                break

                        if free:
                            result["alternative_slots"].append({
                                "start": slot_start,
                                "end": slot_end,
                            })
                            if len(result["alternative_slots"]) >= 3:
                                break

                        cursor += timedelta(minutes=30)

            finally:
                db.close()
        except Exception:
            pass

        return result

    # ============================================================
    # 智能卡片构建
    # ============================================================

    async def _build_card(self, ticket_type: str, params: dict, user_input: str) -> dict:
        """
        构建智能确认卡片。

        核心改进：
        - 从用户输入解析自然时间表达（"明天早上"→具体日期+弹性时段）
        - 从用户历史预定推断偏好（常用房间、平均时长、常用时间）
        - 检测会议室冲突，提供替代房间/时间建议
        """
        now = datetime.now()

        if ticket_type == "admin":
            extra = params.get("extra", {})
            service_type = extra.get("service_type", "")

            # 1. 智能解析时间
            parsed_date, parsed_start, parsed_end, is_explicit_duration = (
                self._parse_time_expression(user_input, extra)
            )

            # 2. 获取用户偏好
            user_prefs = self._get_user_preferences(user_id="")
            preferred_room_id = user_prefs.get("preferred_room_id")

            # 根据用户历史平均时长调整 end_time
            # 但用户显式指定时长的（如"两小时"）不覆盖
            if user_prefs.get("avg_duration_minutes") and not is_explicit_duration:
                from datetime import datetime as dt, timedelta
                avg_min = user_prefs["avg_duration_minutes"]
                st = dt.strptime(parsed_start, "%H:%M")
                et = st + timedelta(minutes=avg_min)
                if et <= dt.strptime("20:00", "%H:%M"):
                    parsed_end = et.strftime("%H:%M")

            # 3. 冲突检测
            availability = self._check_availability_with_alternatives(
                parsed_date, parsed_start, parsed_end, preferred_room_id,
            )

            # 4. 获取会议室列表（含设备标签 + 设备需求匹配）
            room_options = []
            raw_rooms = []  # 保留原始数据用于排序

            # 设备关键词 → emoji 标签
            AMENITY_MAP = {
                "视频会议": "📹视频", "投影仪": "📽投影", "投影": "📽投影",
                "电话会议": "📞电话", "白板": "📝白板", "白板墙": "📝白板",
                "音响": "🔊音响", "茶水": "🍵茶水", "显示器": "🖥显示",
                "站立桌": "🧍站立", "WiFi": "📶WiFi", "wifi": "📶WiFi",
            }
            # 从用户输入中提取设备需求（作用域延到 card 构建）
            requested_amenity_keys = set()
            for kw in AMENITY_MAP:
                if kw.lower() in user_input.lower():
                    requested_amenity_keys.add(kw)

            try:
                from db.models import MeetingRoom
                from sqlalchemy import create_engine
                from sqlalchemy.orm import sessionmaker
                import os

                db_path = os.path.join("data", "ticket_dispatch.db")
                engine = create_engine(
                    f"sqlite:///{db_path}", connect_args={"check_same_thread": False}
                )
                SessionLocal = sessionmaker(bind=engine)
                db = SessionLocal()
                try:
                    raw_rooms = db.query(MeetingRoom).filter(
                        MeetingRoom.is_active == 1,
                        MeetingRoom.status == "available",
                    ).all()
                finally:
                    db.close()
            except Exception:
                raw_rooms = []

            if not raw_rooms:
                room_options = [
                    {"value": str(i), "label": f"{n} ({c}人)"}
                    for i, n, c in [
                        (1, "A101 星空厅", 6), (2, "A201 银河厅", 12),
                        (3, "A301 宇宙厅", 25), (4, "B101 创意坊", 8),
                        (5, "B201 静思阁", 4),
                    ]
                ]
            else:
                # 为每个房间生成设备标签
                room_entries = []
                for r in raw_rooms:
                    amenities = r.amenities or []
                    if isinstance(amenities, str):
                        try:
                            amenities = json.loads(amenities)
                        except Exception:
                            amenities = [amenities]
                    tags = []
                    for am in amenities:
                        for kw, emoji in AMENITY_MAP.items():
                            if kw.lower() in am.lower() and emoji not in tags:
                                tags.append(emoji)
                                break
                    # 计算匹配度
                    match_count = 0
                    if requested_amenity_keys:
                        for am in amenities:
                            for kw in requested_amenity_keys:
                                if kw.lower() in am.lower():
                                    match_count += 1

                    room_entries.append((r, tags, match_count))

                # 有设备需求时：匹配房间排前面
                if requested_amenity_keys:
                    room_entries.sort(key=lambda x: (-x[2], x[0].id))

                room_options = []
                for r, tags, _ in room_entries:
                    tag_str = " ".join(tags[:3]) if tags else ""
                    label = f"{r.name} ({r.capacity}人) — {r.location}"
                    if tag_str:
                        label += f"  [{tag_str}]"
                    room_options.append({"value": str(r.id), "label": label})

            # 4.5 容量关键字检测
            CAPACITY_BIG = {"大", "大点", "大一点", "更大", "最大的", "大的", "大型"}
            CAPACITY_SMALL = {"小", "小点", "小一点", "更小", "最小的", "小的", "小型"}
            capacity_filter = None  # "big" | "small" | None
            import re as _re
            # "X人" / "X个人" → 精确容量
            people_match = _re.search(r'(\d+)\s*[个]?人', user_input)
            min_capacity = int(people_match.group(1)) if people_match else 0

            for kw in CAPACITY_BIG:
                if kw in user_input:
                    capacity_filter = "big"
                    break
            if not capacity_filter:
                for kw in CAPACITY_SMALL:
                    if kw in user_input:
                        capacity_filter = "small"
                        break

            # 按容量排序房间
            room_hint = ""  # 稍后追加设备/冲突信息
            if capacity_filter or min_capacity > 0:
                try:
                    room_with_cap = []
                    for opt in room_options:
                        rid = int(opt["value"])
                        cap = 0
                        for r in raw_rooms:
                            if r.id == rid:
                                cap = r.capacity
                                break
                        room_with_cap.append((opt, cap))

                    if min_capacity > 0:
                        room_with_cap.sort(key=lambda x: (0 if x[1] >= min_capacity else 1, -x[1]))
                        room_hint = f"🔍 筛选 ≥{min_capacity}人"
                    elif capacity_filter == "big":
                        room_with_cap.sort(key=lambda x: -x[1])
                        room_hint = "🔍 优先大会议室"
                    elif capacity_filter == "small":
                        room_with_cap.sort(key=lambda x: x[1])
                        room_hint = "🔍 优先小会议室"

                    room_options = [opt for opt, _ in room_with_cap]
                except Exception:
                    pass

            # 5. 默认选中的会议室：首选 > 容量/设备匹配 > 第一个
            default_room = str(preferred_room_id) if preferred_room_id else None
            if not default_room and room_options:
                default_room = room_options[0]["value"]

            # 如果首选不可用，自动推荐第一个可用房间
            if availability.get("conflict") and availability["available_rooms"]:
                for r in availability["available_rooms"]:
                    if r["available"]:
                        default_room = str(r["id"])
                        break

            # 6. 生成独立起始/结束时间选项（30分钟粒度，8:00-20:00）
            from datetime import datetime as dt, timedelta

            # 起始时间选项
            start_options = []
            t = dt.strptime("08:00", "%H:%M")
            max_t = dt.strptime("19:30", "%H:%M")
            while t <= max_t:
                val = t.strftime("%H:%M")
                is_rec = (val == parsed_start)
                label = val if not is_rec else f"{val} ✨"
                start_options.append({"value": val, "label": label})
                t += timedelta(minutes=30)

            # 结束时间选项（默认 = 起始 + 时长）
            end_options = []
            t = dt.strptime("08:30", "%H:%M")
            max_t = dt.strptime("20:00", "%H:%M")
            while t <= max_t:
                val = t.strftime("%H:%M")
                is_rec = (val == parsed_end)
                label = val if not is_rec else f"{val} ✨"
                end_options.append({"value": val, "label": label})
                t += timedelta(minutes=30)

            # 7. 构建描述文字
            time_desc = {
                "09:00": "早上", "10:00": "上午", "12:00": "中午",
                "14:00": "下午", "18:00": "晚上",
            }
            friendly_time = ""
            for k, label in time_desc.items():
                if parsed_start.startswith(k):
                    friendly_time = label
                    break
            if not friendly_time:
                friendly_time = parsed_start

            weekday_names = ["周一", "周二", "周三", "周四", "周五", "周六", "周日"]
            try:
                target_dt = datetime.strptime(parsed_date, "%Y-%m-%d")
                weekday_label = weekday_names[target_dt.weekday()]
                date_label = f"{parsed_date} ({weekday_label})"
            except Exception:
                date_label = parsed_date

            dur_min = int(
                (dt.strptime(parsed_end, "%H:%M") - dt.strptime(parsed_start, "%H:%M"))
                .total_seconds() / 60
            )
            desc_parts = [f"📅 {date_label}  {friendly_time} {parsed_start}-{parsed_end}（{dur_min}分钟）"]
            if is_explicit_duration:
                desc_parts.append(f"⏱️ 按您要求的时长：{dur_min} 分钟")
            elif user_prefs.get("total_bookings", 0) > 0:
                desc_parts.append(
                    f"💡 根据历史记录，推荐时长 {user_prefs['avg_duration_minutes']} 分钟"
                )
            if availability.get("conflict"):
                c = availability["conflict"]
                desc_parts.append(
                    f"⚠️ {c['room_name']} 在 {c['time']} 已被「{c['booking_title']}」占用"
                )

            # 追加设备/冲突提示到 room_hint
            if requested_amenity_keys:
                amenity_str = ", ".join(requested_amenity_keys)
                room_hint = (
                    f"{room_hint} | 🔍 设备: {amenity_str}" if room_hint
                    else f"🔍 筛选设备: {amenity_str}"
                )
            if availability.get("conflict") and not room_hint:
                room_hint = "已根据可用性筛选"

            card = {
                "type": "booking",
                "title": "📋 会议室预定",
                "description": "\n".join(desc_parts),
                "fields": [
                    {
                        "key": "room_id", "label": "会议室", "type": "select",
                        "options": room_options,
                        "value": default_room,
                        "required": True,
                        "hint": room_hint,
                    },
                    {
                        "key": "date", "label": "日期", "type": "date",
                        "value": parsed_date, "required": True,
                    },
                    {
                        "key": "start_time", "label": "开始时间", "type": "select",
                        "options": start_options,
                        "value": parsed_start,
                        "required": True,
                        "hint": "选择会议开始时间",
                    },
                    {
                        "key": "end_time", "label": "结束时间", "type": "select",
                        "options": end_options,
                        "value": parsed_end,
                        "required": True,
                        "hint": f"推荐时长 {dur_min} 分钟",
                    },
                    {
                        "key": "title", "label": "会议主题", "type": "text",
                        "placeholder": "输入会议主题...",
                        "value": user_prefs.get("recent_titles", [None])[0] if user_prefs.get("recent_titles") else "",
                        "required": True,
                    },
                ],
                "confirm_text": "确认预定",
                "action": "/api/meeting-rooms/{room_id}/book",
                "success_message": (
                    f"✅ {parsed_date} {parsed_start}-{parsed_end} 会议室预定成功！"
                ),
                "fallback_url": "/meeting-rooms",
                "fallback_text": "查看会议室日历",
                # 冲突提示 + 替代建议
                "alerts": [],
            }

            # 添加冲突提示
            if availability.get("conflict"):
                card["alerts"].append({
                    "type": "warning",
                    "message": (
                        f"⚠️ {availability['conflict']['room_name']} 在该时段已被预定。"
                        f"已自动推荐可用会议室。"
                    ),
                })
            # 生成替代时段建议（基于可用起始时间 + 推荐时长）
            alt_slots = []
            for s_opt in start_options:
                s_val = s_opt["value"]
                if s_val == parsed_start:
                    continue
                s_dt = dt.strptime(s_val, "%H:%M")
                e_dt = s_dt + timedelta(minutes=dur_min)
                if e_dt <= dt.strptime("20:00", "%H:%M"):
                    alt_slots.append({"start": s_val, "end": e_dt.strftime("%H:%M")})
            for slot in alt_slots[:3]:
                card["alerts"].append({
                    "type": "info",
                    "message": f"💡 也可选择 {slot['start']}-{slot['end']} 时段",
                })

            if not card["alerts"]:
                del card["alerts"]

            return card

        elif ticket_type == "leave":
            extra = params.get("extra", {})
            from datetime import datetime as dt, timedelta

            # 1. 从 LLM extra 字段提取
            llm_start = extra.get("start_date", "")
            llm_end = extra.get("end_date", "")
            llm_days = extra.get("total_days", 0)
            llm_leave_type = extra.get("leave_type", "")

            # 2. 日期范围兜底：用 _parse_date_range 从用户原始输入解析
            parsed_range = self._parse_date_range(user_input, extra)

            default_start = llm_start or parsed_range.get("start_date", "")
            default_end = llm_end or parsed_range.get("end_date", "")
            total_days = llm_days or parsed_range.get("total_days", 0)

            # 3. 如果还没有结束日期但有开始日期和天数，推算结束日期
            if default_start and total_days and not default_end:
                try:
                    sd = dt.strptime(default_start, "%Y-%m-%d")
                    ed = sd + timedelta(days=total_days - 1)
                    default_end = ed.strftime("%Y-%m-%d")
                except Exception:
                    pass

            # 4. 如果还没有天数但有开始和结束，计算天数
            if default_start and default_end and not total_days:
                try:
                    sd = dt.strptime(default_start, "%Y-%m-%d")
                    ed = dt.strptime(default_end, "%Y-%m-%d")
                    total_days = (ed - sd).days + 1
                except Exception:
                    pass

            # 5. 推断请假类型（LLM 优先 → 关键词兜底）
            default_leave_type = llm_leave_type or "年假"
            if not llm_leave_type:
                type_keywords = [
                    (["病假", "看病", "医院", "不舒服", "生病"], "病假"),
                    (["事假", "有事", "私事", "办事"], "事假"),
                    (["调休", "补休", "加班调休"], "调休"),
                    (["婚假", "结婚", "婚礼"], "婚假"),
                    (["产假", "陪产假", "生育"], "产假"),
                    (["年假", "休假", "度假", "旅游"], "年假"),
                ]
                for kws, leave_type in type_keywords:
                    if any(kw in user_input for kw in kws):
                        default_leave_type = leave_type
                        break

            # 6. 生成描述
            desc_parts = ["请确认以下请假信息，信息已根据您的输入预填："]
            if default_start:
                try:
                    sd = dt.strptime(default_start, "%Y-%m-%d")
                    wd = ["周一", "周二", "周三", "周四", "周五", "周六", "周日"][sd.weekday()]
                    desc_parts.append(f"📅 {default_start} ({wd}) 起")
                except Exception:
                    desc_parts.append(f"📅 {default_start} 起")
            if total_days:
                desc_parts.append(f"共 {total_days} 天")
            if default_leave_type:
                desc_parts.append(f"类型：{default_leave_type}")

            return {
                "type": "confirm",
                "title": "🏖️ 请假申请",
                "description": "\n".join(desc_parts),
                "fields": [
                    {
                        "key": "leave_type", "label": "请假类型", "type": "select",
                        "options": [
                            {"value": "年假", "label": "年假"},
                            {"value": "病假", "label": "病假"},
                            {"value": "事假", "label": "事假"},
                            {"value": "调休", "label": "调休"},
                            {"value": "婚假", "label": "婚假"},
                        ],
                        "value": default_leave_type,
                        "required": True,
                    },
                    {
                        "key": "start_date", "label": "开始日期", "type": "date",
                        "value": default_start, "required": True,
                    },
                    {
                        "key": "end_date", "label": "结束日期", "type": "date",
                        "value": default_end, "required": True,
                    },
                    {
                        "key": "total_days", "label": "天数", "type": "number",
                        "value": str(total_days) if total_days else "",
                        "min": 1, "max": 30,
                        "required": True,
                    },
                ],
                "confirm_text": "提交请假申请",
                "action": "/api/tickets/",
                "method": "POST",
                "body_template": {
                    "user_input": user_input,
                    "ticket_type": "leave",
                    "priority": params.get("priority", "P2"),
                },
                "success_message": "请假申请已提交！可在工单管理页面查看进度。",
                "fallback_url": "/tickets",
                "fallback_text": "查看工单",
            }

        elif ticket_type == "expense":
            extra = params.get("extra", {})
            import re

            # 1. 推断报销类型（LLM优先 → 关键词兜底）
            default_expense_type = extra.get("expense_type", "")
            if not default_expense_type:
                type_keywords = [
                    (["差旅", "出差", "住宿", "机票", "火车票", "酒店"], "差旅费"),
                    (["办公", "文具", "打印", "耗材", "纸张"], "办公用品"),
                    (["打车", "地铁", "公交", "出租车", "交通", "停车"], "交通费"),
                    (["餐", "吃饭", "聚餐", "招待", "宴请", "外卖"], "餐费"),
                    (["培训", "课程", "学习", "考试", "认证"], "培训费"),
                ]
                for kws, exp_type in type_keywords:
                    if any(kw in user_input for kw in kws):
                        default_expense_type = exp_type
                        break
            if not default_expense_type:
                default_expense_type = "差旅费"

            # 2. 推断金额（LLM优先 → 正则兜底）
            default_amount = extra.get("amount", "")
            if not default_amount:
                amount_match = re.search(
                    r'(\d+(?:\.\d{1,2})?)\s*(?:元|块|钱|￥|¥)', user_input
                )
                if amount_match:
                    default_amount = amount_match.group(1)

            # 3. 生成描述
            desc_parts = ["请确认报销信息："]
            if default_expense_type:
                desc_parts.append(f"类型：{default_expense_type}")
            if default_amount:
                desc_parts.append(f"金额：{default_amount} 元")

            return {
                "type": "confirm",
                "title": "💰 报销申请",
                "description": "\n".join(desc_parts),
                "fields": [
                    {
                        "key": "expense_type", "label": "报销类型", "type": "select",
                        "options": [
                            {"value": "差旅费", "label": "差旅费"},
                            {"value": "办公用品", "label": "办公用品"},
                            {"value": "交通费", "label": "交通费"},
                            {"value": "餐费", "label": "餐费"},
                            {"value": "培训费", "label": "培训费"},
                        ],
                        "value": default_expense_type,
                        "required": True,
                    },
                    {
                        "key": "amount", "label": "金额（元）", "type": "number",
                        "value": str(default_amount) if default_amount else "",
                        "min": 0, "required": True,
                    },
                    {
                        "key": "description", "label": "说明", "type": "text",
                        "value": params.get("description", user_input),
                        "required": False,
                    },
                ],
                "confirm_text": "提交报销申请",
                "action": "/api/tickets/",
                "method": "POST",
                "body_template": {
                    "user_input": user_input,
                    "ticket_type": "expense",
                    "priority": params.get("priority", "P2"),
                },
                "success_message": "报销申请已提交！请保留原始发票。",
                "fallback_url": "/tickets",
                "fallback_text": "查看工单",
            }

        elif ticket_type == "it_fault":
            # IT 故障：先搜索知识库提供解决方案
            rag_answer = ""
            try:
                from services.knowledge_service import KnowledgeService
                ks = KnowledgeService()
                await ks.initialize()
                docs = await ks.search(user_input, top_k=3)
                if docs:
                    from agents.sub_agents.enterprise_rag import EnterpriseRAGAgent
                    rag = EnterpriseRAGAgent()
                    rag.knowledge_service = ks
                    rag._initialized = True
                    doc_context = rag._build_doc_context(docs)
                    rag_answer = await rag._synthesize(
                        user_input, docs, ""
                    )
            except Exception:
                pass

            description_parts = []
            if rag_answer:
                # 截取前 500 字作为摘要
                short = rag_answer[:500]
                if len(rag_answer) > 500:
                    short += "..."
                description_parts.append(f"🔍 **知识库匹配到以下解决方案：**\n\n{short}")
                description_parts.append("\n---\n💡 如果以上方案未解决您的问题，请点击下方按钮创建工单。")
            else:
                description_parts.append("未在知识库中找到相关解决方案，请确认是否创建工单。")

            return {
                "type": "confirm",
                "title": "🔧 IT 故障排查",
                "description": "\n".join(description_parts),
                "fields": [
                    {
                        "key": "title", "label": "问题标题", "type": "text",
                        "value": params.get("title", user_input[:30]), "required": True,
                    },
                    {
                        "key": "description", "label": "详细描述", "type": "text",
                        "value": params.get("description", user_input), "required": True,
                    },
                ],
                "confirm_text": "仍需要帮助，创建工单",
                "dismiss_text": "问题已解决",
                "action": "/api/tickets/",
                "method": "POST",
                "body_template": {
                    "user_input": user_input,
                    "ticket_type": "it_fault",
                    "priority": params.get("priority", "P2"),
                },
                "success_message": "IT工单已创建，工程师将尽快处理。",
                "fallback_url": "/tickets",
                "fallback_text": "查看工单进度",
            }

        # 默认卡
        return {
            "type": "confirm",
            "title": f"{TICKET_TYPE_CONFIG.get(ticket_type, {}).get('emoji', '📋')} 确认创建工单",
            "description": f"即将创建 {TICKET_TYPE_CONFIG.get(ticket_type, {}).get('label', '工单')}：",
            "fields": [
                {"key": "title", "label": "标题", "type": "text",
                 "value": params.get("title", user_input[:30]), "required": True},
                {"key": "description", "label": "描述", "type": "text",
                 "value": params.get("description", user_input), "required": False},
            ],
            "confirm_text": "确认创建",
            "action": "/api/tickets/",
            "method": "POST",
            "body_template": {
                "user_input": user_input,
                "ticket_type": ticket_type,
                "priority": params.get("priority", "P2"),
            },
            "success_message": "工单已创建！",
        }

    async def _extract_params(
        self,
        user_input: str,
        urgency: str = "medium",
        conversation_history: str = "",
    ) -> dict:
        """
        使用 LLM Function Calling 从用户输入中提取工单参数。

        v3.2: bind_tools + tool_choice="auto" 优先，
        模型未调用 tool 时自动降级为 prompt→JSON。
        """
        from datetime import datetime as dt_now, timedelta

        today_dt = dt_now.now()
        today_str = today_dt.strftime("%Y-%m-%d")
        weekday_cn = ["周一", "周二", "周三", "周四", "周五", "周六", "周日"]
        today_label = f"{today_str} ({weekday_cn[today_dt.weekday()]})"
        tomorrow_str = (today_dt + timedelta(days=1)).strftime("%Y-%m-%d")
        day_after_str = (today_dt + timedelta(days=2)).strftime("%Y-%m-%d")

        ticket_type_options = "\n".join([
            f"  - {k}: {v['label']}（如：{'/'.join(v['category_options'][:3])}...）"
            for k, v in TICKET_TYPE_CONFIG.items()
        ])

        history_section = ""
        if conversation_history:
            history_section = f"## 对话历史\n{conversation_history}\n\n"

        system_prompt = (
            "你是一个企业工单系统的参数提取器。调用 extract_ticket_params 函数提取工单信息。\n"
            f"今天的日期是 {today_label}。\n"
            f"明天={tomorrow_str}，后天={day_after_str}。\n"
            "- '下周2' → 下周周二 → 从今天算起找到下周二的日期（YYYY-MM-DD）\n"
            "- '3天' / '请假3天' → 开始日期=当天或明天，结束日期=开始日期+天数-1\n"
            f"默认 priority={urgency}（用户未指定时用 P2）。\n"
            "leave 类型的 start_date/end_date 必须按今天日期推算，不能留空。\n"
            "\n"
            "优先级判断：\n"
            "P0: 系统宕机、核心业务中断、多人受影响\n"
            "P1: 影响工作效率但可暂时绕过\n"
            "P2: 一般故障/申请，有替代方案\n"
            "P3: 咨询、非紧急问题"
        )

        user_prompt = (
            f"## 工单类型\n{ticket_type_options}\n\n"
            f"{history_section}"
            f"## 用户输入\n{user_input}\n\n"
            f"请调用 extract_ticket_params 函数提取工单参数。"
        )

        try:
            # ── 主路径：Function Calling ──
            response = await self.llm_extract.ainvoke([
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ])

            data = None
            if response.tool_calls:
                tool_call = response.tool_calls[0]
                args = tool_call.get("args", {})
                if isinstance(args, str):
                    args = json.loads(args)
                data = args
                self.logger.info(
                    f"[_extract_params:FC] ticket_type={data.get('ticket_type')}, "
                    f"title={data.get('title', '')[:30]}"
                )
            else:
                # ── Fallback: prompt→JSON ──
                self.logger.info("[_extract_params] 模型未调用 tool，降级为 prompt→JSON")
                try:
                    data = self._parse_json(response.content)
                except ValueError:
                    raise

            if data is None:
                raise ValueError("无法获取提取结果")

            ticket_type = data.get("ticket_type", "it_fault")
            if ticket_type not in TICKET_TYPE_CONFIG:
                ticket_type = "it_fault"

            return {
                "ticket_type": ticket_type,
                "title": data.get("title", user_input[:30]),
                "description": data.get("description", user_input),
                "category": data.get("category", "其他"),
                "priority": data.get("priority", "P2"),
                "extra": data.get("extra", {}),
            }

        except Exception as e:
            self.logger.warning(f"参数提取失败: {e}，使用规则兜底")
            return self._fallback_extract(user_input, urgency)

    @staticmethod
    def _parse_json(text: str) -> dict:
        """从 LLM 文本中提取 JSON（DashScope 兼容）"""
        import json as _json, re as _re
        # 1. 直接解析
        try:
            return _json.loads(text.strip())
        except _json.JSONDecodeError:
            pass
        # 2. 提取 ```json ... ``` 块
        m = _re.search(r'```(?:json)?\s*([\s\S]*?)```', text)
        if m:
            try:
                return _json.loads(m.group(1).strip())
            except _json.JSONDecodeError:
                pass
        # 3. 提取第一个 { ... } 对
        m = _re.search(r'\{[\s\S]*\}', text)
        if m:
            raw = m.group(0)
            try:
                return _json.loads(raw)
            except _json.JSONDecodeError:
                pass
        raise ValueError(f"无法提取 JSON: {text[:200]}")

    def _fallback_extract(self, user_input: str, urgency: str) -> dict:
        """规则兜底：基于关键词快速推断 ticket_type"""
        text = user_input.lower()

        # 关键词匹配 ticket_type
        leave_keywords = ["请假", "休假", "年假", "病假", "事假", "调休", "婚假", "产假"]
        expense_keywords = ["报销", "差旅", "发票", "费用", "账单"]
        admin_keywords = ["会议室", "快递", "寄送", "资产", "访客", "预定", "预约"]

        if any(kw in text for kw in leave_keywords):
            ticket_type = "leave"
            category = "其他"
        elif any(kw in text for kw in expense_keywords):
            ticket_type = "expense"
            category = "其他"
        elif any(kw in text for kw in admin_keywords):
            ticket_type = "admin"
            category = "其他"
        else:
            ticket_type = "it_fault"
            category = "其他"

        return {
            "ticket_type": ticket_type,
            "title": user_input[:30],
            "description": user_input,
            "category": category,
            "priority": "P2",
            "extra": {},
        }

    @staticmethod
    def _build_extra_payload(ticket_type: str, params: dict) -> dict:
        """根据 ticket_type 构建扩展字段 payload"""
        extra = params.get("extra", {})

        if ticket_type == "leave":
            return {
                "leave_type": extra.get("leave_type", ""),
                "start_date": extra.get("start_date", ""),
                "end_date": extra.get("end_date", ""),
                "total_days": extra.get("total_days", 0),
                "reason": extra.get("reason", params.get("description", "")),
            }
        elif ticket_type == "expense":
            return {
                "expense_type": extra.get("expense_type", ""),
                "amount": extra.get("amount", 0.0),
                "has_invoice": extra.get("has_invoice", False),
            }
        elif ticket_type == "admin":
            return {
                "service_type": extra.get("service_type", ""),
                "time_slot": extra.get("time_slot", ""),
                "location": extra.get("location", ""),
            }
        else:  # it_fault
            return {
                "suggested_skill": params.get("suggested_engineer_skill", "通用"),
                "affected_users": extra.get("affected_users", 1),
            }

    def _build_response(self, ticket: dict, ticket_type: str) -> str:
        """根据 ticket_type 构建差异化用户响应"""
        config = TICKET_TYPE_CONFIG.get(ticket_type, TICKET_TYPE_CONFIG["it_fault"])
        priority_emoji = {"P0": "🔴", "P1": "🟠", "P2": "🟡", "P3": "🟢"}
        emoji = priority_emoji.get(ticket["priority"], "📋")
        type_emoji = config["emoji"]

        # 基础信息
        lines = [
            f"{type_emoji} **{config['label']}工单已创建**",
            "",
            f"**工单编号**：{ticket['ticket_number']}",
            f"**标题**：{ticket['title']}",
            f"**优先级**：{emoji} {ticket['priority']}",
            f"**分类**：{ticket['category']}",
            f"**状态**：{ticket['status']}",
        ]

        # 类型特有信息
        payload = ticket.get("payload", {})
        if ticket_type == "leave" and payload:
            if leave_type := payload.get("leave_type"):
                lines.append(f"**请假类型**：{leave_type}")
            if start := payload.get("start_date"):
                lines.append(f"**开始日期**：{start}")
            if end := payload.get("end_date"):
                lines.append(f"**结束日期**：{end}")
            if days := payload.get("total_days"):
                lines.append(f"**天数**：{days}天")

        elif ticket_type == "expense" and payload:
            if expense_type := payload.get("expense_type"):
                lines.append(f"**报销类型**：{expense_type}")
            if amount := payload.get("amount"):
                lines.append(f"**金额**：¥{amount}")
            lines.append(f"**是否有发票**：{'是' if payload.get('has_invoice') else '待确认'}")

        elif ticket_type == "admin" and payload:
            if service_type := payload.get("service_type"):
                lines.append(f"**服务类型**：{service_type}")
            if time_slot := payload.get("time_slot"):
                lines.append(f"**时间段**：{time_slot}")

        # 进度提示
        lines.append("")
        if ticket["priority"] in ("P0", "P1"):
            lines.append("⚡ 您的请求已标记为高优先级，将优先处理。")
        else:
            lines.append("您的工单已进入处理队列，请耐心等待。")

        # 备注
        lines.append("")
        if ticket_type == "leave":
            lines.append("💡 请确认已通过OA系统同步提交请假审批。")
        elif ticket_type == "expense":
            lines.append("💡 请保留原始发票，后续需提交至财务部。")
        elif ticket_type == "admin":
            lines.append("💡 行政人员将在1个工作日内确认并回复。")
        else:
            lines.append("💡 如需查询工单进度，可随时询问我。")

        return "\n".join(lines)

    # ============================================================
    # 卡片回复处理（v3.1 卡片锁）
    # ============================================================

    async def classify_card_response(
        self, user_text: str, card: dict, ticket_type: str,
    ) -> str:
        """
        分类用户对卡片的回复意图。

        v3.2: bind_tools + tool_choice="auto" 优先，
        模型未调用 tool 时自动降级为 prompt→JSON。
        最终 fallback: 关键词规则。

        Returns: "confirm" | "modify" | "cancel" | "new_topic"
        """
        card_desc = card.get("description", "")
        card_fields = card.get("fields", [])

        # 根据卡片类型生成 new_topic 的正例
        if ticket_type == "booking":
            topic_examples = "'请假流程''年假几天''帮我查VPN''打印机怎么用'"
        elif ticket_type in ("it_fault", "it_request"):
            topic_examples = "'请假流程''预定会议室''报销单在哪''你是AI吗'"
        else:
            topic_examples = "'预定会议室''VPN怎么连''打印机坏了''你是AI吗'"

        system_prompt = (
            "你是企业服务台的意图分类器。用户看到了一张确认卡片，然后回复了一句话。\n"
            "请调用 classify_intent 函数判断用户的意图。\n\n"
            "分类标准：\n"
            "- confirm: 用户确认/同意卡片内容，要求执行操作。"
            "包括简短确认（'好的''行''ok'）和长句确认（'行吧那就帮我定了吧谢谢你'）\n"
            "- modify: 用户想修改卡片的某个参数（改时间、换房间、改金额、换主题等）\n"
            "- cancel: 用户想取消/放弃/不要这张卡片（'算了''不定了''取消吧''不要了'）。"
            "注意：cancel 是指放弃卡片操作，不是请假/离开的意思。\n"
            "- new_topic: 用户完全换了话题，问的是和这张卡片不相关的事"
            f"（{topic_examples}）。\n"
            "核心判断：用户的话是否仍然围绕这张卡片？围绕卡片=confirm/modify/cancel，完全不相关=new_topic。"
        )

        try:
            # ── 主路径：Function Calling ──
            response = await self.llm_classify.ainvoke([
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": (
                    f"卡片类型：{ticket_type}\n"
                    f"卡片标题：{card.get('title', '')}\n"
                    f"卡片描述：{card_desc[:300]}\n"
                    f"卡片字段：{json.dumps([f.get('label','') for f in card_fields], ensure_ascii=False)}\n\n"
                    f"用户回复：\"{user_text}\"\n\n"
                    f"请调用 classify_intent 函数分类用户的意图。"
                )},
            ])

            data = None
            if response.tool_calls:
                tool_call = response.tool_calls[0]
                args = tool_call.get("args", {})
                if isinstance(args, str):
                    args = json.loads(args)
                data = args
            else:
                # ── Fallback: prompt→JSON ──
                try:
                    data = self._parse_json(response.content)
                except ValueError:
                    data = None

            if data:
                intent = data.get("intent", "confirm")
                if intent not in ("confirm", "modify", "cancel", "new_topic"):
                    intent = "confirm"
                self.logger.info(
                    f"[classify_card_response] → {intent} "
                    f"(reason: {data.get('reason', '')[:60]})"
                )
                return intent
            else:
                raise ValueError("无法获取分类结果")

        except Exception as e:
            self.logger.warning(f"[classify_card_response] 分类失败: {e}，关键词兜底")
            # ── 最终 fallback: 关键词规则 ──
            card_title = card.get("title", "")
            card_desc_text = card.get("description", "")
            card_text = card_title + card_desc_text
            confirm_kw = ["确认", "好的", "行", "可以", "是", "对", "ok", "yes", "提交", "预定"]
            cancel_kw = ["算了", "不要", "取消", "不了", "不用"]
            modify_kw = ["改", "换", "调整", "修改", "换成", "改成"]
            if any(kw in user_text for kw in cancel_kw):
                return "cancel"
            if any(kw in user_text for kw in modify_kw):
                return "modify"
            card_keywords = set(card_text.replace(" ", ""))
            user_keywords = set(user_text.replace(" ", ""))
            overlap = card_keywords & user_keywords
            if len(overlap) >= 2 or any(kw in user_text for kw in confirm_kw):
                return "confirm"
            return "new_topic"

    async def execute_card(self, card: dict, user_text: str) -> str:
        """
        执行卡片确认后的实际操作。

        根据卡片类型：
        - admin: 预定会议室（调用 MeetingRoom API）
        - leave/expense/it_fault: 创建工单（写入 DB）
        """
        card_type = card.get("type", "")
        action_url = card.get("action", "")
        confirm_text = card.get("confirm_text", "")

        # admin / booking 类型 → 预定会议室
        if card_type == "booking" or "meeting-rooms" in action_url:
            return await self._execute_booking_card(card, user_text)

        # 其他类型 → 创建工单
        return await self._execute_ticket_card(card, user_text)

    async def _execute_booking_card(self, card: dict, user_text: str) -> str:
        """执行会议室预定卡片"""
        try:
            from db.models import MeetingRoom, MeetingRoomBooking
            from sqlalchemy import create_engine
            from sqlalchemy.orm import sessionmaker
            import os

            db_path = os.path.join("data", "ticket_dispatch.db")
            engine = create_engine(
                f"sqlite:///{db_path}", connect_args={"check_same_thread": False}
            )
            SessionLocal = sessionmaker(bind=engine)
            db = SessionLocal()

            try:
                fields = {f["key"]: f.get("value", "") for f in card.get("fields", [])}
                room_id = int(fields.get("room_id", 1))
                date = fields.get("date", datetime.now().strftime("%Y-%m-%d"))
                time_slot = fields.get("time_slot", "14:00-15:00")
                title = fields.get("title", "会议")
                parts = time_slot.split("-")
                start_time = parts[0].strip() if parts else "14:00"
                end_time = parts[1].strip() if len(parts) > 1 else "15:00"

                # 冲突检查
                existing = db.query(MeetingRoomBooking).filter(
                    MeetingRoomBooking.room_id == room_id,
                    MeetingRoomBooking.date == date,
                    MeetingRoomBooking.status == "confirmed",
                    MeetingRoomBooking.start_time < end_time,
                    MeetingRoomBooking.end_time > start_time,
                ).first()

                if existing:
                    return (
                        f"⚠️ 该时段已被「{existing.title}」占用，预定失败。\n"
                        f"请重新选择会议室或时段。"
                    )

                booking = MeetingRoomBooking(
                    room_id=room_id,
                    date=date,
                    start_time=start_time,
                    end_time=end_time,
                    booked_by="web_user",
                    title=title,
                    description="通过聊天卡片预定",
                    status="confirmed",
                )
                db.add(booking)
                db.commit()

                # 获取房间名
                room = db.query(MeetingRoom).filter(
                    MeetingRoom.id == room_id,
                ).first()
                room_name = room.name if room else f"会议室 #{room_id}"

                card["success_message"] = card.get("success_message", "").replace(
                    "会议室预定成功！",
                    f"{room_name} 预定成功！",
                )

                fallback_url = card.get("fallback_url", "/meeting-rooms")
                fallback_text = card.get("fallback_text", "查看会议室日历")

                return (
                    f"{card.get('success_message', '✅ 会议室预定成功！')}\n\n"
                    f"📅 {date}  {start_time}-{end_time}\n"
                    f"🏢 {room_name}\n"
                    f"📝 {title}\n\n"
                    f"[查看会议室日历]({fallback_url})"
                )
            finally:
                db.close()

        except Exception as e:
            logger.error(f"[_execute_booking_card] 失败: {e}")
            return f"会议室预定失败：{e}\n请稍后重试或联系行政人员。"

    async def _execute_ticket_card(self, card: dict, user_text: str) -> str:
        """执行工单创建卡片（IT/请假/报销等）"""
        try:
            body_template = card.get("body_template", {})
            ticket_type = body_template.get("ticket_type", "it_fault")
            priority = body_template.get("priority", "P2")
            fields = {f["key"]: f.get("value", "") for f in card.get("fields", [])}
            title = fields.get("title", user_text[:30])
            description = fields.get("description", user_text)

            # 创建工单
            ticket = self.db_router.ticket.add_ticket(
                ticket_type=ticket_type,
                title=title,
                description=description,
                category=body_template.get("category", "其他"),
                priority=priority,
                requester_id=fields.get("requester_name", "web_user"),
                requester_name=fields.get("requester_name", ""),
                trace_id=f"card_{datetime.now().strftime('%Y%m%d%H%M%S')}",
                payload={},
            )

            success_msg = card.get("success_message", "工单创建成功！")
            fallback_url = card.get("fallback_url", "/tickets")
            fallback_text = card.get("fallback_text", "查看工单")

            return (
                f"{success_msg}\n\n"
                f"📋 工单号：**{ticket.get('ticket_number', 'N/A')}**\n"
                f"📝 标题：{title}\n\n"
                f"[{fallback_text}]({fallback_url})"
            )
        except Exception as e:
            logger.error(f"[_execute_ticket_card] 失败: {e}")
            return f"工单创建失败：{e}\n请稍后重试。"

    async def rebuild_card(
        self, old_card: dict, user_text: str, ticket_type: str,
    ) -> dict:
        """
        用户想修改卡片参数，用新输入重新提取参数并重建卡片。
        """
        try:
            params = await self._extract_params(user_text)

            # 合并旧卡片的参数（保留未修改的部分）
            old_fields = {f["key"]: f.get("value", "") for f in old_card.get("fields", [])}
            for key, val in old_fields.items():
                if key not in params.get("extra", {}):
                    if isinstance(params.get("extra"), dict):
                        params["extra"][key] = val

            return await self._build_card(ticket_type, params, user_text)
        except Exception as e:
            logger.error(f"[rebuild_card] 失败: {e}")
            # 返回原卡片，仅更新描述
            old_card["description"] = (
                f"（已尝试按「{user_text}」调整，但参数解析失败，请手动修改）\n\n"
                + old_card.get("description", "")
            )
            return old_card

    @classmethod
    def get_ticket(cls, ticket_id: int) -> dict | None:
        """查询工单（DB）"""
        from db.db_router import DatabaseRouter
        db = DatabaseRouter()
        return db.ticket.get_ticket(ticket_id)

    @classmethod
    def get_all_tickets(cls, ticket_type: str = None, limit: int = 50) -> list[dict]:
        """获取所有工单（DB）"""
        from db.db_router import DatabaseRouter
        db = DatabaseRouter()
        return db.ticket.list_tickets(ticket_type=ticket_type, limit=limit)

    async def execute_stream(self, message: AgentMessage) -> AsyncGenerator[str, None]:
        """流式执行"""
        yield "[TicketDispatch] 正在分析工单需求..."
        yield "[TicketDispatch] 正在识别工单类型（IT/请假/报销/行政）..."
        yield "[TicketDispatch] 正在提取关键参数..."
        yield "[TicketDispatch] 正在创建工单..."
        yield "[TicketDispatch] 工单已创建，结果返回给编排器"


# 自动注册到全局注册中心
def _register():
    agent_registry.register(
        TicketDispatchSubAgent.__agent_declaration__,
        TicketDispatchSubAgent,
    )

_register()

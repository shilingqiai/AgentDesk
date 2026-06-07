"""
工单派发子Agent — 处理工单创建与派发请求

负责：
- 从用户输入中提取工单参数（标题、描述、优先级、分类）
- 创建工单记录
- 返回工单状态给编排器

YOU ARE A SUB-AGENT. DO NOT REPLY TO USER DIRECTLY.
MUST return structured findings to the Orchestrator.
"""

from __future__ import annotations

import logging
import uuid
from typing import AsyncGenerator
from datetime import datetime

from agents.base_sub_agent import BaseSubAgent
from agents.a2a.protocol import AgentMessage
from agents.orchestrator.agent_declaration import agent_declaration
from agents.orchestrator.agent_registry import agent_registry
from config.model_provider import create_chat_model

logger = logging.getLogger("agent.ticket_dispatch")


@agent_declaration(
    agent_id="ticket_dispatch",
    name="工单派发Agent",
    description=(
        "负责创建、查询和派发IT工单。当用户明确要求提交工单、派工程师、"
        "安排人员处理、提交申请时调用此Agent。"
        "从用户输入中提取工单参数（标题、描述、优先级、分类），创建工单记录并返回状态。"
    ),
    capabilities=[
        "ticket_creation",
        "ticket_query",
        "engineer_dispatch",
        "parameter_extraction",
        "status_tracking",
    ],
    knowledge_domains=[
        "ticket_management",
        "dispatch_workflow",
        "sla_enforcement",
    ],
    priority=2,
)
class TicketDispatchSubAgent(BaseSubAgent):
    """
    工单派发子Agent

    职责：
    1. 从用户输入中使用LLM提取工单参数
    2. 创建工单记录（当前为内存存储，生产环境对接真实工单系统）
    3. 返回工单状态给编排器
    """

    agent_id = "ticket_dispatch"

    # 内存工单存储（生产环境应替换为数据库+工单系统API）
    _tickets: dict[str, dict] = {}

    def __init__(self):
        super().__init__()
        self.llm = create_chat_model(temperature=0.1)

    async def execute(self, message: AgentMessage) -> AgentMessage:
        """
        执行工单派发任务

        编排器委派的消息格式：
            payload.user_input: 用户原始输入
            payload.task: 任务描述
            payload.intent_category: 意图类别
            payload.urgency: 紧急程度

        返回格式：
            payload.ticket_id: 工单ID
            payload.ticket_summary: 工单摘要
            payload.direct_response: 可展示给用户的工单状态消息
            payload.status: 工单状态
        """
        user_input = message.payload.get("user_input", "")
        task = message.payload.get("task", "")
        urgency = message.payload.get("urgency", "medium")

        self.logger.info(
            f"[TicketDispatch] 处理工单请求 (trace={message.trace_id[:8]}...): {task}"
        )

        try:
            # 1. 使用 LLM 提取工单参数
            ticket_params = await self._extract_params(user_input, urgency)

            # 2. 创建工单
            ticket = self._create_ticket(ticket_params, message.trace_id)

            # 3. 生成用户响应
            response = self._build_response(ticket)

            return AgentMessage.create_response(
                from_agent=self.agent_id,
                to_agent=message.from_agent,
                payload={
                    "direct_response": response,
                    "ticket_id": ticket["ticket_id"],
                    "ticket_summary": ticket["title"],
                    "status": ticket["status"],
                    "priority": ticket["priority"],
                    "summary": f"工单 {ticket['ticket_id'][:8]} 已创建",
                    "needs_escalation": ticket["priority"] in ("P0", "P1"),
                },
                original_message=message,
                success=True,
            )

        except Exception as e:
            self.logger.error(f"工单派发失败: {e}")
            return self.create_error_response(message, str(e))

    async def _extract_params(self, user_input: str, urgency: str) -> dict:
        """使用 LLM 从用户输入中提取工单参数"""
        prompt = f"""你是一个工单系统参数提取器。从用户输入中提取工单信息。

用户输入：{user_input}
默认紧急度：{urgency}

请输出 JSON 格式（不要其他文字）：
{{
    "title": "工单标题（简洁，<20字）",
    "description": "工单详细描述",
    "category": "网络故障|系统运维|账号管理|硬件故障|软件问题|其他",
    "priority": "P0|P1|P2|P3",
    "suggested_engineer_skill": "网络|系统|安全|桌面运维|通用"
}}

优先级判断标准：
- P0: 系统宕机、核心业务中断、多人受影响
- P1: 影响工作效率但可暂时绕过
- P2: 一般故障，有替代方案
- P3: 咨询、非紧急问题"""

        try:
            import json
            response = await self.llm.ainvoke([{"role": "user", "content": prompt}])
            content = response.content.strip()
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0].strip()
            elif "```" in content:
                content = content.split("```")[1].split("```")[0].strip()
            return json.loads(content)
        except Exception as e:
            self.logger.warning(f"工单参数提取失败，使用默认值: {e}")
            return {
                "title": user_input[:30],
                "description": user_input,
                "category": "其他",
                "priority": "P2",
                "suggested_engineer_skill": "通用",
            }

    def _create_ticket(self, params: dict, trace_id: str) -> dict:
        """创建工单记录"""
        ticket_id = f"TK-{datetime.now().strftime('%Y%m%d')}-{uuid.uuid4().hex[:6].upper()}"
        ticket = {
            "ticket_id": ticket_id,
            "title": params.get("title", "未命名工单"),
            "description": params.get("description", ""),
            "category": params.get("category", "其他"),
            "priority": params.get("priority", "P2"),
            "status": "已创建",
            "suggested_skill": params.get("suggested_engineer_skill", "通用"),
            "created_at": datetime.now().isoformat(),
            "trace_id": trace_id,
        }
        self._tickets[ticket_id] = ticket
        self.logger.info(f"工单已创建: {ticket_id} [{ticket['priority']}] {ticket['title']}")
        return ticket

    def _build_response(self, ticket: dict) -> str:
        """构建用户可读的工单状态响应"""
        priority_emoji = {"P0": "🔴", "P1": "🟠", "P2": "🟡", "P3": "🟢"}
        emoji = priority_emoji.get(ticket["priority"], "📋")

        return (
            f"{emoji} **工单已创建**\n\n"
            f"**工单编号**：{ticket['ticket_id']}\n"
            f"**标题**：{ticket['title']}\n"
            f"**优先级**：{ticket['priority']}\n"
            f"**分类**：{ticket['category']}\n"
            f"**状态**：{ticket['status']}\n\n"
            f"您的工单已进入处理队列。"
            f"{'由于优先级较高，将优先安排工程师处理。' if ticket['priority'] in ('P0', 'P1') else ''}\n\n"
            f"💡 如需查询工单进度，可随时询问我。"
        )

    @classmethod
    def get_ticket(cls, ticket_id: str) -> dict | None:
        """查询工单"""
        return cls._tickets.get(ticket_id)

    @classmethod
    def get_all_tickets(cls) -> list[dict]:
        """获取所有工单"""
        return list(cls._tickets.values())

    async def execute_stream(self, message: AgentMessage) -> AsyncGenerator[str, None]:
        """流式执行"""
        yield "[TicketDispatch] 正在分析工单需求..."
        yield "[TicketDispatch] 正在提取工单参数..."
        yield "[TicketDispatch] 正在创建工单..."
        yield "[TicketDispatch] 工单已创建，结果返回给编排器"


# 自动注册到全局注册中心
def _register():
    agent_registry.register(
        TicketDispatchSubAgent.__agent_declaration__,
        TicketDispatchSubAgent,
    )

_register()

"""
HR咨询子Agent — 验证Agent声明系统的可扩展性

这是一个示例Agent，展示如何用 @agent_declaration 装饰器
快速注册一个新的专业Agent到Copilot Studio编排架构中。

知识域: HR政策、请假流程、福利查询、入职指引
与其他Agent知识域不重叠。

YOU ARE A SUB-AGENT. DO NOT REPLY TO USER DIRECTLY.
MUST return structured findings to the Orchestrator.
"""

from __future__ import annotations

import logging
from typing import AsyncGenerator

from agents.base_sub_agent import BaseSubAgent
from agents.a2a.protocol import AgentMessage
from agents.orchestrator.agent_declaration import agent_declaration
from agents.orchestrator.agent_registry import agent_registry
from config.model_provider import create_chat_model

logger = logging.getLogger("agent.hr_consultant")


# HR 知识库（内嵌示例数据）
HR_KNOWLEDGE_BASE = {
    "请假政策": (
        "公司请假政策：\n"
        "1. 年假：入职满1年享5天，满3年享10天，满5年享15天\n"
        "2. 病假：需提供医院证明，每年累计不超过30天\n"
        "3. 事假：需提前1天申请，每年累计不超过10天\n"
        "4. 婚假：3天，需提供结婚证\n"
        "5. 产假/陪产假：按国家规定执行\n"
        "请假流程：OA系统提交 → 直属领导审批 → HR备案"
    ),
    "福利查询": (
        "公司福利政策：\n"
        "1. 五险一金：按国家规定缴纳\n"
        "2. 补充商业保险：入职即享\n"
        "3. 年度体检：每年一次\n"
        "4. 餐补：每月500元\n"
        "5. 交通补贴：每月300元\n"
        "6. 培训津贴：每年2000元"
    ),
    "入职指引": (
        "新员工入职流程：\n"
        "1. 签收Offer → 准备入职材料（身份证、学历证明、离职证明）\n"
        "2. 入职当天：HR办理入职手续 → 领取设备（电脑、门禁卡）\n"
        "3. IT开通账号（邮箱、OA、VPN）\n"
        "4. 参加新员工培训（第一周）\n"
        "5. 试用期：3个月，期间有导师带教\n"
        "常见问题：\n"
        "- 设备申请：入职前1周由HR统一向IT提交\n"
        "- 账号开通：入职当天生效"
    ),
    "报销政策": (
        "报销流程：\n"
        "1. 在OA系统提交报销申请\n"
        "2. 上传发票照片（需清晰完整）\n"
        "3. 直属领导审批（3个工作日内）\n"
        "4. 财务审核（5个工作日内）\n"
        "5. 打款到工资卡\n"
        "注意事项：\n"
        "- 单笔超过5000元需部门总监审批\n"
        "- 发票日期需在报销周期内\n"
        "- 差旅报销需附带行程单"
    ),
}


@agent_declaration(
    agent_id="hr_consultant",
    name="HR咨询Agent",
    description=(
        "负责人力资源政策咨询，包括请假流程、福利查询、入职指引、报销政策等。"
        "当用户询问请假、年假、福利、入职、社保、报销等HR相关问题时调用此Agent。"
        "使用内嵌HR知识库进行匹配和回答。"
    ),
    capabilities=[
        "policy_lookup",
        "leave_inquiry",
        "benefits_query",
        "onboarding_guide",
        "expense_guide",
    ],
    knowledge_domains=[
        "hr_policy",
        "leave_management",
        "employee_benefits",
        "onboarding",
        "expense_reimbursement",
    ],
    priority=4,
)
class HRConsultantSubAgent(BaseSubAgent):
    """
    HR咨询子Agent

    职责：
    1. 匹配用户问题到HR知识库
    2. 使用LLM生成自然语言回答
    3. 返回结构化结果给编排器

    这是一个轻量级示例Agent，用于验证：
    - Agent声明系统的可扩展性
    - 新Agent无需修改编排器代码即可注册
    - 非重叠知识域原则
    """

    agent_id = "hr_consultant"

    def __init__(self):
        super().__init__()
        self.llm = create_chat_model(temperature=0.3)

    async def execute(self, message: AgentMessage) -> AgentMessage:
        """
        执行HR咨询任务

        流程：
        1. 关键词匹配 → 查找HR知识库
        2. LLM生成自然回答
        3. 返回结构化结果
        """
        user_input = message.payload.get("user_input", "")
        task = message.payload.get("task", "")

        self.logger.info(
            f"[HR Agent] 处理HR咨询 (trace={message.trace_id[:8]}...): {task}"
        )

        try:
            # 1. 关键词匹配知识库
            matched_topic, knowledge_entry = self._match_knowledge(user_input)

            # 2. 使用LLM生成回答
            if knowledge_entry:
                response = await self._generate_response(
                    user_input, matched_topic, knowledge_entry,
                )
                confidence = 0.8
            else:
                response = await self._generate_unknown_response(user_input)
                confidence = 0.3

            return AgentMessage.create_response(
                from_agent=self.agent_id,
                to_agent=message.from_agent,
                payload={
                    "direct_response": response,
                    "summary": f"HR咨询完成: {matched_topic or '未匹配'}",
                    "confidence": confidence,
                    "matched_topic": matched_topic,
                    "needs_escalation": matched_topic is None,
                },
                original_message=message,
                success=True,
            )

        except Exception as e:
            self.logger.error(f"HR咨询处理失败: {e}")
            return self.create_error_response(message, str(e))

    def _match_knowledge(self, user_input: str) -> tuple[str, str]:
        """
        关键词匹配HR知识库

        Args:
            user_input: 用户输入

        Returns:
            (主题名称, 知识内容) 或 ("", "")
        """
        user_lower = user_input.lower()

        keyword_map = {
            "请假政策": ["请假", "年假", "病假", "事假", "婚假", "产假", "陪产假", "休假"],
            "福利查询": ["福利", "五险一金", "保险", "体检", "餐补", "补贴", "公积金", "社保"],
            "入职指引": ["入职", "新人", "新员工", "报到", "试用期", "设备", "账号", "门禁", "电脑"],
            "报销政策": ["报销", "发票", "差旅", "费用", "oa", "审批"],
        }

        for topic, keywords in keyword_map.items():
            if any(kw in user_lower for kw in keywords):
                return topic, HR_KNOWLEDGE_BASE.get(topic, "")

        return "", ""

    async def _generate_response(
        self, user_input: str, topic: str, knowledge: str,
    ) -> str:
        """使用LLM生成自然语言HR回答"""
        try:
            prompt = f"""你是企业HR助手。请根据以下知识库内容回答员工的问题。

知识库内容 ({topic}):
{knowledge}

员工问题: {user_input}

要求：
1. 回答准确、简洁、友好
2. 基于知识库内容，不要编造
3. 如果知识库不完全覆盖问题，诚实告知
4. 引导员工到正确的HR渠道（如OA系统、HR邮箱）
5. 控制在150字以内"""

            response = await self.llm.ainvoke([{"role": "user", "content": prompt}])
            return response.content.strip()
        except Exception as e:
            # LLM 失败时的兜底直接输出知识库内容
            return f"关于**{topic}**：\n\n{knowledge[:300]}"

    async def _generate_unknown_response(self, user_input: str) -> str:
        """未匹配到知识时的响应"""
        available_topics = "、".join(HR_KNOWLEDGE_BASE.keys())
        return (
            f"关于您询问的问题，HR知识库目前未收录相关信息。\n\n"
            f"我可以帮您解答以下HR问题：\n"
            f"- 请假政策（年假、病假、事假等）\n"
            f"- 福利查询（五险一金、商业保险、补贴等）\n"
            f"- 入职指引（入职流程、设备领取等）\n"
            f"- 报销政策（报销流程、发票要求等）\n\n"
            f"如需进一步帮助，请联系HR部门邮箱: hr@company.com"
        )

    async def execute_stream(self, message: AgentMessage) -> AsyncGenerator[str, None]:
        """流式执行"""
        yield "[HR Agent] 正在查询HR政策知识库..."
        yield "[HR Agent] 正在生成回答..."
        yield "[HR Agent] 结果已返回给编排器"


# 自动注册到全局注册中心
def _register():
    agent_registry.register(
        HRConsultantSubAgent.__agent_declaration__,
        HRConsultantSubAgent,
    )

_register()

"""
行政设施子Agent — 会议室、访客、食堂、办公设施

面向员工的日常行政服务：
- 会议室查询与预定引导
- 访客登记指引
- 食堂/餐饮查询
- 办公设施报修指引
- 公司地址/交通指引

YOU ARE A SUB-AGENT. DO NOT REPLY TO USER DIRECTLY.
"""

from __future__ import annotations

import logging
from typing import AsyncGenerator

from agents.base_sub_agent import BaseSubAgent
from agents.a2a.protocol import AgentMessage
from agents.orchestrator.agent_declaration import agent_declaration
from agents.orchestrator.agent_registry import agent_registry
from config.model_provider import create_chat_model

logger = logging.getLogger("agent.facilities")

# 行政知识库（示例数据，实际可对接OA/飞书文档）
FACILITIES_KNOWLEDGE = {
    "会议室预定": (
        "会议室预定流程：\n"
        "1. 在飞书/企业微信日历中查看空闲会议室\n"
        "2. 选择时段 → 添加参会人 → 预定\n"
        "3. 系统自动发送会议邀请\n\n"
        "会议室资源：\n"
        "- A座3楼: 301(8人)、302(12人)、303(20人)\n"
        "- B座5楼: 501(6人)、502(10人)\n"
        "- 视频会议室: 需提前半天预约\n"
        "规则：最多提前14天预定，单次会议不超过4小时"
    ),
    "访客登记": (
        "访客登记流程：\n"
        "1. 员工在OA/飞书中提交访客申请\n"
        "   - 填写：访客姓名、手机号、来访日期、访问事由\n"
        "2. 审批通过后，访客收到短信（含通行二维码）\n"
        "3. 访客到前台扫码通行\n\n"
        "注意事项：\n"
        "- 提前1天申请，当天紧急访客需电话前台\n"
        "- 访客需携带身份证在前台核验\n"
        "- 团体来访（>5人）需提前3天申请"
    ),
    "食堂餐饮": (
        "食堂信息：\n"
        "- A座B1层：员工食堂（早7:30-9:00、午11:30-13:00、晚17:30-19:00）\n"
        "- B座1层：咖啡厅（8:00-18:00）\n"
        "- 周边推荐：步行5分钟有商圈\n\n"
        "今日菜单查询：打开飞书 → 工作台 → 食堂菜单\n"
        "餐补：每月自动充值500元到工卡"
    ),
    "办公设施": (
        "办公设施报修：\n"
        "1. 灯管/空调/门禁故障 → 飞书提交「设施报修」工单\n"
        "2. IT设备（电脑/打印机/网络）→ 飞书提交「IT报修」工单\n"
        "3. 饮用水/文具领用 → 各楼层前台\n\n"
        "紧急报修电话：\n"
        "- 物业值班: 010-XXXX-XXXX (24小时)\n"
        "- IT值班: 010-XXXX-YYYY (工作日9:00-18:00)"
    ),
    "公司地址": (
        "公司地址：\n"
        "- 主办公区：北京市朝阳区XXX路XX号  XX大厦A座\n"
        "- 分办公区：北京市海淀区YYY路YY号  YY科技园B座\n\n"
        "交通指引：\n"
        "- 地铁：10号线 XX站 B口出 步行500米\n"
        "- 公交：XX路、YY路 XX站下车\n"
        "- 驾车：园区有地下停车场，员工免费（需在OA登记车牌）\n"
        "- 班车：工作日早晚各2班（详情见OA-班车时刻表）"
    ),
    "快递收发": (
        "快递收发：\n"
        "- 公司快递统一送至各楼层前台\n"
        "- 寄件：前台扫码填写电子面单（顺丰/中通可选）\n"
        "- 到付件：前台代收后通知领取\n"
        "- 大件物品：需提前通知行政部安排搬运\n\n"
        "公司地址（收件用）：\n"
        "北京市朝阳区XXX路XX号 XX大厦A座 X层 [姓名/部门]"
    ),
}


@agent_declaration(
    agent_id="facilities",
    name="行政服务Agent",
    description=(
        "负责会议室预定、访客登记、食堂餐饮、办公设施报修、公司地址等日常行政服务。"
        "当用户询问会议室、访客、食堂、报修、地址、快递等行政问题时调用此Agent。"
        "这是一个高频低摩擦场景，日均员工使用频次最高。"
    ),
    capabilities=[
        "meeting_room_booking",
        "visitor_registration",
        "catering_info",
        "facility_repair",
        "office_navigation",
        "mail_delivery",
    ],
    knowledge_domains=[
        "meeting_rooms",
        "visitor_management",
        "catering",
        "facility_maintenance",
        "office_location",
        "mail_services",
    ],
    priority=3,
)
class FacilitiesSubAgent(BaseSubAgent):
    """
    行政设施子Agent

    这是使用频次最高的Agent（日均查询量 > IT咨询）。
    覆盖员工日常行政服务的6大场景。

    复用模式：与HR Agent相同的内嵌知识库+LLM生成回答模式。
    """

    agent_id = "facilities"

    def __init__(self):
        super().__init__()
        self.llm = create_chat_model(temperature=0.3)

    async def execute(self, message: AgentMessage) -> AgentMessage:
        user_input = message.payload.get("user_input", "")
        task = message.payload.get("task", "")

        self.logger.info(
            f"[Facilities] 处理行政咨询 (trace={message.trace_id[:8]}...): {task}"
        )

        try:
            matched_topic, knowledge = self._match_knowledge(user_input)

            if knowledge:
                response = await self._generate_response(user_input, matched_topic, knowledge)
                confidence = 0.85
            else:
                response = self._unknown_response()
                confidence = 0.3

            return AgentMessage.create_response(
                from_agent=self.agent_id,
                to_agent=message.from_agent,
                payload={
                    "direct_response": response,
                    "summary": f"行政咨询: {matched_topic or '未匹配'}",
                    "confidence": confidence,
                    "matched_topic": matched_topic,
                },
                original_message=message,
                success=True,
            )
        except Exception as e:
            self.logger.error(f"行政咨询失败: {e}")
            return self.create_error_response(message, str(e))

    def _match_knowledge(self, user_input: str) -> tuple[str, str]:
        user_lower = user_input.lower()
        keyword_map = {
            "会议室预定": ["会议室", "预定", "开会", "视频会议", "会客"],
            "访客登记": ["访客", "来访", "客人", "参观"],
            "食堂餐饮": ["食堂", "餐厅", "吃饭", "午饭", "晚餐", "菜单", "咖啡"],
            "办公设施": ["报修", "灯", "空调", "门禁", "打印机", "饮水", "文具"],
            "公司地址": ["地址", "怎么走", "地铁", "公交", "停车", "开车", "班车", "在哪"],
            "快递收发": ["快递", "寄件", "包裹", "收件"],
        }
        for topic, keywords in keyword_map.items():
            if any(kw in user_lower for kw in keywords):
                return topic, FACILITIES_KNOWLEDGE.get(topic, "")
        return "", ""

    async def _generate_response(self, user_input: str, topic: str, knowledge: str) -> str:
        try:
            prompt = f"""你是企业行政助手。根据知识库回答员工的问题。

知识库内容 ({topic}):
{knowledge}

员工问题: {user_input}

要求：回答简洁实用，引导员工到正确的渠道操作，控制在150字以内"""
            response = await self.llm.ainvoke([{"role": "user", "content": prompt}])
            return response.content.strip()
        except Exception:
            return f"关于**{topic}**：\n\n{knowledge[:300]}"

    def _unknown_response(self) -> str:
        topics = "、".join(FACILITIES_KNOWLEDGE.keys())
        return (
            f"关于您的问题，行政知识库暂未收录。\n\n"
            f"我可以帮您查询：{topics}\n\n"
            f"如需人工协助，请联系行政部邮箱: admin@company.com"
        )

    async def execute_stream(self, message: AgentMessage) -> AsyncGenerator[str, None]:
        yield "[Facilities] 查询行政服务信息..."
        yield "[Facilities] 生成回复..."
        yield "[Facilities] 结果已返回给编排器"


def _register():
    agent_registry.register(
        FacilitiesSubAgent.__agent_declaration__,
        FacilitiesSubAgent,
    )

_register()

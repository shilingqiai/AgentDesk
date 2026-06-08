"""
三轨道路由器 — Hub & Spoke 架构的核心（v3 Function Calling 版）

v3 改进：
    - 使用 with_structured_output() 替代 prompt→JSON 解析
    - 废除 _extract_json() 和 JSON 修复重试逻辑
    - RouterDecision 直接由 LLM 以原生结构化输出返回
    - confidence < 0.7 不猜测，返回 clarify 让编排器主动反问用户

轨道判定：
    fast:    知识查询/方法问答 → EnterpriseRAGAgent (FAISS + LLM)
    action:  需要调API/创建工单 → TicketDispatchSubAgent
    complex: 多步骤复合指令 → TaskPlanner + 多Agent
    clarify: AI不确定 → 编排器反问用户
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional, Literal

from pydantic import BaseModel, Field
from langchain_core.language_models.chat_models import BaseChatModel

from config.model_provider import create_chat_model

logger = logging.getLogger("orchestrator.router")


# ============================================================
# Pydantic 结构化输出
# ============================================================

class RouterDecision(BaseModel):
    """Router 输出的结构化决策（Pydantic Model）"""
    track: Literal["fast", "action", "complex", "clarify"] = Field(
        description="路由轨道: fast=查资料, action=办事情, complex=复合指令, clarify=不确定需反问"
    )
    confidence: float = Field(
        description="置信度 0.0-1.0。当 < 0.7 时，编排器会主动反问用户以澄清意图",
        ge=0.0, le=1.0,
    )
    reason: str = Field(
        default="", description="路由理由（一句话，用于日志和调试）"
    )
    requires_tools: list[str] = Field(
        default_factory=list,
        description="需要的工具/API列表，如 ['jira_api', 'oa_system']。仅 action/complex 轨道需要",
    )


# ============================================================
# RouteResult (兼容层)
# ============================================================

@dataclass
class RouteResult:
    """
    路由结果（兼容旧版 RouteResult）

    与 RouterDecision 的关系：
        RouterDecision 是 LLM 输出的结构化 JSON
        RouteResult 是内部使用的 dataclass（向后兼容 graph_workflow）
    """
    track: str = "clarify"
    agent_id: str = ""
    reason: str = ""
    params: dict = field(default_factory=dict)
    confidence: float = 0.0
    requires_tools: list[str] = field(default_factory=list)

    @classmethod
    def from_decision(cls, decision: RouterDecision) -> "RouteResult":
        """从 RouterDecision 创建 RouteResult"""
        # 将 track 映射到 agent_id
        agent_map = {
            "fast": "enterprise_rag",
            "action": "ticket_dispatch",
            "complex": "",
            "clarify": "",
        }
        return cls(
            track=decision.track,
            agent_id=agent_map.get(decision.track, ""),
            reason=decision.reason,
            confidence=decision.confidence,
            requires_tools=decision.requires_tools,
        )

    # 向后兼容属性
    @property
    def category(self) -> str:
        mapping = {"fast": "knowledge_query", "action": "ticket_request",
                   "complex": "multi_step", "clarify": "uncertain"}
        return mapping.get(self.track, "uncertain")

    @property
    def urgency(self) -> str:
        return "medium" if self.track == "complex" else "low"

    @property
    def target_agent(self) -> str:
        return self.agent_id

    @property
    def keywords(self) -> list[str]:
        return []

    @property
    def summary(self) -> str:
        return self.reason


# ============================================================
# Router
# ============================================================

class Router:
    """
    语义路由器（v3 — Function Calling）

    职责：
        一次 LLM 调用，通过 with_structured_output 直接返回 RouterDecision。
        废除关键词硬匹配和 JSON 解析 — 模型原生保证结构化输出。

    使用方式:
        router = Router()
        decision = await router.decide(user_input, agent_descriptions, conversation_history)
        result = RouteResult.from_decision(decision)
    """

    def __init__(self, llm: BaseChatModel = None):
        base_llm = llm or create_chat_model(model_type="router", temperature=0)
        self.llm = base_llm.with_structured_output(RouterDecision)
        self._initialize_prompt()

    def _initialize_prompt(self):
        """初始化路由提示词 — 强调不确定时返回 clarify"""
        self.system_prompt = (
            "你是企业AI服务台的路由器。分析用户输入，判定走哪条轨道。\n\n"
            "## 轨道定义\n\n"
            "**fast** — 知识查询/政策咨询/故障排查/方法问答\n"
            "  例：'VPN怎么连' '请假政策是什么' '食堂在哪' '病假'\n"
            "  注意：短追问（≤5字）如'病假''第二步呢'结合对话历史通常走此轨道\n\n"
            "**action** — 需要调接口/创建工单/提交申请/执行操作\n"
            "  例：'帮我提交一个网络故障工单' '申请一台新电脑'\n"
            "  例：'我想请假3天' '报销差旅费500元' '帮我预定会议室'\n"
            "  注意：请假/报销/预定/申请等操作类请求也走此轨道\n\n"
            "**complex** — 涉及2个以上独立任务，或需要多Agent协作\n"
            "  例：'查天气然后请假再取消会议室'\n\n"
            "**clarify** — 以下情况必须返回 clarify：\n"
            "  1. 输入过于模糊，无法判断是查询还是操作（如只输入'请假'）\n"
            "  2. 输入有歧义，可能是查询也可能是操作\n"
            "  3. 与IT/HR/工单/企业服务完全无关（如'股票行情''电影推荐'）\n"
            "  4. AI不确定答案时，宁可反问也不要猜测\n"
            "请始终设置 confidence < 0.7 当你需要 clarify 时。"
        )

    async def decide(
        self,
        user_input: str,
        agent_descriptions: str = "",
        conversation_history: str = "",
    ) -> RouterDecision:
        """
        语义路由决策（v3 — Function Calling）

        with_structured_output 直接返回 RouterDecision 实例，
        无需 JSON 解析、无需重试、无需 json_repair。

        Args:
            user_input: 用户输入文本
            agent_descriptions: 可用 Agent 描述
            conversation_history: 对话历史上下文

        Returns:
            RouterDecision（LLM 原生结构化输出）
        """
        agent_list = agent_descriptions or (
            "enterprise_rag(企业知识库问答: IT/HR/行政), "
            "ticket_dispatch(工单派发: 创建/查询多类型工单-IT故障/请假/报销/行政)"
        )

        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": (
                f"## 可用Agent\n{agent_list}\n\n"
                f"## 对话历史\n{conversation_history or '（首轮对话，无历史）'}\n\n"
                f"分析以下用户输入并返回路由决策：{user_input}"
            )},
        ]

        try:
            decision = await self.llm.ainvoke(messages)
            # with_structured_output 返回的直接就是 RouterDecision 实例
            if not isinstance(decision, RouterDecision):
                # 极少数模型可能返回 dict → 手动构造
                decision = RouterDecision(**decision)

            # 只做 track 合法性检查
            if decision.track not in ("fast", "action", "complex", "clarify"):
                decision = RouterDecision(
                    track="clarify", confidence=0.3,
                    reason=f"LLM返回未知轨道: {decision.track}",
                )

            logger.info(
                f"[Router] track={decision.track}, "
                f"confidence={decision.confidence:.0%}, "
                f"reason={decision.reason[:60] if decision.reason else ''}"
            )
            return decision

        except Exception as e:
            logger.error(f"[Router] structured output 失败: {e}，fallback=clarify")
            return RouterDecision(
                track="clarify", confidence=0.2,
                reason=f"结构化输出异常: {str(e)[:80]}",
            )

    # 向后兼容别名
    async def route(
        self,
        user_input: str,
        agent_descriptions: str = "",
        conversation_history: str = "",
    ) -> RouteResult:
        """向后兼容的路由方法（返回 RouteResult 而非 RouterDecision）"""
        decision = await self.decide(user_input, agent_descriptions, conversation_history)
        return RouteResult.from_decision(decision)


# 向后兼容导出
IntentClassifier = Router
IntentResult = RouteResult

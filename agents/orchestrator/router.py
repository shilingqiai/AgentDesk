"""
三轨道路由器 — Hub & Spoke 架构的核心（v3.1 prompt→JSON 版）

轨道判定：
    fast:    知识查询/方法问答 → EnterpriseRAGAgent (FAISS + LLM)
    action:  需要调API/创建工单 → TicketDispatchSubAgent
    complex: 多步骤复合指令 → TaskPlanner + 多Agent
    clarify: AI不确定 → 编排器反问用户

DashScope 兼容：不使用 with_structured_output（API 不支持 function_calling），
改用 prompt→JSON 解析 + json_repair 兜底。
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Optional, Literal

from pydantic import BaseModel, Field
from langchain_core.language_models.chat_models import BaseChatModel

from config.model_provider import create_chat_model

logger = logging.getLogger("orchestrator.router")


# ============================================================
# RouterDecision
# ============================================================

class RouterDecision(BaseModel):
    """Router 输出的结构化决策"""
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
    """路由结果（兼容 graph_workflow）"""
    track: str = "clarify"
    agent_id: str = ""
    reason: str = ""
    params: dict = field(default_factory=dict)
    confidence: float = 0.0
    requires_tools: list[str] = field(default_factory=list)

    @classmethod
    def from_decision(cls, decision: RouterDecision) -> "RouteResult":
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
    语义路由器（v3.1 prompt→JSON）

    DashScope 兼容：用纯 LLM + prompt 输出 JSON，避免 with_structured_output
    不兼容问题。json_repair 兜底修复格式错误。
    """

    def __init__(self, llm: BaseChatModel = None):
        self.llm = llm or create_chat_model(model_type="main", temperature=0)
        self._initialize_prompt()

    def _initialize_prompt(self):
        self.system_prompt = (
            "你是企业AI服务台的路由器。分析用户输入，判定走哪条轨道。\n\n"
            "## 轨道定义\n\n"
            "**fast** — 知识查询/政策咨询/故障排查/方法问答\n"
            "  例：'VPN怎么连' '请假政策是什么' '食堂在哪' '病假'\n\n"
            "**action** — 需要调接口/创建工单/提交申请/执行操作\n"
            "  例：'帮我提交一个网络故障工单' '申请一台新电脑'\n"
            "  例：'我想请假3天' '报销差旅费500元' '帮我预定会议室'\n\n"
            "**complex** — 涉及2个以上独立任务，或需要多Agent协作\n"
            "  例：'查天气然后请假再取消会议室'\n\n"
            "**clarify** — 以下情况必须返回 clarify：\n"
            "  1. 输入过于模糊，无法判断是查询还是操作\n"
            "  2. 输入有歧义，可能是查询也可能是操作\n"
            "  3. 与IT/HR/工单/企业服务完全无关\n"
            "  4. AI不确定答案时，宁可反问也不要猜测\n\n"
            "## 输出格式（严格 JSON，不要 markdown 包裹，不要注释）\n"
            '{"track":"fast|action|complex|clarify","confidence":0.0-1.0,'
            '"reason":"一句话理由","requires_tools":[]}'
        )

    @staticmethod
    def _extract_json(text: str) -> dict:
        """从 LLM 文本中提取 JSON 对象"""
        # 1. 直接解析
        try:
            return json.loads(text.strip())
        except json.JSONDecodeError:
            pass

        # 2. 提取 ```json ... ``` 块
        m = re.search(r'```(?:json)?\s*([\s\S]*?)```', text)
        if m:
            try:
                return json.loads(m.group(1).strip())
            except json.JSONDecodeError:
                pass

        # 3. 提取第一个 { ... } 对
        m = re.search(r'\{[\s\S]*\}', text)
        if m:
            raw = m.group(0)
            try:
                return json.loads(raw)
            except json.JSONDecodeError:
                # 4. json_repair 兜底
                try:
                    from json_repair import repair_json
                    return json.loads(repair_json(raw))
                except Exception:
                    pass

        raise ValueError(f"无法从 LLM 输出中提取 JSON: {text[:200]}")

    async def decide(
        self,
        user_input: str,
        agent_descriptions: str = "",
        conversation_history: str = "",
    ) -> RouterDecision:
        """语义路由决策（prompt→JSON）"""
        agent_list = agent_descriptions or (
            "enterprise_rag(企业知识库问答: IT/HR/行政), "
            "ticket_dispatch(工单派发: 创建/查询多类型工单-IT故障/请假/报销/行政)"
        )

        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": (
                f"## 可用Agent\n{agent_list}\n\n"
                f"## 对话历史\n{conversation_history or '（首轮对话，无历史）'}\n\n"
                f"分析以下用户输入并返回路由决策 JSON：{user_input}"
            )},
        ]

        try:
            response = await self.llm.ainvoke(messages)
            data = self._extract_json(response.content)

            decision = RouterDecision(
                track=data.get("track", "clarify"),
                confidence=float(data.get("confidence", 0.5)),
                reason=str(data.get("reason", "")),
                requires_tools=data.get("requires_tools", []),
            )

            if decision.track not in ("fast", "action", "complex", "clarify"):
                decision = RouterDecision(
                    track="clarify", confidence=0.3,
                    reason=f"LLM 返回未知轨道: {decision.track}",
                )

        except Exception as e:
            logger.error(f"[Router] 决策失败: {e}，fallback=clarify")
            decision = RouterDecision(
                track="clarify", confidence=0.2,
                reason=f"决策异常: {str(e)[:80]}",
            )

        logger.info(
            f"[Router] track={decision.track}, "
            f"confidence={decision.confidence:.0%}, "
            f"reason={decision.reason[:60] if decision.reason else ''}"
        )
        return decision

    async def route(
        self,
        user_input: str,
        agent_descriptions: str = "",
        conversation_history: str = "",
    ) -> RouteResult:
        """向后兼容的路由方法"""
        decision = await self.decide(user_input, agent_descriptions, conversation_history)
        return RouteResult.from_decision(decision)


# 向后兼容导出
IntentClassifier = Router
IntentResult = RouteResult

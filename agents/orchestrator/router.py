"""
三轨道路由器 — Hub & Spoke 架构的核心（v3.2 bind_tools 版）

轨道判定：
    fast:    知识查询/方法问答 → EnterpriseRAGAgent (FAISS + LLM)
    action:  需要调API/创建工单 → TicketDispatchSubAgent
    complex: 多步骤复合指令 → TaskPlanner + 多Agent
    clarify: AI不确定 → 编排器反问用户

DashScope 兼容：使用 bind_tools + tool_choice="auto" 触发原生 Function Calling，
直接解析 response.tool_calls[0]["args"] 获得结构化决策。
prompt→JSON 作为 fallback（模型未调用 tool 时自动降级）。
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
# OpenAI Function Calling 工具定义
# ============================================================

ROUTER_TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "route_decision",
        "description": "分析用户输入，返回路由决策。判断用户意图属于哪种轨道。",
        "parameters": {
            "type": "object",
            "properties": {
                "track": {
                    "type": "string",
                    "enum": ["fast", "action", "complex", "clarify"],
                    "description": (
                        "路由轨道 — fast: 知识查询/政策咨询/故障排查/方法问答（如VPN怎么连、请假政策）；"
                        "action: 需要调接口/创建工单/提交申请（如请假、报销、预定会议室、创建IT工单）；"
                        "complex: 涉及2个以上独立任务或多Agent协作；"
                        "clarify: 输入模糊/有歧义/无关话题/AI不确定"
                    ),
                },
                "confidence": {
                    "type": "number",
                    "description": "置信度 0.0-1.0。不确定时设 < 0.7",
                    "minimum": 0.0,
                    "maximum": 1.0,
                },
                "reason": {
                    "type": "string",
                    "description": "一句话理由",
                },
                "requires_tools": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "需要的工具/API列表",
                },
            },
            "required": ["track", "confidence", "reason"],
        },
    },
}


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
    语义路由器（v3.2 bind_tools 版）

    优先使用 bind_tools 触发原生 Function Calling：
      - qwen-max 原生支持 tool_calls，直接返回结构化参数
      - 无需 JSON 解析、无需 regex、无需 json_repair
      - tool_choice="auto" 兼容性好，DashScope 全支持

    Fallback: 模型未调用 tool 时自动降级到 prompt→JSON 解析。
    """

    def __init__(self, llm: BaseChatModel = None):
        base_llm = llm or create_chat_model(model_type="main", temperature=0)
        self.llm = base_llm
        # bind_tools + auto: DashScope 已验证全支持（auto/required/object）
        self.llm_with_tools = base_llm.bind_tools(
            [ROUTER_TOOL_SCHEMA], tool_choice="auto"
        )
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
            "  4. AI不确定答案时，宁可反问也不要猜测"
        )

    # ── JSON 提取 fallback ──────────────────────────────

    @staticmethod
    def _extract_json(text: str) -> dict:
        """从 LLM 文本中提取 JSON 对象（prompt→JSON fallback）"""
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

    # ── 核心决策方法 ────────────────────────────────────

    async def decide(
        self,
        user_input: str,
        agent_descriptions: str = "",
        conversation_history: str = "",
    ) -> RouterDecision:
        """
        语义路由决策（v3.2 — bind_tools 优先，prompt→JSON fallback）

        优先通过 bind_tools 触发原生 Function Calling，
        直接从 response.tool_calls 获取结构化参数。
        如果模型未调用 tool，自动降级为 prompt→JSON 解析。
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
            # ── 主路径：Function Calling ──
            response = await self.llm_with_tools.ainvoke(messages)

            if response.tool_calls:
                # 原生 Function Calling 返回 → 直接解析 args
                tool_call = response.tool_calls[0]
                args = tool_call.get("args", {})

                if isinstance(args, str):
                    # 极少数模型可能返回 JSON 字符串
                    args = json.loads(args)

                decision = RouterDecision(
                    track=args.get("track", "clarify"),
                    confidence=float(args.get("confidence", 0.5)),
                    reason=str(args.get("reason", "")),
                    requires_tools=args.get("requires_tools", []),
                )

                if decision.track not in ("fast", "action", "complex", "clarify"):
                    decision = RouterDecision(
                        track="clarify", confidence=0.3,
                        reason=f"LLM 返回未知轨道: {decision.track}",
                    )

                logger.info(
                    f"[Router:FC] track={decision.track}, "
                    f"confidence={decision.confidence:.0%}, "
                    f"reason={decision.reason[:60] if decision.reason else ''}"
                )
                return decision

            # ── Fallback: 模型未调用 tool → prompt→JSON ──
            logger.info("[Router] 模型未调用 tool，降级为 prompt→JSON")
            data = self._extract_json(response.content)

            decision = RouterDecision(
                track=data.get("track", "clarify"),
                confidence=float(data.get("confidence", 0.5)),
                reason=str(data.get("reason", "")),
                requires_tools=data.get("requires_tools", []),
            )

        except Exception as e:
            logger.error(f"[Router] 决策失败: {e}，fallback=clarify")
            decision = RouterDecision(
                track="clarify", confidence=0.2,
                reason=f"决策异常: {str(e)[:80]}",
            )

        # 合法性兜底
        if decision.track not in ("fast", "action", "complex", "clarify"):
            decision = RouterDecision(
                track="clarify", confidence=0.3,
                reason=f"LLM 返回未知轨道: {decision.track}",
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

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
    track: Literal["fast", "action_query", "action_create", "complex", "clarify"] = Field(
        description="路由轨道: fast=查资料, action_query=查个人数据, action_create=办事情, complex=复合指令, clarify=不确定需反问"
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
                    "enum": ["fast", "action_query", "action_create", "complex", "clarify"],
                    "description": (
                        "路由轨道 — fast: 知识查询/政策咨询/故障排查/方法问答（如VPN怎么连、请假政策）。"
                        "但若对话历史显示助手已提供方案且用户说无效（还是不行/没用/无法解决），勿走此轨道；"
                        "action_query: 纯数据查询，用户只想查看个人数据（如查剩余年假、查工单进度），无意发起申请；"
                        "action_create: 单步操作，执行不需要合规检查的操作（如创建IT工单、寄快递、简单行政请求）；"
                        "complex: 发起申请类操作，需要查政策+查余额+合规判断（如请假、报销），无论参数是否完整；"
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
            "action_query": "tool_agent",
            "action_create": "ticket_dispatch",
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
        mapping = {"fast": "knowledge_query", "action_query": "personal_query",
                   "action_create": "ticket_request", "complex": "multi_step",
                   "clarify": "uncertain"}
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
            "你是企业AI服务台的路由器。分析用户输入和对话历史，判定走哪条轨道。\n\n"
            "## 核心原则：判断用户的业务意图，而非匹配关键词\n\n"
            "## 重要：多轮对话中的升级信号\n"
            "如果对话历史显示助手已经提供了排障步骤/解决方案，且用户回复表示无效（如'还是不行'"
            "'无法解决''没用''试过了''还是连不上'等），用户意图是「创建工单」而非「再次查询」。"
            "此时必须走 **action_create** 轨道！！切勿走 fast 重复回答相同内容。\n\n"
            "## 轨道判定规则（按优先级）\n\n"
            "### 1. 先判断：是否含「我」或具体人名？\n"
            "含「我」「我的」「张三」等 → 排除 fast（fast 只能回答通用政策，不能回答个人数据）\n\n"
            "### 2. 再判断：用户的谓语是什么？\n"
            "- 谓语是「查/看/还剩/还有/多少」（无意发起申请）→ 纯查询意图\n"
            "- 谓语是「想/要/帮我/申请/提交/请」（有意发起申请）→ 操作意图\n\n"
            "### 3. 轨道选择\n\n"
            "**action_query** — 用户只想「查看/了解」个人数据，无发起申请意图：\n"
            "  特征：谓语是查/看/还剩/还有，不包含想/要/帮我/申请\n"
            "  例：'我还有几天年假' '查我的剩余年假' '我的工单进度怎么样'\n"
            "       '张三的年假还剩多少' '帮我查一下我的余额'\n"
            "  注：'帮我查'仍是查询意图，不是操作意图\n\n"
            "**action_create** — 用户要「执行」一个不需要合规检查的操作：\n"
            "  特征：IT故障报修、简单行政请求、方案无效后的升级\n"
            "  例：'VPN连不上帮我建工单' '打印机坏了' '帮我寄个快递'\n"
            "       '还是不行' '试过了没用'（对话历史有排障建议时）\n"
            "  注意：请假/报销类不属此轨道！即使说'帮我请假'也走 complex\n\n"
            "**complex** — 用户意图是「发起申请」，需 RAG查政策 + 工具查余额 + 合规判断：\n"
            "  特征：核心意图是请假/报销申请，无论是否提供完整参数\n"
            "  例：'我要休年假' '请3天病假' '想报销差旅费' '申请年假'\n"
            "       '帮我提交请假' '我要报销' '休假' '请假'\n"
            "  关键：即使只说'我要休假'没写天数 → 仍走 complex（下游Agent做Slot Filling反问）\n"
            "  关键：即使只说'报销'没写金额 → 仍走 complex\n"
            "  关键：'帮我请假'虽然含'帮我'，但核心意图是申请，走 complex 不是 action_create\n\n"
            "**fast** — 用户问的是通用政策/流程/知识（不含个人数据）：\n"
            "  特征：问规定/流程/方法，不涉及「我」或具体人\n"
            "  例：'年假政策是什么' '怎么报销' '食堂在哪' 'VPN怎么排查'\n"
            "  注意：含「我」字的（如'我的年假'）绝对不是 fast！\n"
            "  注意：如果用户之前刚收到过类似答案却说没用/不行，不要走此轨道\n\n"
            "**clarify** — 以下情况必须返回 clarify：\n"
            "  1. 输入过于模糊，无法判断意图（仅'你好''在吗'等寒暄）\n"
            "  2. 与IT/HR/工单/企业服务完全无关\n"
            "  3. AI不确定答案时，宁可反问也不要猜测"
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

                if decision.track not in ("fast", "action_query", "action_create", "complex", "clarify"):
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
        if decision.track not in ("fast", "action_query", "action_create", "complex", "clarify"):
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

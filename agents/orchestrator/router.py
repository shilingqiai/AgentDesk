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
    """
    Router 输出的结构化决策

    v7: 新增 dynamic 轨道 — 所有需要多工具编排的请求统一走 DynamicActionAgent (ReAct循环)
        fast → 通用知识查询 (RAG)
        dynamic → 需要工具编排 (ReAct自由编排，替代旧的 action_query/action_create/complex)
        clarify → AI不确定时反问
    """
    track: Literal["fast", "dynamic", "complex", "clarify"] = Field(
        description="路由轨道: fast=知识查询, dynamic=工具编排(ReAct), complex=请假/报销固定DAG, clarify=反问"
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
                    "enum": ["fast", "dynamic", "complex", "clarify"],
                    "description": (
                        "路由轨道。优先判断：是否能一步回答 ——\n"
                        "fast: 通用知识/政策/流程查询，不涉及个人信息和实际操作"
                        "（如'年假政策''VPN怎么排查''食堂在哪'）→ RAG直接回答\n"
                        "dynamic: 需要多个工具编排的请求，涉及查询+判断+操作组合"
                        "（如'查库存再建单''准备入职设备''会议室预定'），"
                        "让DynamicActionAgent自主决定调用哪些工具和顺序。\n"
                        "complex: 请假/报销 — 固定DAG（查政策+查余额→合规检查→确认卡片），"
                        "即使参数不完整也走此轨道\n"
                        "clarify: 输入模糊/AI不确定"
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
            "dynamic": "dynamic_action",
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
        mapping = {"fast": "knowledge_query", "dynamic": "dynamic_action",
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
        self.last_response = None  # 最近一次 LLM 响应（供 Token Budget 扣减）

    def _initialize_prompt(self):
        self.system_prompt = (
            "你是企业AI服务台的路由器。分析用户输入和对话历史，判定走哪条轨道。\n\n"
            "## 核心原则：判断用户的业务意图，而非匹配关键词\n\n"
            "## 重要：多轮对话中的话题切换\n"
            "对话历史可能包含已完成的旧话题（如已提交的请假工单）。"
            "如果用户当前输入的历史话题完全无关（如历史是请假、当前是设备领用），"
            "必须只根据当前输入判定轨道，不要让历史话题影响判断。\n"
            "例: 对话历史是请假流程 → 用户当前说'帮我给新入职前端准备设备'\n"
            "  → 这跟请假无关，走 dynamic（设备领用场景），不要关联请假上下文。\n\n"
            "## 重要：多轮对话中的升级信号\n"
            "如果对话历史显示助手已经提供了排障步骤/解决方案，且用户回复表示无效（如'还是不行'"
            "'无法解决''没用''试过了''还是连不上'等），用户意图是「创建工单」而非「再次查询」。"
            "此时必须走 **dynamic** 轨道（让 DynamicActionAgent 自动升级为工单）。\n\n"
            "## 轨道判定（按优先级）\n\n"
            "### 第一步：是否只需回答通用知识？\n"
            "**fast** — 用户问通用政策/流程/方法，不涉及个人数据、不需要执行操作：\n"
            "  例：'年假政策是什么' 'VPN怎么排查' '食堂在哪' '入职需要什么设备'\n"
            "  特征：纯问答，不需要查数据库、不需要创建工单\n"
            "  反例：含「帮我」「我要」等操作词的不是 fast\n"
            "  反例：含「我」「张三」等个人指代的不是 fast\n\n"
            "### 第二步：是否请假/报销？\n"
            "**complex** — 请假/报销走固定 DAG（查政策 ∥ 查余额 → 合规检查 → 确认卡片）：\n"
            "  例：'我想请3天年假' '我要申请病假' '报销昨天的差旅费'\n"
            "  特征：流程固定，必须经过政策查询+余额/额度检查+合规确认\n"
            "  即使参数不完整也走 complex — complex_track_node 会通过 TicketDispatch 做 Slot Filling\n"
            "  注意：'年假政策是什么'这类纯政策问答走 fast，不是 complex\n\n"
            "### 第三步：是否需要工具编排？\n"
            "**dynamic** — 用户意图涉及以下任一场景时走此轨道：\n"
            "  - 需要查询数据+根据结果做判断（查库存→领用/采购）\n"
            "  - 需要多个工具编排（搜知识库+查库存+建工单）\n"
            "  - 创建工单、设备领用、采购申请、IT故障报告\n"
            "  - 会议室预定\n"
            "  - 方案无效后的升级（'还是不行'→自动建工单）\n"
            "  DynamicActionAgent 拥有全部工具能力，会自主决定调用哪些工具、什么顺序、何时确认。\n\n"
            "### 第四步：不确定时 clarify\n"
            "**clarify** — 输入极其模糊或完全不相关时反问。"
            "宁可反问也不走错轨道。"
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
            self.last_response = response  # 供 Token Budget 扣减

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

                if decision.track not in ("fast", "dynamic", "complex", "clarify"):
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
        if decision.track not in ("fast", "dynamic", "complex", "clarify"):
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

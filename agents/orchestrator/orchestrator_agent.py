"""
Orchestrator Agent — 中央编排器

参考 Microsoft Copilot Studio 的 Orchestrator 模式：

核心职责：
1. Intent Classification — 理解用户意图
2. Task Planning — 制定执行计划
3. Agent Delegation — 委派任务给专业子Agent
4. Response Synthesis — 合成子Agent结果，统一响应用户
5. Human-in-the-Loop — 高风险操作等待人工确认

Single Response Principle：
    编排器是唯一与用户直接对话的Agent。
    所有子Agent静默返回结果给编排器。

调用模式（来自 Microsoft 最佳实践）：
    invoke → wait for all → combine → respond
"""

from __future__ import annotations

import logging
from typing import AsyncGenerator, Optional

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, AIMessage

from config.model_provider import create_chat_model

from .agent_registry import agent_registry, AgentRegistry
from .intent_classifier import IntentClassifier, IntentResult
from .task_planner import TaskPlanner, ExecutionPlan
from .response_synthesizer import ResponseSynthesizer

from agents.a2a.protocol import AgentMessage, MessageIntent
from agents.a2a.message_bus import message_bus, MessageBus
from agents.a2a.context_manager import context_manager, ContextManager

logger = logging.getLogger("orchestrator")


class OrchestratorAgent:
    """
    中央编排器 Agent

    这是整个多Agent系统的核心调度器。
    参考 Microsoft Copilot Studio 的编排器设计：
    - 使用 LLM 进行意图识别和任务规划
    - 通过 AgentRegistry 发现和调用子Agent
    - 执行 Single Response Principle
    - 支持 Human-in-the-Loop

    使用方式：
        orchestrator = OrchestratorAgent()
        async for token in orchestrator.process_stream(user_input, thread_id):
            yield token
    """

    def __init__(
        self,
        registry: AgentRegistry = None,
        llm: BaseChatModel = None,
    ):
        # Agent 注册中心
        self.registry = registry or agent_registry

        # LLM 模型
        self.llm = llm or create_chat_model(model_type="main", temperature=0)

        # 初始化各组件
        self.intent_classifier = IntentClassifier(self.llm)
        self.task_planner = TaskPlanner(self.llm, self.registry)
        self.response_synthesizer = ResponseSynthesizer(self.llm)

        # A2A 基础设施
        self.message_bus: MessageBus = message_bus
        self.context_manager: ContextManager = context_manager

        self.logger = logging.getLogger("orchestrator.agent")

    def _validate_agent_exists(self, agent_id: str) -> bool:
        """验证 Agent 是否已注册"""
        agent = self.registry.get_agent(agent_id)
        if agent is None:
            logger.warning(
                f"Agent '{agent_id}' 未注册，将跳过"
            )
            return False
        return True

    async def process(
        self,
        user_input: str,
        thread_id: str = "default",
        user_id: str = "default_user",
        debug_mode: bool = False,
    ) -> str:
        """
        处理用户输入（同步版本）

        Args:
            user_input: 用户输入文本
            thread_id: 会话线程ID
            user_id: 用户ID
            debug_mode: 是否输出调试信息

        Returns:
            编排器合成的最终响应
        """
        result_parts = []
        async for token in self.process_stream(
            user_input, thread_id, user_id, debug_mode,
        ):
            result_parts.append(token)
        return "".join(result_parts)

    async def process_stream(
        self,
        user_input: str,
        thread_id: str = "default",
        user_id: str = "default_user",
        debug_mode: bool = False,
    ) -> AsyncGenerator[str, None]:
        """
        处理用户输入（流式版本）

        编排流程：
        1. 分类意图
        2. 制定计划
        3. 委派子Agent
        4. 合成响应

        Args:
            user_input: 用户输入文本
            thread_id: 会话线程ID
            user_id: 用户ID
            debug_mode: 是否输出调试信息

        Yields:
            流式响应文本（包括进度提示和最终回复）
        """
        import uuid
        trace_id = str(uuid.uuid4())

        # ==========================================
        # Step 1: 创建共享上下文
        # ==========================================
        yield "[ORCHESTRATOR] 🔍 正在分析您的需求...\n"

        ctx = self.context_manager.create_context(
            trace_id=trace_id,
            user_input=user_input,
            user_id=user_id,
            session_id=thread_id,
        )

        # ==========================================
        # Step 2: 意图分类
        # ==========================================
        agent_descriptions = self.registry.get_routing_descriptions()
        intent_result = await self.intent_classifier.classify(
            user_input, agent_descriptions,
        )

        ctx.detected_intent = intent_result.category
        ctx.urgency = intent_result.urgency
        self.context_manager.update_context(
            trace_id,
            detected_intent=intent_result.category,
            urgency=intent_result.urgency,
        )

        if debug_mode:
            yield (
                f"[DEBUG] 意图: {intent_result.category} | "
                f"紧急度: {intent_result.urgency} | "
                f"置信度: {intent_result.confidence:.0%}\n"
            )

        # 无法识别的意图 → 兜底
        if intent_result.category == "other" and intent_result.confidence < 0.4:
            yield "[ORCHESTRATOR] 无法确定您的需求类型。\n"
            yield self._get_fallback_response(user_input)
            self.context_manager.clear_context(trace_id)
            return

        # ==========================================
        # Step 3: 制定计划
        # ==========================================
        yield "[ORCHESTRATOR] 📋 正在规划处理方案...\n"

        plan = await self.task_planner.plan(intent_result, user_input)

        if not plan.steps:
            yield "[ORCHESTRATOR] 未找到合适的处理方案。\n"
            yield self._get_fallback_response(user_input)
            self.context_manager.clear_context(trace_id)
            return

        if debug_mode:
            yield f"[DEBUG] 规划步骤: {len(plan.steps)} 步\n"
            for i, step in enumerate(plan.steps):
                yield f"[DEBUG]   {i+1}. {step.agent_id}: {step.task}\n"

        # ==========================================
        # Step 4: 委派子Agent执行
        # ==========================================
        agent_results: dict[str, AgentMessage] = {}

        for i, step in enumerate(plan.steps):
            agent_id = step.agent_id

            if not self._validate_agent_exists(agent_id):
                continue

            yield f"[ORCHESTRATOR] 🤖 调用 {agent_id}...\n"

            # 创建委派消息
            delegation = AgentMessage.create_delegation(
                from_agent="orchestrator",
                to_agent=agent_id,
                payload={
                    "user_input": user_input,
                    "task": step.task,
                    "intent_category": intent_result.category,
                    "urgency": intent_result.urgency,
                    "params": step.params,
                },
                context=ctx.to_dict(),
                trace_id=trace_id,
            )

            self.message_bus.record(delegation)

            # 调用子Agent
            agent_instance = self.registry.get_agent(agent_id)
            if agent_instance is None:
                logger.error(f"Agent '{agent_id}' 实例获取失败")
                continue

            try:
                # 流式输出子Agent的进度信息
                async for progress in agent_instance.execute_stream(delegation):
                    yield progress + "\n"

                # 获取最终结果
                result = await agent_instance.execute(delegation)
                self.message_bus.record(result)
                agent_results[agent_id] = result

                # 合并结果到上下文
                self.context_manager.merge_agent_result(
                    trace_id, agent_id, result.payload,
                )

            except Exception as e:
                logger.error(f"Agent '{agent_id}' 执行失败: {e}")
                error_msg = AgentMessage.create_response(
                    from_agent=agent_id,
                    to_agent="orchestrator",
                    payload={"error": str(e)},
                    original_message=delegation,
                    success=False,
                    error=str(e),
                )
                self.message_bus.record(error_msg)
                agent_results[agent_id] = error_msg

        # ==========================================
        # Step 5: 合成响应
        # ==========================================
        yield "[ORCHESTRATOR] 📝 正在整理回复...\n\n"

        if debug_mode:
            debug_view = self.response_synthesizer.synthesize_debug(
                agent_results,
                plan.steps,
                trace_id,
            )
            yield debug_view + "\n\n"

        response = await self.response_synthesizer.synthesize(
            agent_results, user_input, debug_mode,
        )
        yield response

        # 清理上下文
        self.context_manager.clear_context(trace_id)

        # 日志摘要
        if debug_mode:
            summary = self.message_bus.trace_summary(trace_id)
            yield f"\n\n[DEBUG] 调用摘要: {summary.get('call_chain', 'none')}"

    def _get_fallback_response(self, user_input: str) -> str:
        """获取兜底响应"""
        return (
            "抱歉，我暂时无法处理您的请求。\n\n"
            "我可以帮您：\n"
            "- 🔧 **IT技术支持**：排查故障、软件使用指南\n"
            "- 📋 **工单提交**：提交和处理IT工单\n"
            "- 📊 **效能分析**：查看工单处理效率\n"
            "- 💼 **HR咨询**：请假、福利、入职政策\n\n"
            "请描述您的具体需求，我会尽力为您服务。"
        )

    def get_call_summary(self, trace_id: str) -> dict:
        """获取调用链摘要"""
        return self.message_bus.trace_summary(trace_id)


# 全局编排器实例
orchestrator = OrchestratorAgent()

"""
IT咨询子Agent — RAG知识库自服务

参考 Microsoft Copilot Studio 的专业Agent模式：
- 知识域: IT故障排查、软件使用指南、系统配置
- 能力: RAG检索、故障排查步骤生成、FAQ匹配
- 与其他Agent知识域不重叠

基于现有 ConsultantAgent 改造，复用 consultants/ 组件的核心逻辑。

YOU ARE A SUB-AGENT. DO NOT REPLY TO USER DIRECTLY.
MUST return structured findings to the Orchestrator.
"""

from __future__ import annotations

import logging
from typing import AsyncGenerator

from agents.base_sub_agent import BaseSubAgent
from agents.a2a.protocol import AgentMessage, MessageIntent
from agents.orchestrator.agent_declaration import agent_declaration
from agents.orchestrator.agent_registry import agent_registry

from agents.consultant import (
    KnowledgeRetriever,
    ConsultationClassifier,
    ResponseGenerator,
    ConsultationProcessor,
)
from config.model_provider import create_chat_model

logger = logging.getLogger("agent.it_consultant")


@agent_declaration(
    agent_id="it_consultant",
    name="IT咨询Agent",
    description=(
        "负责IT故障排查、软件使用指南、系统配置等知识自服务。"
        "当用户询问'如何/怎么'类IT问题时调用此Agent。"
        "通过RAG检索知识库生成排查步骤。"
        "如果知识库无法解决，返回升级建议由编排器决定是否创建工单。"
    ),
    capabilities=[
        "rag_search",
        "it_troubleshooting",
        "software_guide",
        "faq_matching",
        "escalation_recommendation",
    ],
    knowledge_domains=[
        "it_support",
        "software_guide",
        "system_configuration",
        "network_troubleshooting",
    ],
    priority=1,
)
class ITConsultantSubAgent(BaseSubAgent):
    """
    IT咨询子Agent

    职责：
    1. 从FAISS知识库检索相关内容
    2. 使用LLM生成自然语言排查步骤
    3. 如果知识库无法覆盖，建议升级为工单
    4. 返回结构化结果给编排器（不直接回复用户）

    复用现有组件: KnowledgeRetriever, ConsultationClassifier,
                  ResponseGenerator, ConsultationProcessor
    """

    agent_id = "it_consultant"

    def __init__(self):
        super().__init__()

        # 复用现有的咨询组件
        self.llm = create_chat_model(temperature=0.3)
        self.knowledge_retriever = KnowledgeRetriever()
        self.consultation_classifier = ConsultationClassifier(self.llm)
        self.response_generator = ResponseGenerator(self.llm)
        self.consultation_processor = ConsultationProcessor(
            self.knowledge_retriever,
            self.consultation_classifier,
            self.response_generator,
        )

        self._initialized = False

    async def _ensure_initialized(self):
        """延迟初始化知识库（避免在注册时触发重IO操作）"""
        if not self._initialized:
            await self.knowledge_retriever.initialize()
            self._initialized = True
            self.logger.info("IT咨询Agent知识库已初始化")

    async def execute(self, message: AgentMessage) -> AgentMessage:
        """
        执行IT咨询任务

        编排器委派的消息格式：
            payload.user_input: 用户原始输入
            payload.task: 任务描述
            payload.intent_category: 意图类别

        返回格式：
            payload.direct_response: 直接可展示的回答（如有）
            payload.knowledge_sources: 知识库来源
            payload.confidence: 置信度
            payload.needs_escalation: 是否需要升级为工单
        """
        await self._ensure_initialized()

        user_input = message.payload.get("user_input", "")
        task = message.payload.get("task", "")

        self.logger.info(
            f"[IT Agent] 处理咨询请求 (trace={message.trace_id[:8]}...): {task}"
        )

        try:
            # 使用现有的 ConsultationProcessor 处理
            # consult_stream 返回流式回答，我们收集完整结果
            response_parts = []
            async for token in self.consultation_processor.process_consultation_stream(
                user_input, message.context.get("session_id", "default"),
            ):
                response_parts.append(token)

            full_response = "".join(response_parts)

            # 判断是否需要升级
            needs_escalation = self._check_escalation_needed(full_response)

            return AgentMessage.create_response(
                from_agent=self.agent_id,
                to_agent=message.from_agent,
                payload={
                    "direct_response": full_response,
                    "summary": f"IT知识库检索完成，回答长度: {len(full_response)}字",
                    "confidence": 0.85,
                    "needs_escalation": needs_escalation,
                    "knowledge_sources": ["IT知识库 (FAISS)"],
                },
                original_message=message,
                success=True,
            )

        except Exception as e:
            self.logger.error(f"IT咨询处理失败: {e}")
            return self.create_error_response(message, str(e))

    def _check_escalation_needed(self, response: str) -> bool:
        """
        检查是否需要升级为工单

        判断标准：
        - 回答中包含"无法解决"/"需要人工"/"建议提交工单"等
        - 回答过短（<20字）可能表示知识库无结果
        """
        escalation_keywords = [
            "无法解决", "需要人工", "建议提交工单", "请联系",
            "暂时无法", "超出范围", "需要工程师",
        ]
        if any(kw in response for kw in escalation_keywords):
            return True
        if len(response) < 20:
            return True
        return False

    async def execute_stream(self, message: AgentMessage) -> AsyncGenerator[str, None]:
        """
        流式执行（向编排器报告进度）

        注意：这些进度信息是给编排器的，不是给用户的。
        编排器可以选择性地透传给用户。
        """
        await self._ensure_initialized()

        user_input = message.payload.get("user_input", "")

        yield f"[IT Agent] 正在检索知识库..."

        async for token in self.consultation_processor.process_consultation_stream(
            user_input, message.context.get("session_id", "default"),
        ):
            # 不暴露具体内容，只报告进度
            pass

        yield f"[IT Agent] 检索完成，生成回答中..."
        yield f"[IT Agent] 结果已返回给编排器"


# 自动注册到全局注册中心
def _register():
    """模块加载时自动注册到全局注册中心"""
    agent_registry.register(
        ITConsultantSubAgent.__agent_declaration__,
        ITConsultantSubAgent,
    )

_register()

"""
EnterpriseRAGAgent — 统一企业知识库 RAG Agent

替代旧的 it_consultant / hr_consultant / facilities 三个"伪 Agent"。

核心改进：
1. 不再用关键词匹配区分 IT/HR/行政 — FAISS 向量相似度自动跨领域检索
2. 一次检索返回 top_k 篇最相关文档（可能横跨 IT + HR + 行政）
3. LLM 根据检索到的多领域文档直接合成回答
4. 解决了"病假属于 HR 还是 IT？"这类跨领域问题

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
from services.knowledge_service import KnowledgeService

logger = logging.getLogger("agent.enterprise_rag")


@agent_declaration(
    agent_id="enterprise_rag",
    name="企业知识库RAG Agent",
    description=(
        "统一企业知识库问答Agent。覆盖IT故障排查、HR政策咨询、行政服务指引等所有领域。"
        "通过FAISS向量检索自动匹配最相关文档，LLM跨领域合成回答。"
        "适用于所有知识查询类问题（80%+的请求）。"
    ),
    capabilities=[
        "rag_search", "cross_domain_qa", "it_troubleshooting",
        "hr_policy_lookup", "facilities_guide", "faq_matching",
    ],
    knowledge_domains=[
        "it_support", "network_troubleshooting", "system_configuration",
        "hr_policy", "leave_management", "employee_benefits",
        "meeting_rooms", "facility_maintenance", "office_location",
    ],
    priority=1,
)
class EnterpriseRAGAgent(BaseSubAgent):
    """
    统一企业 RAG Agent

    工作流程：
    1. FAISS 向量检索（跨所有领域，不设 category 过滤）
    2. LLM 根据检索到的文档合成回答
    3. 返回结构化结果给编排器

    与旧版的区别：
    - 旧: Router → 关键词匹配 agent_id → 对应 Agent 的内嵌知识库
    - 新: Router → EnterpriseRAGAgent → FAISS → LLM 合成
    """

    agent_id = "enterprise_rag"

    def __init__(self):
        super().__init__()
        self.llm = create_chat_model(temperature=0)
        self.knowledge_service = KnowledgeService()
        self._initialized = False

    async def _ensure_initialized(self):
        if not self._initialized:
            await self.knowledge_service.initialize()
            self._initialized = True
            doc_count = self.knowledge_service.get_documents_count()
            self.logger.info(f"EnterpriseRAGAgent 已初始化，知识库共 {doc_count} 篇文档")

    async def execute(self, message: AgentMessage) -> AgentMessage:
        """
        执行统一 RAG 问答

        编排器委派的消息格式：
            payload.user_input: 用户原始输入
            payload.conversation_history: 对话历史（可选）
            payload.task: 任务描述

        返回格式：
            payload.direct_response: 直接可展示的回答
            payload.sources: 引用来源
            payload.confidence: 置信度
            payload.needs_escalation: 是否需要升级
        """
        await self._ensure_initialized()

        user_input = message.payload.get("user_input", "")
        conversation_history = message.payload.get("conversation_history", "")
        task = message.payload.get("task", "")

        self.logger.info(
            f"[EnterpriseRAG] 处理咨询 (trace={message.trace_id[:8]}...): "
            f"query=\"{user_input[:50]}\""
        )

        try:
            # 1. FAISS 向量检索 — 跨所有领域，不设 category 过滤
            docs = await self.knowledge_service.search(user_input, top_k=5)

            if not docs:
                return AgentMessage.create_response(
                    from_agent=self.agent_id,
                    to_agent=message.from_agent,
                    payload={
                        "direct_response": (
                            "抱歉，我在知识库中没有找到相关的信息。\n\n"
                            "建议：\n"
                            "- 尝试用不同的关键词描述您的问题\n"
                            "- 提交工单让工程师协助处理\n"
                            "- 拨打服务台热线获取人工支持"
                        ),
                        "sources": [],
                        "confidence": 0.0,
                        "needs_escalation": True,
                        "summary": "知识库未检索到相关文档",
                    },
                    original_message=message,
                    success=True,
                )

            # 2. LLM 根据检索到的所有文档合成回答
            response = await self._synthesize(user_input, docs, conversation_history)

            # 3. 判断是否需要升级
            needs_escalation = self._check_escalation_needed(response, docs)

            return AgentMessage.create_response(
                from_agent=self.agent_id,
                to_agent=message.from_agent,
                payload={
                    "direct_response": response,
                    "sources": [
                        {"category": d.get("category", ""), "score": d.get("score", 0)}
                        for d in docs
                    ],
                    "confidence": docs[0].get("score", 0) if docs else 0,
                    "needs_escalation": needs_escalation,
                    "summary": f"检索到 {len(docs)} 篇文档，回答 {len(response)} 字",
                },
                original_message=message,
                success=True,
            )

        except Exception as e:
            self.logger.error(f"EnterpriseRAG 处理失败: {e}")
            return self.create_error_response(message, str(e))

    async def _synthesize(
        self,
        user_input: str,
        docs: list[dict],
        conversation_history: str = "",
    ) -> str:
        """LLM 合成回答 — 基于跨领域检索到的文档"""
        # 构建文档上下文（标注分类来源）
        doc_context_parts = []
        for i, doc in enumerate(docs):
            category = doc.get("category", "通用")
            content = doc.get("content", "")
            score = doc.get("score", 0)
            doc_context_parts.append(
                f"[文档{i+1}] (分类:{category}, 相关度:{score:.2f})\n{content}"
            )
        doc_context = "\n\n".join(doc_context_parts)

        history_section = ""
        if conversation_history:
            history_section = (
                f"## 对话历史\n{conversation_history}\n\n"
                f"注意：如果用户当前是追问，请结合历史上下文理解。\n\n"
            )

        prompt = (
            f"你是一个企业员工服务台的AI助手。请根据以下知识库文档回答用户问题。\n\n"
            f"{history_section}"
            f"## 检索到的相关文档（共{len(docs)}篇，可能跨多个领域）\n"
            f"{doc_context}\n\n"
            f"## 用户问题\n{user_input}\n\n"
            f"回答要求：\n"
            f"1. 基于文档内容回答，不要编造\n"
            f"2. 如果文档只部分覆盖了问题，诚实告知\n"
            f"3. 如果涉及多个领域（如IT+HR），自然整合\n"
            f"4. 简洁清晰，控制在300字以内\n"
            f"5. 如有必要，引导用户到正确的操作渠道（OA、飞书、工单系统等）"
        )

        try:
            response = await self.llm.ainvoke([{"role": "user", "content": prompt}])
            return response.content.strip()
        except Exception as e:
            self.logger.error(f"LLM 合成失败: {e}")
            # 兜底：直接返回检索到的文档内容
            top_doc = docs[0] if docs else {"content": "未找到相关信息"}
            return f"根据知识库检索结果：\n\n{top_doc.get('content', '')[:500]}"

    def _check_escalation_needed(self, response: str, docs: list[dict]) -> bool:
        """检查是否需要升级为工单"""
        # 所有文档相关度都很低 → 可能需要升级
        if docs and all(d.get("score", 0) < 0.3 for d in docs):
            return True
        # 回答中包含升级信号词
        escalation_signals = [
            "无法解决", "需要人工", "建议提交工单", "请联系",
            "暂时无法", "超出范围", "需要工程师",
        ]
        if any(sig in response for sig in escalation_signals):
            return True
        return False

    async def execute_stream(self, message: AgentMessage) -> AsyncGenerator[str, None]:
        """流式执行（向编排器报告进度）"""
        yield "[EnterpriseRAG] 正在向量检索知识库..."
        yield "[EnterpriseRAG] 检索到相关文档，生成回答中..."
        yield "[EnterpriseRAG] 回答已生成，返回给编排器"


# 自动注册
def _register():
    agent_registry.register(
        EnterpriseRAGAgent.__agent_declaration__,
        EnterpriseRAGAgent,
    )

_register()

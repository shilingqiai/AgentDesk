"""
EnterpriseRAGAgent 测试

测试覆盖：
- 正常 RAG 问答流程
- 空检索结果降级
- LLM 合成失败兜底
- 自动升级判断
- v3.2: 用户升级信号检测 + 短路
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch


class TestEnterpriseRAG:
    """EnterpriseRAGAgent 业务逻辑测试"""

    @pytest.mark.asyncio
    async def test_execute_with_docs(self):
        """正常流程：检索到文档 → LLM 合成回答"""
        from agents.sub_agents.enterprise_rag import EnterpriseRAGAgent
        from agents.a2a.protocol import AgentMessage

        agent = EnterpriseRAGAgent()

        # Mock knowledge_service
        mock_ks = AsyncMock()
        mock_ks.initialize = AsyncMock()
        mock_ks.search = AsyncMock(return_value=[
            {"id": 1, "content": "VPN排查步骤", "category": "网络故障",
             "score": 0.92, "keywords": ["VPN"]},
        ])
        mock_ks.get_documents_count = MagicMock(return_value=20)
        agent.knowledge_service = mock_ks
        agent._initialized = True

        # Mock LLM
        from tests.conftest import create_mock_llm
        agent.llm = create_mock_llm(["VPN排查步骤：1.检查网络 2.检查客户端..."])

        msg = AgentMessage.create_delegation(
            from_agent="orchestrator",
            to_agent="enterprise_rag",
            payload={"user_input": "VPN怎么连", "task": "知识库问答"},
            trace_id="test-trace",
        )

        result = await agent.execute(msg)

        assert result.success is True
        assert result.payload is not None
        assert len(result.payload.get("sources", [])) == 1
        assert result.payload["sources"][0]["score"] == 0.92
        assert result.payload["needs_escalation"] is False

    @pytest.mark.asyncio
    async def test_execute_no_docs(self):
        """知识库无相关文档 → 返回升级建议"""
        from agents.sub_agents.enterprise_rag import EnterpriseRAGAgent
        from agents.a2a.protocol import AgentMessage

        agent = EnterpriseRAGAgent()

        mock_ks = AsyncMock()
        mock_ks.initialize = AsyncMock()
        mock_ks.search = AsyncMock(return_value=[])
        mock_ks.get_documents_count = MagicMock(return_value=10)
        agent.knowledge_service = mock_ks
        agent._initialized = True

        msg = AgentMessage.create_delegation(
            from_agent="orchestrator",
            to_agent="enterprise_rag",
            payload={"user_input": "不存在的问题XYZ", "task": "问答"},
            trace_id="test-trace-2",
        )

        result = await agent.execute(msg)

        assert result.success is True
        assert result.payload["confidence"] == 0.0
        assert result.payload["needs_escalation"] is True
        assert "知识库中没有找到" in result.payload.get("direct_response", "")

    @pytest.mark.asyncio
    async def test_execute_llm_failure_fallback(self):
        """LLM 合成失败 → 返回文档内容兜底"""
        from agents.sub_agents.enterprise_rag import EnterpriseRAGAgent
        from agents.a2a.protocol import AgentMessage

        agent = EnterpriseRAGAgent()

        mock_ks = AsyncMock()
        mock_ks.initialize = AsyncMock()
        mock_ks.search = AsyncMock(return_value=[
            {"id": 1, "content": "VPN排查步骤内容", "category": "网络故障",
             "score": 0.95, "keywords": ["VPN"]},
        ])
        mock_ks.get_documents_count = MagicMock(return_value=10)
        agent.knowledge_service = mock_ks
        agent._initialized = True

        # LLM 调用失败
        from tests.conftest import create_mock_llm
        mock_llm = create_mock_llm()
        mock_llm.ainvoke = AsyncMock(side_effect=Exception("LLM API 超时"))
        agent.llm = mock_llm

        msg = AgentMessage.create_delegation(
            from_agent="orchestrator",
            to_agent="enterprise_rag",
            payload={"user_input": "VPN排查", "task": "问答"},
            trace_id="test-trace-3",
        )

        result = await agent.execute(msg)

        assert result.success is True
        # 应返回文档内容兜底
        assert "VPN排查步骤内容" in result.payload.get("direct_response", "")

    # ── v3.2: 升级信号检测 ──

    @pytest.mark.asyncio
    async def test_execute_escalation_signal_short_circuit(self):
        """用户说'还是不行' + 有对话历史 → 短路返回工单建议"""
        from agents.sub_agents.enterprise_rag import EnterpriseRAGAgent
        from agents.a2a.protocol import AgentMessage

        agent = EnterpriseRAGAgent()
        agent._initialized = True

        msg = AgentMessage.create_delegation(
            from_agent="orchestrator",
            to_agent="enterprise_rag",
            payload={
                "user_input": "还是不行，按照你说的步骤都试了",
                "task": "知识库问答",
                "conversation_history": (
                    "用户: VPN怎么排查\n"
                    "助手: 请按以下步骤排查：1.检查网络 2.检查客户端版本..."
                ),
            },
            trace_id="test-trace-esc",
        )

        # knowledge_service 不应被调用（短路）
        mock_ks = AsyncMock()
        mock_ks.initialize = AsyncMock()
        mock_ks.search = AsyncMock()
        agent.knowledge_service = mock_ks

        result = await agent.execute(msg)

        assert result.success is True
        assert result.payload["needs_escalation"] is True
        assert "工单" in result.payload.get("direct_response", "")
        # 确认没有调用 FAISS 检索
        mock_ks.search.assert_not_called()

    @pytest.mark.asyncio
    async def test_execute_no_escalation_without_history(self):
        """首次输入'VPN还是连不上' → 不应短路（正常搜索）"""
        from agents.sub_agents.enterprise_rag import EnterpriseRAGAgent
        from agents.a2a.protocol import AgentMessage

        agent = EnterpriseRAGAgent()
        agent._initialized = True

        mock_ks = AsyncMock()
        mock_ks.initialize = AsyncMock()
        mock_ks.search = AsyncMock(return_value=[
            {"id": 1, "content": "VPN连接问题排查", "category": "网络故障",
             "score": 0.85, "keywords": ["VPN"]},
        ])
        mock_ks.get_documents_count = MagicMock(return_value=10)
        agent.knowledge_service = mock_ks

        from tests.conftest import create_mock_llm
        agent.llm = create_mock_llm(["请检查VPN客户端版本和网络设置..."])

        msg = AgentMessage.create_delegation(
            from_agent="orchestrator",
            to_agent="enterprise_rag",
            payload={
                "user_input": "VPN还是连不上",
                "task": "知识库问答",
                "conversation_history": "",  # 无历史
            },
            trace_id="test-trace-no-esc",
        )

        result = await agent.execute(msg)
        # 不应短路 — 走正常 RAG 流程
        assert result.success is True

    def test_check_user_escalation_signal_positive(self):
        """用户说'还是不行' + 有排障历史 → True"""
        from agents.sub_agents.enterprise_rag import EnterpriseRAGAgent

        agent = EnterpriseRAGAgent()
        result = agent._check_user_escalation_signal(
            user_input="按照你说的做了，还是不行",
            conversation_history="助手: 检查VPN客户端版本...用户: 检查了",
        )
        assert result is True

    def test_check_user_escalation_signal_no_history(self):
        """用户说'还是不行' 但没有对话历史 → False"""
        from agents.sub_agents.enterprise_rag import EnterpriseRAGAgent

        agent = EnterpriseRAGAgent()
        result = agent._check_user_escalation_signal(
            user_input="还是不行",
            conversation_history="",
        )
        assert result is False

    def test_check_user_escalation_signal_no_signal_word(self):
        """用户正常提问 + 有历史 → False"""
        from agents.sub_agents.enterprise_rag import EnterpriseRAGAgent

        agent = EnterpriseRAGAgent()
        result = agent._check_user_escalation_signal(
            user_input="VPN怎么配置",
            conversation_history="助手: 请检查网络连接...",
        )
        assert result is False

    @pytest.mark.asyncio
    async def test_check_escalation_low_scores(self):
        """所有文档相关度低 → 触发升级"""
        from agents.sub_agents.enterprise_rag import EnterpriseRAGAgent

        agent = EnterpriseRAGAgent()

        docs = [
            {"score": 0.15, "content": "不太相关", "category": "其他"},
            {"score": 0.20, "content": "也不太相关", "category": "其他"},
        ]
        needs = agent._check_escalation_needed("回答内容", docs)
        assert needs is True

    @pytest.mark.asyncio
    async def test_check_escalation_signals(self):
        """回答含升级信号词 → 触发升级"""
        from agents.sub_agents.enterprise_rag import EnterpriseRAGAgent

        agent = EnterpriseRAGAgent()

        docs = [{"score": 0.8, "content": "相关内容", "category": "测试"}]

        # 含"建议提交工单"信号
        response = "这个问题比较复杂，建议提交工单让工程师处理"
        assert agent._check_escalation_needed(response, docs) is True

    def test_build_doc_context(self):
        """构建文档上下文文本"""
        from agents.sub_agents.enterprise_rag import EnterpriseRAGAgent

        agent = EnterpriseRAGAgent()
        docs = [
            {"content": "文档内容1", "category": "IT", "score": 0.9},
            {"content": "文档内容2", "category": "HR", "score": 0.7},
        ]

        context = agent._build_doc_context(docs)
        assert "[文档1]" in context
        assert "[文档2]" in context
        assert "IT" in context
        assert "HR" in context

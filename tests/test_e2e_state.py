"""
E2E 测试 — 维度 1: State 过渡断言

验证 LangGraph 节点执行后 TicketState 各字段变化符合预期：
- 路由后 track/agent_id/confidence/intent 设置正确
- 卡片锁短路逻辑
- 对话阶段追踪更新
- resolved 从 False→True
- re_evaluate escalation 转向 dynamic
"""

import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

from agents.graph.state import create_initial_state, TicketState


class TestRouteNodeState:
    """route_node 执行后的 State 验证"""

    def test_sets_track_confidence_and_intent(self):
        """mock Router LLM → 断言 track='fast', confidence=0.95, agent_id='enterprise_rag'"""
        from agents.graph.nodes.router import route_node
        from agents.orchestrator.router import RouteResult

        state = create_initial_state("VPN怎么排查", thread_id="test-route-001")

        mock_result = RouteResult(
            track="fast", agent_id="enterprise_rag", reason="VPN排查",
            confidence=0.95,
        )

        # Router.route() 是实例方法 — 在类上 patch 它
        with patch(
            "agents.orchestrator.router.Router.route",
            new_callable=AsyncMock,
            return_value=mock_result,
        ):
            result = asyncio.run(route_node(state))

        assert result["track"] == "fast"
        assert result["agent_id"] == "enterprise_rag"
        assert result["confidence"] == 0.95
        assert result["intent"] == "knowledge_query"
        assert result["resolved"] is False
        assert result["urgency"] in ("low", "medium", "high")

    def test_card_lock_short_circuits(self):
        """pending_card_type 存在 → track='card_lock', 不走 Router"""
        from agents.graph.nodes.router import route_node

        state = create_initial_state("确认预定", thread_id="test-card-001")
        state["pending_card_type"] = "admin"

        result = asyncio.run(route_node(state))

        assert result["track"] == "card_lock"
        assert result["agent_id"] == "ticket_dispatch"
        assert result["confidence"] == 1.0
        assert result["intent"] == "admin"
        assert result["resolved"] is False

    def test_self_help_phase_re_evaluates(self):
        """self_help phase='self_help_provided' → track='re_evaluate'"""
        from agents.graph.nodes.router import route_node

        state = create_initial_state("还是不行", thread_id="test-re-001")
        from agents.graph.state import _sh_set
        _sh_set(state, phase="self_help_provided", topic="VPN故障排查", summary="检查网络连通性...")

        result = asyncio.run(route_node(state))

        assert result["track"] == "re_evaluate"
        assert result["resolved"] is False

    def test_dynamic_card_lock_short_circuits(self):
        """pending_card_type 以 'dynamic_' 开头 → track='dynamic', agent_id='dynamic_action'"""
        from agents.graph.nodes.router import route_node

        state = create_initial_state("确认", thread_id="test-dyn-001")
        state["pending_card_type"] = "dynamic_interrupt"

        result = asyncio.run(route_node(state))

        assert result["track"] == "dynamic"
        assert result["agent_id"] == "dynamic_action"
        assert result["confidence"] == 1.0
        assert result["intent"] == "dynamic_resume"


class TestNodeStateTransitions:
    """各轨道节点执行后的 State 变化断言"""

    def test_resolved_false_to_true(self):
        """respond_node → resolved 从 False 变为 True"""
        from agents.graph.nodes.terminal import respond_node

        state = create_initial_state("测试", thread_id="test-respond")
        state["final_response"] = "处理完成。"
        assert state["resolved"] is False

        result = asyncio.run(respond_node(state))
        assert result["resolved"] is True
        assert len(result["messages"]) == 2  # 原始 + AI 回复

    def test_clarification_sets_resolved_and_response(self):
        """clarification_node → resolved=True + final_response 非空"""
        from agents.graph.nodes.terminal import clarification_node

        state = create_initial_state("嗯？", thread_id="test-clarify")
        state["track"] = "clarify"
        state["confidence"] = 0.1

        result = asyncio.run(clarification_node(state))
        assert result["resolved"] is True
        assert len(result["final_response"]) > 20
        assert "不太确定" in result["final_response"] or "抱歉" in result["final_response"]

    def test_respond_node_adds_human_review_warning(self):
        """needs_human_review=True + track=dynamic → final_response 含人工审核警告"""
        from agents.graph.nodes.terminal import respond_node

        state = create_initial_state("创建工单", thread_id="test-review")
        state["final_response"] = "工单已创建"
        state["needs_human_review"] = True
        state["track"] = "dynamic"

        result = asyncio.run(respond_node(state))
        assert "人工审核" in result["final_response"]
        assert result["resolved"] is True

    @pytest.mark.asyncio
    async def test_fast_track_sets_conversation_phase(self):
        """fast_track_node 成功后 → self_help.phase='self_help_provided' + self_help.track='fast'"""
        from agents.graph.nodes.fast import fast_track_node
        from agents.orchestrator.agent_registry import agent_registry
        from agents.graph.state import _sh_get

        rag_agent = agent_registry.get_agent("enterprise_rag")
        if rag_agent is None:
            pytest.skip("EnterpriseRAG agent 未注册")

        orig_ks = rag_agent.knowledge_service
        orig_initialized = rag_agent._initialized

        mock_ks = AsyncMock()
        mock_ks.search = AsyncMock(return_value=[
            {"id": 1, "content": "VPN排查步骤", "category": "IT-网络",
             "score": 0.95, "keywords": ["VPN"]},
        ])
        rag_agent.knowledge_service = mock_ks
        rag_agent._initialized = True

        async def mock_synth(user_text, docs, history):
            yield "VPN排查步骤：1.检查网络 2.检查客户端"
        orig_synth = rag_agent._synthesize_stream
        rag_agent._synthesize_stream = mock_synth

        with patch(
            "agents.graph.nodes.fast.get_stream_writer",
            return_value=lambda x: None,
        ):
            state = create_initial_state("VPN怎么排查", thread_id="test-fast-phase")
            result = await fast_track_node(state)

        assert _sh_get(result, "phase") == "self_help_provided"
        assert _sh_get(result, "track") == "fast"
        assert result["resolved"] is True
        assert "VPN" in result["final_response"]

        rag_agent.knowledge_service = orig_ks
        rag_agent._initialized = orig_initialized
        rag_agent._synthesize_stream = orig_synth

    @pytest.mark.asyncio
    async def test_re_evaluate_escalation_sets_dynamic(self):
        """用户反馈无效 → escalation → track='dynamic', urgency='high'"""
        from agents.graph.nodes.router import re_evaluate_node

        state = create_initial_state("试了没用", thread_id="test-re-eval")
        from agents.graph.state import _sh_set
        _sh_set(state, phase="self_help_provided", topic="VPN故障排查",
                summary="检查网络连通性，确认VPN客户端版本")

        mock_llm = MagicMock()
        mock_response = MagicMock()
        mock_response.content = '{"intent":"escalation","reason":"用户反馈方案无效"}'
        mock_llm.ainvoke = AsyncMock(return_value=mock_response)

        with patch(
            "config.model_provider.create_chat_model",
            return_value=mock_llm,
        ):
            result = await re_evaluate_node(state)

        assert result["track"] == "dynamic"
        assert result["agent_id"] == "dynamic_action"
        assert result["urgency"] == "high"
        assert result["confidence"] == 0.9
        assert result["resolved"] is False


class TestRoutingFunctions:
    """条件边分发函数 — 纯函数测试"""

    def test_route_after_route_all_tracks(self):
        """4 轨 + 1 内部轨 → 正确目标节点名"""
        from agents.graph.routing import route_after_route

        assert route_after_route({"track": "fast"}) == "fast_track"
        assert route_after_route({"track": "dynamic"}) == "dynamic_action"
        assert route_after_route({"track": "complex"}) == "complex_track"
        assert route_after_route({"track": "card_lock"}) == "action_track"
        assert route_after_route({"track": "clarify"}) == "clarification"
        assert route_after_route({"track": "unknown"}) == "clarification"
        assert route_after_route({"track": ""}) == "clarification"

    def test_after_re_evaluate_all_intents(self):
        """re_evaluate 的4种意图 → 正确下一条边"""
        from agents.graph.routing import after_re_evaluate

        state = {
            "agent_results": {"re_evaluate": {"intent": "escalation"}},
            "self_help": {"phase": "", "topic": "", "summary": "", "track": ""},
        }
        assert after_re_evaluate(state) == "dynamic_action"

        state["agent_results"]["re_evaluate"]["intent"] = "new_topic"
        assert after_re_evaluate(state) == "route"

        state["agent_results"]["re_evaluate"]["intent"] = "confirm"
        assert after_re_evaluate(state) == "respond"

        state["agent_results"]["re_evaluate"]["intent"] = "follow_up"
        assert after_re_evaluate(state) == "fast_track"

        state["self_help"]["track"] = "dynamic"
        assert after_re_evaluate(state) == "dynamic_action"

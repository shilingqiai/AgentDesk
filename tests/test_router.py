"""
Router 路由决策测试

测试覆盖：
- 四轨道正确分发（fast/action/complex/clarify）
- 低置信度强制转为 clarify
- JSON 解析失败降级
- JSON 修复与重试

策略：mock config.model_provider.create_chat_model 返回纯 AsyncMock，
避免 ChatOpenAI Pydantic 模型的字段限制。
"""

import pytest
from unittest.mock import patch, AsyncMock, MagicMock
from agents.orchestrator.router import Router, RouterDecision, RouteResult


# ============================================================
# 辅助
# ============================================================

class MockLLMResponse:
    """模拟 LLM 返回对象"""
    def __init__(self, content: str):
        self.content = content


def create_mock_chain(responses: list[str]):
    """创建顺序返回预设响应的 mock chain"""
    call_count = [0]
    mock = AsyncMock()

    async def side_effect(*args, **kwargs):
        idx = min(call_count[0], len(responses) - 1)
        resp = responses[idx]
        call_count[0] += 1
        return MockLLMResponse(resp)

    mock.ainvoke = side_effect
    return mock


# ============================================================
# 测试
# ============================================================

class TestRouterDecision:
    """Router 结构化决策测试"""

    @pytest.mark.asyncio
    async def test_fast_track(self):
        """fast 轨道：知识查询类问题"""
        from tests.conftest import ROUTER_FAST_RESPONSE

        mock_chain = create_mock_chain([ROUTER_FAST_RESPONSE])

        with patch("agents.orchestrator.router.create_chat_model",
                   return_value=MagicMock()):
            router = Router()
            router.llm.ainvoke = mock_chain.ainvoke

            with patch.object(router, 'route_prompt') as mock_prompt:
                mock_prompt.__or__ = MagicMock(return_value=mock_chain)

                decision = await router.decide(
                    "VPN怎么排查连接失败的问题？",
                    agent_descriptions="enterprise_rag(企业知识库问答), ticket_dispatch(工单派发)",
                )

        assert decision.track == "fast"
        assert decision.confidence == 0.95
        assert "VPN" in decision.reason

    @pytest.mark.asyncio
    async def test_action_track_leave(self):
        """action 轨道：请假申请"""
        from tests.conftest import ROUTER_ACTION_LEAVE_RESPONSE

        mock_chain = create_mock_chain([ROUTER_ACTION_LEAVE_RESPONSE])

        with patch("agents.orchestrator.router.create_chat_model",
                   return_value=MagicMock()):
            router = Router()
            router.llm.ainvoke = mock_chain.ainvoke

            with patch.object(router, 'route_prompt') as mock_prompt:
                mock_prompt.__or__ = MagicMock(return_value=mock_chain)

                decision = await router.decide(
                    "我想请3天年假",
                    agent_descriptions="enterprise_rag, ticket_dispatch",
                )

        assert decision.track == "action"
        assert decision.confidence == 0.90

    @pytest.mark.asyncio
    async def test_action_track_it(self):
        """action 轨道：IT 故障工单"""
        from tests.conftest import ROUTER_ACTION_IT_RESPONSE

        mock_chain = create_mock_chain([ROUTER_ACTION_IT_RESPONSE])

        with patch("agents.orchestrator.router.create_chat_model",
                   return_value=MagicMock()):
            router = Router()
            router.llm.ainvoke = mock_chain.ainvoke

            with patch.object(router, 'route_prompt') as mock_prompt:
                mock_prompt.__or__ = MagicMock(return_value=mock_chain)

                decision = await router.decide(
                    "帮我提交一个网络故障工单",
                    agent_descriptions="enterprise_rag, ticket_dispatch",
                )

        assert decision.track == "action"
        assert decision.requires_tools == ["jira_api"]

    @pytest.mark.asyncio
    async def test_clarify_track(self):
        """clarify 轨道：输入模糊"""
        from tests.conftest import ROUTER_CLARIFY_RESPONSE

        mock_chain = create_mock_chain([ROUTER_CLARIFY_RESPONSE])

        with patch("agents.orchestrator.router.create_chat_model",
                   return_value=MagicMock()):
            router = Router()
            router.llm.ainvoke = mock_chain.ainvoke

            with patch.object(router, 'route_prompt') as mock_prompt:
                mock_prompt.__or__ = MagicMock(return_value=mock_chain)

                decision = await router.decide(
                    "嗯？",
                    agent_descriptions="enterprise_rag, ticket_dispatch",
                )

        assert decision.track == "clarify"
        assert decision.confidence == 0.20

    @pytest.mark.asyncio
    async def test_complex_track(self):
        """complex 轨道：多步骤复合指令"""
        from tests.conftest import ROUTER_COMPLEX_RESPONSE

        mock_chain = create_mock_chain([ROUTER_COMPLEX_RESPONSE])

        with patch("agents.orchestrator.router.create_chat_model",
                   return_value=MagicMock()):
            router = Router()
            router.llm.ainvoke = mock_chain.ainvoke

            with patch.object(router, 'route_prompt') as mock_prompt:
                mock_prompt.__or__ = MagicMock(return_value=mock_chain)

                decision = await router.decide(
                    "帮我查天气然后请假再取消会议室",
                    agent_descriptions="enterprise_rag, ticket_dispatch",
                )

        assert decision.track == "complex"

    @pytest.mark.asyncio
    async def test_json_parse_error_fallback(self):
        """JSON 解析失败 → 返回 clarify"""
        mock_chain = create_mock_chain(["not valid json at all {{{"])

        with patch("agents.orchestrator.router.create_chat_model",
                   return_value=MagicMock()):
            router = Router()
            router.llm.ainvoke = mock_chain.ainvoke

            with patch.object(router, 'route_prompt') as mock_prompt:
                mock_prompt.__or__ = MagicMock(return_value=mock_chain)

                decision = await router.decide("测试输入")

        assert decision.track == "clarify"
        assert decision.confidence == 0.0

    @pytest.mark.asyncio
    async def test_invalid_track_fallback(self):
        """LLM 返回未知轨道 → 强制 clarify"""
        mock_chain = create_mock_chain([
            '{"track":"unknown_track","confidence":0.8,"reason":"test","requires_tools":[]}',
        ])

        with patch("agents.orchestrator.router.create_chat_model",
                   return_value=MagicMock()):
            router = Router()
            router.llm.ainvoke = mock_chain.ainvoke

            with patch.object(router, 'route_prompt') as mock_prompt:
                mock_prompt.__or__ = MagicMock(return_value=mock_chain)

                decision = await router.decide("测试")

        assert decision.track == "clarify"

    @pytest.mark.asyncio
    async def test_route_backward_compat(self):
        """向后兼容的 route() 方法返回 RouteResult"""
        from tests.conftest import ROUTER_FAST_RESPONSE

        mock_chain = create_mock_chain([ROUTER_FAST_RESPONSE])

        with patch("agents.orchestrator.router.create_chat_model",
                   return_value=MagicMock()):
            router = Router()
            router.llm.ainvoke = mock_chain.ainvoke

            with patch.object(router, 'route_prompt') as mock_prompt:
                mock_prompt.__or__ = MagicMock(return_value=mock_chain)

                result = await router.route("VPN怎么连")

        assert isinstance(result, RouteResult)
        assert result.track == "fast"
        assert result.agent_id == "enterprise_rag"
        assert result.category == "knowledge_query"

    @pytest.mark.asyncio
    async def test_llm_call_failure(self):
        """LLM 调用异常 → 返回 clarify"""
        mock_chain = AsyncMock()
        mock_chain.ainvoke = AsyncMock(side_effect=Exception("API timeout"))

        with patch("agents.orchestrator.router.create_chat_model",
                   return_value=MagicMock()):
            router = Router()

            with patch.object(router, 'route_prompt') as mock_prompt:
                mock_prompt.__or__ = MagicMock(return_value=mock_chain)

                decision = await router.decide("test")

        assert decision.track == "clarify"
        assert decision.confidence == 0.0


class TestRouteResult:
    """RouteResult 数据类测试"""

    def test_from_decision_fast(self):
        """从 RouterDecision 创建 RouteResult（fast）"""
        decision = RouterDecision(
            track="fast", confidence=0.95, reason="VPN排查",
        )
        result = RouteResult.from_decision(decision)
        assert result.track == "fast"
        assert result.agent_id == "enterprise_rag"
        assert result.category == "knowledge_query"

    def test_from_decision_action(self):
        """从 RouterDecision 创建 RouteResult（action）"""
        decision = RouterDecision(
            track="action", confidence=0.88, reason="创建工单",
        )
        result = RouteResult.from_decision(decision)
        assert result.track == "action"
        assert result.agent_id == "ticket_dispatch"
        assert result.urgency == "low"


class TestJsonExtract:
    """JSON 提取辅助函数测试"""

    def test_extract_markdown_json(self):
        """从 markdown json 代码块提取"""
        from agents.orchestrator.router import Router
        result = Router._extract_json('```json\n{"key":"value"}\n```')
        assert result == '{"key":"value"}'

    def test_extract_plain_json(self):
        """从文本中提取 JSON"""
        from agents.orchestrator.router import Router
        result = Router._extract_json('前缀 {"a":1} 后缀')
        assert result == '{"a":1}'

    def test_extract_no_braces(self):
        """无花括号 → 原样返回"""
        from agents.orchestrator.router import Router
        result = Router._extract_json("plain text")
        assert result == "plain text"

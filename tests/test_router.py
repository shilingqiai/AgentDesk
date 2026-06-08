"""
Router 路由决策测试

测试覆盖：
- 四轨道正确分发（fast/action/complex/clarify）
- 低置信度转为 clarify 检验
- 未知轨道兜底
- LLM 异常降级

v3.1: prompt→JSON（DashScope 兼容，不用 with_structured_output）
"""

import pytest
from unittest.mock import patch, AsyncMock, MagicMock
from agents.orchestrator.router import Router, RouterDecision, RouteResult


# ============================================================
# 辅助
# ============================================================

def make_mock_llm(json_str: str):
    """创建返回指定 JSON 字符串的 mock LLM"""
    mock = MagicMock()
    mock_response = MagicMock()
    mock_response.content = json_str
    mock.ainvoke = AsyncMock(return_value=mock_response)
    return mock


# ============================================================
# 测试
# ============================================================

class TestRouterDecision:
    """Router 结构化决策测试 (v3.1 prompt→JSON)"""

    @pytest.mark.asyncio
    async def test_fast_track(self):
        """fast 轨道：知识查询类问题"""
        json_str = '{"track":"fast","confidence":0.95,"reason":"用户询问VPN排查方法","requires_tools":[]}'
        mock_llm = make_mock_llm(json_str)

        with patch("agents.orchestrator.router.create_chat_model",
                   return_value=MagicMock()):
            router = Router()
            router.llm = mock_llm

            result = await router.decide(
                "VPN怎么排查连接失败的问题？",
                agent_descriptions="enterprise_rag(企业知识库问答), ticket_dispatch(工单派发)",
            )

        assert result.track == "fast"
        assert result.confidence == 0.95
        assert "VPN" in result.reason

    @pytest.mark.asyncio
    async def test_action_track_leave(self):
        """action 轨道：请假申请"""
        json_str = '{"track":"action","confidence":0.90,"reason":"用户申请请假","requires_tools":[]}'
        mock_llm = make_mock_llm(json_str)

        with patch("agents.orchestrator.router.create_chat_model",
                   return_value=MagicMock()):
            router = Router()
            router.llm = mock_llm

            result = await router.decide(
                "我想请3天年假",
                agent_descriptions="enterprise_rag, ticket_dispatch",
            )

        assert result.track == "action"
        assert result.confidence == 0.90

    @pytest.mark.asyncio
    async def test_action_track_it(self):
        """action 轨道：IT 故障工单"""
        json_str = '{"track":"action","confidence":0.88,"reason":"用户提交IT故障工单","requires_tools":["jira_api"]}'
        mock_llm = make_mock_llm(json_str)

        with patch("agents.orchestrator.router.create_chat_model",
                   return_value=MagicMock()):
            router = Router()
            router.llm = mock_llm

            result = await router.decide(
                "帮我提交一个网络故障工单",
                agent_descriptions="enterprise_rag, ticket_dispatch",
            )

        assert result.track == "action"
        assert result.requires_tools == ["jira_api"]

    @pytest.mark.asyncio
    async def test_clarify_track(self):
        """clarify 轨道：输入模糊"""
        json_str = '{"track":"clarify","confidence":0.20,"reason":"输入过于模糊无法判断意图","requires_tools":[]}'
        mock_llm = make_mock_llm(json_str)

        with patch("agents.orchestrator.router.create_chat_model",
                   return_value=MagicMock()):
            router = Router()
            router.llm = mock_llm

            result = await router.decide(
                "嗯？",
                agent_descriptions="enterprise_rag, ticket_dispatch",
            )

        assert result.track == "clarify"
        assert result.confidence == 0.20

    @pytest.mark.asyncio
    async def test_complex_track(self):
        """complex 轨道：多步骤复合指令"""
        json_str = '{"track":"complex","confidence":0.75,"reason":"多步骤复合指令","requires_tools":[]}'
        mock_llm = make_mock_llm(json_str)

        with patch("agents.orchestrator.router.create_chat_model",
                   return_value=MagicMock()):
            router = Router()
            router.llm = mock_llm

            result = await router.decide(
                "帮我查天气然后请假再取消会议室",
                agent_descriptions="enterprise_rag, ticket_dispatch",
            )

        assert result.track == "complex"

    @pytest.mark.asyncio
    async def test_json_in_markdown_block(self):
        """LLM 返回 markdown 包裹的 JSON → _extract_json 提取"""
        json_str = '```json\n{"track":"fast","confidence":0.85,"reason":"查询VPN","requires_tools":[]}\n```'
        mock_llm = make_mock_llm(json_str)

        with patch("agents.orchestrator.router.create_chat_model",
                   return_value=MagicMock()):
            router = Router()
            router.llm = mock_llm

            result = await router.decide("VPN怎么连")

        assert result.track == "fast"
        assert result.confidence == 0.85

    @pytest.mark.asyncio
    async def test_invalid_track_fallback(self):
        """LLM 返回未知轨道 → 强制 clarify"""
        json_str = '{"track":"unknown_track","confidence":0.8,"reason":"test","requires_tools":[]}'
        mock_llm = make_mock_llm(json_str)

        with patch("agents.orchestrator.router.create_chat_model",
                   return_value=MagicMock()):
            router = Router()
            router.llm = mock_llm

            result = await router.decide("测试")

        assert result.track == "clarify"

    @pytest.mark.asyncio
    async def test_route_backward_compat(self):
        """向后兼容的 route() 方法返回 RouteResult"""
        json_str = '{"track":"fast","confidence":0.95,"reason":"VPN排查","requires_tools":[]}'
        mock_llm = make_mock_llm(json_str)

        with patch("agents.orchestrator.router.create_chat_model",
                   return_value=MagicMock()):
            router = Router()
            router.llm = mock_llm

            result = await router.route("VPN怎么连")

        assert isinstance(result, RouteResult)
        assert result.track == "fast"
        assert result.agent_id == "enterprise_rag"
        assert result.category == "knowledge_query"

    @pytest.mark.asyncio
    async def test_llm_call_failure(self):
        """LLM 调用异常 → 返回 clarify"""
        mock = MagicMock()
        mock.ainvoke = AsyncMock(side_effect=Exception("API timeout"))

        with patch("agents.orchestrator.router.create_chat_model",
                   return_value=MagicMock()):
            router = Router()
            router.llm = mock

            result = await router.decide("test")

        assert result.track == "clarify"
        assert result.confidence == 0.2


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

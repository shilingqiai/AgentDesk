"""
Router 路由决策测试

测试覆盖：
- 四轨道正确分发（fast/action/complex/clarify）
- bind_tools (Function Calling) 主路径
- prompt→JSON fallback 路径
- 低置信度转为 clarify 检验
- 未知轨道兜底
- LLM 异常降级

v3.2: bind_tools + tool_calls 优先，prompt→JSON fallback
"""

import pytest
from unittest.mock import patch, AsyncMock, MagicMock
from agents.orchestrator.router import Router, RouterDecision, RouteResult


# ============================================================
# 辅助
# ============================================================

def make_mock_llm_with_tool_calls(track: str, confidence: float = 0.9,
                                   reason: str = "", requires_tools: list = None):
    """创建返回 tool_calls 的 mock LLM（bind_tools 主路径）"""
    mock = MagicMock()
    mock_response = MagicMock()
    mock_response.tool_calls = [{
        "name": "route_decision",
        "args": {
            "track": track,
            "confidence": confidence,
            "reason": reason or f"Mock {track} decision",
            "requires_tools": requires_tools or [],
        },
    }]
    mock_response.content = ""  # tool_calls 时 content 通常为空
    mock.ainvoke = AsyncMock(return_value=mock_response)
    return mock


def make_mock_llm_with_json(json_str: str):
    """创建返回 JSON 字符串的 mock LLM（prompt→JSON fallback 路径）"""
    mock = MagicMock()
    mock_response = MagicMock()
    mock_response.tool_calls = []   # 无 tool_calls → 触发 fallback
    mock_response.content = json_str
    mock.ainvoke = AsyncMock(return_value=mock_response)
    return mock


# ============================================================
# 测试
# ============================================================

class TestRouterDecision:
    """Router 结构化决策测试 (v3.2 bind_tools)"""

    @pytest.mark.asyncio
    async def test_fast_track_fc(self):
        """fast 轨道：Function Calling 主路径"""
        mock_llm = make_mock_llm_with_tool_calls(
            "fast", confidence=0.95, reason="用户询问VPN排查方法"
        )

        with patch("agents.orchestrator.router.create_chat_model",
                   return_value=MagicMock()):
            router = Router()
            router.llm_with_tools = mock_llm

            result = await router.decide(
                "VPN怎么排查连接失败的问题？",
                agent_descriptions="enterprise_rag(企业知识库问答), ticket_dispatch(工单派发)",
            )

        assert result.track == "fast"
        assert result.confidence == 0.95
        assert "VPN" in result.reason

    @pytest.mark.asyncio
    async def test_action_track_leave_fc(self):
        """action 轨道：请假申请 — Function Calling"""
        mock_llm = make_mock_llm_with_tool_calls(
            "action", confidence=0.90, reason="用户申请请假"
        )

        with patch("agents.orchestrator.router.create_chat_model",
                   return_value=MagicMock()):
            router = Router()
            router.llm_with_tools = mock_llm

            result = await router.decide(
                "我想请3天年假",
                agent_descriptions="enterprise_rag, ticket_dispatch",
            )

        assert result.track == "action"
        assert result.confidence == 0.90

    @pytest.mark.asyncio
    async def test_action_track_it_fc(self):
        """action 轨道：IT 故障工单 — Function Calling"""
        mock_llm = make_mock_llm_with_tool_calls(
            "action", confidence=0.88, reason="用户提交IT故障工单",
            requires_tools=["jira_api"],
        )

        with patch("agents.orchestrator.router.create_chat_model",
                   return_value=MagicMock()):
            router = Router()
            router.llm_with_tools = mock_llm

            result = await router.decide(
                "帮我提交一个网络故障工单",
                agent_descriptions="enterprise_rag, ticket_dispatch",
            )

        assert result.track == "action"
        assert result.requires_tools == ["jira_api"]

    @pytest.mark.asyncio
    async def test_clarify_track_fc(self):
        """clarify 轨道：输入模糊 — Function Calling"""
        mock_llm = make_mock_llm_with_tool_calls(
            "clarify", confidence=0.20, reason="输入过于模糊无法判断意图"
        )

        with patch("agents.orchestrator.router.create_chat_model",
                   return_value=MagicMock()):
            router = Router()
            router.llm_with_tools = mock_llm

            result = await router.decide(
                "嗯？",
                agent_descriptions="enterprise_rag, ticket_dispatch",
            )

        assert result.track == "clarify"
        assert result.confidence == 0.20

    @pytest.mark.asyncio
    async def test_complex_track_fc(self):
        """complex 轨道：多步骤复合指令 — Function Calling"""
        mock_llm = make_mock_llm_with_tool_calls(
            "complex", confidence=0.75, reason="多步骤复合指令"
        )

        with patch("agents.orchestrator.router.create_chat_model",
                   return_value=MagicMock()):
            router = Router()
            router.llm_with_tools = mock_llm

            result = await router.decide(
                "帮我查天气然后请假再取消会议室",
                agent_descriptions="enterprise_rag, ticket_dispatch",
            )

        assert result.track == "complex"

    # ── Fallback 路径测试 ──

    @pytest.mark.asyncio
    async def test_fallback_prompt_json(self):
        """prompt→JSON fallback：无 tool_calls 时降级解析 content JSON"""
        json_str = '{"track":"fast","confidence":0.85,"reason":"查询VPN","requires_tools":[]}'
        mock_llm = make_mock_llm_with_json(json_str)

        with patch("agents.orchestrator.router.create_chat_model",
                   return_value=MagicMock()):
            router = Router()
            router.llm_with_tools = mock_llm

            result = await router.decide("VPN怎么连")

        assert result.track == "fast"
        assert result.confidence == 0.85

    @pytest.mark.asyncio
    async def test_fallback_json_in_markdown_block(self):
        """prompt→JSON fallback：LLM 返回 markdown 包裹的 JSON"""
        json_str = '```json\n{"track":"fast","confidence":0.85,"reason":"查询VPN","requires_tools":[]}\n```'
        mock_llm = make_mock_llm_with_json(json_str)

        with patch("agents.orchestrator.router.create_chat_model",
                   return_value=MagicMock()):
            router = Router()
            router.llm_with_tools = mock_llm

            result = await router.decide("VPN怎么连")

        assert result.track == "fast"
        assert result.confidence == 0.85

    # ── 兜底测试 ──

    @pytest.mark.asyncio
    async def test_invalid_track_fallback(self):
        """LLM 返回未知轨道 → 强制 clarify"""
        mock_llm = make_mock_llm_with_tool_calls(
            "unknown_track", confidence=0.8, reason="test"
        )

        with patch("agents.orchestrator.router.create_chat_model",
                   return_value=MagicMock()):
            router = Router()
            router.llm_with_tools = mock_llm

            result = await router.decide("测试")

        assert result.track == "clarify"

    @pytest.mark.asyncio
    async def test_route_backward_compat(self):
        """向后兼容的 route() 方法返回 RouteResult"""
        mock_llm = make_mock_llm_with_tool_calls(
            "fast", confidence=0.95, reason="VPN排查"
        )

        with patch("agents.orchestrator.router.create_chat_model",
                   return_value=MagicMock()):
            router = Router()
            router.llm_with_tools = mock_llm

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
            router.llm_with_tools = mock

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

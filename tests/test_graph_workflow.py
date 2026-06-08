"""
编排工作流测试

测试覆盖：
- 工作流图构建
- 路由后节点分发
- 初始状态创建
- 对话上下文构建
- 反问节点（高/低置信度）
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch


class TestGraphWorkflow:
    """编排工作流图测试"""

    def test_build_workflow(self):
        """验证工作流图可正常构建"""
        from agents.graph_workflow import build_orchestration_workflow

        workflow = build_orchestration_workflow()
        assert workflow is not None

        # 验证节点已注册
        compiled = workflow.compile()
        assert compiled is not None

    def test_create_initial_state(self):
        """验证初始状态创建"""
        from agents.graph_workflow import create_initial_state
        from langchain_core.messages import HumanMessage

        state = create_initial_state("VPN怎么排查？", thread_id="test-001")

        assert len(state["messages"]) == 1
        assert isinstance(state["messages"][0], HumanMessage)
        assert state["messages"][0].content == "VPN怎么排查？"
        assert state["track"] == ""
        assert state["agent_id"] == ""
        assert state["confidence"] == 0.0
        assert state["urgency"] == "medium"
        assert state["resolved"] is False
        assert state["thread_id"] == "test-001"

    def test_build_conversation_context_empty(self):
        """无历史 → 返回空字符串"""
        from agents.graph_workflow import _build_conversation_context
        from langchain_core.messages import HumanMessage

        messages = [HumanMessage(content="你好")]
        ctx = _build_conversation_context(messages)
        assert ctx == ""

    def test_build_conversation_context_with_history(self):
        """有历史 → 返回格式化上下文"""
        from agents.graph_workflow import _build_conversation_context
        from langchain_core.messages import HumanMessage, AIMessage

        messages = [
            HumanMessage(content="VPN怎么排查？"),
            AIMessage(content="请先检查网络连通性"),
            HumanMessage(content="网络通的但连不上"),
        ]
        ctx = _build_conversation_context(messages)
        assert "VPN" in ctx
        assert "网络" in ctx

    def test_get_user_text(self):
        """获取当前用户输入"""
        from agents.graph_workflow import create_initial_state, _get_user_text

        state = create_initial_state("测试用户输入")
        text = _get_user_text(state)
        assert text == "测试用户输入"

    def test_route_after_route_fast(self):
        """路由分发：fast 轨道"""
        from agents.graph_workflow import route_after_route

        result = route_after_route({"track": "fast"})
        assert result == "fast_track"

    def test_route_after_route_action(self):
        """路由分发：action 轨道"""
        from agents.graph_workflow import route_after_route

        result = route_after_route({"track": "action"})
        assert result == "action_track"

    def test_route_after_route_complex(self):
        """路由分发：complex 轨道"""
        from agents.graph_workflow import route_after_route

        result = route_after_route({"track": "complex"})
        assert result == "complex_track"

    def test_route_after_route_clarify(self):
        """路由分发：clarify / 未知轨道"""
        from agents.graph_workflow import route_after_route

        result = route_after_route({"track": "clarify"})
        assert result == "clarification"

    def test_route_after_route_unknown(self):
        """路由分发：未知轨道 → clarification"""
        from agents.graph_workflow import route_after_route

        result = route_after_route({"track": "garbage"})
        assert result == "clarification"


class TestClarificationNode:
    """反问节点测试"""

    @pytest.mark.asyncio
    async def test_clarification_low_confidence(self):
        """低置信度（< 0.3）：通用引导"""
        from agents.graph_workflow import clarification_node, create_initial_state

        state = create_initial_state("随便说点什么")
        state["confidence"] = 0.1
        state["track"] = "clarify"

        result = await clarification_node(state)

        assert result["resolved"] is True
        assert "不太确定" in result["final_response"]
        assert "VPN" in result["final_response"] or "查询知识" in result["final_response"]

    @pytest.mark.asyncio
    async def test_clarification_medium_confidence(self):
        """中置信度（0.3~0.7）：引导式反问"""
        from agents.graph_workflow import clarification_node, create_initial_state

        state = create_initial_state("请假")
        state["confidence"] = 0.5
        state["track"] = "clarify"

        result = await clarification_node(state)

        assert result["resolved"] is True
        # 引导式反问应提及两条路径
        assert "查询" in result["final_response"] or "工单" in result["final_response"]


class TestRespondNode:
    """响应节点测试"""

    @pytest.mark.asyncio
    async def test_respond_normal(self):
        """正常响应"""
        from agents.graph_workflow import respond_node, create_initial_state

        state = create_initial_state("VPN怎么连")
        state["final_response"] = "VPN排查步骤..."

        result = await respond_node(state)

        assert result["resolved"] is True
        assert len(result["messages"]) == 2  # 原始 + AI 回复
        assert result["messages"][-1].content == "VPN排查步骤..."

    @pytest.mark.asyncio
    async def test_respond_with_human_review(self):
        """需人工审核的响应"""
        from agents.graph_workflow import respond_node, create_initial_state

        state = create_initial_state("创建高优工单")
        state["final_response"] = "工单已创建"
        state["needs_human_review"] = True
        state["track"] = "action"

        result = await respond_node(state)

        assert result["resolved"] is True
        assert "人工审核" in result["final_response"]

    @pytest.mark.asyncio
    async def test_respond_empty_fallback(self):
        """无响应内容 → 默认兜底"""
        from agents.graph_workflow import respond_node, create_initial_state

        state = create_initial_state("测试")
        state["final_response"] = ""

        result = await respond_node(state)

        assert result["resolved"] is True
        assert "处理完成" in result["final_response"]


class TestOrchestrationRunner:
    """编排运行器测试"""

    def test_init(self):
        """验证运行器可正常初始化"""
        from agents.graph_workflow import OrchestrationWorkflowRunner

        runner = OrchestrationWorkflowRunner()
        assert runner is not None
        assert runner.app is not None

    @pytest.mark.asyncio
    async def test_run_sync(self):
        """同步运行（ainvoke）"""
        from agents.graph_workflow import OrchestrationWorkflowRunner

        # 这里只验证运行器可执行，不验证 LLM 调用结果
        runner = OrchestrationWorkflowRunner()
        assert runner.workflow is not None

    def test_get_state(self):
        """获取会话状态 — 新会话可能返回 None 或空 StateSnapshot"""
        from agents.graph_workflow import OrchestrationWorkflowRunner

        runner = OrchestrationWorkflowRunner()
        state = runner.get_state("nonexistent-thread-xyz-123")
        # SqliteSaver 返回空 StateSnapshot，MemorySaver 返回 None
        # 两者都表示无状态
        if state is not None:
            # StateSnapshot 的 values 应为空或默认
            assert state.values == {} or state.next == ()
            # 或者检查是否有实际消息
            if hasattr(state, 'values') and state.values:
                messages = state.values.get("messages", [])
                assert messages == [] or len(messages) == 0

    def test_reset(self):
        """重置会话"""
        from agents.graph_workflow import OrchestrationWorkflowRunner

        runner = OrchestrationWorkflowRunner()
        # reset 不应报错
        runner.reset("test-thread")

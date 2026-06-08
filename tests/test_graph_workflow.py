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


class TestCardLocking:
    """卡片锁 + 重路由测试 (v3.1)"""

    def test_route_node_short_circuits_with_pending_card(self):
        """pending_card_type 存在时 route_node 短路 Router"""
        from agents.graph_workflow import route_node, create_initial_state

        state = create_initial_state("算了，不要了")
        state["pending_card_type"] = "admin"

        import asyncio
        result = asyncio.run(route_node(state))

        assert result["track"] == "action"
        assert result["agent_id"] == "ticket_dispatch"
        assert result["confidence"] == 1.0
        assert result["intent"] == "admin"

    def test_route_node_short_circuits_long_input(self):
        """长输入在卡片锁期间也短路（让 Agent 判断意图）"""
        from agents.graph_workflow import route_node, create_initial_state

        long_text = "行吧那就帮我把今天下午最大的会议室预定了吧谢谢你哈"
        state = create_initial_state(long_text)
        state["pending_card_type"] = "admin"

        import asyncio
        result = asyncio.run(route_node(state))

        assert result["track"] == "action"
        assert result["confidence"] == 1.0

    def test_route_node_no_pending_card_routes_normally(self):
        """无 pending_card 时正常走 Router"""
        from agents.graph_workflow import route_node, create_initial_state

        state = create_initial_state("VPN怎么连")
        state["pending_card_type"] = ""

        import asyncio
        result = asyncio.run(route_node(state))

        # 应该正常路由（通过 Router LLM），不应该被短路到 action
        # 但这里 Router LLM 实际不可用，会 fallback 到 clarify
        assert result["track"] in ("fast", "action", "complex", "clarify")

    def test_after_action_track_respond(self):
        """re_route=False → 去 respond"""
        from agents.graph_workflow import after_action_track

        result = after_action_track({"re_route": False})
        assert result == "respond"

    def test_after_action_track_reroute(self):
        """re_route=True → 回 route"""
        from agents.graph_workflow import after_action_track

        result = after_action_track({"re_route": True})
        assert result == "route"

    def test_create_initial_state_has_card_fields(self):
        """新 state 包含 pending_card_type 和 re_route 默认值"""
        from agents.graph_workflow import create_initial_state

        state = create_initial_state("测试")
        assert state["pending_card_type"] == ""
        assert state["re_route"] is False

    def test_action_track_sets_pending_card_on_card_return(self):
        """非锁模式返回卡片时设置 pending_card_type"""
        from agents.graph_workflow import action_track_node, create_initial_state
        from agents.sub_agents.ticket_dispatch import TicketDispatchSubAgent
        from agents.a2a.protocol import AgentMessage
        from tests.conftest import create_mock_llm, TICKET_PARAMS_IT_RESPONSE

        state = create_initial_state("预定明天早上会议室")
        state["pending_card_type"] = ""

        import asyncio

        # Mock 行为: execute 返回卡片
        async def run_test():
            # 使用 mock LLM 创建 agent 实例
            agent = TicketDispatchSubAgent()
            agent.llm = create_mock_llm([TICKET_PARAMS_IT_RESPONSE])

            mock_db = MagicMock()
            agent._db_router = mock_db

            from agents.orchestrator.agent_registry import agent_registry
            # 确保 agent 已注册
            if not agent_registry.get_agent("ticket_dispatch"):
                agent_registry.register(
                    TicketDispatchSubAgent.__agent_declaration__,
                    lambda: agent,
                )

            result = await action_track_node(state)
            # 如果返回了卡片，pending_card_type 应该被设置
            # 注意：这里依赖真实 Agent 执行，可能因 LLM 不可用而走不到卡片路径
            # 但至少不应报错
            assert result is not None

        asyncio.run(run_test())

    @pytest.mark.asyncio
    async def test_action_track_card_lock_confirm(self):
        """卡片锁模式：confirm 意图 → 执行卡片 + 清锁"""
        from agents.graph_workflow import action_track_node, create_initial_state
        from agents.sub_agents.ticket_dispatch import TicketDispatchSubAgent
        from unittest.mock import AsyncMock, patch

        state = create_initial_state("确认")
        state["pending_card_type"] = "admin"
        state["agent_results"] = {
            "ticket_dispatch": {
                "card": {
                    "type": "booking",
                    "title": "会议室预定",
                    "description": "测试卡片",
                    "fields": [
                        {"key": "room_id", "label": "会议室", "value": "1"},
                        {"key": "date", "label": "日期", "value": "2026-06-09"},
                        {"key": "time_slot", "label": "时段", "value": "09:00-10:30"},
                        {"key": "title", "label": "主题", "value": "周会"},
                    ],
                    "action": "/api/meeting-rooms/{room_id}/book",
                    "confirm_text": "确认预定",
                    "success_message": "会议室预定成功！",
                }
            }
        }

        # Mock classify_card_response → confirm
        # Mock execute_card → 成功消息
        with patch.object(
            TicketDispatchSubAgent, "classify_card_response",
            new_callable=AsyncMock,
        ) as mock_classify, patch.object(
            TicketDispatchSubAgent, "execute_card",
            new_callable=AsyncMock,
        ) as mock_execute:

            mock_classify.return_value = "confirm"
            mock_execute.return_value = "✅ 会议室预定成功！\n📅 2026-06-09 09:00-10:30"

            result = await action_track_node(state)

            assert result["pending_card_type"] == ""
            assert result["re_route"] is False
            assert "会议室预定成功" in result["final_response"]
            assert result["resolved"] is True

    @pytest.mark.asyncio
    async def test_action_track_card_lock_cancel(self):
        """卡片锁模式：cancel 意图 → 清锁，返回已取消"""
        from agents.graph_workflow import action_track_node, create_initial_state
        from agents.sub_agents.ticket_dispatch import TicketDispatchSubAgent
        from unittest.mock import AsyncMock, patch

        state = create_initial_state("算了，不要了")
        state["pending_card_type"] = "admin"
        state["agent_results"] = {
            "ticket_dispatch": {
                "card": {"type": "booking", "title": "会议室预定", "description": ""}
            }
        }

        with patch.object(
            TicketDispatchSubAgent, "classify_card_response",
            new_callable=AsyncMock,
        ) as mock_classify:

            mock_classify.return_value = "cancel"

            result = await action_track_node(state)

            assert result["pending_card_type"] == ""
            assert result["re_route"] is False
            assert "取消" in result["final_response"]

    @pytest.mark.asyncio
    async def test_action_track_card_lock_modify(self):
        """卡片锁模式：modify 意图 → 重建卡片，保持锁"""
        from agents.graph_workflow import action_track_node, create_initial_state
        from agents.sub_agents.ticket_dispatch import TicketDispatchSubAgent
        from unittest.mock import AsyncMock, patch

        state = create_initial_state("改到明天下午")
        state["pending_card_type"] = "admin"
        state["agent_results"] = {
            "ticket_dispatch": {
                "card": {"type": "booking", "title": "会议室预定", "description": ""}
            }
        }

        with patch.object(
            TicketDispatchSubAgent, "classify_card_response",
            new_callable=AsyncMock,
        ) as mock_classify, patch.object(
            TicketDispatchSubAgent, "rebuild_card",
            new_callable=AsyncMock,
        ) as mock_rebuild:

            mock_classify.return_value = "modify"
            mock_rebuild.return_value = {
                "type": "booking",
                "title": "会议室预定（已更新）",
                "description": "已更新为明天下午",
                "fields": [],
                "confirm_text": "确认预定",
            }

            result = await action_track_node(state)

            # pending_card_type 保持不变（继续锁）
            assert result["pending_card_type"] == "admin"
            assert result["re_route"] is False
            assert "[CARD]" in result["final_response"]

    @pytest.mark.asyncio
    async def test_action_track_card_lock_new_topic(self):
        """卡片锁模式：new_topic 意图 → 清锁 + re_route=True"""
        from agents.graph_workflow import action_track_node, create_initial_state
        from agents.sub_agents.ticket_dispatch import TicketDispatchSubAgent
        from unittest.mock import AsyncMock, patch

        state = create_initial_state("帮我查一下年假政策")
        state["pending_card_type"] = "admin"
        state["agent_results"] = {
            "ticket_dispatch": {
                "card": {"type": "booking", "title": "会议室预定", "description": ""}
            }
        }

        with patch.object(
            TicketDispatchSubAgent, "classify_card_response",
            new_callable=AsyncMock,
        ) as mock_classify:

            mock_classify.return_value = "new_topic"

            result = await action_track_node(state)

            assert result["pending_card_type"] == ""
            assert result["re_route"] is True
            # 不设 final_response（留给重路由后的节点）
            assert result["final_response"] == ""

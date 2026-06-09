"""
TicketDispatchSubAgent 测试

测试覆盖：
- IT故障工单提取与创建
- 请假工单提取与创建
- 报销工单提取与创建
- 参数提取失败降级
- 规则兜底（关键词匹配）
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch


class TestTicketDispatch:
    """工单派发 Agent 业务逻辑测试"""

    @pytest.mark.asyncio
    async def test_execute_it_fault(self):
        """IT 故障工单：提取参数 + RAG-first 卡片"""
        from agents.sub_agents.ticket_dispatch import TicketDispatchSubAgent
        from agents.a2a.protocol import AgentMessage
        from tests.conftest import create_mock_llm, TICKET_PARAMS_IT_RESPONSE

        agent = TicketDispatchSubAgent()
        # v3.2: _extract_params 使用 llm_extract（bind_tools 版本）
        agent.llm_extract = create_mock_llm([TICKET_PARAMS_IT_RESPONSE])

        # Mock DB Router
        mock_db = MagicMock()
        agent._db_router = mock_db

        msg = AgentMessage.create_delegation(
            from_agent="orchestrator",
            to_agent="ticket_dispatch",
            payload={"user_input": "网络故障", "task": "创建工单", "urgency": "medium"},
            trace_id="test-trace",
        )

        result = await agent.execute(msg)

        assert result.success is True
        assert result.payload["ticket_type"] == "it_fault"
        # v3: IT 类型返回 RAG-first 确认卡片
        assert result.payload["return_card"] is True
        assert result.payload["card"]["type"] == "confirm"
        assert "IT" in result.payload["card"]["title"]
        assert result.payload["card"]["confirm_text"] == "仍需要帮助，创建工单"

    @pytest.mark.asyncio
    async def test_execute_leave(self):
        """请假工单：提取参数 + 确认卡片"""
        from agents.sub_agents.ticket_dispatch import TicketDispatchSubAgent
        from agents.a2a.protocol import AgentMessage
        from tests.conftest import create_mock_llm, TICKET_PARAMS_LEAVE_RESPONSE

        agent = TicketDispatchSubAgent()
        agent.llm_extract = create_mock_llm([TICKET_PARAMS_LEAVE_RESPONSE])

        mock_db = MagicMock()
        agent._db_router = mock_db

        msg = AgentMessage.create_delegation(
            from_agent="orchestrator",
            to_agent="ticket_dispatch",
            payload={"user_input": "申请年假5天", "task": "创建请假工单", "urgency": "low"},
            trace_id="test-trace-leave",
        )

        result = await agent.execute(msg)

        assert result.success is True
        assert result.payload["ticket_type"] == "leave"
        # v3: leave 类型返回确认卡片
        assert result.payload["return_card"] is True
        assert result.payload["card"]["type"] == "confirm"
        assert "请假" in result.payload["card"]["title"]
        assert result.payload["card"]["confirm_text"] == "提交请假申请"

    @pytest.mark.asyncio
    async def test_execute_expense(self):
        """报销工单：提取参数 + 确认卡片"""
        from agents.sub_agents.ticket_dispatch import TicketDispatchSubAgent
        from agents.a2a.protocol import AgentMessage
        from tests.conftest import create_mock_llm, TICKET_PARAMS_EXPENSE_RESPONSE

        agent = TicketDispatchSubAgent()
        agent.llm_extract = create_mock_llm([TICKET_PARAMS_EXPENSE_RESPONSE])

        mock_db = MagicMock()
        agent._db_router = mock_db

        msg = AgentMessage.create_delegation(
            from_agent="orchestrator",
            to_agent="ticket_dispatch",
            payload={"user_input": "报销差旅费2500", "task": "创建报销工单", "urgency": "medium"},
            trace_id="test-trace-expense",
        )

        result = await agent.execute(msg)

        assert result.success is True
        assert result.payload["ticket_type"] == "expense"
        # v3: expense 类型返回确认卡片
        assert result.payload["return_card"] is True
        assert result.payload["card"]["type"] == "confirm"
        assert "报销" in result.payload["card"]["title"]

    @pytest.mark.asyncio
    async def test_extract_params_json_repair(self):
        """JSON 格式有问题 → json_repair 修复（prompt→JSON fallback）"""
        from agents.sub_agents.ticket_dispatch import TicketDispatchSubAgent
        from tests.conftest import create_mock_llm

        agent = TicketDispatchSubAgent()

        # 返回有问题的 JSON（尾逗号）→ 触发 json_repair 修复
        bad_json = '{"ticket_type":"it_fault","title":"测试","description":"test","category":"其他","priority":"P2","extra":{},}'  # noqa: E501
        agent.llm_extract = create_mock_llm([bad_json])

        params = await agent._extract_params("网络故障")
        assert params["ticket_type"] == "it_fault"

    @pytest.mark.asyncio
    async def test_fallback_extract_leave_keywords(self):
        """规则兜底：关键词匹配请假"""
        from agents.sub_agents.ticket_dispatch import TicketDispatchSubAgent

        agent = TicketDispatchSubAgent()
        params = agent._fallback_extract("我想请年假3天", "low")

        assert params["ticket_type"] == "leave"
        assert len(params["title"]) > 0

    @pytest.mark.asyncio
    async def test_fallback_extract_expense_keywords(self):
        """规则兜底：关键词匹配报销"""
        from agents.sub_agents.ticket_dispatch import TicketDispatchSubAgent

        agent = TicketDispatchSubAgent()
        params = agent._fallback_extract("报销差旅费用200元", "medium")

        assert params["ticket_type"] == "expense"

    @pytest.mark.asyncio
    async def test_fallback_extract_admin_keywords(self):
        """规则兜底：关键词匹配行政"""
        from agents.sub_agents.ticket_dispatch import TicketDispatchSubAgent

        agent = TicketDispatchSubAgent()
        params = agent._fallback_extract("帮我预定一个会议室", "low")

        assert params["ticket_type"] == "admin"

    @pytest.mark.asyncio
    async def test_fallback_extract_it_default(self):
        """规则兜底：无匹配关键词 → 默认 IT"""
        from agents.sub_agents.ticket_dispatch import TicketDispatchSubAgent

        agent = TicketDispatchSubAgent()
        params = agent._fallback_extract("系统报错了，帮我看看", "high")

        assert params["ticket_type"] == "it_fault"

    def test_build_extra_payload_leave(self):
        """构建请假扩展字段"""
        from agents.sub_agents.ticket_dispatch import TicketDispatchSubAgent

        params = {
            "extra": {
                "leave_type": "年假",
                "start_date": "2026-06-15",
                "total_days": 5,
            },
            "description": "休假",
        }
        payload = TicketDispatchSubAgent._build_extra_payload("leave", params)
        assert payload["leave_type"] == "年假"
        assert payload["total_days"] == 5

    def test_build_extra_payload_expense(self):
        """构建报销扩展字段"""
        from agents.sub_agents.ticket_dispatch import TicketDispatchSubAgent

        params = {
            "extra": {"expense_type": "差旅费", "amount": 1500, "has_invoice": True},
        }
        payload = TicketDispatchSubAgent._build_extra_payload("expense", params)
        assert payload["amount"] == 1500
        assert payload["has_invoice"] is True

    def test_build_response_leave(self):
        """构建请假工单响应"""
        from agents.sub_agents.ticket_dispatch import TicketDispatchSubAgent

        agent = TicketDispatchSubAgent()
        ticket = {
            "ticket_number": "TK-001", "title": "年假申请",
            "priority": "P3", "category": "年假", "status": "created",
            "payload": {"leave_type": "年假", "total_days": 5},
        }
        resp = agent._build_response(ticket, "leave")
        assert "请假申请" in resp
        assert "年假" in resp
        assert "TK-001" in resp

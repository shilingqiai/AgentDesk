"""
测试基础设施 — 共享 fixtures 和 mock 对象

使用方法:
    pytest tests/ -v
    pytest tests/ -v --cov=. --cov-report=term
"""

from __future__ import annotations

import os
import sys
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

# 确保项目根目录在 Python path 中
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ============================================================
# Mock LLM
# ============================================================

class MockLLMResponse:
    """模拟 LLM 返回对象（v3.2 bind_tools 兼容）"""
    def __init__(self, content: str, tool_calls: list = None):
        self.content = content
        # tool_calls=None → 模拟 bind_tools 返回值（空列表=未调用 tool）
        self.tool_calls = tool_calls if tool_calls is not None else []

    def __repr__(self):
        tc = f", tool_calls={self.tool_calls}" if self.tool_calls else ""
        return f"MockLLMResponse(content={self.content[:50]!r}{tc})"


def create_mock_llm(responses: list[str] = None):
    """
    创建模拟 LLM，按顺序返回预设响应。
    返回的 mock 对象模拟 bind_tools 行为：tool_calls=[]，触发 prompt→JSON fallback。

    Args:
        responses: 每次调用的返回内容列表

    Returns:
        AsyncMock 实例
    """
    mock = AsyncMock()
    if responses:
        mock.ainvoke = AsyncMock(side_effect=[
            MockLLMResponse(r, tool_calls=[]) for r in responses
        ])
    else:
        mock.ainvoke = AsyncMock(return_value=MockLLMResponse("{}", tool_calls=[]))
    return mock


def create_mock_llm_with_tool_calls(tool_args: dict):
    """
    创建模拟 bind_tools LLM，返回 tool_calls 响应。

    Args:
        tool_args: tool_calls[0]["args"] 的内容
    """
    mock = AsyncMock()
    mock_response = MockLLMResponse("", tool_calls=[
        {"name": "extract_ticket_params", "args": tool_args}
    ])
    mock.ainvoke = AsyncMock(return_value=mock_response)
    return mock


# ============================================================
# 常用 LLM 响应预设
# ============================================================

ROUTER_FAST_RESPONSE = (
    '{"track":"fast","confidence":0.95,"reason":"用户询问VPN排查方法","requires_tools":[]}'
)

ROUTER_ACTION_LEAVE_RESPONSE = (
    '{"track":"action","confidence":0.90,"reason":"用户申请请假","requires_tools":[]}'
)

ROUTER_ACTION_IT_RESPONSE = (
    '{"track":"action","confidence":0.88,"reason":"用户提交IT故障工单","requires_tools":["jira_api"]}'
)

ROUTER_CLARIFY_RESPONSE = (
    '{"track":"clarify","confidence":0.20,"reason":"输入过于模糊无法判断意图","requires_tools":[]}'
)

ROUTER_COMPLEX_RESPONSE = (
    '{"track":"complex","confidence":0.75,"reason":"多步骤复合指令","requires_tools":[]}'
)

TICKET_PARAMS_IT_RESPONSE = (
    '{"ticket_type":"it_fault","title":"网络故障工单","description":"办公室网络连接不稳定","category":"网络故障","priority":"P2","extra":{"suggested_engineer_skill":"网络","affected_users":5}}'
)

TICKET_PARAMS_LEAVE_RESPONSE = (
    '{"ticket_type":"leave","title":"年假申请","description":"申请年假5天","category":"年假","priority":"P3","extra":{"leave_type":"年假","start_date":"2026-06-15","end_date":"2026-06-19","total_days":5,"reason":"个人休假"}}'
)

TICKET_PARAMS_EXPENSE_RESPONSE = (
    '{"ticket_type":"expense","title":"差旅报销","description":"出差上海差旅费报销","category":"差旅费","priority":"P2","extra":{"expense_type":"差旅费","amount":2500.0,"has_invoice":true}}'
)

TICKET_PARAMS_INVALID_JSON = (
    'invalid json response without proper structure'
)


# ============================================================
# Fixtures
# ============================================================

@pytest.fixture
def mock_llm_fast():
    """返回 fast 轨道的 mock LLM"""
    return create_mock_llm([ROUTER_FAST_RESPONSE])


@pytest.fixture
def mock_llm_action_leave():
    """返回 action(请假) 轨道的 mock LLM"""
    return create_mock_llm([ROUTER_ACTION_LEAVE_RESPONSE])


@pytest.fixture
def mock_llm_action_it():
    """返回 action(IT故障) 轨道的 mock LLM"""
    return create_mock_llm([ROUTER_ACTION_IT_RESPONSE])


@pytest.fixture
def mock_llm_clarify():
    """返回 clarify 轨道的 mock LLM"""
    return create_mock_llm([ROUTER_CLARIFY_RESPONSE])


@pytest.fixture
def mock_llm_complex():
    """返回 complex 轨道的 mock LLM"""
    return create_mock_llm([ROUTER_COMPLEX_RESPONSE])


@pytest.fixture
def mock_embedding():
    """模拟 embedding 向量"""
    import numpy as np
    return np.random.randn(1024).astype('float32').tolist()


@pytest.fixture
def sample_user_inputs():
    """常用测试输入"""
    return {
        "fast": "VPN怎么排查连接失败的问题？",
        "action_it": "帮我提交一个网络故障工单，办公室网络上不去",
        "action_leave": "我想请3天年假，下周一到周三",
        "action_expense": "报销昨天出差的交通费200元",
        "clarify": "嗯？",
        "complex": "帮我查一下会议室有没有空的，然后预定一间，再把这事通知团队",
    }


@pytest.fixture
def sample_conversation_history():
    """模拟对话历史"""
    return "用户: 我的电脑连不上VPN\n助手: 请先检查网络连通性"


@pytest.fixture
def sample_agent_descriptions():
    """模拟 Agent 描述"""
    return (
        "enterprise_rag(企业知识库问答: IT/HR/行政), "
        "ticket_dispatch(工单派发: 创建/查询多类型工单-IT故障/请假/报销/行政)"
    )

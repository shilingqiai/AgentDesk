"""
轻量 A2A 协议 — 标准化 Agent 间任务委托与结果返回格式

设计原则:
- Single Response: 子 Agent 永远 reply_to_user=False，只返回结构化结果给编排器
- 最小抽象: 一个 AgentMessage 数据类 + create_delegation/create_response 两个工厂方法
- trace_id 全链路追踪
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Any


@dataclass
class AgentMessage:
    """标准化的 Agent 间消息 — 统一委托与响应格式"""

    from_agent: str = ""
    to_agent: str = ""
    payload: dict[str, Any] = field(default_factory=dict)
    success: bool = True
    error: str | None = None
    trace_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])

    @classmethod
    def create_delegation(
        cls,
        from_agent: str,
        to_agent: str,
        payload: dict,
        context: dict = None,       # 已废弃，保留兼容
        trace_id: str = "",
    ) -> "AgentMessage":
        """编排器 → 子 Agent 任务委派"""
        return cls(
            from_agent=from_agent,
            to_agent=to_agent,
            payload=payload,
            trace_id=trace_id or uuid.uuid4().hex[:12],
        )

    @classmethod
    def create_response(
        cls,
        from_agent: str,
        to_agent: str,
        payload: dict,
        original_message: "AgentMessage" = None,   # 可选，用于继承 trace_id
        success: bool = True,
        error: str = None,
    ) -> "AgentMessage":
        """子 Agent → 编排器 结果返回"""
        return cls(
            from_agent=from_agent,
            to_agent=to_agent,
            payload=payload,
            success=success,
            error=error,
            trace_id=original_message.trace_id if original_message else uuid.uuid4().hex[:12],
        )

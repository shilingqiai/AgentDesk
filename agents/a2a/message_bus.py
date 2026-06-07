"""
Agent Message Bus — 消息总线

参考 Copilot Studio 的 Agent 间通信追踪：
- 记录所有 AgentMessage 的传递链路
- 支持 trace_id 关联一次用户请求的所有Agent调用
- 结构化日志输出
"""

from __future__ import annotations

import logging
from typing import Any, Optional
from collections import defaultdict
from datetime import datetime

from .protocol import AgentMessage

logger = logging.getLogger("a2a.message_bus")


class MessageBus:
    """
    Agent 消息总线

    职责：
    1. 记录所有 Agent 间消息的传递链路
    2. 以 trace_id 关联一次用户请求的所有 Agent 调用
    3. 提供调用链查询能力（用于调试和可观测性）

    使用方式：
        bus = MessageBus()
        bus.record(message)
        trace = bus.get_trace(trace_id)
    """

    def __init__(self):
        # trace_id → [AgentMessage, ...]
        self._traces: dict[str, list[AgentMessage]] = defaultdict(list)
        # agent_id → 调用计数
        self._call_counts: dict[str, int] = defaultdict(int)

    def record(self, message: AgentMessage) -> None:
        """
        记录一条 Agent 间消息

        Args:
            message: AgentMessage 实例
        """
        self._traces[message.trace_id].append(message)

        # 更新调用计数
        self._call_counts[message.to_agent] += 1

        # 结构化日志
        log_data = message.to_log_dict()
        if message.error:
            logger.warning(
                f"[A2A] {message.from_agent} → {message.to_agent} "
                f"[{message.intent.value}] ❌ {message.error}",
                extra=log_data,
            )
        else:
            logger.info(
                f"[A2A] {message.from_agent} → {message.to_agent} "
                f"[{message.intent.value}] ✓",
                extra=log_data,
            )

    def get_trace(self, trace_id: str) -> list[AgentMessage]:
        """
        获取一次用户请求的完整 Agent 调用链

        Args:
            trace_id: 追踪ID

        Returns:
            按时间排序的 AgentMessage 列表
        """
        messages = self._traces.get(trace_id, [])
        return sorted(messages, key=lambda m: m.created_at)

    def get_call_counts(self) -> dict[str, int]:
        """获取每个 Agent 的累计调用次数"""
        return dict(self._call_counts)

    def clear_trace(self, trace_id: str) -> None:
        """清除指定追踪链"""
        self._traces.pop(trace_id, None)

    def clear_all(self) -> None:
        """清除所有追踪记录"""
        self._traces.clear()
        self._call_counts.clear()

    def trace_summary(self, trace_id: str) -> dict[str, Any]:
        """
        生成追踪链摘要

        Args:
            trace_id: 追踪ID

        Returns:
            包含 Agent 数量、调用顺序、成功/失败统计的字典
        """
        messages = self.get_trace(trace_id)
        if not messages:
            return {"error": "trace not found"}

        agents_called = list(dict.fromkeys(
            m.to_agent for m in messages
            if m.to_agent and m.intent.value != "response"
        ))

        success_count = sum(1 for m in messages if m.success)
        error_count = sum(1 for m in messages if not m.success)

        return {
            "trace_id": trace_id,
            "total_messages": len(messages),
            "agents_called": agents_called,
            "call_chain": " → ".join(agents_called) if agents_called else "none",
            "success_count": success_count,
            "error_count": error_count,
            "duration": self._estimate_duration(messages),
            "messages": [m.to_log_dict() for m in messages],
        }

    def _estimate_duration(self, messages: list[AgentMessage]) -> str:
        """估算调用链耗时"""
        if len(messages) < 2:
            return "0ms"
        try:
            start = datetime.fromisoformat(messages[0].created_at)
            end = datetime.fromisoformat(messages[-1].created_at)
            delta = (end - start).total_seconds() * 1000
            return f"{delta:.0f}ms"
        except (ValueError, IndexError):
            return "unknown"


# 全局消息总线实例（单例模式）
message_bus = MessageBus()

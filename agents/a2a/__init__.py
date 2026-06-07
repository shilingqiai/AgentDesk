"""
A2A (Agent-to-Agent) Communication Protocol

参考 Microsoft Copilot Studio 的 Agent 间通信模式：
- 标准化的 AgentMessage 格式
- 消息总线（日志 + 追踪）
- 跨 Agent 上下文管理

核心原则 — Single Response Principle:
    只有编排器（Orchestrator）直接响应用户，
    子 Agent 静默返回结构化结果给编排器。
"""

from .protocol import AgentMessage, MessageIntent, AgentRole
from .message_bus import MessageBus
from .context_manager import ContextManager

__all__ = [
    "AgentMessage",
    "MessageIntent",
    "AgentRole",
    "MessageBus",
    "ContextManager",
]

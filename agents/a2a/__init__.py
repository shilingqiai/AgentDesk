"""
A2A (Agent-to-Agent) Communication Protocol

核心原则 — Single Response Principle:
    只有编排器（Orchestrator）直接响应用户，
    子 Agent 静默返回结构化结果给编排器。
"""

from .protocol import AgentMessage

__all__ = ["AgentMessage"]

"""
A2A Protocol — 标准化的 Agent 间消息格式

参考 Microsoft Copilot Studio 的 Agent-to-Agent 通信协议：
- 每个消息有明确的 intent（delegate / query / handoff / notify）
- reply_to_user 标志：False = 子Agent静默返回，True = 编排器可回复用户
- 结构化 payload + 共享 context
"""

from __future__ import annotations

import uuid
from enum import Enum
from dataclasses import dataclass, field
from typing import Any, Optional
from datetime import datetime


class MessageIntent(str, Enum):
    """消息意图类型 — 与 Copilot Studio A2A 协议对齐"""
    DELEGATE = "delegate"     # 编排器委派任务给子Agent
    QUERY = "query"           # Agent间查询（获取信息）
    HANDOFF = "handoff"       # Agent间转交（会话转移）
    NOTIFY = "notify"         # 通知（状态更新、事件）
    RESPONSE = "response"     # 子Agent返回执行结果


class AgentRole(str, Enum):
    """Agent角色"""
    ORCHESTRATOR = "orchestrator"
    SUB_AGENT = "sub_agent"


@dataclass
class AgentMessage:
    """
    标准化的 Agent 间消息格式

    对应 Copilot Studio 中的任务委派上下文：
    - 子Agent收到此消息后执行任务
    - 子Agent返回此消息给编排器
    - reply_to_user=False 确保子Agent不直接回复用户
    """

    # 消息元数据
    message_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    trace_id: str = field(default="")              # 全局追踪ID（一次用户请求共享）
    from_agent: str = field(default="")             # 发送方 Agent ID
    to_agent: str = field(default="")               # 接收方 Agent ID

    # 消息内容
    intent: MessageIntent = MessageIntent.DELEGATE
    payload: dict[str, Any] = field(default_factory=dict)   # 结构化业务数据
    context: dict[str, Any] = field(default_factory=dict)   # 共享上下文

    # Single Response Principle 控制
    reply_to_user: bool = False    # False=子Agent，True=编排器可回复

    # 错误处理
    error: Optional[str] = None
    success: bool = True

    # 时间戳
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())

    @classmethod
    def create_delegation(
        cls,
        from_agent: str,
        to_agent: str,
        payload: dict,
        context: dict = None,
        trace_id: str = "",
    ) -> "AgentMessage":
        """快捷方法：创建委派消息（编排器 → 子Agent）"""
        return cls(
            message_id=str(uuid.uuid4()),
            trace_id=trace_id or str(uuid.uuid4()),
            from_agent=from_agent,
            to_agent=to_agent,
            intent=MessageIntent.DELEGATE,
            payload=payload,
            context=context or {},
            reply_to_user=False,
        )

    @classmethod
    def create_response(
        cls,
        from_agent: str,
        to_agent: str,
        payload: dict,
        original_message: "AgentMessage",
        success: bool = True,
        error: str = None,
    ) -> "AgentMessage":
        """快捷方法：创建响应消息（子Agent → 编排器）"""
        return cls(
            message_id=str(uuid.uuid4()),
            trace_id=original_message.trace_id,
            from_agent=from_agent,
            to_agent=to_agent,
            intent=MessageIntent.RESPONSE,
            payload=payload,
            context=original_message.context,
            reply_to_user=False,  # 子Agent永远不直接回复用户
            success=success,
            error=error,
        )

    @classmethod
    def create_handoff(
        cls,
        from_agent: str,
        to_agent: str,
        payload: dict,
        context: dict = None,
        trace_id: str = "",
    ) -> "AgentMessage":
        """快捷方法：创建转交消息（Agent间会话转移）"""
        return cls(
            message_id=str(uuid.uuid4()),
            trace_id=trace_id or str(uuid.uuid4()),
            from_agent=from_agent,
            to_agent=to_agent,
            intent=MessageIntent.HANDOFF,
            payload=payload,
            context=context or {},
            reply_to_user=False,
        )

    def to_log_dict(self) -> dict:
        """转换为日志友好的字典格式"""
        return {
            "message_id": self.message_id,
            "trace_id": self.trace_id,
            "from_agent": self.from_agent,
            "to_agent": self.to_agent,
            "intent": self.intent.value,
            "payload_keys": list(self.payload.keys()) if self.payload else [],
            "reply_to_user": self.reply_to_user,
            "success": self.success,
            "error": self.error,
            "created_at": self.created_at,
        }

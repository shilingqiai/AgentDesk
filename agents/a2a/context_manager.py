"""
跨Agent上下文管理器

参考 Copilot Studio 的上下文共享机制：
- 编排器与子Agent之间共享用户上下文
- 支持上下文合并、选择性传递
- 防止上下文污染（子Agent不互相泄露内部状态）
"""

from __future__ import annotations

from typing import Any, Optional
from dataclasses import dataclass, field


@dataclass
class SharedContext:
    """
    跨Agent共享的上下文信息

    编排器在委派任务时传递此上下文给子Agent，
    子Agent在返回结果时可追加自己的分析结果。
    """
    # 用户信息
    user_id: str = "default_user"
    session_id: str = "default_session"

    # 对话历史摘要（而非完整历史，避免上下文膨胀）
    conversation_summary: str = ""

    # 当前请求信息
    original_user_input: str = ""
    detected_intent: str = ""
    urgency: str = "medium"  # high | medium | low

    # 业务上下文（从之前Agent调用中积累）
    business_context: dict[str, Any] = field(default_factory=dict)

    # 前序Agent结果（用于链式调用）
    previous_agent_results: dict[str, Any] = field(default_factory=dict)

    # 元数据
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        """转换为字典（传递给AgentMessage.context）"""
        return {
            "user_id": self.user_id,
            "session_id": self.session_id,
            "conversation_summary": self.conversation_summary,
            "original_user_input": self.original_user_input,
            "detected_intent": self.detected_intent,
            "urgency": self.urgency,
            "business_context": self.business_context,
            "previous_agent_results": self.previous_agent_results,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "SharedContext":
        """从字典恢复"""
        return cls(
            user_id=data.get("user_id", "default_user"),
            session_id=data.get("session_id", "default_session"),
            conversation_summary=data.get("conversation_summary", ""),
            original_user_input=data.get("original_user_input", ""),
            detected_intent=data.get("detected_intent", ""),
            urgency=data.get("urgency", "medium"),
            business_context=data.get("business_context", {}),
            previous_agent_results=data.get("previous_agent_results", {}),
            metadata=data.get("metadata", {}),
        )


class ContextManager:
    """
    上下文管理器

    职责：
    1. 为每次编排调用创建和维护 SharedContext
    2. 合并子Agent返回的上下文增量
    3. 控制上下文大小，防止膨胀
    """

    def __init__(self, max_context_size: int = 5000):
        self.max_context_size = max_context_size
        # trace_id → SharedContext
        self._contexts: dict[str, SharedContext] = {}

    def create_context(
        self,
        trace_id: str,
        user_input: str,
        user_id: str = "default_user",
        session_id: str = "default_session",
    ) -> SharedContext:
        """
        创建新的共享上下文

        Args:
            trace_id: 追踪ID
            user_input: 用户原始输入
            user_id: 用户ID
            session_id: 会话ID

        Returns:
            SharedContext 实例
        """
        ctx = SharedContext(
            user_id=user_id,
            session_id=session_id,
            original_user_input=user_input,
        )
        self._contexts[trace_id] = ctx
        return ctx

    def get_context(self, trace_id: str) -> Optional[SharedContext]:
        """获取指定追踪的上下文"""
        return self._contexts.get(trace_id)

    def update_context(self, trace_id: str, **kwargs) -> None:
        """
        增量更新上下文

        Args:
            trace_id: 追踪ID
            **kwargs: 要更新的字段
        """
        ctx = self._contexts.get(trace_id)
        if ctx:
            for key, value in kwargs.items():
                if hasattr(ctx, key):
                    setattr(ctx, key, value)

    def merge_agent_result(self, trace_id: str, agent_id: str, result: dict) -> None:
        """
        合并子Agent的返回结果到上下文

        Args:
            trace_id: 追踪ID
            agent_id: Agent ID
            result: Agent返回的payload
        """
        ctx = self._contexts.get(trace_id)
        if ctx:
            ctx.previous_agent_results[agent_id] = result

    def clear_context(self, trace_id: str) -> None:
        """清除上下文"""
        self._contexts.pop(trace_id, None)

    def clear_all(self) -> None:
        """清除所有上下文"""
        self._contexts.clear()


# 全局上下文管理器实例
context_manager = ContextManager()

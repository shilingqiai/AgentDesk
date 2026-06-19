"""
AgentContext — 单一身份源 (Single Source of Truth)

所有 Agent / Node / API 路径通过 AgentContext 传递用户身份，
禁止裸字符串 fallback（如 "web_user" / "" / None）。

用法:
    ctx = AgentContext.from_state(state)       # 从 LangGraph State
    ctx = AgentContext.from_request(request)   # 从 FastAPI Request
    ctx = AgentContext(user_name="张三", ...)  # 直接构造
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

logger = logging.getLogger("agent.context")


@dataclass
class AgentContext:
    """
    不可变身份上下文。

    user_name 和 user_id 在当前原型阶段可以相同（都用姓名），
    生产环境对接 SSO 后 user_id 改为工号/UPN。
    """

    user_name: str
    user_id: str = ""
    role: str = "employee"
    thread_id: str = ""

    def __post_init__(self):
        # user_id 默认等于 user_name（原型阶段）
        if not self.user_id:
            self.user_id = self.user_name

    @property
    def is_admin(self) -> bool:
        return self.role == "admin"

    @property
    def is_authenticated(self) -> bool:
        """拒绝空身份写入 DB"""
        return bool(self.user_name) and self.user_name not in ("", "anonymous", "web_user")

    def validate(self, *, allow_anonymous: bool = False) -> AgentContext:
        """
        验证身份有效性。不合法时抛出 ValueError。

        Args:
            allow_anonymous: 是否允许匿名用户（默认不允许）
        """
        if not allow_anonymous and not self.is_authenticated:
            raise ValueError(
                f"AgentContext 身份无效: user_name={self.user_name!r}。"
                f"拒绝写入数据库。"
            )
        return self

    # ── 工厂方法 ──

    @classmethod
    def from_state(cls, state: dict) -> AgentContext:
        """从 LangGraph TicketState 构建"""
        return cls(
            user_name=state.get("user_name", ""),
            user_id=state.get("user_name", ""),  # 原型: user_id = user_name
            role=state.get("role", "employee"),
            thread_id=state.get("thread_id", ""),
        )

    @classmethod
    def from_request(cls, request) -> AgentContext:
        """从 FastAPI Request 构建（兼容 web.routes 和 api.tickets）"""
        user_name = getattr(request.state, "user_name", "") or ""
        role = getattr(request.state, "role", "employee") or "employee"
        return cls(
            user_name=user_name,
            user_id=user_name,
            role=role,
        )

    @classmethod
    def from_message_payload(cls, payload: dict) -> AgentContext:
        """从 AgentMessage.payload 构建（A2A 委派消息）"""
        user_name = payload.get("user_id", "") or payload.get("user_name", "")
        return cls(
            user_name=user_name,
            user_id=user_name,
            role=payload.get("role", "employee"),
        )

    def to_dict(self) -> dict:
        return {
            "user_name": self.user_name,
            "user_id": self.user_id,
            "role": self.role,
            "thread_id": self.thread_id,
        }

"""
EventBus — 轻量异步事件总线

实现发布/订阅模式，审批/通知/审计/仪表盘通过事件解耦。

设计原则:
- 零外部依赖，纯 Python asyncio
- Handler 异常隔离: 一个 handler 崩溃不影响其他订阅者
- 同步 emit: 所有 handler 并发执行，emit 等待全部完成
- 事件类型标准化: 所有事件使用命名约定

使用方式:
    from services.event_bus import EventBus

    EventBus.subscribe("ticket.created", my_handler)
    await EventBus.emit("ticket.created", ticket_id=42, ticket_number="TK-...")
"""

from __future__ import annotations

import asyncio
import logging
from collections import defaultdict
from typing import Callable, Awaitable

logger = logging.getLogger("event.bus")

# ── 事件类型常量 ──────────────────────────────────────

class EventType:
    """事件类型枚举"""
    TICKET_CREATED        = "ticket.created"
    TICKET_STATUS_CHANGED = "ticket.status_changed"
    APPROVAL_STEP_APPROVED  = "approval.step_approved"
    APPROVAL_STEP_REJECTED  = "approval.step_rejected"
    APPROVAL_COMPLETED    = "approval.completed"
    APPROVAL_REJECTED     = "approval.rejected"
    SLA_BREACHED          = "sla.breached"


Handler = Callable[..., Awaitable[None]]


class EventBus:
    """
    异步事件总线 — 全局单例模式。

    Handler 签名: async def handler(event_type: str, **data) -> None
    """

    _handlers: dict[str, list[Handler]] = defaultdict(list)

    @classmethod
    def subscribe(cls, event_type: str, handler: Handler) -> None:
        """
        订阅事件。

        Args:
            event_type: 事件类型字符串（推荐使用 EventType 常量）
            handler:   async callable(event_type, **data)
        """
        cls._handlers[event_type].append(handler)
        logger.debug(f"[EventBus] 订阅: {event_type} → {handler.__name__}")

    @classmethod
    def unsubscribe(cls, event_type: str, handler: Handler) -> None:
        """取消订阅"""
        if handler in cls._handlers[event_type]:
            cls._handlers[event_type].remove(handler)

    @classmethod
    async def emit(cls, event_type: str, **data) -> None:
        """
        发射事件 — 所有订阅 handler 并发执行。

        Handler 异常被捕获并记录日志，不会传播到调用方。
        """
        handlers = cls._handlers.get(event_type, [])
        if not handlers:
            return

        async def _safe_invoke(handler: Handler):
            try:
                await handler(event_type=event_type, **data)
            except Exception as e:
                logger.error(
                    f"[EventBus] Handler '{handler.__name__}' "
                    f"处理事件 '{event_type}' 失败: {e}",
                    exc_info=True,
                )

        tasks = [_safe_invoke(h) for h in handlers]
        await asyncio.gather(*tasks)

    @classmethod
    def clear(cls) -> None:
        """清空所有订阅（仅用于测试）"""
        cls._handlers.clear()

    @classmethod
    def subscriber_count(cls, event_type: str = None) -> int:
        """获取订阅者数量（用于测试和监控）"""
        if event_type:
            return len(cls._handlers.get(event_type, []))
        return sum(len(v) for v in cls._handlers.values())

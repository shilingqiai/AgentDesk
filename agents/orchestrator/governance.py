"""
治理与审计追踪 — 参考 Microsoft Copilot Studio 的 Governance 模型

Microsoft Copilot Studio 的治理能力：
- Agent 身份管理（Entra-based identities）
- 数据访问边界（parent agent 不能绕过子agent的限制）
- 审计日志（separate transcripts per agent）
- 遥测关联（parent-child session identifiers）

本模块实现：
1. 审计日志记录
2. Agent 调用链追踪
3. 关键决策点日志
4. 合规检查
"""

from __future__ import annotations

import logging
import json
from datetime import datetime
from enum import Enum
from dataclasses import dataclass, field, asdict
from typing import Any, Optional

logger = logging.getLogger("orchestrator.governance")


class AuditEventType(str, Enum):
    """审计事件类型"""
    # 编排事件
    ORCHESTRATION_START = "orchestration_start"
    ORCHESTRATION_END = "orchestration_end"
    # Agent 事件
    AGENT_CALLED = "agent_called"
    AGENT_RESPONDED = "agent_responded"
    AGENT_FAILED = "agent_failed"
    # 控制事件
    HUMAN_REVIEW_REQUESTED = "human_review_requested"
    HUMAN_REVIEW_COMPLETED = "human_review_completed"
    HUMAN_REVIEW_TIMEOUT = "human_review_timeout"
    ESCALATION = "escalation"
    FALLBACK = "fallback"
    # 数据事件
    DATA_ACCESS = "data_access"
    DATA_MODIFICATION = "data_modification"


@dataclass
class AuditEvent:
    """审计事件"""
    event_id: str
    event_type: AuditEventType
    trace_id: str
    agent_id: str = "orchestrator"
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    details: dict[str, Any] = field(default_factory=dict)
    user_id: str = "anonymous"
    session_id: str = ""

    def to_log_entry(self) -> str:
        """转换为日志字符串"""
        return json.dumps(asdict(self), ensure_ascii=False, default=str)


class AuditTrail:
    """
    审计追踪

    记录编排过程中的所有关键事件。
    参考 Copilot Studio 的 per-agent transcript 模式。

    使用方式：
        audit = AuditTrail()
        audit.record(AuditEvent(...))
        events = audit.get_events_by_trace(trace_id)
    """

    def __init__(self, max_events_per_trace: int = 1000):
        self.max_events_per_trace = max_events_per_trace
        # trace_id → [AuditEvent]
        self._traces: dict[str, list[AuditEvent]] = {}
        # event_id → AuditEvent (全局索引)
        self._events: dict[str, AuditEvent] = {}

    def record(self, event: AuditEvent) -> None:
        """记录审计事件"""
        import uuid
        if not event.event_id:
            event.event_id = str(uuid.uuid4())

        # 按 trace 分组
        if event.trace_id not in self._traces:
            self._traces[event.trace_id] = []
        self._traces[event.trace_id].append(event)

        # 限制每个 trace 的事件数量
        if len(self._traces[event.trace_id]) > self.max_events_per_trace:
            self._traces[event.trace_id] = self._traces[event.trace_id][
                -self.max_events_per_trace:
            ]

        # 全局索引
        self._events[event.event_id] = event

        # 输出到结构化日志
        logger.info(
            f"[Audit] {event.event_type.value} | "
            f"trace={event.trace_id[:8]}... | "
            f"agent={event.agent_id}",
            extra={"audit_event": event.to_log_entry()},
        )

    def get_events_by_trace(self, trace_id: str) -> list[AuditEvent]:
        """获取指定追踪链的所有审计事件"""
        return self._traces.get(trace_id, [])

    def get_event(self, event_id: str) -> Optional[AuditEvent]:
        """获取指定事件"""
        return self._events.get(event_id)

    def get_trace_summary(self, trace_id: str) -> dict[str, Any]:
        """生成追踪链审计摘要"""
        events = self.get_events_by_trace(trace_id)
        if not events:
            return {"error": "no events found"}

        # 按类型统计
        type_counts = {}
        agent_calls = []
        human_reviews = []
        errors = []

        for e in events:
            type_counts[e.event_type.value] = type_counts.get(e.event_type.value, 0) + 1

            if e.event_type == AuditEventType.AGENT_CALLED:
                agent_calls.append(e.details.get("target_agent", "unknown"))
            elif e.event_type == AuditEventType.HUMAN_REVIEW_REQUESTED:
                human_reviews.append(e.details)
            elif e.event_type in (AuditEventType.AGENT_FAILED, AuditEventType.ESCALATION):
                errors.append(e.details.get("error", ""))

        # 计算耗时
        duration = "unknown"
        if len(events) >= 2:
            try:
                start = datetime.fromisoformat(events[0].timestamp)
                end = datetime.fromisoformat(events[-1].timestamp)
                duration = f"{(end - start).total_seconds():.1f}s"
            except (ValueError, IndexError):
                pass

        return {
            "trace_id": trace_id,
            "total_events": len(events),
            "event_types": type_counts,
            "agent_calls": agent_calls,
            "human_reviews": len(human_reviews),
            "errors": len(errors),
            "duration": duration,
            "timeline": [
                {
                    "type": e.event_type.value,
                    "agent": e.agent_id,
                    "time": e.timestamp,
                    "details": e.details,
                }
                for e in events
            ],
        }

    def clear_trace(self, trace_id: str) -> None:
        """清除追踪链"""
        events = self._traces.pop(trace_id, [])
        for e in events:
            self._events.pop(e.event_id, None)

    def clear_all(self) -> None:
        """清除所有审计记录"""
        self._traces.clear()
        self._events.clear()


class GovernanceChecker:
    """
    合规检查器

    在关键操作节点进行合规检查：
    - 验证Agent权限
    - 检查数据访问边界
    - 确认必要的人工审核已完成
    """

    @staticmethod
    def check_agent_permission(agent_id: str, action: str) -> bool:
        """
        检查Agent是否有权限执行指定操作

        规则：
        - 子Agent不得执行编排操作
        - 编排器不得直接修改数据（必须通过子Agent）
        - 效能分析Agent不能创建工单
        """
        # 子Agent不能做编排
        if action.startswith("orchestrate") and agent_id != "orchestrator":
            logger.warning(f"[Governance] 权限拒绝: {agent_id} 不能执行 {action}")
            return False

        # 效能分析Agent是只读的
        if agent_id == "analytics" and action in ("create_ticket", "modify_sla", "delete"):
            logger.warning(f"[Governance] 权限拒绝: {agent_id} 不能执行写操作 {action}")
            return False

        return True

    @staticmethod
    def check_data_boundary(agent_id: str, requested_domain: str) -> bool:
        """
        检查Agent是否在允许的数据域内

        每个Agent的知识域是声明式的、不重叠的。
        """
        from agents.orchestrator.agent_registry import agent_registry

        decl = agent_registry.get_declaration(agent_id)
        if decl is None:
            return False

        if requested_domain not in decl.knowledge_domains:
            logger.warning(
                f"[Governance] 数据域越界: {agent_id} 请求访问 "
                f"'{requested_domain}'，但知识域为 {decl.knowledge_domains}"
            )
            return False

        return True


# 全局审计追踪实例
audit_trail = AuditTrail()
governance_checker = GovernanceChecker()

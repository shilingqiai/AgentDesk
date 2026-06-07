"""
编排器遥测 — 性能监控与可观测性

参考 Microsoft Copilot Studio 的可观测性模式：
- Agent 365 monitoring
- Custom metrics with role-scoped analytics
- Per-agent performance tracking

本模块实现：
1. 编排调用指标收集
2. Agent 性能统计
3. 意图分类准确率追踪
4. Human-in-the-Loop 触发率
"""

from __future__ import annotations

import time
import logging
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Optional
from datetime import datetime

logger = logging.getLogger("orchestrator.telemetry")


@dataclass
class AgentMetrics:
    """单个Agent的性能指标"""
    agent_id: str
    total_calls: int = 0
    successful_calls: int = 0
    failed_calls: int = 0
    total_duration_ms: float = 0.0
    avg_duration_ms: float = 0.0
    last_called_at: Optional[str] = None

    def record_call(self, success: bool, duration_ms: float):
        """记录一次调用"""
        self.total_calls += 1
        if success:
            self.successful_calls += 1
        else:
            self.failed_calls += 1
        self.total_duration_ms += duration_ms
        self.avg_duration_ms = self.total_duration_ms / self.total_calls
        self.last_called_at = datetime.now().isoformat()

    @property
    def success_rate(self) -> float:
        """成功率"""
        if self.total_calls == 0:
            return 1.0
        return self.successful_calls / self.total_calls


@dataclass
class OrchestrationMetrics:
    """编排器总体指标"""
    total_orchestrations: int = 0
    total_duration_ms: float = 0.0
    avg_duration_ms: float = 0.0
    avg_agents_per_orchestration: float = 0.0
    human_review_triggers: int = 0
    fallback_triggers: int = 0
    escalation_triggers: int = 0

    # 意图分布
    intent_distribution: dict[str, int] = field(default_factory=lambda: defaultdict(int))

    # 紧急度分布
    urgency_distribution: dict[str, int] = field(default_factory=lambda: defaultdict(int))


class TelemetryCollector:
    """
    遥测收集器

    职责：
    1. 收集编排调用指标
    2. 追踪Agent性能
    3. 统计意图分类分布
    4. 生成可观测性报告

    使用方式：
        telemetry = TelemetryCollector()
        telemetry.start_orchestration(trace_id)
        # ... 执行编排 ...
        telemetry.end_orchestration(trace_id, success=True)
        report = telemetry.get_report()
    """

    def __init__(self):
        # trace_id → 开始时间戳
        self._active_orchestrations: dict[str, float] = {}
        # agent_id → AgentMetrics
        self._agent_metrics: dict[str, AgentMetrics] = defaultdict(
            lambda: AgentMetrics(agent_id="")
        )
        # 编排指标
        self.orchestration = OrchestrationMetrics()

        # 分类准确率追踪
        self._classification_feedback: list[dict] = []

    def start_orchestration(self, trace_id: str) -> None:
        """开始一次编排调用"""
        self._active_orchestrations[trace_id] = time.time()
        self.orchestration.total_orchestrations += 1
        logger.debug(f"[Telemetry] 编排开始: {trace_id[:8]}...")

    def end_orchestration(
        self,
        trace_id: str,
        success: bool = True,
        intent: str = "",
        urgency: str = "",
        agents_called: int = 0,
        human_review: bool = False,
        escalated: bool = False,
        fallback: bool = False,
    ) -> None:
        """结束一次编排调用"""
        start_time = self._active_orchestrations.pop(trace_id, None)
        if start_time is None:
            return

        duration_ms = (time.time() - start_time) * 1000
        self.orchestration.total_duration_ms += duration_ms
        self.orchestration.avg_duration_ms = (
            self.orchestration.total_duration_ms
            / self.orchestration.total_orchestrations
        )
        self.orchestration.avg_agents_per_orchestration = (
            (self.orchestration.avg_agents_per_orchestration
             * (self.orchestration.total_orchestrations - 1)
             + agents_called)
            / self.orchestration.total_orchestrations
        )

        # 分布统计
        if intent:
            self.orchestration.intent_distribution[intent] += 1
        if urgency:
            self.orchestration.urgency_distribution[urgency] += 1

        # 控制流统计
        if human_review:
            self.orchestration.human_review_triggers += 1
        if escalated:
            self.orchestration.escalation_triggers += 1
        if fallback:
            self.orchestration.fallback_triggers += 1

        logger.info(
            f"[Telemetry] 编排完成: {trace_id[:8]}... "
            f"耗时={duration_ms:.0f}ms, 成功={success}, "
            f"Agent数={agents_called}"
        )

    def record_agent_call(
        self, agent_id: str, success: bool, duration_ms: float,
    ) -> None:
        """记录Agent调用"""
        metrics = self._agent_metrics[agent_id]
        if metrics.agent_id == "":
            metrics.agent_id = agent_id
        metrics.record_call(success, duration_ms)

    def record_classification_feedback(
        self,
        user_input: str,
        predicted_intent: str,
        actual_intent: str,
    ) -> None:
        """
        记录分类反馈（用于准确率追踪）

        Args:
            user_input: 用户输入
            predicted_intent: AI预测的意图
            actual_intent: 实际意图（来自用户反馈或后续确认）
        """
        self._classification_feedback.append({
            "user_input": user_input[:200],
            "predicted": predicted_intent,
            "actual": actual_intent,
            "timestamp": datetime.now().isoformat(),
        })

    def get_agent_metrics(self, agent_id: str) -> Optional[AgentMetrics]:
        """获取指定Agent的指标"""
        return self._agent_metrics.get(agent_id)

    def get_all_agent_metrics(self) -> dict[str, AgentMetrics]:
        """获取所有Agent的指标"""
        return dict(self._agent_metrics)

    def get_classification_accuracy(self) -> float:
        """获取意图分类准确率"""
        if not self._classification_feedback:
            return 1.0
        correct = sum(
            1 for f in self._classification_feedback
            if f["predicted"] == f["actual"]
        )
        return correct / len(self._classification_feedback)

    def get_report(self) -> dict[str, Any]:
        """生成完整的遥测报告"""
        return {
            "orchestration": {
                "total": self.orchestration.total_orchestrations,
                "avg_duration_ms": round(self.orchestration.avg_duration_ms, 1),
                "avg_agents_per_call": round(
                    self.orchestration.avg_agents_per_orchestration, 1
                ),
                "human_review_rate": (
                    self.orchestration.human_review_triggers
                    / max(self.orchestration.total_orchestrations, 1)
                ),
                "escalation_rate": (
                    self.orchestration.escalation_triggers
                    / max(self.orchestration.total_orchestrations, 1)
                ),
                "fallback_rate": (
                    self.orchestration.fallback_triggers
                    / max(self.orchestration.total_orchestrations, 1)
                ),
            },
            "intent_distribution": dict(self.orchestration.intent_distribution),
            "urgency_distribution": dict(self.orchestration.urgency_distribution),
            "agents": {
                agent_id: {
                    "total_calls": m.total_calls,
                    "success_rate": round(m.success_rate, 2),
                    "avg_duration_ms": round(m.avg_duration_ms, 1),
                    "last_called": m.last_called_at,
                }
                for agent_id, m in self._agent_metrics.items()
            },
            "classification_accuracy": round(
                self.get_classification_accuracy(), 2
            ),
        }

    def reset(self) -> None:
        """重置所有指标"""
        self._active_orchestrations.clear()
        self._agent_metrics.clear()
        self.orchestration = OrchestrationMetrics()
        self._classification_feedback.clear()


# 全局遥测收集器实例
telemetry = TelemetryCollector()

"""
三层控制模型 — 参考 Microsoft Copilot Studio 的控制架构

Microsoft Copilot Studio 的三层决策模型：
┌─────────────────────────────────┐
│   AI Orchestrator Layer         │  ← LLM自由决策
├─────────────────────────────────┤
│   Hybrid (Intercept) Layer      │  ← AI在边界内操作，人工检查点
├─────────────────────────────────┤
│   Deterministic Layer           │  ← 纯规则，无AI参与
└─────────────────────────────────┘

本模块实现三层判断逻辑，确保：
- 高风险操作必须经过 Hybrid 层的人工确认
- 简单的确定性判断不走 LLM（节省成本和延迟）
- AI 决策在明确的边界内运行
"""

from __future__ import annotations

import logging
from enum import Enum
from dataclasses import dataclass
from typing import Any, Optional
from datetime import datetime, time

logger = logging.getLogger("orchestrator.control")


class ControlLevel(str, Enum):
    """控制层级"""
    AI = "ai_orchestrator"         # LLM自由决策
    HYBRID = "hybrid_intercept"    # AI+规则+人工检查点
    DETERMINISTIC = "deterministic" # 纯规则，无AI


class ActionRisk(str, Enum):
    """操作风险等级"""
    LOW = "low"         # 纯信息查询，无需确认
    MEDIUM = "medium"   # 修改数据，建议确认
    HIGH = "high"       # 高风险操作，必须确认
    CRITICAL = "critical"  # 关键操作，多重确认


@dataclass
class ControlDecision:
    """控制决策结果"""
    level: ControlLevel
    risk: ActionRisk
    needs_human_review: bool
    reason: str
    allowed: bool = True


class DeterministicRules:
    """
    确定性规则层 (Deterministic Layer)

    纯 Python 规则判断，不依赖 LLM。
    所有判断必须明确、可预测、快速。
    """

    # 营业时间（9:00-22:00）
    BUSINESS_START = 9
    BUSINESS_END = 22

    # 高优SLA阈值
    HIGH_PRIORITY_SLA_HOURS = 2
    MEDIUM_PRIORITY_SLA_HOURS = 4
    LOW_PRIORITY_SLA_HOURS = 8

    @staticmethod
    def is_business_hours() -> bool:
        """判断当前是否在营业时间内"""
        now = datetime.now()
        hour = now.hour
        return DeterministicRules.BUSINESS_START <= hour < DeterministicRules.BUSINESS_END

    @staticmethod
    def is_weekend() -> bool:
        """判断是否为周末"""
        return datetime.now().weekday() >= 5

    @staticmethod
    def validate_required_fields(data: dict, required: list[str]) -> list[str]:
        """
        验证必填字段

        Args:
            data: 数据字典
            required: 必填字段列表

        Returns:
            缺失字段列表（空列表表示验证通过）
        """
        return [f for f in required if not data.get(f)]

    @staticmethod
    def get_sla_hours(priority: str) -> int:
        """根据优先级返回 SLA 小时数"""
        sla_map = {
            "P0": DeterministicRules.HIGH_PRIORITY_SLA_HOURS,
            "P1": DeterministicRules.HIGH_PRIORITY_SLA_HOURS,
            "high": DeterministicRules.HIGH_PRIORITY_SLA_HOURS,
            "P2": DeterministicRules.MEDIUM_PRIORITY_SLA_HOURS,
            "medium": DeterministicRules.MEDIUM_PRIORITY_SLA_HOURS,
            "P3": DeterministicRules.LOW_PRIORITY_SLA_HOURS,
            "low": DeterministicRules.LOW_PRIORITY_SLA_HOURS,
        }
        return sla_map.get(priority.lower(), DeterministicRules.LOW_PRIORITY_SLA_HOURS)

    @staticmethod
    def check_out_of_hours_risk() -> ControlDecision:
        """检查非工作时间风险"""
        if not DeterministicRules.is_business_hours():
            return ControlDecision(
                level=ControlLevel.DETERMINISTIC,
                risk=ActionRisk.MEDIUM,
                needs_human_review=True,
                reason="当前为非营业时间 (9:00-22:00)，操作可能需要延迟处理",
            )
        return ControlDecision(
            level=ControlLevel.DETERMINISTIC,
            risk=ActionRisk.LOW,
            needs_human_review=False,
            reason="营业时间内，无需特殊处理",
        )


class HybridInterceptRules:
    """
    混合拦截层 (Hybrid Intercept Layer)

    AI 在边界内操作，关键节点由人工检查。
    组合使用规则判断和 LLM 分析。
    """

    @staticmethod
    def needs_human_review(
        intent: str,
        urgency: str,
        action_type: str,
        confidence: float,
    ) -> ControlDecision:
        """
        判断是否需要人工审核

        Args:
            intent: 意图类别
            urgency: 紧急度
            action_type: 操作类型 (create_ticket / modify_sla / delete / query)
            confidence: AI置信度

        Returns:
            ControlDecision
        """
        # 纯查询操作 → AI层，不需要审核
        if action_type in ("query", "search", "analyze"):
            return ControlDecision(
                level=ControlLevel.AI,
                risk=ActionRisk.LOW,
                needs_human_review=False,
                reason="查询类操作，AI自由处理",
            )

        # 低置信度 + 非查询 → 需要人工确认
        if confidence < 0.5 and action_type != "query":
            return ControlDecision(
                level=ControlLevel.HYBRID,
                risk=ActionRisk.MEDIUM,
                needs_human_review=True,
                reason=f"AI置信度较低 ({confidence:.0%})，建议人工确认",
            )

        # 创建高优工单 → 必须确认
        if action_type == "create_ticket" and urgency == "high":
            return ControlDecision(
                level=ControlLevel.HYBRID,
                risk=ActionRisk.HIGH,
                needs_human_review=True,
                reason="创建高优先级工单需要人工确认",
            )

        # SLA修改 → 必须确认
        if action_type == "modify_sla":
            return ControlDecision(
                level=ControlLevel.HYBRID,
                risk=ActionRisk.CRITICAL,
                needs_human_review=True,
                reason="SLA修改为关键操作，需要多重确认",
            )

        # 删除操作 → 必须确认
        if action_type == "delete":
            return ControlDecision(
                level=ControlLevel.HYBRID,
                risk=ActionRisk.CRITICAL,
                needs_human_review=True,
                reason="删除操作不可逆，需要人工确认",
            )

        # 非工作时间 + 紧急请求 → 需要确认
        if urgency == "high" and not DeterministicRules.is_business_hours():
            return ControlDecision(
                level=ControlLevel.HYBRID,
                risk=ActionRisk.MEDIUM,
                needs_human_review=True,
                reason="非工作时间的紧急请求，建议确认",
            )

        # 默认：AI可以处理
        return ControlDecision(
            level=ControlLevel.AI,
            risk=ActionRisk.LOW,
            needs_human_review=False,
            reason="标准操作，AI可自主处理",
        )


class ControlLayerManager:
    """
    控制层管理器

    统一管理三层控制的决策流程：
    1. 先经过 Deterministic 层（快速规则判断）
    2. 再经过 Hybrid 层（组合判断）
    3. 其余交给 AI 层

    使用方式：
        manager = ControlLayerManager()
        decision = manager.evaluate(intent, urgency, action_type, confidence)
        if decision.needs_human_review:
            # 进入 Human-in-the-Loop
            ...
    """

    def __init__(self):
        self.deterministic = DeterministicRules()
        self.hybrid = HybridInterceptRules()

    def evaluate(
        self,
        intent: str,
        urgency: str,
        action_type: str = "query",
        confidence: float = 0.5,
    ) -> ControlDecision:
        """
        评估操作的控制层级

        按优先级从低到高：
        1. Deterministic 规则（最快）
        2. Hybrid 规则（需要时）
        3. 默认 AI 层

        Args:
            intent: 意图类别
            urgency: 紧急度
            action_type: 操作类型
            confidence: AI置信度

        Returns:
            ControlDecision
        """
        # Layer 1: Deterministic 检查
        if not self.deterministic.is_business_hours():
            out_of_hours = self.deterministic.check_out_of_hours_risk()
            if out_of_hours.needs_human_review and urgency == "high":
                logger.info(f"[Control] Deterministic层拦截: {out_of_hours.reason}")
                return out_of_hours

        # Layer 2: Hybrid 检查
        hybrid_decision = self.hybrid.needs_human_review(
            intent, urgency, action_type, confidence,
        )
        if hybrid_decision.needs_human_review:
            logger.info(f"[Control] Hybrid层拦截: {hybrid_decision.reason}")
            return hybrid_decision

        # Layer 3: AI 层自由决策
        logger.debug(f"[Control] AI层自由决策: intent={intent}, urgency={urgency}")
        return ControlDecision(
            level=ControlLevel.AI,
            risk=ActionRisk.LOW,
            needs_human_review=False,
            reason="AI层自主处理",
        )

    def get_escalation_reason(self, agent_results: dict) -> Optional[str]:
        """
        判断是否需要升级的理由

        Args:
            agent_results: 子Agent返回结果

        Returns:
            升级理由，None表示不需要升级
        """
        if not agent_results:
            return "所有Agent均未执行"

        all_failed = all(
            not r.get("success", False)
            for r in agent_results.values()
        )
        if all_failed:
            return "所有Agent执行失败"

        return None


# 全局控制层管理器
control_manager = ControlLayerManager()

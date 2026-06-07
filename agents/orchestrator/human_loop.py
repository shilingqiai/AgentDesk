"""
Human-in-the-Loop 实现

参考 Microsoft Copilot Studio 的 Human-in-the-Loop 模式：
- 高风险操作暂停执行，等待人工确认
- 支持 approve / reject / modify 三种决策
- 超时自动降级处理

触发场景：
- 创建高优先级工单 (P0/P1)
- SLA超时自动升级
- 非工作时间提交紧急请求
- 匹配到技能不最优的工程师
- AI置信度低于阈值
"""

from __future__ import annotations

import logging
from enum import Enum
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Optional, Callable

logger = logging.getLogger("orchestrator.human_loop")


class HumanDecision(str, Enum):
    """人工决策结果"""
    APPROVED = "approved"
    REJECTED = "rejected"
    MODIFIED = "modified"  # 修改后批准
    TIMEOUT = "timeout"    # 超时自动处理


@dataclass
class ReviewRequest:
    """
    人工审核请求

    包含足够的信息让审核者做出决策。
    """
    request_id: str
    intent: str
    urgency: str
    action_type: str
    summary: str
    details: dict[str, Any] = field(default_factory=dict)
    agent_results: dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    expires_at: str = field(default_factory=lambda: (
        datetime.now() + timedelta(minutes=5)
    ).isoformat())


@dataclass
class ReviewResult:
    """人工审核结果"""
    request_id: str
    decision: HumanDecision
    reviewer: str = "system"
    comment: str = ""
    modifications: dict[str, Any] = field(default_factory=dict)
    decided_at: str = field(default_factory=lambda: datetime.now().isoformat())


class HumanInTheLoop:
    """
    Human-in-the-Loop 管理器

    职责：
    1. 创建审核请求
    2. 管理审核队列
    3. 处理审核结果
    4. 超时自动降级

    使用方式：
        hitl = HumanInTheLoop()
        request = hitl.create_review_request(intent, urgency, action_type, summary)
        # ... 等待人工决策 ...
        result = hitl.process_decision(request.request_id, HumanDecision.APPROVED)
    """

    def __init__(self, timeout_minutes: int = 5):
        self.timeout_minutes = timeout_minutes
        # request_id → ReviewRequest
        self._pending: dict[str, ReviewRequest] = {}
        # request_id → ReviewResult
        self._decisions: dict[str, ReviewResult] = {}

    def create_review_request(
        self,
        intent: str,
        urgency: str,
        action_type: str,
        summary: str,
        details: dict = None,
        agent_results: dict = None,
    ) -> ReviewRequest:
        """创建审核请求"""
        import uuid
        request = ReviewRequest(
            request_id=str(uuid.uuid4()),
            intent=intent,
            urgency=urgency,
            action_type=action_type,
            summary=summary,
            details=details or {},
            agent_results=agent_results or {},
        )
        self._pending[request.request_id] = request
        logger.info(
            f"[HumanLoop] 创建审核请求 {request.request_id[:8]}...: "
            f"{action_type} (紧急度={urgency})"
        )
        return request

    def process_decision(
        self,
        request_id: str,
        decision: HumanDecision,
        reviewer: str = "user",
        comment: str = "",
        modifications: dict = None,
    ) -> ReviewResult:
        """处理人工决策"""
        request = self._pending.pop(request_id, None)
        if request is None:
            logger.warning(f"[HumanLoop] 审核请求 {request_id} 不存在或已过期")
            return ReviewResult(
                request_id=request_id,
                decision=HumanDecision.TIMEOUT,
                reviewer="system",
                comment="请求已过期",
            )

        result = ReviewResult(
            request_id=request_id,
            decision=decision,
            reviewer=reviewer,
            comment=comment,
            modifications=modifications or {},
        )
        self._decisions[request_id] = result

        logger.info(
            f"[HumanLoop] 审核完成 {request_id[:8]}...: "
            f"{decision.value} by {reviewer}"
        )
        return result

    def check_timeout(self, request_id: str) -> Optional[ReviewResult]:
        """
        检查审核请求是否超时

        超时策略：
        - 高紧急度 → 自动批准（不阻塞紧急处理）
        - 中/低紧急度 → 自动拒绝（安全优先）

        Returns:
            超时则返回自动决策结果，否则返回 None
        """
        request = self._pending.get(request_id)
        if request is None:
            return None

        expires = datetime.fromisoformat(request.expires_at)
        if datetime.now() > expires:
            # 超时自动决策
            if request.urgency == "high":
                decision = HumanDecision.APPROVED
                comment = "超时自动批准（高紧急度）"
            else:
                decision = HumanDecision.REJECTED
                comment = "超时自动拒绝（安全优先）"

            logger.warning(f"[HumanLoop] 审核超时 {request_id[:8]}...: {comment}")

            return self.process_decision(
                request_id=request_id,
                decision=decision,
                reviewer="system_timeout",
                comment=comment,
            )

        return None

    def get_pending_count(self) -> int:
        """获取待审核数量"""
        # 先清理超时的
        for req_id in list(self._pending.keys()):
            self.check_timeout(req_id)
        return len(self._pending)

    def is_pending(self, request_id: str) -> bool:
        """检查是否有待处理的审核"""
        return request_id in self._pending

    def format_for_display(self, request: ReviewRequest) -> str:
        """格式化为可显示的审核请求"""
        return (
            f"⚠️ **需要您的确认**\n\n"
            f"**操作类型**: {request.action_type}\n"
            f"**意图**: {request.intent}\n"
            f"**紧急度**: {'🔴 高' if request.urgency == 'high' else '🟡 中' if request.urgency == 'medium' else '🟢 低'}\n"
            f"**摘要**: {request.summary}\n"
            f"**详情**: {request.details}\n\n"
            f"请回复 **批准** 或 **拒绝** 来做出决策。\n"
            f"（{self.timeout_minutes}分钟内未回复将自动{'批准' if request.urgency == 'high' else '拒绝'}）"
        )


# 全局 Human-in-the-Loop 实例
human_in_the_loop = HumanInTheLoop()

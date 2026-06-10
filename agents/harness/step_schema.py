"""
Step Schema — Agent 间类型安全的 Input/Output 契约

每个 Agent 的输出 TypedDict 就是下游 Agent 的输入类型。
Harness 在传递时自动校验字段完整性，面试时可讲"这就是类型安全的 Agent 契约层"。

对比旧方案：
    agent_results: dict  # ← 无类型，Agent B 不知道 Agent A 输出了什么
    agent_results: LeavePolicyResult  # ← compile-time check + runtime validation
"""

from __future__ import annotations

from typing import TypedDict, NotRequired, Any, Optional


# ============================================================
# Step 配置
# ============================================================

class StepConfig:
    """
    单个 Agent 步骤的配置

    核心字段：
    - agent_id:  哪个 Agent 执行
    - task:      任务描述（给 Agent 的 prompt）
    - output_key: 结果存入 Harness 上下文的 key 名
    - output_schema: 输出类型（TypedDict），用于下游校验
    - input_from: 依赖的上游 step output_key 列表（DAG 入度）
    - timeout:   超时秒数
    - retries:   失败重试次数
    """

    def __init__(
        self,
        agent_id: str,
        task: str,
        output_key: str,
        output_schema: type | None = None,
        input_from: list[str] | None = None,
        timeout: float = 30.0,
        retries: int = 1,
        fallback_agent: str | None = None,
    ):
        self.agent_id = agent_id
        self.task = task
        self.output_key = output_key
        self.output_schema = output_schema
        self.input_from = input_from or []
        self.timeout = timeout
        self.retries = retries
        self.fallback_agent = fallback_agent

    def __repr__(self) -> str:
        deps = "→".join(self.input_from) if self.input_from else "∅"
        return (
            f"Step({self.agent_id} | {deps} → [{self.output_key}] "
            f"timeout={self.timeout}s retries={self.retries})"
        )


def parallel(*steps: StepConfig) -> list[StepConfig]:
    """
    标记一组 Step 可并行执行（无相互依赖）。

    使用方式：
        harness.run(task, [
            parallel(step_policy, step_balance),  # 并行组
            step_compliance,                       # 汇聚后执行
        ])
    """
    return list(steps)


# ============================================================
# 请假合规检查 — 类型安全的 Agent 间契约
# ============================================================

class LeavePolicyResult(TypedDict):
    """RAG Agent 输出：年假政策查询结果"""
    max_consecutive_days: int           # 最长连续休假天数
    requires_approval_above_days: int   # 超过此天数需要审批
    blackout_dates: list[str]           # 不可休假的日期范围
    policy_summary: str                 # 政策原文摘要


class LeaveBalanceResult(TypedDict):
    """ToolAgent 输出：员工年假余额查询结果"""
    employee_name: str
    annual_leave_total: int             # 年假总额
    annual_leave_used: int              # 已用年假
    annual_leave_remaining: int         # 剩余年假
    sick_leave_remaining: int           # 剩余病假
    query_time: str                     # 查询时间


class ComplianceResult(TypedDict):
    """TicketDispatch 输出：合规检查 + 确认卡片"""
    passed: bool                        # 是否通过合规检查
    checks: list[dict]                  # 各项检查结果
    warnings: list[str]                 # 警告信息
    card: dict                          # 确认卡片（如通过）
    summary: str                        # 合规检查摘要

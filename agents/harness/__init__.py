"""
Agent Harness — 多 Agent DAG 编排框架

实现了 HR 面试常考的 Agent Harness 三要素：
1. 步骤编排 (Steps)    — DAG 拓扑，支持 parallel→converge
2. 结果传递 (Passing)   — 类型安全的 Output→Input 契约
3. 角色拆解 (Roles)     — 每个 Step 声明 agent_id + task + schema

使用方式：
    harness = AgentHarness()
    result = await harness.run(task, [
        parallel(step_rag, step_tool),
        step_compliance,
    ])
"""

from .harness import AgentHarness, StepResult, HarnessResult
from .step_schema import (
    StepConfig,
    parallel,
    LeavePolicyResult,
    LeaveBalanceResult,
    ComplianceResult,
)

__all__ = [
    "AgentHarness",
    "StepConfig",
    "StepResult",
    "HarnessResult",
    "parallel",
    "LeavePolicyResult",
    "LeaveBalanceResult",
    "ComplianceResult",
]

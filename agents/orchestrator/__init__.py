"""
Orchestrator 模块 — Copilot Studio 风格的多Agent编排器

核心组件：
- AgentRegistry: Agent 声明注册中心
- Router: 语义路由器（Pydantic结构化输出）
- TaskPlanner: 任务规划与分解
- ResponseSynthesizer: 响应合成（Single Response Principle）

控制与治理：
- ControlLayerManager: 三层控制模型（AI/Hybrid/Deterministic）
- HumanInTheLoop: 人工审核
- AuditTrail: 审计追踪
- TelemetryCollector: 可观测性指标
"""

from .agent_declaration import AgentDeclaration, agent_declaration, set_registry
from .agent_registry import AgentRegistry, agent_registry
from .router import Router, RouteResult, RouterDecision
from .task_planner import TaskPlanner, ExecutionPlan, PlanStep
from .response_synthesizer import ResponseSynthesizer

from .control_layers import (
    ControlLayerManager, ControlLevel, ActionRisk, ControlDecision,
    DeterministicRules, HybridInterceptRules, control_manager,
)
from .human_loop import (
    HumanInTheLoop, HumanDecision, ReviewRequest, ReviewResult, human_in_the_loop,
)
from .governance import (
    AuditTrail, AuditEvent, AuditEventType, GovernanceChecker,
    audit_trail, governance_checker,
)
from .telemetry import (
    TelemetryCollector, AgentMetrics, OrchestrationMetrics, telemetry,
)

__all__ = [
    "AgentDeclaration", "agent_declaration", "set_registry",
    "AgentRegistry", "agent_registry",
    "Router", "RouteResult", "RouterDecision",
    "TaskPlanner", "ExecutionPlan", "PlanStep",
    "ResponseSynthesizer",
    "ControlLayerManager", "ControlLevel", "ActionRisk", "ControlDecision",
    "DeterministicRules", "HybridInterceptRules", "control_manager",
    "HumanInTheLoop", "HumanDecision", "ReviewRequest", "ReviewResult", "human_in_the_loop",
    "AuditTrail", "AuditEvent", "AuditEventType", "GovernanceChecker",
    "audit_trail", "governance_checker",
    "TelemetryCollector", "AgentMetrics", "OrchestrationMetrics", "telemetry",
]

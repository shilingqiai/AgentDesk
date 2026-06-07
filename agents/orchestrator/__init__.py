"""
Orchestrator 模块 — Copilot Studio 风格的多Agent编排器

核心组件：
- AgentRegistry: Agent 声明注册中心
- AgentDeclaration: Agent 声明数据结构
- IntentClassifier: LLM 意图识别
- TaskPlanner: 任务规划与分解
- ResponseSynthesizer: 响应合成（Single Response Principle）
- OrchestratorAgent: 中央编排器主入口

控制与治理（Phase 3-4）：
- ControlLayerManager: 三层控制模型（AI/Hybrid/Deterministic）
- HumanInTheLoop: Human-in-the-Loop 人工审核
- AuditTrail: 审计追踪
- GovernanceChecker: 合规检查
- TelemetryCollector: 可观测性指标

架构参考：
    Microsoft Copilot Studio Multi-Agent Orchestration
    Router Pattern + Single Response Principle + Human-in-the-Loop
"""

from .agent_declaration import AgentDeclaration, agent_declaration, set_registry
from .agent_registry import AgentRegistry, agent_registry
from .intent_classifier import IntentClassifier, IntentResult
from .task_planner import TaskPlanner, ExecutionPlan, PlanStep
from .response_synthesizer import ResponseSynthesizer
from .orchestrator_agent import OrchestratorAgent, orchestrator

# 控制与治理 (Phase 3)
from .control_layers import (
    ControlLayerManager,
    ControlLevel,
    ActionRisk,
    ControlDecision,
    DeterministicRules,
    HybridInterceptRules,
    control_manager,
)
from .human_loop import (
    HumanInTheLoop,
    HumanDecision,
    ReviewRequest,
    ReviewResult,
    human_in_the_loop,
)

# 审计与遥测 (Phase 4)
from .governance import (
    AuditTrail,
    AuditEvent,
    AuditEventType,
    GovernanceChecker,
    audit_trail,
    governance_checker,
)
from .telemetry import (
    TelemetryCollector,
    AgentMetrics,
    OrchestrationMetrics,
    telemetry,
)

__all__ = [
    # Agent 声明系统
    "AgentDeclaration",
    "agent_declaration",
    "set_registry",
    # Agent 注册中心
    "AgentRegistry",
    "agent_registry",
    # 编排组件
    "IntentClassifier",
    "IntentResult",
    "TaskPlanner",
    "ExecutionPlan",
    "PlanStep",
    "ResponseSynthesizer",
    # 编排器主入口
    "OrchestratorAgent",
    "orchestrator",
    # 控制层
    "ControlLayerManager",
    "ControlLevel",
    "ActionRisk",
    "ControlDecision",
    "DeterministicRules",
    "HybridInterceptRules",
    "control_manager",
    # Human-in-the-Loop
    "HumanInTheLoop",
    "HumanDecision",
    "ReviewRequest",
    "ReviewResult",
    "human_in_the_loop",
    # 审计
    "AuditTrail",
    "AuditEvent",
    "AuditEventType",
    "GovernanceChecker",
    "audit_trail",
    "governance_checker",
    # 遥测
    "TelemetryCollector",
    "AgentMetrics",
    "OrchestrationMetrics",
    "telemetry",
]

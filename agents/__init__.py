"""
多Agent智能层 — Hub & Spoke 编排架构

模块:
- orchestrator/: 编排器（AgentRegistry, Router, TaskPlanner, ResponseSynthesizer）
- a2a/: Agent间通信协议
- sub_agents/: 专业子Agent（EnterpriseRAG, TicketDispatch）
- graph_workflow.py: LangGraph编排工作流
"""

from .a2a import AgentMessage, MessageIntent, AgentRole, MessageBus, ContextManager
from .orchestrator import (
    AgentDeclaration, agent_declaration, AgentRegistry, agent_registry,
    Router, RouteResult, TaskPlanner, ExecutionPlan, PlanStep,
    ResponseSynthesizer,
    ControlLayerManager, HumanInTheLoop,
    AuditTrail, TelemetryCollector, telemetry, audit_trail,
)
from .base_sub_agent import BaseSubAgent
from .sub_agents import EnterpriseRAGAgent, TicketDispatchSubAgent
from .graph_workflow import orchestration_runner, OrchestrationWorkflowRunner

__all__ = [
    "AgentMessage", "MessageIntent", "AgentRole", "MessageBus", "ContextManager",
    "AgentDeclaration", "agent_declaration", "AgentRegistry", "agent_registry",
    "Router", "RouteResult",
    "TaskPlanner", "ExecutionPlan", "PlanStep",
    "ResponseSynthesizer",
    "ControlLayerManager", "HumanInTheLoop",
    "AuditTrail", "TelemetryCollector", "telemetry", "audit_trail",
    "BaseSubAgent",
    "EnterpriseRAGAgent", "TicketDispatchSubAgent",
    "orchestration_runner", "OrchestrationWorkflowRunner",
]

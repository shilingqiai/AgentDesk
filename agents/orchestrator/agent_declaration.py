"""
Agent Declaration System

参考 Microsoft Copilot Studio 的 Agent 声明模式：
- 每个 Agent 声明自己的 identity、capabilities、knowledge 域
- 声明式路由：编排器根据声明选择正确的 Agent
- 非重叠知识域：每个 Agent 的知识域互斥

使用方式：
    @AgentDeclaration(
        agent_id="it_consultant",
        name="IT咨询Agent",
        description="负责IT故障排查和知识自服务...",
        capabilities=["rag_search", "troubleshooting"],
        knowledge_domains=["it_support", "software_guide"],
    )
    class ITConsultantSubAgent(BaseSubAgent):
        ...
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Optional


@dataclass
class AgentDeclaration:
    """
    Agent 声明 — 每个子Agent必须提供的元数据

    对应 Copilot Studio 中的 Agent 描述：
    - agent_id: 全局唯一标识
    - name: 显示名称
    - description: 功能描述（编排器用于路由决策，必须准确且区分度高）
    - capabilities: 能力列表（用于能力发现）
    - knowledge_domains: 知识域（必须与其他Agent不重叠）
    - priority: 路由优先级（数字越小越优先）
    """

    agent_id: str
    name: str
    description: str
    capabilities: list[str] = field(default_factory=list)
    knowledge_domains: list[str] = field(default_factory=list)
    priority: int = 10

    # 可选：input/output schema 引用（用于类型检查）
    input_schema: Optional[type] = None
    output_schema: Optional[type] = None

    # 元数据
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        """转换为字典（供编排器用于路由决策）"""
        return {
            "agent_id": self.agent_id,
            "name": self.name,
            "description": self.description,
            "capabilities": self.capabilities,
            "knowledge_domains": self.knowledge_domains,
            "priority": self.priority,
            "metadata": self.metadata,
        }

    def to_routing_prompt(self) -> str:
        """
        生成用于 LLM 路由决策的描述文本

        编排器将此文本嵌入路由 prompt 中，
        让 LLM 根据 Agent 描述选择正确的 Agent。
        """
        capabilities_str = "、".join(self.capabilities[:5]) if self.capabilities else "通用处理"
        domains_str = "、".join(self.knowledge_domains[:3]) if self.knowledge_domains else "通用域"
        return (
            f"- **{self.name}** (ID: `{self.agent_id}`)\n"
            f"  描述: {self.description}\n"
            f"  能力: {capabilities_str}\n"
            f"  知识域: {domains_str}"
        )


# ============================================================
# Agent 注册装饰器
# ============================================================

# 全局注册表引用（在运行时注入）
_registry_ref: Optional["AgentRegistry"] = None


def set_registry(registry: "AgentRegistry") -> None:
    """设置全局注册表引用（在 AgentRegistry 初始化时调用）"""
    global _registry_ref
    _registry_ref = registry


def agent_declaration(
    agent_id: str,
    name: str,
    description: str,
    capabilities: list[str] = None,
    knowledge_domains: list[str] = None,
    priority: int = 10,
    **metadata,
) -> Callable:
    """
    装饰器：将类标记为 Agent 声明

    使用方式：
        @agent_declaration(
            agent_id="it_consultant",
            name="IT咨询Agent",
            description="负责IT故障排查...",
            capabilities=["rag_search"],
            knowledge_domains=["it_support"],
        )
        class ITConsultantSubAgent(BaseSubAgent):
            ...

    装饰器会自动：
    1. 将声明附加到类的 __agent_declaration__ 属性
    2. 将类注册到全局 AgentRegistry
    """
    def decorator(cls):
        decl = AgentDeclaration(
            agent_id=agent_id,
            name=name,
            description=description,
            capabilities=capabilities or [],
            knowledge_domains=knowledge_domains or [],
            priority=priority,
            metadata=metadata,
        )
        # 将声明附加到类
        cls.__agent_declaration__ = decl

        # 自动注册到全局注册表
        if _registry_ref is not None:
            _registry_ref.register(decl, cls)

        return cls
    return decorator

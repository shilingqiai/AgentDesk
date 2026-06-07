"""
Agent Registry — Agent 注册中心

参考 Microsoft Copilot Studio 的 Agent 发现与路由机制：
- 统一管理所有 Agent 声明
- 支持按意图/能力/知识域发现 Agent
- 生成 LLM 路由决策的 Agent 描述列表

Single Response Principle:
    注册中心确保每个子Agent明确标记为 reply_to_user=False
"""

from __future__ import annotations

import logging
from typing import Any, Optional

from .agent_declaration import AgentDeclaration, set_registry

logger = logging.getLogger("orchestrator.registry")


class AgentRegistry:
    """
    Agent 注册中心（单例模式）

    职责：
    1. 管理所有已注册的 Agent 声明
    2. 提供 Agent 发现 API（按ID、能力、知识域）
    3. 生成 LLM 路由决策用的 Agent 描述列表

    使用方式：
        registry = AgentRegistry()
        registry.register(declaration, agent_class)
        agent = registry.get_agent("it_consultant")
    """

    def __init__(self):
        # agent_id → AgentDeclaration
        self._declarations: dict[str, AgentDeclaration] = {}
        # agent_id → Agent 类引用
        self._agent_classes: dict[str, type] = {}
        # agent_id → Agent 实例（懒加载）
        self._agent_instances: dict[str, Any] = {}

        # 将自身注册为全局引用，供装饰器使用
        set_registry(self)

    def register(self, declaration: AgentDeclaration, agent_class: type) -> None:
        """
        注册一个 Agent

        Args:
            declaration: Agent 声明
            agent_class: Agent 实现类

        Raises:
            ValueError: 如果 agent_id 已存在
        """
        agent_id = declaration.agent_id

        if agent_id in self._declarations:
            existing = self._declarations[agent_id]
            logger.warning(
                f"Agent '{agent_id}' 已注册（{existing.name}），将被覆盖"
            )

        self._declarations[agent_id] = declaration
        self._agent_classes[agent_id] = agent_class

        # 清除缓存的实例（让下次获取时重新初始化）
        self._agent_instances.pop(agent_id, None)

        logger.info(
            f"✓ 注册 Agent: {declaration.name} (ID: {agent_id}) "
            f"[能力: {len(declaration.capabilities)}, "
            f"知识域: {len(declaration.knowledge_domains)}]"
        )

    def get_agent(self, agent_id: str):
        """
        获取 Agent 实例（懒加载）

        Args:
            agent_id: Agent 唯一标识

        Returns:
            Agent 实例，如果不存在返回 None
        """
        # 返回缓存的实例
        if agent_id in self._agent_instances:
            return self._agent_instances[agent_id]

        # 懒加载创建实例
        agent_class = self._agent_classes.get(agent_id)
        if agent_class is None:
            logger.error(f"Agent '{agent_id}' 未注册")
            return None

        try:
            instance = agent_class()
            self._agent_instances[agent_id] = instance
            return instance
        except Exception as e:
            logger.error(f"无法实例化 Agent '{agent_id}': {e}")
            return None

    def get_declaration(self, agent_id: str) -> Optional[AgentDeclaration]:
        """获取 Agent 声明"""
        return self._declarations.get(agent_id)

    def list_agents(self) -> list[dict]:
        """
        列出所有已注册的 Agent

        Returns:
            Agent 声明字典列表（按优先级排序）
        """
        agents = sorted(
            self._declarations.values(),
            key=lambda d: d.priority,
        )
        return [a.to_dict() for a in agents]

    def get_routing_descriptions(self, exclude_orchestrator: bool = True) -> str:
        """
        生成 LLM 路由决策用的 Agent 描述列表

        编排器将此文本嵌入路由 prompt，让 LLM 选择正确的 Agent。

        Args:
            exclude_orchestrator: 是否排除编排器自身

        Returns:
            格式化后的 Agent 描述文本
        """
        agents = sorted(
            self._declarations.values(),
            key=lambda d: d.priority,
        )

        descriptions = []
        for agent in agents:
            if exclude_orchestrator and agent.agent_id == "orchestrator":
                continue
            descriptions.append(agent.to_routing_prompt())

        if not descriptions:
            return "（暂无可用Agent）"

        return "\n\n".join(descriptions)

    def discover_by_capability(self, capability: str) -> list[AgentDeclaration]:
        """
        按能力发现 Agent

        Args:
            capability: 能力名称

        Returns:
            拥有该能力的 Agent 声明列表
        """
        return [
            d for d in self._declarations.values()
            if capability in d.capabilities
        ]

    def discover_by_domain(self, domain: str) -> list[AgentDeclaration]:
        """
        按知识域发现 Agent

        Args:
            domain: 知识域名称

        Returns:
            覆盖该知识域的 Agent 声明列表
        """
        return [
            d for d in self._declarations.values()
            if domain in d.knowledge_domains
        ]

    def discover_by_intent(self, intent_description: str, llm=None) -> Optional[AgentDeclaration]:
        """
        按意图描述发现最匹配的 Agent（使用 LLM 路由）

        Args:
            intent_description: 意图描述文本
            llm: LLM 实例（用于匹配判断）

        Returns:
            最匹配的 Agent 声明，如果没有匹配返回 None
        """
        # 简单规则匹配：在 Agent 描述中搜索关键词
        intent_lower = intent_description.lower()
        for agent in sorted(
            self._declarations.values(),
            key=lambda d: d.priority,
        ):
            if agent.agent_id == "orchestrator":
                continue
            # 匹配能力关键词
            for cap in agent.capabilities:
                if cap.lower().replace("_", " ") in intent_lower:
                    return agent
            # 匹配知识域关键词
            for domain in agent.knowledge_domains:
                if domain.lower().replace("_", " ") in intent_lower:
                    return agent

        return None

    @property
    def agent_count(self) -> int:
        """已注册的 Agent 数量"""
        return len(self._declarations)


# 全局 Agent 注册中心实例
agent_registry = AgentRegistry()

"""
BaseSubAgent — 子Agent抽象基类

参考 Microsoft Copilot Studio 的子Agent设计规范：

1. 子Agent是"研究者"而非"响应者"
2. 子Agent通过 execute() 接收编排器委派 → 执行任务 → 返回结构化结果
3. 子Agent永远不直接回复用户（Single Response Principle）
4. 子Agent的指令中明确：MUST return findings to the orchestrator, NEVER reply to user

使用方式：
    @agent_declaration(
        agent_id="it_consultant",
        name="IT咨询Agent",
        ...
    )
    class ITConsultantSubAgent(BaseSubAgent):
        async def execute(self, message: AgentMessage) -> AgentMessage:
            ...
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import AsyncGenerator, Optional

from agents.a2a.protocol import AgentMessage, AgentRole


class BaseSubAgent(ABC):
    """
    子Agent抽象基类

    所有专业子Agent必须继承此类并实现 execute() 方法。

    核心规则（来自 Microsoft Copilot Studio 最佳实践）：
    1. MUST return findings to orchestrator — NEVER reply to user
    2. 使用强指令语言：MUST / NEVER / ONLY
    3. 知识域与其他Agent不重叠
    4. 不持有其他Agent的引用
    """

    # 子类必须设置 agent_id（与 AgentDeclaration 中的一致）
    agent_id: str = ""

    # Agent 角色
    role: AgentRole = AgentRole.SUB_AGENT

    def __init__(self):
        self.logger = logging.getLogger(f"agent.{self.agent_id}")

        # 获取声明信息（由 @agent_declaration 装饰器设置）
        self.declaration = getattr(self.__class__, "__agent_declaration__", None)

    @abstractmethod
    async def execute(self, message: AgentMessage) -> AgentMessage:
        """
        执行Agent任务（核心方法）

        编排器通过此方法委派任务给子Agent。

        Args:
            message: 编排器发送的委派消息
                     - message.payload: 任务参数
                     - message.context: 共享上下文
                     - message.trace_id: 追踪ID

        Returns:
            AgentMessage(reply_to_user=False):
                - payload: 结构化执行结果
                - success: 是否成功
                - error: 错误信息（如有）

        IMPORTANT:
            子Agent必须返回 reply_to_user=False 的消息。
            只有编排器有权对用户响应。
        """
        ...

    async def execute_stream(self, message: AgentMessage) -> AsyncGenerator[str, None]:
        """
        流式执行（可选覆盖）

        子Agent可以在执行过程中流式输出进度信息（给编排器，不是给用户）。
        编排器可以选择性地将这些进度信息透传给用户。

        Args:
            message: 编排器发送的委派消息

        Yields:
            进度信息字符串（如 "[子Agent] 正在检索知识库..."）
        """
        # 默认实现：直接调用 execute，不产生中间流
        result = await self.execute(message)
        if result.success:
            yield f"[AGENT:{self.agent_id}] 执行完成"
        else:
            yield f"[AGENT:{self.agent_id}] 执行失败: {result.error}"

    def validate_message(self, message: AgentMessage) -> Optional[str]:
        """
        验证输入消息的有效性

        Args:
            message: 输入消息

        Returns:
            错误描述字符串，如果验证通过返回 None
        """
        if not message.payload:
            return "消息 payload 为空"
        if not message.to_agent:
            return "消息缺少 to_agent 字段"
        return None

    def create_error_response(
        self,
        original_message: AgentMessage,
        error: str,
    ) -> AgentMessage:
        """
        创建错误响应消息

        Args:
            original_message: 原始委派消息
            error: 错误描述

        Returns:
            带有错误信息的 AgentMessage
        """
        return AgentMessage.create_response(
            from_agent=self.agent_id,
            to_agent=original_message.from_agent,
            payload={"error": error},
            original_message=original_message,
            success=False,
            error=error,
        )

    def __repr__(self) -> str:
        name = self.declaration.name if self.declaration else self.agent_id
        return f"<{name} (ID: {self.agent_id})>"

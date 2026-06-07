"""
响应合成器 — Orchestrator 的最终阶段

参考 Microsoft Copilot Studio 的 Single Response Principle：
- 子Agent不直接回复用户
- 编排器收集所有子Agent结果后，合成为统一的用户响应
- 响应中不暴露内部Agent调用细节（除非调试模式）

调用流程：
    invoke sub-agents → wait for all results → combine → respond to user
"""

from __future__ import annotations

import logging
from typing import Optional

from langchain.prompts import PromptTemplate
from langchain_core.language_models.chat_models import BaseChatModel

from agents.a2a.protocol import AgentMessage

logger = logging.getLogger("orchestrator.response_synthesizer")


class ResponseSynthesizer:
    """
    响应合成器

    职责：
    1. 收集所有子Agent的执行结果
    2. 合成为自然、连贯的用户响应
    3. 在调试模式下可显示Agent调用过程
    4. 处理部分Agent失败的情况（优雅降级）

    使用方式：
        synthesizer = ResponseSynthesizer(llm)
        response = await synthesizer.synthesize(agent_results, user_input, debug_mode)
    """

    def __init__(self, llm: BaseChatModel):
        self.llm = llm
        self._initialize_prompt()

    def _initialize_prompt(self):
        """初始化合成提示词模板"""
        self.synthesize_prompt = PromptTemplate(
            input_variables=["user_input", "agent_results", "debug_mode"],
            template=(
                "你是一个企业智能服务台的响应合成器。\n\n"
                "## 规则\n\n"
                "1. 你收到多个专业Agent的分析结果，需要将它们合成为一条自然、连贯的回复\n"
                "2. 不要暴露内部Agent名称或调用细节（如'IT Agent说...'）\n"
                "3. 如果有Agent返回了知识库内容，优先使用知识库内容\n"
                "4. 如果有Agent返回了工单信息，清晰告知工单状态\n"
                "5. 如果部分Agent失败，优雅地告知用户哪些部分暂时不可用\n"
                "6. 保持专业、简洁的企业服务台语气\n"
                "7. 如果需要用户补充信息，主动询问\n\n"
                "## 用户原始问题\n\n"
                "{user_input}\n\n"
                "## Agent分析结果\n\n"
                "{agent_results}\n\n"
                "## 调试模式\n\n"
                "{debug_mode}\n\n"
                "请直接输出合成后的用户回复（纯文本，不要JSON）。"
            ),
        )

    async def synthesize(
        self,
        agent_results: dict[str, AgentMessage],
        user_input: str,
        debug_mode: bool = False,
    ) -> str:
        """
        合成子Agent结果为用户响应

        Args:
            agent_results: {agent_id: AgentMessage} 各Agent的执行结果
            user_input: 用户原始输入
            debug_mode: 是否显示调试信息

        Returns:
            合成后的用户响应文本
        """
        # 格式化 Agent 结果
        results_text = self._format_agent_results(agent_results)

        # 如果没有有效结果
        if not results_text.strip():
            return (
                "抱歉，我暂时无法处理您的请求。"
                "请稍后重试或联系IT服务台获取人工支持。"
            )

        # 简单响应（不需要 LLM 合成）：单个Agent成功
        if len(agent_results) == 1:
            single_result = list(agent_results.values())[0]
            if single_result.success and single_result.payload.get("direct_response"):
                response = single_result.payload["direct_response"]
                if debug_mode:
                    agent_id = list(agent_results.keys())[0]
                    response = (
                        f"[DEBUG] 来源: {agent_id}\n"
                        f"[DEBUG] trace_id: {single_result.trace_id}\n\n"
                        f"{response}"
                    )
                return response

        # 复杂响应：使用 LLM 合成
        try:
            chain = self.synthesize_prompt | self.llm
            response = await chain.ainvoke({
                "user_input": user_input,
                "agent_results": results_text,
                "debug_mode": "是" if debug_mode else "否",
            })
            return response.content.strip()
        except Exception as e:
            logger.error(f"响应合成失败: {e}")
            return self._fallback_synthesize(agent_results)

    def _format_agent_results(self, agent_results: dict[str, AgentMessage]) -> str:
        """格式化Agent结果用于 prompt"""
        if not agent_results:
            return "（无Agent结果）"

        lines = []
        for agent_id, msg in agent_results.items():
            success = "✓" if msg.success else "❌"
            if msg.error:
                lines.append(f"- {agent_id} {success}: 错误 - {msg.error}")
            elif msg.payload:
                # 简化 payload 输出（避免过长）
                for key, value in msg.payload.items():
                    if isinstance(value, str) and len(value) > 500:
                        value = value[:500] + "..."
                    lines.append(f"- {agent_id} {success}: {key} = {value}")
            else:
                lines.append(f"- {agent_id} {success}: 无内容")

        return "\n".join(lines)

    def _fallback_synthesize(self, agent_results: dict[str, AgentMessage]) -> str:
        """兜底合成：不使用 LLM，直接拼接结果"""
        parts = []
        for agent_id, msg in agent_results.items():
            if not msg.success:
                continue

            # 提取 direct_response 或第一个文本值
            direct = msg.payload.get("direct_response", "")
            if direct:
                parts.append(direct)
            else:
                for value in msg.payload.values():
                    if isinstance(value, str) and len(value) > 20:
                        parts.append(value)
                        break

        if parts:
            return "\n\n".join(parts)
        return "处理完成，但无法生成有效回复。请提交工单获取人工支持。"

    def synthesize_debug(
        self,
        agent_results: dict[str, AgentMessage],
        plan_steps: list,
        trace_id: str,
    ) -> str:
        """
        生成调试视图（显示完整Agent调用链路）

        Args:
            agent_results: Agent 执行结果
            plan_steps: 执行计划步骤
            trace_id: 追踪ID

        Returns:
            调试格式的响应
        """
        lines = [
            f"```",
            f"🔍 Agent 调用追踪 (trace_id: {trace_id})",
            f"",
            f"📋 执行计划:",
        ]

        for i, step in enumerate(plan_steps):
            agent_id = step.agent_id if hasattr(step, 'agent_id') else step.get('agent_id', '?')
            task = step.task if hasattr(step, 'task') else step.get('task', '?')
            lines.append(f"  {i+1}. {agent_id}: {task}")

        lines.append("")
        lines.append("📊 执行结果:")

        for agent_id, msg in agent_results.items():
            status = "✅" if msg.success else "❌"
            lines.append(f"  {status} {agent_id}: {msg.payload.get('summary', msg.error or '完成')}")

        lines.append("```")
        return "\n".join(lines)

"""
ToolAgent — MCP 风格工具调用 Agent

演示 Agent 如何通过 ToolRegistry 发现并调用工具。
当用户请求需要外部工具支持（天气查询、日期计算等）时，
此 Agent 从 LLM 输出中解析工具调用意图，执行工具并返回结果。

这是 P2 "MCP 工具调用 demo" 的核心实现。
"""

from __future__ import annotations

import json
import logging
import re
from typing import AsyncGenerator

from agents.base_sub_agent import BaseSubAgent
from agents.a2a.protocol import AgentMessage
from agents.orchestrator.agent_declaration import agent_declaration
from agents.orchestrator.agent_registry import agent_registry
from agents.tools import tool_registry
from config.model_provider import create_chat_model

logger = logging.getLogger("agent.tool_agent")


@agent_declaration(
    agent_id="tool_agent",
    name="工具调用Agent",
    description=(
        "负责调用外部工具和API完成任务。支持天气查询、日期计算、工单状态查询等工具。"
        "当用户请求需要实时数据或外部服务时调用此Agent。"
    ),
    capabilities=["tool_use", "external_api", "weather_query", "date_calculation"],
    knowledge_domains=["external_services", "utility_tools"],
    priority=3,
)
class ToolAgent(BaseSubAgent):
    """
    MCP 风格工具调用 Agent

    工作流程：
    1. 接收编排器委派（含 user_input + 可用工具列表）
    2. LLM 分析输入，判断需要调用哪个工具及参数
    3. 通过 ToolRegistry.invoke() 执行工具
    4. 返回结构化工具结果给编排器
    """

    agent_id = "tool_agent"

    def __init__(self):
        super().__init__()
        self.llm = create_chat_model(temperature=0)

    async def execute(self, message: AgentMessage) -> AgentMessage:
        """执行工具调用"""
        user_input = message.payload.get("user_input", "")
        task = message.payload.get("task", "")

        self.logger.info(f"[ToolAgent] 处理工具调用请求: {user_input[:60]}")

        try:
            # 获取可用工具描述
            tools_desc = tool_registry.get_tools_for_llm()

            if not tools_desc:
                return AgentMessage.create_response(
                    from_agent=self.agent_id,
                    to_agent=message.from_agent,
                    payload={
                        "direct_response": "暂无可用工具。",
                        "tool_result": None,
                    },
                    original_message=message,
                    success=True,
                )

            # LLM 解析工具调用意图
            tool_call = await self._parse_tool_call(user_input, tools_desc, task)

            if tool_call is None:
                return AgentMessage.create_response(
                    from_agent=self.agent_id,
                    to_agent=message.from_agent,
                    payload={
                        "direct_response": "无法识别需要调用哪个工具，请提供更具体的信息。",
                        "tool_result": None,
                        "available_tools": tool_registry.list_tools()[0].name if tool_registry.list_tools() else "",
                    },
                    original_message=message,
                    success=True,
                )

            # 执行工具
            tool_name = tool_call["name"]
            tool_args = tool_call.get("args", {})

            self.logger.info(f"[ToolAgent] 调用工具: {tool_name}({tool_args})")
            result = await tool_registry.invoke(tool_name, **tool_args)

            if result.success:
                response_text = self._format_tool_result(result)
                return AgentMessage.create_response(
                    from_agent=self.agent_id,
                    to_agent=message.from_agent,
                    payload={
                        "direct_response": response_text,
                        "tool_result": result.data,
                        "tool_name": result.tool_name,
                        "duration_ms": result.duration_ms,
                        "summary": f"工具 {tool_name} 执行成功 ({result.duration_ms:.0f}ms)",
                    },
                    original_message=message,
                    success=True,
                )
            else:
                return AgentMessage.create_response(
                    from_agent=self.agent_id,
                    to_agent=message.from_agent,
                    payload={
                        "direct_response": f"工具调用失败：{result.error}",
                        "tool_result": None,
                        "tool_name": result.tool_name,
                    },
                    original_message=message,
                    success=False,
                    error=result.error,
                )

        except Exception as e:
            self.logger.error(f"[ToolAgent] 执行失败: {e}")
            return self.create_error_response(message, str(e))

    async def _parse_tool_call(
        self, user_input: str, tools_desc: str, task: str = "",
    ) -> dict | None:
        """
        LLM 解析工具调用意图 → 返回 {name, args}

        与 Router 一致：prompt→JSON + json_repair 兜底
        """
        prompt = (
            "你是一个工具调用解析器。根据用户输入，选择最合适的工具并提取参数。\n\n"
            f"## 可用工具\n{tools_desc}\n\n"
            f"## 用户输入\n{user_input}\n"
            + (f"## 任务描述\n{task}\n\n" if task else "\n")
            + "请返回 JSON（不要 markdown 包裹）：\n"
              '{"tool":"工具名","args":{"参数名":"值"}}\n\n'
              '如果不需要调用任何工具，返回 {"tool":null}'
        )

        try:
            response = await self.llm.ainvoke([{"role": "user", "content": prompt}])
            data = self._extract_json(response.content)

            tool_name = data.get("tool")
            if not tool_name:
                return None

            return {"name": tool_name, "args": data.get("args", {})}

        except Exception as e:
            self.logger.warning(f"[ToolAgent] 工具调用解析失败: {e}")
            return None

    @staticmethod
    def _extract_json(text: str) -> dict:
        """与 Router 一致的 JSON 提取逻辑"""
        try:
            return json.loads(text.strip())
        except json.JSONDecodeError:
            pass
        m = re.search(r'```(?:json)?\s*([\s\S]*?)```', text)
        if m:
            try:
                return json.loads(m.group(1).strip())
            except json.JSONDecodeError:
                pass
        m = re.search(r'\{[\s\S]*\}', text)
        if m:
            raw = m.group(0)
            try:
                return json.loads(raw)
            except json.JSONDecodeError:
                try:
                    from json_repair import repair_json
                    return json.loads(repair_json(raw))
                except Exception:
                    pass
        raise ValueError(f"无法提取 JSON: {text[:200]}")

    @staticmethod
    def _format_tool_result(result) -> str:
        """格式化工具结果为用户可读文本"""
        data = result.data
        if not isinstance(data, dict):
            return str(data)

        tool = result.tool_name

        if tool == "weather_query":
            return (
                f"🌤 **{data.get('city', '未知')} 天气**\n\n"
                f"- 温度：{data.get('temp', '--')}°C\n"
                f"- 天气：{data.get('weather', '--')}\n"
                f"- 湿度：{data.get('humidity', '--')}%\n"
                f"- 风力：{data.get('wind', '--')}\n"
                f"- 更新时间：{data.get('query_time', '--')}"
            )

        if tool == "date_calculator":
            return (
                f"📅 计算结果\n\n"
                f"日期：**{data.get('date', '--')}** ({data.get('weekday', '')})"
            )

        if tool == "ticket_status":
            if data.get("found"):
                return (
                    f"📋 工单 **{data.get('ticket_number')}**\n\n"
                    f"- 标题：{data.get('title', '')}\n"
                    f"- 状态：{data.get('status', '')}\n"
                    f"- 优先级：{data.get('priority', '')}"
                )
            return f"未找到工单 {data.get('ticket_number', '')}"

        return json.dumps(data, ensure_ascii=False, indent=2)

    async def execute_stream(self, message: AgentMessage) -> AsyncGenerator[str, None]:
        yield "[ToolAgent] 正在分析工具调用意图..."
        yield "[ToolAgent] 正在执行工具..."
        yield "[ToolAgent] 工具执行完成"


# 自动注册
def _register():
    import agents.tools.builtin_tools  # noqa: F401  确保工具已注册
    agent_registry.register(
        ToolAgent.__agent_declaration__,
        ToolAgent,
    )

_register()

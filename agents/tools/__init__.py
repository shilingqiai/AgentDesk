"""
MCP 工具注册与调用框架

实现 Model Context Protocol (MCP) 风格的 Agent Tool Use：
- 声明式工具注册（name/description/parameters）
- 标准化调用接口
- 工具执行结果结构化返回

使用方式：
    from agents.tools import tool_registry, ToolDefinition

    @tool_registry.register(
        name="web_search",
        description="在网上搜索信息",
        parameters={"query": {"type": "string", "description": "搜索查询词"}},
    )
    async def web_search(query: str) -> dict:
        ...
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Callable, Any, Awaitable

logger = logging.getLogger("agent.tools")


@dataclass
class ToolDefinition:
    """工具定义（MCP 风格）"""
    name: str
    description: str
    parameters: dict[str, dict]        # {param_name: {type, description, required}}
    handler: Callable[..., Awaitable[Any]]
    category: str = "general"          # general | external | internal


@dataclass
class ToolResult:
    """工具执行结果"""
    tool_name: str
    success: bool
    data: Any = None
    error: str = ""
    duration_ms: float = 0


class ToolRegistry:
    """
    Agent 工具注册中心

    管理所有可用工具的定义与调用。Router / Agent 可通过此注册中心
    发现并调用工具，实现 Agent Tool Use。
    """

    def __init__(self):
        self._tools: dict[str, ToolDefinition] = {}

    def register(
        self,
        name: str,
        description: str,
        parameters: dict[str, dict],
        category: str = "general",
    ):
        """装饰器：注册工具"""
        def decorator(func: Callable[..., Awaitable[Any]]):
            self._tools[name] = ToolDefinition(
                name=name,
                description=description,
                parameters=parameters,
                handler=func,
                category=category,
            )
            logger.info(f"[ToolRegistry] 注册工具: {name} ({category})")
            return func
        return decorator

    def get_tool(self, name: str) -> ToolDefinition | None:
        """获取工具定义"""
        return self._tools.get(name)

    def list_tools(self, category: str = None) -> list[ToolDefinition]:
        """列出所有工具"""
        tools = list(self._tools.values())
        if category:
            tools = [t for t in tools if t.category == category]
        return tools

    def get_tools_for_llm(self) -> str:
        """生成供 LLM prompt 使用的工具描述"""
        lines = []
        for tool in self._tools.values():
            params_desc = ", ".join(
                f"{k}: {v.get('type', 'str')}" for k, v in tool.parameters.items()
            )
            lines.append(f"- {tool.name}({params_desc}): {tool.description}")
        return "\n".join(lines)

    def get_tools_as_openai_schemas(self, category: str = None) -> list[dict]:
        """
        将已注册工具转为 OpenAI Function Calling 格式。

        用于 bind_tools / tool_choice="auto"。

        Args:
            category: 可选，按分类过滤

        Returns:
            [{"type": "function", "function": {"name": ..., "description": ..., "parameters": ...}}, ...]
        """
        schemas = []
        tools = self.list_tools(category=category)
        for tool in tools:
            # 将 ToolRegistry 的 parameters 格式转为 JSON Schema
            properties = {}
            required = []
            for param_name, param_def in tool.parameters.items():
                properties[param_name] = {
                    "type": param_def.get("type", "string"),
                    "description": param_def.get("description", ""),
                }
                if param_def.get("required"):
                    required.append(param_name)

            schema = {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": {
                        "type": "object",
                        "properties": properties,
                        "required": required,
                    },
                },
            }
            schemas.append(schema)
        return schemas

    async def invoke(self, name: str, **kwargs) -> ToolResult:
        """调用工具"""
        import time

        tool = self._tools.get(name)
        if tool is None:
            return ToolResult(tool_name=name, success=False, error=f"工具 '{name}' 未注册")

        start = time.time()
        try:
            data = await tool.handler(**kwargs)
            duration = (time.time() - start) * 1000
            return ToolResult(tool_name=name, success=True, data=data, duration_ms=duration)
        except Exception as e:
            duration = (time.time() - start) * 1000
            logger.error(f"[ToolRegistry] 工具 '{name}' 执行失败: {e}")
            return ToolResult(tool_name=name, success=False, error=str(e), duration_ms=duration)


# 全局工具注册中心
tool_registry = ToolRegistry()

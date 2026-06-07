"""
任务规划器 — Orchestrator 的第二阶段

参考 Microsoft Copilot Studio 的 Task Planning 模式：
- 接收到意图分类结果后，制定执行计划
- 决定调用哪个/哪些子Agent，调用顺序
- 支持并行调用（多个Agent同时处理不同子任务）

计划结构：
    [
        {"agent_id": "it_consultant", "task": "检索VPN排查知识", "params": {...}},
        {"agent_id": "ticket_dispatch", "task": "创建工单", "params": {...}},
    ]
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Optional

from langchain.prompts import PromptTemplate
from langchain_core.language_models.chat_models import BaseChatModel

from .router import RouteResult as IntentResult  # 兼容别名

logger = logging.getLogger("orchestrator.task_planner")


@dataclass
class PlanStep:
    """执行计划中的单步"""
    agent_id: str
    task: str
    params: dict = field(default_factory=dict)
    depends_on: Optional[str] = None  # 依赖的前置步骤（agent_id）


@dataclass
class ExecutionPlan:
    """执行计划"""
    steps: list[PlanStep]
    parallel_groups: list[list[int]]  # 可并行执行的步骤索引组
    needs_human_review: bool
    reasoning: str = ""


class TaskPlanner:
    """
    任务规划器

    职责：
    1. 接收意图分类结果
    2. 制定执行计划（调用哪些Agent、顺序、并行度）
    3. 判断是否需要 Human-in-the-Loop

    使用方式：
        planner = TaskPlanner(llm, agent_registry)
        plan = await planner.plan(intent_result, user_input)
    """

    def __init__(self, llm: BaseChatModel, agent_registry=None):
        self.llm = llm
        self.agent_registry = agent_registry
        self._initialize_prompt()

    def _initialize_prompt(self):
        """初始化规划提示词模板"""
        self.plan_prompt = PromptTemplate(
            input_variables=["agent_list", "intent_info", "user_input"],
            template=(
                "你是一个企业智能服务台的编排规划器。根据用户意图制定Agent调用计划。\n\n"
                "## 可用Agent\n\n"
                "{agent_list}\n\n"
                "## 意图分析\n\n"
                "{intent_info}\n\n"
                "## 用户原始输入\n\n"
                "{user_input}\n\n"
                "## 规划规则\n\n"
                "1. 优先选择最匹配的Agent处理\n"
                "2. 如果问题需要多个Agent协作，按依赖关系排序\n"
                "3. 无依赖的Agent可以并行调用\n"
                "4. 以下情况需要人工审核：\n"
                "   - 创建高优先级工单（P0/P1）\n"
                "   - 涉及敏感数据或权限变更\n"
                "   - 非工作时间提交的紧急请求\n"
                "5. 如果没有任何Agent匹配，返回空计划\n\n"
                "## 输出格式\n\n"
                "请严格输出以下 JSON 格式：\n"
                "{{\n"
                '  "steps": [\n'
                '    {{"agent_id": "agent_id", "task": "任务描述", "params": {{}}, "depends_on": null}}\n'
                "  ],\n"
                '  "needs_human_review": false,\n'
                '  "reasoning": "规划理由"\n'
                "}}"
            ),
        )

    async def plan(
        self,
        intent_result: IntentResult,
        user_input: str,
    ) -> ExecutionPlan:
        """
        制定执行计划

        Args:
            intent_result: 意图分类结果
            user_input: 用户原始输入

        Returns:
            ExecutionPlan 执行计划
        """
        # 简单场景：直接匹配，不需要 LLM
        if intent_result.confidence > 0.8 and intent_result.target_agent:
            return self._simple_plan(intent_result)

        # 复杂场景：使用 LLM 制定多步骤计划
        return await self._llm_plan(intent_result, user_input)

    def _simple_plan(self, intent_result: IntentResult) -> ExecutionPlan:
        """简单计划：高置信度单Agent路由"""
        step = PlanStep(
            agent_id=intent_result.target_agent,
            task=intent_result.summary,
            params={"user_input": ""},  # 由 delegate_node 注入
        )

        needs_review = (
            intent_result.urgency == "high"
            or intent_result.category == "ticket_request"
        )

        return ExecutionPlan(
            steps=[step],
            parallel_groups=[[0]],
            needs_human_review=needs_review,
            reasoning=(
                f"高置信度({intent_result.confidence:.0%})匹配: "
                f"{intent_result.target_agent}"
            ),
        )

    async def _llm_plan(
        self,
        intent_result: IntentResult,
        user_input: str,
    ) -> ExecutionPlan:
        """LLM 规划：复杂/低置信度场景"""
        try:
            # 获取 Agent 列表
            agent_list = "（从注册中心获取）"
            if self.agent_registry:
                agent_list = self.agent_registry.get_routing_descriptions()

            intent_info = (
                f"类别: {intent_result.category}\n"
                f"紧急度: {intent_result.urgency}\n"
                f"置信度: {intent_result.confidence:.0%}\n"
                f"摘要: {intent_result.summary}\n"
                f"推荐Agent: {intent_result.target_agent or '无'}"
            )

            chain = self.plan_prompt | self.llm
            response = await chain.ainvoke({
                "agent_list": agent_list,
                "intent_info": intent_info,
                "user_input": user_input,
            })

            content = response.content.strip()
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0].strip()
            elif "```" in content:
                content = content.split("```")[1].split("```")[0].strip()

            data = json.loads(content)
            steps = [
                PlanStep(
                    agent_id=s.get("agent_id", ""),
                    task=s.get("task", ""),
                    params=s.get("params", {}),
                    depends_on=s.get("depends_on"),
                )
                for s in data.get("steps", [])
            ]

            return ExecutionPlan(
                steps=steps,
                parallel_groups=[[i] for i in range(len(steps))],
                needs_human_review=data.get("needs_human_review", False),
                reasoning=data.get("reasoning", ""),
            )

        except Exception as e:
            logger.error(f"LLM 规划失败: {e}，使用简单兜底")
            return ExecutionPlan(
                steps=[],
                parallel_groups=[],
                needs_human_review=False,
                reasoning=f"规划失败: {str(e)}",
            )

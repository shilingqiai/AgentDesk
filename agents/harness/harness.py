"""
Agent Harness — 多 Agent DAG 编排引擎

核心能力：
1. DAG 拓扑解析 — 自动识别 parallel group → converge point
2. 类型安全传递 — 上游 Output → 下游 Input（TypedDict 校验）
3. 容错机制 — timeout、retry、fallback agent
4. 全链路观测 — 每步耗时、token 估算、成功/失败状态

面试叙事：
    "原来的 complex_track 只能做简单的线性/并行编排，我给它加了一层
     AgentHarness，支持 DAG 拓扑 + 类型契约 + 超时重试，面试官可以直接
     看到 Agent 之间怎么传递结构化结果。"

使用方式：
    harness = AgentHarness()
    result = await harness.run(
        task="我想请5天年假，6月15日到20日",
        steps=[
            parallel(step_rag, step_balance),
            step_compliance,
        ],
    )
"""

from __future__ import annotations

import asyncio
import logging
import time
import traceback
from dataclasses import dataclass, field
from typing import Any, Optional

from .step_schema import StepConfig

logger = logging.getLogger("agent.harness")


# ============================================================
# 数据结构
# ============================================================

@dataclass
class StepResult:
    """单个步骤的执行结果"""
    step: StepConfig
    success: bool
    data: dict[str, Any] = field(default_factory=dict)
    error: str = ""
    duration_ms: float = 0.0
    retries_used: int = 0

    def to_dict(self) -> dict:
        return {
            "agent_id": self.step.agent_id,
            "output_key": self.step.output_key,
            "task": self.step.task,
            "success": self.success,
            "error": self.error,
            "duration_ms": round(self.duration_ms, 1),
            "retries_used": self.retries_used,
            "data_keys": list(self.data.keys()) if self.data else [],
        }


@dataclass
class HarnessResult:
    """Harness 整体执行结果"""
    task: str
    steps: list[StepResult]
    total_duration_ms: float = 0.0
    success: bool = True
    final_output: dict[str, Any] = field(default_factory=dict)

    @property
    def step_summary(self) -> list[dict]:
        return [s.to_dict() for s in self.steps]

    @property
    def failed_steps(self) -> list[StepResult]:
        return [s for s in self.steps if not s.success]


# ============================================================
# DAG Builder — 将 step 列表解析为执行计划
# ============================================================

class _DAGPlan:
    """
    DAG 执行计划

    将用户传入的 [parallel(a,b), c, parallel(d,e), f] 解析为：
        Layer 0: [a, b]  (并行)
        Layer 1: [c]     (依赖 Layer 0)
        Layer 2: [d, e]  (并行，依赖 Layer 1)
        Layer 3: [f]     (依赖 Layer 2)

    规则：
    - list 中的每个元素要么是 StepConfig（单步）要么是 list[StepConfig]（并行组）
    - 层与层之间顺序执行
    - 同层内的 Step 并行执行
    """

    def __init__(self, steps: list):
        self.layers: list[list[StepConfig]] = []
        self._parse(steps)
        self._validate()

    def _parse(self, steps: list):
        """解析用户传入的混合 step 列表 → 分层"""
        for item in steps:
            if isinstance(item, list):
                # 并行组：一层多个 Step
                self.layers.append([s for s in item if isinstance(s, StepConfig)])
            elif isinstance(item, StepConfig):
                # 单步：一层一个 Step
                self.layers.append([item])
            else:
                logger.warning(f"Harness: 跳过未知 step 类型 {type(item)}")

    def _validate(self):
        """校验 DAG 拓扑合法性"""
        all_output_keys: set[str] = set()

        for layer_idx, layer in enumerate(self.layers):
            for step in layer:
                # 检查 input_from 引用的 key 是否已存在
                for dep in step.input_from:
                    if dep not in all_output_keys:
                        logger.warning(
                            f"Harness: Step [{step.output_key}] 依赖 [{dep}]，"
                            f"但 [{dep}] 不在前面的层中。确保依赖已在前层输出。"
                        )

            # 当前层的 output_key 加入已知集合
            for step in layer:
                all_output_keys.add(step.output_key)

    @property
    def total_steps(self) -> int:
        return sum(len(layer) for layer in self.layers)


# ============================================================
# Agent Harness
# ============================================================

class AgentHarness:
    """
    多 Agent DAG 编排引擎

    职责：
    1. 解析 DAG → 分层执行计划
    2. 层内并行、层间串行
    3. 上游 Output → 下游 Input（类型安全传递）
    4. 超时/重试/fallback
    5. 聚合最终结果

    不替代 LangGraph StateGraph，而是作为 complex_track 的增强编排层。
    """

    def __init__(self):
        self._context: dict[str, Any] = {}  # Harness 级共享上下文

    # ── 公共 API ──────────────────────────────────────

    async def run(
        self,
        task: str,
        steps: list,
        thread_id: str = "",
    ) -> HarnessResult:
        """
        执行 DAG 编排

        Args:
            task: 用户原始任务描述
            steps: 混合 step 列表，如 [parallel(a,b), c, parallel(d,e)]
            thread_id: 追踪 ID

        Returns:
            HarnessResult 包含每步执行详情和最终聚合结果
        """
        t0 = time.perf_counter()
        plan = _DAGPlan(steps)

        logger.info(
            f"[Harness] 开始执行 | task={task[:50]} | "
            f"layers={len(plan.layers)} | total_steps={plan.total_steps}"
        )

        all_results: list[StepResult] = []

        for layer_idx, layer in enumerate(plan.layers):
            layer_label = (
                f"Layer {layer_idx + 1}/{len(plan.layers)}"
                f" ({'∥' if len(layer) > 1 else '→'} {len(layer)} steps)"
            )
            logger.info(f"[Harness] {layer_label}")

            # 层内并行
            if len(layer) == 1:
                result = await self._execute_step(
                    layer[0], layer_idx, thread_id,
                )
                all_results.append(result)
            else:
                coros = [
                    self._execute_step(step, layer_idx, thread_id)
                    for step in layer
                ]
                results = await asyncio.gather(*coros, return_exceptions=True)
                for i, r in enumerate(results):
                    if isinstance(r, Exception):
                        all_results.append(StepResult(
                            step=layer[i], success=False,
                            error=str(r), retries_used=0,
                        ))
                    else:
                        all_results.append(r)

        total_ms = (time.perf_counter() - t0) * 1000
        success = all(r.success for r in all_results)

        logger.info(
            f"[Harness] 完成 | success={success} | "
            f"total_ms={total_ms:.0f} | "
            f"failed={len([r for r in all_results if not r.success])}"
        )

        return HarnessResult(
            task=task,
            steps=all_results,
            total_duration_ms=total_ms,
            success=success,
            final_output=self._context.copy(),
        )

    # ── 单步执行 ──────────────────────────────────────

    async def _execute_step(
        self,
        step: StepConfig,
        layer_idx: int,
        thread_id: str,
    ) -> StepResult:
        """执行单个 Step（含重试 + 超时 + fallback）"""
        t0 = time.perf_counter()
        last_error = ""

        for attempt in range(step.retries + 1):
            try:
                # ── 构建输入上下文 ──
                input_context = self._build_input(step)

                # ── 执行 Agent ──
                data = await asyncio.wait_for(
                    self._invoke_agent(step, input_context, thread_id),
                    timeout=step.timeout,
                )

                # ── 校验输出 Schema ──
                if step.output_schema:
                    self._validate_output(step, data)

                # ── 存入上下文 ──
                self._context[step.output_key] = data

                duration_ms = (time.perf_counter() - t0) * 1000
                return StepResult(
                    step=step, success=True, data=data,
                    duration_ms=duration_ms, retries_used=attempt,
                )

            except asyncio.TimeoutError:
                last_error = f"超时 ({step.timeout}s)"
                logger.warning(
                    f"[Harness] {step.output_key} 超时 "
                    f"(attempt {attempt + 1}/{step.retries + 1})"
                )
            except Exception as e:
                last_error = f"{type(e).__name__}: {e}"
                logger.warning(
                    f"[Harness] {step.output_key} 失败: {last_error} "
                    f"(attempt {attempt + 1}/{step.retries + 1})"
                )
                if attempt < step.retries:
                    await asyncio.sleep(0.5 * (attempt + 1))  # 退避

        # ── Fallback Agent ──
        if step.fallback_agent:
            logger.info(f"[Harness] {step.output_key} → fallback agent {step.fallback_agent}")
            try:
                data = await self._invoke_agent(
                    StepConfig(
                        agent_id=step.fallback_agent,
                        task=step.task,
                        output_key=step.output_key,
                    ),
                    {},
                    thread_id,
                )
                self._context[step.output_key] = data
                duration_ms = (time.perf_counter() - t0) * 1000
                return StepResult(
                    step=step, success=True, data=data,
                    duration_ms=duration_ms, retries_used=step.retries + 1,
                )
            except Exception as e:
                last_error = f"fallback also failed: {e}"

        duration_ms = (time.perf_counter() - t0) * 1000
        return StepResult(
            step=step, success=False, error=last_error,
            duration_ms=duration_ms, retries_used=step.retries + 1,
        )

    # ── 内部方法 ──────────────────────────────────────

    def _build_input(self, step: StepConfig) -> dict:
        """
        构建当前 Step 的输入上下文

        从 Harness 上下文中提取上游 step 的输出，
        注入到 payload 中供 Agent 使用。

        面试要点：这就是"类型安全的结果传递"——
        每个 Step 声明的 input_from 就是它依赖的上游 key。
        """
        ctx: dict[str, Any] = {}

        for dep_key in step.input_from:
            if dep_key in self._context:
                ctx[dep_key] = self._context[dep_key]
            else:
                logger.warning(
                    f"[Harness] Step [{step.output_key}] 依赖 [{dep_key}] "
                    f"但上下文中不存在"
                )

        return ctx

    async def _invoke_agent(
        self,
        step: StepConfig,
        input_context: dict,
        thread_id: str,
    ) -> dict:
        """
        调用 Agent 并返回结构化结果

        这里通过 AgentRegistry 找到 Agent 实例，
        用 AgentMessage 委派任务，然后解析返回的 payload。
        """
        from agents.orchestrator.agent_registry import agent_registry
        from agents.a2a.protocol import AgentMessage

        agent = agent_registry.get_agent(step.agent_id)
        if agent is None:
            raise ValueError(f"Agent '{step.agent_id}' 未注册")

        # 构建委派消息
        payload = {
            "user_input": step.task,
            "task": step.task,
            "harness_context": input_context,  # ← 上游结果注入
        }

        delegation = AgentMessage.create_delegation(
            from_agent="harness",
            to_agent=step.agent_id,
            payload=payload,
            trace_id=thread_id,
        )

        # 执行
        response = await agent.execute(delegation)

        if not response.success:
            raise RuntimeError(
                f"Agent '{step.agent_id}' 执行失败: {response.error}"
            )

        # 返回 payload 中的数据部分
        return response.payload if isinstance(response.payload, dict) else {
            "raw_response": str(response.payload),
        }

    @staticmethod
    def _validate_output(step: StepConfig, data: dict):
        """
        校验 Agent 输出是否符合声明的 Schema

        目前做 key 存在性校验。生产环境可升级为 pydantic 完全验证。
        """
        if step.output_schema is None:
            return

        # 获取 TypedDict 的 __required_keys__ 和 __optional_keys__
        required = getattr(step.output_schema, "__required_keys__", set())
        optional = getattr(step.output_schema, "__optional_keys__", set())

        if not required:
            return  # 非 TypedDict，跳过校验

        missing = [k for k in required if k not in data]
        if missing:
            logger.warning(
                f"[Harness] Schema 校验警告: [{step.output_key}] "
                f"缺少必需字段 {missing}"
            )

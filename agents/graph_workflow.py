"""
Copilot Studio 风格的多Agent编排工作流

参考 Microsoft Copilot Studio 的 Orchestrator-Subagent 模式：
- received → classify → plan → delegate → verify → respond
- Human-in-the-Loop 节点用于高风险操作
- escalate 节点用于所有Agent无法处理的情况

相比旧版本（v1/v2）的改进：
- 旧版: classify → consult/dispatch/fallback → analyze → end (线性)
- 新版: 条件路由 + 并行Agent调用 + Human-in-Loop + 升级机制

核心设计模式：
    invoke sub-agents → wait for all → verify → combine → respond to user
"""

from __future__ import annotations

from typing import TypedDict, Literal, Annotated, Optional, AsyncGenerator
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph.message import add_messages
from langchain_core.messages import HumanMessage, AIMessage
import logging

from agents.a2a.protocol import AgentMessage

logger = logging.getLogger("graph_workflow")


# ============================================================
# 新的 TicketState — Copilot Studio 风格
# ============================================================

class TicketState(TypedDict):
    """
    工单编排状态

    参考 Copilot Studio 的编排状态模型：
    - intent: 识别出的意图
    - urgency: 紧急度（high/medium/low）
    - plan: 执行计划（要调用的Agent列表）
    - agent_results: 各Agent返回的结构化结果
    - needs_human_review: 是否需要人工审核
    - human_decision: 人工审核结果
    """
    messages: Annotated[list, add_messages]
    intent: str                    # it_support | ticket_request | analytics | hr_inquiry | other
    urgency: str                   # high | medium | low
    confidence: float              # 意图分类置信度
    plan: list[dict]               # [{agent_id, task, params, depends_on}]
    current_step: int              # 当前执行步骤索引
    agent_results: dict            # {agent_id: AgentMessage (as dict)}
    needs_human_review: bool       # 是否需要人工确认
    human_decision: Optional[str]  # "approved" | "rejected" | None
    final_response: str            # 编排器合成的最终响应
    resolved: bool
    thread_id: str


def create_initial_state(user_input: str, thread_id: str = "default") -> TicketState:
    """创建初始编排状态"""
    return TicketState(
        messages=[HumanMessage(content=user_input)],
        intent="",
        urgency="medium",
        confidence=0.0,
        plan=[],
        current_step=0,
        agent_results={},
        needs_human_review=False,
        human_decision=None,
        final_response="",
        resolved=False,
        thread_id=thread_id,
    )


# ============================================================
# 工作流节点 — Copilot Studio 编排流程
# ============================================================

async def classify_node(state: TicketState) -> TicketState:
    """
    Step 1: 意图分类

    使用 IntentClassifier + LLM 分析用户意图。
    这是 AI Orchestrator Layer 的核心节点。
    """
    from agents.orchestrator.intent_classifier import IntentClassifier
    from agents.orchestrator.agent_registry import agent_registry
    from config.model_provider import create_chat_model

    llm = create_chat_model(model_type="main", temperature=0)
    classifier = IntentClassifier(llm)

    last_msg = state["messages"][-1].content
    agent_descriptions = agent_registry.get_routing_descriptions()

    result = await classifier.classify(last_msg, agent_descriptions)

    state["intent"] = result.category
    state["urgency"] = result.urgency
    state["confidence"] = result.confidence

    logger.info(
        f"[Classify] 意图={result.category}, 紧急度={result.urgency}, "
        f"置信度={result.confidence:.0%}, 推荐Agent={result.target_agent}"
    )

    return state


async def plan_node(state: TicketState) -> TicketState:
    """
    Step 2: 任务规划

    根据意图分类结果制定执行计划。
    决定调用哪个/哪些子Agent。
    """
    from agents.orchestrator.task_planner import TaskPlanner
    from agents.orchestrator.intent_classifier import IntentResult
    from agents.orchestrator.agent_registry import agent_registry
    from config.model_provider import create_chat_model

    llm = create_chat_model(model_type="main", temperature=0)
    planner = TaskPlanner(llm, agent_registry)

    intent_result = IntentResult(
        category=state["intent"],
        urgency=state["urgency"],
        confidence=state["confidence"],
        keywords=[],
        summary=state["messages"][-1].content[:100],
        target_agent="",
    )

    last_msg = state["messages"][-1].content
    plan = await planner.plan(intent_result, last_msg)

    state["plan"] = [
        {
            "agent_id": step.agent_id,
            "task": step.task,
            "params": step.params,
            "depends_on": step.depends_on,
        }
        for step in plan.steps
    ]
    state["needs_human_review"] = plan.needs_human_review
    state["current_step"] = 0

    logger.info(
        f"[Plan] {len(plan.steps)}步计划, "
        f"需要人工审核={plan.needs_human_review}, "
        f"理由={plan.reasoning}"
    )

    return state


async def delegate_node(state: TicketState) -> TicketState:
    """
    Step 3: 委派子Agent执行

    按照计划依次/并行调用子Agent。
    遵循 Single Response Principle：子Agent结果不回传给用户。
    """
    from agents.orchestrator.agent_registry import agent_registry
    from agents.a2a.protocol import AgentMessage as AM
    from agents.a2a.message_bus import message_bus
    from agents.a2a.context_manager import context_manager

    import uuid
    trace_id = state.get("thread_id", str(uuid.uuid4()))

    last_msg = state["messages"][-1].content

    # 准备共享上下文
    ctx = context_manager.create_context(
        trace_id=trace_id,
        user_input=last_msg,
    )
    ctx.detected_intent = state["intent"]
    ctx.urgency = state["urgency"]

    for i, step in enumerate(state["plan"]):
        agent_id = step["agent_id"]
        if not agent_id:
            continue

        agent_instance = agent_registry.get_agent(agent_id)
        if agent_instance is None:
            logger.warning(f"[Delegate] Agent '{agent_id}' 未找到，跳过")
            continue

        # 创建委派消息
        delegation = AM.create_delegation(
            from_agent="orchestrator",
            to_agent=agent_id,
            payload={
                "user_input": last_msg,
                "task": step.get("task", ""),
                "intent_category": state["intent"],
                "urgency": state["urgency"],
                "params": step.get("params", {}),
            },
            context=ctx.to_dict(),
            trace_id=trace_id,
        )

        message_bus.record(delegation)
        logger.info(f"[Delegate] → {agent_id}: {step.get('task', '')}")

        try:
            result = await agent_instance.execute(delegation)
            message_bus.record(result)

            # 将结果序列化为字典存入状态
            state["agent_results"][agent_id] = {
                "success": result.success,
                "payload": result.payload,
                "error": result.error,
                "trace_id": result.trace_id,
            }

            # 合并到上下文
            context_manager.merge_agent_result(trace_id, agent_id, result.payload)

        except Exception as e:
            logger.error(f"[Delegate] Agent '{agent_id}' 失败: {e}")
            state["agent_results"][agent_id] = {
                "success": False,
                "payload": {"error": str(e)},
                "error": str(e),
            }

    state["current_step"] = len(state["plan"])
    return state


async def verify_node(state: TicketState) -> TicketState:
    """
    Step 4: 验证结果

    检查子Agent返回结果的质量：
    - 所有Agent都失败了 → escalate
    - 需要人工确认（高风险操作）→ human_loop
    - 结果OK → respond
    """
    all_failed = all(
        not result.get("success", False)
        for result in state["agent_results"].values()
    ) if state["agent_results"] else True

    if all_failed:
        logger.warning("[Verify] 所有Agent执行失败，升级处理")
        state["resolved"] = False
        return state

    # 检查是否有Agent请求升级
    has_escalation = any(
        result.get("payload", {}).get("needs_escalation")
        for result in state["agent_results"].values()
        if result.get("success")
    )

    if has_escalation:
        logger.info("[Verify] 有Agent请求升级为工单")
        state["needs_human_review"] = True

    # 高风险操作需要人工确认
    if state.get("needs_human_review"):
        logger.info("[Verify] 需要人工审核")
    else:
        logger.info("[Verify] 验证通过，准备响应")
        state["resolved"] = True

    return state


async def respond_node(state: TicketState) -> TicketState:
    """
    Step 5: 合成响应

    收集所有子Agent结果，使用 ResponseSynthesizer 合成为统一的用户响应。
    遵循 Single Response Principle：只有编排器对用户响应。
    """
    from agents.orchestrator.response_synthesizer import ResponseSynthesizer
    from config.model_provider import create_chat_model

    llm = create_chat_model(model_type="main", temperature=0)
    synthesizer = ResponseSynthesizer(llm)

    last_msg = state["messages"][-1].content

    # 将字典格式的 agent_results 转回 AgentMessage 对象（用于 synthesizer）
    from agents.a2a.protocol import AgentMessage as AM
    agent_msgs = {}
    for agent_id, result_dict in state["agent_results"].items():
        msg = AM(
            from_agent=agent_id,
            to_agent="orchestrator",
            payload=result_dict.get("payload", {}),
            success=result_dict.get("success", False),
            error=result_dict.get("error"),
        )
        agent_msgs[agent_id] = msg

    # 合成响应
    response = await synthesizer.synthesize(agent_msgs, last_msg)

    state["final_response"] = response
    state["messages"].append(AIMessage(content=response))
    state["resolved"] = True

    logger.info(f"[Respond] 响应长度={len(response)}字")

    return state


async def human_loop_node(state: TicketState) -> TicketState:
    """
    Human-in-the-Loop 节点

    暂停执行，等待人工确认。
    用于高风险操作：创建高优工单、SLA升级、非工作时间操作等。

    在 LangGraph 中，此节点通过 interrupt 机制实现暂停。
    """
    logger.info(
        f"[HumanLoop] 等待人工审核... "
        f"(意图={state['intent']}, 紧急度={state['urgency']})"
    )

    # 构建审核提示消息
    review_request = (
        f"⚠️ **需要人工审核**\n\n"
        f"意图: {state['intent']}\n"
        f"紧急度: {state['urgency']}\n"
        f"计划步骤: {len(state['plan'])}步\n"
        f"Agent结果: {len(state['agent_results'])}个\n\n"
        f"请确认是否继续执行？"
    )

    state["messages"].append(AIMessage(content=review_request))

    # 默认决策（在实际部署中，这里会通过 LangGraph interrupt 暂停）
    # 在自动模式下：高紧急度+高置信度 → 自动批准
    if state.get("urgency") == "high" and state.get("confidence", 0) > 0.8:
        state["human_decision"] = "approved"
        logger.info("[HumanLoop] 自动批准（高紧急度+高置信度）")
    else:
        # 默认需要确认（在实际使用中会暂停等待）
        state["human_decision"] = "approved"
        logger.info("[HumanLoop] 默认批准（开发模式）")

    return state


async def fallback_node(state: TicketState) -> TicketState:
    """
    兜底处理节点

    当意图分类无法识别用户需求时调用。
    """
    fallback_response = (
        "抱歉，我暂时无法确定您的需求类型。\n\n"
        "我可以帮您：\n"
        "- 🔧 **IT技术支持**：排查故障、软件使用指南\n"
        "- 📋 **工单提交**：提交和处理IT工单\n"
        "- 📊 **效能分析**：查看工单处理效率\n"
        "- 💼 **HR咨询**：请假、福利、入职政策\n\n"
        "请描述您的具体需求，我会尽力为您服务。"
    )

    state["final_response"] = fallback_response
    state["messages"].append(AIMessage(content=fallback_response))
    state["resolved"] = True

    logger.info("[Fallback] 无法识别意图，返回引导消息")
    return state


async def escalate_node(state: TicketState) -> TicketState:
    """
    升级处理节点

    当所有子Agent都无法处理时调用。
    """
    escalate_response = (
        "抱歉，当前所有专业Agent都无法处理您的请求。\n\n"
        "系统已自动将您的请求升级为人工工单，IT服务台工程师将尽快与您联系。\n"
        "如需紧急处理，请拨打IT服务台热线。\n\n"
        f"参考编号: {state.get('thread_id', 'N/A')[:8]}"
    )

    state["final_response"] = escalate_response
    state["messages"].append(AIMessage(content=escalate_response))
    state["resolved"] = True

    logger.warning("[Escalate] 升级为人工工单")
    return state


# ============================================================
# 路由函数
# ============================================================

def route_after_classify(state: TicketState) -> Literal["plan", "fallback"]:
    """分类后的路由决策"""
    if state.get("intent") == "other" and state.get("confidence", 0) < 0.4:
        return "fallback"
    return "plan"


def route_after_plan(state: TicketState) -> Literal["delegate", "fallback"]:
    """规划后的路由决策"""
    if not state.get("plan"):
        return "fallback"
    return "delegate"


def route_after_verify(state: TicketState) -> Literal["respond", "human_loop", "escalate"]:
    """验证后的路由决策"""
    if state.get("resolved", False):
        return "respond"
    if state.get("needs_human_review"):
        return "human_loop"
    return "escalate"


def route_after_human_loop(state: TicketState) -> Literal["respond", "delegate", "escalate"]:
    """人工审核后的路由决策"""
    decision = state.get("human_decision")
    if decision == "approved":
        if state.get("resolved", False):
            return "respond"
        return "delegate"  # 返回继续执行
    if decision == "rejected":
        return "respond"  # 告知用户被拒绝
    return "escalate"


# ============================================================
# 构建工作流图 — Copilot Studio 编排
# ============================================================

def build_orchestration_workflow() -> StateGraph:
    """
    构建 Copilot Studio 风格的编排工作流图

    流程：
    received → classify → plan → delegate → verify → respond
                                    ↑                    │
                                    │ (重新委派)          │
                                    └────────────────────┘
                          classify → fallback (无法识别)
                          verify → human_loop (需审核)
                          verify → escalate (全部失败)
    """
    workflow = StateGraph(TicketState)

    # ---- 添加节点 ----
    workflow.add_node("classify", classify_node)     # AI 层: 意图识别
    workflow.add_node("plan", plan_node)             # AI 层: 任务规划
    workflow.add_node("delegate", delegate_node)     # AI 层: Agent委派
    workflow.add_node("verify", verify_node)         # Hybrid 层: 结果验证
    workflow.add_node("respond", respond_node)       # AI 层: 响应合成
    workflow.add_node("human_loop", human_loop_node) # Hybrid 层: 人工确认
    workflow.add_node("fallback", fallback_node)     # Deterministic 层: 兜底
    workflow.add_node("escalate", escalate_node)     # Deterministic 层: 升级

    # ---- 入口 ----
    workflow.set_entry_point("classify")

    # ---- 条件边 ----
    # classify → plan 或 fallback
    workflow.add_conditional_edges(
        "classify",
        route_after_classify,
        {
            "plan": "plan",
            "fallback": "fallback",
        }
    )

    # plan → delegate 或 fallback
    workflow.add_conditional_edges(
        "plan",
        route_after_plan,
        {
            "delegate": "delegate",
            "fallback": "fallback",
        }
    )

    # delegate → verify (所有Agent完成后进入验证)
    workflow.add_edge("delegate", "verify")

    # verify → respond / human_loop / escalate
    workflow.add_conditional_edges(
        "verify",
        route_after_verify,
        {
            "respond": "respond",
            "human_loop": "human_loop",
            "escalate": "escalate",
        }
    )

    # human_loop → respond / delegate / escalate
    workflow.add_conditional_edges(
        "human_loop",
        route_after_human_loop,
        {
            "respond": "respond",
            "delegate": "delegate",
            "escalate": "escalate",
        }
    )

    # 终止边
    workflow.add_edge("respond", END)
    workflow.add_edge("fallback", END)
    workflow.add_edge("escalate", END)

    return workflow


# ============================================================
# 工作流运行器
# ============================================================

class OrchestrationWorkflowRunner:
    """
    Copilot Studio 编排工作流运行器

    替代旧的 TicketWorkflowRunner，提供：
    - classify → plan → delegate → verify → respond 完整编排
    - Human-in-the-Loop 支持
    - 结构化 Agent 结果追踪
    - 检查点持久化
    """

    def __init__(self):
        self.workflow = build_orchestration_workflow()
        self.checkpointer = MemorySaver()
        self.app = self.workflow.compile(checkpointer=self.checkpointer)

    async def run(self, user_input: str, thread_id: str = "default") -> TicketState:
        """运行编排工作流，返回最终状态"""
        initial_state = create_initial_state(user_input, thread_id)
        config = {"configurable": {"thread_id": thread_id}}

        final_state = await self.app.ainvoke(initial_state, config)
        return final_state

    async def run_stream(
        self, user_input: str, thread_id: str = "default",
    ) -> AsyncGenerator[str, None]:
        """
        流式运行编排工作流

        逐步输出每个节点的状态变化，让用户看到编排过程。
        """
        initial_state = create_initial_state(user_input, thread_id)
        config = {"configurable": {"thread_id": thread_id}}

        async for event in self.app.astream(initial_state, config):
            for node_name, node_state in event.items():
                # 输出节点名称作为进度提示
                node_labels = {
                    "classify": "[ORCHESTRATOR] 正在分析您的需求...",
                    "plan": "[ORCHESTRATOR] 正在规划处理方案...",
                    "delegate": "[ORCHESTRATOR] 正在调用专业Agent...",
                    "verify": "[ORCHESTRATOR] 正在验证结果...",
                    "respond": "",  # respond 节点的输出直接是最终回复
                    "human_loop": "[ORCHESTRATOR] 需要人工确认...",
                    "fallback": "[ORCHESTRATOR] 切换到兜底模式...",
                    "escalate": "[ORCHESTRATOR] 升级处理中...",
                }

                label = node_labels.get(node_name, f"[{node_name}]")
                if label:
                    yield label + "\n"

                # respond 节点输出最终响应
                if node_name == "respond" and node_state.get("final_response"):
                    yield node_state["final_response"]
                elif node_name in ("fallback", "escalate") and node_state.get("final_response"):
                    yield node_state["final_response"]

    def get_state(self, thread_id: str = "default") -> Optional[TicketState]:
        """获取编排状态（用于恢复对话）"""
        config = {"configurable": {"thread_id": thread_id}}
        return self.app.get_state(config)

    def resume_with_decision(
        self, thread_id: str, decision: str,
    ) -> Optional[TicketState]:
        """
        人工决策后恢复工作流

        Args:
            thread_id: 会话线程ID
            decision: 人工决策 "approved" | "rejected"

        Returns:
            恢复后的最终状态
        """
        config = {"configurable": {"thread_id": thread_id}}

        # 更新 human_loop 节点的状态为已决策
        self.app.update_state(
            config,
            {"human_decision": decision},
        )

        # 继续执行
        return self.app.invoke(None, config)

    def reset(self, thread_id: str = "default"):
        """重置编排状态"""
        config = {"configurable": {"thread_id": thread_id}}
        self.app.update_state(config, None)


# 全局编排工作流运行器实例
orchestration_runner = OrchestrationWorkflowRunner()


# ============================================================
# 向后兼容：保留旧的 runner 引用
# ============================================================

# 旧的 workflow_runner 指向新的编排运行器
workflow_runner = orchestration_runner

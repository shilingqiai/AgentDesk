"""
LangGraph 工单调度工作流

使用 LangGraph 状态图管理工单完整生命周期：
received → classified → (consulting | dispatching) → resolved

替代原来的手写状态管理（SharedState/StateEnum），提供：
- 类型安全的状态定义
- 条件路由（分类结果决定下一步）
- 检查点持久化（支持对话恢复）
- Agent间状态传递
"""

from typing import TypedDict, Literal, Annotated
from typing import AsyncGenerator, Optional
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph.message import add_messages
from langchain_core.messages import HumanMessage, AIMessage
import logging

logger = logging.getLogger(__name__)


# ============================================================
# 状态定义
# ============================================================

class TicketState(TypedDict):
    """工单状态"""
    messages: Annotated[list, add_messages]
    current_agent: str           # classify | consult | dispatch | analyze
    category: str                # appointment | query | other
    priority: str                # P0 | P1 | P2 | P3
    ticket_info: dict            # 提取的工单信息
    engineer_id: Optional[str]   # 分配的工程师ID
    sla_deadline: Optional[str]  # SLA截止时间
    resolved: bool               # 工单是否已解决


def create_initial_state(user_input: str) -> TicketState:
    """创建初始工单状态"""
    return TicketState(
        messages=[HumanMessage(content=user_input)],
        current_agent="classify",
        category="",
        priority="P2",
        ticket_info={},
        engineer_id=None,
        sla_deadline=None,
        resolved=False,
    )


# ============================================================
# 工作流节点
# ============================================================

async def classify_node(state: TicketState) -> TicketState:
    """分类节点：调用 TaskClassificationAgent 进行意图识别"""
    from agents.task_classification_agent import TaskClassificationAgent
    from agents.appointment_agent import AppointmentAgent
    from agents.consultant_agent import ConsultantAgent

    task_agent = TaskClassificationAgent(
        AppointmentAgent(),
        ConsultantAgent()
    )

    last_msg = state["messages"][-1].content
    result = await task_agent.classify_task(last_msg)

    state["category"] = result if result in ("appointment", "query") else "other"
    state["current_agent"] = "classify"

    logger.info(f"工单分类结果: {state['category']}")
    return state


async def consult_node(state: TicketState) -> TicketState:
    """咨询节点：RAG知识库自服务"""
    from agents.consultant_agent import ConsultantAgent

    last_msg = state["messages"][-1].content

    async with ConsultantAgent() as agent:
        response = await agent.consult(last_msg)
        state["messages"].append(AIMessage(content=response))
        state["current_agent"] = "consult"
        state["resolved"] = True

    logger.info("知识库自服务完成")
    return state


async def dispatch_node(state: TicketState) -> TicketState:
    """调度节点：工单派发"""
    from agents.appointment_agent import AppointmentAgent

    last_msg = state["messages"][-1].content
    agent = AppointmentAgent()

    response_parts = []
    async for token in agent.run_stream(user_input=last_msg):
        response_parts.append(token)

    response = "".join(response_parts)
    state["messages"].append(AIMessage(content=response))
    state["current_agent"] = "dispatch"
    state["resolved"] = True

    logger.info("工单派发完成")
    return state


async def analyze_node(state: TicketState) -> TicketState:
    """分析节点：工程师效能分析"""
    from agents.user_behavior_agent import UserBehaviorAgent

    agent = UserBehaviorAgent()
    analysis = agent.get_user_analysis()
    if analysis:
        state["ticket_info"]["analysis"] = analysis

    state["current_agent"] = "analyze"
    return state


async def fallback_node(state: TicketState) -> TicketState:
    """兜底节点：处理无法分类的请求"""
    reply = "暂不支持该类型任务。请提交与IT工单调度相关的问题。"
    state["messages"].append(AIMessage(content=reply))
    state["current_agent"] = "classify"
    state["resolved"] = True
    return state


# ============================================================
# 路由函数
# ============================================================

def route_by_category(state: TicketState) -> Literal["consult", "dispatch", "fallback"]:
    """根据分类结果路由到对应节点"""
    category = state.get("category", "other")
    if category == "query":
        return "consult"
    elif category == "appointment":
        return "dispatch"
    else:
        return "fallback"


def should_continue(state: TicketState) -> Literal["analyze", "end"]:
    """判断是否需要继续到分析节点"""
    if state.get("resolved", False):
        return "analyze"
    return "end"


# ============================================================
# 构建工作流图
# ============================================================

def build_ticket_workflow() -> StateGraph:
    """构建工单调度工作流图"""

    workflow = StateGraph(TicketState)

    # 添加节点
    workflow.add_node("classify", classify_node)
    workflow.add_node("consult", consult_node)
    workflow.add_node("dispatch", dispatch_node)
    workflow.add_node("analyze", analyze_node)
    workflow.add_node("fallback", fallback_node)

    # 设置入口
    workflow.set_entry_point("classify")

    # 条件边：分类 → 路由
    workflow.add_conditional_edges(
        "classify",
        route_by_category,
        {
            "consult": "consult",
            "dispatch": "dispatch",
            "fallback": "fallback",
        }
    )

    # 处理后进入分析（效能追踪）
    workflow.add_conditional_edges(
        "consult",
        should_continue,
        {"analyze": "analyze", "end": END}
    )
    workflow.add_conditional_edges(
        "dispatch",
        should_continue,
        {"analyze": "analyze", "end": END}
    )

    # 分析和兜底直接结束
    workflow.add_edge("analyze", END)
    workflow.add_edge("fallback", END)

    return workflow


# ============================================================
# 工作流运行器
# ============================================================

class TicketWorkflowRunner:
    """工单工作流运行器 — 替代原来的 chat_handler 手动编排"""

    def __init__(self):
        self.workflow = build_ticket_workflow()
        self.checkpointer = MemorySaver()
        self.app = self.workflow.compile(checkpointer=self.checkpointer)

    async def run(self, user_input: str, thread_id: str = "default") -> TicketState:
        """运行工作流，返回最终状态"""
        initial_state = create_initial_state(user_input)
        config = {"configurable": {"thread_id": thread_id}}

        final_state = await self.app.ainvoke(initial_state, config)
        return final_state

    async def run_stream(self, user_input: str, thread_id: str = "default") -> AsyncGenerator[str, None]:
        """流式运行工作流，逐步返回输出"""
        initial_state = create_initial_state(user_input)
        config = {"configurable": {"thread_id": thread_id}}

        async for event in self.app.astream(initial_state, config):
            for node_name, node_state in event.items():
                yield f"[{node_name}] "
                if node_state.get("messages"):
                    last_msg = node_state["messages"][-1]
                    if hasattr(last_msg, "content"):
                        yield last_msg.content

    def get_state(self, thread_id: str = "default") -> Optional[TicketState]:
        """获取对话状态（用于恢复对话）"""
        config = {"configurable": {"thread_id": thread_id}}
        return self.app.get_state(config)

    def reset(self, thread_id: str = "default"):
        """重置对话"""
        config = {"configurable": {"thread_id": thread_id}}
        self.app.update_state(config, None)


# 全局工作流运行器实例
workflow_runner = TicketWorkflowRunner()

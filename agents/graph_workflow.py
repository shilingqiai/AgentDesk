"""
Copilot Studio 多Agent编排工作流 — Hub & Spoke 三级路由 (v12)

⚠️ 此文件已拆分为 agents/graph/ 子模块。
   本文件保留为向后兼容的 re-export 层，所有旧 import 路径仍然有效。

新模块结构:
    agents/graph/
        state.py          — TicketState + 辅助函数
        routing.py        — route_after_* 条件边分发
        streaming.py      — _yield_stream_event
        nodes/
            router.py     — route_node, re_evaluate_node
            fast.py       — fast_track_node
            action.py     — action_track_node
            complex.py    — complex_track_node
            dynamic.py    — dynamic_action_node, dynamic_interrupt_node, after_*
            terminal.py   — clarification_node, respond_node
        workflow.py       — build_orchestration_workflow, OrchestrationWorkflowRunner

架构 (v8: 4轨):
    用户输入 → Router (语义路由)
                  │
        ┌─────────┼─────────┬──────────┐
        ▼         ▼         ▼          ▼
      fast     dynamic   complex    clarification
      (80%)    (15%)      (5%)      (不确定→反问)
        │         │         │
        ▼         ▼         ▼
    EnterpriseRAG  DynamicAction  TaskPlanner
    (FAISS+LLM)  (ReAct自由编排)  (多Agent委派)

流式输出：
    [THINKING] → 前端显示"思考中..."
    [ROUTE]    → 更新侧边栏路由轨道
    [STREAM]   → 逐字流式输出到对话气泡
    [CARD]     → 确认卡片
    [REACT]    → ReAct 思维链
    [DONE]     → 完成标记
"""

# ============================================================
# 向后兼容 re-exports — 所有旧 import 路径继续有效
# ============================================================

# State
from agents.graph.state import (  # noqa: F401
    TicketState,
    create_initial_state,
    _build_conversation_context,
    _maybe_compress_history,
    _get_user_text,
    _reset_self_help_state,
    _generate_rag_topic,
    _detect_topic_from_history,
    _serialize_messages,
    COMPRESS_THRESHOLD,
    KEEP_RECENT,
)

# Routing
from agents.graph.routing import (  # noqa: F401
    route_after_route,
    after_re_evaluate,
    after_action_track,
)

# Streaming
from agents.graph.streaming import _yield_stream_event  # noqa: F401

# Nodes
from agents.graph.nodes.router import route_node, re_evaluate_node  # noqa: F401
from agents.graph.nodes.fast import fast_track_node  # noqa: F401
from agents.graph.nodes.action import action_track_node  # noqa: F401
from agents.graph.nodes.complex import complex_track_node  # noqa: F401
from agents.graph.nodes.dynamic import (  # noqa: F401
    dynamic_action_node,
    dynamic_interrupt_node,
    after_dynamic_action,
    after_dynamic_interrupt,
)
from agents.graph.nodes.terminal import clarification_node, respond_node  # noqa: F401

# Workflow
from agents.graph.workflow import (  # noqa: F401
    _classify_dynamic_response,
    build_orchestration_workflow,
    OrchestrationWorkflowRunner,
    STREAM_CHUNK_SIZE,
    STREAM_DELAY,
    orchestration_runner,
    workflow_runner,
)

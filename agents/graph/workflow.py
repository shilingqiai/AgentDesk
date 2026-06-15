"""
编排工作流构建器 + 运行器 — Hub & Spoke 主入口

从 agents.graph_workflow 拆分出来。
依赖: state, routing, streaming, 所有 nodes
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import AsyncGenerator

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.types import Command
from langchain_core.messages import AIMessage

from agents.graph.state import (
    TicketState, create_initial_state, _get_user_text,
)
from agents.graph.routing import route_after_route, after_re_evaluate, after_action_track
from agents.graph.streaming import _yield_stream_event
from agents.graph.nodes.router import route_node, re_evaluate_node
from agents.graph.nodes.fast import fast_track_node
from agents.graph.nodes.action import action_track_node
from agents.graph.nodes.complex import complex_track_node
from agents.graph.nodes.dynamic import (
    dynamic_action_node, dynamic_interrupt_node,
    after_dynamic_action, after_dynamic_interrupt,
)
from agents.graph.nodes.terminal import clarification_node, respond_node

logger = logging.getLogger("graph.workflow")


# ============================================================
# 构建工作流图
# ============================================================

async def _classify_dynamic_response(
    user_text: str,
    prev_values: dict,
) -> dict:
    """
    v8: 分类用户对动态卡片的回复意图 (AskUserQuestion 风格)

    当图被 interrupt() 冻结后, 用户通过文本回复(而非按钮),
    需要 LLM 判断意图: confirm / modify / cancel / new_topic

    Returns: {"action": "...", "feedback": "..."}
    """
    from config.model_provider import create_chat_model

    # 获取卡片信息作为分类上下文
    single_card = prev_values.get("dynamic_interrupt_card", {})
    if single_card:
        cards = [single_card]
    else:
        agent_results = prev_values.get("agent_results", {})
        dynamic_result = agent_results.get("dynamic_action", {})
        cards = dynamic_result.get("cards", [])
    card_descriptions = "\n".join(
        f"- {c.get('title', '')}: {c.get('description', '')[:200]}"
        for c in cards
    ) if cards else "(no card info)"

    system_prompt = (
        "你是企业服务台的意图分类器。用户看到了一张或多张确认卡片, 然后回复了一句话。\n"
        "请调用 classify_intent 函数判断用户的意图。\n\n"
        "分类标准:\n"
        "- confirm: 用户确认/同意卡片内容, 要求执行操作。\n"
        "  例: '好的''行''确认''可以''没问题''就这样''yes''ok''confirm'\n"
        "- modify: 用户想修改卡片的某个参数。\n"
        "  例: '把显示器改成LG的''数量改成2台''不要耳机''加一个鼠标'\n"
        "  关键: 仍围绕卡片内容, 但要求调整\n"
        "  ★ '不要X' = modify（移除某物品），不是 cancel！\n"
        "  ★ '不要了''算了''取消' = cancel（放弃整个操作）\n"
        "- cancel: 用户想完全取消/放弃整个操作。\n"
        "  例: '算了''取消''不要了''不用了''不搞了''cancel'\n"
        "  仅当整句表达放弃意图才判 cancel\n"
        "- new_topic: 用户完全换了话题, 与当前卡片无关。\n"
        "  例: '帮我查下请假政策''会议室怎么预定''VPN怎么连'\n"
        "核心判断: 用户的话是否仍然围绕这张卡片?\n"
        "围绕卡片=confirm/modify/cancel, 完全不相关=new_topic。"
    )

    classify_tool = {
        "type": "function",
        "function": {
            "name": "classify_intent",
            "description": "分类用户对确认卡片的回复意图",
            "parameters": {
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": ["confirm", "modify", "cancel", "new_topic"],
                        "description": "用户意图分类",
                    },
                    "feedback": {
                        "type": "string",
                        "description": "如果是modify, 简述用户要求的修改内容; 其他情况为空字符串",
                    },
                },
                "required": ["action"],
            },
        },
    }

    try:
        llm = create_chat_model(model_type="main", temperature=0)
        llm_with_tool = llm.bind_tools([classify_tool], tool_choice="auto")

        response = await llm_with_tool.ainvoke([
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": (
                f"卡片内容:\n{card_descriptions}\n\n"
                f"用户回复: \"{user_text}\"\n\n"
                f"请调用 classify_intent 函数分类用户意图。"
            )},
        ])

        if response.tool_calls:
            args = response.tool_calls[0].get("args", {})
            if isinstance(args, str):
                import json as _json
                args = _json.loads(args)
            return {
                "action": args.get("action", "confirm"),
                "feedback": args.get("feedback", ""),
            }
        else:
            # Fallback: 关键词规则（仅在 LLM tool_call 未触发时使用）
            text_lower = user_text.lower().strip()

            cancel_words = ["取消", "算了", "不要了", "不用了", "cancel", "放弃"]
            if text_lower in ("算了", "取消", "cancel", "不要了", "不用了", "别",
                             "放弃", "算了不搞了"):
                return {"action": "cancel", "feedback": ""}
            if any(text_lower == w for w in cancel_words):
                return {"action": "cancel", "feedback": ""}

            confirm_words = ["确认", "好的", "行", "可以", "是", "yes", "ok", "对", "好"]
            if text_lower in confirm_words or (
                len(text_lower) < 6 and any(text_lower == w for w in confirm_words)
            ):
                return {"action": "confirm", "feedback": ""}

            if len(user_text) < 30 and "?" not in user_text and "？" not in user_text:
                return {"action": "confirm", "feedback": ""}
            else:
                return {"action": "modify", "feedback": user_text}

    except Exception as e:
        logger.error(f"[ClassifyDynamic] 分类失败: {e}，兜底为 confirm")
        return {"action": "confirm", "feedback": ""}


def build_orchestration_workflow() -> StateGraph:
    workflow = StateGraph(TicketState)

    workflow.add_node("route", route_node)
    workflow.add_node("re_evaluate", re_evaluate_node)
    workflow.add_node("fast_track", fast_track_node)
    workflow.add_node("dynamic_action", dynamic_action_node)   # v10: ReAct 自由编排
    workflow.add_node("dynamic_interrupt", dynamic_interrupt_node)  # v10: 中断确认
    workflow.add_node("action_track", action_track_node)
    workflow.add_node("complex_track", complex_track_node)
    workflow.add_node("clarification", clarification_node)
    workflow.add_node("respond", respond_node)

    workflow.set_entry_point("route")

    # route → 六路分发
    workflow.add_conditional_edges("route", route_after_route, {
        "re_evaluate": "re_evaluate",
        "fast_track": "fast_track",
        "dynamic_action": "dynamic_action",
        "action_track": "action_track",
        "complex_track": "complex_track",
        "clarification": "clarification",
    })

    # re_evaluate → 五路分发
    workflow.add_conditional_edges("re_evaluate", after_re_evaluate, {
        "action_track": "action_track",
        "dynamic_action": "dynamic_action",
        "fast_track": "fast_track",
        "route": "route",
        "respond": "respond",
    })

    workflow.add_edge("fast_track", "respond")
    workflow.add_conditional_edges("dynamic_action", after_dynamic_action, {
        "dynamic_action": "dynamic_action",       # v10: self-loop 继续迭代
        "dynamic_interrupt": "dynamic_interrupt",  # v10: 中断确认
        "respond": "respond",
    })
    workflow.add_conditional_edges("dynamic_interrupt", after_dynamic_interrupt, {
        "dynamic_action": "dynamic_action",  # v10: 确认后继续推理
        "respond": "respond",
    })
    workflow.add_conditional_edges("action_track", after_action_track, {
        "route": "route",
        "respond": "respond",
    })
    workflow.add_edge("complex_track", "respond")
    workflow.add_edge("clarification", "respond")
    workflow.add_edge("respond", END)

    return workflow


# ============================================================
# 工作流运行器 — 带流式输出
# ============================================================

STREAM_CHUNK_SIZE = 3
STREAM_DELAY = 0.025


class OrchestrationWorkflowRunner:
    """Hub & Spoke 编排工作流运行器 (v3)"""

    def __init__(self):
        self._ensure_agents_loaded()
        self.workflow = build_orchestration_workflow()
        self.checkpointer = None  # 懒加载，首调时在事件循环内初始化
        self.app = None

    async def _ensure_app(self):
        """
        懒加载 AsyncSqliteSaver + 编译图。

        必须在事件循环内首次调用（run / run_stream 入口自动触发）。
        init 阶段不创建异步资源，避免 Windows ProactorEventLoop 导入时挂起。
        """
        if self.app is not None:
            return

        import aiosqlite
        import os as _os

        _os.makedirs("data", exist_ok=True)
        conn = await aiosqlite.connect("data/checkpoints.db")
        self.checkpointer = AsyncSqliteSaver(conn)
        self.app = self.workflow.compile(checkpointer=self.checkpointer)
        logger.info("AsyncSqliteSaver 初始化完成 — 会话状态持久化到 data/checkpoints.db")

    @staticmethod
    def _ensure_agents_loaded():
        """确保子Agent模块已导入并注册"""
        try:
            import agents.sub_agents.enterprise_rag     # noqa: F401
            import agents.sub_agents.ticket_dispatch    # noqa: F401
            import agents.sub_agents.tool_agent         # noqa: F401
            import agents.sub_agents.dynamic_action_agent  # noqa: F401
            import agents.tools.builtin_tools           # noqa: F401
        except ImportError as e:
            logger.warning(f"Agent 模块加载警告: {e}")

    async def run(self, user_input: str, thread_id: str = "default",
                  user_name: str = "", role: str = "employee") -> TicketState:
        await self._ensure_app()
        initial_state = create_initial_state(
            user_input, thread_id, user_name=user_name, role=role,
        )
        config = {"configurable": {"thread_id": thread_id}}

        # 从 checkpointer 恢复跨 turn 状态
        prev = await asyncio.to_thread(self.app.get_state, config)
        if prev and prev.values:
            if prev.values.get("conversation_phase") == "self_help_provided":
                initial_state["conversation_phase"] = "self_help_provided"
                initial_state["last_rag_topic"] = prev.values.get("last_rag_topic", "")
                initial_state["last_rag_summary"] = prev.values.get("last_rag_summary", "")
                initial_state["last_track_type"] = prev.values.get("last_track_type", "")
            if prev.values.get("conversation_summary"):
                initial_state["conversation_summary"] = prev.values["conversation_summary"]
            if not initial_state["user_name"] and prev.values.get("user_name"):
                initial_state["user_name"] = prev.values["user_name"]
                initial_state["role"] = prev.values.get("role", "employee")
            # v12: 恢复 dynamic agent 跨 turn 状态（follow_up 无卡片锁场景）
            if prev.values.get("last_track_type") == "dynamic":
                initial_state["dynamic_agent_messages"] = prev.values.get(
                    "dynamic_agent_messages", [],
                )
                initial_state["dynamic_iteration"] = prev.values.get(
                    "dynamic_iteration", 0,
                )

        return await self.app.ainvoke(initial_state, config)

    async def run_stream(
        self, user_input: str, thread_id: str = "default",
        user_name: str = "", role: str = "employee",
    ) -> AsyncGenerator[str, None]:
        """
        流式运行编排工作流。

        输出令牌：
          [THINKING] <文字>            — 更新"思考中..."文字
          [ROUTE] <轨道描述>           — 路由判定结果
          [CLARIFY] <反问文字>         — AI 反问用户澄清意图
          [FAST]/[ACTION]/[COMPLEX]    — 轨道入口
          [STREAM]<文字片段>           — 流式回答片段
          [CARD]<JSON>                 — 确认卡片
          [DONE]                       — 完成
        """
        initial_state = create_initial_state(user_input, thread_id,
                                            user_name=user_name, role=role)
        config = {"configurable": {"thread_id": thread_id}}

        # ── 懒加载 checkpointer + 编译图 ──
        await self._ensure_app()

        # ── 从 checkpointer 恢复状态 ──
        prev = await asyncio.to_thread(self.app.get_state, config)
        pending_card = ""
        if prev and prev.values:
            pending_card = prev.values.get("pending_card_type", "")

            if prev.values.get("conversation_phase") == "self_help_provided":
                initial_state["conversation_phase"] = "self_help_provided"
                initial_state["last_rag_topic"] = prev.values.get("last_rag_topic", "")
                initial_state["last_rag_summary"] = prev.values.get("last_rag_summary", "")
                initial_state["last_track_type"] = prev.values.get("last_track_type", "")
                logger.info(
                    f"[Stream] 恢复 self_help_provided, "
                    f"topic={initial_state['last_rag_topic']}, "
                    f"last_track={initial_state['last_track_type']}"
                )
            if pending_card:
                prev_agent_results = prev.values.get("agent_results", {})
                if prev_agent_results:
                    initial_state["agent_results"] = prev_agent_results
                if pending_card.startswith("dynamic_"):
                    initial_state["dynamic_agent_messages"] = prev.values.get(
                        "dynamic_agent_messages", [],
                    )
                    initial_state["dynamic_agent_proposals"] = prev.values.get(
                        "dynamic_agent_proposals", {},
                    )
                    logger.info(
                        f"[Stream] 恢复 dynamic agent 状态: "
                        f"{len(initial_state['dynamic_agent_messages'])} msgs, "
                        f"{len(initial_state['dynamic_agent_proposals'])} proposals"
                    )

            if prev.values.get("last_track_type") == "dynamic" and not pending_card:
                initial_state["dynamic_agent_messages"] = prev.values.get(
                    "dynamic_agent_messages", [],
                )
                initial_state["dynamic_iteration"] = prev.values.get(
                    "dynamic_iteration", 0,
                )
                logger.info(
                    f"[Stream] 恢复 dynamic agent 状态 (follow_up 无卡片锁): "
                    f"{len(initial_state['dynamic_agent_messages'])} msgs, "
                    f"iter={initial_state['dynamic_iteration']}"
                )

            if not initial_state["user_name"] and prev.values.get("user_name"):
                initial_state["user_name"] = prev.values["user_name"]
                initial_state["role"] = prev.values.get("role", "employee")

            if prev.values.get("conversation_summary"):
                initial_state["conversation_summary"] = prev.values["conversation_summary"]
                logger.info(
                    f"[Stream] 恢复 conversation_summary: "
                    f"{len(initial_state['conversation_summary'])} 字"
                )

        if pending_card:
            # ── v8/v9: dynamic_* → 图已冻结, 需分类意图后 Command(resume) ──
            if pending_card in ("dynamic_confirm:interrupt", "dynamic_interrupt"):
                decision = await _classify_dynamic_response(
                    user_input, prev.values,
                )
                logger.info(
                    f"[Stream:v8] dynamic_confirm:interrupt → "
                    f"classified as {decision.get('action')}"
                )

                if decision.get("action") == "new_topic":
                    await self.app.ainvoke(
                        Command(resume={"action": "cancel"}), config,
                    )
                    yield f"[ROUTE] 🔄 Switching topic...\n"
                    yield "[THINKING] 🔍 Analyzing new request...\n"
                    initial_state["pending_card_type"] = ""
                else:
                    yield f"[ROUTE] 📋 Processing ({decision.get('action')})...\n"
                    yield "[THINKING] 🔍 Resuming agent...\n"

                    async for event in self.app.astream(
                        Command(resume=decision), config,
                        stream_mode=["updates", "custom"],
                    ):
                        if isinstance(event, tuple) and len(event) == 2:
                            mode, data = event
                            if mode == "custom":
                                if isinstance(data, str) and (
                                    data.startswith("[REACT]") or data.startswith("[CARD]")
                                    or data.startswith("[THINKING]")
                                ):
                                    yield f"{data}\n"
                                else:
                                    yield f"[STREAM]{data}\n"
                                continue
                            elif mode == "updates":
                                for node_name, node_state in data.items():
                                    for _y in _yield_stream_event(node_name, node_state):
                                        yield _y
                        else:
                            for node_name, node_state in event.items():
                                for _y in _yield_stream_event(node_name, node_state):
                                    yield _y

                    resumed_state = await asyncio.to_thread(self.app.get_state, config)
                    if resumed_state and resumed_state.interrupts:
                        yield "[INTERRUPT]\n"
                    else:
                        yield "[DONE]\n"
                    return

            elif pending_card.startswith("dynamic_confirm:"):
                initial_state["pending_card_type"] = pending_card
                yield f"[ROUTE] 📋 Processing your confirmation...\n"
                yield "[THINKING] 🔍 Resuming agent with your approval...\n"

            else:
                # ── 旧卡片锁: 直送 action_track ──
                initial_state["pending_card_type"] = pending_card
                initial_state["track"] = "card_lock"

                yield f"[ROUTE] 📋 处理您对「{pending_card}」卡片的回复\n"
                yield "[THINKING] 🔍 正在理解您的意图...\n"

                async for event in self.app.astream(initial_state, config):
                    for node_name, node_state in event.items():
                        if node_name == "action_track":
                            yield "[ACTION] ⚡ 动作通道 · 处理卡片回复\n"
                            if node_state.get("re_route"):
                                yield "[ROUTE] 🔄 切换话题，重新分析...\n"
                                yield "[THINKING] 🔍 正在路由到对应轨道...\n"
                        for _y in _yield_stream_event(node_name, node_state):
                            yield _y

                yield "[DONE]\n"
                return

        # ================================================================
        # 非锁模式：graph 全权接管
        # ================================================================
        yield "[THINKING] 🔍 正在分析您的问题...\n"

        async for event in self.app.astream(
            initial_state, config, stream_mode=["updates", "custom"],
        ):
            if isinstance(event, tuple) and len(event) == 2:
                mode, data = event
                if mode == "custom":
                    if isinstance(data, str) and (
                        data.startswith("[REACT]") or data.startswith("[CARD]")
                        or data.startswith("[THINKING]")
                    ):
                        yield f"{data}\n"
                    else:
                        yield f"[STREAM]{data}\n"
                    continue
                elif mode == "updates":
                    for node_name, node_state in data.items():
                        for _y in _yield_stream_event(node_name, node_state):
                            yield _y
            else:
                for node_name, node_state in event.items():
                    for _y in _yield_stream_event(node_name, node_state):
                        yield _y

        # v8: 检查 graph 是否被 interrupt() 冻结
        final_state = await asyncio.to_thread(self.app.get_state, config)
        if final_state and final_state.interrupts:
            yield "[INTERRUPT]\n"
            logger.info(
                f"[Stream:v8] Graph interrupted — {len(final_state.interrupts)} interrupt(s) pending"
            )
        else:
            yield "[DONE]\n"

    async def get_state(self, thread_id: str = "default"):
        """获取会话状态。"""
        await self._ensure_app()
        config = {"configurable": {"thread_id": thread_id, "checkpoint_ns": ""}}
        return await asyncio.to_thread(self.app.get_state, config)

    async def reset(self, thread_id: str = "default"):
        """重置会话状态。"""
        await self._ensure_app()
        config = {"configurable": {"thread_id": thread_id, "checkpoint_ns": ""}}
        await asyncio.to_thread(self.app.update_state, config, None)


# 全局实例
orchestration_runner = OrchestrationWorkflowRunner()
workflow_runner = orchestration_runner

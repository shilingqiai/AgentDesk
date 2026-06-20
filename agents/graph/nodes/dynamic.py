"""
动态动作节点 — ReAct 循环 + 中断确认

v10: 真正的中途中断 — 每次图节点执行 = 一次 LLM 调用 + 工具执行。
create_ticket 返回 proposed 时设置 pending_card，由 dynamic_interrupt_node 调用 interrupt()。

从 agents.graph_workflow 拆分出来。
依赖: state.py, agents.sub_agents.dynamic_action_agent
"""

from __future__ import annotations

import asyncio
import json
import logging
import time as _time

from langgraph.config import get_stream_writer
from langgraph.types import interrupt

from agents.graph.state import (
    TicketState, _get_user_text, _build_conversation_context, _generate_rag_topic,
    _sh_set,
    _generate_rag_topic, _detect_topic_from_history,
)

logger = logging.getLogger("graph.nodes.dynamic")

# v12: 执行追踪记录
from services.trace_store import TraceStore

# ══ 延迟诊断开关 ══
_LATENCY_DEBUG = True


async def dynamic_action_node(state: TicketState) -> TicketState:
    """
    动态动作节点 (v10) — 真正的中途中断 ReAct

    每次图节点执行 = 一次 LLM 调用 + 工具执行。
    如果 create_ticket 返回 proposed → 设置 pending_card 后返回，
    由独立的 dynamic_interrupt_node 调用 interrupt() 冻结图。

    自循环边: dynamic_action → dynamic_action（继续迭代）
    中断边:   dynamic_action → dynamic_interrupt → dynamic_action（确认后继续）
    终结边:   dynamic_action → respond（LLM 完成）
    """
    from agents.sub_agents.dynamic_action_agent import DynamicActionAgent

    writer = get_stream_writer()
    agent = DynamicActionAgent()
    agent._proposals = {}  # v10: 每次迭代都初始化，供 _tool_create_ticket 存储卡片
    user_text = _get_user_text(state)

    # v12-fix: 每次迭代都设置用户身份（self-loop 创建新实例，仅 is_first 设置会丢失）
    agent._last_user_input = user_text
    agent._last_user_name = state.get("user_name", "")
    agent._last_user_role = state.get("role", "employee")
    agent._last_trace_id = state.get("thread_id", "")

    # ================================================================
    # GATE A: 初始化或继续 ReAct
    # ================================================================
    iteration = state.get("dynamic_iteration", 0)
    messages = state.get("dynamic_agent_messages", [])
    is_first = (iteration == 0 and len(messages) == 0)

    if is_first:
        await agent._ensure_inventory_seeded()

        messages = [
            {"role": "system", "content": agent._build_system_prompt(state.get("user_name", ""))},
        ]

        # v11: 结构化 XML 注入 — 隔离历史与当前，明确话题边界
        conv_hist = _build_conversation_context(
            state["messages"], summary=state.get("conversation_summary", ""),
        )
        if conv_hist:
            _last_topic = _detect_topic_from_history(state["messages"])
            history_block = (
                f"<conversation_history>\n"
                f"Previous topic: {_last_topic}\n"
                f"Note: This is for context only. The current request may be on a different topic.\n"
                f"{conv_hist}\n"
                f"</conversation_history>"
            )
            messages.append({"role": "user", "content": history_block})

        messages.append({
            "role": "user",
            "content": (
                f"<current_request>\n{user_text}\n</current_request>\n\n"
                "Process the <current_request> above. "
                "Only use <conversation_history> if it is clearly the SAME topic "
                "(e.g., slot filling where the user is providing missing details). "
                "If <current_request> is a completely different topic, ignore the history."
            ),
        })

        logger.info(
            f"[DynamicAction:v11] FIRST execution — iter={iteration}, "
            f"topic={_detect_topic_from_history(state['messages']) if conv_hist else 'N/A'}"
        )

    if iteration >= agent.MAX_REACT_ITERATIONS:
        logger.warning(f"[DynamicAction:v9] 达到最大迭代 {iteration}，强制总结")
        messages.append({
            "role": "user",
            "content": "已达到最大操作步数。请基于已有结果给用户简洁总结。",
        })
        final = await agent.llm.ainvoke(messages)
        state["final_response"] = final.content.strip() or "处理超时，请重新提交。"
        state["agent_results"]["dynamic_action"] = {
            "success": True,
            "payload": {"direct_response": state["final_response"]},
            "error": None,
        }
        state["resolved"] = True
        state["dynamic_agent_messages"] = []
        state["dynamic_iteration"] = 0

        # v12: 记录执行追踪（超时）
        TraceStore.record(
            thread_id=state.get("thread_id", ""),
            user_name=state.get("user_name", ""),
            user_input=user_text,
            track="dynamic",
            iterations=iteration,
            steps=trace_steps,
            final_response=state["final_response"],
            success=False,
            error="达到最大迭代次数",
        )

        return state

    # v12: 执行步骤收集器（存储于 state，跨 self-loop 持久化）
    if is_first:
        state["_trace_steps"] = []
    trace_steps = state.setdefault("_trace_steps", [])

    logger.info(f"[DynamicAction:v9] 迭代 {iteration + 1}/{agent.MAX_REACT_ITERATIONS}")

    # ================================================================
    # GATE C: 一次 LLM 调用
    # ================================================================
    thought_event = json.dumps({
        "event": "thought",
        "text": f"正在分析第 {iteration + 1} 步...",
    }, ensure_ascii=False)
    writer(f"[REACT]{thought_event}")

    try:
        _t_llm = _time.time()
        response = await agent.llm_with_tools.ainvoke(messages)
        if _LATENCY_DEBUG:
            logger.info(f"[⏱️ LATENCY] DynamicAction LLM调用 (iter={iteration + 1}): {_time.time() - _t_llm:.2f}s, tool_calls={len(response.tool_calls) if response.tool_calls else 0}")
    except Exception as e:
        logger.error(f"[DynamicAction:v9] LLM 调用失败: {e}")
        state["final_response"] = "抱歉，处理请求时出现问题。请稍后重试。"
        state["resolved"] = True
        state["dynamic_agent_messages"] = []
        return state

    # ── 无 tool_calls → 最终回答 ──
    if not response.tool_calls:
        final_answer = response.content.strip() if response.content else "处理完成。"
        logger.info(f"[DynamicAction:v9] FINAL — iter {iteration + 1}, 无 tool_calls")

        final_event = json.dumps({
            "event": "final",
            "text": final_answer,
            "iterations": iteration + 1,
        }, ensure_ascii=False)
        writer(f"[REACT]{final_event}")

        state["final_response"] = final_answer
        state["agent_results"]["dynamic_action"] = {
            "success": True,
            "payload": {"direct_response": final_answer},
            "error": None,
        }
        # v11: 如果 final answer 是反问（含问号），标记为持续对话
        if "?" in final_answer or "？" in final_answer:
            topic = _generate_rag_topic(user_text, final_answer)
            _sh_set(state, phase="self_help_provided", topic=topic,
                    summary=final_answer[:150], track="dynamic")
            logger.info(
                f"[DynamicAction:v11] 反问结尾 → phase=self_help_provided, "
                f"topic={topic}, last_track=dynamic"
            )
        state["resolved"] = True
        state["dynamic_agent_messages"] = []
        state["dynamic_iteration"] = 0

        # v12: 记录执行追踪
        TraceStore.record(
            thread_id=state.get("thread_id", ""),
            user_name=state.get("user_name", ""),
            user_input=user_text,
            track="dynamic",
            iterations=iteration + 1,
            steps=trace_steps,
            final_response=final_answer,
            success=True,
        )

        return state

    # ================================================================
    # GATE B: 执行工具 — create_ticket 提议时设置 pending 返回
    # ================================================================
    thought = response.content[:200] if response.content else ""

    # v16: 预处理所有 tool_calls（解析 args）
    _parsed_calls = []
    for i, tc in enumerate(response.tool_calls):
        _name = tc.get("name", "")
        _args = tc.get("args", {})
        if isinstance(_args, str):
            try:
                _args = json.loads(_args)
            except (json.JSONDecodeError, TypeError):
                _args = {"_raw": _args}
        if not isinstance(_args, dict):
            _args = {"_raw": str(_args)}
        _parsed_calls.append({
            "i": i, "name": _name, "args": _args,
            "call_id": f"call_{iteration}_{_name}_{i}",
        })

    # v16: 无 create_ticket 且多个 tool_calls → 并行执行
    _has_create_ticket = any(c["name"] == "create_ticket" for c in _parsed_calls)
    if not _has_create_ticket and len(_parsed_calls) > 1:
        if _LATENCY_DEBUG:
            logger.info(f"[LATENCY] Parallel tool execution: {len(_parsed_calls)} tools")

        # Push all tool_call events first
        for c in _parsed_calls:
            writer(f"[REACT]{json.dumps({'event': 'tool_call', 'text': agent._describe_tool_call(c['name'], c['args']), 'tool': c['name'], 'args': c['args']}, ensure_ascii=False)}")

        # Execute in parallel
        async def _run_one(c):
            obs = await agent._execute_tool(c["name"], c["args"])
            obs_summary = agent._summarize_observation(c["name"], obs)
            return {**c, "observation": obs, "obs_summary": obs_summary}

        _results = await asyncio.gather(*[_run_one(c) for c in _parsed_calls])

        # Process results in original order
        for c in _results:
            writer(f"[REACT]{json.dumps({'event': 'tool_result', 'text': c['obs_summary'], 'tool': c['name']}, ensure_ascii=False)}")
            obs_data = {}
            try:
                obs_data = json.loads(c["observation"]) if isinstance(c["observation"], str) else c["observation"]
            except (json.JSONDecodeError, TypeError):
                pass
            trace_steps.append({
                "step": len(trace_steps) + 1,
                "tool": c["name"],
                "args_summary": agent._describe_tool_call(c["name"], c["args"])[:100],
                "result_summary": c["obs_summary"][:120],
                "success": not (isinstance(obs_data, dict) and obs_data.get("error")),
            })
            messages.append({
                "role": "assistant", "content": thought,
                "tool_calls": [{
                    "id": c["call_id"], "type": "function",
                    "function": {"name": c["name"], "arguments": json.dumps(c["args"], ensure_ascii=False)},
                }],
            })
            messages.append({
                "role": "tool", "tool_call_id": c["call_id"],
                "content": c["observation"] if isinstance(c["observation"], str) else json.dumps(c["observation"], ensure_ascii=False),
            })

        # Continue self-loop (no create_ticket, no interrupt)
        state["dynamic_agent_messages"] = messages
        state["dynamic_iteration"] = iteration + 1
        state["resolved"] = False
        logger.info(f"[DynamicAction:v16] Parallel done — {len(_results)} tools, iter={iteration + 1}")
        return state

    # v16: 串行路径（单个 tool_call 或含 create_ticket）
    for c in _parsed_calls:
        tool_name = c["name"]
        tool_args = c["args"]
        tool_call_id = c["call_id"]
        i = c["i"]

        # 推送: 工具调用
        tool_call_event = json.dumps({
            "event": "tool_call",
            "text": agent._describe_tool_call(tool_name, tool_args),
            "tool": tool_name,
            "args": tool_args,
        }, ensure_ascii=False)
        writer(f"[REACT]{tool_call_event}")

        # 执行工具
        _t_tool = _time.time()
        observation = await agent._execute_tool(tool_name, tool_args)
        if _LATENCY_DEBUG:
            logger.info(f"[⏱️ LATENCY] 工具 {tool_name}: {_time.time() - _t_tool:.2f}s")

        # 推送: 工具结果
        obs_summary = agent._summarize_observation(tool_name, observation)
        tool_result_event = json.dumps({
            "event": "tool_result",
            "text": obs_summary,
            "tool": tool_name,
        }, ensure_ascii=False)
        writer(f"[REACT]{tool_result_event}")

        # v12: 收集步骤追踪
        obs_data = {}
        try:
            obs_data = json.loads(observation) if isinstance(observation, str) else observation
        except (json.JSONDecodeError, TypeError):
            pass
        trace_steps.append({
            "step": len(trace_steps) + 1,
            "tool": tool_name,
            "args_summary": agent._describe_tool_call(tool_name, tool_args)[:100],
            "result_summary": obs_summary[:120],
            "success": not (isinstance(obs_data, dict) and obs_data.get("error")),
        })

        # 追加 assistant 消息（含 tool_call）
        messages.append({
            "role": "assistant",
            "content": thought,
            "tool_calls": [{
                "id": tool_call_id,
                "type": "function",
                "function": {
                    "name": tool_name,
                    "arguments": json.dumps(tool_args, ensure_ascii=False),
                },
            }],
        })

        # ★★★ create_ticket 提议 → 设置 pending, 由独立节点调用 interrupt() ★★★
        if tool_name == "create_ticket":
            try:
                data = json.loads(observation) if isinstance(observation, str) else observation
            except (json.JSONDecodeError, TypeError):
                data = {}

            if data.get("status") == "proposed" and not agent._execution_mode:
                # 不追加 tool result — 等用户确认后再追加
                proposals = getattr(agent, '_proposals', {})
                card = {}
                if proposals:
                    last_key = list(proposals.keys())[-1]
                    card = proposals[last_key].get("card", {})

                state["dynamic_agent_messages"] = messages
                state["dynamic_interrupt_card"] = card
                state["dynamic_pending_tool"] = {
                    "name": tool_name,
                    "args": tool_args,
                    "tool_call_id": tool_call_id,
                }
                state["pending_card_type"] = "dynamic_interrupt"
                state["resolved"] = False
                state["dynamic_iteration"] = iteration  # 还没完成这一轮

                # 推送卡片到前端
                if card:
                    writer(f"[CARD]{json.dumps(card, ensure_ascii=False)}")

                logger.info(
                    f"[DynamicAction:v10] 🛑 PROPOSED — "
                    f"card={card.get('title', '?')[:40]}, "
                    f"iter={iteration}, tool_call_id={tool_call_id}"
                )
                # ★ 不在这里 interrupt() — 由 dynamic_interrupt_node 统一处理

                # v12: 记录执行追踪（中断等待确认）
                TraceStore.record(
                    thread_id=state.get("thread_id", ""),
                    user_name=state.get("user_name", ""),
                    user_input=user_text,
                    track="dynamic",
                    iterations=iteration + 1,
                    steps=trace_steps,
                    final_response=f"等待确认: {card.get('title', '?')}",
                    success=True,
                )

                return state

        # 非 create_ticket / 非 proposed → 正常追加 tool result
        messages.append({
            "role": "tool",
            "tool_call_id": tool_call_id,
            "content": observation if isinstance(observation, str)
            else json.dumps(observation, ensure_ascii=False),
        })

    # ── 无中断 → 保存消息, 自循环 ──
    state["dynamic_agent_messages"] = messages
    state["dynamic_iteration"] = iteration + 1
    state["resolved"] = False

    logger.info(
        f"[DynamicAction:v10] 继续 — iter {iteration + 1}, "
        f"{len(messages)} msgs saved"
    )
    return state


async def dynamic_interrupt_node(state: TicketState) -> TicketState:
    """
    中断节点 (v10) — 单一 interrupt() 调用点，处理用户确认/取消/修改。

    dynamic_action_node 设置 pending_card_type="dynamic_interrupt" 后进入此节点。
    首次执行: interrupt() 冻结图，等待用户操作。
    Replay:   interrupt() 返回决策 → 执行确认/取消/修改 → 返回 state。
    """
    from agents.sub_agents.dynamic_action_agent import DynamicActionAgent

    card = state.get("dynamic_interrupt_card", {})
    pending_tool = state.get("dynamic_pending_tool", {})
    messages = state.get("dynamic_agent_messages", [])
    user_text = _get_user_text(state)

    logger.info(
        f"[DynamicInterrupt:v10] 等待用户确认: "
        f"card={card.get('title', '?')[:40]}"
    )

    # ★ 唯一的 interrupt() 调用点 — replay 时正确返回决策
    decision = interrupt({
        "type": "dynamic_interrupt",
        "card": card,
        "message": f"请确认: {card.get('title', '')}",
    })

    action = decision.get("action", "confirm")
    logger.info(
        f"[DynamicInterrupt:v10] RESUME decision={action}, "
        f"iter={state.get('dynamic_iteration', 0)}"
    )

    # ── 执行用户决定 ──
    agent = DynamicActionAgent()
    agent._proposals = {}
    await agent._ensure_inventory_seeded()

    if action == "confirm":
        agent._execution_mode = True
        agent._last_user_input = user_text
        agent._last_user_name = state.get("user_name", "")
        agent._last_user_role = state.get("role", "employee")
        agent._last_trace_id = state.get("thread_id", "")

        result_json = await agent._tool_create_ticket(pending_tool.get("args", {}))
        result = json.loads(result_json) if isinstance(result_json, str) else {}
        agent._execution_mode = False

        if result.get("executed"):
            ticket_no = result.get("ticket_number", "?")
            observation = json.dumps(result, ensure_ascii=False)
            logger.info(f"[DynamicInterrupt:v10] TicketDispatch 落库成功: {ticket_no}")

            # 追加 tool result 到对话历史
            messages.append({
                "role": "tool",
                "tool_call_id": pending_tool.get(
                    "tool_call_id",
                    f"call_resume_{state.get('dynamic_iteration', 0)}",
                ),
                "content": observation,
            })

            # ── v13: 确认成功 → 让 LLM 生成总结后结束 ──
            summary_prompt = (
                "操作已成功执行。请根据以上对话历史，生成一段简洁的总结回复给用户。\n\n"
                "格式要求:\n"
                f"- 开头告知工单号: {ticket_no}\n"
                "- 列出已确认的物品清单（从 tool result 中提取）\n"
                "- 如有库存不足/缺货的物品，主动提示并建议替代方案\n"
                "- 语气: 专业、简练\n"
                "- 不超过300字"
            )
            messages.append({"role": "user", "content": summary_prompt})
            try:
                final_msg = await agent.llm.ainvoke(messages)
                state["final_response"] = final_msg.content.strip() or (
                    f"工单 {ticket_no} 已创建，感谢您的确认。"
                )
            except Exception:
                state["final_response"] = (
                    f"工单 {ticket_no} 已创建，感谢您的确认。"
                )

            state["dynamic_agent_messages"] = []
            state["pending_card_type"] = ""
            state["dynamic_interrupt_card"] = {}
            state["dynamic_pending_tool"] = {}
            state["dynamic_iteration"] = 0
            state["resolved"] = True
            logger.info(
                f"[DynamicInterrupt:v13] 确认成功，工单 {ticket_no}，"
                f"生成最终总结后结束"
            )
            return state
        else:
            observation = json.dumps(
                {"executed": False, "error": result.get("error", "执行失败")},
                ensure_ascii=False,
            )
    elif action == "cancel":
        observation = json.dumps(
            {"executed": False, "message": "用户取消"},
            ensure_ascii=False,
        )
        # cancel → 直接结束，不再继续
        state["final_response"] = "好的，已取消。还有其他需要帮您的吗？"
        state["dynamic_agent_messages"] = []
        state["pending_card_type"] = ""
        state["dynamic_interrupt_card"] = {}
        state["dynamic_pending_tool"] = {}
        state["dynamic_iteration"] = 0
        state["resolved"] = True
        logger.info("[DynamicInterrupt:v13] 用户取消，直接结束")
        return state
    else:  # modify
        feedback = decision.get("feedback", "")
        observation = json.dumps(
            {"executed": False, "message": f"用户要求修改: {feedback}"},
            ensure_ascii=False,
        )

    # ★ modify: 把 tool result 追加到对话历史 — LLM 在下一轮看到修改要求
    messages.append({
        "role": "tool",
        "tool_call_id": pending_tool.get(
            "tool_call_id",
            f"call_resume_{state.get('dynamic_iteration', 0)}",
        ),
        "content": observation,
    })

    state["dynamic_agent_messages"] = messages
    state["pending_card_type"] = ""
    state["dynamic_interrupt_card"] = {}
    state["dynamic_pending_tool"] = {}
    state["dynamic_iteration"] = state.get("dynamic_iteration", 0) + 1
    state["resolved"] = False

    logger.info(
        f"[DynamicInterrupt:v13] modify — "
        f"{len(messages)} msgs, iter={state['dynamic_iteration']}, "
        f"返回 dynamic_action 继续推理修改"
    )
    return state


def after_dynamic_action(state: TicketState) -> str:
    """
    dynamic_action 之后的路由:
      - resolved=True → respond
      - pending_card="dynamic_interrupt" → dynamic_interrupt
      - 其他 pending → respond
      - 其他 → dynamic_action (自循环)
    """
    if state.get("resolved"):
        return "respond"
    pending = state.get("pending_card_type", "")
    if pending == "dynamic_interrupt":
        return "dynamic_interrupt"  # v10: 独立中断节点
    if pending:
        return "respond"
    return "dynamic_action"


def after_dynamic_interrupt(state: TicketState) -> str:
    """
    dynamic_interrupt 之后的路由:
      - resolved=True → respond
      - 其他 → dynamic_action (继续 ReAct)
    """
    if state.get("resolved"):
        return "respond"
    return "dynamic_action"

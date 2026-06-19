"""
复杂通道节点 — 请假/报销固定 DAG (RAG ∥ ToolAgent → TicketDispatch)

从 agents.graph_workflow 拆分出来。
依赖: state.py
"""

from __future__ import annotations

import asyncio
import json
import logging
import re as _re

from langgraph.config import get_stream_writer

from agents.graph.state import TicketState, _get_user_text, _build_conversation_context

logger = logging.getLogger("graph.nodes.complex")


async def complex_track_node(state: TicketState) -> TicketState:
    """
    复杂通道：请假/报销固定 DAG (RAG ∥ 余额查询 → TicketDispatch)，非 DAG 场景串行兜底。

    v6: 对请假/报销场景实现并行编排（RAG ∥ leave_balance_query → TicketDispatch），
    其他场景走串行 TaskPlanner 路径。
    并行结果通过 Python 变量传递（不写 LangGraph State），避免 Race Condition。
    """
    from agents.orchestrator.task_planner import TaskPlanner
    from agents.orchestrator.agent_registry import agent_registry
    from agents.orchestrator.response_synthesizer import ResponseSynthesizer
    from agents.a2a.protocol import AgentMessage as AM
    from agents.orchestrator.router import RouteResult
    from config.model_provider import create_chat_model

    user_text = _get_user_text(state)
    conversation_history = _build_conversation_context(
        state["messages"], summary=state.get("conversation_summary", ""),
    )
    user_name = state.get("user_name", "")
    if not user_name:
        logger.warning("[ComplexTrack] user_name 为空，降级为 fast")
        state["final_response"] = "无法获取您的用户身份，请刷新页面后重试。"
        state["resolved"] = True
        return state
    llm = create_chat_model(model_type="main", temperature=0)

    # ================================================================
    # v6: 请假/报销 → 并行 DAG 模式
    # ================================================================
    _leave_pattern = _re.compile(r'请.*?假|休.*?假|年假|病假|事假|调休|婚假|产假|天假')
    is_leave = bool(_leave_pattern.search(user_text))
    expense_keywords = ["报销", "差旅", "费用"]
    is_expense = any(kw in user_text for kw in expense_keywords)

    if is_leave or is_expense:
        logger.info(
            f"[ComplexTrack:v6] 检测到 {'请假' if is_leave else '报销'}请求，"
            f"启用并行 DAG 模式"
        )
        writer = get_stream_writer()

        try:
            # ── Step 1+2: 并行 RAG查政策 + 余额查询 ──
            rag_agent = agent_registry.get_agent("enterprise_rag")
            ticket_agent = agent_registry.get_agent("ticket_dispatch")

            if not rag_agent or not ticket_agent:
                logger.error("[ComplexTrack:v6] Agent 未就绪，降级为串行")
                state["final_response"] = "系统初始化未完成，请稍后重试。"
                state["resolved"] = True
                return state

            await rag_agent._ensure_initialized()

            # ★ 推送: 阶段1 — 并行查询
            stage1_label = (
                f"📚 正在查询{'请假' if is_leave else '报销'}政策与个人余额..."
            )
            logger.info(f"[ComplexTrack:v9] {stage1_label}")
            writer(f"[THINKING] {stage1_label}")

            async def _rag_policy():
                """并行任务 A: 查询政策"""
                docs = await rag_agent.knowledge_service.search(user_text, top_k=3)
                if docs:
                    policy_text = await rag_agent._synthesize(
                        user_text, docs, conversation_history,
                    )
                    return {
                        "success": True,
                        "policy_text": policy_text,
                        "sources": [
                            {"category": d.get("category", ""), "score": d.get("score", 0)}
                            for d in docs
                        ],
                    }
                return {"success": True, "policy_text": "", "sources": []}

            async def _tool_balance():
                """并行任务 B: 查询员工余额 — 直接调用 leave_balance_query 工具"""
                from agents.tools import tool_registry
                result = await tool_registry.invoke(
                    "leave_balance_query", employee_name=user_name,
                )
                if result.success and result.data:
                    return {
                        "success": True,
                        "tool_result": result.data,
                        "direct_response": json.dumps(result.data, ensure_ascii=False),
                    }
                return {"success": False, "error": result.error}

            # 并行执行（Python 变量传递，不写 LangGraph State）
            policy_result, balance_result = await asyncio.gather(
                _rag_policy(), _tool_balance(),
            )

            logger.info(
                f"[ComplexTrack:v6] 并行完成 — "
                f"policy={policy_result['success']}, balance={balance_result['success']}"
            )

            # 存入 state（并行完成后安全写入）
            state["parallel_rag_result"] = policy_result
            state["parallel_tool_result"] = balance_result

            # ★ 推送: 阶段2 — 合规检查
            writer(f"[THINKING] 🔍 正在进行合规检查并生成确认卡片...")

            # ── Step 3: TicketDispatch 合规检查 + 生成卡片 ──
            policy_text = policy_result.get("policy_text", "")
            balance_data = balance_result.get("tool_result", {})
            balance_text = balance_result.get("direct_response", "")

            compliance_input = (
                f"{user_text}\n\n"
                f"[系统预查询结果]\n"
                f"政策信息：{policy_text[:500] if policy_text else '未查到相关政策'}\n"
                f"员工余额：{balance_text if balance_text else '未查到余额数据'}\n"
                f"余额数据(JSON)：{json.dumps(balance_data, ensure_ascii=False) if balance_data else '无'}"
            )

            ticket_msg = AM.create_delegation(
                from_agent="orchestrator", to_agent="ticket_dispatch",
                payload={
                    "user_input": compliance_input,
                    "task": "合规检查并生成确认卡片",
                    "intent_category": "complex",
                    "urgency": state.get("urgency", "medium"),
                    "user_id": user_name,
                    "user_name": user_name,
                    "role": state.get("role", "employee"),
                    "pre_checked": True,
                    "policy_result": policy_result,
                    "balance_result": balance_result,
                },
                trace_id=state.get("thread_id", ""),
            )

            result = await ticket_agent.execute(ticket_msg)

            state["agent_results"] = {
                "enterprise_rag": {"success": policy_result["success"],
                                   "payload": policy_result, "error": None},
                "leave_balance_query": {"success": balance_result["success"],
                                       "payload": balance_result, "error": None},
                "ticket_dispatch": {"success": result.success,
                                    "payload": result.payload, "error": result.error},
            }

            if result.success and result.payload.get("return_card"):
                card = result.payload.get("card", {})
                ticket_type = result.payload.get("ticket_type", "leave" if is_leave else "expense")
                state["final_response"] = (
                    "📋 **请确认以下信息**\n\n"
                    + card.get("description", "")
                    + "\n[CARD]" + json.dumps(card, ensure_ascii=False)
                )
                state["agent_results"]["ticket_dispatch"]["card"] = card
                state["pending_card_type"] = ticket_type
                logger.info(f"[ComplexTrack:v6] 设置卡片锁 pending_card_type={ticket_type}")
            elif result.success and result.payload.get("direct_response"):
                state["final_response"] = result.payload["direct_response"]
            elif result.error:
                state["final_response"] = (
                    f"合规检查未能完成：{result.error}\n\n"
                    "请稍后重试，或拨打服务台热线获取人工支持。"
                )
            else:
                state["final_response"] = "申请已提交，如需查看详情请稍后查询。"

        except Exception as e:
            logger.error(f"[ComplexTrack:v6] 并行 DAG 执行失败: {e}，降级为串行")
            # 降级：直接走 TicketDispatch
            ticket_agent = agent_registry.get_agent("ticket_dispatch")
            if ticket_agent:
                try:
                    fallback_msg = AM.create_delegation(
                        from_agent="orchestrator", to_agent="ticket_dispatch",
                        payload={
                            "user_input": user_text,
                            "task": "提取参数并创建工单",
                            "intent_category": "complex",
                            "urgency": state.get("urgency", "medium"),
                            "user_id": user_name,
                            "user_name": user_name,
                            "role": state.get("role", "employee"),
                        },
                        trace_id=state.get("thread_id", ""),
                    )
                    result = await ticket_agent.execute(fallback_msg)
                    state["agent_results"]["ticket_dispatch"] = {
                        "success": result.success, "payload": result.payload, "error": result.error,
                    }
                    state["final_response"] = (
                        result.payload.get("direct_response", "申请已提交。")
                        if result.success
                        else f"操作失败：{result.error}"
                    )
                except Exception as e2:
                    state["final_response"] = f"申请处理失败：{e2}"

        state["resolved"] = True
        return state

    # ================================================================
    # 非请假/报销场景：串行 TaskPlanner 路径（防御性兜底）
    # v8: Router 应只将请假/报销路由到 complex，保留此路径防御未知输入
    # ================================================================

    logger.warning(
        f"[ComplexTrack:v8:FALLBACK] 非请假/报销请求走串行路径 — "
        f"这可能表示 Router 将非 DAG 请求误判为 complex。"
        f" input={user_text[:80]}"
    )

    intent_result = RouteResult(track="complex", reason=user_text[:100])
    planner = TaskPlanner(llm, agent_registry)

    plan_input = user_text
    if conversation_history:
        plan_input = f"对话历史:\n{conversation_history}\n\n当前输入: {user_text}"

    plan = await planner.plan(intent_result, plan_input)

    state["needs_human_review"] = state.get("needs_human_review", False) or plan.needs_human_review

    agent_results = {}
    for step in plan.steps:
        agent_id = step.agent_id
        if not agent_id:
            continue
        agent_instance = agent_registry.get_agent(agent_id)
        if agent_instance is None:
            logger.warning(f"[ComplexTrack] Agent '{agent_id}' 未注册，跳过")
            continue

        delegation = AM.create_delegation(
            from_agent="orchestrator", to_agent=agent_id,
            payload={
                "user_input": user_text, "task": step.task,
                "params": step.params, "intent_category": "complex",
                "conversation_history": conversation_history,
            },
            trace_id=state.get("thread_id", ""),
        )
        try:
            result = await agent_instance.execute(delegation)
            agent_results[agent_id] = {
                "success": result.success, "payload": result.payload, "error": result.error,
            }
        except Exception as e:
            logger.error(f"[ComplexTrack] Agent '{agent_id}' 失败: {e}")
            agent_results[agent_id] = {"success": False, "payload": {}, "error": str(e)}

    state["agent_results"] = agent_results

    # 合成
    agent_msgs = {}
    for aid, rdict in agent_results.items():
        agent_msgs[aid] = AM(
            from_agent=aid, to_agent="orchestrator",
            payload=rdict.get("payload", {}),
            success=rdict.get("success", False),
            error=rdict.get("error"),
        )

    # v6.1: 检测 TicketDispatch 是否返回了确认卡片
    td_result = agent_results.get("ticket_dispatch", {})
    if td_result.get("success") and td_result.get("payload", {}).get("return_card"):
        card = td_result["payload"].get("card", {})
        ticket_type = td_result["payload"].get("ticket_type", "")
        state["final_response"] = (
            "📋 **请确认以下信息**\n\n"
            + card.get("description", "")
            + "\n[CARD]" + json.dumps(card, ensure_ascii=False)
        )
        state["agent_results"]["ticket_dispatch"]["card"] = card
        state["pending_card_type"] = ticket_type
        logger.info(f"[ComplexTrack:serial] 设置卡片锁 pending_card_type={ticket_type}")
    else:
        synthesizer = ResponseSynthesizer(llm)
        response = await synthesizer.synthesize(agent_msgs, user_text)
        state["final_response"] = response
    state["resolved"] = True
    return state

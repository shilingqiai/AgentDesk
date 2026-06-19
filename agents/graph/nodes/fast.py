"""
极速通道节点 — EnterpriseRAG 知识检索 + 真流式 token 输出

从 agents.graph_workflow 拆分出来。
依赖: state.py
"""

from __future__ import annotations

import logging
from langgraph.config import get_stream_writer
from agents.graph.state import (
    TicketState, _get_user_text, _build_conversation_context, _generate_rag_topic,
    _sh_get, _sh_set,
)

logger = logging.getLogger("graph.nodes.fast")


async def fast_track_node(state: TicketState) -> TicketState:
    """
    极速通道 (80%)：EnterpriseRAG — 真流式 token 输出 + 状态持久化

    v4 流式：通过 LangGraph StreamWriter 发射每个 token，
    run_stream 端以 stream_mode=["values","custom"] 接收并转为 [STREAM] 标签。
    """
    from agents.orchestrator.agent_registry import agent_registry

    user_text = _get_user_text(state)
    conversation_history = _build_conversation_context(
        state["messages"], summary=state.get("conversation_summary", ""),
    )
    phase = _sh_get(state, "phase", "initial")

    agent_instance = agent_registry.get_agent("enterprise_rag")
    if agent_instance is None:
        logger.error("[FastTrack] EnterpriseRAGAgent 未注册！")
        state["final_response"] = "系统初始化未完成，请稍后重试。"
        state["resolved"] = True
        return state

    try:
        await agent_instance._ensure_initialized()

        # 检索
        docs = await agent_instance.knowledge_service.search(user_text, top_k=5)

        writer = get_stream_writer()
        full_response = ""

        if not docs:
            full_response = (
                "抱歉，我在知识库中没有找到相关的信息。\n\n"
                "建议提交工单让工程师协助处理。"
            )
            writer(full_response)
        else:
            # v4 follow_up 上下文注入
            if phase == "self_help_provided":
                topic = _sh_get(state, "topic", "")
                conversation_history = (
                    f"用户对上一轮「{topic}」方案提出了追问。\n{conversation_history}"
                )

            # 真流式 token 发射
            async for token in agent_instance._synthesize_stream(
                user_text, docs, conversation_history,
            ):
                full_response += token
                writer(token)

        # 流式完成后设置状态
        state["final_response"] = full_response.strip()
        source_list = [{"category": d.get("category", ""), "score": d.get("score", 0)}
                       for d in docs] if docs else []
        state["agent_results"]["enterprise_rag"] = {
            "success": True,
            "payload": {"direct_response": full_response.strip(), "sources": source_list},
            "error": None,
        }
        topic = _generate_rag_topic(user_text, full_response)
        _sh_set(state, phase="self_help_provided", topic=topic,
                summary=full_response[:150], track="fast")
        state["resolved"] = True
        logger.info(
            f"[FastTrack] phase → self_help_provided, "
            f"topic={topic}, tokens={len(full_response)}"
        )

    except Exception as e:
        logger.error(f"[FastTrack] 执行失败: {e}")
        state["final_response"] = "抱歉，处理您的请求时出现了问题。请稍后重试。"
        state["resolved"] = True

    return state

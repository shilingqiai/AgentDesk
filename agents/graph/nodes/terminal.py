"""
终端节点 — clarification（反问）+ respond（最终响应）

从 agents.graph_workflow 拆分出来。
依赖: state.py
"""

from __future__ import annotations

import logging
from langchain_core.messages import AIMessage
from agents.graph.state import TicketState, _get_user_text

logger = logging.getLogger("graph.nodes.terminal")


async def clarification_node(state: TicketState) -> TicketState:
    """
    反问节点：AI 不确定用户意图时，主动反问澄清

    触发条件：
    - Router 返回 track="clarify"
    - Router 返回 confidence < 0.7（route_node 强制转为 clarify）
    - LLM JSON 解析失败

    v4: self_help_provided 阶段的反问带上下文，不再问通用的"查还是办"。
    """
    user_text = _get_user_text(state)
    confidence = state.get("confidence", 0)
    topic = state.get("last_rag_topic", "")

    # v4: 有上下文时反问更精准
    if topic and state.get("conversation_phase") == "self_help_provided":
        state["final_response"] = (
            f"关于「{topic}」的方案似乎没有解决您的问题。您是想要：\n\n"
            f"1. 我再提供其他思路？\n"
            f"2. 直接提交工单让工程师处理？\n\n"
            f"请告诉我您的想法。"
        )
    elif confidence < 0.3:
        # 完全无法理解 → 通用引导
        state["final_response"] = (
            "抱歉，我不太确定您想做什么。\n\n"
            "您可以这样对我说：\n"
            "- 🔍 **查询知识**：'VPN怎么排查？''请假流程是什么？''食堂在哪？'\n"
            "- 📋 **提交工单**：'帮我提交一个网络故障工单'\n"
            "- 🏢 **行政服务**：'会议室怎么预定？''快递怎么寄？'\n\n"
            "请描述您的具体需求，我会尽力帮您解决。"
        )
    else:
        # 有一定理解但不确信 → 引导式反问
        state["final_response"] = (
            f"我不太确定您的具体需求，想跟您确认一下：\n\n"
            f"您是想要：\n"
            f"1. **查询相关信息**（如政策、流程、故障排查方法）？\n"
            f"2. **提交一个工单**（让工程师或HR处理）？\n\n"
            f"请告诉我具体内容，我会帮您处理。"
        )

    state["resolved"] = True
    logger.info(f"[Clarify] 反问用户 (confidence={confidence:.0%}, input=\"{user_text[:50]}\")")
    return state


async def respond_node(state: TicketState) -> TicketState:
    """最终响应节点：记录消息 + 人工审核拦截"""
    if not state.get("final_response"):
        state["final_response"] = "处理完成，如有疑问请咨询服务台。"

    if state.get("needs_human_review") and state.get("track") in ("dynamic", "complex"):
        state["final_response"] = (
            "⚠️ 此操作需要人工审核确认。\n\n"
            f"{state['final_response']}\n\n"
            "---\n💡 工单已创建但需要管理员审核后才会派发。"
        )

    state["messages"].append(AIMessage(content=state["final_response"]))
    state["resolved"] = True
    return state

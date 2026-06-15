"""
工单编排状态 — TicketState TypedDict + 辅助函数

从 agents.graph_workflow 拆分出来，作为最底层模块（无内部依赖）。
"""

from __future__ import annotations

import logging
from typing import TypedDict, Optional, Annotated
from langgraph.graph.message import add_messages
from langchain_core.messages import HumanMessage, AIMessage

logger = logging.getLogger("graph.state")


# ============================================================
# TicketState
# ============================================================

class TicketState(TypedDict):
    """工单编排状态"""
    messages: Annotated[list, add_messages]
    track: str
    agent_id: str
    intent: str
    urgency: str
    confidence: float
    plan: list[dict]
    current_step: int
    agent_results: dict
    needs_human_review: bool
    human_decision: Optional[str]
    final_response: str
    resolved: bool
    thread_id: str
    pending_card_type: str    # "" = 无锁, "admin"/"leave"/"expense"/"it_fault" = 卡片锁定中
    re_route: bool            # True = action_track 处理完回 Router 重路由
    # v4: 对话阶段追踪
    conversation_phase: str   # "initial" | "self_help_provided"
    last_rag_topic: str       # 上一轮 RAG 回答的主题（如 "VPN故障排查"）
    last_rag_summary: str     # 上一轮 RAG 回答的核心内容（100字摘要）
    # v5: 用户身份
    user_name: str            # 当前用户姓名（如 "张三"）
    role: str                 # 角色: "employee" | "admin"
    # v6: 并行节点结果隔离（防 Race Condition）
    parallel_rag_result: dict     # RAG 查询结果（complex track 并行）
    parallel_tool_result: dict    # ToolAgent 查询结果（complex track 并行）
    # v7: 动态Agent跨turn状态
    dynamic_agent_messages: list  # 序列化的消息历史（跨turn恢复）
    dynamic_agent_proposals: dict # 上一轮的提议（确认后执行）
    # v9: 真正的中断控制（每次迭代一个工具调用，create_ticket 后立即 interrupt）
    dynamic_iteration: int        # 当前 ReAct 迭代次数
    dynamic_interrupt_card: dict  # 当前触发中断的单张卡片
    dynamic_pending_tool: dict    # 等待确认的工具信息 {name, args, tool_call_id}
    # v11: 上一轮轨道类型 — 供 re_evaluate 正确分发 follow_up
    last_track_type: str          # "fast" | "dynamic" | ""
    # v12: 对话历史递进摘要 — 超过阈值轮数后触发 LLM 压缩
    conversation_summary: str     # 早期对话的递进摘要（语义要点）


def create_initial_state(user_input: str, thread_id: str = "default",
                        user_name: str = "", role: str = "employee") -> TicketState:
    return TicketState(
        messages=[HumanMessage(content=user_input)],
        track="", agent_id="", intent="", urgency="medium",
        confidence=0.0, plan=[], current_step=0, agent_results={},
        needs_human_review=False, human_decision=None,
        final_response="", resolved=False, thread_id=thread_id,
        pending_card_type="", re_route=False,
        conversation_phase="initial", last_rag_topic="", last_rag_summary="",
        user_name=user_name, role=role,
        parallel_rag_result={}, parallel_tool_result={},
        dynamic_agent_messages=[], dynamic_agent_proposals={},
        dynamic_iteration=0, dynamic_interrupt_card={}, dynamic_pending_tool={},
        last_track_type="", conversation_summary="",
    )


# ============================================================
# 辅助函数
# ============================================================

def _build_conversation_context(messages: list, max_turns: int = 5,
                                summary: str = "") -> str:
    """
    构建多轮对话上下文 — 最近 N 轮原文 + 早期递进摘要。

    v12: 当 conversation_summary 存在时，以语义摘要替代早期消息截断。
    格式: <conversation_summary> + <recent_history>
    """
    if len(messages) <= 1:
        return summary  # 仅当前消息时只有摘要

    recent = messages[-(max_turns * 2):]
    lines = []
    # 仅取最近 N-1 轮（第 N 轮是当前用户输入，由节点单独处理）
    for msg in recent[:-1]:
        role = "用户" if isinstance(msg, HumanMessage) else "助手"
        content = msg.content[:300] if hasattr(msg, 'content') else str(msg)[:300]
        lines.append(f"{role}: {content}")

    recent_text = "\n".join(lines) if lines else ""

    if summary:
        return (
            f"<conversation_summary>\n{summary}\n</conversation_summary>\n\n"
            f"<recent_history>\n{recent_text}\n</recent_history>"
        )
    return recent_text


# ── 对话历史压缩阈值 ──
# 消息数 ≥ COMPRESS_THRESHOLD 时触发 LLM 压缩
# 压缩范围: messages[:-KEEP_RECENT]，保留最近 KEEP_RECENT 条原文
COMPRESS_THRESHOLD = 10   # 消息数（约 5 轮）
KEEP_RECENT = 6           # 保留最近 6 条（约 3 轮）


async def _maybe_compress_history(messages: list, existing_summary: str = "") -> str:
    """
    当对话消息超过阈值时，用 LLM 压缩早期消息为递进摘要。

    策略:
      - messages ≤ COMPRESS_THRESHOLD → 返回 existing_summary（不压缩）
      - messages > COMPRESS_THRESHOLD → 压缩 messages[:-KEEP_RECENT] 部分
      - 优先使用规则压缩（零延迟），回退 LLM 压缩

    递进摘要: 新摘要 = LLM(已有摘要 + 待压缩消息) → 累积式
    """
    if len(messages) <= COMPRESS_THRESHOLD:
        return existing_summary

    # 待压缩范围: 去掉最近 KEEP_RECENT 条
    to_compress = messages[:-KEEP_RECENT]

    # 仅压缩 HumanMessage 和 AIMessage 的文本
    compressible = []
    for m in to_compress:
        if isinstance(m, HumanMessage):
            compressible.append(f"用户: {m.content[:200]}")
        elif isinstance(m, AIMessage):
            compressible.append(f"助手: {m.content[:200]}")

    if not compressible:
        return existing_summary

    raw_text = "\n".join(compressible[-40:])  # 最多取最近 40 条待压缩

    # ── 规则兜底: 待压缩部分 ≤ 4 条 → 直接拼接，不调 LLM ──
    if len(compressible) <= 4:
        base = existing_summary + "\n" if existing_summary else ""
        return (base + raw_text)[:800]

    # ── LLM 压缩 ──
    try:
        from config.model_provider import create_chat_model
        llm = create_chat_model(model_type="main", temperature=0)

        prompt = (
            "你是对话摘要器。请将以下对话历史压缩为一段简洁的摘要（≤200字），"
            "提取关键事实：用户身份、问题领域、已尝试方案、已做决策、待确认事项。\n\n"
            "规则:\n"
            "- 保留具体参数值（如工单号、时间、人名、设备型号）\n"
            "- 保留未解决的问题和待确认事项\n"
            "- 语言简洁，只记录事实和决策点\n"
        )
        if existing_summary:
            prompt += (
                f"\n\n**已有摘要:**\n{existing_summary}\n\n"
                f"**新增对话:**\n{raw_text}\n\n"
                "请将上述内容合并为一份递进摘要（≤300字）。"
            )
        else:
            prompt += f"\n\n**对话内容:**\n{raw_text}\n\n请生成摘要（≤200字）。"

        response = await llm.ainvoke(prompt)
        summary = response.content.strip()[:500]
        logger.info(
            f"[Compress] 压缩 {len(compressible)} 条消息 → "
            f"{len(summary)} 字摘要"
        )
        return summary

    except Exception as e:
        logger.warning(f"[Compress] LLM 摘要失败: {e}，使用规则拼接")
        base = existing_summary + "\n" if existing_summary else ""
        return (base + raw_text)[:800]


def _get_user_text(state: TicketState) -> str:
    """获取当前用户输入"""
    return state["messages"][-1].content


def _reset_self_help_state(state: TicketState) -> None:
    """清空 self-help 追踪状态（防幽灵上下文）"""
    state["conversation_phase"] = "initial"
    state["last_rag_topic"] = ""
    state["last_rag_summary"] = ""
    state["last_track_type"] = ""


def _generate_rag_topic(user_input: str, response: str) -> str:
    """从用户输入和 RAG 回答中提取简要主题（规则兜底，不调 LLM）"""
    topic = user_input[:20].replace("\n", " ").strip()
    return topic if topic else "企业服务咨询"


def _detect_topic_from_history(messages: list) -> str:
    """
    从最近几轮对话中提取话题标签（纯规则匹配，零延迟）。

    仅用于给 LLM 提供话题提示，不参与路由决策。
    """
    from langchain_core.messages import HumanMessage as HM

    # 取最近 2 轮用户消息
    user_msgs = [m for m in messages[-8:]
                 if isinstance(m, HM) or (
                     isinstance(m, dict) and m.get("role") == "user"
                 )]
    recent_text = " ".join([
        (m.content if hasattr(m, "content") else m.get("content", ""))[:80]
        for m in user_msgs[-2:]
    ])

    topic_keywords = {
        "请假/休假": ["请假", "年假", "病假", "事假", "调休", "休假", "休", "假期"],
        "入职/设备领用": ["入职", "设备", "电脑", "笔记本", "显示器", "领用", "采购", "资产"],
        "IT/故障报修": ["VPN", "网络", "故障", "报修", "连不上", "打不开", "电脑坏", "打印机"],
        "报销": ["报销", "发票", "差旅", "费用"],
        "会议室": ["会议室", "预定", "开会", "预约"],
        "查政策/知识": ["政策", "流程", "怎么", "如何", "规定", "查询", "在哪"],
    }

    for topic, keywords in topic_keywords.items():
        if any(kw in recent_text for kw in keywords):
            return topic

    return "通用咨询"


def _serialize_messages(messages: list) -> list:
    """将 LangChain 消息列表序列化为可存储的字典列表"""
    if not messages:
        return []
    result = []
    for m in messages:
        entry = {"role": m.get("role", "") if isinstance(m, dict) else getattr(m, "role", "system")}
        content = m.get("content", "") if isinstance(m, dict) else getattr(m, "content", "")
        entry["content"] = str(content)[:2000]
        if isinstance(m, dict) and m.get("tool_calls"):
            entry["tool_calls"] = m["tool_calls"]
        elif hasattr(m, "tool_calls") and m.tool_calls:
            entry["tool_calls"] = m.tool_calls
        if isinstance(m, dict) and m.get("tool_call_id"):
            entry["tool_call_id"] = m["tool_call_id"]
        elif hasattr(m, "tool_call_id") and m.tool_call_id:
            entry["tool_call_id"] = m.tool_call_id
        result.append(entry)
    return result

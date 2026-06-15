"""
流式输出辅助 — 将图节点事件转为前端流式标签

从 agents.graph_workflow 拆分出来。
"""

from __future__ import annotations


def _yield_stream_event(node_name: str, node_state: dict):
    """将图节点事件转为流式标签 yield（生成器辅助函数）"""
    # 安全保护: 非 dict 跳过（如 interrupt 产生的 tuple）
    if not isinstance(node_state, dict):
        return

    # respond 节点只负责持久化，不产出行内输出
    if node_name == "respond":
        return

    if node_name == "re_evaluate":
        pass  # 静默执行
    elif node_name == "fast_track":
        yield "[FAST] 📚 企业知识库检索 · EnterpriseRAG\n"
    elif node_name == "action_track":
        yield "[CARD_RESPONSE] 📋 处理卡片回复\n"
    elif node_name == "dynamic_action":
        yield "[DYNAMIC] 🧠 动态编排 · ReAct 循环\n"
    elif node_name == "dynamic_interrupt":
        yield "[INTERRUPT_CARD] 📋 等待确认\n"
    elif node_name == "complex_track":
        yield "[COMPLEX] 🧩 复杂通道 · 多步骤编排\n"
    elif node_name == "clarification":
        yield "[CLARIFY] 🤔 AI 需要确认您的意图\n"
    elif node_name == "route":
        track = node_state.get("track", "")
        if track and track not in ("card_lock", "re_evaluate"):
            track_names = {
                "fast": "🔍 极速通道 · 企业知识库问答",
                "dynamic": "🧠 动态编排 · ReAct 自由工具调用",
                "complex": "🧩 复杂通道 · 请假/报销合规检查",
            }
            yield f"[ROUTE] {track_names.get(track, track)}\n"

    # 文本 / 卡片输出
    resp = node_state.get("final_response", "")
    if not resp:
        return
    if "[CARD]" in resp:
        parts = resp.split("[CARD]", 1)
        text_part = parts[0].strip()
        card_part = parts[1].strip() if len(parts) > 1 else ""
        if text_part:
            yield f"[STREAM]{text_part}\n"
        if card_part:
            yield f"[CARD]{card_part}\n"
    else:
        yield f"[STREAM]{resp}\n"

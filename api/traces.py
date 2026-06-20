"""
执行追踪 API

提供 Agent 执行追踪的查询，供 Admin 面板可视化。
"""

from __future__ import annotations

from fastapi import APIRouter, Query

from services.trace_store import TraceStore

router = APIRouter(prefix="/api/traces", tags=["执行追踪"])


@router.get("/", summary="获取最近执行追踪")
async def list_traces(limit: int = Query(default=10, ge=1, le=50, description="返回条数")):
    """
    返回最近 N 条 ReAct 执行追踪（倒序 — 最新的在前）。

    每条追踪包含:
    - thread_id, user_name, timestamp, track
    - user_input (截断至 120 字符)
    - steps: [{step, tool, args_summary, result_summary, success}]
    - tool_count, iterations, success, error
    """
    traces = TraceStore.get_recent(limit=limit)
    return {
        "success": True,
        "count": len(traces),
        "total_stored": TraceStore.count(),
        "traces": traces,
    }

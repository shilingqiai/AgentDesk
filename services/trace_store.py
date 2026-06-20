"""
Execution Trace Store — 内存环形缓冲区

存储最近 N 条 ReAct 执行追踪，供 Admin 面板可视化。
不持久化到 DB，重启后清空。
"""

from __future__ import annotations

import time
from collections import deque
from datetime import datetime, timezone, timedelta
from typing import Optional

# Asia/Shanghai 时区
TZ_SHANGHAI = timezone(timedelta(hours=8))


class TraceStore:
    """轻量内存追踪存储 — 环形缓冲区，最多保留 50 条"""

    _traces: deque[dict] = deque(maxlen=50)

    @classmethod
    def record(
        cls,
        *,
        thread_id: str = "",
        user_name: str = "",
        user_input: str = "",
        track: str = "dynamic",
        iterations: int = 0,
        steps: list[dict] = None,
        final_response: str = "",
        success: bool = True,
        error: str = "",
    ):
        """记录一条执行追踪"""
        trace = {
            "id": f"tr-{int(time.time() * 1000)}",
            "thread_id": thread_id or "",
            "user_name": user_name or "",
            "timestamp": datetime.now(TZ_SHANGHAI).strftime("%H:%M:%S"),
            "timestamp_iso": datetime.now(TZ_SHANGHAI).isoformat(),
            "track": track,
            "user_input": (user_input or "")[:120],
            "iterations": iterations,
            "steps": steps or [],
            "tool_count": len(steps) if steps else 0,
            "final_response": (final_response or "")[:150],
            "success": success,
            "error": error or "",
        }
        cls._traces.append(trace)

    @classmethod
    def get_recent(cls, limit: int = 10) -> list[dict]:
        """获取最近 N 条追踪（倒序——最新的在前）"""
        traces = list(cls._traces)
        traces.reverse()
        return traces[:limit]

    @classmethod
    def get_by_thread(cls, thread_id: str) -> Optional[dict]:
        """按 thread_id 查找最近的追踪"""
        for t in reversed(cls._traces):
            if t["thread_id"] == thread_id:
                return t
        return None

    @classmethod
    def clear(cls):
        """清空所有追踪"""
        cls._traces.clear()

    @classmethod
    def count(cls) -> int:
        return len(cls._traces)

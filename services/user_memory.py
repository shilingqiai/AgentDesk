"""
用户记忆层 — 跨会话持久化用户画像

三层记忆能力:
1. 问题模式检测: 7天内同类问题出现≥2次 → 主动建议走工单而非再次给出相同答案
2. 历史方案复用: 上次类似问题的解决方案 → 注入 System Prompt
3. 累计统计: 本季度请假次数/消费金额等 → 审批时提示

存储: SQLite（轻量零依赖）
"""

from __future__ import annotations

import json
import logging
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime, timedelta

logger = logging.getLogger("user_memory")

MEMORY_DB = "data/user_memory.db"


@dataclass
class UserMemory:
    """用户快照记忆"""
    user_id: str
    recent_topics: list[dict] = field(default_factory=list)    # 最近对话主题
    recent_tickets: list[dict] = field(default_factory=list)   # 最近工单
    preferences: dict = field(default_factory=dict)            # 偏好设置
    leave_balance_snapshot: dict | None = None                 # 假期余额缓存
    updated_at: str = ""


class UserMemoryStore:
    """基于 SQLite 的轻量用户记忆存储"""

    def __init__(self, db_path: str = MEMORY_DB):
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        import os
        os.makedirs("data", exist_ok=True)
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS user_memory (
                    user_id TEXT PRIMARY KEY,
                    memory_json TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
            """)
            conn.commit()

    def get(self, user_id: str) -> UserMemory:
        """获取用户记忆"""
        if not user_id:
            return UserMemory(user_id="anonymous")
        with sqlite3.connect(self.db_path) as conn:
            row = conn.execute(
                "SELECT memory_json FROM user_memory WHERE user_id = ?",
                (user_id,)
            ).fetchone()
            if row:
                data = json.loads(row[0])
                return UserMemory(**data)
        return UserMemory(user_id=user_id)

    def save(self, memory: UserMemory):
        """保存用户记忆"""
        memory.updated_at = datetime.now().isoformat()
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "INSERT OR REPLACE INTO user_memory VALUES (?, ?, ?)",
                (memory.user_id, json.dumps(memory.__dict__, ensure_ascii=False, default=str),
                 memory.updated_at)
            )
            conn.commit()

    def record_interaction(
        self, user_id: str, topic: str, resolved: bool,
        solution: str = "", ticket: dict = None,
    ):
        """记录一次用户交互"""
        if not user_id:
            return
        m = self.get(user_id)

        m.recent_topics.append({
            "topic": topic,
            "resolved": resolved,
            "solution": solution,
            "timestamp": datetime.now().isoformat(),
        })
        m.recent_topics = m.recent_topics[-20:]  # 最多保留 20 条

        if ticket:
            m.recent_tickets.append(ticket)
            m.recent_tickets = m.recent_tickets[-10:]

        self.save(m)

    def inject_memory_context(self, user_id: str) -> str:
        """
        生成注入到 System Prompt 的上下文文本。
        返回空字符串表示无记忆可用。
        """
        if not user_id:
            return ""

        m = self.get(user_id)
        parts = []

        # ── 1. 问题模式检测 ──
        cutoff = (datetime.now() - timedelta(days=7)).isoformat()
        recent = [t for t in m.recent_topics if t.get("timestamp", "") >= cutoff]

        if recent:
            topic_counts: dict[str, int] = {}
            for t in recent:
                key = t.get("topic", "通用咨询")
                topic_counts[key] = topic_counts.get(key, 0) + 1

            frequent = [k for k, v in topic_counts.items() if v >= 2]
            if frequent:
                parts.append(
                    f"⚠️ 过去7天内用户反复遇到: {', '.join(frequent)}。"
                    f"如本次仍然无法解决，应主动建议创建工单而非重复相同方案。"
                )

            # 上次解决方案
            last_resolved = [t for t in recent if t.get("resolved") and t.get("solution")]
            if last_resolved:
                last = last_resolved[-1]
                parts.append(
                    f"💡 上次类似问题「{last.get('topic', '')}」的解决方案: "
                    f"{last.get('solution', '')[:150]}"
                )

        # ── 2. 近期工单 ──
        if m.recent_tickets:
            active_tickets = [
                t for t in m.recent_tickets
                if t.get("status") not in ("closed", "cancelled")
            ]
            if active_tickets:
                parts.append(f"📋 用户有 {len(active_tickets)} 个进行中的工单:")
                for t in active_tickets[-3:]:
                    parts.append(
                        f"  - [{t.get('type', '?')}] {t.get('title', '')} "
                        f"({t.get('status', '?')})"
                    )

        # ── 3. 累计统计 ──
        leave_count = sum(
            1 for t in m.recent_topics
            if any(kw in t.get("topic", "") for kw in ["请假", "休假", "年假", "病假", "事假"])
        )
        if leave_count > 0:
            parts.append(f"📊 用户本季度已申请请假 {leave_count} 次")

        return "\n".join(parts) if parts else ""


# 全局实例
user_memory_store = UserMemoryStore()

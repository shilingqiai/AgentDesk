"""
工单派发子Agent — 处理多类型工单创建与派发请求

负责：
- 识别工单类型（IT故障/请假/报销/行政服务）
- 从用户输入中提取对应参数
- 通过 TicketRepository 持久化工单
- 返回工单状态给编排器

支持的工单类型：
    it_fault  — IT故障报修（网络/硬件/系统）
    leave     — 请假申请（年假/病假/事假）
    expense   — 报销申请（差旅/办公/餐费）
    admin     — 行政服务（会议室/快递/资产领用）

YOU ARE A SUB-AGENT. DO NOT REPLY TO USER DIRECTLY.
MUST return structured findings to the Orchestrator.
"""

from __future__ import annotations

import json
import logging
import uuid
from typing import AsyncGenerator
from datetime import datetime

from agents.base_sub_agent import BaseSubAgent
from agents.a2a.protocol import AgentMessage
from agents.orchestrator.agent_declaration import agent_declaration
from agents.orchestrator.agent_registry import agent_registry
from config.model_provider import create_chat_model

logger = logging.getLogger("agent.ticket_dispatch")


# ============================================================
# 工单类型定义
# ============================================================

TICKET_TYPE_CONFIG = {
    "it_fault": {
        "label": "IT故障报修",
        "emoji": "🔧",
        "category_options": ["网络故障", "系统运维", "账号管理", "硬件故障", "软件问题", "安全事件", "其他"],
        "response_prefix": "已为您创建IT故障工单",
    },
    "leave": {
        "label": "请假申请",
        "emoji": "🏖️",
        "category_options": ["年假", "病假", "事假", "婚假", "产假", "调休", "其他"],
        "response_prefix": "已为您提交请假申请",
        "extra_fields": ["leave_type", "start_date", "end_date", "total_days", "reason"],
    },
    "expense": {
        "label": "报销申请",
        "emoji": "💰",
        "category_options": ["差旅费", "办公用品", "餐费", "交通费", "培训费", "其他"],
        "response_prefix": "已为您提交报销申请",
        "extra_fields": ["expense_type", "amount", "has_invoice", "description"],
    },
    "admin": {
        "label": "行政服务",
        "emoji": "🏢",
        "category_options": ["会议室预定", "快递寄送", "资产领用", "访客登记", "办公环境", "其他"],
        "response_prefix": "已为您创建行政服务请求",
        "extra_fields": ["service_type", "time_slot", "location", "description"],
    },
}


@agent_declaration(
    agent_id="ticket_dispatch",
    name="工单派发Agent",
    description=(
        "负责创建、查询和派发多类型工单。支持：IT故障报修、请假申请、报销申请、行政服务请求。"
        "当用户需要提交工单、请假、报销、预定会议室等操作时调用此Agent。"
        "从用户输入中提取工单参数，创建工单记录并返回状态。"
    ),
    capabilities=[
        "ticket_creation",
        "ticket_query",
        "parameter_extraction",
        "status_tracking",
        "leave_application",
        "expense_claim",
        "admin_service",
    ],
    knowledge_domains=[
        "ticket_management",
        "dispatch_workflow",
        "sla_enforcement",
        "leave_management",
        "expense_claim",
        "admin_service",
    ],
    priority=2,
)
class TicketDispatchSubAgent(BaseSubAgent):
    """
    工单派发子Agent（v2 — DB持久化 + 多类型支持）

    职责：
    1. 从用户输入中使用LLM识别工单类型并提取参数
    2. 通过 TicketRepository 持久化工单记录
    3. 返回结构化工单状态给编排器
    """

    agent_id = "ticket_dispatch"

    def __init__(self):
        super().__init__()
        self.llm = create_chat_model(temperature=0.1)
        self._db_router = None

    @property
    def db_router(self):
        """懒加载 DatabaseRouter"""
        if self._db_router is None:
            from db.db_router import DatabaseRouter
            self._db_router = DatabaseRouter()
        return self._db_router

    async def execute(self, message: AgentMessage) -> AgentMessage:
        """
        执行工单派发任务

        编排器委派的消息格式：
            payload.user_input: 用户原始输入
            payload.task: 任务描述
            payload.intent_category: 意图类别
            payload.urgency: 紧急程度
            payload.conversation_history: 对话历史（可选）

        返回格式：
            payload.ticket_id: 工单ID
            payload.ticket_number: 工单号
            payload.ticket_type: 工单类型
            payload.direct_response: 可展示给用户的工单状态消息
            payload.status: 工单状态
        """
        user_input = message.payload.get("user_input", "")
        task = message.payload.get("task", "")
        urgency = message.payload.get("urgency", "medium")
        conversation_history = message.payload.get("conversation_history", "")

        self.logger.info(
            f"[TicketDispatch] 处理工单请求 (trace={message.trace_id[:8]}...): "
            f"task=\"{task[:50]}\""
        )

        try:
            # 1. 使用 LLM 提取工单参数（含类型识别）
            ticket_params = await self._extract_params(
                user_input, urgency, conversation_history,
            )

            ticket_type = ticket_params.get("ticket_type", "it_fault")

            # 1.5 判断是否应该返回确认卡片
            if self._should_return_card(ticket_type, ticket_params):
                card = await self._build_card(ticket_type, ticket_params, user_input)
                return AgentMessage.create_response(
                    from_agent=self.agent_id,
                    to_agent=message.from_agent,
                    payload={
                        "direct_response": "",
                        "return_card": True,
                        "card": card,
                        "ticket_type": ticket_type,
                        "summary": f"[{TICKET_TYPE_CONFIG[ticket_type]['label']}] 等待用户确认",
                    },
                    original_message=message,
                    success=True,
                )

            # 2. 构建 payload（扩展字段）
            extra_payload = self._build_extra_payload(ticket_type, ticket_params)

            # 3. 创建工单（写入 DB）
            ticket = self.db_router.ticket.add_ticket(
                ticket_type=ticket_type,
                title=ticket_params.get("title", user_input[:30]),
                description=ticket_params.get("description", user_input),
                category=ticket_params.get("category", "其他"),
                priority=ticket_params.get("priority", "P2"),
                requester_id=message.payload.get("user_id", ""),
                trace_id=message.trace_id,
                payload=extra_payload,
            )

            # 4. 生成用户响应
            response = self._build_response(ticket, ticket_type)

            return AgentMessage.create_response(
                from_agent=self.agent_id,
                to_agent=message.from_agent,
                payload={
                    "direct_response": response,
                    "ticket_id": ticket["id"],
                    "ticket_number": ticket["ticket_number"],
                    "ticket_type": ticket_type,
                    "ticket_summary": ticket["title"],
                    "status": ticket["status"],
                    "priority": ticket["priority"],
                    "summary": (
                        f"[{TICKET_TYPE_CONFIG[ticket_type]['label']}] "
                        f"工单 {ticket['ticket_number']} 已创建"
                    ),
                    "needs_escalation": ticket["priority"] in ("P0", "P1"),
                },
                original_message=message,
                success=True,
            )

        except Exception as e:
            self.logger.error(f"工单派发失败: {e}")
            return self.create_error_response(message, str(e))

    @staticmethod
    def _should_return_card(ticket_type: str, params: dict) -> bool:
        """
        判断是否应该返回确认卡片而非直接创建工单。

        规则：
        - admin 类型（会议室预定等）→ 始终返回卡片
        - leave 类型 → 始终返回卡片（需用户确认日期/类型）
        - expense 类型 → 始终返回卡片（需用户确认金额/类型）
        - it_fault 类型 → 返回 RAG-first 卡片（先提供解决方案）
        """
        if ticket_type in ("admin", "leave", "expense", "it_fault"):
            return True
        return False

    # ============================================================
    # 时间表达式解析
    # ============================================================

    @staticmethod
    def _parse_time_expression(user_input: str, extra: dict) -> tuple:
        """
        从用户输入中智能解析日期和时间段。

        支持：
        - "明天早上" → (2026-06-09, 09:00, 10:30)
        - "今天下午" → (2026-06-08, 14:00, 16:00)
        - "后天上午" → (2026-06-10, 09:00, 11:00)
        - "周五下午3点" → (next Friday, 15:00, 16:30)
        - "下午2点到4点" → (today, 14:00, 16:00)

        Returns:
            (date_str, start_time, end_time)  均为字符串
        """
        from datetime import datetime, timedelta

        today = datetime.now()
        date_str = today.strftime("%Y-%m-%d")
        start_time = "14:00"
        end_time = "16:00"

        text = user_input

        # 1. 解析日期
        if "明天" in text:
            target = today + timedelta(days=1)
            date_str = target.strftime("%Y-%m-%d")
        elif "后天" in text:
            target = today + timedelta(days=2)
            date_str = target.strftime("%Y-%m-%d")
        elif "今天" in text:
            date_str = today.strftime("%Y-%m-%d")
        elif "大后天" in text:
            target = today + timedelta(days=3)
            date_str = target.strftime("%Y-%m-%d")

        # 星期几解析
        weekdays = {"周一": 0, "周二": 1, "周三": 2, "周四": 3,
                     "周五": 4, "周六": 5, "周日": 6,
                     "星期一": 0, "星期二": 1, "星期三": 2, "星期四": 3,
                     "星期五": 4, "星期六": 5, "星期日": 6}
        for name, wd in weekdays.items():
            if name in text:
                current_wd = today.weekday()
                days_ahead = wd - current_wd
                if days_ahead <= 0:
                    days_ahead += 7
                target = today + timedelta(days=days_ahead)
                date_str = target.strftime("%Y-%m-%d")
                break

        # 如果 extra 中有 start_date，优先使用
        if extra.get("start_date"):
            date_str = extra["start_date"]

        # 2. 解析时间段
        # 精确时间: "下午3点", "14:00", "2点到4点"
        import re

        # 尝试匹配 "X点到Y点" 或 "X:00-Y:00"
        time_range = re.search(
            r'(\d{1,2})[点:：](\d{0,2})?\s*[到至\-~]\s*(\d{1,2})[点:：](\d{0,2})?',
            text
        )
        if time_range:
            h1 = int(time_range.group(1))
            h2 = int(time_range.group(3))
            # 处理下午
            if "下午" in text and h1 < 12 and h1 >= 1:
                h1 += 12
            if "下午" in text and h2 < 12 and h2 >= 1:
                h2 += 12
            start_time = f"{h1:02d}:00"
            end_time = f"{h2:02d}:00"
            return date_str, start_time, end_time

        # "早上/上午/中午/下午/晚上" 关键时段
        if any(t in text for t in ("早上", "上午", "早晨")):
            start_time = "09:00"
            end_time = "10:30"
        elif "中午" in text:
            start_time = "12:00"
            end_time = "13:30"
        elif any(t in text for t in ("下午", "午后")):
            # 检查是否有具体时间，如 "下午3点"
            hour_match = re.search(r'(\d{1,2})点', text)
            if hour_match:
                h = int(hour_match.group(1))
                if h < 9 and h >= 1:  # 下午1-8点 → 13-20点
                    h += 12
                h = max(8, min(20, h))
                start_time = f"{h:02d}:00"
                end_h = min(h + 2, 20)
                end_time = f"{end_h:02d}:00"
            else:
                start_time = "14:00"
                end_time = "16:00"
        elif "晚上" in text:
            start_time = "18:00"
            end_time = "20:00"

        # extra 中的 time_slot 优先级
        if extra.get("time_slot"):
            slot = extra["time_slot"]
            if "-" in slot:
                parts = slot.split("-")
                if len(parts) == 2:
                    start_time = parts[0].strip()
                    end_time = parts[1].strip()

        return date_str, start_time, end_time

    # ============================================================
    # 用户偏好记忆
    # ============================================================

    @staticmethod
    def _get_user_preferences(user_id: str = "") -> dict:
        """
        从历史预定记录推断用户偏好。

        Returns:
            {
                "preferred_room_id": int or None,
                "avg_duration_minutes": int (default 60),
                "preferred_start_time": "09:00" or None,
                "recent_titles": ["周会", "项目评审", ...],
                "total_bookings": int,
            }
        """
        prefs = {
            "preferred_room_id": None,
            "avg_duration_minutes": 60,
            "preferred_start_time": None,
            "recent_titles": [],
            "total_bookings": 0,
        }

        try:
            from db.models import MeetingRoomBooking
            from sqlalchemy import create_engine, func
            from sqlalchemy.orm import sessionmaker
            import os

            db_path = os.path.join("data", "ticket_dispatch.db")
            engine = create_engine(
                f"sqlite:///{db_path}", connect_args={"check_same_thread": False}
            )
            SessionLocal = sessionmaker(bind=engine)
            db = SessionLocal()
            try:
                query = db.query(MeetingRoomBooking).filter(
                    MeetingRoomBooking.is_active == 1,
                )
                if user_id:
                    query = query.filter(MeetingRoomBooking.booked_by == user_id)

                bookings = query.order_by(
                    MeetingRoomBooking.created_at.desc()
                ).limit(10).all()

                if not bookings:
                    return prefs

                prefs["total_bookings"] = len(bookings)

                # 最常用的会议室
                room_counts = {}
                for b in bookings:
                    room_counts[b.room_id] = room_counts.get(b.room_id, 0) + 1
                prefs["preferred_room_id"] = max(room_counts, key=room_counts.get)

                # 平均时长
                durations = []
                start_times = []
                for b in bookings:
                    try:
                        from datetime import datetime
                        s = datetime.strptime(b.start_time, "%H:%M")
                        e = datetime.strptime(b.end_time, "%H:%M")
                        diff = (e - s).total_seconds() / 60
                        if diff > 0:
                            durations.append(diff)
                        start_times.append(b.start_time)
                    except Exception:
                        pass

                if durations:
                    avg_dur = int(sum(durations) / len(durations))
                    # 取整到 30 分钟
                    avg_dur = round(avg_dur / 30) * 30
                    prefs["avg_duration_minutes"] = max(30, avg_dur)

                # 最常用的开始时间
                if start_times:
                    from collections import Counter
                    prefs["preferred_start_time"] = Counter(start_times).most_common(1)[0][0]

                # 最近的会议主题
                prefs["recent_titles"] = [
                    b.title for b in bookings[:5] if b.title
                ]

            finally:
                db.close()
        except Exception:
            pass

        return prefs

    # ============================================================
    # 冲突检测 + 替代方案
    # ============================================================

    @staticmethod
    def _check_availability_with_alternatives(
        date_str: str,
        start_time: str,
        end_time: str,
        preferred_room_id: int = None,
    ) -> dict:
        """
        检查会议室可用性，如果冲突则提供替代方案。

        Returns:
            {
                "available": bool,
                "conflict": None or {"room_name": "...", "booking_title": "..."},
                "available_rooms": [{"id": 1, "name": "A101", "available": True}, ...],
                "alternative_slots": [{"start": "10:00", "end": "11:30"}, ...],
            }
        """
        result = {
            "available": True,
            "conflict": None,
            "available_rooms": [],
            "alternative_slots": [],
        }

        try:
            from db.models import MeetingRoom, MeetingRoomBooking
            from sqlalchemy import create_engine
            from sqlalchemy.orm import sessionmaker
            from datetime import datetime, timedelta
            import os

            db_path = os.path.join("data", "ticket_dispatch.db")
            engine = create_engine(
                f"sqlite:///{db_path}", connect_args={"check_same_thread": False}
            )
            SessionLocal = sessionmaker(bind=engine)
            db = SessionLocal()
            try:
                # 1. 检查首选房间是否可用
                if preferred_room_id:
                    conflict = db.query(MeetingRoomBooking).filter(
                        MeetingRoomBooking.room_id == preferred_room_id,
                        MeetingRoomBooking.date == date_str,
                        MeetingRoomBooking.status == "confirmed",
                        MeetingRoomBooking.is_active == 1,
                        MeetingRoomBooking.start_time < end_time,
                        MeetingRoomBooking.end_time > start_time,
                    ).first()

                    if conflict:
                        room = db.query(MeetingRoom).filter(
                            MeetingRoom.id == preferred_room_id
                        ).first()
                        result["available"] = False
                        result["conflict"] = {
                            "room_name": room.name if room else str(preferred_room_id),
                            "booking_title": conflict.title,
                            "time": f"{conflict.start_time}-{conflict.end_time}",
                        }

                # 2. 查找同时段可用的所有房间
                all_rooms = db.query(MeetingRoom).filter(
                    MeetingRoom.is_active == 1,
                    MeetingRoom.status == "available",
                ).all()

                for room in all_rooms:
                    has_conflict = db.query(MeetingRoomBooking).filter(
                        MeetingRoomBooking.room_id == room.id,
                        MeetingRoomBooking.date == date_str,
                        MeetingRoomBooking.status == "confirmed",
                        MeetingRoomBooking.is_active == 1,
                        MeetingRoomBooking.start_time < end_time,
                        MeetingRoomBooking.end_time > start_time,
                    ).first()

                    entry = {
                        "id": room.id,
                        "name": room.name,
                        "capacity": room.capacity,
                        "location": room.location,
                        "available": not bool(has_conflict),
                    }
                    if has_conflict:
                        entry["conflict_with"] = has_conflict.title
                    result["available_rooms"].append(entry)

                # 按可用优先 + 首选房间优先排序
                result["available_rooms"].sort(
                    key=lambda r: (
                        not r["available"],
                        0 if r["id"] == preferred_room_id else 1,
                    )
                )

                # 3. 如果首选不可用，为该房间找邻近时段
                if result["conflict"] and preferred_room_id:
                    # 在该日期已有的预定
                    existing = db.query(MeetingRoomBooking).filter(
                        MeetingRoomBooking.room_id == preferred_room_id,
                        MeetingRoomBooking.date == date_str,
                        MeetingRoomBooking.status == "confirmed",
                        MeetingRoomBooking.is_active == 1,
                    ).order_by(MeetingRoomBooking.start_time).all()

                    # 构建占用时段列表
                    busy_slots = [
                        (b.start_time, b.end_time) for b in existing
                    ]

                    # 在 08:00-20:00 范围内找空闲窗口
                    day_start = datetime.strptime("08:00", "%H:%M")
                    day_end = datetime.strptime("20:00", "%H:%M")
                    req_duration = (
                        datetime.strptime(end_time, "%H:%M") -
                        datetime.strptime(start_time, "%H:%M")
                    )

                    cursor = day_start
                    while cursor + req_duration <= day_end:
                        slot_start = cursor.strftime("%H:%M")
                        slot_end = (cursor + req_duration).strftime("%H:%M")

                        # 检查是否与占用冲突
                        free = True
                        for bs, be in busy_slots:
                            if slot_start < be and slot_end > bs:
                                free = False
                                break

                        if free:
                            result["alternative_slots"].append({
                                "start": slot_start,
                                "end": slot_end,
                            })
                            if len(result["alternative_slots"]) >= 3:
                                break

                        cursor += timedelta(minutes=30)

            finally:
                db.close()
        except Exception:
            pass

        return result

    # ============================================================
    # 智能卡片构建
    # ============================================================

    async def _build_card(self, ticket_type: str, params: dict, user_input: str) -> dict:
        """
        构建智能确认卡片。

        核心改进：
        - 从用户输入解析自然时间表达（"明天早上"→具体日期+弹性时段）
        - 从用户历史预定推断偏好（常用房间、平均时长、常用时间）
        - 检测会议室冲突，提供替代房间/时间建议
        """
        now = datetime.now()

        if ticket_type == "admin":
            extra = params.get("extra", {})
            service_type = extra.get("service_type", "")

            # 1. 智能解析时间
            parsed_date, parsed_start, parsed_end = self._parse_time_expression(
                user_input, extra
            )

            # 2. 获取用户偏好
            user_prefs = self._get_user_preferences(user_id="")
            preferred_room_id = user_prefs.get("preferred_room_id")

            # 根据用户历史平均时长调整 end_time
            if user_prefs.get("avg_duration_minutes"):
                from datetime import datetime as dt, timedelta
                avg_min = user_prefs["avg_duration_minutes"]
                st = dt.strptime(parsed_start, "%H:%M")
                et = st + timedelta(minutes=avg_min)
                if et <= dt.strptime("20:00", "%H:%M"):
                    parsed_end = et.strftime("%H:%M")

            # 3. 冲突检测
            availability = self._check_availability_with_alternatives(
                parsed_date, parsed_start, parsed_end, preferred_room_id,
            )

            # 4. 获取会议室列表（用于下拉）
            room_options = []
            try:
                from db.models import MeetingRoom
                from sqlalchemy import create_engine
                from sqlalchemy.orm import sessionmaker
                import os

                db_path = os.path.join("data", "ticket_dispatch.db")
                engine = create_engine(
                    f"sqlite:///{db_path}", connect_args={"check_same_thread": False}
                )
                SessionLocal = sessionmaker(bind=engine)
                db = SessionLocal()
                try:
                    rooms = db.query(MeetingRoom).filter(
                        MeetingRoom.is_active == 1,
                        MeetingRoom.status == "available",
                    ).all()
                    room_options = [
                        {
                            "value": str(r.id),
                            "label": f"{r.name} ({r.capacity}人) — {r.location}",
                        }
                        for r in rooms
                    ]
                finally:
                    db.close()
            except Exception:
                room_options = [
                    {"value": str(i), "label": f"{n} ({c}人)"}
                    for i, n, c in [
                        (1, "A101 星空厅", 6), (2, "A201 银河厅", 12),
                        (3, "A301 宇宙厅", 25), (4, "B101 创意坊", 8),
                        (5, "B201 静思阁", 4),
                    ]
                ]

            # 5. 默认选中的会议室：首选 > 第一个可用的
            default_room = str(preferred_room_id) if preferred_room_id else None
            if not default_room and room_options:
                default_room = room_options[0]["value"]

            # 如果首选不可用，自动推荐第一个可用房间
            if availability.get("conflict") and availability["available_rooms"]:
                for r in availability["available_rooms"]:
                    if r["available"]:
                        default_room = str(r["id"])
                        break

            # 6. 生成时段选项（包含当前时长和邻近时段）
            time_options = [
                {"value": f"{parsed_start}-{parsed_end}",
                 "label": f"{parsed_start}-{parsed_end} ✨ 推荐"},
            ]
            # 添加源自用户偏好的其他常见时段
            alt_slots = availability.get("alternative_slots", [])
            for slot in alt_slots[:2]:
                slot_val = f"{slot['start']}-{slot['end']}"
                if slot_val not in [o["value"] for o in time_options]:
                    time_options.append({
                        "value": slot_val,
                        "label": f"{slot['start']}-{slot['end']} 🔄 备选",
                    })

            # 7. 构建描述文字
            time_desc = {
                "09:00": "早上", "10:00": "上午", "12:00": "中午",
                "14:00": "下午", "18:00": "晚上",
            }
            friendly_time = ""
            for t, label in time_desc.items():
                if parsed_start.startswith(t):
                    friendly_time = label
                    break
            if not friendly_time:
                friendly_time = parsed_start

            weekday_names = ["周一", "周二", "周三", "周四", "周五", "周六", "周日"]
            try:
                target_dt = datetime.strptime(parsed_date, "%Y-%m-%d")
                weekday_label = weekday_names[target_dt.weekday()]
                date_label = f"{parsed_date} ({weekday_label})"
            except Exception:
                date_label = parsed_date

            desc_parts = [f"📅 {date_label}  {friendly_time} {parsed_start}-{parsed_end}"]
            if user_prefs.get("total_bookings", 0) > 0:
                desc_parts.append(
                    f"💡 根据您的历史记录，推荐时长 {user_prefs['avg_duration_minutes']} 分钟"
                )
            if availability.get("conflict"):
                c = availability["conflict"]
                desc_parts.append(
                    f"⚠️ {c['room_name']} 在 {c['time']} 已被「{c['booking_title']}」占用，"
                    f"已自动推荐其他可用会议室"
                )

            card = {
                "type": "booking",
                "title": "📋 会议室预定",
                "description": "\n".join(desc_parts),
                "fields": [
                    {
                        "key": "room_id", "label": "会议室", "type": "select",
                        "options": room_options,
                        "value": default_room,
                        "required": True,
                        "hint": "已根据可用性筛选" if availability.get("conflict") else "",
                    },
                    {
                        "key": "date", "label": "日期", "type": "date",
                        "value": parsed_date, "required": True,
                    },
                    {
                        "key": "time_slot", "label": "时间段", "type": "select",
                        "options": time_options,
                        "value": time_options[0]["value"],
                        "required": True,
                    },
                    {
                        "key": "title", "label": "会议主题", "type": "text",
                        "placeholder": "输入会议主题...",
                        "value": user_prefs.get("recent_titles", [None])[0] if user_prefs.get("recent_titles") else "",
                        "required": True,
                    },
                ],
                "confirm_text": "确认预定",
                "action": "/api/meeting-rooms/{room_id}/book",
                "success_message": (
                    f"✅ {parsed_date} {parsed_start}-{parsed_end} 会议室预定成功！"
                ),
                "fallback_url": "/meeting-rooms",
                "fallback_text": "查看会议室日历",
                # 冲突提示 + 替代建议
                "alerts": [],
            }

            # 添加冲突提示
            if availability.get("conflict"):
                card["alerts"].append({
                    "type": "warning",
                    "message": (
                        f"⚠️ {availability['conflict']['room_name']} 在该时段已被预定。"
                        f"已自动推荐可用会议室。"
                    ),
                })
            for slot in alt_slots[:3]:
                if slot["start"] != parsed_start:
                    card["alerts"].append({
                        "type": "info",
                        "message": f"💡 也可选择 {slot['start']}-{slot['end']} 时段",
                    })

            if not card["alerts"]:
                del card["alerts"]

            return card

        elif ticket_type == "leave":
            extra = params.get("extra", {})
            # 解析日期
            parsed_date, _, _ = self._parse_time_expression(user_input, extra)
            default_start = parsed_date if extra.get("start_date") or "明天" in user_input or "后天" in user_input else ""

            total_days = extra.get("total_days", 0)
            default_end = ""
            if default_start and total_days:
                try:
                    from datetime import datetime as dt, timedelta
                    sd = dt.strptime(default_start, "%Y-%m-%d")
                    ed = sd + timedelta(days=total_days - 1)
                    default_end = ed.strftime("%Y-%m-%d")
                except Exception:
                    pass

            return {
                "type": "confirm",
                "title": "🏖️ 请假申请",
                "description": "请确认以下请假信息，信息已根据您的输入预填：",
                "fields": [
                    {
                        "key": "leave_type", "label": "请假类型", "type": "select",
                        "options": [
                            {"value": "年假", "label": "年假"},
                            {"value": "病假", "label": "病假"},
                            {"value": "事假", "label": "事假"},
                            {"value": "调休", "label": "调休"},
                            {"value": "婚假", "label": "婚假"},
                        ],
                        "value": extra.get("leave_type", "年假"),
                        "required": True,
                    },
                    {
                        "key": "start_date", "label": "开始日期", "type": "date",
                        "value": default_start, "required": True,
                    },
                    {
                        "key": "end_date", "label": "结束日期", "type": "date",
                        "value": default_end, "required": True,
                    },
                    {
                        "key": "total_days", "label": "天数", "type": "number",
                        "value": str(total_days) if total_days else "",
                        "min": 1, "max": 30,
                        "required": True,
                    },
                ],
                "confirm_text": "提交请假申请",
                "action": "/api/tickets/",
                "method": "POST",
                "body_template": {
                    "user_input": user_input,
                    "ticket_type": "leave",
                    "priority": params.get("priority", "P2"),
                },
                "success_message": "请假申请已提交！可在工单管理页面查看进度。",
                "fallback_url": "/tickets",
                "fallback_text": "查看工单",
            }

        elif ticket_type == "expense":
            extra = params.get("extra", {})
            return {
                "type": "confirm",
                "title": "💰 报销申请",
                "description": "请确认报销信息：",
                "fields": [
                    {
                        "key": "expense_type", "label": "报销类型", "type": "select",
                        "options": [
                            {"value": "差旅费", "label": "差旅费"},
                            {"value": "办公用品", "label": "办公用品"},
                            {"value": "交通费", "label": "交通费"},
                            {"value": "餐费", "label": "餐费"},
                            {"value": "培训费", "label": "培训费"},
                        ],
                        "value": extra.get("expense_type", "差旅费"),
                        "required": True,
                    },
                    {
                        "key": "amount", "label": "金额（元）", "type": "number",
                        "value": str(extra.get("amount", "")),
                        "min": 0, "required": True,
                    },
                    {
                        "key": "description", "label": "说明", "type": "text",
                        "value": params.get("description", user_input),
                        "required": False,
                    },
                ],
                "confirm_text": "提交报销申请",
                "action": "/api/tickets/",
                "method": "POST",
                "body_template": {
                    "user_input": user_input,
                    "ticket_type": "expense",
                    "priority": params.get("priority", "P2"),
                },
                "success_message": "报销申请已提交！请保留原始发票。",
                "fallback_url": "/tickets",
                "fallback_text": "查看工单",
            }

        elif ticket_type == "it_fault":
            # IT 故障：先搜索知识库提供解决方案
            rag_answer = ""
            try:
                from services.knowledge_service import KnowledgeService
                ks = KnowledgeService()
                await ks.initialize()
                docs = await ks.search(user_input, top_k=3)
                if docs:
                    from agents.sub_agents.enterprise_rag import EnterpriseRAGAgent
                    rag = EnterpriseRAGAgent()
                    rag.knowledge_service = ks
                    rag._initialized = True
                    doc_context = rag._build_doc_context(docs)
                    rag_answer = await rag._synthesize(
                        user_input, docs, ""
                    )
            except Exception:
                pass

            description_parts = []
            if rag_answer:
                # 截取前 500 字作为摘要
                short = rag_answer[:500]
                if len(rag_answer) > 500:
                    short += "..."
                description_parts.append(f"🔍 **知识库匹配到以下解决方案：**\n\n{short}")
                description_parts.append("\n---\n💡 如果以上方案未解决您的问题，请点击下方按钮创建工单。")
            else:
                description_parts.append("未在知识库中找到相关解决方案，请确认是否创建工单。")

            return {
                "type": "confirm",
                "title": "🔧 IT 故障排查",
                "description": "\n".join(description_parts),
                "fields": [
                    {
                        "key": "title", "label": "问题标题", "type": "text",
                        "value": params.get("title", user_input[:30]), "required": True,
                    },
                    {
                        "key": "description", "label": "详细描述", "type": "text",
                        "value": params.get("description", user_input), "required": True,
                    },
                ],
                "confirm_text": "仍需要帮助，创建工单",
                "dismiss_text": "问题已解决",
                "action": "/api/tickets/",
                "method": "POST",
                "body_template": {
                    "user_input": user_input,
                    "ticket_type": "it_fault",
                    "priority": params.get("priority", "P2"),
                },
                "success_message": "IT工单已创建，工程师将尽快处理。",
                "fallback_url": "/tickets",
                "fallback_text": "查看工单进度",
            }

        # 默认卡
        return {
            "type": "confirm",
            "title": f"{TICKET_TYPE_CONFIG.get(ticket_type, {}).get('emoji', '📋')} 确认创建工单",
            "description": f"即将创建 {TICKET_TYPE_CONFIG.get(ticket_type, {}).get('label', '工单')}：",
            "fields": [
                {"key": "title", "label": "标题", "type": "text",
                 "value": params.get("title", user_input[:30]), "required": True},
                {"key": "description", "label": "描述", "type": "text",
                 "value": params.get("description", user_input), "required": False},
            ],
            "confirm_text": "确认创建",
            "action": "/api/tickets/",
            "method": "POST",
            "body_template": {
                "user_input": user_input,
                "ticket_type": ticket_type,
                "priority": params.get("priority", "P2"),
            },
            "success_message": "工单已创建！",
        }

    async def _extract_params(
        self,
        user_input: str,
        urgency: str = "medium",
        conversation_history: str = "",
    ) -> dict:
        """
        使用 LLM 从用户输入中提取工单参数

        核心改进（v2）：
        - 自动识别 ticket_type
        - 针对不同 ticket_type 提取不同字段
        - JSON 解析失败时使用 json_repair 修复
        - 最多重试 2 次
        """
        history_section = ""
        if conversation_history:
            history_section = (
                f"## 对话历史\n{conversation_history}\n\n"
                f"注意：结合上下文理解用户意图。\n\n"
            )

        ticket_type_options = "\n".join([
            f"  - {k}: {v['label']}（如：{'/'.join(v['category_options'][:3])}...）"
            for k, v in TICKET_TYPE_CONFIG.items()
        ])

        prompt = f"""你是一个企业工单系统的参数提取器。识别工单类型并提取信息。

{history_section}
## 工单类型
{ticket_type_options}

## 用户输入
{user_input}

## 默认值
- 紧急度: {urgency}
- 如用户未明确指定，priority 默认为 P2

## 输出 JSON（严格按此格式，不要其他文字）
{{
    "ticket_type": "it_fault|leave|expense|admin",
    "title": "工单标题（简洁，<20字）",
    "description": "工单详细描述",
    "category": "具体分类（从上述分类中选）",
    "priority": "P0|P1|P2|P3",
    "extra": {{
        "leave_type": "年假|病假|事假（仅 leave 类型）",
        "start_date": "开始日期（仅 leave 类型，YYYY-MM-DD）",
        "end_date": "结束日期（仅 leave 类型）",
        "total_days": 0,
        "expense_type": "差旅费|办公用品...（仅 expense 类型）",
        "amount": 0.0,
        "has_invoice": true,
        "service_type": "会议室|快递...（仅 admin 类型）",
        "time_slot": "时间段（仅 admin 类型）"
    }}
}}

优先级判断标准：
- P0: 系统宕机、核心业务中断、多人受影响
- P1: 影响工作效率但可暂时绕过
- P2: 一般故障/申请，有替代方案
- P3: 咨询、非紧急问题"""

        for attempt in range(2):
            try:
                response = await self.llm.ainvoke([{"role": "user", "content": prompt}])
                content = response.content.strip()

                # 提取 JSON（兼容 markdown 代码块）
                content = self._extract_json(content)

                data = json.loads(content)

                # 验证 ticket_type
                ticket_type = data.get("ticket_type", "it_fault")
                if ticket_type not in TICKET_TYPE_CONFIG:
                    ticket_type = "it_fault"

                return {
                    "ticket_type": ticket_type,
                    "title": data.get("title", user_input[:30]),
                    "description": data.get("description", user_input),
                    "category": data.get("category", "其他"),
                    "priority": data.get("priority", "P2"),
                    "extra": data.get("extra", {}),
                }

            except (json.JSONDecodeError, KeyError, ValueError) as e:
                self.logger.warning(f"参数提取尝试 {attempt + 1}/2 失败: {e}")

                # 尝试 json_repair
                if attempt == 0:
                    try:
                        from json_repair import repair_json
                        content = repair_json(content)
                        self.logger.info("JSON 修复成功，重试解析")
                        # 继续循环以重新解析
                    except ImportError:
                        pass

        # 所有重试失败 → 规则兜底
        self.logger.warning("LLM 参数提取全部失败，使用规则兜底")
        return self._fallback_extract(user_input, urgency)

    def _fallback_extract(self, user_input: str, urgency: str) -> dict:
        """规则兜底：基于关键词快速推断 ticket_type"""
        text = user_input.lower()

        # 关键词匹配 ticket_type
        leave_keywords = ["请假", "休假", "年假", "病假", "事假", "调休", "婚假", "产假"]
        expense_keywords = ["报销", "差旅", "发票", "费用", "账单"]
        admin_keywords = ["会议室", "快递", "寄送", "资产", "访客", "预定", "预约"]

        if any(kw in text for kw in leave_keywords):
            ticket_type = "leave"
            category = "其他"
        elif any(kw in text for kw in expense_keywords):
            ticket_type = "expense"
            category = "其他"
        elif any(kw in text for kw in admin_keywords):
            ticket_type = "admin"
            category = "其他"
        else:
            ticket_type = "it_fault"
            category = "其他"

        return {
            "ticket_type": ticket_type,
            "title": user_input[:30],
            "description": user_input,
            "category": category,
            "priority": "P2",
            "extra": {},
        }

    @staticmethod
    def _extract_json(content: str) -> str:
        """从 LLM 响应中提取 JSON 内容"""
        if "```json" in content:
            return content.split("```json")[1].split("```")[0].strip()
        elif "```" in content:
            return content.split("```")[1].split("```")[0].strip()
        # 尝试找到第一个 { 和最后一个 }
        brace_start = content.find("{")
        brace_end = content.rfind("}")
        if brace_start != -1 and brace_end != -1 and brace_end > brace_start:
            return content[brace_start:brace_end + 1].strip()
        return content

    @staticmethod
    def _build_extra_payload(ticket_type: str, params: dict) -> dict:
        """根据 ticket_type 构建扩展字段 payload"""
        extra = params.get("extra", {})

        if ticket_type == "leave":
            return {
                "leave_type": extra.get("leave_type", ""),
                "start_date": extra.get("start_date", ""),
                "end_date": extra.get("end_date", ""),
                "total_days": extra.get("total_days", 0),
                "reason": extra.get("reason", params.get("description", "")),
            }
        elif ticket_type == "expense":
            return {
                "expense_type": extra.get("expense_type", ""),
                "amount": extra.get("amount", 0.0),
                "has_invoice": extra.get("has_invoice", False),
            }
        elif ticket_type == "admin":
            return {
                "service_type": extra.get("service_type", ""),
                "time_slot": extra.get("time_slot", ""),
                "location": extra.get("location", ""),
            }
        else:  # it_fault
            return {
                "suggested_skill": params.get("suggested_engineer_skill", "通用"),
                "affected_users": extra.get("affected_users", 1),
            }

    def _build_response(self, ticket: dict, ticket_type: str) -> str:
        """根据 ticket_type 构建差异化用户响应"""
        config = TICKET_TYPE_CONFIG.get(ticket_type, TICKET_TYPE_CONFIG["it_fault"])
        priority_emoji = {"P0": "🔴", "P1": "🟠", "P2": "🟡", "P3": "🟢"}
        emoji = priority_emoji.get(ticket["priority"], "📋")
        type_emoji = config["emoji"]

        # 基础信息
        lines = [
            f"{type_emoji} **{config['label']}工单已创建**",
            "",
            f"**工单编号**：{ticket['ticket_number']}",
            f"**标题**：{ticket['title']}",
            f"**优先级**：{emoji} {ticket['priority']}",
            f"**分类**：{ticket['category']}",
            f"**状态**：{ticket['status']}",
        ]

        # 类型特有信息
        payload = ticket.get("payload", {})
        if ticket_type == "leave" and payload:
            if leave_type := payload.get("leave_type"):
                lines.append(f"**请假类型**：{leave_type}")
            if start := payload.get("start_date"):
                lines.append(f"**开始日期**：{start}")
            if end := payload.get("end_date"):
                lines.append(f"**结束日期**：{end}")
            if days := payload.get("total_days"):
                lines.append(f"**天数**：{days}天")

        elif ticket_type == "expense" and payload:
            if expense_type := payload.get("expense_type"):
                lines.append(f"**报销类型**：{expense_type}")
            if amount := payload.get("amount"):
                lines.append(f"**金额**：¥{amount}")
            lines.append(f"**是否有发票**：{'是' if payload.get('has_invoice') else '待确认'}")

        elif ticket_type == "admin" and payload:
            if service_type := payload.get("service_type"):
                lines.append(f"**服务类型**：{service_type}")
            if time_slot := payload.get("time_slot"):
                lines.append(f"**时间段**：{time_slot}")

        # 进度提示
        lines.append("")
        if ticket["priority"] in ("P0", "P1"):
            lines.append("⚡ 您的请求已标记为高优先级，将优先处理。")
        else:
            lines.append("您的工单已进入处理队列，请耐心等待。")

        # 备注
        lines.append("")
        if ticket_type == "leave":
            lines.append("💡 请确认已通过OA系统同步提交请假审批。")
        elif ticket_type == "expense":
            lines.append("💡 请保留原始发票，后续需提交至财务部。")
        elif ticket_type == "admin":
            lines.append("💡 行政人员将在1个工作日内确认并回复。")
        else:
            lines.append("💡 如需查询工单进度，可随时询问我。")

        return "\n".join(lines)

    # ============================================================
    # 查询方法
    # ============================================================

    @classmethod
    def get_ticket(cls, ticket_id: int) -> dict | None:
        """查询工单（DB）"""
        from db.db_router import DatabaseRouter
        db = DatabaseRouter()
        return db.ticket.get_ticket(ticket_id)

    @classmethod
    def get_all_tickets(cls, ticket_type: str = None, limit: int = 50) -> list[dict]:
        """获取所有工单（DB）"""
        from db.db_router import DatabaseRouter
        db = DatabaseRouter()
        return db.ticket.list_tickets(ticket_type=ticket_type, limit=limit)

    async def execute_stream(self, message: AgentMessage) -> AsyncGenerator[str, None]:
        """流式执行"""
        yield "[TicketDispatch] 正在分析工单需求..."
        yield "[TicketDispatch] 正在识别工单类型（IT/请假/报销/行政）..."
        yield "[TicketDispatch] 正在提取关键参数..."
        yield "[TicketDispatch] 正在创建工单..."
        yield "[TicketDispatch] 工单已创建，结果返回给编排器"


# 自动注册到全局注册中心
def _register():
    agent_registry.register(
        TicketDispatchSubAgent.__agent_declaration__,
        TicketDispatchSubAgent,
    )

_register()

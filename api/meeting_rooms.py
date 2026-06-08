"""
会议室管理 API

提供会议室查询、可用性检查、预定/取消等功能。
"""

from __future__ import annotations

import os
from typing import Optional
from datetime import datetime, timedelta

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field
from sqlalchemy import create_engine, func
from sqlalchemy.orm import Session, sessionmaker

router = APIRouter(prefix="/api/meeting-rooms", tags=["会议室管理"])

DB_PATH = os.path.join("data", "ticket_dispatch.db")
os.makedirs("data", exist_ok=True)
_engine = create_engine(
    f"sqlite:///{DB_PATH}",
    connect_args={"check_same_thread": False},
    echo=False,
)
_SessionLocal = sessionmaker(bind=_engine)


def _get_db() -> Session:
    return _SessionLocal()


# ============================================================
# 默认会议室数据
# ============================================================

DEFAULT_ROOMS = [
    {"name": "A101 星空厅", "capacity": 6, "location": "A栋 1层",
     "amenities": ["投影仪", "白板", "WiFi"], "description": "小型讨论室，适合6人以下团队"},
    {"name": "A201 银河厅", "capacity": 12, "location": "A栋 2层",
     "amenities": ["投影仪", "白板", "视频会议", "WiFi"], "description": "中型会议室，配视频会议设备"},
    {"name": "A301 宇宙厅", "capacity": 25, "location": "A栋 3层",
     "amenities": ["投影仪", "音响", "视频会议", "白板", "茶水"], "description": "大型会议室，可容纳25人"},
    {"name": "B101 创意坊", "capacity": 8, "location": "B栋 1层",
     "amenities": ["白板墙", "站立桌", "WiFi", "显示器"], "description": "创意讨论室，开放式布局"},
    {"name": "B201 静思阁", "capacity": 4, "location": "B栋 2层",
     "amenities": ["白板", "WiFi", "电话会议"], "description": "小型洽谈室，适合1对1面试或私密会议"},
]


def _ensure_tables():
    """确保会议室相关表存在"""
    from db.models import Base
    Base.metadata.create_all(bind=_engine)


def _seed_rooms():
    """初始化默认会议室数据"""
    _ensure_tables()
    db = _get_db()
    try:
        from db.models import MeetingRoom
        existing = db.query(func.count(MeetingRoom.id)).scalar()
        if existing == 0:
            for room in DEFAULT_ROOMS:
                db.add(MeetingRoom(**room))
            db.commit()
    finally:
        db.close()


# ============================================================
# 请求/响应模型
# ============================================================

class BookingRequest(BaseModel):
    """预定会议室请求"""
    date: str = Field(..., description="预定日期 YYYY-MM-DD")
    start_time: str = Field(..., description="开始时间 HH:MM")
    end_time: str = Field(..., description="结束时间 HH:MM")
    title: str = Field(..., description="会议主题")
    description: Optional[str] = Field(default="", description="会议描述")
    booked_by: Optional[str] = Field(default="", description="预定人")


# ============================================================
# 端点
# ============================================================

@router.get("/", summary="获取会议室列表")
async def list_rooms(
    status: Optional[str] = Query(default=None, description="筛选状态: available/maintenance"),
):
    """获取所有可用会议室"""
    try:
        _seed_rooms()
        db = _get_db()
        try:
            from db.models import MeetingRoom
            query = db.query(MeetingRoom).filter(MeetingRoom.is_active == 1)
            if status:
                query = query.filter(MeetingRoom.status == status)
            rooms = query.order_by(MeetingRoom.name).all()

            return {
                "status": "success",
                "data": [
                    {
                        "id": r.id,
                        "name": r.name,
                        "capacity": r.capacity,
                        "location": r.location,
                        "amenities": r.amenities or [],
                        "description": r.description or "",
                        "status": r.status,
                    }
                    for r in rooms
                ],
            }
        finally:
            db.close()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取会议室列表失败: {str(e)}")


@router.get("/{room_id}", summary="获取会议室详情")
async def get_room(room_id: int):
    """获取单个会议室详情"""
    try:
        _seed_rooms()
        db = _get_db()
        try:
            from db.models import MeetingRoom
            room = db.query(MeetingRoom).filter(
                MeetingRoom.id == room_id,
                MeetingRoom.is_active == 1,
            ).first()
            if not room:
                raise HTTPException(status_code=404, detail="会议室不存在")

            return {
                "status": "success",
                "data": {
                    "id": room.id,
                    "name": room.name,
                    "capacity": room.capacity,
                    "location": room.location,
                    "amenities": room.amenities or [],
                    "description": room.description or "",
                    "status": room.status,
                },
            }
        finally:
            db.close()
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取会议室失败: {str(e)}")


@router.get("/{room_id}/availability", summary="检查会议室可用时段")
async def check_availability(
    room_id: int,
    date: str = Query(..., description="查询日期 YYYY-MM-DD"),
):
    """
    返回会议室在指定日期的可用时段。

    时间范围: 08:00-20:00，以 30 分钟为粒度。
    返回每个时段是否可用。
    """
    try:
        _seed_rooms()
        db = _get_db()
        try:
            from db.models import MeetingRoom, MeetingRoomBooking

            room = db.query(MeetingRoom).filter(
                MeetingRoom.id == room_id, MeetingRoom.is_active == 1,
            ).first()
            if not room:
                raise HTTPException(status_code=404, detail="会议室不存在")

            if room.status != "available":
                return {
                    "status": "success",
                    "data": {
                        "room_id": room_id,
                        "room_name": room.name,
                        "date": date,
                        "room_status": room.status,
                        "slots": [],
                        "bookings": [],
                    },
                }

            # 查询当天已有的预定
            bookings = db.query(MeetingRoomBooking).filter(
                MeetingRoomBooking.room_id == room_id,
                MeetingRoomBooking.date == date,
                MeetingRoomBooking.status == "confirmed",
                MeetingRoomBooking.is_active == 1,
            ).all()

            booking_list = [
                {
                    "id": b.id,
                    "start_time": b.start_time,
                    "end_time": b.end_time,
                    "title": b.title,
                    "booked_by": b.booked_by,
                }
                for b in bookings
            ]

            # 生成 30 分钟时段 (08:00-20:00)
            slots = []
            current = datetime.strptime("08:00", "%H:%M")
            end = datetime.strptime("20:00", "%H:%M")

            while current < end:
                slot_start = current.strftime("%H:%M")
                slot_end = (current + timedelta(minutes=30)).strftime("%H:%M")

                # 检查是否与已有预定冲突
                is_booked = False
                booked_title = ""
                for b in bookings:
                    if b.start_time < slot_end and b.end_time > slot_start:
                        is_booked = True
                        booked_title = b.title
                        break

                slots.append({
                    "start": slot_start,
                    "end": slot_end,
                    "available": not is_booked,
                    "booked_title": booked_title if is_booked else "",
                })

                current += timedelta(minutes=30)

            return {
                "status": "success",
                "data": {
                    "room_id": room_id,
                    "room_name": room.name,
                    "date": date,
                    "room_status": room.status,
                    "slots": slots,
                    "bookings": booking_list,
                },
            }
        finally:
            db.close()
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"查询可用时段失败: {str(e)}")


@router.post("/{room_id}/book", summary="预定会议室")
async def book_room(room_id: int, request: BookingRequest):
    """预定会议室"""
    try:
        _seed_rooms()
        db = _get_db()
        try:
            from db.models import MeetingRoom, MeetingRoomBooking

            room = db.query(MeetingRoom).filter(
                MeetingRoom.id == room_id, MeetingRoom.is_active == 1,
            ).first()
            if not room:
                raise HTTPException(status_code=404, detail="会议室不存在")
            if room.status != "available":
                raise HTTPException(status_code=400, detail=f"会议室当前状态: {room.status}")

            # 验证时间
            try:
                start_dt = datetime.strptime(request.start_time, "%H:%M")
                end_dt = datetime.strptime(request.end_time, "%H:%M")
            except ValueError:
                raise HTTPException(status_code=400, detail="时间格式错误，应为 HH:MM")

            if end_dt <= start_dt:
                raise HTTPException(status_code=400, detail="结束时间必须晚于开始时间")

            # 时间范围检查 (08:00-20:00)
            if start_dt < datetime.strptime("08:00", "%H:%M") or \
               end_dt > datetime.strptime("20:00", "%H:%M"):
                raise HTTPException(status_code=400, detail="预定时间需在 08:00-20:00 之间")

            # 检查冲突
            conflict = db.query(MeetingRoomBooking).filter(
                MeetingRoomBooking.room_id == room_id,
                MeetingRoomBooking.date == request.date,
                MeetingRoomBooking.status == "confirmed",
                MeetingRoomBooking.is_active == 1,
                MeetingRoomBooking.start_time < request.end_time,
                MeetingRoomBooking.end_time > request.start_time,
            ).first()

            if conflict:
                raise HTTPException(
                    status_code=409,
                    detail=f"该时段已被预定: {conflict.title} ({conflict.start_time}-{conflict.end_time})",
                )

            # 创建预定
            booking = MeetingRoomBooking(
                room_id=room_id,
                date=request.date,
                start_time=request.start_time,
                end_time=request.end_time,
                title=request.title,
                description=request.description or "",
                booked_by=request.booked_by or "",
                status="confirmed",
            )
            db.add(booking)
            db.commit()
            db.refresh(booking)

            return {
                "status": "success",
                "message": f"会议室 {room.name} 预定成功！",
                "data": {
                    "booking_id": booking.id,
                    "room_name": room.name,
                    "date": booking.date,
                    "start_time": booking.start_time,
                    "end_time": booking.end_time,
                    "title": booking.title,
                },
            }
        finally:
            db.close()
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"预定失败: {str(e)}")


@router.delete("/bookings/{booking_id}", summary="取消预定")
async def cancel_booking(booking_id: int):
    """取消会议室预定（软删除）"""
    try:
        db = _get_db()
        try:
            from db.models import MeetingRoomBooking

            booking = db.query(MeetingRoomBooking).filter(
                MeetingRoomBooking.id == booking_id,
                MeetingRoomBooking.is_active == 1,
            ).first()
            if not booking:
                raise HTTPException(status_code=404, detail="预定记录不存在")

            booking.status = "cancelled"
            booking.is_active = 0
            db.commit()

            return {
                "status": "success",
                "message": "预定已取消",
            }
        finally:
            db.close()
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"取消失败: {str(e)}")


@router.get("/bookings/list", summary="获取预定列表")
async def list_bookings(
    date: Optional[str] = Query(default=None, description="日期 YYYY-MM-DD，默认为今天"),
    room_id: Optional[int] = Query(default=None, description="会议室 ID"),
):
    """获取会议室预定列表"""
    try:
        if not date:
            date = datetime.now().strftime("%Y-%m-%d")

        _seed_rooms()
        db = _get_db()
        try:
            from db.models import MeetingRoomBooking

            query = db.query(MeetingRoomBooking).filter(
                MeetingRoomBooking.is_active == 1,
                MeetingRoomBooking.status == "confirmed",
            )
            if date:
                query = query.filter(MeetingRoomBooking.date == date)
            if room_id:
                query = query.filter(MeetingRoomBooking.room_id == room_id)

            bookings = query.order_by(
                MeetingRoomBooking.date,
                MeetingRoomBooking.start_time,
            ).all()

            return {
                "status": "success",
                "data": [
                    {
                        "id": b.id,
                        "room_id": b.room_id,
                        "date": b.date,
                        "start_time": b.start_time,
                        "end_time": b.end_time,
                        "title": b.title,
                        "description": b.description,
                        "booked_by": b.booked_by,
                        "status": b.status,
                    }
                    for b in bookings
                ],
            }
        finally:
            db.close()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取预定列表失败: {str(e)}")

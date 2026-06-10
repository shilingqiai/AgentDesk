"""
数据库模型 — 企业员工AI服务台

包含知识库、工单、文档反馈、人工审核等模型。
"""
from sqlalchemy import Column, Integer, String, DateTime, Text, JSON, Float, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from datetime import datetime

Base = declarative_base()


class KnowledgeDocument(Base):
    """知识库文档"""
    __tablename__ = 'knowledge_documents'
    id = Column(Integer, primary_key=True)
    content = Column(Text, nullable=False)
    category = Column(String, nullable=False)
    keywords = Column(JSON, nullable=True)
    embedding = Column(JSON, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    is_active = Column(Integer, default=1)


class Ticket(Base):
    """
    工单模型 — 支持 IT故障 / 请假 / 报销 / 行政 等多种类型

    ticket_type 枚举:
        it_fault  — IT故障报修（网络/硬件/系统）
        leave     — 请假申请（年假/病假/事假）
        expense   — 报销申请（差旅/办公/餐费）
        admin     — 行政服务（会议室/快递/资产）
    """
    __tablename__ = 'tickets'

    id = Column(Integer, primary_key=True)
    ticket_number = Column(String(30), unique=True, nullable=False, index=True)
    ticket_type = Column(String(20), nullable=False, default="it_fault",
                         comment="it_fault | leave | expense | admin")
    title = Column(String(200), nullable=False)
    description = Column(Text, nullable=False)
    category = Column(String(50), nullable=False, default="其他")
    priority = Column(String(10), nullable=False, default="P2",
                      comment="P0=紧急 P1=高 P2=中 P3=低")
    status = Column(String(20), nullable=False, default="created",
                    comment="created→assigned→processing→resolved→closed")
    requester_id = Column(String(50), nullable=True, default="")
    requester_name = Column(String(50), nullable=True, default="")
    assigned_to = Column(String(50), nullable=True, default="")
    trace_id = Column(String(50), nullable=True, default="")
    payload = Column(JSON, nullable=True, comment="扩展字段: 请假天数/报销金额/会议室时间等")
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    is_active = Column(Integer, default=1)


class DocumentFeedback(Base):
    """知识库文档反馈（用户点赞/踩）"""
    __tablename__ = 'document_feedback'

    id = Column(Integer, primary_key=True)
    doc_id = Column(Integer, nullable=False, index=True, comment="关联 knowledge_documents.id")
    is_helpful = Column(Integer, nullable=False, default=1, comment="1=有帮助 0=无帮助")
    user_id = Column(String(50), nullable=True, default="")
    comment = Column(Text, nullable=True, default="")
    created_at = Column(DateTime, default=datetime.utcnow)


class HumanReviewRecord(Base):
    """人工审核记录 — 持久化 Human-in-the-Loop 审核队列"""
    __tablename__ = 'human_review_records'

    id = Column(Integer, primary_key=True)
    request_id = Column(String(50), unique=True, nullable=False, index=True)
    thread_id = Column(String(50), nullable=False, default="")
    intent = Column(String(50), nullable=False, default="")
    urgency = Column(String(20), nullable=False, default="medium")
    action_type = Column(String(30), nullable=False, default="")
    summary = Column(Text, nullable=True, default="")
    details = Column(JSON, nullable=True)
    status = Column(String(20), nullable=False, default="pending",
                    comment="pending→approved→rejected→timeout")
    decision = Column(String(20), nullable=True, default="")
    reviewer = Column(String(50), nullable=True, default="")
    comment = Column(Text, nullable=True, default="")
    created_at = Column(DateTime, default=datetime.utcnow)
    decided_at = Column(DateTime, nullable=True)


class MeetingRoom(Base):
    """
    会议室模型

    记录企业内可预定的会议室资源。
    """
    __tablename__ = 'meeting_rooms'

    id = Column(Integer, primary_key=True)
    name = Column(String(50), nullable=False, unique=True, comment="会议室名称，如 A101")
    capacity = Column(Integer, nullable=False, default=10, comment="容纳人数")
    location = Column(String(100), nullable=False, default="", comment="楼层/区域")
    amenities = Column(JSON, nullable=True, comment="设备: ['投影仪','白板','视频会议']")
    description = Column(Text, nullable=True, default="")
    status = Column(String(20), nullable=False, default="available",
                    comment="available | maintenance | closed")
    is_active = Column(Integer, default=1)
    created_at = Column(DateTime, default=datetime.utcnow)


class MeetingRoomBooking(Base):
    """
    会议室预定记录

    记录每次会议室预定的详细信息。
    """
    __tablename__ = 'meeting_room_bookings'

    id = Column(Integer, primary_key=True)
    room_id = Column(Integer, nullable=False, index=True, comment="关联 meeting_rooms.id")
    date = Column(String(10), nullable=False, comment="预定日期 YYYY-MM-DD")
    start_time = Column(String(5), nullable=False, comment="开始时间 HH:MM")
    end_time = Column(String(5), nullable=False, comment="结束时间 HH:MM")
    booked_by = Column(String(50), nullable=False, default="", comment="预定人")
    title = Column(String(200), nullable=False, default="", comment="会议主题")
    description = Column(Text, nullable=True, default="")
    status = Column(String(20), nullable=False, default="confirmed",
                    comment="confirmed | cancelled")
    created_at = Column(DateTime, default=datetime.utcnow)
    is_active = Column(Integer, default=1)


class InventoryItem(Base):
    """
    库存物品模型 (v7 新增)

    管理企业办公设备和物品库存。
    DynamicActionAgent 通过此表查询库存状态，支持模糊搜索。
    """
    __tablename__ = 'inventory_items'

    id = Column(Integer, primary_key=True)
    item_name = Column(String(100), nullable=False, comment="物品名称（含型号）")
    category = Column(String(50), nullable=False, default="电子设备",
                      comment="分类: 电子设备 | 外设 | 办公家具 | 耗材")
    stock = Column(Integer, nullable=False, default=0, comment="当前库存数")
    min_threshold = Column(Integer, nullable=False, default=2, comment="最低库存阈值")
    unit_price = Column(Integer, nullable=True, default=0, comment="单价(元)")
    supplier = Column(String(100), nullable=True, default="", comment="供应商")
    description = Column(Text, nullable=True, default="")
    is_active = Column(Integer, default=1)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

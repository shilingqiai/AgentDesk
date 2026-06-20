"""
数据库模型 — 企业员工AI服务台

包含知识库、工单、文档反馈、人工审核等模型。
"""
from sqlalchemy import Column, Integer, String, DateTime, Text, JSON, Float, ForeignKey
from sqlalchemy.orm import declarative_base
from datetime import datetime, timezone

_utcnow = lambda: datetime.now(timezone.utc)

Base = declarative_base()


class KnowledgeDocument(Base):
    """知识库文档"""
    __tablename__ = 'knowledge_documents'
    id = Column(Integer, primary_key=True)
    content = Column(Text, nullable=False)
    category = Column(String, nullable=False)
    keywords = Column(JSON, nullable=True)
    embedding = Column(JSON, nullable=True)
    created_at = Column(DateTime, default=_utcnow)
    updated_at = Column(DateTime, default=_utcnow, onupdate=_utcnow)
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
                    comment="created→pending_approval→approved→processing→completed | rejected")
    requester_id = Column(String(50), nullable=True, default="")
    requester_name = Column(String(50), nullable=True, default="")
    assigned_to = Column(String(50), nullable=True, default="")
    trace_id = Column(String(50), nullable=True, default="")
    current_approver = Column(String(64), nullable=True, default="",
                             comment="当前审批人姓名（冗余，方便 tickets 页直接筛选）")
    approver_chain = Column(JSON, nullable=True,
                           comment="审批链快照: ['王经理','李HR']")
    history = Column(JSON, nullable=True,
                    comment="操作时间线: [{action, by, time, detail}]")
    payload = Column(JSON, nullable=True, comment="扩展字段: 请假天数/报销金额/会议室时间等")
    created_at = Column(DateTime, default=_utcnow)
    updated_at = Column(DateTime, default=_utcnow, onupdate=_utcnow)
    is_active = Column(Integer, default=1)


class DocumentFeedback(Base):
    """知识库文档反馈（用户点赞/踩）"""
    __tablename__ = 'document_feedback'

    id = Column(Integer, primary_key=True)
    doc_id = Column(Integer, nullable=False, index=True, comment="关联 knowledge_documents.id")
    is_helpful = Column(Integer, nullable=False, default=1, comment="1=有帮助 0=无帮助")
    user_id = Column(String(50), nullable=True, default="")
    comment = Column(Text, nullable=True, default="")
    created_at = Column(DateTime, default=_utcnow)


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
    created_at = Column(DateTime, default=_utcnow)
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
    created_at = Column(DateTime, default=_utcnow)


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
    created_at = Column(DateTime, default=_utcnow)
    is_active = Column(Integer, default=1)


class ApprovalWorkflow(Base):
    """
    审批流模型 — 企业审批流程引擎

    每个需要审批的工单生成一条审批流。
    审批链由确定性规则确定（不经过 LLM），保证合规可审计。
    """
    __tablename__ = 'approval_workflows'

    id = Column(Integer, primary_key=True)
    ticket_id = Column(Integer, ForeignKey('tickets.id'), nullable=False, index=True)
    workflow_type = Column(String(32), nullable=False, default="leave",
                         comment="leave | expense | procurement")
    current_step = Column(Integer, nullable=False, default=0, comment="当前审批节点序号")
    total_steps = Column(Integer, nullable=False, default=1)
    status = Column(String(16), nullable=False, default="pending",
                  comment="pending → approved → rejected")
    created_at = Column(DateTime, default=_utcnow)
    updated_at = Column(DateTime, default=_utcnow, onupdate=_utcnow)


class ApprovalStep(Base):
    """
    审批节点 — 审批流中的每一步

    每个节点对应一个审批人，按 step_order 顺序执行。
    """
    __tablename__ = 'approval_steps'

    id = Column(Integer, primary_key=True)
    workflow_id = Column(Integer, ForeignKey('approval_workflows.id'), nullable=False, index=True)
    step_order = Column(Integer, nullable=False, comment="审批顺序: 1, 2, 3...")
    approver = Column(String(64), nullable=False, comment="审批人")
    approver_role = Column(String(32), nullable=False, default="",
                         comment="审批角色: department_manager | hr | finance | vp")
    status = Column(String(16), nullable=False, default="pending",
                  comment="pending → approved → rejected")
    comment = Column(String(500), nullable=True, default="")
    created_at = Column(DateTime, default=_utcnow)
    decided_at = Column(DateTime, nullable=True)


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
    created_at = Column(DateTime, default=_utcnow)
    updated_at = Column(DateTime, default=_utcnow, onupdate=_utcnow)


class AuditLog(Base):
    """
    审计日志模型 (v12 新增) — Event-Driven Architecture

    记录所有系统事件，用于合规审计和问题追溯。
    由 AuditHandler 通过 EventBus 订阅自动写入。
    """
    __tablename__ = 'audit_logs'

    id = Column(Integer, primary_key=True)
    event = Column(String(50), nullable=False, index=True,
                   comment="事件类型: ticket.created / approval.completed 等")
    ticket_id = Column(Integer, nullable=True, index=True,
                       comment="关联工单 ID")
    ticket_number = Column(String(30), nullable=True, default="",
                           comment="关联工单号")
    operator = Column(String(50), nullable=True, default="system",
                      comment="操作人")
    data = Column(JSON, nullable=True, comment="事件附加数据")
    created_at = Column(DateTime, default=_utcnow)

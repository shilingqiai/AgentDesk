"""
测试: 事件总线 (EventBus + Event Handlers)

覆盖:
- subscribe / emit 基本流程
- 多个 handler 订阅同一事件
- handler 异常隔离
- 事件数据正确传递
- EventType 常量
"""

import pytest
from services.event_bus import EventBus, EventType


class TestEventBusBasic:
    """EventBus 基础功能测试"""

    def setup_method(self):
        """每个测试前清空订阅"""
        EventBus.clear()

    @pytest.mark.asyncio
    async def test_subscribe_and_emit(self):
        """基本订阅/发射流程"""
        received = []

        async def handler(event_type, **data):
            received.append(data)

        EventBus.subscribe("test.event", handler)
        await EventBus.emit("test.event", key="value", num=42)

        assert len(received) == 1
        assert received[0]["key"] == "value"
        assert received[0]["num"] == 42

    @pytest.mark.asyncio
    async def test_multiple_handlers(self):
        """多个 handler 订阅同一事件"""
        results = []

        async def h1(event_type, **data):
            results.append("h1")

        async def h2(event_type, **data):
            results.append("h2")

        async def h3(event_type, **data):
            results.append("h3")

        EventBus.subscribe("test.multi", h1)
        EventBus.subscribe("test.multi", h2)
        EventBus.subscribe("test.multi", h3)
        await EventBus.emit("test.multi")

        assert len(results) == 3
        assert "h1" in results
        assert "h2" in results
        assert "h3" in results

    @pytest.mark.asyncio
    async def test_handler_exception_isolation(self):
        """一个 handler 崩溃不影响其他 handler"""
        results = []

        async def good_handler(event_type, **data):
            results.append("good")

        async def bad_handler(event_type, **data):
            raise RuntimeError("handler crash")

        EventBus.subscribe("test.crash", good_handler)
        EventBus.subscribe("test.crash", bad_handler)
        EventBus.subscribe("test.crash", good_handler)

        # 不应抛出异常
        await EventBus.emit("test.crash")

        assert results == ["good", "good"]

    @pytest.mark.asyncio
    async def test_emit_no_subscribers(self):
        """无订阅者时 emit 不报错"""
        await EventBus.emit("test.nonexistent")  # 不抛异常

    @pytest.mark.asyncio
    async def test_event_data_passed_correctly(self):
        """事件数据完整传递"""
        captured = {}

        async def handler(event_type, **data):
            captured["type"] = event_type
            captured["data"] = data

        EventBus.subscribe(EventType.TICKET_CREATED, handler)
        await EventBus.emit(
            EventType.TICKET_CREATED,
            ticket_id=42,
            ticket_number="TK-20260619-ABC123",
            ticket_type="leave",
        )

        assert captured["type"] == "ticket.created"
        assert captured["data"]["ticket_id"] == 42
        assert captured["data"]["ticket_number"] == "TK-20260619-ABC123"
        assert captured["data"]["ticket_type"] == "leave"

    @pytest.mark.asyncio
    async def test_unsubscribe(self):
        """取消订阅后不再接收事件"""
        results = []

        async def handler(event_type, **data):
            results.append("x")

        EventBus.subscribe("test.unsub", handler)
        EventBus.unsubscribe("test.unsub", handler)
        await EventBus.emit("test.unsub")

        assert len(results) == 0

    def test_subscriber_count(self):
        """订阅者计数"""
        EventBus.clear()

        async def h(event_type, **data):
            pass

        assert EventBus.subscriber_count() == 0
        EventBus.subscribe("e1", h)
        assert EventBus.subscriber_count("e1") == 1
        EventBus.subscribe("e2", h)
        assert EventBus.subscriber_count() == 2

    @pytest.mark.asyncio
    async def test_event_type_constants(self):
        """EventType 常量完整性"""
        assert EventType.TICKET_CREATED == "ticket.created"
        assert EventType.TICKET_STATUS_CHANGED == "ticket.status_changed"
        assert EventType.APPROVAL_STEP_APPROVED == "approval.step_approved"
        assert EventType.APPROVAL_STEP_REJECTED == "approval.step_rejected"
        assert EventType.APPROVAL_COMPLETED == "approval.completed"
        assert EventType.APPROVAL_REJECTED == "approval.rejected"
        assert EventType.SLA_BREACHED == "sla.breached"


class TestDashboardHandler:
    """DashboardHandler 计数与事件记录"""

    def setup_method(self):
        EventBus.clear()
        from services.event_handlers import DashboardHandler
        # 重置计数器
        DashboardHandler._counters = {
            "tickets_created_today": 0,
            "tickets_approved_today": 0,
            "tickets_rejected_today": 0,
        }
        DashboardHandler._recent_events = []

    @pytest.mark.asyncio
    async def test_counts_ticket_created(self):
        from services.event_handlers import DashboardHandler as DH

        EventBus.subscribe(EventType.TICKET_CREATED, DH.on_ticket_created)
        await EventBus.emit(EventType.TICKET_CREATED,
                            ticket_number="TK-001", ticket_type="leave",
                            requester_name="张三")

        stats = DH.get_stats()
        assert stats["counters"]["tickets_created_today"] == 1
        assert len(stats["recent_events"]) == 1
        assert stats["recent_events"][0]["ticket"] == "TK-001"

    @pytest.mark.asyncio
    async def test_counts_status_changed_to_approved(self):
        from services.event_handlers import DashboardHandler as DH

        EventBus.subscribe(EventType.TICKET_STATUS_CHANGED, DH.on_status_changed)
        await EventBus.emit(EventType.TICKET_STATUS_CHANGED,
                            ticket_number="TK-001",
                            from_status="pending_approval", to_status="approved")

        stats = DH.get_stats()
        assert stats["counters"]["tickets_approved_today"] == 1

    @pytest.mark.asyncio
    async def test_counts_status_changed_to_rejected(self):
        from services.event_handlers import DashboardHandler as DH

        EventBus.subscribe(EventType.TICKET_STATUS_CHANGED, DH.on_status_changed)
        await EventBus.emit(EventType.TICKET_STATUS_CHANGED,
                            ticket_number="TK-001",
                            from_status="pending_approval", to_status="rejected")

        stats = DH.get_stats()
        assert stats["counters"]["tickets_rejected_today"] == 1

    @pytest.mark.asyncio
    async def test_recent_events_capped(self):
        from services.event_handlers import DashboardHandler as DH

        EventBus.subscribe(EventType.TICKET_CREATED, DH.on_ticket_created)
        for i in range(25):
            await EventBus.emit(EventType.TICKET_CREATED,
                                ticket_number=f"TK-{i:03d}",
                                ticket_type="leave", requester_name="张三")

        stats = DH.get_stats()
        # _recent_events 存储上限 20，get_stats() 对外返回最近 10 条
        assert len(DH._recent_events) == 20
        assert len(stats["recent_events"]) == 10

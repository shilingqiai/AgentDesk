"""
测试: SLA Engine 超时检测与自动升级

覆盖:
- SLA 基础计算 (deadline, check_status, escalation levels)
- 规则引擎 (SLARule, SLA_RULES)
- 超时检测 (get_breached)
- 格式化和通知消息
- 自动升级动作 (escalate priority)
- SLA 规则配置完整性
- SLAScheduler 基础生命周期
"""

import pytest
import asyncio
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch, PropertyMock

from services.sla_engine import (
    SLAEngine,
    SLAStatus,
    SLASummary,
    SLARule,
    SLA_HOURS,
    SLA_RULES,
    ESCALATION_LEVELS,
    BUSINESS_START,
    BUSINESS_END,
    _add_business_hours,
    _elapsed_business_hours,
)


# ═══════════════════════════════════════════════════════
# 测试: SLA 基础计算
# ═══════════════════════════════════════════════════════

class TestSLABasicCalculation:
    """SLA 基础计算功能"""

    def test_sla_hours_config(self):
        """SLA 优先级→时长配置"""
        assert SLA_HOURS["P0"] == 2
        assert SLA_HOURS["P1"] == 2
        assert SLA_HOURS["P2"] == 4
        assert SLA_HOURS["P3"] == 8

    def test_escalation_levels(self):
        """升级阶梯配置"""
        assert len(ESCALATION_LEVELS) == 3
        assert ESCALATION_LEVELS[0][0] == 0.5
        assert ESCALATION_LEVELS[1][0] == 0.75
        assert ESCALATION_LEVELS[2][0] == 1.0

    def test_add_business_hours_basic(self):
        """工作小时累加"""
        # 周一 10:00 开始，加 4 工作小时
        from datetime import datetime, timedelta
        start = datetime(2026, 6, 15, 10, 0, 0)  # Monday 10:00
        result = _add_business_hours(start, 4)
        # 10:00 + 4 工作小时 = 14:00（在同一天）
        assert result.hour == 14
        assert result.day == 15

    def test_add_business_hours_overnight(self):
        """跨天工作小时累加"""
        # 21:00 开始，加 3 小时 → 第二天 11:00
        start = datetime(2026, 6, 15, 21, 0, 0)  # 21:00
        result = _add_business_hours(start, 3)
        # 21:00-22:00 = 1h, then next day 9:00-11:00 = 2h
        assert result.hour == 11
        assert result.day == 16

    def test_get_deadline_p2(self):
        """P2 工单 deadline 计算"""
        start = datetime(2026, 6, 15, 10, 0, 0)
        deadline = SLAEngine.get_deadline("P2", start)
        # P2 = 4h, 10:00 + 4 = 14:00
        assert deadline.hour == 14
        assert deadline.day == 15

    def test_get_deadline_p0(self):
        """P0 工单 deadline 计算"""
        start = datetime(2026, 6, 15, 10, 0, 0)
        deadline = SLAEngine.get_deadline("P0", start)
        # P0 = 2h, 10:00 + 2 = 12:00
        assert deadline.hour == 12

    def test_check_status_not_breached(self):
        """未超时工单检测"""
        now = datetime.utcnow()
        # 刚刚创建的 P3 工单（8h SLA）
        ticket = {
            "id": 1, "ticket_number": "TK-001",
            "priority": "P3", "created_at": now,
        }
        status = SLAEngine.check_status(ticket)
        assert status.is_breached is False
        assert status.escalation_level == 0
        assert status.sla_hours == 8

    def test_check_status_breached(self):
        """超时工单检测"""
        # 10 小时前创建的 P2 工单（4h SLA）→ 已超时
        created = datetime.utcnow() - timedelta(hours=10)
        ticket = {
            "id": 2, "ticket_number": "TK-002",
            "priority": "P2", "created_at": created,
        }
        status = SLAEngine.check_status(ticket)
        assert status.is_breached is True
        assert status.escalation_level == 3  # > 100%

    def test_check_status_escalation_levels(self):
        """阶梯升级检测"""
        now = datetime.utcnow()

        # 刚创建 P2 (4h SLA) → 0%
        status = SLAEngine.check_status({
            "id": 1, "ticket_number": "TK-001",
            "priority": "P2", "created_at": now,
        })
        assert status.escalation_level == 0

        # 3 小时前创建 (75%) → level 1
        created = now - timedelta(hours=3)
        status = SLAEngine.check_status({
            "id": 2, "ticket_number": "TK-002",
            "priority": "P2", "created_at": created,
        })
        assert status.escalation_level >= 1

    def test_check_status_str_datetime(self):
        """字符串 datetime 输入"""
        created = (datetime.utcnow() - timedelta(hours=6)).isoformat()
        ticket = {
            "id": 3, "ticket_number": "TK-003",
            "priority": "P2", "created_at": created,
        }
        status = SLAEngine.check_status(ticket)
        assert status.priority == "P2"

    def test_check_status_default_priority(self):
        """默认优先级 P3（未指定时）"""
        ticket = {
            "id": 4, "ticket_number": "TK-004",
            "created_at": datetime.utcnow(),
        }
        status = SLAEngine.check_status(ticket)
        assert status.sla_hours == 8  # P3 = 8h
        assert status.priority == "P3"


# ═══════════════════════════════════════════════════════
# 测试: SLA 规则定义
# ═══════════════════════════════════════════════════════

class TestSLARules:
    """SLA 规则配置"""

    def test_all_rules_have_required_fields(self):
        """所有规则都有必需字段"""
        required = {"key", "label", "duration_h", "action"}
        for key, rule in SLA_RULES.items():
            for field in required:
                assert getattr(rule, field) is not None, \
                    f"Rule '{key}' missing field '{field}'"

    def test_rules_have_valid_actions(self):
        """规则动作在合法范围内"""
        valid_actions = {"notify_admin", "escalate", "remind_approver", "auto_cancel"}
        for key, rule in SLA_RULES.items():
            assert rule.action in valid_actions, \
                f"Rule '{key}' has invalid action '{rule.action}'"

    def test_get_rules_returns_list(self):
        """get_rules() 返回规则列表"""
        rules = SLAEngine.get_rules()
        assert isinstance(rules, list)
        assert len(rules) >= 4  # ticket_response, ticket_resolution, approval_step, meeting_confirm
        for r in rules:
            assert "key" in r
            assert "label" in r
            assert "duration_h" in r
            assert "action" in r


# ═══════════════════════════════════════════════════════
# 测试: 通知消息
# ═══════════════════════════════════════════════════════

class TestSLAMessages:
    """SLA 通知消息格式化"""

    def test_format_escalation_message_level_0(self):
        """level 0 = 无通知"""
        status = SLAStatus(
            ticket_id=1, ticket_number="TK-001", priority="P3",
            created_at=datetime.utcnow(), sla_hours=8,
            deadline=datetime.utcnow() + timedelta(hours=8),
            elapsed_hours=0, remaining_hours=8,
            escalation_level=0, is_breached=False,
        )
        msg = SLAEngine.format_escalation_message(status)
        assert msg == ""

    def test_format_escalation_message_level_2(self):
        """level 2 = 通知主管"""
        status = SLAStatus(
            ticket_id=2, ticket_number="TK-002", priority="P2",
            created_at=datetime.utcnow() - timedelta(hours=3.5),
            sla_hours=4,
            deadline=datetime.utcnow() - timedelta(hours=0.5),
            elapsed_hours=3.5, remaining_hours=0.5,
            escalation_level=2, is_breached=False,
        )
        msg = SLAEngine.format_escalation_message(status)
        assert "TK-002" in msg
        assert "通知主管" in msg or "SLA" in msg

    def test_format_escalation_message_level_3(self):
        """level 3 = 紧急"""
        status = SLAStatus(
            ticket_id=3, ticket_number="TK-003", priority="P1",
            created_at=datetime.utcnow() - timedelta(hours=5),
            sla_hours=2,
            deadline=datetime.utcnow() - timedelta(hours=3),
            elapsed_hours=5, remaining_hours=0,
            escalation_level=3, is_breached=True,
        )
        msg = SLAEngine.format_escalation_message(status)
        assert "TK-003" in msg
        assert "紧急" in msg or "SLA" in msg


# ═══════════════════════════════════════════════════════
# 测试: 超时查询
# ═══════════════════════════════════════════════════════

class TestGetBreached:
    """get_breached 查询"""

    def test_get_breached_empty_db(self):
        """空数据库 → 无超时工单"""
        from db.db_router import DatabaseRouter
        db = DatabaseRouter("sqlite:///:memory:")
        try:
            session = db.session_manager.Session()
            breached = SLAEngine.get_breached(session)
            assert len(breached) == 0
        finally:
            db.close()

    def test_get_breached_no_active_tickets(self):
        """数据库有已完成的工单，但不计入超时"""
        from db.db_router import DatabaseRouter
        from db.models import Ticket
        db = DatabaseRouter("sqlite:///:memory:")
        try:
            session = db.session_manager.Session()
            # 创建已完成的工单
            t = Ticket(
                ticket_number="TK-DONE-001",
                ticket_type="it_fault",
                title="已完成的工单",
                description="已完成",
                status="completed",
                priority="P2",
                created_at=datetime.utcnow() - timedelta(hours=48),
                is_active=1,
            )
            session.add(t)
            session.commit()

            breached = SLAEngine.get_breached(session)
            assert len(breached) == 0  # completed 不在 active_statuses() 中
        finally:
            db.close()

    def test_get_breached_with_breached_ticket(self):
        """有超时工单"""
        from db.db_router import DatabaseRouter
        from db.models import Ticket
        db = DatabaseRouter("sqlite:///:memory:")
        try:
            session = db.session_manager.Session()
            t = Ticket(
                ticket_number="TK-BREACH-001",
                ticket_type="it_fault",
                title="超时工单",
                description="已超时未处理",
                status="created",
                priority="P0",  # 2h SLA
                created_at=datetime.utcnow() - timedelta(hours=5),
                is_active=1,
            )
            session.add(t)
            session.commit()

            breached = SLAEngine.get_breached(session)
            assert len(breached) >= 1
            assert breached[0].ticket_number == "TK-BREACH-001"
            assert breached[0].is_breached is True
        finally:
            db.close()


# ═══════════════════════════════════════════════════════
# 测试: 自动升级
# ═══════════════════════════════════════════════════════

class TestAutoEscalation:
    """自动升级逻辑"""

    def test_escalate_priority(self):
        """优先级提升: P3→P2, P2→P1, P1→P0, P0→P0"""
        assert SLAEngine._escalate_priority("P3") == "P2"
        assert SLAEngine._escalate_priority("P2") == "P1"
        assert SLAEngine._escalate_priority("P1") == "P0"
        assert SLAEngine._escalate_priority("P0") == "P0"  # 已最高

    def test_escalate_priority_unknown(self):
        """未知优先级默认 → P1"""
        result = SLAEngine._escalate_priority("unknown")
        assert result == "P1"

    @pytest.mark.asyncio
    async def test_escalate_ticket_priority_change(self):
        """单工单升级: 优先级实际改变"""
        from db.db_router import DatabaseRouter
        from db.models import Ticket
        db = DatabaseRouter("sqlite:///:memory:")
        try:
            session = db.session_manager.Session()
            t = Ticket(
                ticket_number="TK-ESC-001",
                ticket_type="it_fault",
                title="待升级工单",
                description="测试升级",
                status="created",
                priority="P2",
                is_active=1,
            )
            session.add(t)
            session.commit()

            result = await SLAEngine.escalate_ticket(
                t.id, reason="测试升级", db_session=session
            )
            assert result["success"] is True
            assert result["old_priority"] == "P2"
            assert result["new_priority"] == "P1"
        finally:
            db.close()

    @pytest.mark.asyncio
    async def test_escalate_ticket_not_found(self):
        """升级不存在的工单"""
        from db.db_router import DatabaseRouter
        db = DatabaseRouter("sqlite:///:memory:")
        try:
            session = db.session_manager.Session()
            result = await SLAEngine.escalate_ticket(
                99999, reason="测试", db_session=session
            )
            assert result["success"] is False
            assert "error" in result
        finally:
            db.close()

    @pytest.mark.asyncio
    async def test_escalate_ticket_history_appended(self):
        """升级后 history 追加记录"""
        from db.db_router import DatabaseRouter
        from db.models import Ticket
        db = DatabaseRouter("sqlite:///:memory:")
        try:
            session = db.session_manager.Session()
            t = Ticket(
                ticket_number="TK-HIST-001",
                ticket_type="it_fault",
                title="历史记录测试",
                description="测试",
                status="created",
                priority="P3",
                history=[{"action": "created", "by": "张三", "time": "..."}],
                is_active=1,
            )
            session.add(t)
            session.commit()

            result = await SLAEngine.escalate_ticket(
                t.id, reason="SLA 超时升级", db_session=session
            )
            assert result["success"] is True

            # 重新加载验证 history
            t2 = session.query(Ticket).filter(Ticket.id == t.id).first()
            assert len(t2.history) >= 2
            last = t2.history[-1]
            assert last["action"] == "sla_escalate"
            assert "SLA" in last["detail"]
        finally:
            db.close()


# ═══════════════════════════════════════════════════════
# 测试: SLAScheduler
# ═══════════════════════════════════════════════════════

class TestSLAScheduler:
    """SLA 调度器"""

    @pytest.mark.asyncio
    async def test_scheduler_start_stop(self):
        """调度器启动和停止"""
        from services.sla_scheduler import SLAScheduler

        # 用短间隔启动
        await SLAScheduler.start(interval=600)  # 10 分钟，不会在测试中触发
        assert SLAScheduler.is_running() is True

        await SLAScheduler.stop()
        assert SLAScheduler.is_running() is False

    @pytest.mark.asyncio
    async def test_scheduler_double_start(self):
        """重复启动不创建第二个任务"""
        from services.sla_scheduler import SLAScheduler

        await SLAScheduler.start(interval=600)
        task_count = 1 if SLAScheduler.is_running() else 0
        await SLAScheduler.start(interval=600)  # 应该跳过
        assert SLAScheduler.is_running() is True
        await SLAScheduler.stop()

    @pytest.mark.asyncio
    async def test_run_once_returns_summary(self):
        """手动执行一次检测返回 summary（结构验证）"""
        from services.sla_scheduler import SLAScheduler

        summary = await SLAScheduler.run_once()
        assert "breached_count" in summary
        assert "active_sla_count" in summary
        assert "rules" in summary
        assert isinstance(summary["breached_tickets"], list)
        assert isinstance(summary["approval_deadlines"], list)

    @pytest.mark.asyncio
    async def test_run_once_empty_db(self):
        """空数据库手动检测"""
        from services.sla_scheduler import SLAScheduler

        summary = await SLAScheduler.run_once()
        assert summary["breached_count"] == 0
        assert "rules" in summary


# ═══════════════════════════════════════════════════════
# 测试: SLASummary
# ═══════════════════════════════════════════════════════

class TestSLASummary:
    """SLA Summary 数据结构"""

    def test_empty_summary(self):
        """空 summary"""
        s = SLASummary()
        assert s.breached_count == 0
        assert s.warning_count == 0
        assert s.active_sla_count == 0
        assert s.breached_tickets == []
        assert s.approval_deadlines == []
        assert s.rules == []

    def test_summary_with_data(self):
        """带数据的 summary"""
        s = SLASummary(
            breached_count=3,
            warning_count=5,
            active_sla_count=10,
            breached_tickets=[{"id": 1, "ticket_number": "TK-001"}],
            approval_deadlines=[{"step_id": 1, "approver": "王经理"}],
            rules=[{"key": "test", "label": "测试"}],
        )
        assert s.breached_count == 3
        assert len(s.breached_tickets) == 1
        assert len(s.approval_deadlines) == 1


# ═══════════════════════════════════════════════════════
# 测试: 审批步骤 SLA
# ═══════════════════════════════════════════════════════

class TestApprovalStepSLA:
    """审批步骤 SLA 检测"""

    @pytest.mark.asyncio
    async def test_check_approval_steps_empty(self):
        """无审批步骤 → 空列表"""
        from db.db_router import DatabaseRouter
        db = DatabaseRouter("sqlite:///:memory:")
        try:
            session = db.session_manager.Session()
            results = await SLAEngine.check_approval_steps(session)
            assert results == []
        finally:
            db.close()

    @pytest.mark.asyncio
    async def test_check_approval_steps_with_pending(self):
        """有待审批步骤 — 检测超时"""
        from db.db_router import DatabaseRouter
        from db.models import Ticket, ApprovalWorkflow, ApprovalStep
        db = DatabaseRouter("sqlite:///:memory:")
        try:
            session = db.session_manager.Session()

            # 创建工单
            t = Ticket(
                ticket_number="TK-APPROVAL-SLA-001",
                ticket_type="leave",
                title="请假审批 SLA 测试",
                description="测试",
                status="pending_approval",
                priority="P2",
                is_active=1,
            )
            session.add(t)
            session.flush()

            # 创建审批流
            wf = ApprovalWorkflow(
                ticket_id=t.id,
                workflow_type="leave",
                current_step=0,
                total_steps=2,
                status="pending",
            )
            session.add(wf)
            session.flush()

            # 创建审批步骤（20h 前 → 约 11 工作小时 > 8h SLA）
            long_ago = datetime.utcnow() - timedelta(hours=20)
            step = ApprovalStep(
                workflow_id=wf.id,
                step_order=1,
                approver="王经理",
                approver_role="department_manager",
                status="pending",
                created_at=long_ago,
            )
            session.add(step)
            session.commit()

            summary = SLASummary()
            results = await SLAEngine.check_approval_steps(session, summary)
            # 应该至少找到 1 个审批步骤
            assert len(results) >= 1
            assert results[0]["approver"] == "王经理"
            # 10 小时前创建 → 超时（8h SLA）
            assert results[0]["is_breached"] is True
        finally:
            db.close()


# ═══════════════════════════════════════════════════════
# 测试: 边界情况
# ═══════════════════════════════════════════════════════

class TestSLAEdgeCases:
    """边界情况和异常处理"""

    def test_elapsed_business_hours_zero(self):
        """刚创建的工单 — 耗时 ≈ 0"""
        now = datetime.utcnow()
        elapsed = _elapsed_business_hours(now, now)
        assert elapsed == 0.0

    def test_elapsed_business_hours_positive(self):
        """正耗时"""
        start = datetime.utcnow() - timedelta(hours=3)
        elapsed = _elapsed_business_hours(start)
        assert elapsed > 0

    def test_sla_status_repr(self):
        """SLAStatus 对象创建"""
        s = SLAStatus(
            ticket_id=42, ticket_number="TK-042", priority="P2",
            created_at=datetime.utcnow(), sla_hours=4,
            deadline=datetime.utcnow() + timedelta(hours=4),
            elapsed_hours=1.5, remaining_hours=2.5,
            escalation_level=1, is_breached=False,
        )
        assert s.ticket_id == 42
        assert s.ticket_number == "TK-042"
        assert s.is_breached is False

    def test_business_hours_midnight(self):
        """午夜前后 SLA 计算"""
        # 22:00（工作结束）开始 + 1h → 第二天 9:00
        start = datetime(2026, 6, 15, 22, 0, 0)
        result = _add_business_hours(start, 1)
        assert result.hour == 9
        assert result.day == 16

    def test_business_hours_before_start(self):
        """工作时间开始前的计算"""
        # 8:00（工作开始前）+ 1h → 9:00（第一个工作小时从 9:00 开始累积）
        start = datetime(2026, 6, 15, 8, 0, 0)
        result = _add_business_hours(start, 1)
        assert result.hour == 9
        assert result.day == 15


# ═══════════════════════════════════════════════════════
# 测试: get_sla_summary API
# ═══════════════════════════════════════════════════════

class TestSLASummaryAPI:
    """get_sla_summary() API"""

    @pytest.mark.asyncio
    async def test_get_sla_summary_with_breached(self):
        """有超时工单时的 summary"""
        from db.db_router import DatabaseRouter
        from db.models import Ticket
        db = DatabaseRouter("sqlite:///:memory:")
        try:
            session = db.session_manager.Session()

            t = Ticket(
                ticket_number="TK-SUMMARY-001",
                ticket_type="it_fault",
                title="Summary 测试工单",
                description="测试",
                status="created",
                priority="P0",
                created_at=datetime.utcnow() - timedelta(hours=5),
                is_active=1,
            )
            session.add(t)
            session.commit()

            summary = await SLAEngine.get_sla_summary(session)
            assert summary["breached_count"] >= 1
            assert len(summary["breached_tickets"]) >= 1
            assert len(summary["rules"]) >= 4
        finally:
            db.close()

    @pytest.mark.asyncio
    async def test_get_sla_summary_all_fields(self):
        """返回所有必需字段"""
        summary = await SLAEngine.get_sla_summary()
        required_fields = [
            "breached_count", "warning_count", "active_sla_count",
            "breached_tickets", "approval_deadlines", "rules",
        ]
        for field in required_fields:
            assert field in summary, f"Missing field: {field}"

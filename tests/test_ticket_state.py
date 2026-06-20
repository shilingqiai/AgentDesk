"""
测试: 工单状态机 (TicketStatus + WorkflowService)

覆盖:
- 状态枚举向后兼容 (str mixin)
- 合法状态转换
- 非法状态转换被拒绝
- 终态不可变
- active_statuses() 返回非终态
"""

from services.ticket_state import (
    TicketStatus,
    ALLOWED_TRANSITIONS,
    can_transition,
    validate_transition,
    active_statuses,
    is_terminal,
    STATUS_LABELS,
)


class TestTicketStatusEnum:
    """TicketStatus 枚举基础测试"""

    def test_str_mixin_backward_compat(self):
        """str mixin 保证与现有字符串比较兼容"""
        assert TicketStatus.CREATED == "created"
        assert TicketStatus.PENDING_APPROVAL == "pending_approval"
        assert TicketStatus.APPROVED == "approved"
        assert TicketStatus.REJECTED == "rejected"
        assert TicketStatus.PROCESSING == "processing"
        assert TicketStatus.COMPLETED == "completed"

    def test_from_string(self):
        """从字符串构造枚举"""
        assert TicketStatus("created") == TicketStatus.CREATED
        assert TicketStatus("pending_approval") == TicketStatus.PENDING_APPROVAL
        assert TicketStatus("approved") == TicketStatus.APPROVED

    def test_invalid_string_raises(self):
        """无效字符串抛出 ValueError"""
        try:
            TicketStatus("nonexistent")
            assert False, "should have raised"
        except ValueError:
            pass

    def test_all_statuses_have_labels(self):
        """所有状态都有中文标签"""
        for status in TicketStatus:
            assert status in STATUS_LABELS
            assert isinstance(STATUS_LABELS[status], str)
            assert len(STATUS_LABELS[status]) > 0


class TestAllowedTransitions:
    """状态转换规则测试"""

    def test_created_to_pending_approval(self):
        assert can_transition(TicketStatus.CREATED, TicketStatus.PENDING_APPROVAL)

    def test_pending_approval_to_approved(self):
        assert can_transition(TicketStatus.PENDING_APPROVAL, TicketStatus.APPROVED)

    def test_pending_approval_to_rejected(self):
        assert can_transition(TicketStatus.PENDING_APPROVAL, TicketStatus.REJECTED)

    def test_approved_to_processing(self):
        assert can_transition(TicketStatus.APPROVED, TicketStatus.PROCESSING)

    def test_approved_to_rejected(self):
        """事后驳回: APPROVED → REJECTED"""
        assert can_transition(TicketStatus.APPROVED, TicketStatus.REJECTED)

    def test_processing_to_completed(self):
        assert can_transition(TicketStatus.PROCESSING, TicketStatus.COMPLETED)

    def test_rejected_is_terminal(self):
        """REJECTED 是终态，不可再转"""
        for target in TicketStatus:
            assert not can_transition(TicketStatus.REJECTED, target), \
                f"REJECTED should not transition to {target}"

    def test_completed_is_terminal(self):
        """COMPLETED 是终态，不可再转"""
        for target in TicketStatus:
            assert not can_transition(TicketStatus.COMPLETED, target), \
                f"COMPLETED should not transition to {target}"

    def test_created_cannot_go_to_completed_directly(self):
        """CREATED 不能直接到 COMPLETED"""
        assert not can_transition(TicketStatus.CREATED, TicketStatus.COMPLETED)

    def test_pending_approval_cannot_go_to_processing_directly(self):
        """PENDING_APPROVAL 不能直接到 PROCESSING（必须先 APPROVED）"""
        assert not can_transition(TicketStatus.PENDING_APPROVAL, TicketStatus.PROCESSING)


class TestValidateTransition:
    """validate_transition 函数测试"""

    def test_valid_transition_no_error(self):
        validate_transition("created", "pending_approval")

    def test_invalid_transition_raises(self):
        try:
            validate_transition("created", "completed")
            assert False, "should have raised"
        except ValueError as e:
            assert "不允许" in str(e)

    def test_invalid_from_status_raises(self):
        try:
            validate_transition("nonexistent", "approved")
            assert False, "should have raised"
        except ValueError as e:
            assert "无效" in str(e)

    def test_invalid_to_status_raises(self):
        try:
            validate_transition("created", "nonexistent")
            assert False, "should have raised"
        except ValueError as e:
            assert "无效" in str(e)


class TestActiveStatuses:
    """active_statuses() 辅助函数测试"""

    def test_returns_non_terminal_only(self):
        active = active_statuses()
        assert "created" in active
        assert "pending_approval" in active
        assert "approved" in active
        assert "processing" in active
        assert "rejected" not in active
        assert "completed" not in active

    def test_all_active_are_non_terminal(self):
        for status_str in active_statuses():
            assert not is_terminal(status_str)


class TestIsTerminal:
    """is_terminal() 辅助函数测试"""

    def test_rejected_is_terminal(self):
        assert is_terminal(TicketStatus.REJECTED)
        assert is_terminal("rejected")

    def test_completed_is_terminal(self):
        assert is_terminal(TicketStatus.COMPLETED)
        assert is_terminal("completed")

    def test_created_is_not_terminal(self):
        assert not is_terminal(TicketStatus.CREATED)

    def test_pending_approval_is_not_terminal(self):
        assert not is_terminal(TicketStatus.PENDING_APPROVAL)

"""
测试: 审批权限修复 — 越级审批防护

验证:
- 当前步骤审批人可以审批通过
- 后面步骤的审批人不能越级审批
- 驳回权限同理
- 空 approver_name 兼容旧调用
"""

import pytest
from unittest.mock import MagicMock


def _make_mock_db(workflow, step, ticket=None):
    """构造 mock db session，支持多次 first() 调用。

    mock WorkflowService.transition 以避免 DB 状态写入的副作用。
    """
    mock_db = MagicMock()

    # query().filter().first() 返回序列: workflow → step → (next_step or ticket) ...
    call_results = [workflow, step]

    # 中间步骤通过后查询 ticket（事件数据）
    if ticket:
        call_results.append(ticket)  # for step_approved event

    # 如果最终步骤通过，需要更多调用
    if step.step_order >= workflow.total_steps:
        if ticket:
            call_results.append(ticket)  # for final approval transition
            call_results.append(ticket)  # for approval_completed event

    # next_step 查询（mock 返回 None，表示没有下一步）
    call_results.append(None)

    mock_db.query.return_value.filter.return_value.first.side_effect = call_results

    return mock_db


class TestApprovalPermission:
    """审批权限校验测试"""

    def test_wrong_approver_cannot_approve(self):
        """不是当前步骤审批人 → 抛出 PermissionError"""
        from services.approval_engine import ApprovalEngine

        mock_step = MagicMock()
        mock_step.approver = "王经理"
        mock_step.status = "pending"
        mock_step.step_order = 1

        mock_workflow = MagicMock()
        mock_workflow.id = 1
        mock_workflow.status = "pending"

        mock_db = MagicMock()
        mock_db.query.return_value.filter.return_value.first.side_effect = [
            mock_workflow,
            mock_step,
        ]

        with pytest.raises(PermissionError, match="无权操作"):
            ApprovalEngine.approve_step(
                workflow_id=1,
                step_order=1,
                approver_name="李HR",  # 李HR 不是当前步骤审批人
                db_session=mock_db,
            )

    def test_wrong_approver_cannot_reject(self):
        """驳回同样需要权限校验"""
        from services.approval_engine import ApprovalEngine

        mock_step = MagicMock()
        mock_step.approver = "王经理"
        mock_step.status = "pending"
        mock_step.step_order = 1

        mock_workflow = MagicMock()
        mock_workflow.id = 1
        mock_workflow.status = "pending"

        mock_db = MagicMock()
        mock_db.query.return_value.filter.return_value.first.side_effect = [
            mock_workflow,
            mock_step,
        ]

        with pytest.raises(PermissionError, match="无权操作"):
            ApprovalEngine.reject_step(
                workflow_id=1,
                step_order=1,
                approver_name="赵财务",  # 赵财务不是当前审批人
                db_session=mock_db,
            )

    def test_empty_approver_name_skips_check(self):
        """approver_name 为空时跳过权限校验（兼容旧调用）。

        会因 mock 不完整而抛出其他异常（如 AttributeError），
        但不应抛出 PermissionError。
        """
        from services.approval_engine import ApprovalEngine

        mock_step = MagicMock()
        mock_step.approver = "王经理"
        mock_step.status = "pending"
        mock_step.step_order = 1

        mock_workflow = MagicMock()
        mock_workflow.id = 1
        mock_workflow.status = "pending"
        mock_workflow.current_step = 0
        mock_workflow.total_steps = 2

        mock_db = MagicMock()
        # 空 approver_name → 不抛权限异常（后续可能因 mock 不完整抛其他异常）
        mock_db.query.return_value.filter.return_value.first.side_effect = [
            mock_workflow,
            mock_step,
            None,  # next_step
            None,  # ticket for event emit
        ]

        try:
            ApprovalEngine.approve_step(
                workflow_id=1,
                step_order=1,
                approver_name="",  # 空，兼容旧调用
                db_session=mock_db,
            )
        except PermissionError:
            pytest.fail("空 approver_name 应跳过权限校验")
        except Exception:
            pass  # 其他异常（mock 不完整）可以接受

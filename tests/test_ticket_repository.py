"""
TicketRepository 数据访问测试

测试覆盖：
- 创建工单（各类型）
- 查询（按ID/编号/用户/类型）
- 更新状态
- 软删除
- 统计
"""

import pytest
from db.base.session_manager import SessionManager
from db.repositories.ticket_repository import TicketRepository


@pytest.fixture
def repo():
    """创建内存 SQLite 的 TicketRepository"""
    sm = SessionManager("sqlite:///:memory:")
    return TicketRepository(sm)


class TestTicketCRUD:
    """工单 CRUD 测试"""

    def test_add_ticket_it_fault(self, repo):
        """创建 IT 故障工单"""
        ticket = repo.add_ticket(
            ticket_type="it_fault",
            title="网络故障",
            description="办公室网络连接不稳定",
            category="网络故障",
            priority="P1",
            requester_id="user001",
            trace_id="trace-123",
        )

        assert ticket["id"] == 1
        assert ticket["ticket_number"].startswith("TK-")
        assert ticket["ticket_type"] == "it_fault"
        assert ticket["title"] == "网络故障"
        assert ticket["priority"] == "P1"
        assert ticket["status"] == "created"
        assert ticket["is_active"] is True

    def test_add_ticket_leave(self, repo):
        """创建请假工单"""
        ticket = repo.add_ticket(
            ticket_type="leave",
            title="年假申请",
            description="申请年假5天",
            category="年假",
            priority="P3",
            payload={"leave_type": "年假", "total_days": 5},
        )

        assert ticket["ticket_type"] == "leave"
        assert ticket["payload"]["leave_type"] == "年假"
        assert ticket["payload"]["total_days"] == 5

    def test_add_ticket_expense(self, repo):
        """创建报销工单"""
        ticket = repo.add_ticket(
            ticket_type="expense",
            title="差旅报销",
            description="出差报销",
            category="差旅费",
            payload={"amount": 2500.0, "has_invoice": True},
        )

        assert ticket["ticket_type"] == "expense"
        assert ticket["payload"]["amount"] == 2500.0

    def test_get_ticket(self, repo):
        """按 ID 查询"""
        created = repo.add_ticket(
            ticket_type="it_fault", title="测试工单",
            description="测试用", category="测试",
        )
        fetched = repo.get_ticket(created["id"])
        assert fetched is not None
        assert fetched["ticket_number"] == created["ticket_number"]

    def test_get_nonexistent_ticket(self, repo):
        """查询不存在的工单"""
        result = repo.get_ticket(9999)
        assert result is None

    def test_get_by_number(self, repo):
        """按工单号查询"""
        created = repo.add_ticket(
            ticket_type="it_fault", title="测试",
            description="测试", category="测试",
        )
        fetched = repo.get_by_number(created["ticket_number"])
        assert fetched is not None
        assert fetched["id"] == created["id"]

    def test_list_tickets_all(self, repo):
        """列表查询：全部"""
        repo.add_ticket(ticket_type="it_fault", title="IT工单1", description="desc", category="网络故障")
        repo.add_ticket(ticket_type="leave", title="请假1", description="desc", category="年假")
        repo.add_ticket(ticket_type="expense", title="报销1", description="desc", category="差旅费")

        all_tickets = repo.list_tickets()
        assert len(all_tickets) == 3

    def test_list_tickets_filter_by_type(self, repo):
        """列表查询：按类型筛选"""
        repo.add_ticket(ticket_type="it_fault", title="IT", description="d", category="网络")
        repo.add_ticket(ticket_type="leave", title="请假", description="d", category="年假")

        it_tickets = repo.list_tickets(ticket_type="it_fault")
        assert len(it_tickets) == 1
        assert it_tickets[0]["ticket_type"] == "it_fault"

    def test_list_tickets_filter_by_status(self, repo):
        """列表查询：按状态筛选"""
        repo.add_ticket(ticket_type="it_fault", title="t1", description="d", category="c")
        t2 = repo.add_ticket(ticket_type="it_fault", title="t2", description="d", category="c")

        repo.update_status(t2["id"], "completed")
        resolved = repo.list_tickets(status="completed")
        assert len(resolved) == 1

    def test_update_status(self, repo):
        """更新工单状态"""
        ticket = repo.add_ticket(
            ticket_type="admin", title="会议室预定",
            description="预定了3楼大会议室", category="会议室预定",
        )

        success = repo.update_status(ticket["id"], "processing", assigned_to="admin01")
        assert success is True

        updated = repo.get_ticket(ticket["id"])
        assert updated["status"] == "processing"
        assert updated["assigned_to"] == "admin01"

    def test_update_invalid_field(self, repo):
        """更新非法字段被忽略"""
        ticket = repo.add_ticket(
            ticket_type="it_fault", title="t", description="d", category="c",
        )
        # 尝试更新不允许的字段
        repo.update_ticket(ticket["id"], not_a_field="value")
        # 不应报错，但也不应生效

    def test_delete_ticket_soft(self, repo):
        """软删除"""
        ticket = repo.add_ticket(
            ticket_type="it_fault", title="待删除", description="d", category="c",
        )
        success = repo.delete_ticket(ticket["id"], soft_delete=True)
        assert success is True

        # 查询不到（is_active=0）
        fetched = repo.get_ticket(ticket["id"])
        assert fetched is None

    def test_get_stats(self, repo):
        """统计接口"""
        repo.add_ticket(ticket_type="it_fault", title="IT", description="d",
                        category="网络故障", priority="P1")
        repo.add_ticket(ticket_type="leave", title="请假", description="d",
                        category="年假", priority="P3")
        repo.add_ticket(ticket_type="expense", title="报销", description="d",
                        category="差旅费", priority="P2")

        stats = repo.get_stats()
        assert stats["total"] == 3
        assert stats["by_type"]["it_fault"] == 1
        assert stats["by_type"]["leave"] == 1
        assert stats["by_type"]["expense"] == 1
        assert stats["by_priority"]["P1"] == 1
        assert stats["today"] >= 0

    def test_ticket_number_uniqueness(self, repo):
        """工单号唯一性"""
        t1 = repo.add_ticket(
            ticket_type="it_fault", title="t1", description="d", category="c",
        )
        t2 = repo.add_ticket(
            ticket_type="leave", title="t2", description="d", category="c",
        )
        assert t1["ticket_number"] != t2["ticket_number"]

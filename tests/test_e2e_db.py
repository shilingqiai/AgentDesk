"""
E2E 测试 — 维度 3: 数据库落地验证

验证业务数据准确写入 SQLite：
- 工单字段、格式、payload 持久化
- 提议 vs 执行模式下的 DB 状态差异
- 库存种子数据完整性
"""

import json
import pytest
from db.base.session_manager import SessionManager
from db.repositories.ticket_repository import TicketRepository


async def _make_seeded_agent():
    """创建注入内存 DB + 种子库存的 DynamicActionAgent

    同时将 TicketDispatch 的 DB 指向同一内存库，确保工单写入可验证。
    pre_extracted 参数已由 DynamicActionAgent 提供，TicketDispatch 不会调 LLM。
    """
    from agents.sub_agents.dynamic_action_agent import DynamicActionAgent
    from agents.orchestrator.agent_registry import agent_registry
    from db.db_router import DatabaseRouter

    db = DatabaseRouter("sqlite:///:memory:")

    agent = DynamicActionAgent()
    agent._db_router = db
    agent._inventory_seeded = False
    await agent._ensure_inventory_seeded()

    # ★ 关键: TicketDispatch 共享同一个内存 DB
    td_agent = agent_registry.get_agent("ticket_dispatch")
    if td_agent is not None:
        td_agent._db_router = db

    return agent


class TestTicketCreation:
    """工单创建 — 格式与字段验证"""

    def test_ticket_format_and_fields(self, in_memory_ticket_repo):
        """验证工单号格式、默认状态、活跃标志"""
        repo = in_memory_ticket_repo
        ticket = repo.add_ticket(
            ticket_type="admin", title="设备领用",
            description="ThinkPad+显示器", category="资产领用",
            priority="P2", requester_name="张三",
        )

        assert ticket["ticket_number"].startswith("TK-")
        parts = ticket["ticket_number"].split("-")
        assert len(parts) == 3
        assert len(parts[1]) == 8  # YYYYMMDD
        assert len(parts[2]) == 6  # XXXXXX
        assert len(ticket["ticket_number"]) == 18  # TK-YYYYMMDD-XXXXXX

        assert ticket["id"] == 1
        assert ticket["status"] == "created"
        assert ticket["ticket_type"] == "admin"
        assert ticket["title"] == "设备领用"
        assert ticket["is_active"] is True
        assert ticket["priority"] == "P2"
        assert ticket["requester_name"] == "张三"

    def test_ticket_payload_persistence(self, in_memory_ticket_repo):
        """JSON payload 字段 — 写后读完全一致"""
        repo = in_memory_ticket_repo
        payload = {
            "leave_type": "年假",
            "start_date": "2026-06-16",
            "end_date": "2026-06-20",
            "total_days": 5,
            "reason": "个人休假",
        }
        ticket = repo.add_ticket(
            ticket_type="leave", title="年假申请",
            description="申请年假5天", category="年假",
            priority="P3", requester_name="张三",
            payload=payload,
        )

        fetched = repo.get_ticket(ticket["id"])
        assert fetched is not None
        assert fetched["payload"]["leave_type"] == "年假"
        assert fetched["payload"]["total_days"] == 5
        assert fetched["payload"]["start_date"] == "2026-06-16"
        assert fetched["payload"]["end_date"] == "2026-06-20"
        assert fetched["payload"]["reason"] == "个人休假"

    def test_independent_ticket_numbers(self, in_memory_ticket_repo):
        """多个工单 → 工单号各不同"""
        repo = in_memory_ticket_repo
        t1 = repo.add_ticket(ticket_type="admin", title="A", description="a", category="c")
        t2 = repo.add_ticket(ticket_type="leave", title="B", description="b", category="c")
        t3 = repo.add_ticket(ticket_type="expense", title="C", description="c", category="c")

        numbers = {t1["ticket_number"], t2["ticket_number"], t3["ticket_number"]}
        assert len(numbers) == 3
        assert t1["id"] == 1
        assert t2["id"] == 2
        assert t3["id"] == 3


class TestProposalVsExecution:
    """提议 vs 执行模式 — DB 写入时机验证"""

    @pytest.mark.asyncio
    async def test_proposal_does_not_write_db(self):
        """提议模式 create_ticket → DB 无记录；确认后执行 → DB 有记录"""
        agent = await _make_seeded_agent()

        tickets_before = agent._db_router.ticket.list_tickets()
        assert len(tickets_before) == 0

        # Phase 1: 提议模式 — 不落库
        agent._execution_mode = False
        agent._last_user_name = "张三"
        agent._last_trace_id = "trace-proposal-test"

        proposal_json = await agent._tool_create_ticket({
            "ticket_type": "admin", "title": "设备领用",
            "description": "测试工单", "priority": "P2",
            "extra": {"service_type": "asset_requisition"},
        })
        proposal = json.loads(proposal_json)
        assert proposal.get("status") == "proposed"

        tickets_mid = agent._db_router.ticket.list_tickets()
        assert len(tickets_mid) == 0, "提议模式不应写入 DB"

        # Phase 2: 执行模式 — 落库
        agent._execution_mode = True
        exec_json = await agent._tool_create_ticket({
            "ticket_type": "admin", "title": "设备领用",
            "description": "测试工单", "priority": "P2",
            "extra": {"service_type": "asset_requisition"},
        })
        exec_result = json.loads(exec_json)
        assert exec_result.get("executed") is True
        assert exec_result["ticket_number"].startswith("TK-")

        tickets_after = agent._db_router.ticket.list_tickets()
        assert len(tickets_after) == 1

    @pytest.mark.asyncio
    async def test_confirmed_ticket_in_db(self):
        """执行后工单可通过 get_by_number 查回，所有字段匹配"""
        agent = await _make_seeded_agent()
        agent._execution_mode = True
        agent._last_user_name = "李四"
        agent._last_trace_id = "trace-confirm-test"

        exec_json = await agent._tool_create_ticket({
            "ticket_type": "leave", "title": "年假申请",
            "description": "申请年假3天", "priority": "P3",
            "extra": {
                "leave_type": "年假", "total_days": 3,
                "start_date": "2026-07-01", "end_date": "2026-07-03",
            },
        })
        exec_result = json.loads(exec_json)
        ticket_no = exec_result["ticket_number"]

        db_ticket = agent._db_router.ticket.get_by_number(ticket_no)
        assert db_ticket is not None
        assert db_ticket["ticket_type"] == "leave"
        assert db_ticket["title"] == "年假申请"
        assert db_ticket["status"] == "pending_approval"  # 请假工单有审批链，自动进入待审批状态
        assert db_ticket["requester_name"] == "李四"
        assert db_ticket["is_active"] is True


class TestInventorySeeding:
    """库存种子数据验证"""

    @pytest.mark.asyncio
    async def test_seed_creates_21_items(self):
        """_ensure_inventory_seeded 在空内存 DB 中创建 21 条种子数据"""
        agent = await _make_seeded_agent()
        session = agent._get_session()
        try:
            from db.models import InventoryItem
            items = session.query(InventoryItem).filter(
                InventoryItem.is_active == 1
            ).all()
            assert len(items) == 21

            thinkpad = [i for i in items if "ThinkPad" in i.item_name]
            assert len(thinkpad) == 1
            assert thinkpad[0].stock == 8
            assert thinkpad[0].min_threshold == 2
            assert thinkpad[0].unit_price == 14999

            lg = [i for i in items if "LG" in i.item_name and "显示器" in i.item_name]
            assert len(lg) == 1
            assert lg[0].stock == 0
            assert lg[0].min_threshold == 3
        finally:
            session.close()

    @pytest.mark.asyncio
    async def test_inventory_not_double_seeded(self):
        """重复调用 _ensure_inventory_seeded 不会重复插入"""
        agent = await _make_seeded_agent()
        await agent._ensure_inventory_seeded()  # 守卫应跳过

        session = agent._get_session()
        try:
            from db.models import InventoryItem
            count = session.query(InventoryItem).filter(
                InventoryItem.is_active == 1
            ).count()
            assert count == 21
        finally:
            session.close()


class TestTicketQuery:
    """工单查询与更新"""

    def test_list_tickets_filter_by_type(self, in_memory_ticket_repo):
        """list_tickets 可按类型筛选"""
        repo = in_memory_ticket_repo
        repo.add_ticket(ticket_type="it_fault", title="网络故障", description="d", category="c")
        repo.add_ticket(ticket_type="it_fault", title="打印机坏了", description="d", category="c")
        repo.add_ticket(ticket_type="leave", title="年假", description="d", category="c")

        it_tickets = repo.list_tickets(ticket_type="it_fault")
        assert len(it_tickets) == 2

        leave_tickets = repo.list_tickets(ticket_type="leave")
        assert len(leave_tickets) == 1

    def test_update_status_transitions(self, in_memory_ticket_repo):
        """工单状态可正常流转"""
        repo = in_memory_ticket_repo
        ticket = repo.add_ticket(ticket_type="it_fault", title="测试", description="d", category="c")
        assert ticket["status"] == "created"

        repo.update_status(ticket["id"], "processing", assigned_to="engineer1")
        updated = repo.get_ticket(ticket["id"])
        assert updated["status"] == "processing"
        assert updated["assigned_to"] == "engineer1"

        repo.update_status(ticket["id"], "completed")
        resolved = repo.get_ticket(ticket["id"])
        assert resolved["status"] == "completed"

    def test_delete_ticket_soft(self, in_memory_ticket_repo):
        """软删除后 get_ticket 返回 None"""
        repo = in_memory_ticket_repo
        ticket = repo.add_ticket(ticket_type="admin", title="测试", description="d", category="c")
        assert repo.get_ticket(ticket["id"]) is not None

        repo.delete_ticket(ticket["id"], soft_delete=True)
        assert repo.get_ticket(ticket["id"]) is None

    def test_get_stats_summary(self, in_memory_ticket_repo):
        """统计信息包含正确的汇总数据"""
        repo = in_memory_ticket_repo
        repo.add_ticket(ticket_type="it_fault", title="A", description="d", category="c",
                        priority="P1", status="created")
        repo.add_ticket(ticket_type="leave", title="B", description="d", category="c",
                        priority="P3", status="created")
        repo.add_ticket(ticket_type="it_fault", title="C", description="d", category="c",
                        priority="P2", status="completed")

        stats = repo.get_stats()
        assert stats["total"] == 3
        assert stats["by_type"]["it_fault"] == 2
        assert stats["by_type"]["leave"] == 1
        assert stats["by_status"]["created"] == 2
        assert stats["by_status"]["completed"] == 1

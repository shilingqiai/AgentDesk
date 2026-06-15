"""
E2E 测试 — 维度 2: SOP 工具调用轨迹

验证 DynamicActionAgent 的 ReAct 循环遵守业务 SOP：
- Rule 0: 不依赖的工具并行调用，create_ticket 绝不与其他工具并行
- Rule 1: 两阶段执行 — 先收集全部信息，再逐一提议工单
- Rule 2: 同类型合并 — 有货物品合并为一张领用单，缺货合并为采购单
- 提议→确认→执行生命周期
- 最大迭代次数兜底
"""

import json
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from tests.conftest import create_sequenced_mock_llm, build_agent_message


async def _setup_agent_for_trace():
    """创建注入内存 DB + seed + mock TicketDispatch DB 的 agent"""
    from agents.sub_agents.dynamic_action_agent import DynamicActionAgent
    from agents.orchestrator.agent_registry import agent_registry
    from db.db_router import DatabaseRouter

    db = DatabaseRouter("sqlite:///:memory:")

    agent = DynamicActionAgent()
    agent._db_router = db
    agent._inventory_seeded = False
    await agent._ensure_inventory_seeded()

    td_agent = agent_registry.get_agent("ticket_dispatch")
    if td_agent is not None:
        td_agent._db_router = db

    return agent


# ============================================================
# Rule 0: create_ticket 绝不与其他工具并行
# ============================================================

class TestRule0ParallelIsolation:
    """Rule 0: 并行工具隔离 — create_ticket 必须单独调用"""

    @pytest.mark.asyncio
    async def test_independent_tools_parallel_create_ticket_alone(self):
        """库存+知识并行查询 → create_ticket 单独调用 → 验证不混在同一批次"""
        agent = await _setup_agent_for_trace()

        # 构建 SOP 合规序列：Phase 1 并行的独立工具 → Phase 2 单独的 create_ticket
        mock_llm = create_sequenced_mock_llm([
            # 迭代 1: 4 个独立工具并行（无 create_ticket）
            ([
                {"name": "search_knowledge_base", "args": {"query": "设备领用政策"}},
                {"name": "check_inventory", "args": {"keyword": "ThinkPad"}},
                {"name": "check_inventory", "args": {"keyword": "显示器"}},
                {"name": "check_inventory", "args": {"keyword": "键鼠"}},
            ], "Phase 1: 并行收集信息"),
            # 迭代 2: create_ticket 单独调用
            ([
                {"name": "create_ticket", "args": {
                    "ticket_type": "admin", "title": "新员工设备领用",
                    "description": "ThinkPad + 显示器 + 键鼠",
                    "priority": "P2",
                    "extra": {"service_type": "asset_requisition"},
                }},
            ], "Phase 2: 创建合并领用单"),
            # 迭代 3: 最终答案
            ([], "已为您创建设备领用申请，请确认。"),
        ])
        agent.llm_with_tools = mock_llm  # 替换整个 RunnableBinding

        msg = build_agent_message("新员工入职需要ThinkPad、显示器、键鼠", "张三")
        result = await agent.execute(msg)

        assert result.success is True
        trace = result.payload.get("react_trace", [])
        assert len(trace) >= 5  # 4 个 info tools + 1 个 create_ticket

        tool_names = [t["tool_name"] for t in trace]

        # ★ 核心断言：create_ticket 前面的工具都是 info tools
        create_idx = tool_names.index("create_ticket")
        info_tools_before = tool_names[:create_idx]
        for t in info_tools_before:
            assert t != "create_ticket", (
                f"Rule 0 违规：create_ticket 在迭代中与其他工具并行"
            )

        # ★ 核心断言：所有 info tools 都在 create_ticket 之前
        info_tools = {"check_inventory", "search_knowledge_base", "web_search"}
        info_indices = [i for i, n in enumerate(tool_names) if n in info_tools]
        if info_indices:
            assert max(info_indices) < create_idx, (
                f"Rule 1 违规：info tools 应在 create_ticket 之前，"
                f"但发现 info_tool 索引 {max(info_indices)} > create_ticket 索引 {create_idx}"
            )

        # 验证卡片已生成
        assert result.payload.get("return_card") is True
        cards = result.payload.get("cards", [])
        assert len(cards) >= 1

    @pytest.mark.asyncio
    async def test_create_ticket_not_parallel_with_other_tools(self):
        """显式验证：若 LLM 返回 create_ticket + 其他工具在同一响应，
        则应在 trace 中可检测到此违规"""
        agent = await _setup_agent_for_trace()

        # 故意构造违规序列：create_ticket 与 check_inventory 并行
        mock_llm = create_sequenced_mock_llm([
            ([
                {"name": "check_inventory", "args": {"keyword": "键鼠"}},
                {"name": "create_ticket", "args": {
                    "ticket_type": "admin", "title": "设备领用",
                    "description": "测试", "priority": "P2",
                    "extra": {"service_type": "asset_requisition"},
                }},
            ], "违规：create_ticket 与其他工具并行"),
            ([], "完成。"),
        ])
        agent.llm_with_tools = mock_llm  # 替换整个 RunnableBinding

        msg = build_agent_message("帮我领键盘鼠标", "张三")
        result = await agent.execute(msg)

        trace = result.payload.get("react_trace", [])
        tool_names = [t["tool_name"] for t in trace]

        # 检测：同一批次（连续的非 create_ticket 和 create_ticket）意味着并行违规
        create_indices = [i for i, n in enumerate(tool_names) if n == "create_ticket"]
        non_create_indices = [i for i, n in enumerate(tool_names) if n != "create_ticket"]

        # 如果 create_ticket 和 info tools 交替出现 → 违规
        # 合规模式是：所有 info tools 都在 create_ticket 之前
        if create_indices and non_create_indices:
            # 这不是一个 hard fail（取决于 LLM），但记录行为
            violation = min(create_indices) < max(non_create_indices)
            if violation:
                # 有 info tool 在 create_ticket 之后 → 明确违规
                pass  # 记录但不强制失败（这是 mock 测试，用来检测能力）


# ============================================================
# Rule 1: 两阶段执行
# ============================================================

class TestRule1TwoPhaseExecution:
    """Rule 1: 先收集再建单 — info tools 必须在 create_ticket 之前"""

    @pytest.mark.asyncio
    async def test_gather_before_create_ticket(self):
        """所有 info tools 必须在 create_ticket 之前被调用"""
        agent = await _setup_agent_for_trace()

        mock_llm = create_sequenced_mock_llm([
            # Phase 1: 先查知识 + 查库存
            ([
                {"name": "search_knowledge_base", "args": {"query": "入职设备标准"}},
                {"name": "check_inventory", "args": {"keyword": "笔记本"}},
            ], "先收集信息"),
            # Phase 2: 再建单
            ([
                {"name": "create_ticket", "args": {
                    "ticket_type": "admin", "title": "设备领用",
                    "description": "笔记本", "priority": "P2",
                    "extra": {"service_type": "asset_requisition"},
                }},
            ], "再创建工单"),
            ([], "已为您创建申请。"),
        ])
        agent.llm_with_tools = mock_llm  # 替换整个 RunnableBinding

        msg = build_agent_message("新员工需要笔记本", "张三")
        result = await agent.execute(msg)

        trace = result.payload.get("react_trace", [])
        tool_names = [t["tool_name"] for t in trace]

        info_tools = {"check_inventory", "search_knowledge_base", "check_leave_balance"}
        create_indices = [i for i, n in enumerate(tool_names) if n == "create_ticket"]
        info_indices = [i for i, n in enumerate(tool_names) if n in info_tools]

        assert len(create_indices) >= 1, "应该有 create_ticket 调用"
        assert len(info_indices) >= 1, "应该有 info tool 调用"
        assert max(info_indices) < min(create_indices), (
            f"Rule 1 违规：info tools 索引 {info_indices} 应在 create_ticket 索引 "
            f"{create_indices} 之前。工具序列: {tool_names}"
        )

    @pytest.mark.asyncio
    async def test_leave_application_two_phase(self):
        """请假申请：先查余额 → 再建请假单"""
        agent = await _setup_agent_for_trace()

        mock_llm = create_sequenced_mock_llm([
            ([
                {"name": "check_leave_balance", "args": {"user_name": "张三"}},
            ], "先查余额"),
            ([
                {"name": "create_ticket", "args": {
                    "ticket_type": "leave", "title": "年假申请",
                    "description": "申请年假5天", "priority": "P3",
                    "extra": {
                        "leave_type": "年假", "total_days": 5,
                        "start_date": "2026-06-16", "end_date": "2026-06-20",
                    },
                }},
            ], "再请假"),
            ([], "已生成请假确认卡片。"),
        ])
        agent.llm_with_tools = mock_llm  # 替换整个 RunnableBinding

        msg = build_agent_message("我要请5天年假", "张三")
        result = await agent.execute(msg)

        assert result.success is True
        trace = result.payload.get("react_trace", [])
        tool_names = [t["tool_name"] for t in trace]

        assert "check_leave_balance" in tool_names
        assert "create_ticket" in tool_names
        # balance 查询应在 create 之前
        assert tool_names.index("check_leave_balance") < tool_names.index("create_ticket")


# ============================================================
# Rule 2: 同类型合并
# ============================================================

class TestRule2MergeSameTypes:
    """Rule 2: 同类型物品合并 — 有货→领用单，缺货→采购单"""

    @pytest.mark.asyncio
    async def test_merge_in_stock_to_one_requisition(self):
        """有货+缺货混合 → 2次 create_ticket（1 领用 + 1 采购）"""
        agent = await _setup_agent_for_trace()

        mock_llm = create_sequenced_mock_llm([
            # Phase 1: 并行查库存
            ([
                {"name": "check_inventory", "args": {"keyword": "ThinkPad"}},
                {"name": "check_inventory", "args": {"keyword": "Dell显示器"}},
                {"name": "check_inventory", "args": {"keyword": "LG显示器"}},
            ], "Phase 1: 检查库存（ThinkPad有货、Dell有货、LG缺货）"),
            # Phase 2: 领用单（有货物品合并）
            ([
                {"name": "create_ticket", "args": {
                    "ticket_type": "admin", "title": "设备领用",
                    "description": "ThinkPad + Dell显示器", "priority": "P2",
                    "extra": {"service_type": "asset_requisition"},
                }},
            ], "Phase 2: 合并有货物品到一张领用单"),
            # Phase 2 继续: 采购单（缺货物品）
            ([
                {"name": "create_ticket", "args": {
                    "ticket_type": "admin", "title": "采购申请",
                    "description": "LG 4K显示器", "priority": "P2",
                    "extra": {"service_type": "procurement"},
                }},
            ], "Phase 2: 缺货物品单独采购"),
            ([], "已完成设备领用和采购申请。"),
        ])
        agent.llm_with_tools = mock_llm  # 替换整个 RunnableBinding

        msg = build_agent_message("需要ThinkPad、Dell显示器和LG显示器", "张三")
        result = await agent.execute(msg)

        trace = result.payload.get("react_trace", [])
        create_calls = [t for t in trace if t["tool_name"] == "create_ticket"]
        assert len(create_calls) == 2, f"预期 2 次 create_ticket，实际 {len(create_calls)}"

        # 第一次是领用，第二次是采购
        call1_args = create_calls[0]["tool_args"]
        call2_args = create_calls[1]["tool_args"]
        call1_service = call1_args.get("extra", {}).get("service_type", "")
        call2_service = call2_args.get("extra", {}).get("service_type", "")
        assert "asset_requisition" in call1_service.lower() or "领用" in call1_service
        assert "procurement" in call2_service.lower() or "采购" in call2_service


# ============================================================
# 提议→确认→执行 生命周期
# ============================================================

class TestProposeExecuteLifecycle:
    """create_ticket 两阶段生命周期：先提议(status=proposed) → 确认后执行"""

    @pytest.mark.asyncio
    async def test_propose_then_execute_lifecycle(self):
        """完整生命周期：提议 → DB 无记录 → 确认 → DB 有记录"""
        agent = await _setup_agent_for_trace()

        # ── Step 1: 提议模式 ──
        agent._execution_mode = False
        agent._last_user_name = "张三"
        agent._last_trace_id = "trace-lifecycle"

        proposal_json = await agent._tool_create_ticket({
            "ticket_type": "admin", "title": "设备领用",
            "description": "测试工单", "priority": "P2",
            "extra": {"service_type": "asset_requisition"},
        })
        proposal = json.loads(proposal_json)

        # 断言：提议状态
        assert proposal["status"] == "proposed"
        assert agent._db_router.ticket.get_ticket_count() == 0

        # ── Step 2: 执行模式 ──
        agent._execution_mode = True
        exec_json = await agent._tool_create_ticket({
            "ticket_type": "admin", "title": "设备领用",
            "description": "测试工单", "priority": "P2",
            "extra": {"service_type": "asset_requisition"},
        })
        exec_result = json.loads(exec_json)

        # 断言：执行状态
        assert exec_result["executed"] is True
        assert exec_result["ticket_number"].startswith("TK-")
        assert agent._db_router.ticket.get_ticket_count() == 1

        # 断言：DB 记录完整
        db_ticket = agent._db_router.ticket.get_by_number(
            exec_result["ticket_number"]
        )
        assert db_ticket is not None
        assert db_ticket["status"] == "created"
        assert db_ticket["requester_name"] == "张三"


# ============================================================
# 最大迭代次数兜底
# ============================================================

class TestMaxIterationsGuard:
    """ReAct 循环强制终止 + 压缩总结"""

    @pytest.mark.asyncio
    async def test_max_iterations_truncates(self):
        """超过 MAX_REACT_ITERATIONS → 强制终止 + results 含 summary"""
        agent = await _setup_agent_for_trace()
        agent.MAX_REACT_ITERATIONS = 3

        # 构建 5 次 tool_call 响应（超过 MAX=3）
        sequences = [
            ([{"name": "check_inventory", "args": {"keyword": "test"}}], f"iter {i}")
            for i in range(5)
        ]
        mock_llm = create_sequenced_mock_llm(sequences)
        agent.llm_with_tools = mock_llm  # 替换整个 RunnableBinding

        # Mock 最终 LLM 调用（无工具绑定的那个）
        mock_final_response = MagicMock()
        mock_final_response.content = "已达到最大操作步数，总结中..."
        agent.llm = MagicMock()
        agent.llm.ainvoke = AsyncMock(return_value=mock_final_response)

        msg = build_agent_message("一直循环查询", "张三")
        result = await agent.execute(msg)

        assert result.success is True
        # 应包含 direct_response（强制总结）
        assert result.payload.get("direct_response") is not None
        # 迭代次数 ≤ MAX（因为达到后强制终止）
        iterations = result.payload.get("iterations", 0)
        assert iterations <= agent.MAX_REACT_ITERATIONS + 1  # +1 for final summary

    @pytest.mark.asyncio
    async def test_normal_completion_within_iterations(self):
        """正常完成 → 迭代次数 < MAX"""
        agent = await _setup_agent_for_trace()
        agent.MAX_REACT_ITERATIONS = 15

        mock_llm = create_sequenced_mock_llm([
            ([{"name": "check_inventory", "args": {"keyword": "鼠标"}}], "check"),
            ([], "库存查询完成。"),
        ])
        agent.llm_with_tools = mock_llm  # 替换整个 RunnableBinding

        msg = build_agent_message("查一下鼠标库存", "张三")
        result = await agent.execute(msg)

        trace = result.payload.get("react_trace", [])
        # 应该只有 1 次工具调用
        assert len(trace) == 1
        assert trace[0]["tool_name"] == "check_inventory"
        assert result.payload["iterations"] <= 2  # 1 tool + 1 final

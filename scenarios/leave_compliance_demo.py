#!/usr/bin/env python3
"""
请假合规检查 — 多 Agent DAG 编排端到端演示

场景：员工想请 5 天年假，系统并行调用 RAG（查政策）+ ToolAgent（查余额），
      结果汇聚到 TicketDispatch（合规检查 + 生成确认卡片）。

DAG 拓扑：
                     "我想请5天年假，6月15日到20日"
                                │
                 ┌──────────────┴──────────────┐
                 ▼                             ▼
           Step 1: RAG                    Step 2: ToolAgent
           "查询年假政策"                "查询员工年假余额"
           → LeavePolicyResult           → LeaveBalanceResult
                 │                             │
                 └──────────────┬──────────────┘
                                ▼
                      Step 3: TicketDispatch
                      "合规检查 + 生成确认卡片"
                      输入: policy + balance
                      → ComplianceResult

面试叙事要点：
1. "原方案需要 3 次 LLM 串行调用（RAG→判断是否够→生成卡片）"
2. "改进后 RAG 和余额查询并行（2 个 LLM 调用同时发出）"
3. "汇聚后 TicketDispatch 拿到两个结构化结果直接做合规判断"
4. "整个过程类型安全：LeavePolicy + LeaveBalance → ComplianceResult"

运行方式：
    python scenarios/leave_compliance_demo.py
"""

from __future__ import annotations

import asyncio
import json
import logging
import sys
from pathlib import Path

# Windows 终端 UTF-8 编码
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# 确保项目根在 sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agents.harness import AgentHarness, StepConfig, parallel
from agents.harness.step_schema import (
    LeavePolicyResult,
    LeaveBalanceResult,
    ComplianceResult,
)
from agents.orchestrator.agent_registry import agent_registry
from agents.a2a.protocol import AgentMessage

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("scenario.leave")


# ============================================================
# Step 1: RAG — 查询年假政策
# ============================================================

async def rag_leave_policy(user_input: str) -> LeavePolicyResult:
    """
    调用 EnterpriseRAG 查询年假政策。

    真实场景中 RAG 从知识库检索政策文档，LLM 从文档中提取结构化字段。
    这里为 demo 清晰性使用 LLM 直接提取，生产环境应走 RAG 检索流程。
    """
    from config.model_provider import create_chat_model

    llm = create_chat_model(temperature=0)
    prompt = (
        "你是一个 HR 政策查询助手。根据企业年假政策，提取以下结构化信息。\n\n"
        "## 企业年假政策（默认）\n"
        "- 员工每年享有 5-20 天年假（按工龄）\n"
        "- 连续休假不超过 10 个工作日\n"
        "- 超过 3 个工作日的年假需提前 3 天申请并经直属主管审批\n"
        "- 年终（12月25日-1月5日）为封账期，不可休年假\n"
        "- 法定节假日不计入年假天数\n\n"
        f"用户咨询：{user_input}\n\n"
        "返回 JSON（不要 markdown 包裹）：\n"
        '{"max_consecutive_days": 10, "requires_approval_above_days": 3, '
        '"blackout_dates": ["12/25-1/5"], "policy_summary": "..."}'
    )

    response = await llm.ainvoke([{"role": "user", "content": prompt}])

    # 解析
    import re
    text = response.content.strip()
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r"\{[\s\S]*\}", text)
        data = json.loads(m.group(0)) if m else {}

    return LeavePolicyResult(
        max_consecutive_days=int(data.get("max_consecutive_days", 10)),
        requires_approval_above_days=int(data.get("requires_approval_above_days", 3)),
        blackout_dates=data.get("blackout_dates", []),
        policy_summary=data.get("policy_summary", ""),
    )


# ============================================================
# Step 2: ToolAgent — 查年假余额
# ============================================================

async def tool_leave_balance(employee_name: str) -> LeaveBalanceResult:
    """
    调用 ToolAgent 查询员工年假余额。

    ToolAgent 内部通过 tool_registry 发现并调用 leave_balance_query 工具。
    """
    from agents.sub_agents.tool_agent import ToolAgent

    agent = ToolAgent()
    message = AgentMessage.create_delegation(
        from_agent="harness",
        to_agent="tool_agent",
        payload={
            "user_input": f"查询员工 {employee_name} 的年假余额",
            "task": f"查询 {employee_name} 的假期余额",
        },
    )

    response = await agent.execute(message)

    if not response.success:
        raise RuntimeError(f"ToolAgent 执行失败: {response.error}")

    tool_result = response.payload.get("tool_result", {})
    if not tool_result:
        # ToolAgent 可能没找到工具，用默认值
        return LeaveBalanceResult(
            employee_name=employee_name,
            annual_leave_total=15,
            annual_leave_used=3,
            annual_leave_remaining=12,
            sick_leave_remaining=5,
            query_time="2026-06-10 10:00",
        )

    return LeaveBalanceResult(
        employee_name=tool_result.get("employee_name", employee_name),
        annual_leave_total=int(tool_result.get("annual_leave_total", 0)),
        annual_leave_used=int(tool_result.get("annual_leave_used", 0)),
        annual_leave_remaining=int(tool_result.get("annual_leave_remaining", 0)),
        sick_leave_remaining=int(tool_result.get("sick_leave_remaining", 0)),
        query_time=tool_result.get("query_time", ""),
    )


# ============================================================
# Step 3: TicketDispatch — 合规检查 + 生成卡片
# ============================================================

async def ticket_compliance_check(
    user_input: str,
    policy: LeavePolicyResult,
    balance: LeaveBalanceResult,
) -> ComplianceResult:
    """
    TicketDispatch 执行合规检查。

    输入来自 Step 1（policy）和 Step 2（balance），
    检查后生成预填确认卡片。

    面试要点：这就是 "type-safe result passing"，
    LeavePolicyResult + LeaveBalanceResult → ComplianceResult
    """
    from config.model_provider import create_chat_model

    # 提取用户请求中的天数和日期
    llm = create_chat_model(temperature=0)
    prompt = (
        "你是一个请假合规检查器。根据政策、余额和用户请求，进行合规检查。\n\n"
        f"## 年假政策\n{json.dumps(policy, ensure_ascii=False)}\n\n"
        f"## 员工余额\n{json.dumps(balance, ensure_ascii=False)}\n\n"
        f"## 用户请求\n{user_input}\n\n"
        "## 检查项\n"
        "1. 请假天数是否 ≤ 最长连续休假天数\n"
        "2. 剩余年假是否 ≥ 请假天数\n"
        "3. 日期是否在封账期 (blackout_dates) 内\n"
        "4. 是否触发审批要求\n\n"
        "返回 JSON：\n"
        '{"passed": true/false, '
        '"checks": [{"name":"...", "passed":true/false, "detail":"..."}], '
        '"warnings": ["..."]}'
    )

    response = await llm.ainvoke([{"role": "user", "content": prompt}])

    import re
    text = response.content.strip()
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r"\{[\s\S]*\}", text)
        data = json.loads(m.group(0)) if m else {}

    passed = data.get("passed", False)
    checks = data.get("checks", [])
    warnings = data.get("warnings", [])

    # 生成确认卡片
    card = {
        "type": "leave",
        "title": "请假申请确认",
        "employee": balance["employee_name"],
        "leave_type": "年假",
        "days_requested": 5,
        "date_range": "2026-06-15 ~ 2026-06-20",
        "remaining_after": balance["annual_leave_remaining"] - 5,
        "checks": checks,
        "warnings": warnings,
        "requires_approval": any("审批" in c.get("detail", "") for c in checks),
    }

    summary = (
        f"✅ 合规检查通过" if passed else "❌ 合规检查未通过"
    ) + f" | 剩余年假: {balance['annual_leave_remaining']}天 | "
    if warnings:
        summary += f"⚠️ {'; '.join(warnings)}"

    return ComplianceResult(
        passed=passed,
        checks=checks,
        warnings=warnings,
        card=card,
        summary=summary,
    )


# ============================================================
# 主演示
# ============================================================

async def demo_harness_mode():
    """使用 AgentHarness 执行 DAG 编排"""
    print("=" * 62)
    print("  请假合规检查 — Agent Harness DAG 编排演示")
    print("=" * 62)

    harness = AgentHarness()

    step_rag = StepConfig(
        agent_id="enterprise_rag",
        task="查询年假政策：最长连续休假天数、审批门槛、封账期",
        output_key="policy",
        output_schema=LeavePolicyResult,
        timeout=30.0,
        retries=1,
    )

    step_balance = StepConfig(
        agent_id="tool_agent",
        task="查询员工张三的年假余额",
        output_key="balance",
        output_schema=LeaveBalanceResult,
        timeout=15.0,
        retries=1,
    )

    step_compliance = StepConfig(
        agent_id="ticket_dispatch",
        task="合规检查：对比政策和余额，生成确认卡片",
        output_key="compliance",
        output_schema=ComplianceResult,
        input_from=["policy", "balance"],  # ← DAG 汇聚点
        timeout=30.0,
        retries=2,
        fallback_agent="enterprise_rag",   # fallback
    )

    print("\n📋 DAG 拓扑:")
    print("  Layer 0: [RAG ∥ ToolAgent]  并行")
    print("  Layer 1: [TicketDispatch]    汇聚 ↓")
    print(f"\n  Steps: {step_rag}\n         {step_balance}\n         {step_compliance}")

    print("\n🚀 开始执行...\n")

    result = await harness.run(
        task="我想请5天年假，6月15日到20日",
        steps=[
            parallel(step_rag, step_balance),
            step_compliance,
        ],
        thread_id="demo-leave-001",
    )

    # ── 输出结果 ──
    print("\n" + "─" * 62)
    print("📊 执行结果")
    print("─" * 62)
    print(f"  总耗时: {result.total_duration_ms:.0f}ms")
    print(f"  成功: {'✅' if result.success else '❌'}")
    print(f"  步骤数: {len(result.steps)}")

    for i, sr in enumerate(result.steps):
        icon = "✅" if sr.success else "❌"
        print(f"\n  Step {i + 1}: {icon} {sr.step.agent_id}")
        print(f"    任务: {sr.step.task[:60]}")
        print(f"    耗时: {sr.duration_ms:.0f}ms")
        print(f"    重试: {sr.retries_used}")
        if sr.success:
            print(f"    输出 keys: {list(sr.data.keys())}")
        else:
            print(f"    错误: {sr.error}")

    # ── 合规检查摘要 ──
    compliance = result.final_output.get("compliance", {})
    if compliance:
        print("\n" + "─" * 62)
        print("📋 合规检查详情")
        print("─" * 62)
        print(f"  通过: {'✅' if compliance.get('passed') else '❌'}")
        for check in compliance.get("checks", []):
            icon = "✅" if check.get("passed") else "❌"
            print(f"  {icon} {check.get('name')}: {check.get('detail', '')}")
        if compliance.get("warnings"):
            for w in compliance["warnings"]:
                print(f"  ⚠️  {w}")
        print(f"\n  📇 确认卡片: {json.dumps(compliance.get('card', {}), ensure_ascii=False, indent=2)}")

    print("\n" + "=" * 62)
    return result


async def demo_direct_mode():
    """
    直接调用模式（不用 Harness）—— 展示无 Harness 时的手动编排。

    用于对比：相同逻辑在有无 Harness 时的代码差异。
    """
    print("\n\n")
    print("=" * 62)
    print("  对比：无 Harness 的手动编排")
    print("=" * 62)

    import time

    t0 = time.perf_counter()

    # 手动并行
    policy_task = rag_leave_policy("我想请5天年假，6月15日到20日")
    balance_task = tool_leave_balance("张三")
    policy, balance = await asyncio.gather(policy_task, balance_task)

    # 手动传递结果
    compliance = await ticket_compliance_check(
        "我想请5天年假，6月15日到20日", policy, balance,
    )

    elapsed = (time.perf_counter() - t0) * 1000

    print(f"\n  耗时: {elapsed:.0f}ms")
    print(f"  政策: {policy['policy_summary'][:60]}...")
    print(f"  余额: {balance['annual_leave_remaining']}天剩余")
    print(f"  合规: {compliance['summary']}")

    print("\n  ⚠️ 问题：无超时/重试/fallback/类型校验/步骤观测")
    print("  → 这就是 Harness 要做的事情。\n")


# ============================================================
# 入口
# ============================================================

async def main():
    # 确保 Agent 已注册
    import agents.sub_agents.enterprise_rag     # noqa: F401
    import agents.sub_agents.ticket_dispatch    # noqa: F401
    import agents.sub_agents.tool_agent         # noqa: F401

    # 确保工具已注册
    import agents.tools.builtin_tools           # noqa: F401

    # 初始化 RAG Agent 的知识库
    rag = agent_registry.get_agent("enterprise_rag")
    if rag:
        await rag._ensure_initialized()
        print(f"✅ EnterpriseRAG 就绪（{rag.knowledge_service.get_documents_count()} 篇文档）\n")

    # Harness 模式
    await demo_harness_mode()

    # 对比模式
    await demo_direct_mode()


if __name__ == "__main__":
    asyncio.run(main())

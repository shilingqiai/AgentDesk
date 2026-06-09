"""
DashScope Function Calling 健壮性验证脚本

测试维度：
  1. 基础 Function Calling (bind_tools + tool_choice="auto")
  2. with_structured_output (简单 Pydantic)
  3. with_structured_output (嵌套 Pydantic — TicketParams 等价)
  4. 不同 tool_choice 参数兼容性
  5. 流式模式下的 Function Calling
  6. 对照：当前 prompt→JSON 方案

模型：qwen-max (主模型) + qwen-mt-flash (路由模型)
"""

from __future__ import annotations

import json
import os
import sys
import time
import asyncio
import re
from typing import Literal
from dataclasses import dataclass, field

from pydantic import BaseModel, Field
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, ToolMessage
from langchain_openai import ChatOpenAI
from pydantic import SecretStr
from dotenv import load_dotenv

load_dotenv()

# ── 配置 ──────────────────────────────────────────────
DASHSCOPE_BASE = "https://dashscope.aliyuncs.com/compatible-mode/v1"
API_KEY = os.getenv("DASHSCOPE_API_KEY", "")
MAIN_MODEL = os.getenv("LLM_MODEL", "qwen-max")
ROUTER_MODEL = os.getenv("ROUTER_MODEL", "qwen-mt-flash")

if not API_KEY:
    print("[ERROR] DASHSCOPE_API_KEY 未设置，请检查 .env 文件")
    sys.exit(1)


def create_llm(model: str, temperature: float = 0) -> ChatOpenAI:
    return ChatOpenAI(
        model=model,
        api_key=SecretStr(API_KEY),
        base_url=DASHSCOPE_BASE,
        temperature=temperature,
    )


# ── Pydantic 模型（对应项目实际使用） ────────────────

class RouterDecision(BaseModel):
    """Router 路由决策"""
    track: Literal["fast", "action", "complex", "clarify"] = Field(
        description="路由轨道"
    )
    confidence: float = Field(description="置信度 0.0-1.0", ge=0.0, le=1.0)
    reason: str = Field(description="路由理由")
    requires_tools: list[str] = Field(default_factory=list, description="需要的工具")


class TicketExtra(BaseModel):
    """工单扩展字段（嵌套）"""
    leave_type: str = Field(default="", description="请假类型")
    start_date: str = Field(default="", description="开始日期 YYYY-MM-DD")
    end_date: str = Field(default="", description="结束日期 YYYY-MM-DD")
    total_days: int = Field(default=0, description="请假天数")


class TicketParams(BaseModel):
    """工单参数（含嵌套）"""
    ticket_type: Literal["it_fault", "leave", "expense", "admin"] = Field(
        description="工单类型"
    )
    title: str = Field(description="工单标题")
    description: str = Field(description="工单描述")
    category: str = Field(description="具体分类")
    priority: Literal["P0", "P1", "P2", "P3"] = Field(default="P2", description="优先级")
    extra: TicketExtra = Field(default_factory=TicketExtra, description="扩展字段")


class CardIntent(BaseModel):
    """卡片意图分类"""
    intent: Literal["confirm", "modify", "cancel", "new_topic"] = Field(
        description="用户意图"
    )
    reason: str = Field(default="", description="分类理由")


# ── 工具定义（模拟项目中的 Function Calling） ────────

ROUTER_TOOL = {
    "type": "function",
    "function": {
        "name": "RouterDecision",
        "description": "分析用户输入，返回路由决策",
        "parameters": {
            "type": "object",
            "properties": {
                "track": {
                    "type": "string",
                    "enum": ["fast", "action", "complex", "clarify"],
                    "description": "路由轨道: fast=查资料, action=办事情, complex=复合指令, clarify=反问",
                },
                "confidence": {
                    "type": "number",
                    "description": "置信度 0.0-1.0",
                    "minimum": 0.0,
                    "maximum": 1.0,
                },
                "reason": {"type": "string", "description": "一句话理由"},
                "requires_tools": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "需要的工具列表",
                },
            },
            "required": ["track", "confidence", "reason"],
        },
    },
}


# ── 测试用例 ──────────────────────────────────────────

TEST_INPUTS = {
    "router": ["VPN怎么连", "我想请3天年假", "帮我查天气然后请假再取消会议室", "嗯？"],
    "ticket_params": [
        "我想请3天年假，从明天开始",
        "帮我提交一个网络故障工单，VPN连不上了",
        "报销差旅费500元，有发票",
    ],
    "card_intent": [
        ("好的，确认预定", "会议室预定"),
        ("算了，不定了", "会议室预定"),
        ("帮我换成A201房间", "会议室预定"),
        ("对了，请假流程是什么", "会议室预定"),
    ],
}


# ── 辅助函数 ──────────────────────────────────────────

def parse_json(text: str) -> dict:
    """解析 LLM JSON 输出（与项目一致的健壮提取）"""
    # 直接解析
    try:
        return json.loads(text.strip())
    except json.JSONDecodeError:
        pass
    # markdown 块
    m = re.search(r'```(?:json)?\s*([\s\S]*?)```', text)
    if m:
        try:
            return json.loads(m.group(1).strip())
        except json.JSONDecodeError:
            pass
    # 第一个 { ... }
    m = re.search(r'\{[\s\S]*\}', text)
    if m:
        raw = m.group(0)
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            try:
                from json_repair import repair_json
                return json.loads(repair_json(raw))
            except Exception:
                pass
    raise ValueError(f"无法提取 JSON: {text[:200]}")


# ══════════════════════════════════════════════════════
# 测试 1: 基础 Function Calling (bind_tools + auto)
# ══════════════════════════════════════════════════════

@dataclass
class TestResult:
    name: str
    passed: bool
    duration_ms: float
    error: str = ""
    detail: str = ""


async def test_bind_tools_auto(model_name: str) -> list[TestResult]:
    """测试 bind_tools + tool_choice='auto'"""
    results = []
    llm = create_llm(model_name, temperature=0)
    llm_with_tools = llm.bind_tools([ROUTER_TOOL], tool_choice="auto")

    for user_input in TEST_INPUTS["router"]:
        name = f"bind_tools(auto) | {model_name} | '{user_input[:20]}'"
        t0 = time.perf_counter()
        try:
            response = await llm_with_tools.ainvoke([
                SystemMessage(content="你是企业AI服务台的路由器。分析用户输入，调用 RouterDecision 函数。"),
                HumanMessage(content=user_input),
            ])

            # 检查是否返回了 tool_calls
            if hasattr(response, 'tool_calls') and response.tool_calls:
                tc = response.tool_calls[0]
                args = tc.get("args", {})
                track = args.get("track", "?")
                conf = args.get("confidence", "?")
                results.append(TestResult(
                    name=name, passed=True,
                    duration_ms=(time.perf_counter() - t0) * 1000,
                    detail=f"track={track}, confidence={conf}",
                ))
            elif response.content and "{" in response.content:
                # 模型没走 tool_call，但返回了 JSON 文本
                results.append(TestResult(
                    name=name, passed=False,
                    duration_ms=(time.perf_counter() - t0) * 1000,
                    error="模型未调用 tool，返回了普通文本",
                    detail=f"content[:100]={response.content[:100]}",
                ))
            else:
                results.append(TestResult(
                    name=name, passed=False,
                    duration_ms=(time.perf_counter() - t0) * 1000,
                    error="模型既无 tool_calls 也无 JSON 内容",
                    detail=f"content={str(response)[:200]}",
                ))
        except Exception as e:
            results.append(TestResult(
                name=name, passed=False,
                duration_ms=(time.perf_counter() - t0) * 1000,
                error=f"异常: {type(e).__name__}: {str(e)[:150]}",
            ))

    return results


# ══════════════════════════════════════════════════════
# 测试 2: with_structured_output (简单 Pydantic)
# ══════════════════════════════════════════════════════

async def test_structured_output_simple(model_name: str) -> list[TestResult]:
    """测试 with_structured_output(RouterDecision) — 简单模型"""
    results = []
    llm = create_llm(model_name, temperature=0)

    try:
        structured_llm = llm.with_structured_output(RouterDecision)
    except Exception as e:
        return [TestResult(
            name=f"with_structured_output(RouterDecision) | {model_name}",
            passed=False, duration_ms=0,
            error=f"初始化失败: {type(e).__name__}: {str(e)[:150]}",
        )]

    for user_input in TEST_INPUTS["router"]:
        name = f"structured(RouterDecision) | {model_name} | '{user_input[:20]}'"
        t0 = time.perf_counter()
        try:
            decision = await structured_llm.ainvoke([
                SystemMessage(content="你是企业AI服务台的路由器。"),
                HumanMessage(content=user_input),
            ])

            if isinstance(decision, RouterDecision):
                results.append(TestResult(
                    name=name, passed=True,
                    duration_ms=(time.perf_counter() - t0) * 1000,
                    detail=f"track={decision.track}, confidence={decision.confidence:.0%}",
                ))
            elif isinstance(decision, dict):
                results.append(TestResult(
                    name=name, passed=True,
                    duration_ms=(time.perf_counter() - t0) * 1000,
                    detail=f"(dict) track={decision.get('track')}, conf={decision.get('confidence')}",
                ))
            else:
                results.append(TestResult(
                    name=name, passed=False,
                    duration_ms=(time.perf_counter() - t0) * 1000,
                    error=f"返回类型异常: {type(decision).__name__}",
                    detail=f"value={str(decision)[:200]}",
                ))
        except Exception as e:
            results.append(TestResult(
                name=name, passed=False,
                duration_ms=(time.perf_counter() - t0) * 1000,
                error=f"{type(e).__name__}: {str(e)[:200]}",
            ))

    return results


# ══════════════════════════════════════════════════════
# 测试 3: with_structured_output (嵌套 Pydantic)
# ══════════════════════════════════════════════════════

async def test_structured_output_nested(model_name: str) -> list[TestResult]:
    """测试 with_structured_output(TicketParams) — 嵌套模型"""
    results = []
    llm = create_llm(model_name, temperature=0)

    try:
        structured_llm = llm.with_structured_output(TicketParams)
    except Exception as e:
        return [TestResult(
            name=f"with_structured_output(TicketParams) | {model_name}",
            passed=False, duration_ms=0,
            error=f"初始化失败: {type(e).__name__}: {str(e)[:150]}",
        )]

    for user_input in TEST_INPUTS["ticket_params"]:
        name = f"structured(TicketParams) | {model_name} | '{user_input[:25]}'"
        t0 = time.perf_counter()
        try:
            params = await structured_llm.ainvoke([
                SystemMessage(content="你是工单参数提取器。"),
                HumanMessage(content=user_input),
            ])

            if isinstance(params, TicketParams):
                results.append(TestResult(
                    name=name, passed=True,
                    duration_ms=(time.perf_counter() - t0) * 1000,
                    detail=(f"type={params.ticket_type}, title={params.title}, "
                            f"extra.leave_type={params.extra.leave_type}, "
                            f"extra.start_date={params.extra.start_date}"),
                ))
            elif isinstance(params, dict):
                results.append(TestResult(
                    name=name, passed=True,
                    duration_ms=(time.perf_counter() - t0) * 1000,
                    detail=f"(dict) type={params.get('ticket_type')}",
                ))
            else:
                results.append(TestResult(
                    name=name, passed=False,
                    duration_ms=(time.perf_counter() - t0) * 1000,
                    error=f"返回类型异常: {type(params).__name__}",
                ))
        except Exception as e:
            results.append(TestResult(
                name=name, passed=False,
                duration_ms=(time.perf_counter() - t0) * 1000,
                error=f"{type(e).__name__}: {str(e)[:200]}",
            ))

    return results


# ══════════════════════════════════════════════════════
# 测试 4: tool_choice 参数兼容性矩阵
# ══════════════════════════════════════════════════════

async def test_tool_choice_matrix(model_name: str) -> list[TestResult]:
    """测试不同 tool_choice 值的行为"""
    results = []
    user_input = "我想请3天年假"

    choices = [
        ("auto", "auto"),
        ("none", "none"),
        ("required", "required"),
        # 对象格式 — LangChain with_structured_output 使用的格式
        ("object", {"type": "function", "function": {"name": "RouterDecision"}}),
    ]

    for label, tool_choice in choices:
        name = f"tool_choice={label} | {model_name} | '{user_input}'"
        t0 = time.perf_counter()
        try:
            llm = create_llm(model_name, temperature=0)
            llm_with_tools = llm.bind_tools([ROUTER_TOOL], tool_choice=tool_choice)
            response = await llm_with_tools.ainvoke([
                SystemMessage(content="你是路由器。分析用户输入，调用 RouterDecision 函数。"),
                HumanMessage(content=user_input),
            ])

            if hasattr(response, 'tool_calls') and response.tool_calls:
                tc = response.tool_calls[0]
                args = tc.get("args", {})
                results.append(TestResult(
                    name=name, passed=True,
                    duration_ms=(time.perf_counter() - t0) * 1000,
                    detail=f"[OK] tool_call: track={args.get('track')}",
                ))
            else:
                results.append(TestResult(
                    name=name, passed=False,
                    duration_ms=(time.perf_counter() - t0) * 1000,
                    error=f"未触发 tool_call",
                    detail=f"content[:80]={str(response.content)[:80]}",
                ))
        except Exception as e:
            results.append(TestResult(
                name=name, passed=False,
                duration_ms=(time.perf_counter() - t0) * 1000,
                error=f"{type(e).__name__}: {str(e)[:200]}",
            ))

    return results


# ══════════════════════════════════════════════════════
# 测试 5: 对照 — 当前 prompt→JSON 方案
# ══════════════════════════════════════════════════════

ROUTER_PROMPT = (
    "你是企业AI服务台的路由器。分析用户输入，判定走哪条轨道。\n\n"
    "**fast** — 知识查询/政策咨询/故障排查/方法问答\n"
    "**action** — 需要调接口/创建工单/提交申请/执行操作\n"
    "**complex** — 涉及2个以上独立任务\n"
    "**clarify** — 输入模糊/有歧义/无关话题/AI不确定\n\n"
    '输出严格 JSON（不要 markdown 包裹）：\n'
    '{"track":"fast|action|complex|clarify","confidence":0.0-1.0,'
    '"reason":"一句话理由","requires_tools":[]}'
)


async def test_prompt_to_json(model_name: str) -> list[TestResult]:
    """测试当前 prompt→JSON 方案（基准对照）"""
    results = []
    llm = create_llm(model_name, temperature=0)

    for user_input in TEST_INPUTS["router"]:
        name = f"prompt→JSON | {model_name} | '{user_input[:20]}'"
        t0 = time.perf_counter()
        success = True
        detail = ""
        error = ""

        try:
            response = await llm.ainvoke([
                SystemMessage(content=ROUTER_PROMPT),
                HumanMessage(content=f"分析以下用户输入并返回路由决策 JSON：{user_input}"),
            ])

            # 尝试多层 JSON 提取
            try:
                data = parse_json(response.content)
            except ValueError:
                data = None

            if data and "track" in data:
                detail = f"track={data['track']}, confidence={data.get('confidence','?')}"
            else:
                success = False
                error = "JSON 提取失败"
                detail = f"raw[:120]={response.content[:120]}"

        except Exception as e:
            success = False
            error = f"{type(e).__name__}: {str(e)[:150]}"

        results.append(TestResult(
            name=name, passed=success,
            duration_ms=(time.perf_counter() - t0) * 1000,
            error=error, detail=detail,
        ))

    return results


# ══════════════════════════════════════════════════════
# 测试 6: 原生 HTTP 调用（绕过 LangChain，直接测 API）
# ══════════════════════════════════════════════════════

async def test_raw_http_function_calling(model_name: str) -> list[TestResult]:
    """直接用 HTTP 请求测试 DashScope 原生 Function Calling 支持"""
    import aiohttp

    results = []
    user_input = "我想请3天年假"

    payload = {
        "model": model_name,
        "messages": [
            {"role": "system", "content": "你是企业AI服务台的路由器。分析用户输入，调用 RouterDecision 函数返回路由决策。"},
            {"role": "user", "content": user_input},
        ],
        "tools": [ROUTER_TOOL],
        "tool_choice": "auto",
        "temperature": 0,
    }

    name = f"原生HTTP FC | {model_name} | '{user_input}'"
    t0 = time.perf_counter()

    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{DASHSCOPE_BASE}/chat/completions",
                headers={
                    "Authorization": f"Bearer {API_KEY}",
                    "Content-Type": "application/json",
                },
                json=payload,
                timeout=aiohttp.ClientTimeout(total=30),
            ) as resp:
                body = await resp.json()

                if resp.status != 200:
                    results.append(TestResult(
                        name=name, passed=False,
                        duration_ms=(time.perf_counter() - t0) * 1000,
                        error=f"HTTP {resp.status}: {json.dumps(body, ensure_ascii=False)[:300]}",
                    ))
                else:
                    choice = body.get("choices", [{}])[0]
                    msg = choice.get("message", {})

                    # 检查 tool_calls
                    if msg.get("tool_calls"):
                        tc = msg["tool_calls"][0]
                        args_str = tc.get("function", {}).get("arguments", "{}")
                        if isinstance(args_str, str):
                            args = json.loads(args_str)
                        else:
                            args = args_str
                        results.append(TestResult(
                            name=name, passed=True,
                            duration_ms=(time.perf_counter() - t0) * 1000,
                            detail=f"[OK] tool_call: track={args.get('track')}, confidence={args.get('confidence')}",
                        ))
                    elif msg.get("content") and "{" in msg.get("content", ""):
                        results.append(TestResult(
                            name=name, passed=False,
                            duration_ms=(time.perf_counter() - t0) * 1000,
                            error="模型未调用 tool，返回了 JSON 文本（auto 模式下模型选择不调用函数）",
                            detail=f"content[:120]={msg['content'][:120]}",
                        ))
                    else:
                        results.append(TestResult(
                            name=name, passed=False,
                            duration_ms=(time.perf_counter() - t0) * 1000,
                            error="无 tool_calls，无 JSON",
                            detail=f"content[:120]={msg.get('content', '')[:120]}",
                        ))
    except Exception as e:
        results.append(TestResult(
            name=name, passed=False,
            duration_ms=(time.perf_counter() - t0) * 1000,
            error=f"{type(e).__name__}: {str(e)[:200]}",
        ))

    # 同时测一次 streaming 模式
    stream_payload = {**payload, "stream": True}
    stream_name = f"原生HTTP FC (stream) | {model_name} | '{user_input}'"
    t1 = time.perf_counter()

    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{DASHSCOPE_BASE}/chat/completions",
                headers={
                    "Authorization": f"Bearer {API_KEY}",
                    "Content-Type": "application/json",
                },
                json=stream_payload,
                timeout=aiohttp.ClientTimeout(total=30),
            ) as resp:
                if resp.status != 200:
                    results.append(TestResult(
                        name=stream_name, passed=False,
                        duration_ms=(time.perf_counter() - t1) * 1000,
                        error=f"HTTP {resp.status}",
                    ))
                else:
                    # 收集所有 SSE 事件
                    full_text = ""
                    tool_calls_in_stream = []
                    async for line in resp.content:
                        line_str = line.decode("utf-8").strip()
                        if line_str.startswith("data: "):
                            data_str = line_str[6:]
                            if data_str == "[DONE]":
                                break
                            try:
                                chunk = json.loads(data_str)
                                delta = chunk.get("choices", [{}])[0].get("delta", {})
                                if delta.get("content"):
                                    full_text += delta["content"]
                                if delta.get("tool_calls"):
                                    tool_calls_in_stream.extend(delta["tool_calls"])
                            except json.JSONDecodeError:
                                pass

                    if tool_calls_in_stream:
                        results.append(TestResult(
                            name=stream_name, passed=True,
                            duration_ms=(time.perf_counter() - t1) * 1000,
                            detail=f"[OK] stream tool_calls: {len(tool_calls_in_stream)} 个",
                        ))
                    elif full_text:
                        results.append(TestResult(
                            name=stream_name, passed=False,
                            duration_ms=(time.perf_counter() - t1) * 1000,
                            error="流式模式无 tool_calls",
                            detail=f"text[:100]={full_text[:100]}",
                        ))
                    else:
                        results.append(TestResult(
                            name=stream_name, passed=False,
                            duration_ms=(time.perf_counter() - t1) * 1000,
                            error="流式模式无输出",
                        ))
    except Exception as e:
        results.append(TestResult(
            name=stream_name, passed=False,
            duration_ms=(time.perf_counter() - t1) * 1000,
            error=f"{type(e).__name__}: {str(e)[:200]}",
        ))

    return results


# ══════════════════════════════════════════════════════
# 主流程
# ══════════════════════════════════════════════════════

async def main():
    print("=" * 70)
    print("DashScope Function Calling 健壮性验证")
    print(f"主模型: {MAIN_MODEL}    路由模型: {ROUTER_MODEL}")
    print(f"Base URL: {DASHSCOPE_BASE}")
    print("=" * 70)

    all_results: list[TestResult] = []

    # ── 每个测试函数在两模型上各跑一次 ──
    test_suites = [
        ("[1] 基础 Function Calling (bind_tools + auto)", test_bind_tools_auto),
        ("[2] with_structured_output (简单 Pydantic)", test_structured_output_simple),
        ("[3] with_structured_output (嵌套 Pydantic)", test_structured_output_nested),
        ("[4] tool_choice 兼容性矩阵", test_tool_choice_matrix),
        ("[5] 对照: 当前 prompt->JSON", test_prompt_to_json),
        ("[6] 原生 HTTP Function Calling", test_raw_http_function_calling),
    ]

    for label, test_fn in test_suites:
        print(f"\n{'─' * 70}")
        print(f"  {label}")
        print(f"{'─' * 70}")

        for model_name in [MAIN_MODEL, ROUTER_MODEL]:
            results = await test_fn(model_name)
            all_results.extend(results)

            for r in results:
                status = "PASS" if r.passed else "FAIL"
                ms = f"{r.duration_ms:.0f}ms"
                print(f"  [{status}] {r.name}")
                if r.error:
                    print(f"      ERR: {r.error}")
                if r.detail:
                    print(f"      MSG: {r.detail}")

    # ── 汇总 ──
    print(f"\n{'=' * 70}")
    print("汇总报告")
    print(f"{'=' * 70}")

    passed = [r for r in all_results if r.passed]
    failed = [r for r in all_results if not r.passed]

    print(f"  总计: {len(all_results)} 个测试")
    print(f"  通过: {len(passed)} [PASS]")
    print(f"  失败: {len(failed)} [FAIL]")

    if passed:
        avg_ms = sum(r.duration_ms for r in passed) / len(passed)
        print(f"  平均延迟: {avg_ms:.0f}ms (通过用例)")

    if failed:
        print(f"\n  失败用例明细:")
        for r in failed:
            print(f"    [FAIL] {r.name}")
            print(f"       {r.error[:150]}")

    # 按维度总结
    print(f"\n{'─' * 70}")
    print("能力矩阵总结")
    print(f"{'─' * 70}")

    dims = {
        "bind_tools (auto)": lambda r: "bind_tools(auto)" in r.name,
        "with_structured_output (simple)": lambda r: "structured(RouterDecision)" in r.name,
        "with_structured_output (nested)": lambda r: "structured(TicketParams)" in r.name,
        "tool_choice=required": lambda r: "tool_choice=required" in r.name,
        "tool_choice=object": lambda r: "tool_choice=object" in r.name,
        "prompt→JSON": lambda r: "prompt→JSON" in r.name,
        "原生HTTP FC": lambda r: "原生HTTP FC" in r.name,
        "原生HTTP FC (stream)": lambda r: "原生HTTP FC (stream)" in r.name,
    }

    for dim_name, filter_fn in dims.items():
        relevant = [r for r in all_results if filter_fn(r)]
        if not relevant:
            continue
        pct = sum(1 for r in relevant if r.passed) / len(relevant) * 100
        bar = "[GREEN]" if pct >= 80 else ("[YELLOW]" if pct >= 50 else "[RED]")
        models = set()
        for r in relevant:
            if MAIN_MODEL in r.name:
                models.add(MAIN_MODEL)
            if ROUTER_MODEL in r.name:
                models.add(ROUTER_MODEL)
        model_str = ", ".join(sorted(models))
        print(f"  {bar} {dim_name}: {pct:.0f}% ({sum(1 for r in relevant if r.passed)}/{len(relevant)}) — {model_str}")

    # 结论
    print(f"\n{'=' * 70}")
    print("结论与建议")
    print(f"{'=' * 70}")

    structured_pass = sum(1 for r in all_results if r.passed and "structured(" in r.name)
    structured_total = sum(1 for r in all_results if "structured(" in r.name)
    prompt_pass = sum(1 for r in all_results if r.passed and "prompt→JSON" in r.name)
    prompt_total = sum(1 for r in all_results if "prompt→JSON" in r.name)

    print(f"  with_structured_output 可用性: {structured_pass}/{structured_total}")
    print(f"  prompt→JSON 健壮性:          {prompt_pass}/{prompt_total}")

    if structured_pass == structured_total and prompt_pass == prompt_total:
        print("\n  [PASS] 两种方案均可工作，建议迁移到 with_structured_output（更低延迟、更可靠）")
    elif structured_pass > 0:
        print("\n  [WARN] with_structured_output 部分可用，建议逐步迁移")
        print("     先用简单 Pydantic，确认稳定后再迁移嵌套模型")
    else:
        print("\n  [FAIL] with_structured_output 完全不可用")
        print("     继续使用 prompt->JSON 方案（当前最稳定）")
        print("     或考虑：换用 DashScope 原生 SDK / 换用支持 FC 的模型")

    return all_results


if __name__ == "__main__":
    all_results = asyncio.run(main())

"""
Latency measurement script — measures LLM + Embedding latency directly

Usage:
    python scripts/latency_test.py              # Measure LLM + Embedding latency
    python scripts/latency_test.py --full        # Full end-to-end (needs server running)
"""

from __future__ import annotations
import asyncio
import time
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.settings import settings
from config.model_provider import create_chat_model, create_embedding_model


def sep(title: str):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


async def measure_llm(model_type: str, model_name: str, prompt: str, with_tools: bool = False):
    """Measure single LLM call latency"""
    llm = create_chat_model(model_type=model_type, temperature=0)

    if with_tools:
        tool_schema = {
            "type": "function",
            "function": {
                "name": "route_decision",
                "description": "classify user intent and return routing decision",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "track": {"type": "string", "enum": ["fast", "dynamic", "complex", "clarify"]},
                        "confidence": {"type": "number"},
                        "reason": {"type": "string"},
                    },
                    "required": ["track", "confidence", "reason"],
                },
            },
        }
        llm = llm.bind_tools([tool_schema], tool_choice="auto")

    t0 = time.time()
    try:
        response = await llm.ainvoke([{"role": "user", "content": prompt}])
        elapsed = time.time() - t0

        content_preview = (
            (response.content or "")[:80].replace("\n", " ")
            if hasattr(response, 'content') else ""
        )
        tool_info = ""
        if hasattr(response, 'tool_calls') and response.tool_calls:
            tool_info = f", tool_calls={len(response.tool_calls)}"

        usage = ""
        if hasattr(response, 'usage_metadata') and response.usage_metadata:
            u = response.usage_metadata
            usage = f", input={u.get('input_tokens','?')} output={u.get('output_tokens','?')}"

        print(f"  [OK] {model_name}: {elapsed:.2f}s | {content_preview[:60]}{tool_info}{usage}")
        return elapsed
    except Exception as e:
        elapsed = time.time() - t0
        print(f"  [FAIL] {model_name}: {elapsed:.2f}s -> {str(e)[:120]}")
        return None


async def measure_embedding(text: str):
    """Measure single embedding call latency"""
    t0 = time.time()
    try:
        embeddings = create_embedding_model()
        t_create = time.time()
        result = embeddings.embed_query(text)
        t_total = time.time()
        print(f"  [OK] Embedding: create={t_create - t0:.2f}s, API={t_total - t_create:.2f}s, total={t_total - t0:.2f}s")
        print(f"       dims={len(result)}, text='{text[:50]}'")
        return t_total - t0
    except Exception as e:
        elapsed = time.time() - t0
        print(f"  [FAIL] Embedding: {elapsed:.2f}s -> {str(e)[:120]}")
        return None


async def measure_end_to_end(port: int = 8001):
    """HTTP end-to-end latency test (requires running server)"""
    import httpx

    url = f"http://127.0.0.1:{port}/api/v3/chat"
    payload = {
        "user_input": "hello",
        "thread_id": f"lt_{int(time.time())}",
        "user_id": "test_user",
    }

    print(f"\n  Sending request to {url}...")
    t0 = time.time()
    first_tt = None
    token_count = 0

    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(120.0)) as client:
            async with client.stream("POST", url, json=payload) as resp:
                async for line in resp.aiter_lines():
                    if line.startswith("data: "):
                        data = line[6:]
                        token_count += 1
                        now = time.time()
                        if first_tt is None:
                            first_tt = now
                            print(f"  First byte: {now - t0:.2f}s -> {data[:60]}")
                        if data.startswith("[DONE]"):
                            break

        total = time.time() - t0
        ttft = (first_tt - t0) if first_tt else 0
        print(f"\n  Total: {total:.2f}s | TTFT: {ttft:.2f}s | events: {token_count}")
        return total
    except Exception as e:
        print(f"  [FAIL] {e}")
        return None


async def main():
    sep("LATENCY DIAGNOSTIC TOOL")

    # Show config
    print(f"\n  Current config:")
    print(f"    LLM_MODEL       = {settings.llm_model}")
    print(f"    ROUTER_MODEL     = {settings.router_model}")
    print(f"    EMBEDDING_MODEL  = {settings.embedding_model}")
    print(f"    LLM_BASE_URL     = {settings.llm_base_url}")
    api_key = settings.dashscope_api_key or ""
    key_info = f"{api_key[:12]}...{api_key[-4:]}" if len(api_key) > 16 else api_key[:20]
    print(f"    API_KEY          = {key_info}")

    if "your_dashscope_api_key_here" in api_key or len(api_key) < 10:
        print(f"\n  *** WARNING: API Key not configured! Set DASHSCOPE_API_KEY in .env ***")
        return

    # ── 1. Raw LLM latency ──
    sep("1. Raw LLM Call Latency")

    short_prompt = "Reply in one sentence: what is SLA?"

    print(f"\n  [Model: {settings.llm_model}]")
    await measure_llm("main", settings.llm_model, short_prompt)

    print(f"\n  [Model: {settings.router_model}]")
    await measure_llm("router", settings.router_model, short_prompt)

    # Compare with faster models
    print(f"\n  [Compare: qwen-turbo-latest]")
    try:
        from langchain_openai import ChatOpenAI
        from pydantic import SecretStr
        turbo_llm = ChatOpenAI(
            model="qwen-turbo-latest",
            api_key=SecretStr(settings.dashscope_api_key or ""),
            base_url=settings.llm_base_url,
            temperature=0,
        )
        t0 = time.time()
        resp = await turbo_llm.ainvoke([{"role": "user", "content": short_prompt}])
        print(f"  [OK] qwen-turbo-latest: {time.time() - t0:.2f}s | {(resp.content or '')[:60]}")
    except Exception as e:
        print(f"  [FAIL] qwen-turbo-latest: {str(e)[:120]}")

    # ── 2. Router-style (with tools) ──
    sep("2. Router-style Call (bind_tools + Function Calling)")

    router_prompt = "help me check mouse inventory"
    await measure_llm("main", settings.llm_model, router_prompt, with_tools=True)
    await measure_llm("router", settings.router_model, router_prompt, with_tools=True)

    # ── 3. Embedding ──
    sep("3. Embedding Call Latency")
    await measure_embedding("help me check mouse inventory")

    # ── 4. Simulated pipeline ──
    sep("4. Simulated Full Pipeline (Router -> Embedding -> Agent)")

    t_total = time.time()

    print(f"\n  [1/3] Router LLM ({settings.llm_model})...")
    t1 = await measure_llm("main", settings.llm_model, router_prompt, with_tools=True)
    t1 = t1 or 30

    print(f"\n  [2/3] Embedding API...")
    t2 = await measure_embedding(router_prompt)
    t2 = t2 or 0.5

    print(f"\n  [3/3] Agent LLM ({settings.llm_model})...")
    t3 = await measure_llm("main", settings.llm_model,
        "Based on: mouse inventory has 12 units. User asks: do we have mice? Answer concisely.")
    t3 = t3 or 30

    pipeline = time.time() - t_total
    print(f"\n  *** Pipeline total: {pipeline:.2f}s = Router({t1:.1f}s) + Embed({t2:.1f}s) + Agent({t3:.1f}s)")

    # ── 5. End-to-end ──
    if "--full" in sys.argv:
        sep("5. End-to-End HTTP Test")
        await measure_end_to_end()

    # ── Summary ──
    sep("DIAGNOSTIC SUMMARY")

    issues = []
    fixes = []

    if t1 and t1 > 10:
        issues.append(f"Router LLM call: {t1:.0f}s — model {settings.llm_model} is too slow")
        fixes.append("Set ROUTER_MODEL=qwen-turbo-latest (estimated 0.5-2s)")

    if t3 and t3 > 10:
        issues.append(f"Agent LLM call: {t3:.0f}s — model {settings.llm_model} is too slow")
        fixes.append("Set LLM_MODEL=qwen-plus (1-5s) or qwen-turbo-latest (0.5-2s)")

    if pipeline > 20:
        issues.append(f"Simulated pipeline: {pipeline:.0f}s — far beyond acceptable (<10s)")

    if "max" in settings.llm_model.lower() or "max" in settings.router_model.lower():
        issues.append(f"*** Using -max model ({settings.llm_model})! These are reasoning models, extremely slow! ***")
        fixes.append("*** URGENT: Change LLM_MODEL from max-series to qwen-plus or qwen-turbo-latest ***")

    if issues:
        print("\n  [ISSUES FOUND]:")
        for i in issues:
            print(f"    - {i}")
    else:
        print("\n  [OK] No major issues detected")

    if fixes:
        print("\n  [RECOMMENDED FIXES]:")
        for f in fixes:
            print(f"    -> {f}")

    print(f"\n  Recommended .env config:")
    print(f"    LLM_MODEL=qwen-plus")
    print(f"    ROUTER_MODEL=qwen-turbo-latest")
    print(f"    (current: LLM_MODEL={settings.llm_model}, ROUTER_MODEL={settings.router_model})")


if __name__ == "__main__":
    asyncio.run(main())

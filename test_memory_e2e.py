"""
E2E memory test -- validates 3 optimizations with real LLM calls
"""
import asyncio
import os
import sys
import time

THREAD_ID = "e2e_mem_test"
TEST_USER = "ZhangSan"

# 6-turn conversation covering 2 topics
TURNS = [
    "Hello, what's the annual leave policy?",
    "I've worked here 8 years, how many days do I get?",
    "What's the approval process for 3 days leave?",
    "What if my leave request gets rejected?",
    "OK thanks. Also, what equipment do new backend engineers get?",
    "What's the purchasing process if I need a better monitor?",
]


async def main():
    print("=" * 60)
    print("[E2E] Memory System Test -- 6-turn conversation")
    print("=" * 60)

    from agents.graph_workflow import OrchestrationWorkflowRunner

    runner = OrchestrationWorkflowRunner()
    await runner.reset(THREAD_ID)

    for i, msg in enumerate(TURNS):
        print(f"\n--- Turn {i+1} ---")
        print(f"[USER] {msg}")

        t0 = time.perf_counter()

        result = await runner.run(
            msg, thread_id=THREAD_ID,
            user_name=TEST_USER, role="employee",
        )

        t1 = time.perf_counter()

        resp = result.get("final_response", "")[:120]
        phase = result.get("conversation_phase", "")
        summary = result.get("conversation_summary", "")
        msg_count = len(result.get("messages", []))

        print(f"[BOT] {resp}...")
        print(f"      {t1-t0:.1f}s | phase={phase} | msgs={msg_count} | summary={len(summary)}c")

        if summary:
            print(f"      summary: {summary[:200]}...")

        await asyncio.sleep(0.3)

    # === Validation ===
    print("\n" + "=" * 60)
    print("VALIDATION")
    print("=" * 60)

    all_ok = True

    # 1. FAISS index on disk
    from services.knowledge_service import FAISS_INDEX_PATH
    if os.path.exists(FAISS_INDEX_PATH):
        import faiss
        idx = faiss.read_index(FAISS_INDEX_PATH)
        size = os.path.getsize(FAISS_INDEX_PATH)
        print(f"[PASS] FAISS index: {idx.ntotal} vectors on disk ({size} bytes)")
    else:
        print("[FAIL] FAISS index: file not found")
        all_ok = False

    # 2. Checkpoints in DB
    import sqlite3
    conn = sqlite3.connect("data/checkpoints.db")
    cp_count = conn.execute(
        "SELECT COUNT(*) FROM checkpoints WHERE thread_id = ?", (THREAD_ID,)
    ).fetchone()[0]
    conn.close()
    ok = cp_count > 0
    print(f"[{'PASS' if ok else 'FAIL'}] Checkpoints: {cp_count} rows for '{THREAD_ID}'")
    if not ok:
        all_ok = False

    # 3. State recovery (simulate restart)
    runner2 = OrchestrationWorkflowRunner()
    recovered = await runner2.get_state(THREAD_ID)
    if recovered and recovered.values:
        msgs = len(recovered.values.get("messages", []))
        summary = recovered.values.get("conversation_summary", "")
        print(f"[PASS] Recovery: {msgs} msgs, summary={len(summary)}c")
    else:
        print("[FAIL] Recovery: no state")
        all_ok = False

    # 4. Summary generation
    final_state = await runner.get_state(THREAD_ID)
    summary = final_state.values.get("conversation_summary", "")
    total_msgs = len(final_state.values.get("messages", []))
    if total_msgs > 10 and len(summary) > 0:
        print(f"[PASS] Summary: {len(summary)}c (threshold=10, actual={total_msgs})")
    elif total_msgs <= 10:
        print(f"[INFO] Summary: below threshold ({total_msgs} <= 10)")
    else:
        print(f"[WARN] Summary: empty ({total_msgs} msgs)")
        all_ok = False

    # Cleanup
    await runner.reset(THREAD_ID)
    print("\nCleaned up.")

    if all_ok:
        print("\n[RESULT] ALL CHECKS PASSED")
    else:
        print("\n[RESULT] SOME CHECKS FAILED")

    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))

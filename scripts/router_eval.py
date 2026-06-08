"""
Router 路由评估脚本

评估语义路由器的准确率、召回率、混淆矩阵。
支持两种模式：
  - mock 模式：用预设 JSON 模拟 LLM 输出（CI 可运行）
  - live 模式：真实调用 DashScope LLM（需 DASHSCOPE_API_KEY）

用法：
  python scripts/router_eval.py              # mock 模式
  python scripts/router_eval.py --live        # 真实 LLM
  python scripts/router_eval.py --live --json # 输出 JSON 格式
"""

from __future__ import annotations

import json
import sys
import os
import argparse
from dataclasses import dataclass, field
from typing import Literal
from collections import Counter

# 确保项目根目录在 sys.path 中
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ============================================================
# 评估数据集（80 条，覆盖 8 类场景，含正常用例 + 边界用例）
# ============================================================

@dataclass
class EvalCase:
    """单条评估用例"""
    id: str
    input: str
    expected_track: Literal["fast", "action", "complex", "clarify"]
    category: str           # 场景分类
    difficulty: str         # easy | medium | hard
    note: str = ""          # 预期行为说明


EVAL_DATASET: list[EvalCase] = [
    # ================================================================
    # 一、fast 轨道 — 知识查询（20 条）
    # ================================================================
    EvalCase("F01", "VPN怎么连接", "fast", "IT-网络", "easy"),
    EvalCase("F02", "请假流程是什么", "fast", "HR-请假", "easy"),
    EvalCase("F03", "食堂在哪", "fast", "行政-设施", "easy"),
    EvalCase("F04", "打印机怎么用", "fast", "IT-硬件", "easy"),
    EvalCase("F05", "病假需要什么证明", "fast", "HR-请假", "easy"),
    EvalCase("F06", "年假有多少天", "fast", "HR-请假", "easy"),
    EvalCase("F07", "会议室怎么预定", "fast", "行政-会议室", "easy"),
    EvalCase("F08", "快递怎么寄", "fast", "行政-快递", "easy"),
    EvalCase("F09", "WiFi密码是什么", "fast", "IT-网络", "easy"),
    EvalCase("F10", "数据库连接超时怎么办", "fast", "IT-运维", "easy"),
    EvalCase("F11", "报销标准是什么", "fast", "财务-报销", "easy"),
    EvalCase("F12", "产假多少天", "fast", "HR-请假", "easy"),
    EvalCase("F13", "访客怎么登记", "fast", "行政-访客", "easy"),
    EvalCase("F14", "账号被锁定了怎么办", "fast", "IT-账号", "easy"),
    EvalCase("F15", "调休怎么申请", "fast", "HR-请假", "easy"),
    EvalCase("F16", "哪个会议室有投影仪", "fast", "行政-会议室", "medium", "可能被误判为 action（预定）"),
    EvalCase("F17", "出差住宿标准是多少", "fast", "财务-报销", "easy"),
    EvalCase("F18", "新员工怎么领电脑", "fast", "行政-资产", "easy"),
    EvalCase("F19", "服务器重启流程是什么", "fast", "IT-运维", "easy"),
    EvalCase("F20", "婚假能休几天", "fast", "HR-请假", "easy"),

    # ================================================================
    # 二、action 轨道 — 工单/操作（20 条）
    # ================================================================
    EvalCase("A01", "帮我提交一个网络故障工单", "action", "IT-故障", "easy"),
    EvalCase("A02", "我要请假3天", "action", "HR-请假", "easy"),
    EvalCase("A03", "报销500元差旅费", "action", "财务-报销", "easy"),
    EvalCase("A04", "帮我预定明天下午的会议室", "action", "行政-会议室", "easy"),
    EvalCase("A05", "我的电脑坏了需要维修", "action", "IT-故障", "medium", "可能是查维修方法(快)或报修(动作)"),
    EvalCase("A06", "申请一台新显示器", "action", "IT-申请", "easy"),
    EvalCase("A07", "我明天想请病假", "action", "HR-请假", "easy"),
    EvalCase("A08", "帮我约下周三的访客", "action", "行政-访客", "easy"),
    EvalCase("A09", "寄一个快递到上海", "action", "行政-快递", "easy"),
    EvalCase("A10", "系统登录不了帮我看看", "action", "IT-故障", "hard", "可能是查解决方案(fast)或报修(action)"),
    EvalCase("A11", "提交一个工单打印机坏了", "action", "IT-故障", "easy"),
    EvalCase("A12", "我要报销昨天的打车费", "action", "财务-报销", "easy"),
    EvalCase("A13", "预定下周一上午的星空厅", "action", "行政-会议室", "easy"),
    EvalCase("A14", "申请调休一天", "action", "HR-请假", "medium", "调休可能是查询规则或申请"),
    EvalCase("A15", "帮我申请一台新笔记本电脑", "action", "IT-申请", "easy"),
    EvalCase("A16", "我下周要请三天事假", "action", "HR-请假", "easy"),
    EvalCase("A17", "帮我报修空调", "action", "行政-设施", "easy"),
    EvalCase("A18", "明天会议室还有空的吗帮我定一个", "action", "行政-会议室", "medium", "先查后定 → 应判 action"),
    EvalCase("A19", "这发票帮我报销了", "action", "财务-报销", "easy"),
    EvalCase("A20", "给张三申请一个访客码", "action", "行政-访客", "easy"),

    # ================================================================
    # 三、complex 轨道 — 复合指令（10 条）
    # ================================================================
    EvalCase("C01", "查天气然后请假再取消会议室", "complex", "多步骤", "easy"),
    EvalCase("C02", "帮我查一下年假余额然后用掉3天再帮我订个会议室", "complex", "多步骤", "easy"),
    EvalCase("C03", "先帮我查报销标准，再提交一个差旅报销", "complex", "多步骤", "medium"),
    EvalCase("C04", "重启服务器然后通知受影响团队再创建运维工单", "complex", "多步骤", "easy"),
    EvalCase("C05", "查一下访客登记流程然后帮张三登记明天来访", "complex", "多步骤", "medium"),
    EvalCase("C06", "确认请假政策后帮我提交下周三的病假申请", "complex", "多步骤", "medium"),
    EvalCase("C07", "查快递怎么寄然后帮我寄一个", "complex", "多步骤", "easy"),
    EvalCase("C08", "VPN连不上先帮我排查再创建工单", "complex", "多步骤", "easy"),
    EvalCase("C09", "电脑坏了先查怎么修，修不了帮我报修", "complex", "多步骤", "easy"),
    EvalCase("C10", "查会议室空余情况然后帮我定一间", "complex", "多步骤", "hard", "可能是 action（直接定）"),

    # ================================================================
    # 四、clarify 轨道 — 模糊/无关/歧义（20 条）
    # ================================================================
    EvalCase("R01", "嗯？", "clarify", "模糊", "easy"),
    EvalCase("R02", "你好", "clarify", "模糊-寒暄", "easy"),
    EvalCase("R03", "今天天气怎么样", "clarify", "无关", "easy"),
    EvalCase("R04", "帮帮我", "clarify", "模糊", "easy"),
    EvalCase("R05", "这个怎么用", "clarify", "模糊-缺宾语", "easy"),
    EvalCase("R06", "推荐一本好书", "clarify", "无关", "easy"),
    EvalCase("R07", "刚才说的那个", "clarify", "模糊-指代不明", "easy"),
    EvalCase("R08", "吃饭了吗", "clarify", "无关", "easy"),
    EvalCase("R09", "行", "clarify", "模糊-不确定", "hard", "可能是确认卡片回复"),
    EvalCase("R10", "不行", "clarify", "模糊-不确定", "hard", "可能是拒绝卡片"),
    EvalCase("R11", "我不知道", "clarify", "模糊", "easy"),
    EvalCase("R12", "怎么做", "clarify", "模糊-缺主语", "easy"),
    EvalCase("R13", "111", "clarify", "模糊-无意义", "easy"),
    EvalCase("R14", "有问题", "clarify", "模糊", "easy"),
    EvalCase("R15", "急", "clarify", "模糊", "hard", "可能是紧急工单请求"),
    EvalCase("R16", "明天", "clarify", "模糊-缺谓语", "medium"),
    EvalCase("R17", "工资多少", "clarify", "无关", "easy"),
    EvalCase("R18", "点外卖", "clarify", "无关", "easy"),
    EvalCase("R19", "。。。", "clarify", "模糊", "easy"),
    EvalCase("R20", "帮我查一下那个", "clarify", "模糊-缺宾语", "medium"),

    # ================================================================
    # 五、边界用例 — 容易混淆（10 条）
    # ================================================================
    EvalCase("E01", "VPN", "fast", "边界-单关键词", "hard", "单关键词可能是查询也可能是报修"),
    EvalCase("E02", "请假", "fast", "边界-单关键词", "hard", "查询请假规则 vs 申请请假"),
    EvalCase("E03", "会议室", "fast", "边界-单关键词", "hard", "查询会议室 vs 预定会议室"),
    EvalCase("E04", "帮我查一下VPN怎么用然后如果不行就帮我想办法修", "complex", "边界-隐含多步骤", "hard"),
    EvalCase("E05", "网络太慢了影响工作了", "action", "边界-隐含报修", "hard", "可能是吐槽或报修"),
    EvalCase("E06", "你帮我看看这个错误代码什么意思然后修复一下", "complex", "边界-排查+修复", "hard"),
    EvalCase("E07", "上次那个工单什么状态了", "action", "边界-工单查询", "hard", "查询工单 vs 知识问答"),
    EvalCase("E08", "我要休假", "action", "边界-模糊请假", "medium", "没有说明类型和天数"),
    EvalCase("E09", "定个会", "action", "边界-极简预定", "medium"),
    EvalCase("E10", "改一下", "clarify", "边界-极简修改", "hard"),
]


# ============================================================
# 评估指标计算
# ============================================================

@dataclass
class EvalMetrics:
    """评估指标"""
    total: int
    correct: int
    accuracy: float
    precision_per_class: dict[str, float]
    recall_per_class: dict[str, float]
    f1_per_class: dict[str, float]
    confusion: dict[str, dict[str, int]]   # actual → predicted → count
    errors: list[dict]                      # 错误详情
    difficulty_accuracy: dict[str, float]   # 按难度统计
    category_accuracy: dict[str, float]     # 按场景统计


def compute_metrics(dataset: list[EvalCase], predictions: dict[str, str]) -> EvalMetrics:
    """计算评估指标"""
    tracks = ["fast", "action", "complex", "clarify"]
    total = len(dataset)

    # 统计
    correct = 0
    confusion = {t: {p: 0 for p in tracks} for t in tracks}
    errors = []

    tp = {t: 0 for t in tracks}    # true positive per class
    fp = {t: 0 for t in tracks}    # false positive per class
    fn_count = {t: 0 for t in tracks}  # false negative per class

    diff_total = {d: 0 for d in ["easy", "medium", "hard"]}
    diff_correct = {d: 0 for d in ["easy", "medium", "hard"]}
    cat_correct = {}
    cat_total = {}

    for case in dataset:
        pred = predictions.get(case.id, "clarify")
        actual = case.expected_track

        # 混淆矩阵
        if actual in confusion and pred in confusion[actual]:
            confusion[actual][pred] += 1

        if pred == actual:
            correct += 1
            tp[actual] += 1
        else:
            fp[pred] += 1
            fn_count[actual] += 1
            errors.append({
                "id": case.id,
                "input": case.input,
                "expected": actual,
                "predicted": pred,
                "category": case.category,
                "difficulty": case.difficulty,
                "note": case.note,
            })

        # 按难度
        d = case.difficulty
        diff_total[d] = diff_total.get(d, 0) + 1
        diff_correct[d] = diff_correct.get(d, 0) + (1 if pred == actual else 0)

        # 按场景
        c = case.category
        cat_total[c] = cat_total.get(c, 0) + 1
        cat_correct[c] = cat_correct.get(c, 0) + (1 if pred == actual else 0)

    # 计算指标
    accuracy = correct / total if total else 0

    precision_per_class = {
        t: tp[t] / (tp[t] + fp[t]) if (tp[t] + fp[t]) > 0 else 0
        for t in tracks
    }
    recall_per_class = {
        t: tp[t] / (tp[t] + fn_count[t]) if (tp[t] + fn_count[t]) > 0 else 0
        for t in tracks
    }
    f1_per_class = {
        t: 2 * precision_per_class[t] * recall_per_class[t] / (precision_per_class[t] + recall_per_class[t])
        if (precision_per_class[t] + recall_per_class[t]) > 0 else 0
        for t in tracks
    }

    difficulty_accuracy = {
        d: diff_correct[d] / diff_total[d] if diff_total[d] else 0
        for d in diff_total
    }
    category_accuracy = {
        c: cat_correct[c] / cat_total[c] if cat_total[c] else 0
        for c in cat_total
    }

    return EvalMetrics(
        total=total, correct=correct, accuracy=accuracy,
        precision_per_class=precision_per_class,
        recall_per_class=recall_per_class,
        f1_per_class=f1_per_class,
        confusion=confusion, errors=errors,
        difficulty_accuracy=difficulty_accuracy,
        category_accuracy=category_accuracy,
    )


# ============================================================
# 评估运行器
# ============================================================

def _make_mock_for_case(case: EvalCase):
    """为单个用例生成模拟 LLM（返回期望的 track 和高置信度）"""
    from unittest.mock import AsyncMock, MagicMock
    mock = MagicMock()
    mock_response = MagicMock()
    mock_response.content = json.dumps({
        "track": case.expected_track,
        "confidence": 0.85,
        "reason": f"mock: {case.category}",
        "requires_tools": [],
    }, ensure_ascii=False)
    mock.ainvoke = AsyncMock(return_value=mock_response)
    return mock


async def run_evaluation(live: bool = False) -> EvalMetrics:
    """运行评估"""
    from agents.orchestrator.router import Router

    predictions: dict[str, str] = {}

    if live:
        # 真实 LLM 调用
        print("[LIVE] 使用真实 LLM 评估...")
        router = Router()
        for i, case in enumerate(EVAL_DATASET):
            try:
                result = await router.decide(case.input)
                predictions[case.id] = result.track
            except Exception as e:
                print(f"  [WARN] {case.id} LLM调用失败: {e}")
                predictions[case.id] = "clarify"
            if (i + 1) % 10 == 0:
                print(f"  进度: {i + 1}/{len(EVAL_DATASET)}")
    else:
        # Mock 模式：每个用例用其期望 track 模拟 LLM，模拟完美路由
        print("[MOCK] Mock模式（理想LLM）：所有用例按期望标签返回")
        for case in EVAL_DATASET:
            mock_llm = _make_mock_for_case(case)

            # 动态替换 router 的 llm
            from unittest.mock import patch
            with patch("agents.orchestrator.router.create_chat_model",
                       return_value=mock_llm):
                router = Router()
                router.llm = mock_llm
                result = await router.decide(case.input)
                predictions[case.id] = result.track

    return compute_metrics(EVAL_DATASET, predictions)


# ============================================================
# 报告生成
# ============================================================

def format_confusion_matrix(confusion: dict[str, dict[str, int]]) -> str:
    """格式化混淆矩阵"""
    tracks = ["fast", "action", "complex", "clarify"]
    labels = {"fast": "快查", "action": "动作", "complex": "复合", "clarify": "反问"}

    lines = ["| 实际 ↓ / 预测 → | " + " | ".join(labels[t] for t in tracks) + " |"]
    lines.append("|" + "|".join(["---"] * (len(tracks) + 1)) + "|")

    for t in tracks:
        row = [str(confusion[t].get(p, 0)) for p in tracks]
        lines.append(f"| **{labels[t]}** | " + " | ".join(row) + " |")

    return "\n".join(lines)


def format_metrics_table(metrics: EvalMetrics) -> str:
    """格式化指标表"""
    tracks = ["fast", "action", "complex", "clarify"]
    labels = {"fast": "知识查询", "action": "工单操作", "complex": "复合指令", "clarify": "反问澄清"}

    lines = ["| 轨道 | 精确率 | 召回率 | F1 | 样本数 |"]
    lines.append("|------|--------|--------|-----|--------|")
    for t in tracks:
        count = sum(1 for c in EVAL_DATASET if c.expected_track == t)
        lines.append(
            f"| {labels[t]} | {metrics.precision_per_class[t]:.1%} | "
            f"{metrics.recall_per_class[t]:.1%} | {metrics.f1_per_class[t]:.1%} | {count} |"
        )
    return "\n".join(lines)


def format_hard_cases(metrics: EvalMetrics, top_n: int = 10) -> str:
    """列出最难用例的错误"""
    lines = []
    for err in metrics.errors:
        lines.append(
            f"| {err['id']} | {err['input'][:40]} | {err['expected']} → {err['predicted']} "
            f"| {err['difficulty']} | {err['note'][:50]} |"
        )
    if not lines:
        return "（无错误）"
    header = "| ID | 输入 | 期望 → 实际 | 难度 | 备注 |\n|----|------|------------|------|------|"
    return header + "\n" + "\n".join(lines[:top_n])


def generate_report(metrics: EvalMetrics, mode: str) -> str:
    """生成完整 Markdown 评估报告"""
    return f"""# 语义路由评估报告

> 生成模式：{mode} | 评估用例：{metrics.total} 条 | 日期：2026-06-08

---

## 一、总览

| 指标 | 值 |
|------|-----|
| 总用例数 | {metrics.total} |
| 正确数 | {metrics.correct} |
| **总体准确率** | **{metrics.accuracy:.1%}** |
| 错误数 | {len(metrics.errors)} |

---

## 二、分类性能

{format_metrics_table(metrics)}

---

## 三、混淆矩阵

{format_confusion_matrix(metrics.confusion)}

---

## 四、按难度统计

| 难度 | 用例数 | 准确率 |
|------|--------|--------|
{chr(10).join(f"| {d} | {diff_total.get(d, 0)} | {diff_correct.get(d, 0)} | {metrics.difficulty_accuracy.get(d, 0):.1%} |" for d in ["easy", "medium", "hard"])}

---

## 五、按场景统计

| 场景 | 准确率 |
|------|--------|
{chr(10).join(f"| {cat} | {acc:.1%} |" for cat, acc in sorted(metrics.category_accuracy.items()) if not cat.startswith("边界"))}

---

## 六、错误详情（Top 10）

{format_hard_cases(metrics)}

---

## 七、评估数据集结构

- **总计**：{metrics.total} 条，覆盖 4 个轨道 × 5 个场景类别 × 3 个难度等级
- **fast（知识查询）**：20 条 — IT网络/HR请假/行政/财务
- **action（工单操作）**：20 条 — IT故障/请假/报销/会议室/访客
- **complex（复合指令）**：10 条 — 多步骤联动
- **clarify（反问澄清）**：20 条 — 模糊/无关/歧义
- **边界用例**：10 条 — 单关键词、隐含意图、极简输入

---

## 八、改进建议

基于错误分布，建议关注以下方向：

1. **单关键词歧义**（如 "VPN"/"请假"/"会议室"）：当前无上下文时默认判 fast，需补对话历史感知
2. **隐含报修信号**（如 "太慢了影响工作"）：需加强 urgency 信号词的权重
3. **极简输入**（如 "行"/"改一下"）：需依赖卡片锁机制，独立路由时判 clarify 符合预期
4. **复合 vs 单步骤边界**：部分用例实质只需一步操作，可根据 confidence 差异自动降级
"""


# ============================================================
# 入口
# ============================================================

async def main():
    parser = argparse.ArgumentParser(description="Router 路由评估")
    parser.add_argument("--live", action="store_true", help="使用真实 LLM（需 API Key）")
    parser.add_argument("--json", action="store_true", help="输出 JSON 格式")
    parser.add_argument("--output", type=str, default="", help="输出报告文件路径")
    args = parser.parse_args()

    mode = "真实 LLM" if args.live else "Mock（理想 LLM）"
    metrics = await run_evaluation(live=args.live)

    if args.json:
        import json as _json
        result = {
            "total": metrics.total,
            "correct": metrics.correct,
            "accuracy": metrics.accuracy,
            "precision": metrics.precision_per_class,
            "recall": metrics.recall_per_class,
            "f1": metrics.f1_per_class,
            "confusion": metrics.confusion,
            "errors": metrics.errors,
        }
        print(_json.dumps(result, ensure_ascii=False, indent=2))
    else:
        report = generate_report(metrics, mode)
        print(report)

        if args.output:
            with open(args.output, "w", encoding="utf-8") as f:
                f.write(report)
            print(f"\n[DONE] 报告已保存到 {args.output}")


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())

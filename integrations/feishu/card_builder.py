"""
飞书消息卡片构建器

封装飞书 Card JSON 格式的构建逻辑，提供语义化的模板方法。

飞书卡片文档:
  https://open.feishu.cn/document/uAjLw4CM/ukzMukzMukzM/feishu-cards/card-components

支持的标签:
  - plain_text: 纯文本
  - lark_md: 飞书 Markdown 子集（粗体、斜体、链接、列表、代码块）
  - div, hr, action (按钮组), note (脚注)

卡片模板颜色:
  - blue: 信息/进行中
  - green: 成功/完成
  - red: 错误/警告
  - yellow: 需确认/等待
"""

from __future__ import annotations

import json
import time
from typing import Optional


class CardBuilder:
    """
    飞书消息卡片构建器

    所有静态方法返回 dict（卡片 JSON 结构），调用者通过
    to_message_content() 转为 Feishu API 的 content 字段。

    使用示例:
        card = CardBuilder.thinking_card()
        content = CardBuilder.to_message_content(card)
        # 发送: CreateMessage(..., msg_type="interactive", content=content)
    """

    # ============================================================
    # 卡片模板
    # ============================================================

    @staticmethod
    def thinking_card() -> dict:
        """创建「正在思考」初始卡片 — 蓝色模板 + loading 动画"""
        return {
            "config": {"wide_screen_mode": True},
            "header": {
                "title": {"tag": "plain_text", "content": "🤔 正在分析中..."},
                "template": "blue",
            },
            "elements": [
                {
                    "tag": "div",
                    "text": {
                        "tag": "lark_md",
                        "content": "**已收到您的消息，AI 正在分析处理...**\n\n请稍候，这通常需要几秒钟。",
                    },
                },
                {
                    "tag": "note",
                    "elements": [
                        {
                            "tag": "plain_text",
                            "content": f"🕐 {CardBuilder._time_str()}",
                        }
                    ],
                },
            ],
        }

    @staticmethod
    def progress_card(
        header_text: str,
        steps: list[dict],
        header_template: str = "blue",
    ) -> dict:
        """
        创建「进度中」卡片

        Args:
            header_text: 卡片标题（如 "📋 正在规划处理方案..."）
            steps: 步骤列表，每项 {"label": "步骤名", "status": "done|running|pending", "detail": "详情(可选)"}
            header_template: 卡片颜色模板
        """
        elements = []

        for i, step in enumerate(steps):
            label = step.get("label", f"步骤 {i+1}")
            status = step.get("status", "pending")
            detail = step.get("detail", "")

            # 步骤图标
            icons = {"done": "✅", "running": "⏳", "pending": "⬜"}
            icon = icons.get(status, "⬜")

            line = f"{icon} **{label}**"
            if detail:
                line += f"\n  _{detail}_"

            elements.append({
                "tag": "div",
                "text": {"tag": "lark_md", "content": line},
            })

        # 脚注
        elements.append({"tag": "hr"})
        elements.append({
            "tag": "note",
            "elements": [
                {
                    "tag": "plain_text",
                    "content": f"🕐 {CardBuilder._time_str()}  ·  AI 编排引擎运行中",
                }
            ],
        })

        return {
            "config": {"wide_screen_mode": True},
            "header": {
                "title": {"tag": "plain_text", "content": header_text},
                "template": header_template,
            },
            "elements": elements,
        }

    @staticmethod
    def result_card(
        title: str,
        content_md: str,
        actions: list[dict] = None,
        header_template: str = "green",
        note: str = None,
    ) -> dict:
        """
        创建「结果」卡片

        Args:
            title: 卡片标题（如 "✅ 处理完成"）
            content_md: 正文（飞书 Markdown 格式）
            actions: 可选按钮列表，每项 {"text": "确认", "type": "primary|danger|default",
                      "value": {"action": "xxx", ...}}
            header_template: 标题颜色模板
            note: 可选脚注文字
        """
        elements = []

        # 正文区域
        elements.append({
            "tag": "div",
            "text": {"tag": "lark_md", "content": content_md},
        })

        # 操作按钮
        if actions:
            elements.append({"tag": "hr"})
            buttons = []
            for a in actions:
                buttons.append({
                    "tag": "button",
                    "text": {
                        "tag": "plain_text",
                        "content": a.get("text", "确认"),
                    },
                    "type": a.get("type", "primary"),
                    "value": a.get("value", {}),
                })
            elements.append({
                "tag": "action",
                "actions": buttons,
                "layout": "bisected",
            })

        # 脚注
        note_text = note or f"🕐 {CardBuilder._time_str()}  ·  企业员工AI服务台"
        elements.append({"tag": "hr"})
        elements.append({
            "tag": "note",
            "elements": [
                {"tag": "plain_text", "content": note_text},
            ],
        })

        return {
            "config": {"wide_screen_mode": True},
            "header": {
                "title": {"tag": "plain_text", "content": title},
                "template": header_template,
            },
            "elements": elements,
        }

    @staticmethod
    def confirm_card(
        title: str,
        description_md: str,
        fields: list[dict] = None,
        confirm_value: dict = None,
        cancel_value: dict = None,
    ) -> dict:
        """
        创建「确认」卡片 — 人工审核用

        Args:
            title: 卡片标题（如 "⚠️ 需要人工审核"）
            description_md: 审核内容描述（Markdown）
            fields: 结构化字段列表 [{"label": "意图", "value": "IT工单提交"}, ...]
            confirm_value: 确认按钮的 value 数据 {"action": "confirm_ticket", ...}
            cancel_value: 取消按钮的 value 数据 {"action": "reject_ticket", ...}
        """
        elements = []

        # 描述
        elements.append({
            "tag": "div",
            "text": {"tag": "lark_md", "content": description_md},
        })

        # 结构化字段
        if fields:
            elements.append({"tag": "hr"})
            for f in fields:
                elements.append({
                    "tag": "div",
                    "fields": [
                        {
                            "tag": "plain_text",
                            "content": f"**{f.get('label', '')}**",
                            "is_short": True,
                        },
                        {
                            "tag": "plain_text",
                            "content": f.get("value", ""),
                            "is_short": True,
                        },
                    ],
                })

        # 确认/取消按钮
        elements.append({"tag": "hr"})
        elements.append({
            "tag": "action",
            "actions": [
                {
                    "tag": "button",
                    "text": {"tag": "plain_text", "content": "✅ 确认"},
                    "type": "primary",
                    "value": confirm_value or {"action": "confirm"},
                },
                {
                    "tag": "button",
                    "text": {"tag": "plain_text", "content": "❌ 取消"},
                    "type": "default",
                    "value": cancel_value or {"action": "cancel"},
                },
            ],
            "layout": "bisected",
        })

        # 脚注
        elements.append({
            "tag": "note",
            "elements": [
                {
                    "tag": "plain_text",
                    "content": f"🕐 {CardBuilder._time_str()}  ·  请确认后继续",
                }
            ],
        })

        return {
            "config": {"wide_screen_mode": True},
            "header": {
                "title": {"tag": "plain_text", "content": title},
                "template": "yellow",
            },
            "elements": elements,
        }

    @staticmethod
    def error_card(error_message: str, detail: str = None) -> dict:
        """
        创建「错误」卡片

        Args:
            error_message: 简短错误描述
            detail: 详细错误信息（可选）
        """
        md = f"**处理请求时出现问题**\n\n{error_message}"
        if detail:
            md += f"\n\n---\n**详情**: {detail[:500]}"

        return {
            "config": {"wide_screen_mode": True},
            "header": {
                "title": {"tag": "plain_text", "content": "❌ 处理失败"},
                "template": "red",
            },
            "elements": [
                {
                    "tag": "div",
                    "text": {"tag": "lark_md", "content": md},
                },
                {
                    "tag": "hr",
                },
                {
                    "tag": "note",
                    "elements": [
                        {
                            "tag": "plain_text",
                            "content": f"🕐 {CardBuilder._time_str()}  ·  如持续出现，请联系IT服务台",
                        }
                    ],
                },
            ],
        }

    # ============================================================
    # 工具方法
    # ============================================================

    @staticmethod
    def to_message_content(card: dict) -> str:
        """将卡片 dict 转为 Feishu message.content JSON 字符串"""
        return json.dumps(card, ensure_ascii=False)

    @staticmethod
    def _time_str() -> str:
        """当前时间字符串"""
        return time.strftime("%H:%M:%S")


# ============================================================
# 快捷函数 — 进度步骤构建
# ============================================================

def make_step(label: str, status: str = "pending", detail: str = "") -> dict:
    """创建单个步骤项"""
    return {"label": label, "status": status, "detail": detail}


def map_stream_tokens_to_steps(tokens: list[str]) -> list[dict]:
    """
    将编排器流式 token 列表映射为卡片步骤列表

    token 格式: "[ORCHESTRATOR] 🔍 正在分析您的需求..."
    提取 [] 中的标签作为步骤标题，后面的 emoji 文本作为详情。

    Args:
        tokens: 编排器输出的 token 列表

    Returns:
        步骤列表 [{"label": "分析需求", "status": "done", "detail": "🔍 正在分析..."}, ...]
    """
    step_names = {
        "classify": "分析需求",
        "plan": "规划方案",
        "delegate": "调用Agent",
        "verify": "验证结果",
        "respond": "整理回复",
    }

    seen = set()
    steps = []
    for token in tokens:
        t = token.strip()
        if not t.startswith("["):
            continue

        # 提取原始 node name
        raw_label = t.split("]")[0].lstrip("[") if "]" in t else ""

        # 映射到中文名称
        label = step_names.get(raw_label, raw_label)

        # 去重
        if label in seen:
            continue
        seen.add(label)

        # 提取详情 text（] 后面的部分）
        detail = ""
        if "]" in t:
            detail = t.split("]", 1)[1].strip()

        steps.append({"label": label, "status": "done", "detail": detail})

    return steps

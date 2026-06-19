"""
Tool 调用错误分类器 — 生产级分层错误处理

面试考点：Tool Calling 失败不是统一 try/except，而是分层处理：
  - ValidationError  → 不重试，直接返回修正提示
  - TransientError   → 指数退避重试（最多 N 次）
  - DependencyError  → 降级到 fallback 或人工
  - UnknownError     → 重试一次，仍失败则升级

用法：
    from agents.tools.error_classifier import classify_error, ToolErrorType

    error_type, message = classify_error("check_inventory", exception)
    if error_type == ToolErrorType.VALIDATION:
        return json.dumps({"error": message, "retry": False})
    elif error_type == ToolErrorType.TRANSIENT:
        # 退避重试
    elif error_type == ToolErrorType.DEPENDENCY:
        # 降级 fallback
"""

from __future__ import annotations

import logging
from enum import Enum
from typing import Optional

logger = logging.getLogger("tools.error_classifier")


class ToolErrorType(str, Enum):
    """工具错误类型 — 决定重试/降级策略"""
    VALIDATION = "validation"       # 参数/输入错误 → 不重试，修正提示
    TRANSIENT = "transient"         # 暂时性错误 → 退避重试（网络/超时/锁）
    DEPENDENCY = "dependency"       # 依赖服务不可用 → 降级 fallback
    UNKNOWN = "unknown"             # 未分类 → 重试一次，仍失败则升级


# ── 错误分类规则 ──────────────────────────────────────────

# 关键错误消息模式 — 命中即分类
_VALIDATION_PATTERNS = [
    "请提供", "参数", "缺少", "为空", "无效", "invalid",
    "不支持", "未知工具", "not found", "does not exist",
    "required", "missing", "格式错误", "ValueError",
    "validation", "assertion",
]

_TRANSIENT_PATTERNS = [
    "timeout", "超时", "连接", "connection", "网络",
    "socket", "refused", "reset", "temporary",
    "try again", "重试", "locked", "busy", "并发",
    "rate limit", "throttle", "too many",
]

_DEPENDENCY_PATTERNS = [
    "service unavailable", "503", "502", "500",
    "database", "数据库", "无法访问", "不可用",
    "down", "offline", "维护", "maintenance",
    "认证失败", "auth", "unauthorized", "forbidden",
    "权限不足", "permission denied",
]


def classify_error(tool_name: str, exception: Exception) -> tuple[ToolErrorType, str]:
    """
    分类工具调用错误

    根据异常类型和消息内容判断错误分类。
    面试要点：看到面试官问"工具调用失败怎么处理"，你能讲出这四种分类
    而不是只说"我加了 try/except"。

    Args:
        tool_name: 工具名称（如 "check_inventory"）
        exception: 捕获的异常

    Returns:
        (error_type, user_message) — 用户友好的错误描述
    """
    exc_type = type(exception).__name__
    exc_msg = str(exception).lower() if exception else ""

    # ── Layer 1: 按异常类型分类 ──
    if exc_type in ("ValueError", "TypeError", "KeyError", "AttributeError"):
        return (
            ToolErrorType.VALIDATION,
            f"工具 [{tool_name}] 参数错误: {str(exception)[:100]}",
        )

    if exc_type in ("TimeoutError", "asyncio.TimeoutError", "ConnectionError",
                    "ConnectionRefusedError", "ConnectionResetError"):
        return (
            ToolErrorType.TRANSIENT,
            f"工具 [{tool_name}] 暂时性错误（网络/超时），可退避重试",
        )

    # ── Layer 2: 按消息内容分类 ──
    msg_lower = exc_msg

    # 依赖错误优先级最高 — 某些库把 503 也包成 RuntimeError
    for pattern in _DEPENDENCY_PATTERNS:
        if pattern in msg_lower:
            return (
                ToolErrorType.DEPENDENCY,
                f"工具 [{tool_name}] 依赖服务不可用: {str(exception)[:100]}",
            )

    for pattern in _VALIDATION_PATTERNS:
        if pattern.lower() in msg_lower:
            return (
                ToolErrorType.VALIDATION,
                f"工具 [{tool_name}] 输入校验失败: {str(exception)[:100]}",
            )

    for pattern in _TRANSIENT_PATTERNS:
        if pattern in msg_lower:
            return (
                ToolErrorType.TRANSIENT,
                f"工具 [{tool_name}] 暂时性错误: {str(exception)[:100]}",
            )

    # ── Layer 3: 兜底 — 未知错误重试一次 ──
    return (
        ToolErrorType.UNKNOWN,
        f"工具 [{tool_name}] 未知错误: {str(exception)[:100]}",
    )


# ── 重试策略 ──────────────────────────────────────────────


def get_retry_config(error_type: ToolErrorType) -> dict:
    """
    根据错误类型返回重试策略

    面试要点：你能说出每种错误的重试次数和退避策略的数字，
    说明你确实在线上调过这些参数。
    """
    configs = {
        ToolErrorType.VALIDATION: {
            "max_retries": 0,
            "backoff_base": 0,
            "strategy": "不重试 — 参数错误重试无意义，直接返回修正提示",
        },
        ToolErrorType.TRANSIENT: {
            "max_retries": 3,
            "backoff_base": 0.5,   # 0.5s → 1.0s → 2.0s 指数退避
            "strategy": "指数退避重试(0.5s/1s/2s) — 网络抖动/超时通常短暂自愈",
        },
        ToolErrorType.DEPENDENCY: {
            "max_retries": 1,
            "backoff_base": 1.0,
            "strategy": "重试1次后降级 — 依赖服务恢复慢，继续重试浪费用户等待时间",
        },
        ToolErrorType.UNKNOWN: {
            "max_retries": 1,
            "backoff_base": 1.0,
            "strategy": "重试1次后升级 — 未知错误模式下安全保守策略",
        },
    }
    return configs[error_type]


def format_tool_error_response(
    tool_name: str,
    error_type: ToolErrorType,
    message: str,
    retries_used: int = 0,
    fallback_available: bool = False,
) -> str:
    """
    生成 LLM 可理解的工具错误响应

    返回结构化 JSON 供 LLM 理解错误性质并决定下一步。
    关键：给 LLM 足够信息判断是修正参数重试还是放弃走降级。
    """
    import json as _json

    return _json.dumps({
        "error": True,
        "tool": tool_name,
        "error_type": error_type.value,
        "message": message,
        "retries_used": retries_used,
        "recoverable": error_type in (ToolErrorType.TRANSIENT, ToolErrorType.UNKNOWN),
        "suggestion": (
            "请修正参数后重试"
            if error_type == ToolErrorType.VALIDATION
            else (
                "可稍后重试，或选择其他工具"
                if error_type == ToolErrorType.TRANSIENT
                else (
                    "降级到 fallback 方案"
                    if error_type == ToolErrorType.DEPENDENCY and fallback_available
                    else "建议转人工处理"
                )
            )
        ),
    }, ensure_ascii=False)

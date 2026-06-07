"""
结构化日志配置 — 支持 Copilot Studio 多Agent可观测性

基于 structlog 的结构化日志，支持：
- Agent 追踪字段（agent_id, trace_id, message_id）
- 控制台输出和JSON格式
- A2A 通信链路日志
"""

import structlog
import logging
from config.settings import settings


def setup_logging():
    """配置结构化日志"""
    log_level = getattr(logging, settings.log_level.upper(), logging.INFO)

    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.UnicodeDecoder(),
            structlog.dev.ConsoleRenderer() if settings.debug
            else structlog.processors.JSONRenderer(),
        ],
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )

    # 设置标准库日志级别
    logging.basicConfig(
        format="%(levelname)s | %(name)s | %(message)s",
        level=log_level,
    )

    # 设置第三方库日志级别（减少噪音）
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)

    return structlog.get_logger()


def get_agent_logger(agent_id: str, trace_id: str = "") -> logging.Logger:
    """
    获取带有 Agent 追踪上下文的日志记录器

    使用方式：
        logger = get_agent_logger("it_consultant", trace_id)
        logger.info("开始知识库检索", extra={"trace_id": trace_id})

    Args:
        agent_id: Agent 唯一标识
        trace_id: 追踪ID

    Returns:
        配置好的 logger 实例
    """
    logger = logging.getLogger(f"agent.{agent_id}")

    # 为 logger 添加 Agent 上下文的 Filter
    if not any(isinstance(f, AgentContextFilter) for f in logger.filters):
        logger.addFilter(AgentContextFilter(agent_id, trace_id))

    return logger


class AgentContextFilter(logging.Filter):
    """
    Agent 上下文过滤器

    自动在日志记录中添加 agent_id 和 trace_id 字段。
    """

    def __init__(self, agent_id: str, trace_id: str = ""):
        super().__init__()
        self.agent_id = agent_id
        self.trace_id = trace_id

    def filter(self, record):
        record.agent_id = self.agent_id
        record.trace_id = self.trace_id
        return True


logger = setup_logging()

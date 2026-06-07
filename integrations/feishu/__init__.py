"""
飞书/Lark 集成模块

支持飞书企业协作平台的 Bot API 接入：
- 接收飞书消息 → 编排Agent处理 → 回复飞书消息
- 支持消息卡片和Markdown格式
- 支持多Agent路由

架构：
    飞书用户 @机器人 → Webhook → FastAPI → Orchestrator → Agent → 回复飞书

使用前需在飞书开放平台创建企业自建应用：
https://open.feishu.cn/app
"""
from .bot_handler import FeishuBotHandler, create_feishu_router
from .card_builder import CardBuilder, make_step, map_stream_tokens_to_steps

__all__ = [
    "FeishuBotHandler",
    "create_feishu_router",
    "CardBuilder",
    "make_step",
    "map_stream_tokens_to_steps",
]

"""模型提供商工厂

基于 DashScope (阿里云百炼) OpenAI 兼容接口，支持切换其他兼容提供商。

配置由 config/settings.py (pydantic-settings) 统一管理。
加载优先级: 系统环境变量 > .env 文件 > 默认值。

使用方式:
    from config.model_provider import create_chat_model, create_embedding_model

    llm = create_chat_model()              # 使用 LLM_MODEL 配置
    router_llm = create_chat_model("router")  # 使用 ROUTER_MODEL 配置
    embeddings = create_embedding_model()
"""

from __future__ import annotations

import ssl
import httpx
import certifi
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from pydantic import SecretStr

from config.settings import settings

# certifi 证书上下文（修复 Windows Python SSL 问题）
_ssl_context = ssl.create_default_context(cafile=certifi.where())

# 共享的 httpx 客户端，使用 certifi 证书
_shared_http_client = None
_shared_async_http_client = None


def _get_http_client() -> httpx.Client:
    """获取共享的 httpx 同步客户端（使用 certifi 证书）"""
    global _shared_http_client
    if _shared_http_client is None:
        _shared_http_client = httpx.Client(
            verify=_ssl_context,
            timeout=60.0,
        )
    return _shared_http_client


def _get_async_http_client() -> httpx.AsyncClient:
    """获取共享的 httpx 异步客户端（使用 certifi 证书）"""
    global _shared_async_http_client
    if _shared_async_http_client is None:
        _shared_async_http_client = httpx.AsyncClient(
            verify=_ssl_context,
            timeout=60.0,
        )
    return _shared_async_http_client


def get_api_key() -> str:
    """获取 API Key"""
    return settings.dashscope_api_key or ""


def create_chat_model(model_type: str = "main", temperature: float = 0):
    """创建聊天模型

    Args:
        model_type: "main" 使用 LLM_MODEL（默认 qwen-plus）
                    "router" 使用 ROUTER_MODEL（默认 qwen-plus）
        temperature: 温度参数（0=确定性, 1=创造性）
    """
    model = settings.router_model if model_type == "router" else settings.llm_model

    return ChatOpenAI(
        model=model,
        api_key=SecretStr(get_api_key()),
        base_url=settings.llm_base_url,
        temperature=temperature,
        http_client=_get_http_client(),
        http_async_client=_get_async_http_client(),
    )


def create_embedding_model():
    """创建 Embedding 模型（默认 text-embedding-v4）"""
    return OpenAIEmbeddings(
        model=settings.embedding_model,
        api_key=SecretStr(get_api_key()),
        base_url=settings.embedding_base_url,
        check_embedding_ctx_length=False,
    )

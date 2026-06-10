"""模型提供商工厂

基于 DashScope (阿里云百炼) OpenAI 兼容接口，支持切换其他兼容提供商。

使用方式:
    from config.model_provider import create_chat_model, create_embedding_model

    llm = create_chat_model()              # 使用 LLM_MODEL 环境变量
    router_llm = create_chat_model("router")  # 使用 ROUTER_MODEL（轻量快速模型）
    embeddings = create_embedding_model()
"""

from __future__ import annotations

import os
import ssl
import httpx
import certifi
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from pydantic import SecretStr

# 环境变量由 config/settings.py (pydantic-settings) 统一管理
# 加载优先级: 系统环境变量 > .env 文件 > 默认值

# 默认 base_url — 阿里云 DashScope 兼容接口
DASHSCOPE_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"

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
    return os.getenv("DASHSCOPE_API_KEY", "")


def get_base_url() -> str:
    """获取 LLM Base URL"""
    return os.getenv("LLM_BASE_URL", DASHSCOPE_BASE_URL)


def create_chat_model(model_type: str = "main", temperature: float = 0):
    """创建聊天模型

    Args:
        model_type: "main" 使用 LLM_MODEL（默认 qwen-plus）
                    "router" 使用 ROUTER_MODEL（默认 qwen-plus）
        temperature: 温度参数（0=确定性, 1=创造性）
    """
    if model_type == "router":
        model = os.getenv("ROUTER_MODEL", "qwen-plus")
    else:
        model = os.getenv("LLM_MODEL", "qwen-plus")

    return ChatOpenAI(
        model=model,
        api_key=SecretStr(get_api_key()),
        base_url=get_base_url(),
        temperature=temperature,
        http_client=_get_http_client(),
        http_async_client=_get_async_http_client(),
    )


def create_embedding_model():
    """创建 Embedding 模型（默认 text-embedding-v4）"""
    model = os.getenv("EMBEDDING_MODEL", "text-embedding-v4")
    base_url = os.getenv("EMBEDDING_BASE_URL", DASHSCOPE_BASE_URL)

    return OpenAIEmbeddings(
        model=model,
        api_key=SecretStr(get_api_key()),
        base_url=base_url,
        check_embedding_ctx_length=False,
    )

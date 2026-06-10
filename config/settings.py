"""
应用程序设置模块

使用 pydantic-settings 实现类型安全的配置管理
"""

from pydantic_settings import BaseSettings
from pydantic import Field
from typing import Optional


class AppSettings(BaseSettings):
    """应用程序设置"""

    # DashScope API
    dashscope_api_key: Optional[str] = Field(default=None, alias="DASHSCOPE_API_KEY")

    # LLM 模型
    llm_model: str = Field(default="qwen-plus", alias="LLM_MODEL")
    llm_base_url: str = Field(
        default="https://dashscope.aliyuncs.com/compatible-mode/v1",
        alias="LLM_BASE_URL"
    )
    router_model: str = Field(default="qwen-plus", alias="ROUTER_MODEL")

    # Embedding 模型
    embedding_model: str = Field(default="text-embedding-v4", alias="EMBEDDING_MODEL")
    embedding_base_url: str = Field(
        default="https://dashscope.aliyuncs.com/compatible-mode/v1",
        alias="EMBEDDING_BASE_URL"
    )

    # 数据库
    database_url: str = Field(
        default="sqlite:///./data/ticket_dispatch.db",
        alias="DATABASE_URL"
    )

    # Redis（可选）
    redis_url: Optional[str] = Field(default=None, alias="REDIS_URL")

    # 应用
    debug: bool = Field(default=False, alias="DEBUG")
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")
    host: str = Field(default="127.0.0.1", alias="HOST")
    port: int = Field(default=8000, alias="PORT")

    # 天气API（可选）
    openweather_api_key: Optional[str] = Field(default=None, alias="OPENWEATHER_API_KEY")

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "case_sensitive": False,
        "extra": "ignore"
    }


settings = AppSettings()

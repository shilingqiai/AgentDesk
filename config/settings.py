"""
应用程序设置模块

使用 pydantic-settings 实现类型安全的配置管理
支持环境变量自动加载和校验
"""

from pydantic_settings import BaseSettings
from pydantic import Field
from typing import Optional


class AppSettings(BaseSettings):
    """应用程序设置"""

    # 模型提供商
    model_provider: str = Field(default="qwen", alias="MODEL_PROVIDER")
    llm_api_key: Optional[str] = Field(default=None, alias="LLM_API_KEY")
    llm_base_url: Optional[str] = Field(default=None, alias="LLM_BASE_URL")
    llm_model: str = Field(default="qwen-plus", alias="LLM_MODEL")

    # Embedding
    embedding_provider: str = Field(default="qwen", alias="EMBEDDING_PROVIDER")
    embedding_api_key: Optional[str] = Field(default=None, alias="EMBEDDING_API_KEY")
    embedding_base_url: Optional[str] = Field(default=None, alias="EMBEDDING_BASE_URL")
    embedding_model: str = Field(default="text-embedding-v3", alias="EMBEDDING_MODEL")

    # 数据库
    database_url: str = Field(
        default="sqlite:///./data/ticket_dispatch.db",
        alias="DATABASE_URL"
    )

    # Redis (可选)
    redis_url: Optional[str] = Field(default=None, alias="REDIS_URL")

    # 应用
    debug: bool = Field(default=False, alias="DEBUG")
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")
    host: str = Field(default="127.0.0.1", alias="HOST")
    port: int = Field(default=8000, alias="PORT")

    # 天气API (可选，用于环境信息展示)
    openweather_api_key: Optional[str] = Field(default=None, alias="OPENWEATHER_API_KEY")

    # Azure OpenAI (可选)
    azure_openai_api_key: Optional[str] = Field(default=None, alias="AZURE_OPENAI_API_KEY")
    azure_openai_endpoint: Optional[str] = Field(default=None, alias="AZURE_OPENAI_ENDPOINT")
    azure_openai_deployment: Optional[str] = Field(default=None, alias="AZURE_OPENAI_DEPLOYMENT")
    azure_openai_version: Optional[str] = Field(default=None, alias="AZURE_OPENAI_VERSION")

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "case_sensitive": False,
        "extra": "ignore"
    }


# 全局设置实例
settings = AppSettings()

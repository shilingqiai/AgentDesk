"""
配置模块

提供应用程序所需的常量和基本配置

.env 文件在此处统一加载到 os.environ，确保所有 os.getenv()
调用都能读取到 .env 中的值。系统环境变量优先（override=False）。
"""

# ⚠️ 必须在其他模块之前加载 .env 到 os.environ
from dotenv import load_dotenv
load_dotenv(override=False)

from .constants import busy_periods_dict
from .settings import settings

__all__ = [
    'busy_periods_dict',
    'settings'
]

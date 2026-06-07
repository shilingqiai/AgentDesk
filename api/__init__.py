"""
API路由模块 — 企业员工AI服务台

提供知识库管理、任务分类的REST API接口
"""
from .knowledge import router as knowledge_router
from .task import router as task_router
from .chat_handler_v3 import router as chat_v3_router

api_routers = [
    knowledge_router,
    task_router,
    chat_v3_router,
]

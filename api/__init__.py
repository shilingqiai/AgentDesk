"""
API路由模块 — 企业员工AI服务台

提供知识库管理、任务分类、工单管理的REST API接口
"""
from .knowledge import router as knowledge_router
from .task import router as task_router
from .chat_handler_v3 import router as chat_v3_router
from .tickets import router as tickets_router
from .meeting_rooms import router as meeting_rooms_router
from .approvals import router as approvals_router

api_routers = [
    knowledge_router,
    task_router,
    chat_v3_router,
    tickets_router,
    meeting_rooms_router,
    approvals_router,
]

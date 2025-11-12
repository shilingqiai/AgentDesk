"""
API路由模块

提供工单调度、知识咨询、任务分类、工程师管理、知识库管理和效能分析的REST API接口
"""

# 导入各业务模块的路由
from .appointment import router as appointment_router
from .consultation import router as consultation_router
from .task import router as task_router
from .knowledge import router as knowledge_router
from .technician import router as technician_router
from .user_behavior_analysis import router as user_behavior_analysis_router
from .user_behavior_analysis import router_underscore as user_behavior_analysis_underscore_router

# 创建API路由列表（用于注册到FastAPI应用）
api_routers = [
    appointment_router,
    consultation_router,
    task_router,
    knowledge_router,
    technician_router,
    user_behavior_analysis_router,
    user_behavior_analysis_underscore_router
]

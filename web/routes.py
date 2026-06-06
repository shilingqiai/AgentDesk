"""
Web界面路由

处理前端页面渲染和聊天功能
"""
import os
from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse, StreamingResponse
from jinja2 import Environment, FileSystemLoader, select_autoescape
from pydantic import BaseModel
from api.chat_handler import ProcessUserInput_stream
from api.chat_handler_v2 import process_user_input_v2, get_conversation_state, reset_conversation
import logging

# 创建logger实例
logger = logging.getLogger(__name__)
# 模板配置 — 直接使用 Jinja2 Environment 避免 Starlette 兼容性问题
_TEMPLATE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "web", "templates")
_jinja_env = Environment(
    loader=FileSystemLoader(_TEMPLATE_DIR),
    autoescape=select_autoescape(["html", "xml"]),
)


def render_template(name: str, context: dict) -> HTMLResponse:
    """渲染Jinja2模板并返回HTML响应"""
    template = _jinja_env.get_template(name)
    content = template.render(**context)
    return HTMLResponse(content=content)

# Web路由器
router = APIRouter(tags=["Web界面"])

class ChatRequest(BaseModel):
    message: str
    state: str | None = None

@router.get("/", response_class=HTMLResponse, summary="主页")
async def read_root(request: Request):
    """渲染主页聊天界面"""
    return render_template("index.html", {"request": request})

@router.post("/chat/stream", summary="流式聊天")
async def chat_stream_endpoint(chat: ChatRequest):
    """处理流式聊天请求"""
    async def token_generator():
        async for token in ProcessUserInput_stream(chat.message):
            yield token
    return StreamingResponse(token_generator(), media_type="text/plain")

@router.post("/chat", summary="兼容性聊天接口")
async def chat_endpoint(chat: ChatRequest):
    """兼容性聊天接口，建议使用/chat/stream"""
    async def token_generator():
        async for token in ProcessUserInput_stream(chat.message):
            yield token
    return StreamingResponse(token_generator(), media_type="text/plain")

@router.post("/chat/v2/stream", summary="流式聊天 (LangGraph)")
async def chat_stream_v2_endpoint(chat: ChatRequest):
    """LangGraph版本流式聊天 — 使用状态图管理Agent协作"""
    async def token_generator():
        async for token in process_user_input_v2(chat.message, thread_id="web"):
            yield token
    return StreamingResponse(token_generator(), media_type="text/plain")

@router.post("/chat/v2/reset", summary="重置LangGraph对话")
async def reset_v2_conversation():
    """重置LangGraph对话状态"""
    reset_conversation("web")
    return {"status": "reset"}

@router.get("/user_behavior", response_class=HTMLResponse, summary="用户行为分析页面")
async def user_behavior_page(request: Request):
    """用户行为分析页面"""
    return render_template("user_behavior_analysis.html", {"request": request})

@router.get("/knowledge", response_class=HTMLResponse, summary="知识库管理页面")
async def knowledge_page(request: Request):
    """知识库管理页面"""
    # 通过API层获取知识库数据
    try:
        from api.knowledge import get_all_knowledge
        
        # 调用API层函数获取数据
        knowledge_data = await get_all_knowledge()
        documents = knowledge_data.get("documents", [])
        categories = knowledge_data.get("categories", [])
        
        return render_template("knowledge_management.html", {
            "request": request,
            "documents": documents,
            "categories": categories
        })
    except Exception as e:
        return render_template("knowledge_management.html", {
            "request": request,
            "documents": [],
            "categories": [],
            "error": str(e)
        })

@router.get("/technician", response_class=HTMLResponse, summary="工程师状态页面")
async def technician_page(request: Request):
    """工程师状态页面"""
    # 通过API层获取工程师数据
    try:
        from api.technician import get_all_technicians
        
        # 调用API层函数获取数据
        technicians = await get_all_technicians()
        
        return render_template("technician.html", {
            "request": request,
            "technicians": technicians
        })
    except Exception as e:
        return render_template("technician.html", {
            "request": request,
            "technicians": [],
            "error": str(e)
        })

@router.get("/technician_schedule", response_class=HTMLResponse, summary="工程师排班页面")
async def technician_schedule_page(request: Request):
    """工程师排班页面"""
    try:
        from api.technician import get_all_technicians_schedule_today
        from config.time_config import time_config
        
        # 获取当前日期
        current_date = time_config.current_date_str()
        
        # 通过API层获取所有工程师的排班数据
        schedules_data = await get_all_technicians_schedule_today()
        
        # 构建排班数据格式 - 直接使用API返回的数据
        schedule = []
        for schedule_item in schedules_data:
            schedule.append({
                "id": schedule_item["technician_id"],
                "name": schedule_item["technician_name"],
                "busy_periods": schedule_item["busy_periods"]
            })
        
        return render_template("technician_schedule.html", {
            "request": request,
            "schedule": schedule,
            "current_date": current_date
        })
    except Exception as e:
        logger.error(f"加载工程师排班数据失败: {str(e)}")
        return render_template("technician_schedule.html", {
            "request": request,
            "schedule": [],
            "error": str(e)
        })

@router.get("/user_behavior_analysis", response_class=HTMLResponse, summary="用户行为分析页面")
async def user_behavior_analysis_page(request: Request):
    """用户行为分析页面"""
    return render_template("user_behavior_analysis.html", {"request": request})

@router.get("/admin", response_class=HTMLResponse, summary="系统管理页面")
async def admin_dashboard(request: Request):
    """系统管理仪表板"""
    try:
        # 通过API层获取系统状态信息
        from api.knowledge import get_all_knowledge
        from api.technician import get_all_technicians
        
        # 获取知识库数据
        knowledge_data = await get_all_knowledge()
        knowledge_count = knowledge_data.get("total_count", 0)
        categories = knowledge_data.get("categories", [])
        
        # 获取工程师数据
        technicians = await get_all_technicians()
        
        # 数据库信息
        db_info = {
            "knowledge_count": knowledge_count,
            "categories_count": len(categories),
            "technicians_count": len(technicians),
            "categories": categories
        }
        
        return render_template("admin_dashboard.html", {
            "request": request,
            "db_info": db_info,
            "technicians": technicians[:5]  # 只显示前5个工程师
        })
    except Exception as e:
        return render_template("admin_dashboard.html", {
            "request": request,
            "db_info": {},
            "technicians": [],
            "error": str(e)
        })

@router.get("/admin/database", response_class=HTMLResponse, summary="数据库管理页面")
async def database_admin_page(request: Request):
    """数据库管理页面"""
    try:
        # 通过API层获取数据库统计信息
        from api.knowledge import get_all_knowledge
        from api.technician import get_all_technicians
        
        # 获取知识库数据
        knowledge_data = await get_all_knowledge()
        
        # 获取工程师数据
        technicians = await get_all_technicians()
        
        stats = {
            "knowledge_documents": knowledge_data.get("total_count", 0),
            "categories": len(knowledge_data.get("categories", [])),
            "technicians": len(technicians),
            "appointments": 0  # TODO: 通过API获取预约数量
        }
        
        return render_template("database_admin.html", {
            "request": request,
            "stats": stats
        })
    except Exception as e:
        return render_template("database_admin.html", {
            "request": request,
            "stats": {},
            "error": str(e)
        })

"""
Web界面路由 — 企业员工AI服务台

处理前端页面渲染和聊天功能。
"""
import os
import logging
from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse, StreamingResponse
from jinja2 import Environment, FileSystemLoader, select_autoescape
from pydantic import BaseModel

logger = logging.getLogger(__name__)

_TEMPLATE_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "web", "templates"
)
_jinja_env = Environment(
    loader=FileSystemLoader(_TEMPLATE_DIR),
    autoescape=select_autoescape(["html", "xml"]),
)

def render_template(name: str, context: dict = None) -> HTMLResponse:
    template = _jinja_env.get_template(name)
    content = template.render(**(context or {}))
    return HTMLResponse(content=content)

router = APIRouter(tags=["Web界面"])

class ChatRequest(BaseModel):
    message: str
    thread_id: str = "web"

# ============================================================
# 页面路由
# ============================================================

@router.get("/", response_class=HTMLResponse, summary="企业员工AI服务台")
async def read_root(request: Request):
    return render_template("index.html", {"request": request})

@router.get("/knowledge", response_class=HTMLResponse, summary="知识库管理")
async def knowledge_page(request: Request):
    try:
        from api.knowledge import get_all_knowledge
        knowledge_data = await get_all_knowledge()
        return render_template("knowledge_management.html", {
            "request": request,
            "documents": knowledge_data.get("documents", []),
            "categories": knowledge_data.get("categories", []),
        })
    except Exception as e:
        return render_template("knowledge_management.html", {
            "request": request,
            "documents": [],
            "categories": [],
            "error": str(e),
        })


@router.get("/tickets", response_class=HTMLResponse, summary="工单管理")
async def tickets_page(request: Request):
    return render_template("tickets.html", {"request": request})


@router.get("/meeting-rooms", response_class=HTMLResponse, summary="会议室预定")
async def meeting_rooms_page(request: Request):
    return render_template("meeting_rooms.html", {"request": request})

# ============================================================
# 聊天API
# ============================================================

@router.post("/chat/stream", summary="多Agent编排流式聊天")
async def chat_stream_endpoint(chat: ChatRequest):
    from agents.graph_workflow import orchestration_runner

    async def token_generator():
        async for token in orchestration_runner.run_stream(
            chat.message, thread_id=chat.thread_id,
        ):
            yield token

    return StreamingResponse(token_generator(), media_type="text/plain")

@router.post("/chat", summary="兼容性聊天接口")
async def chat_endpoint(chat: ChatRequest):
    from agents.graph_workflow import orchestration_runner

    async def token_generator():
        async for token in orchestration_runner.run_stream(
            chat.message, thread_id=chat.thread_id,
        ):
            yield token

    return StreamingResponse(token_generator(), media_type="text/plain")

@router.post("/chat/reset", summary="重置对话")
async def reset_conversation(chat: ChatRequest):
    from agents.graph_workflow import orchestration_runner
    from agents.a2a.context_manager import context_manager
    from agents.a2a.message_bus import message_bus

    orchestration_runner.reset(chat.thread_id)
    context_manager.clear_context(chat.thread_id)
    message_bus.clear_trace(chat.thread_id)
    return {"status": "reset"}

@router.get("/api/agents/list", summary="已注册Agent列表")
async def list_registered_agents():
    from agents.orchestrator.agent_registry import agent_registry
    return agent_registry.list_agents()

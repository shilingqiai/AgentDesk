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
from starlette.middleware.base import BaseHTTPMiddleware

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


# ============================================================
# 身份注入中间件 — 从 Header 解析当前用户
# ============================================================

class IdentityMiddleware(BaseHTTPMiddleware):
    """
    从 HTTP Header 注入用户身份到 request.state。

    企业内网环境下，反向代理（Nginx/网关）在 X-User-Name 中注入当前用户。
    本地开发 / Demo 模式下，前端通过 ChatRequest body 直接传递身份。

    优先级：Header > Body > 默认值（匿名用户）
    """

    async def dispatch(self, request: Request, call_next):
        # 从 header 读取（企业网关注入）
        user_name = request.headers.get("X-User-Name", "")
        role = request.headers.get("X-User-Role", "employee")

        # 存入 request.state 供下游使用
        request.state.user_name = user_name
        request.state.role = role

        response = await call_next(request)
        return response

class ChatRequest(BaseModel):
    message: str
    thread_id: str = "web"
    user_name: str = ""
    role: str = "employee"  # "employee" | "admin"

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
async def chat_stream_endpoint(chat: ChatRequest, request: Request):
    from agents.graph_workflow import orchestration_runner

    # 身份来源：Header 注入 > Body 传递 > 默认值
    user_name = request.state.user_name or chat.user_name or "anonymous"
    role = request.state.role or chat.role or "employee"

    async def token_generator():
        async for token in orchestration_runner.run_stream(
            chat.message,
            thread_id=chat.thread_id,
            user_name=user_name,
            role=role,
        ):
            yield token

    return StreamingResponse(token_generator(), media_type="text/plain")

@router.post("/chat", summary="兼容性聊天接口")
async def chat_endpoint(chat: ChatRequest, request: Request):
    from agents.graph_workflow import orchestration_runner

    user_name = request.state.user_name or chat.user_name or "anonymous"
    role = request.state.role or chat.role or "employee"

    async def token_generator():
        async for token in orchestration_runner.run_stream(
            chat.message,
            thread_id=chat.thread_id,
            user_name=user_name,
            role=role,
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


@router.post("/chat/reset-card", summary="清除卡片锁定状态")
async def reset_card_state(chat: ChatRequest):
    """用户 dismiss 卡片时清除 pending_card_type"""
    from agents.graph_workflow import orchestration_runner

    config = {"configurable": {"thread_id": chat.thread_id}}
    state = orchestration_runner.app.get_state(config)
    if state and state.values:
        orchestration_runner.app.update_state(
            config, {"pending_card_type": ""},
        )
    return {"status": "ok"}

@router.get("/api/agents/list", summary="已注册Agent列表")
async def list_registered_agents():
    from agents.orchestrator.agent_registry import agent_registry
    return agent_registry.list_agents()


@router.get("/api/identity", summary="获取当前用户身份")
async def get_identity(request: Request):
    """返回当前 request 中解析到的用户身份（来自 Header 或默认值）"""
    return {
        "user_name": getattr(request.state, "user_name", ""),
        "role": getattr(request.state, "role", "employee"),
    }

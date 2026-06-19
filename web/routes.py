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

@router.get("/chat", response_class=HTMLResponse, summary="智能服务台 Chat")
async def chat_page(request: Request):
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
    """已迁移至 SPA — 重定向到首页（Tickets 是 SPA 内的 Tab）"""
    return render_template("index.html", {"request": request})


@router.get("/meeting-rooms", response_class=HTMLResponse, summary="会议室预定")
async def meeting_rooms_page(request: Request):
    return render_template("meeting_rooms.html", {"request": request})

@router.get("/approvals", response_class=HTMLResponse, summary="审批门户")
async def approvals_page(request: Request):
    """已迁移至 SPA — 重定向到首页（Approval 是 SPA 内的 Tab）"""
    return render_template("index.html", {"request": request})

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

    return StreamingResponse(
        token_generator(),
        media_type="text/plain; charset=utf-8",
        headers={
            "X-Content-Type-Options": "nosniff",
            "Cache-Control": "no-cache",
        },
    )

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

    return StreamingResponse(
        token_generator(),
        media_type="text/plain; charset=utf-8",
        headers={
            "X-Content-Type-Options": "nosniff",
            "Cache-Control": "no-cache",
        },
    )

@router.post("/chat/reset", summary="重置对话")
async def reset_conversation(chat: ChatRequest):
    from agents.graph_workflow import orchestration_runner
    await orchestration_runner.reset(chat.thread_id)
    return {"status": "reset"}


@router.post("/chat/reset-card", summary="清除卡片锁定状态")
async def reset_card_state(chat: ChatRequest):
    """用户 dismiss 卡片时清除 pending_card_type"""
    from agents.graph_workflow import orchestration_runner

    config = {"configurable": {"thread_id": chat.thread_id}}
    state = await orchestration_runner.app.aget_state(config)
    if state and state.values:
        await orchestration_runner.app.aupdate_state(
            config, {"pending_card_type": ""},
        )
    return {"status": "ok"}


class ResumeRequest(BaseModel):
    """v8: LangGraph interrupt 恢复请求"""
    thread_id: str = "web"
    action: str = "confirm"  # "confirm" | "cancel"
    feedback: str = ""


@router.post("/chat/resume", summary="恢复被 interrupt() 冻结的图 (v8)")
async def resume_interrupted_graph(req: ResumeRequest):
    """
    v8: LangGraph interrupt() 恢复端点。

    前端卡片按钮调用此端点恢复被冻结的图:
      - action=confirm: 执行建单
      - action=cancel: 取消操作

    返回 JSON (卡片按钮不需要流式)。
    """
    from langgraph.types import Command
    from agents.graph_workflow import orchestration_runner

    config = {"configurable": {"thread_id": req.thread_id}}
    decision = {"action": req.action, "feedback": req.feedback}

    logger.info(
        f"[Resume] thread={req.thread_id}, action={req.action}"
    )

    try:
        # 同步 invoke — 卡片按钮不需要流式
        result = await orchestration_runner.app.ainvoke(
            Command(resume=decision), config,
        )

        # 检查是否又产生了新的 interrupt (modify → 新卡片)
        final_state = await orchestration_runner.app.aget_state(config)
        if final_state and final_state.interrupts:
            # 有新卡片 → 返回卡片数据让前端渲染
            interrupts = final_state.interrupts
            cards = []
            for interrupt_data in interrupts:
                if isinstance(interrupt_data, dict):
                    if "cards" in interrupt_data:
                        cards.extend(interrupt_data["cards"])
                    elif "card" in interrupt_data:
                        cards.append(interrupt_data["card"])
            return {
                "status": "interrupted",
                "cards": cards,
                "message": result.get("final_response", ""),
            }

        return {
            "status": "ok",
            "message": result.get("final_response", ""),
        }
    except Exception as e:
        logger.error(f"[Resume] 恢复失败: {e}")
        return {
            "status": "error",
            "message": f"恢复失败: {e}",
        }


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


class HistoryRequest(BaseModel):
    thread_id: str = "web"


@router.get("/chat/history", summary="获取会话历史")
async def get_chat_history(thread_id: str = "web"):
    """从 LangGraph checkpointer 恢复指定 thread 的对话历史。

    前端页面加载时调用，恢复跨页面导航后的聊天记录。
    返回 messages 列表 (role + content) 和当前 pending_card_type。
    """
    import asyncio
    from agents.graph_workflow import orchestration_runner

    try:
        await orchestration_runner._ensure_app()
    except Exception as e:
        logger.warning(f"Checkpointer 初始化失败: {e}")
        return {"messages": [], "pending_card_type": ""}

    config = {"configurable": {"thread_id": thread_id}}
    try:
        state = await asyncio.to_thread(
            orchestration_runner.app.get_state, config,
        )
    except Exception as e:
        logger.warning(f"获取会话状态失败: {e}")
        return {"messages": [], "pending_card_type": ""}

    if not state or not state.values:
        return {"messages": [], "pending_card_type": ""}

    raw_messages = state.values.get("messages", [])
    serialized = []
    for m in raw_messages:
        role = "user" if getattr(m, "type", "") == "human" else "assistant"
        content = getattr(m, "content", "") if hasattr(m, "content") else str(m)
        # 跳过空消息和纯系统消息
        if role == "system" or not str(content).strip():
            continue
        serialized.append({
            "role": role,
            "content": str(content),
        })

    return {
        "messages": serialized,
        "pending_card_type": state.values.get("pending_card_type", ""),
    }

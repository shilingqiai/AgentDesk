"""
FastAPI应用程序 — 企业员工AI服务台

启动方式:
  python app.py                  # 默认 all 模式 (web + feishu)
  python app.py --mode web       # 仅 Web 前端 + API
  python app.py --mode feishu    # 仅飞书 WS 长连接
  python app.py --mode all       # 两者都启动

环境变量:
  SERVICE_MODE=web|feishu|all    （命令行 --mode 优先）
  FEISHU_APP_ID / FEISHU_APP_SECRET
"""
from __future__ import annotations

import sys
import os
import logging
import argparse
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 飞书 Bot Handler（全局实例）
_feishu_handler = None


def parse_mode() -> str:
    """解析启动模式: 命令行 > SERVICE_MODE 环境变量 > 默认 all"""
    if "--mode" in sys.argv:
        idx = sys.argv.index("--mode")
        if idx + 1 < len(sys.argv):
            mode = sys.argv[idx + 1]
        else:
            mode = "all"
        # 从 sys.argv 中移除，避免 uvicorn 误解析
        del sys.argv[idx:idx + 2]
        return mode
    return os.getenv("SERVICE_MODE", "all").lower()


SERVICE_MODE = parse_mode()


async def initialize_system():
    """系统启动时初始化"""
    global _feishu_handler

    try:
        mode_label = {"web": "Web 前端模式", "feishu": "飞书 Bot 模式", "all": "完整模式"}
        logger.info(f"🚀 正在初始化企业员工AI服务台... 模式: {mode_label.get(SERVICE_MODE, SERVICE_MODE)}")

        # 知识库（web / all 模式需要）
        if SERVICE_MODE in ("web", "all"):
            logger.info("📚 初始化知识库...")
            from services.knowledge_service import KnowledgeService
            knowledge_service = KnowledgeService()
            await knowledge_service.initialize()

        # 事件总线 — 注册 Handler（web / all 模式）
        if SERVICE_MODE in ("web", "all"):
            logger.info("📡 初始化事件总线...")
            from services.event_bus import EventBus, EventType
            from services.event_handlers import (
                NotificationHandler as NH,
                AuditHandler as AH,
                DashboardHandler as DH,
            )
            # 通知
            EventBus.subscribe(EventType.TICKET_CREATED, NH.on_ticket_created)
            EventBus.subscribe(EventType.TICKET_STATUS_CHANGED, NH.on_status_changed)
            EventBus.subscribe(EventType.APPROVAL_STEP_APPROVED, NH.on_approval_step_approved)
            EventBus.subscribe(EventType.APPROVAL_COMPLETED, NH.on_approval_completed)
            EventBus.subscribe(EventType.APPROVAL_REJECTED, NH.on_approval_rejected)
            # 审计
            EventBus.subscribe(EventType.TICKET_CREATED, AH.on_any_event)
            EventBus.subscribe(EventType.TICKET_STATUS_CHANGED, AH.on_any_event)
            EventBus.subscribe(EventType.APPROVAL_STEP_APPROVED, AH.on_any_event)
            EventBus.subscribe(EventType.APPROVAL_COMPLETED, AH.on_any_event)
            EventBus.subscribe(EventType.APPROVAL_REJECTED, AH.on_any_event)
            # Dashboard
            EventBus.subscribe(EventType.TICKET_CREATED, DH.on_ticket_created)
            EventBus.subscribe(EventType.TICKET_STATUS_CHANGED, DH.on_status_changed)
            # SLA
            EventBus.subscribe(EventType.SLA_BREACHED, NH.on_sla_breached)
            EventBus.subscribe(EventType.SLA_BREACHED, AH.on_any_event)
            EventBus.subscribe(EventType.SLA_BREACHED, DH.on_sla_breached)
            logger.info("✅ 事件总线已就绪")

            # SLA 定时调度器
            logger.info("⏰ 启动 SLA 定时调度器...")
            from services.sla_scheduler import SLAScheduler
            await SLAScheduler.start()
            logger.info("✅ SLA 调度器已启动")

        # 飞书 WebSocket 长连接（feishu / all 模式）
        if SERVICE_MODE in ("feishu", "all"):
            if os.getenv("FEISHU_APP_ID") and os.getenv("FEISHU_APP_SECRET"):
                logger.info("🔗 启动飞书 WebSocket 长连接...")
                from integrations.feishu.bot_handler import FeishuBotHandler
                _feishu_handler = FeishuBotHandler()
                await _feishu_handler.start_ws()
            else:
                logger.warning(
                    "⚠️  飞书未配置，跳过。"
                    "在 .env 或系统环境变量设置 FEISHU_APP_ID 和 FEISHU_APP_SECRET"
                )

        logger.info("✅ 系统初始化完成！")
    except Exception as e:
        logger.error(f"❌ 系统初始化失败: {e}")
        raise


async def shutdown_system():
    """
    系统关闭时清理所有资源。

    修复 Windows Ctrl+C 卡死：
    - SSE 流式接口未退出 → httpx 连接池未释放 → ProactorEventLoop 死等
    - aiosqlite 连接未关闭 → socket 泄漏
    这三层必须逐层关闭，顺序不能乱。
    """
    global _feishu_handler
    import asyncio

    # 1. 停止 SLA 调度器
    try:
        from services.sla_scheduler import SLAScheduler
        await asyncio.wait_for(SLAScheduler.stop(), timeout=3.0)
        logger.info("⏰ SLA 调度器已停止")
    except asyncio.TimeoutError:
        logger.warning("⚠️ SLA 调度器停止超时（3s），强制跳过")
    except Exception as e:
        logger.warning(f"⚠️ SLA 调度器停止异常: {e}")

    # 2. 关闭飞书 WebSocket 长连接
    if _feishu_handler:
        try:
            await asyncio.wait_for(_feishu_handler.stop_ws(), timeout=3.0)
            logger.info("🔌 飞书 WebSocket 连接已关闭")
        except asyncio.TimeoutError:
            logger.warning("⚠️ 飞书 WebSocket 关闭超时（3s），强制跳过")
        except Exception as e:
            logger.warning(f"⚠️ 飞书 WebSocket 关闭异常: {e}")

    # 2. 关闭 LangGraph checkpointer (aiosqlite)
    try:
        from agents.graph_workflow import orchestration_runner
        await asyncio.wait_for(orchestration_runner.close(), timeout=3.0)
    except asyncio.TimeoutError:
        logger.warning("⚠️ Checkpointer 关闭超时（3s），强制跳过")
    except Exception as e:
        logger.warning(f"⚠️ Checkpointer 关闭异常: {e}")

    # 3. 关闭 httpx 连接池（最底层 — 必须在最后关闭）
    try:
        from config.model_provider import close_http_clients
        await asyncio.wait_for(close_http_clients(), timeout=3.0)
    except asyncio.TimeoutError:
        logger.warning("⚠️ httpx 连接池关闭超时（3s），强制跳过")
    except Exception as e:
        logger.warning(f"⚠️ httpx 连接池关闭异常: {e}")

    logger.info("✅ 所有资源已清理")


def create_app() -> FastAPI:
    """创建FastAPI应用实例"""
    app = FastAPI(
        title="企业员工AI服务台",
        description="Copilot Studio 多Agent编排 — IT自助、HR咨询、行政服务",
        version="2.0.0",
        docs_url="/docs",
        redoc_url="/redoc",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # 身份注入中间件（从 Header 解析 X-User-Name / X-User-Role）
    from web.routes import IdentityMiddleware
    app.add_middleware(IdentityMiddleware)

    from api.exceptions import api_exception_handler, general_exception_handler, BusinessException
    app.add_exception_handler(BusinessException, api_exception_handler)
    app.add_exception_handler(Exception, general_exception_handler)

    # API 路由（所有模式都需要）
    from api import api_routers
    for router in api_routers:
        app.include_router(router)

    # Web 前端路由（web / all 模式）
    if SERVICE_MODE in ("web", "all"):
        from web import router as web_router
        app.include_router(web_router)
        app.mount("/static", StaticFiles(directory="web/static"), name="static")

    # 飞书集成路由（feishu / all 模式）
    if SERVICE_MODE in ("feishu", "all"):
        from integrations.feishu.bot_handler import create_feishu_router
        app.include_router(create_feishu_router())

    @app.on_event("startup")
    async def startup_event():
        await initialize_system()

    @app.on_event("shutdown")
    async def shutdown_event():
        await shutdown_system()

    return app


app = create_app()

if __name__ == "__main__":
    import uvicorn

    # web 模式默认 8001，feishu 模式默认 8002
    default_port = 8001 if SERVICE_MODE != "feishu" else 8002

    parser = argparse.ArgumentParser(description="企业员工AI服务台")
    parser.add_argument("--port", type=int, default=default_port, help=f"监听端口 (默认 {default_port})")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="监听地址")
    args, _ = parser.parse_known_args()

    logger.info(f"启动模式: {SERVICE_MODE} | 端口: {args.port}")

    # ── 修复 Windows Ctrl+C 卡死 ──
    # timeout_graceful_shutdown=5: 5 秒后强制关闭所有未完成的连接
    # 配合 shutdown_system() 的资源清理，确保进程不会无限挂起
    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        timeout_graceful_shutdown=5,
    )

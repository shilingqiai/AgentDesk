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
    """系统关闭时清理"""
    global _feishu_handler
    if _feishu_handler:
        await _feishu_handler.stop_ws()
        logger.info("🔌 飞书 WebSocket 连接已关闭")


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
    uvicorn.run(app, host=args.host, port=args.port)

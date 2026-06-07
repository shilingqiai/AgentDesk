"""
飞书 Bot 消息处理 — 基于官方 lark-oapi SDK WebSocket 长连接

参考官方示例:
  https://open.feishu.cn/document/uAjLw4CM/ukTMukTMukTM/reference/im-v1/message/events/receive

架构:
  飞书 WS 长连接 → 同步回调 → 调度编排器(异步) → SDK HTTP Client 回复
     单聊(p2p): CreateMessage
     群聊: ReplyMessage

环境变量:
    FEISHU_APP_ID=xxx        （必填）
    FEISHU_APP_SECRET=xxx    （必填）
"""

from __future__ import annotations

import os
import json
import logging
import asyncio
from typing import Optional

from fastapi import APIRouter, Request, HTTPException
from fastapi.responses import JSONResponse

logger = logging.getLogger("integration.feishu")

# ============================================================
# 飞书配置 — 从系统环境变量 / .env 读取
# ============================================================

FEISHU_APP_ID = os.getenv("FEISHU_APP_ID", "")
FEISHU_APP_SECRET = os.getenv("FEISHU_APP_SECRET", "")
FEISHU_ENCRYPT_KEY = os.getenv("FEISHU_ENCRYPT_KEY", "")


# ============================================================
# FeishuBotHandler
# ============================================================

class FeishuBotHandler:
    """
    飞书机器人消息处理器 — WebSocket 长连接 + OpenAPI

    参考官方 Python 示例代码实现：
    - WS 长连接接收事件（后台线程）
    - 同步回调处理消息，区分单聊(p2p)/群聊
    - SDK HTTP Client 发送回复（自动管理 token）
    """

    def __init__(self):
        self._api_client = None       # lark_oapi.Client (HTTP)
        self._ws_client = None        # lark_oapi.ws.Client (WebSocket)
        self._ws_started = False
        self._seen_messages: set = set()  # 已处理消息去重

    # ============================================================
    # HTTP API Client (同步)
    # ============================================================

    def _get_api_client(self):
        """获取 HTTP API Client（SDK 自动管理 tenant_access_token）"""
        if self._api_client is None:
            import lark_oapi as lark
            self._api_client = lark.Client.builder() \
                .app_id(FEISHU_APP_ID) \
                .app_secret(FEISHU_APP_SECRET) \
                .build()
        return self._api_client

    # ============================================================
    # WebSocket 长连接
    # ============================================================

    async def start_ws(self):
        """
        启动飞书 WebSocket 长连接

        在独立后台线程中运行 WS Client.start()，
        替换 SDK 模块级 loop 为空闲事件循环。
        """
        if not FEISHU_APP_ID or not FEISHU_APP_SECRET:
            logger.warning(
                "[Feishu] FEISHU_APP_ID 或 FEISHU_APP_SECRET 未配置，"
                "跳过飞书长连接启动。"
            )
            return

        try:
            import lark_oapi as lark
            from lark_oapi.api.im.v1 import P2ImMessageReceiveV1

            # 确保 API Client 已初始化
            self._get_api_client()

            # 注册事件处理器（同步回调）
            event_handler = (
                lark.EventDispatcherHandler.builder("", "")
                .register_p2_im_message_receive_v1(self._on_message)
                .build()
            )

            # 创建 WS Client
            self._ws_client = lark.ws.Client(
                FEISHU_APP_ID,
                FEISHU_APP_SECRET,
                event_handler=event_handler,
                log_level=lark.LogLevel.INFO,
            )

            logger.info("[Feishu] 正在启动 WebSocket 长连接...")

            # 后台线程运行 WS Client
            # SDK 的 Client.start() 内部使用模块级 loop 变量
            # uvicorn 下主线程有 running loop，必须在后台线程中替换为新建的空闲 loop
            import threading

            def _run_ws():
                new_loop = asyncio.new_event_loop()
                asyncio.set_event_loop(new_loop)
                # 替换 SDK 模块级 loop（start() 方法直接引用它）
                import lark_oapi.ws.client as _ws_mod
                _ws_mod.loop = new_loop
                try:
                    self._ws_client.start()
                except Exception as e:
                    logger.error(f"[Feishu] WS 连接异常: {e}", exc_info=True)

            self._ws_thread = threading.Thread(
                target=_run_ws,
                daemon=True,
                name="feishu-ws-client",
            )
            self._ws_thread.start()
            self._ws_started = True

            logger.info("[Feishu] ✅ WebSocket 长连接已启动（后台线程）")

        except Exception as e:
            logger.error(f"[Feishu] 启动 WebSocket 长连接失败: {e}", exc_info=True)
            raise

    async def stop_ws(self):
        """停止飞书 WebSocket 长连接"""
        self._ws_started = False

    # ============================================================
    # 消息回调 — 由 SDK WS Client 在后台线程同步调用
    # 参考: https://open.feishu.cn/document/uAjLw4CM/ukTMukTMukTM/reference/im-v1/message/events/receive
    # ============================================================

    def _on_message(self, data):
        """
        接收消息事件回调（同步方法，必须快速返回）

        ⚠️ 飞书 WS 长连接要求事件回调在 3 秒内返回，否则认为超时。
        因此这里只做：解析消息 → 去重检查 → 启动后台线程 → 立即返回。
        实际的编排处理和回复发送在后台线程中完成。

        参考官方示例:
          https://open.feishu.cn/document/uAjLw4CM/ukTMukTMukTM/reference/im-v1/message/events/receive
        """
        try:
            event_data = data.event       # P2ImMessageReceiveV1Data
            message = event_data.message  # EventMessage
            sender = event_data.sender    # EventSender

            # --- 去重：飞书重连后会重发未确认的消息 ---
            msg_id = message.message_id
            if msg_id in self._seen_messages:
                logger.debug(f"[Feishu] 跳过重复消息: {msg_id}")
                return
            self._seen_messages.add(msg_id)
            # 限长，避免内存泄漏
            if len(self._seen_messages) > 10000:
                self._seen_messages.clear()

            # --- 解析消息内容 ---
            if message.message_type != "text":
                # 非文本消息，群聊中回复提示
                if message.chat_type != "p2p":
                    self._send_reply_sync(
                        message.chat_id, message.chat_type, message.message_id,
                        "请发送文本消息，我暂时还看不懂图片或文件哦～",
                    )
                return

            user_text = json.loads(message.content).get("text", "").strip()

            # 移除 @机器人 前缀
            if user_text.startswith("@") and " " in user_text:
                user_text = user_text.split(" ", 1)[1].strip()

            if not user_text:
                return

            # 提取用户信息
            sender_id = sender.sender_id
            user_id = getattr(sender_id, 'open_id', None) or "unknown"

            logger.info(
                f"[Feishu] 收到消息: user={str(user_id)[:12]}..., "
                f"chat_type={message.chat_type}, "
                f"msg_id={msg_id}, "
                f"text={user_text[:50]}..."
            )

            # --- 启动后台处理（不阻塞，立即返回） ---
            self._fire_orchestrator_async(
                user_text,
                chat_id=message.chat_id,
                chat_type=message.chat_type,
                message_id=message.message_id,
            )

        except Exception as e:
            logger.error(f"[Feishu] 消息回调异常: {e}", exc_info=True)

    def _fire_orchestrator_async(self, user_text: str, chat_id: str,
                                   chat_type: str, message_id: str):
        """
        在后台线程中异步运行编排器，完成后自动发送回复

        ⚠️ 不能在 WS 回调线程中阻塞等待编排结果：
        - 编排器跑 LLM 需要 5-15 秒
        - WS 长连接需要在此期间回应 ping/heartbeat
        - 阻塞会导致 ping timeout → 连接断开 → 消息重发
        - 因此必须在独立线程中 fire-and-forget
        """
        import threading

        def _run_and_reply():
            try:
                from agents.graph_workflow import orchestration_runner

                async def _process():
                    parts = []
                    async for token in orchestration_runner.run_stream(
                        user_text, thread_id=f"feishu_{chat_id}",
                    ):
                        parts.append(token)
                    return "".join(parts)

                response_text = asyncio.run(_process())
                response_text = self._clean_response(response_text)

                if response_text:
                    self._send_reply_sync(chat_id, chat_type, message_id,
                                          response_text[:4000])

            except Exception as e:
                logger.error(f"[Feishu] 后台编排失败: {e}", exc_info=True)
                try:
                    self._send_reply_sync(
                        chat_id, chat_type, message_id,
                        "抱歉，处理您的请求时出现了问题，请稍后重试。",
                    )
                except Exception:
                    pass

        t = threading.Thread(target=_run_and_reply, daemon=True,
                            name=f"feishu-orch-{chat_id[:8]}")
        t.start()

    def _send_reply_sync(self, chat_id: str, chat_type: str,
                          message_id: str, text: str):
        """同步发送回复消息"""
        from lark_oapi.api.im.v1 import (
            CreateMessageRequest, CreateMessageRequestBody,
            ReplyMessageRequest, ReplyMessageRequestBody,
        )

        client = self._get_api_client()
        content = json.dumps({"text": text})

        try:
            if chat_type == "p2p":
                body = CreateMessageRequestBody.builder() \
                    .receive_id(chat_id) \
                    .msg_type("text") \
                    .content(content) \
                    .build()
                request = CreateMessageRequest.builder() \
                    .receive_id_type("chat_id") \
                    .request_body(body) \
                    .build()
                response = client.im.v1.message.create(request)
            else:
                body = ReplyMessageRequestBody.builder() \
                    .content(content) \
                    .msg_type("text") \
                    .build()
                request = ReplyMessageRequest.builder() \
                    .message_id(message_id) \
                    .request_body(body) \
                    .build()
                response = client.im.v1.message.reply(request)

            if response.success():
                logger.info(
                    f"[Feishu] 回复成功({chat_type}): {response.data.message_id}"
                )
            else:
                logger.error(
                    f"[Feishu] 发送失败: code={response.code}, msg={response.msg}"
                )
        except Exception as e:
            logger.error(f"[Feishu] 发送回复异常: {e}", exc_info=True)

    def _clean_response(self, text: str) -> str:
        """清理编排器内部标签"""
        lines = []
        for line in text.split("\n"):
            stripped = line.strip()
            if any(stripped.startswith(prefix) for prefix in [
                "[ORCHESTRATOR]", "[AGENT:", "[DEBUG]", "[PLAN]", "[VERIFY]",
            ]):
                continue
            if stripped:
                lines.append(stripped)
        return "\n".join(lines)


# ============================================================
# FastAPI 路由（健康检查 + Webhook 兼容 + 手动发送）
# ============================================================

def create_feishu_router(handler: FeishuBotHandler = None) -> APIRouter:
    """
    创建飞书集成 FastAPI 路由
    """
    if handler is None:
        handler = FeishuBotHandler()

    router = APIRouter(prefix="/feishu", tags=["飞书集成"])

    @router.post("/event", summary="飞书事件回调（Webhook 兼容）")
    async def feishu_event(request: Request):
        """Webhook 模式事件回调（备用）"""
        body = await request.json()
        challenge = body.get("challenge")
        if challenge:
            return JSONResponse({"challenge": challenge})

        # 使用 SDK 事件分发处理器
        import lark_oapi as lark
        from lark_oapi.core.request import RawRequest

        event_handler = (
            lark.EventDispatcherHandler.builder("", "")
            .register_p2_im_message_receive_v1(handler._on_message)
            .build()
        )

        raw_req = RawRequest()
        raw_req.uri = str(request.url)
        raw_req.headers = dict(request.headers)
        raw_req.body = json.dumps(body).encode("utf-8")

        resp = event_handler.do(raw_req)
        return JSONResponse(
            content=json.loads(resp.content or "{}"),
            status_code=resp.status_code,
        )

    @router.post("/send", summary="主动发送飞书消息")
    async def send_feishu_message(request: Request):
        """手动发送消息到飞书"""
        body = await request.json()
        chat_id = body.get("chat_id")
        text = body.get("text")
        if not chat_id or not text:
            raise HTTPException(status_code=400, detail="chat_id and text required")

        import lark_oapi as lark
        from lark_oapi.api.im.v1 import CreateMessageRequest, CreateMessageRequestBody

        content = json.dumps({"text": text})
        req_body = CreateMessageRequestBody.builder() \
            .receive_id(chat_id) \
            .msg_type("text") \
            .content(content) \
            .build()
        request = CreateMessageRequest.builder() \
            .receive_id_type("chat_id") \
            .request_body(req_body) \
            .build()

        client = handler._get_api_client()
        response = client.im.v1.message.create(request)

        if response.success():
            return {"code": 0, "message_id": response.data.message_id}
        else:
            return {"code": response.code, "msg": response.msg}

    @router.get("/health", summary="飞书集成健康检查")
    async def feishu_health():
        return {
            "status": "ok",
            "mode": "WebSocket 长连接 (SDK)",
            "app_id": FEISHU_APP_ID[:8] + "..." if FEISHU_APP_ID else "not configured",
            "has_secret": bool(FEISHU_APP_SECRET),
            "ws_connected": handler._ws_started,
        }

    return router

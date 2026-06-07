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
import time
import logging
import asyncio
import threading
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
        self._last_card_update: float = 0  # 卡片更新限流时间戳

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
                .register_p2_card_action_trigger(self._on_card_action)
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
        在后台线程中运行编排器，渐进更新卡片

        新的渐进卡片流程：
        ① 发送 thinking 卡片（msg_type="interactive"）→ 获得 card_msg_id
        ② 运行编排器收集流式 token，每遇到进度标签就 PatchMessage 更新卡片
        ③ 编排完成 → PatchMessage 替换为结果卡片

        降级：如果卡片发送失败 → text 文本回复
        """
        # 使用线程而非 asyncio，因为需要在新线程中创建 event loop
        t = threading.Thread(
            target=self._run_orchestrator_with_cards,
            args=(user_text, chat_id, chat_type, message_id),
            daemon=True,
            name=f"feishu-orch-{chat_id[:8]}",
        )
        t.start()

    def _run_orchestrator_with_cards(self, user_text: str, chat_id: str,
                                       chat_type: str, message_id: str):
        """
        编排器后台线程入口：渐进卡片更新

        流程：
        1. 发送 thinking 卡片
        2. 运行编排器流式收集 token
        3. 每遇到进度标签 → 更新卡片步骤
        4. 完成 → 替换为结果卡片
        """
        card_msg_id = None
        try:
            from integrations.feishu.card_builder import (
                CardBuilder, map_stream_tokens_to_steps,
            )

            # ① 发送 thinking 卡片
            thinking_card = CardBuilder.thinking_card()
            card_msg_id = self._send_interactive_card_sync(
                chat_id, chat_type, message_id, thinking_card,
            )

            if not card_msg_id:
                # 卡片发送失败 → 降级为纯文本模式
                logger.warning("[Feishu] 卡片发送失败，降级为纯文本模式")
                self._fallback_text_mode(user_text, chat_id, chat_type, message_id)
                return

            # ② 运行编排器，流式收集进度
            from agents.graph_workflow import orchestration_runner

            step_tokens = []   # 进度标签 token
            body_parts = []    # 最终回复正文
            current_steps = [] # 当前卡片步骤列表

            async def _process():
                async for token in orchestration_runner.run_stream(
                    user_text, thread_id=f"feishu_{chat_id}",
                ):
                    stripped = token.strip()
                    # 判断是否为进度标签（以 [ 开头且紧随标签名）
                    if (stripped.startswith("[") and
                            "]" in stripped[:60] and
                            not stripped.startswith("[DEBUG]")):
                        # 进度标签
                        step_tokens.append(stripped)
                        # 映射为卡片步骤并节流更新
                        current_steps = map_stream_tokens_to_steps(step_tokens)
                        self._update_thinking_card_sync(
                            card_msg_id, current_steps,
                        )
                    else:
                        # 非进度标签 = 正文内容
                        body_parts.append(token)

                return "".join(body_parts)

            response_text = asyncio.run(_process())
            response_text = self._clean_response(response_text)

            if not response_text:
                response_text = "处理完成，但未生成有效回复。请稍后重试。"

            # ③ 替换为结果卡片（强制更新，忽略限流）
            # 检查是否有需要确认的操作
            has_human_review = (
                "[VERIFY] 需要人工审核" in " ".join(step_tokens) or
                "human_loop" in " ".join(step_tokens)
            )

            if has_human_review:
                # 确认卡片（带按钮）
                result_card = CardBuilder.confirm_card(
                    title="⚠️ 需要人工审核",
                    description_md=response_text[:2000],
                    confirm_value={"action": "confirm_operation", "chat_id": chat_id},
                    cancel_value={"action": "cancel_operation", "chat_id": chat_id},
                )
            else:
                # 普通结果卡片
                result_card = CardBuilder.result_card(
                    title="✅ 处理完成",
                    content_md=response_text[:4000],
                )

            self._update_card_sync(card_msg_id, result_card, force=True)

        except Exception as e:
            logger.error(f"[Feishu] 后台编排失败: {e}", exc_info=True)
            try:
                from integrations.feishu.card_builder import CardBuilder

                if card_msg_id:
                    # 更新 thinking 卡片为错误状态
                    error_card = CardBuilder.error_card(
                        "处理您的请求时出现了问题，请稍后重试。",
                        str(e)[:200],
                    )
                    self._update_card_sync(card_msg_id, error_card, force=True)
                else:
                    # 没有卡片，发文本
                    self._send_reply_sync(
                        chat_id, chat_type, message_id,
                        "抱歉，处理您的请求时出现了问题，请稍后重试。",
                    )
            except Exception:
                pass

    def _fallback_text_mode(self, user_text: str, chat_id: str,
                              chat_type: str, message_id: str):
        """纯文本降级模式（卡片不可用时使用）"""
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
            logger.error(f"[Feishu] 文本降级也失败: {e}", exc_info=True)
            self._send_reply_sync(
                chat_id, chat_type, message_id,
                "抱歉，处理您的请求时出现了问题，请稍后重试。",
            )

    # ============================================================
    # 卡片消息发送与更新
    # ============================================================

    def _send_interactive_card_sync(self, chat_id: str, chat_type: str,
                                      message_id: str, card: dict) -> Optional[str]:
        """
        发送交互卡片消息，返回卡片消息 ID

        Args:
            chat_id: 聊天ID
            chat_type: "p2p" 或 "group"
            message_id: 原始用户消息ID（群聊 reply 用）
            card: 卡片 dict（CardBuilder 返回的格式）

        Returns:
            卡片消息 ID（用于后续更新），失败返回 None
        """
        from lark_oapi.api.im.v1 import (
            CreateMessageRequest, CreateMessageRequestBody,
            ReplyMessageRequest, ReplyMessageRequestBody,
        )

        client = self._get_api_client()
        content = json.dumps(card, ensure_ascii=False)

        try:
            if chat_type == "p2p":
                body = CreateMessageRequestBody.builder() \
                    .receive_id(chat_id) \
                    .msg_type("interactive") \
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
                    .msg_type("interactive") \
                    .build()
                request = ReplyMessageRequest.builder() \
                    .message_id(message_id) \
                    .request_body(body) \
                    .build()
                response = client.im.v1.message.reply(request)

            if response.success():
                card_msg_id = response.data.message_id
                logger.info(
                    f"[Feishu] 卡片发送成功({chat_type}): {card_msg_id}"
                )
                return card_msg_id
            else:
                logger.error(
                    f"[Feishu] 卡片发送失败: code={response.code}, msg={response.msg}"
                )
                return None

        except Exception as e:
            logger.error(f"[Feishu] 卡片发送异常: {e}", exc_info=True)
            return None

    def _update_card_sync(self, card_msg_id: str, card: dict,
                            force: bool = False) -> bool:
        """
        通过 PatchMessage 更新卡片内容（渐进更新核心）

        限流策略：距上次更新 < 500ms 时跳过，force=True 时忽略限流。

        Args:
            card_msg_id: 要更新的卡片消息 ID
            card: 新的卡片 dict
            force: 强制更新（忽略限流）

        Returns:
            是否成功
        """
        # 限流（force 忽略）
        if not force:
            elapsed = time.time() - self._last_card_update
            if elapsed < 0.5:
                return False

        from lark_oapi.api.im.v1 import (
            PatchMessageRequest, PatchMessageRequestBody,
        )

        client = self._get_api_client()
        content = json.dumps(card, ensure_ascii=False)

        try:
            body = PatchMessageRequestBody.builder() \
                .content(content) \
                .build()
            request = PatchMessageRequest.builder() \
                .message_id(card_msg_id) \
                .request_body(body) \
                .build()
            response = client.im.v1.message.patch(request)

            self._last_card_update = time.time()

            if response.success():
                return True
            else:
                logger.debug(
                    f"[Feishu] 卡片更新失败: code={response.code}, msg={response.msg}"
                )
                return False

        except Exception as e:
            logger.error(f"[Feishu] 卡片更新异常: {e}", exc_info=True)
            return False

    def _update_thinking_card_sync(self, card_msg_id: str,
                                     steps: list[dict]) -> bool:
        """
        更新 thinking 卡片为进度卡片（限流版）

        自动将步骤列表的最后一项标记为 running，已完成项标记为 done。

        Args:
            card_msg_id: 卡片消息 ID
            steps: 步骤列表（来自 map_stream_tokens_to_steps）

        Returns:
            是否成功
        """
        from integrations.feishu.card_builder import CardBuilder

        # 标记当前步骤
        for i, step in enumerate(steps):
            if i == len(steps) - 1:
                step["status"] = "running"
            else:
                step["status"] = "done"

        # 取标题（最后一个步骤的详情作为标题）
        header = "📋 正在处理..."
        if steps:
            last_step = steps[-1]
            detail = last_step.get("detail", "")
            label = last_step.get("label", "")
            if detail:
                header = f"{detail}"
            elif label:
                header = f"正在{label}..."

        card = CardBuilder.progress_card(header, steps)
        return self._update_card_sync(card_msg_id, card)

    # ============================================================
    # 卡片按钮回调
    # ============================================================

    def _on_card_action(self, data):
        """
        飞书卡片按钮点击回调（同步方法，必须快速返回）

        当用户在飞书中点击卡片按钮时，此回调被 WS 长连接触发。

        回调数据 (P2CardActionTrigger):
            data.event.operator.open_id   — 点击者 open_id
            data.event.action.value       — 按钮绑定的业务数据 (dict)
            data.event.context.open_chat_id    — 聊天ID
            data.event.context.open_message_id — 卡片消息ID

        返回 P2CardActionTriggerResponse 可控制 toast 弹窗和卡片刷新。

        参考:
          https://open.feishu.cn/document/uAjLw4CM/ukTMukTMukTM/feishu-cards/card-components/
            interactive-components/button
        """
        try:
            from lark_oapi.event.callback.model.p2_card_action_trigger import (
                P2CardActionTriggerResponse, CallBackToast,
            )

            action_value = data.event.action.value or {}
            operator = data.event.operator
            user_id = getattr(operator, 'open_id', None) or "unknown"
            chat_id = getattr(data.event.context, 'open_chat_id', None) or ""

            action_type = action_value.get("action", "unknown")

            logger.info(
                f"[Feishu] 卡片按钮点击: user={str(user_id)[:12]}..., "
                f"action={action_type}, chat={chat_id[:12]}..."
            )

            if action_type == "confirm_operation":
                # 确认操作 → 后台处理 + 更新卡片
                self._handle_card_confirm_async(action_value, chat_id,
                                                data.event.context.open_message_id)
                return P2CardActionTriggerResponse(
                    toast=CallBackToast(
                        type="success",
                        content="已确认，正在处理...",
                    )
                )

            elif action_type == "cancel_operation":
                # 取消操作 → 更新卡片
                from integrations.feishu.card_builder import CardBuilder
                cancel_card = CardBuilder.result_card(
                    title="🚫 已取消",
                    content_md="操作已被取消。如需重新提交，请发送消息。",
                    header_template="red",
                )
                self._update_card_sync(
                    data.event.context.open_message_id, cancel_card, force=True,
                )
                return P2CardActionTriggerResponse(
                    toast=CallBackToast(
                        type="info",
                        content="操作已取消",
                    )
                )

            else:
                logger.info(f"[Feishu] 未知卡片按钮: {action_type}")
                return P2CardActionTriggerResponse(
                    toast=CallBackToast(
                        type="info",
                        content="收到",
                    )
                )

        except Exception as e:
            logger.error(f"[Feishu] 卡片按钮回调异常: {e}", exc_info=True)

    def _handle_card_confirm_async(self, action_value: dict, chat_id: str,
                                     card_msg_id: str):
        """
        后台处理卡片确认操作

        目前支持：confirm_operation → 更新卡片为已处理状态
        可扩展：实际工单创建、SLA升级等业务逻辑
        """
        def _process():
            try:
                from integrations.feishu.card_builder import CardBuilder

                result_card = CardBuilder.result_card(
                    title="✅ 已确认处理",
                    content_md="您的请求已提交处理，稍后将有工程师与您联系。\n\n"
                               "如需加急处理，请拨打IT服务台热线。",
                    header_template="green",
                    note=f"操作时间: {CardBuilder._time_str()}",
                )
                self._update_card_sync(card_msg_id, result_card, force=True)

            except Exception as e:
                logger.error(f"[Feishu] 卡片确认后台处理失败: {e}")

        t = threading.Thread(target=_process, daemon=True)
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
            .register_p2_card_action_trigger(handler._on_card_action)
            .build()
        )

        # 保持对 data 的引用，防止被 GC
        handler._event_handler_ref = event_handler

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

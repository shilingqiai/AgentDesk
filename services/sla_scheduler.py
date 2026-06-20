"""
SLA 定时调度器 — 轻量 asyncio 实现

每 5 分钟执行一次全量 SLA 检测，不依赖 Celery/Redis 等外部组件。

设计:
- 单后台任务，asyncio.create_task 启动
- 可优雅停止（通过 asyncio.Event）
- 检测异常隔离 — 一次检测失败不影响后续周期
- 检测结果通过日志记录（生产环境可接入告警系统）

使用方式:
    from services.sla_scheduler import SLAScheduler

    await SLAScheduler.start()   # 在 app startup 中调用
    await SLAScheduler.stop()    # 在 app shutdown 中调用
"""

from __future__ import annotations

import asyncio
import logging

logger = logging.getLogger("sla.scheduler")

# 检测间隔（秒）
DEFAULT_INTERVAL = 300  # 5 分钟


class SLAScheduler:
    """SLA 定时检测调度器"""

    _task: asyncio.Task = None
    _stop_event: asyncio.Event = None
    _interval: int = DEFAULT_INTERVAL

    @classmethod
    async def start(cls, interval: int = None) -> None:
        """
        启动 SLA 定时调度。

        Args:
            interval: 检测间隔（秒），默认 300（5 分钟）
        """
        if cls._task is not None and not cls._task.done():
            logger.warning("[SLAScheduler] 已在运行，跳过重复启动")
            return

        if interval is not None:
            cls._interval = interval

        cls._stop_event = asyncio.Event()
        cls._task = asyncio.create_task(cls._run_loop())
        logger.info(
            f"[SLAScheduler] 已启动 — 检测间隔: {cls._interval}s "
            f"({cls._interval // 60} 分钟)"
        )

    @classmethod
    async def stop(cls) -> None:
        """停止 SLA 定时调度"""
        if cls._stop_event:
            cls._stop_event.set()
        if cls._task and not cls._task.done():
            try:
                await asyncio.wait_for(cls._task, timeout=5.0)
            except asyncio.TimeoutError:
                logger.warning("[SLAScheduler] 停止超时，强制取消")
                cls._task.cancel()
        logger.info("[SLAScheduler] 已停止")

    @classmethod
    async def _run_loop(cls) -> None:
        """主循环 — 每 interval 秒执行一次 SLA 检测"""
        logger.info("[SLAScheduler] 开始循环检测")

        # 首次延迟 30s，等系统完全启动
        await asyncio.sleep(30)

        while not cls._stop_event.is_set():
            try:
                await cls._run_check()
            except Exception as e:
                logger.error(
                    f"[SLAScheduler] SLA 检测异常: {e}", exc_info=True
                )

            # 等待下一次检测（可被 stop 中断）
            try:
                await asyncio.wait_for(
                    cls._stop_event.wait(), timeout=cls._interval
                )
                # 如果 wait 完成说明 stop_event 被设置了
                break
            except asyncio.TimeoutError:
                # timeout 说明 interval 到了，继续下一轮
                continue

    @classmethod
    async def _run_check(cls) -> None:
        """执行一次完整的 SLA 检测"""
        from services.sla_engine import SLAEngine

        logger.debug("[SLAScheduler] 执行 SLA 检测...")
        summary = await SLAEngine.check_all()

        if summary.breached_count > 0:
            logger.warning(
                f"[SLAScheduler] 检测到 {summary.breached_count} 项超时, "
                f"{summary.warning_count} 项预警, "
                f"{summary.active_sla_count} 项在途"
            )
        else:
            logger.debug(
                f"[SLAScheduler] SLA 检测完成 — "
                f"{summary.active_sla_count} 项在途, 0 超时"
            )

    @classmethod
    def is_running(cls) -> bool:
        """查询调度器是否在运行"""
        return cls._task is not None and not cls._task.done()

    @classmethod
    async def run_once(cls) -> dict:
        """
        手动触发一次 SLA 检测（供调试/API 使用）。

        Returns:
            SLA 状态概览 dict
        """
        from services.sla_engine import SLAEngine
        return await SLAEngine.get_sla_summary()

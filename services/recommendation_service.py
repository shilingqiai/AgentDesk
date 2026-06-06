"""
推荐调度服务

职责：
1. 定时生成用户行为推荐
2. 管理推荐调度任务
3. 提供手动触发推荐功能
"""

from datetime import datetime
from typing import List, Dict, Any, Optional
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.interval import IntervalTrigger
import logging

logger = logging.getLogger(__name__)


class RecommendationService:
    """推荐调度服务类（基于 APScheduler）"""

    def __init__(self):
        self._behavior_agent = None
        self.scheduler: Optional[BackgroundScheduler] = None

    @property
    def behavior_agent(self):
        """懒加载用户行为服务"""
        if self._behavior_agent is None:
            from services.user_behavior_service import UserBehaviorService
            self._behavior_agent = UserBehaviorService()
        return self._behavior_agent

    def generate_recommendations_job(self):
        """定时生成推荐的任务"""
        try:
            logger.info("开始执行定时推荐生成任务...")
            recommendations = []

            if recommendations:
                logger.info(f"成功生成 {len(recommendations)} 条推荐:")
                for rec in recommendations:
                    logger.info(f"- {rec['type']}: {rec['content'][:50]}...")
                return recommendations
            else:
                logger.info("本次没有生成新的推荐")
                return None

        except Exception as e:
            logger.error(f"定时推荐生成任务失败: {str(e)}")
            return None

    def start_scheduler(self) -> bool:
        """启动定时任务调度器"""
        if self.scheduler and self.scheduler.running:
            logger.warning("调度器已经在运行中")
            return False

        try:
            self.scheduler = BackgroundScheduler(daemon=True)

            # 每天 9:00、14:00、19:00 检查并生成推荐
            self.scheduler.add_job(
                self.generate_recommendations_job,
                CronTrigger(hour="9,14,19", minute="0"),
                id="daily_recommendation",
                name="每日推荐生成"
            )

            # 每2小时检查一次
            self.scheduler.add_job(
                self.generate_recommendations_job,
                IntervalTrigger(hours=2),
                id="interval_check",
                name="定期推荐检查"
            )

            self.scheduler.start()
            logger.info("推荐调度器已启动（APScheduler）")
            return True

        except Exception as e:
            logger.error(f"启动推荐调度器失败: {str(e)}")
            return False

    def stop_scheduler(self) -> bool:
        """停止定时任务调度器"""
        try:
            if self.scheduler and self.scheduler.running:
                self.scheduler.shutdown(wait=False)
                self.scheduler = None
            logger.info("推荐调度器已停止")
            return True
        except Exception as e:
            logger.error(f"停止推荐调度器失败: {str(e)}")
            return False

    def run_immediate_check(self) -> Optional[List[Dict[str, Any]]]:
        """立即执行一次推荐检查（用于测试或手动触发）"""
        logger.info("执行立即推荐检查...")
        return self.generate_recommendations_job()

    def get_status(self) -> Dict[str, Any]:
        """获取调度器状态"""
        jobs = []
        if self.scheduler:
            jobs = [
                {"id": j.id, "name": j.name, "next_run": str(j.next_run_time)}
                for j in self.scheduler.get_jobs()
            ]
        return {
            "is_running": self.scheduler.running if self.scheduler else False,
            "jobs": jobs,
            "total_jobs": len(jobs),
        }

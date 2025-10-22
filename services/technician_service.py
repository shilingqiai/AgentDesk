# utils/ai/technician_service.py

from typing import List, Dict, Any
from db.db_router import DatabaseRouter
import logging

logger = logging.getLogger(__name__)

class TechnicianService:
    """技师服务类 - 管理技师数据和默认初始化"""
    
    def __init__(self):
        self.db = DatabaseRouter()
        
        # 默认工程师数据（10人，覆盖主流IT技术栈）
        self.default_technicians = [
            {
                "name": "张伟",
                "gender": "男",
                "strength": "擅长后端架构与数据库性能优化，8年Python/Go开发经验，精通MySQL调优与高并发系统设计"
            },
            {
                "name": "王强",
                "gender": "男",
                "strength": "网络架构与安全专家，CCIE认证，精通防火墙策略、VPN部署和DDoS防护"
            },
            {
                "name": "李娜",
                "gender": "女",
                "strength": "数据库管理(MySQL/PostgreSQL/MongoDB)，擅长SQL调优、主从复制与灾备方案"
            },
            {
                "name": "赵敏",
                "gender": "女",
                "strength": "DevOps与CI/CD专家，精通K8s容器编排、Jenkins流水线和云原生架构"
            },
            {
                "name": "刘洋",
                "gender": "男",
                "strength": "前端开发与全栈架构，React/Vue技术栈，擅长性能优化与微前端架构设计"
            },
            {
                "name": "孙丽",
                "gender": "女",
                "strength": "云平台运维(AWS/Azure)，Terraform IaC实践，擅长云资源成本优化与自动化运维"
            },
            {
                "name": "周杰",
                "gender": "男",
                "strength": "系统安全审计与渗透测试，CISSP认证，擅长漏洞挖掘、安全加固与合规审计"
            },
            {
                "name": "吴婷",
                "gender": "女",
                "strength": "大数据平台运维(Hadoop/Spark/Flink)，精通数据管道搭建与实时流处理架构"
            },
            {
                "name": "郑斌",
                "gender": "男",
                "strength": "容器化与微服务治理(K8s/Istio)，Service Mesh架构设计，精通故障定位与容量规划"
            },
            {
                "name": "何静",
                "gender": "女",
                "strength": "IT服务管理(ITIL)与流程优化，10年服务台管理经验，擅长SLA体系搭建与知识库运营"
            }
        ]

    def initialize_default_technicians(self) -> bool:
        """初始化默认工程师数据"""
        try:
            # 检查是否已有工程师数据
            existing_technicians = self.db.technicians.get_all_technicians()

            if existing_technicians:
                logger.info(f"数据库中已有 {len(existing_technicians)} 位工程师，跳过初始化")
                return True

            logger.info("数据库中无工程师数据，开始初始化默认工程师")

            for tech_data in self.default_technicians:
                try:
                    tech_id = self.db.technicians.add_technician(
                        name=tech_data['name'],
                        gender=tech_data['gender'],
                        strength=tech_data['strength']
                    )
                    logger.debug(f"添加工程师: {tech_data['name']} (ID: {tech_id})")

                except Exception as e:
                    logger.error(f"添加工程师 {tech_data['name']} 失败: {e}")
                    return False

            final_count = len(self.db.technicians.get_all_technicians())
            logger.info(f"工程师初始化完成，共添加 {final_count} 位工程师")
            return True

        except Exception as e:
            logger.error(f"工程师初始化失败: {e}")
            return False

    def get_all_technicians(self) -> List[Dict[str, Any]]:
        """获取所有工程师信息"""
        return self.db.technicians.get_all_technicians()

    def get_technician_by_name(self, name: str) -> Dict[str, Any]:
        """根据姓名获取工程师信息"""
        return self.db.technicians.get_technician_by_name(name)

    def get_technician_by_id(self, technician_id: int) -> Dict[str, Any]:
        """根据ID获取工程师信息"""
        return self.db.technicians.get_technician_by_id(technician_id)

    def get_technician_schedules(self, technician_id: int, date) -> List[Dict[str, Any]]:
        """获取工程师指定日期的排班信息"""
        return self.db.technicians.get_technician_schedules(technician_id, date)

    def is_technician_available(self, technician_id: int, start_time, end_time) -> bool:
        """检查工程师在指定时间段是否可用"""
        return self.db.technicians.is_technician_available(technician_id, start_time, end_time)

    def add_technician(self, name: str, gender: str = None, strength: str = None) -> int:
        """添加新工程师"""
        return self.db.technicians.add_technician(name, gender, strength)

    def get_technicians_count(self) -> int:
        """获取工程师总数"""
        technicians = self.db.technicians.get_all_technicians()
        return len(technicians)

    def get_technician_by_id(self, technician_id: int) -> Dict[str, Any]:
        """根据ID获取技师信息"""
        return self.db.technicians.get_technician_by_id(technician_id)

    def get_technician_schedules(self, technician_id: int, date) -> List[Dict[str, Any]]:
        """获取技师指定日期的排班信息"""
        return self.db.technicians.get_technician_schedules(technician_id, date)

    def is_technician_available(self, technician_id: int, start_time, end_time) -> bool:
        """检查技师在指定时间段是否可用"""
        return self.db.technicians.is_technician_available(technician_id, start_time, end_time)

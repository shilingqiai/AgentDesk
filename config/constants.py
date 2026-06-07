"""
应用程序常量

仅保留业务数据常量（如忙碌时段配置）。
状态管理已迁移到 agents/graph_workflow.py 的 TicketState。
"""

# 工程师忙碌时段配置
# { engineer_id: [ {"start": "...", "end": "..."} ] }
busy_periods_dict: dict = {}

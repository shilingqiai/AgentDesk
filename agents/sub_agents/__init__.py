"""
专业子Agent — 企业员工AI服务台

已注册Agent:
- IT咨询: 网络/系统/软件故障排查 (60%+查询)
- HR咨询: 请假/福利/入职/报销政策 (20%)
- 行政服务: 会议室/访客/食堂/设施 (15%+)
"""

from .it_consultant import ITConsultantSubAgent
from .hr_consultant import HRConsultantSubAgent
from .facilities import FacilitiesSubAgent

__all__ = [
    "ITConsultantSubAgent",
    "HRConsultantSubAgent",
    "FacilitiesSubAgent",
]

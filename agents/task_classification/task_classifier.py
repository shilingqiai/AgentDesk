"""
任务分类器 - 专门负责判断用户请求的类型

职责：
1. 接收用户输入，分析其意图
2. 根据预定义的分类规则，将任务归类为：
   - appointment（预约任务）
   - query（查询任务）  
   - pay（支付任务）
   - statistics（统计任务）
   - other（其他任务）
3. 提供清晰的分类结果和置信度
"""

from langchain.prompts import PromptTemplate
from langchain_core.language_models.chat_models import BaseChatModel
from typing import Dict, Any


class TaskClassifier:
    """任务分类器 - 使用LLM进行智能任务分类"""
    
    def __init__(self, llm: BaseChatModel):
        self.llm = llm
        self._initialize_prompt()
        self.chain = self.prompt | self.llm
    
    def _initialize_prompt(self):
        """初始化分类提示词模板"""
        self.prompt = PromptTemplate(
            input_variables=["task"],
            template=(
                "你是一个企业IT工单调度系统的智能分类器。请严格按以下规则分类用户输入：\n\n"
                "核心判断标准 — 用户是在'问方法'还是'要派人'？\n"
                "- query（查询任务）：用户在请教/询问如何解决问题，想自己处理或了解信息\n"
                "- appointment（工单任务）：用户明确要求派工程师处理、提交工单、安排人来做\n\n"
                "分类规则：\n"
                "1. query = 咨询/请教类：\n"
                "   - '怎么排查'、'如何处理'、'步骤是什么'、'为什么'、'是什么原因'\n"
                "   - 询问操作方法、故障排查步骤、配置方法、系统使用指南\n"
                "   - 例：'VPN连接失败怎么排查？' → query\n"
                "   - 例：'数据库性能优化有哪些方法？' → query\n"
                "2. appointment = 派单/工单类：\n"
                "   - 明确说'请帮我'、'派工程师'、'提交工单'、'安排人处理'、'需要人来看'\n"
                "   - 包含明确的求助意愿（要派工程师来做）\n"
                "   - 例：'请派网络工程师处理VPN故障' → appointment\n"
                "   - 例：'帮我提交一个数据库问题的工单' → appointment\n"
                "3. pay = 通知确认类：系统通知、确认回复\n"
                "4. statistics = 统计类：工程师上报完成、汇报工作\n"
                "5. other = 完全无关的话题\n\n"
                "只输出一个英文单词：appointment / query / pay / statistics / other\n\n"
                "任务内容：{task}"
            )
        )
    
    async def classify_task(self, task: str) -> str:
        """
        分类任务
        
        Args:
            task: 用户输入的任务内容
            
        Returns:
            str: 分类结果 ('appointment', 'query', 'pay', 'statistics', 'other')
        """
        try:
            category_msg = await self.chain.ainvoke({"task": task})
            category = category_msg.content.strip().lower()
            
            # 验证分类结果是否有效
            valid_categories = {'appointment', 'query', 'pay', 'statistics', 'other'}
            if category not in valid_categories:
                return 'other'  # 默认归类为其他
                
            return category
            
        except Exception as e:
            print(f"任务分类失败: {str(e)}")
            return 'other'  # 发生错误时默认归类为其他
    
    def get_category_description(self, category: str) -> str:
        """获取分类类别的描述信息"""
        descriptions = {
            'appointment': '工单任务 - 用户提交的工单调度相关请求',
            'query': '查询任务 - 用户咨询IT服务信息、故障处理等',
            'pay': '通知任务 - 工单分配后的通知确认',
            'statistics': '统计任务 - 工程师上报工单完成状态',
            'other': '其他任务 - 与IT运维服务无关的请求'
        }
        return descriptions.get(category, '未知任务类型')

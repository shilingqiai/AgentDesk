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
                "你是一个企业IT工单调度系统的助手，你会处理来自用户和工程师的消息，你的任务是对本次任务进行分类。\n"
                "用户可能会咨询IT服务信息、故障处理方法、服务目录等，这类任务归类为查询任务。\n"
                "用户可能会提交工单请求，比如'请帮我派单处理VPN连接故障，紧急'，这类任务归类为工单任务。\n"
                "调度引擎也可能发来任务，告知用户已分配某位工程师处理工单，这类任务归类为通知任务。\n"
                "工程师可能会发来任务，比如告知某个工单需要升级处理，这类任务归类为工单任务。\n"
                "工程师也可能告知已完成当前工单，这类任务归类为统计任务。\n"
                "如果输入的任务与上述都无关，请归类为其它任务。\n"
                "请将以下任务归类为以下类别，输出只能选择以下之一：\n"
                "1. appointment（工单调度任务）\n"
                "2. query（查询任务）\n"
                "3. pay（通知任务）\n"
                "4. statistics（统计任务）\n"
                "5. other（其它任务）\n"
                "只返回类别英文名。\n\n"
                "举例说明：假如task为'我需要指派一位网络工程师处理数据库连接超时的问题，P1优先级'，则输出appointment。\n"
                "假如输入为'请问怎么重置VPN密码'，则输出query。\n"
                "以下是本次归类任务:\n"
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

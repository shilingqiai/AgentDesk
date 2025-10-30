"""
提示词构建器

负责构建各种类型的提示词
"""

from typing import List, Dict, Any


class PromptBuilder:
    """提示词构建器"""
    
    def __init__(self):
        self.system_prompt = self._create_system_prompt()
        self.classification_prompt_template = self._create_classification_prompt_template()
    
    def _create_system_prompt(self) -> str:
        """创建系统提示词"""
        return (
            "你是一个企业IT运维服务台的智能助手，负责为客户解答关于IT服务、故障排查、系统使用、账号管理、网络配置等相关问题。"
            "我会为你提供相关的知识库信息，请基于这些信息来回答用户的问题。"
            "如果知识库中没有相关信息，请提供合理的兜底回答，比如："
            "- 对于故障问题：抱歉，该问题需要进一步排查，建议您提交工单，我们会安排工程师处理。"
            "- 对于权限问题：请通过OA系统提交权限申请，审批通过后自动开通。"
            "- 对于其他缺失信息：建议您提交工单或联系IT服务台热线获取进一步支持。"
            "请用专业、清晰、简洁的语言回复用户。"
            "如果用户的问题与IT运维服务完全无关，请礼貌地告知用户你只能回答IT运维相关问题。"
            "回答时要自然流畅，不要明显地表现出是在查阅资料。"
        )

    def _create_classification_prompt_template(self) -> str:
        """创建分类提示词模板"""
        return (
            "你是一个分类器，判断用户输入是否是关于企业IT运维的咨询类问题。\n"
            "咨询类问题包括：故障排查方法、系统使用指南、账号管理、网络配置、安全策略、服务目录、工作时间等。\n"
            "非咨询类问题包括：工单提交（我要报修、帮我派单等）、投诉建议、或与IT完全无关的话题。\n"
            "如果是咨询类问题，回答'YES'。如果是工单类问题或完全无关问题，回答'NO'。\n"
            "只回答YES或NO。\n\n"
            "用户输入：{user_input}"
        )

    def build_consultation_prompt(self, user_input: str, knowledge_docs: List[Dict[str, Any]]) -> str:
        """构建咨询提示词"""
        context = self._build_knowledge_context(knowledge_docs)
        return f"{self.system_prompt}\n\n{context}\n用户问题：{user_input}\n\n请回答用户的问题。"

    def build_classification_prompt(self, user_input: str) -> str:
        """构建分类提示词"""
        return self.classification_prompt_template.format(user_input=user_input)

    def _build_knowledge_context(self, knowledge_docs: List[Dict[str, Any]]) -> str:
        """构建知识库上下文"""
        if not knowledge_docs:
            return "没有找到直接相关的知识库信息，请基于你对IT运维的专业知识回答。"

        context = "\n以下是相关的知识库信息：\n"
        for i, doc in enumerate(knowledge_docs, 1):
            context += f"{i}. {doc['content']}\n"
        context += "\n请基于以上信息回答用户问题。如果知识库信息不足以回答问题，请基于你对IT运维的一般了解来补充回答。\n"

        return context

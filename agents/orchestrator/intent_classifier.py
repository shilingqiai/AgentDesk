"""
意图分类器 — Orchestrator 的第一阶段

参考 Microsoft Copilot Studio 的意图识别层：
- 使用 LLM 分析用户输入，判断意图和紧急度
- 从现有 task_classifier.py 迁移 prompt 逻辑
- 输出结构化的意图分类结果（供 TaskPlanner 使用）

关键改进：
- 不只是分类类别，还输出紧急度评估和关键词
- 分类结果直接用于 Agent 路由决策
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Optional

from langchain.prompts import PromptTemplate
from langchain_core.language_models.chat_models import BaseChatModel

logger = logging.getLogger("orchestrator.intent_classifier")


@dataclass
class IntentResult:
    """意图分类结果"""
    category: str          # "it_support" | "ticket_request" | "analytics" | "hr_inquiry" | "other"
    urgency: str           # "high" | "medium" | "low"
    confidence: float      # 0.0 - 1.0
    keywords: list[str] = field(default_factory=list)
    summary: str = ""      # 一句话摘要
    target_agent: str = ""  # 推荐的 Agent ID


class IntentClassifier:
    """
    意图分类器

    职责：
    1. 分析用户输入，识别意图类别
    2. 评估紧急程度
    3. 提取关键词
    4. 推荐目标 Agent

    使用方式：
        classifier = IntentClassifier(llm)
        result = await classifier.classify(user_input, agent_descriptions)
    """

    def __init__(self, llm: BaseChatModel):
        self.llm = llm
        self._initialize_prompt()

    def _initialize_prompt(self):
        """初始化分类提示词模板"""
        self.classify_prompt = PromptTemplate(
            input_variables=["agent_descriptions", "user_input"],
            template=(
                "你是一个企业智能服务台（Enterprise AI Service Desk）的调度编排器。\n\n"
                "你的任务是：分析用户输入，判断意图类别、紧急程度，并推荐最合适的专业Agent处理。\n\n"
                "## 可用的专业Agent\n\n"
                "{agent_descriptions}\n\n"
                "## 分类规则\n\n"
                "1. **IT支持类** (→ it_consultant)：\n"
                "   - 询问IT故障排查方法、软件使用指南、系统配置\n"
                "   - 关键词：怎么、如何、排查、修复、配置、VPN、网络、密码、报错\n"
                "   - 例：'VPN连不上怎么排查？' → it_consultant\n\n"
                "2. **工单派发类** (→ ticket_dispatch)：\n"
                "   - 明确要求派工程师、提交工单、安排人员处理\n"
                "   - 关键词：派工程师、提交工单、帮我修、需要人来、安排处理\n"
                "   - 例：'请派网络工程师处理VPN故障' → ticket_dispatch\n\n"
                "3. **效能分析类** (→ analytics)：\n"
                "   - 查询工单统计、工程师效能、SLA合规情况\n"
                "   - 关键词：统计、分析、效能、报表、工单量、处理速度\n"
                "   - 例：'最近工单处理效率怎么样？' → analytics\n\n"
                "4. **HR咨询类** (→ hr_consultant)：\n"
                "   - 请假政策、福利查询、入职指引、HR流程\n"
                "   - 关键词：请假、年假、福利、入职、社保、公积金、报销\n"
                "   - 例：'请年假需要什么流程？' → hr_consultant\n\n"
                "5. **其他/无关** (→ none)：\n"
                "   - 与IT、HR、工单调度完全无关的内容\n\n"
                "## 紧急度判断\n"
                "- **high**: 影响核心业务、有截止时间、多人受影响、系统宕机\n"
                "- **medium**: 影响工作效率但可暂时绕过\n"
                "- **low**: 一般咨询、非紧急问题\n\n"
                "## 输出格式\n\n"
                "请严格输出以下 JSON 格式（不要添加任何其他文字）：\n"
                "{{\n"
                '  "category": "it_support|ticket_request|analytics|hr_inquiry|other",\n'
                '  "urgency": "high|medium|low",\n'
                '  "confidence": 0.0-1.0,\n'
                '  "keywords": ["关键词1", "关键词2"],\n'
                '  "summary": "一句话描述用户需求",\n'
                '  "target_agent": "agent_id 或 none"\n'
                "}}\n\n"
                "用户输入：{user_input}"
            ),
        )

    async def classify(
        self,
        user_input: str,
        agent_descriptions: str = "",
    ) -> IntentResult:
        """
        分类用户输入

        Args:
            user_input: 用户输入文本
            agent_descriptions: 可用 Agent 的描述文本（来自 AgentRegistry）

        Returns:
            IntentResult 分类结果
        """
        try:
            chain = self.classify_prompt | self.llm
            response = await chain.ainvoke({
                "agent_descriptions": agent_descriptions or "（从注册中心加载）",
                "user_input": user_input,
            })

            # 解析 JSON 响应
            content = response.content.strip()

            # 尝试从可能的 markdown 代码块中提取 JSON
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0].strip()
            elif "```" in content:
                content = content.split("```")[1].split("```")[0].strip()

            result = json.loads(content)

            return IntentResult(
                category=result.get("category", "other"),
                urgency=result.get("urgency", "medium"),
                confidence=float(result.get("confidence", 0.5)),
                keywords=result.get("keywords", []),
                summary=result.get("summary", ""),
                target_agent=result.get("target_agent", ""),
            )

        except (json.JSONDecodeError, KeyError) as e:
            logger.warning(f"意图分类 JSON 解析失败: {e}，使用规则兜底")
            return self._rule_based_fallback(user_input)

        except Exception as e:
            logger.error(f"意图分类失败: {e}")
            return IntentResult(
                category="other",
                urgency="medium",
                confidence=0.0,
                keywords=[],
                summary=f"分类失败: {str(e)}",
                target_agent="",
            )

    def _rule_based_fallback(self, user_input: str) -> IntentResult:
        """规则兜底：当 LLM 输出无法解析时使用简单的关键词匹配"""
        user_lower = user_input.lower()

        # IT 关键词
        it_keywords = ["怎么", "如何", "排查", "修复", "配置", "vpn", "网络",
                       "密码", "报错", "故障", "安装", "连接", "电脑", "系统"]
        # 工单关键词
        ticket_keywords = ["派工程师", "提交工单", "帮我修", "需要人来",
                           "安排", "处理一下", "上门", "帮忙看"]
        # 分析关键词
        analytics_keywords = ["统计", "分析", "效能", "报表", "工单量",
                              "处理速度", "sla", "负载"]
        # HR 关键词
        hr_keywords = ["请假", "年假", "福利", "入职", "社保", "公积金",
                       "报销", "工资", "考勤"]

        # 计数匹配
        scores = {
            "it_support": sum(1 for kw in it_keywords if kw in user_lower),
            "ticket_request": sum(1 for kw in ticket_keywords if kw in user_lower),
            "analytics": sum(1 for kw in analytics_keywords if kw in user_lower),
            "hr_inquiry": sum(1 for kw in hr_keywords if kw in user_lower),
        }

        best_category = max(scores, key=scores.get)
        best_score = scores[best_category]

        if best_score == 0:
            return IntentResult(
                category="other", urgency="medium", confidence=0.3,
                keywords=[], summary="无法识别意图",
                target_agent="",
            )

        # 紧急度简单判断
        urgency = "medium"
        high_urgency_words = ["紧急", "宕机", "影响业务", "马上", "立刻",
                              "deadline", "截止", "急"]
        if any(w in user_lower for w in high_urgency_words):
            urgency = "high"

        return IntentResult(
            category=best_category,
            urgency=urgency,
            confidence=min(best_score * 0.2, 0.7),  # 规则匹配的置信度上限
            keywords=[],
            summary=f"规则匹配: {best_category}",
            target_agent=self._category_to_agent(best_category),
        )

    def _category_to_agent(self, category: str) -> str:
        """将分类类别映射到 Agent ID"""
        mapping = {
            "it_support": "it_consultant",
            "ticket_request": "ticket_dispatch",
            "analytics": "analytics",
            "hr_inquiry": "hr_consultant",
        }
        return mapping.get(category, "")

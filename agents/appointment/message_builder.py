"""
消息构建器

负责构建工单调度相关的各种响应消息
"""

from typing import Dict, Any, List


class MessageBuilder:
    """消息构建器"""

    def __init__(self):
        self.missing_info_prompts = {
            "gender": "请问您需要指定工程师的性别偏好吗？",
            "start_time": "请问您希望工单什么时候开始处理？",
            "duration": "请问预计需要多长时间处理？",
            "project": "请问工单类型是什么？（故障排查/需求变更/技术咨询/系统巡检）",
            "preference": "您对工程师的技术栈有偏好吗？（如网络/数据库/后端等）"
        }

    def create_appointment_success_message(self, tech: Dict[str, Any]) -> str:
        """创建工单派发成功消息"""
        if tech.get('is_recommendation'):
            original_tech = tech.get('original_technician', {})
            return (f"\n系统：已为您派发工单给工程师：{tech['name']}（{tech.get('skill_tag', 'IT运维')}）。派单成功！"
                    f"（原指定的{original_tech.get('name', '')}工程师当前负载过高，{tech['name']}具备相同领域的专业能力）"
                    "请关注工单处理进度，SLA响应时限为2小时。\n")
        else:
            return (f"\n系统：已为您派发工单给工程师：{tech['name']}（{tech.get('skill_tag', 'IT运维')}）。派单成功！"
                    "请关注工单处理进度，SLA响应时限为2小时。\n")

    def create_technician_recommendation_message(self, original_tech: Dict[str, Any],
                                               recommended_tech: Dict[str, Any],
                                               appointment_history: Dict[str, Any],
                                               llm=None) -> str:
        """创建工程师推荐消息，使用LLM生成个性化措辞"""
        project = appointment_history.get('project', 'IT支持服务')
        start_time = appointment_history.get('start_time', '')

        if llm:
            try:
                prompt = f"""
作为一个专业的工单调度助手，用户想派单给{original_tech['name']}工程师处理{project}工单，但{original_tech['name']}工程师当前负载已满。

我找到了一位技能匹配的工程师：
- 姓名：{recommended_tech['name']}
- 技术领域：{recommended_tech.get('strength', '')}

原指定工程师技能：{original_tech.get('strength', '')}

请帮我生成一段专业、简洁的推荐话术，告诉用户原指定工程师负载已满，但推荐工程师具备相同领域技能，可以立即处理，询问用户是否接受推荐。

要求：
1. 语气专业、高效
2. 突出推荐工程师的技能匹配度
3. 明确询问用户意愿
4. 字数控制在80字以内
"""

                response = llm.invoke(prompt)
                if hasattr(response, 'content'):
                    generated_msg = response.content.strip()
                    if generated_msg:
                        return f"\n系统：{generated_msg}\n"

            except Exception as e:
                print(f"LLM生成推荐消息失败: {e}")

        return (f"\n系统：{original_tech['name']}工程师当前负载已满，无法接单。"
                f"推荐{recommended_tech['name']}工程师（技能：{recommended_tech.get('strength', 'IT运维')}），"
                f"与您的工单需求高度匹配，当前可立即接单。请问是否接受推荐？\n")

    def create_recommendation_declined_message(self, llm=None) -> str:
        """创建用户拒绝推荐时的消息"""
        if llm:
            try:
                prompt = """
用户拒绝了我推荐的工程师，请帮我生成一段专业、高效的回复，表达理解并提供其他解决方案。

要求：
1. 表达理解用户的选择
2. 提供其他解决方案（如等待原工程师、放宽技能要求等）
3. 保持专业和高效的语气
4. 字数控制在60字以内
"""
                response = llm.invoke(prompt)
                if hasattr(response, 'content'):
                    generated_msg = response.content.strip()
                    if generated_msg:
                        return f"\n系统：{generated_msg}\n"
            except Exception as e:
                print(f"LLM生成拒绝消息失败: {e}")

        return "\n系统：好的，我理解您的选择。您可以等待原工程师释放负载后处理，或我为您筛选其他匹配的工程师。\n"

    def create_appointment_failure_message(self, technician_name: str) -> str:
        """创建工单派发失败消息"""
        if technician_name and technician_name != "未知":
            from services.appointment_service import AppointmentService
            appointment_service = AppointmentService()
            specific_tech = appointment_service.get_technician_by_name(technician_name)
            if specific_tech:
                return f"\n系统：{technician_name}工程师当前负载已满或不在值班。请等待或让我为您推荐其他匹配的工程师。\n"
            else:
                return f"\n系统：未找到名为'{technician_name}'的工程师。请确认姓名，或让我根据技能需求为您推荐。\n"
        else:
            return "\n系统：当前没有符合条件的工程师空闲。请调整需求或稍后重试。\n"

    def create_missing_info_questions(self, missing_info: List[str]) -> str:
        """根据缺失信息创建询问"""
        questions = [self.missing_info_prompts.get(field, f"请补充{field}信息") for field in missing_info]
        return "\n" + " ".join(questions) + "\n"

    def create_unrelated_message(self) -> str:
        """创建无关请求的消息"""
        return "[REPLY][派单引擎]抱歉，我无法处理该问题。我只能帮您处理IT工单调度相关的请求。请问您需要提交工单吗？\n"

    def create_parse_error_message(self) -> str:
        """创建解析错误消息"""
        return "[REPLY][派单引擎]\n系统：工单信息解析失败，请重试。\n"

    def create_save_failure_message(self) -> str:
        """创建保存失败消息"""
        return "\n系统：工单保存失败，请重试。\n"

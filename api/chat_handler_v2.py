"""
LangGraph-based chat handler (v2)

使用 LangGraph 状态图管理工单调度工作流
相比 v1 的手动状态管理，v2 提供：
- 类型安全的状态定义
- 条件路由与Agent切换
- 检查点持久化（支持对话恢复）
- 统一的流式输出
"""

from agents.graph_workflow import workflow_runner


async def process_user_input_v2(user_input: str, thread_id: str = "default"):
    """
    LangGraph 版本的用户输入处理（流式）

    相比 v1 版本的优势：
    - Agent间切换由Graph自动管理，无需手写状态转移
    - 支持 thread_id 隔离多用户会话
    - 检查点持久化，服务重启后可恢复对话
    """
    async for token in workflow_runner.run_stream(user_input, thread_id):
        yield token


async def process_user_input_v2_sync(user_input: str, thread_id: str = "default"):
    """同步版本，返回完整结果"""
    return await workflow_runner.run(user_input, thread_id)


def get_conversation_state(thread_id: str = "default"):
    """获取当前对话状态"""
    return workflow_runner.get_state(thread_id)


def reset_conversation(thread_id: str = "default"):
    """重置对话"""
    workflow_runner.reset(thread_id)

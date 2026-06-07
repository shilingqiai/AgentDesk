"""
任务分类API — 使用编排器Router进行意图分类
"""
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

router = APIRouter(prefix="/api/task", tags=["任务分类"])


class TaskRequest(BaseModel):
    message: str


@router.post("/classify")
async def classify_task(request: TaskRequest):
    """分类任务 — 使用语义路由器"""
    try:
        from agents.orchestrator.router import Router
        from agents.orchestrator.agent_registry import agent_registry

        router = Router()
        agent_descriptions = agent_registry.get_routing_descriptions()
        result = await router.route(request.message, agent_descriptions)

        return {
            "message": "任务分类成功",
            "data": {
                "track": result.track,
                "agent_id": result.agent_id,
                "reason": result.reason,
                "category": result.category,
                "urgency": result.urgency,
                "confidence": result.confidence,
            },
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

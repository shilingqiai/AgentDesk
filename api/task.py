"""
任务分类API — 使用 Copilot Studio 编排器
"""
from fastapi import APIRouter, HTTPException
from .core.response_models import (
    TaskClassificationRequest,
    TaskClassificationResponse,
    DataResponse,
)

router = APIRouter(prefix="/api/task", tags=["任务分类"])


@router.post("/classify", response_model=DataResponse)
async def classify_task(request: TaskClassificationRequest):
    """分类任务 — 使用编排器的意图分类器"""
    try:
        from agents.orchestrator.intent_classifier import IntentClassifier
        from agents.orchestrator.agent_registry import agent_registry
        from config.model_provider import create_chat_model

        llm = create_chat_model(model_type="router", temperature=0)
        classifier = IntentClassifier(llm)

        agent_descriptions = agent_registry.get_routing_descriptions()
        result = await classifier.classify(request.message, agent_descriptions)

        return DataResponse(
            message="任务分类成功",
            data={
                "category": result.category,
                "urgency": result.urgency,
                "confidence": result.confidence,
                "target_agent": result.target_agent,
                "summary": result.summary,
                "keywords": result.keywords,
            },
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

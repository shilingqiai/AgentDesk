"""
知识库管理API — 支持 CRUD / 向量搜索 / 批量导入 / 用户反馈
"""
from fastapi import APIRouter, HTTPException, UploadFile, File
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
import csv
import io

router = APIRouter(prefix="/api/knowledge", tags=["知识库管理"])


class SearchRequest(BaseModel):
    query: str
    top_k: int = 5
    category: str = None


class KnowledgeCreateRequest(BaseModel):
    content: str
    category: str
    keywords: list[str] = []


class BatchImportRequest(BaseModel):
    """批量导入请求"""
    items: List[KnowledgeCreateRequest] = Field(
        ..., description="知识条目列表", min_length=1, max_length=100
    )


class FeedbackRequest(BaseModel):
    """文档反馈"""
    helpful: bool = Field(..., description="是否有帮助")
    comment: Optional[str] = Field(default="", description="补充说明")


# ============================================================
# 查询端点
# ============================================================

@router.get("/")
async def get_all_knowledge():
    """获取所有知识条目"""
    try:
        from services.knowledge_service import KnowledgeService
        knowledge_service = KnowledgeService()
        if not knowledge_service.initialized:
            await knowledge_service.initialize()
        entries = knowledge_service.get_all_documents()

        try:
            categories = knowledge_service.get_all_categories()
        except Exception as e:
            print(f"获取categories失败: {e}")
            categories = []

        return {
            "documents": entries or [],
            "categories": categories or [],
            "total_count": len(entries) if entries else 0,
            "status": "success"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取知识库失败: {str(e)}")


@router.get("/{knowledge_id}")
async def get_knowledge(knowledge_id: int):
    """获取特定知识条目"""
    try:
        from services.knowledge_service import KnowledgeService
        knowledge_service = KnowledgeService()
        if not knowledge_service.initialized:
            await knowledge_service.initialize()
        entry = knowledge_service.get_document(knowledge_id)
        if not entry:
            raise HTTPException(status_code=404, detail="知识条目不存在")
        return {
            "status": "success",
            "data": entry
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取知识条目失败: {str(e)}")


@router.post("/search")
async def search_knowledge(request: SearchRequest):
    """搜索知识库（向量检索）"""
    try:
        from services.knowledge_service import KnowledgeService
        knowledge_service = KnowledgeService()
        if not knowledge_service.initialized:
            await knowledge_service.initialize()
        results = await knowledge_service.search(
            request.query,
            top_k=request.top_k,
            category=request.category,
        )
        return {
            "status": "success",
            "results": results,
            "data": results,
            "total_found": len(results),
            "query": request.query,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"搜索知识库失败: {str(e)}")


# ============================================================
# CRUD 端点
# ============================================================

@router.post("/")
async def add_knowledge(item: KnowledgeCreateRequest):
    """添加新的知识条目"""
    try:
        from services.knowledge_service import KnowledgeService
        knowledge_service = KnowledgeService()
        if not knowledge_service.initialized:
            await knowledge_service.initialize()
        result = await knowledge_service.add_document(
            content=item.content,
            category=item.category,
            keywords=item.keywords,
        )
        return {
            "status": "success",
            "message": "知识条目添加成功",
            "data": result
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"添加知识条目失败: {str(e)}")


@router.put("/{knowledge_id}")
async def update_knowledge(knowledge_id: int, item: KnowledgeCreateRequest):
    """更新知识条目"""
    try:
        from services.knowledge_service import KnowledgeService
        knowledge_service = KnowledgeService()
        if not knowledge_service.initialized:
            await knowledge_service.initialize()
        result = await knowledge_service.update_document(
            doc_id=knowledge_id,
            content=item.content,
            category=item.category,
            keywords=item.keywords,
        )
        if not result:
            raise HTTPException(status_code=404, detail="知识条目不存在")
        return {
            "status": "success",
            "message": "知识条目更新成功",
            "data": result
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"更新知识条目失败: {str(e)}")


@router.delete("/{knowledge_id}")
async def delete_knowledge(knowledge_id: int):
    """删除知识条目"""
    try:
        from services.knowledge_service import KnowledgeService
        knowledge_service = KnowledgeService()
        if not knowledge_service.initialized:
            await knowledge_service.initialize()
        result = await knowledge_service.delete_document(knowledge_id)
        if not result:
            raise HTTPException(status_code=404, detail="知识条目不存在")
        return {
            "status": "success",
            "message": "知识条目删除成功"
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"删除知识条目失败: {str(e)}")


# ============================================================
# 批量导入
# ============================================================

@router.post("/batch", summary="批量导入知识条目（JSON）")
async def batch_import(request: BatchImportRequest):
    """批量导入知识条目，最多 100 条"""
    try:
        from services.knowledge_service import KnowledgeService
        knowledge_service = KnowledgeService()
        if not knowledge_service.initialized:
            await knowledge_service.initialize()

        success_count = 0
        failed_items = []

        for i, item in enumerate(request.items):
            try:
                await knowledge_service.add_document(
                    content=item.content,
                    category=item.category,
                    keywords=item.keywords,
                )
                success_count += 1
            except Exception as e:
                failed_items.append({"index": i, "error": str(e)})

        return {
            "status": "success",
            "message": f"批量导入完成: {success_count}/{len(request.items)} 条成功",
            "data": {
                "total": len(request.items),
                "success": success_count,
                "failed": len(failed_items),
                "failed_items": failed_items[:10],  # 只返回前10条失败详情
            },
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"批量导入失败: {str(e)}")


@router.post("/import/csv", summary="CSV 文件批量导入")
async def import_csv(file: UploadFile = File(...)):
    """
    上传 CSV 文件批量导入知识条目

    CSV 格式要求 (UTF-8 编码):
        content, category, keywords (分号分隔)
    """
    try:
        content = await file.read()
        text = content.decode("utf-8-sig")  # 支持 BOM
        reader = csv.DictReader(io.StringIO(text))

        from services.knowledge_service import KnowledgeService
        knowledge_service = KnowledgeService()
        if not knowledge_service.initialized:
            await knowledge_service.initialize()

        success_count = 0
        failed_rows = []

        for row_num, row in enumerate(reader, start=2):  # 1-based, skip header
            try:
                content_text = row.get("content", "").strip()
                category = row.get("category", "").strip()
                keywords_raw = row.get("keywords", "")
                keywords = [k.strip() for k in keywords_raw.split(";") if k.strip()]

                if not content_text or not category:
                    failed_rows.append({"row": row_num, "error": "content 或 category 为空"})
                    continue

                await knowledge_service.add_document(
                    content=content_text,
                    category=category,
                    keywords=keywords,
                )
                success_count += 1
            except Exception as e:
                failed_rows.append({"row": row_num, "error": str(e)})

        return {
            "status": "success",
            "message": f"CSV 导入完成: {success_count} 条成功",
            "data": {
                "success": success_count,
                "failed": len(failed_rows),
                "failed_rows": failed_rows[:10],
            },
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"CSV 导入失败: {str(e)}")


# ============================================================
# 用户反馈
# ============================================================

@router.post("/{knowledge_id}/feedback", summary="提交文档反馈")
async def submit_feedback(knowledge_id: int, feedback: FeedbackRequest):
    """用户对知识库文档提交有帮助/无帮助反馈"""
    try:
        from services.knowledge_service import KnowledgeService
        knowledge_service = KnowledgeService()
        if not knowledge_service.initialized:
            await knowledge_service.initialize()

        # 验证文档存在
        entry = knowledge_service.get_document(knowledge_id)
        if not entry:
            raise HTTPException(status_code=404, detail="知识条目不存在")

        # 记录反馈
        knowledge_service.record_feedback(
            doc_id=knowledge_id,
            is_helpful=feedback.helpful,
            comment=feedback.comment or "",
        )

        return {
            "status": "success",
            "message": "反馈已记录，感谢您的参与！",
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"提交反馈失败: {str(e)}")


@router.get("/{knowledge_id}/feedback", summary="获取文档反馈统计")
async def get_feedback_stats(knowledge_id: int):
    """获取某条文档的反馈统计"""
    try:
        from services.knowledge_service import KnowledgeService
        knowledge_service = KnowledgeService()
        if not knowledge_service.initialized:
            await knowledge_service.initialize()

        stats = knowledge_service.get_feedback_stats(knowledge_id)
        return {
            "status": "success",
            "data": stats,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取反馈统计失败: {str(e)}")

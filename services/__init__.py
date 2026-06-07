"""
业务服务层 — 企业员工AI服务台

提供知识库管理、文本向量化等核心服务。
"""
from .text_embedding import embed_input, find_best_match_indices
from .knowledge_service import KnowledgeService

__all__ = [
    'embed_input',
    'find_best_match_indices',
    'KnowledgeService',
]

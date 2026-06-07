"""
数据库模块 — 企业员工AI服务台

提供知识库数据模型、仓库和路由器。
"""
from .db_router import DatabaseRouter, KnowledgeDBRouter
from .repositories import KnowledgeRepository
from .base import SessionManager
from .models import Base, KnowledgeDocument

__all__ = [
    'DatabaseRouter',
    'KnowledgeDBRouter',
    'KnowledgeRepository',
    'SessionManager',
    'Base',
    'KnowledgeDocument',
]

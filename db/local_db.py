"""
本地数据库兼容层 — 企业员工AI服务台

提供知识库的兼容性访问接口。
"""
from .repositories import KnowledgeRepository
from .base import SessionManager
from typing import Dict, List, Optional
import warnings


class LocalKnowledgeDB:
    """知识库数据库的兼容性类"""

    def __init__(self, db_path='sqlite:///data/ticket_dispatch.db'):
        self.session_manager = SessionManager(db_path)
        self.repo = KnowledgeRepository(self.session_manager)

    def add_document(self, content: str, category: str, keywords=None, embedding=None) -> int:
        return self.repo.add_document(content, category, keywords, embedding)

    def get_document(self, doc_id: int) -> Dict:
        return self.repo.get_document(doc_id)

    def get_all_documents(self, include_inactive: bool = False) -> List[Dict]:
        return self.repo.get_all_documents(include_inactive)

    def update_document(self, doc_id: int, content=None, category=None, keywords=None, embedding=None) -> bool:
        return self.repo.update_document(doc_id, content, category, keywords, embedding)

    def delete_document(self, doc_id: int, soft_delete: bool = True) -> bool:
        return self.repo.delete_document(doc_id, soft_delete)

    def search_documents_by_category(self, category: str) -> List[Dict]:
        return self.repo.search_documents_by_category(category)

    def search_documents_by_keywords(self, keywords) -> List[Dict]:
        return self.repo.search_documents_by_keywords(keywords)

    def get_all_categories(self) -> List[str]:
        return self.repo.get_all_categories()

    def get_documents_count(self) -> int:
        return self.repo.get_documents_count()

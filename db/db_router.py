"""
数据库路由器 — 企业员工AI服务台

提供统一的知识库数据访问入口。
"""
from .base import SessionManager
from .repositories import KnowledgeRepository


class DatabaseRouter:
    """数据库路由器"""

    def __init__(self, db_path: str = 'sqlite:///data/ticket_dispatch.db'):
        self.session_manager = SessionManager(db_path)
        self.knowledge_repo = KnowledgeRepository(self.session_manager)

    @property
    def knowledge(self) -> KnowledgeRepository:
        return self.knowledge_repo

    def close(self):
        self.session_manager.close()


class KnowledgeDBRouter:
    """知识库数据库路由器（兼容性类）"""

    def __init__(self, db_type='local', **kwargs):
        self.db_router = DatabaseRouter(**kwargs)
        self.knowledge_repo = self.db_router.knowledge

    def add_document(self, content: str, category: str, keywords=None, embedding=None) -> int:
        return self.knowledge_repo.add_document(content, category, keywords, embedding)

    def get_document(self, doc_id: int):
        return self.knowledge_repo.get_document(doc_id)

    def get_all_documents(self, include_inactive: bool = False):
        return self.knowledge_repo.get_all_documents(include_inactive)

    def update_document(self, doc_id: int, content=None, category=None, keywords=None, embedding=None) -> bool:
        return self.knowledge_repo.update_document(doc_id, content, category, keywords, embedding)

    def delete_document(self, doc_id: int, soft_delete: bool = True) -> bool:
        return self.knowledge_repo.delete_document(doc_id, soft_delete)

    def search_documents_by_category(self, category: str):
        return self.knowledge_repo.search_documents_by_category(category)

    def search_documents_by_keywords(self, keywords):
        return self.knowledge_repo.search_documents_by_keywords(keywords)

    def get_all_categories(self):
        return self.knowledge_repo.get_all_categories()

    def get_documents_count(self) -> int:
        return self.knowledge_repo.get_documents_count()

# services/knowledge_service.py

import numpy as np
import faiss
from typing import List, Dict, Tuple, Optional
from db.db_router import DatabaseRouter
from .text_embedding import embed_input
import logging

logger = logging.getLogger(__name__)

class KnowledgeService:
    """知识库服务类 - 结合数据库存储和向量检索"""
    
    def __init__(self, db_path: str = 'sqlite:///data/ticket_dispatch.db'):
        # 使用统一的DatabaseRouter，符合架构设计
        self.db_router = DatabaseRouter(db_path)
        self.db = self.db_router.knowledge  # 通过router访问knowledge repository
        self.index = None
        self.document_ids = []  # 维护文档ID与索引位置的映射
        self.initialized = False
        self._dirty_count = 0   # 待重建计数（删除累积 > 10 时触发全量重建）
        
        # 默认知识库内容
        self.default_knowledge = [
            {
                "content": "IT服务台的工作时间为工作日周一至周五 9:00-18:00，紧急故障可通过P0工单通道7×24小时响应。",
                "category": "服务时间",
                "keywords": ["工作时间", "服务时间", "几点", "周末", "24小时", "紧急"]
            },
            {
                "content": "VPN连接失败排查步骤：1. 检查本机网络连通性(ping网关) 2. 确认VPN客户端版本≥v3.2.1 3. 验证AD账号密码正确性 4. 检查防火墙是否放行VPN端口(UDP 1194) 5. 如仍不通，提交P1工单处理。",
                "category": "网络故障",
                "keywords": ["VPN", "连接失败", "远程", "网络", "防火墙", "ping"]
            },
            {
                "content": "数据库连接超时常见原因：1. 网络策略变更导致端口不通(默认3306/5432) 2. 数据库最大连接数已满 3. 应用连接池配置不当 4. 数据库服务未启动。建议先检查连接字符串和网络连通性。",
                "category": "系统运维",
                "keywords": ["数据库", "超时", "MySQL", "PostgreSQL", "连接", "连接池"]
            },
            {
                "content": "IT服务台服务中心位于总部大楼3层305室，内线电话8888，外线010-8888XXXX。紧急故障可直接拨打值班手机139XXXXXXXX。",
                "category": "联系方式",
                "keywords": ["地址", "电话", "联系方式", "在哪", "服务台", "位置"]
            },
            {
                "content": "账号锁定/密码重置流程：1. AD账号连续5次输错自动锁定30分钟 2. 员工可通过OA自助解锁 3. 密码重置需提交工单，审批后15分钟内生效 4. 新员工账号创建需HR系统流程完成后自动开通。",
                "category": "账号管理",
                "keywords": ["密码", "账号", "锁定", "重置", "登录", "新员工", "OA"]
            },
            {
                "content": "企业WiFi网络配置：SSID名为Corp-Net（员工用）和Corp-Guest（访客用）。员工连接需使用AD账号密码认证，访客需接待人扫码授权。如遇WiFi频繁断连，请先忘记网络后重新连接。",
                "category": "网络配置",
                "keywords": ["WiFi", "无线", "网络", "上网", "SSID", "断连"]
            },
            {
                "content": "服务器重启标准流程：1. 确认重启窗口时间(非工作时间) 2. 通知受影响的业务团队 3. 备份关键配置和数据 4. 逐台重启(集群环境) 5. 重启后验证服务可达性和数据一致性 6. 如遇重启失败，升级为P1工单。",
                "category": "系统运维",
                "keywords": ["重启", "服务器", "维护", "停机", "备份", "窗口"]
            },
            {
                "content": "信息安全策略要点：所有系统必须使用强密码(12位+大小写+数字+符号)、敏感数据传输必须加密(HTTPS/TLS1.2+)、生产环境未经审批不得直接访问、离职员工账号需在最后工作日24小时内注销。",
                "category": "安全策略",
                "keywords": ["安全", "密码", "加密", "权限", "审计", "合规"]
            },
            {
                "content": "工单SLA政策：P0(紧急)响应15分钟/解决4小时，P1(高)响应1小时/解决8小时，P2(中)响应4小时/解决24小时，P3(低)响应8小时/解决48小时。超时未响应自动升级优先级并通知主管。",
                "category": "SLA政策",
                "keywords": ["SLA", "响应", "解决", "优先级", "超时", "升级", "P0", "P1"]
            },
            {
                "content": "知识库使用指南：IT运维知识库收录常见故障处理、系统使用指南、安全规范和操作手册。支持关键词搜索和分类浏览。如现有文档无法解决问题，可提交工单由专业工程师处理，解决后新方案将录入知识库。",
                "category": "使用指南",
                "keywords": ["知识库", "搜索", "文档", "帮助", "使用", "FAQ"]
            }
        ]

    async def initialize(self):
        """初始化知识库服务"""
        try:
            # 检查数据库中是否已有数据
            existing_docs = self.db.get_all_documents()
            
            if not existing_docs:
                logger.info("数据库为空，初始化默认知识库")
                await self._create_default_knowledge()
            else:
                logger.info(f"从数据库加载了 {len(existing_docs)} 条知识")
            
            # 构建向量索引
            await self._build_vector_index()
            self.initialized = True
            logger.info("知识库服务初始化完成")
            
        except Exception as e:
            logger.error(f"知识库服务初始化失败: {e}")
            raise

    async def _create_default_knowledge(self):
        """创建默认知识库"""
        for knowledge in self.default_knowledge:
            try:
                # 生成嵌入向量
                text_for_embedding = f"{knowledge['content']} {' '.join(knowledge['keywords'])}"
                embedding = embed_input(text_for_embedding)
                
                # 保存到数据库
                self.db.add_document(
                    content=knowledge['content'],
                    category=knowledge['category'],
                    keywords=knowledge['keywords'],
                    embedding=embedding
                )
                logger.debug(f"添加默认知识: {knowledge['content'][:50]}...")
                
            except Exception as e:
                logger.error(f"添加默认知识失败: {e}")

    async def _build_vector_index(self):
        """构建向量索引"""
        try:
            documents = self.db.get_all_documents()
            if not documents:
                logger.warning("没有文档可用于构建索引")
                return

            embeddings = []
            self.document_ids = []
            
            for doc in documents:
                if doc.get('embedding'):
                    embeddings.append(doc['embedding'])
                    self.document_ids.append(doc['id'])
                else:
                    # 如果没有嵌入向量，生成一个
                    logger.warning(f"文档 {doc['id']} 缺少嵌入向量，正在生成...")
                    text_for_embedding = f"{doc['content']} {' '.join(doc.get('keywords', []))}"
                    embedding = embed_input(text_for_embedding)
                    
                    # 更新数据库
                    self.db.update_document(doc['id'], embedding=embedding)
                    
                    embeddings.append(embedding)
                    self.document_ids.append(doc['id'])

            if embeddings:
                # 创建FAISS索引
                embeddings_array = np.array(embeddings).astype('float32')
                dimension = embeddings_array.shape[1]
                self.index = faiss.IndexFlatIP(dimension)  # 内积相似度
                self.index.add(embeddings_array)
                logger.info(f"构建向量索引完成，包含 {len(embeddings)} 个向量")
            else:
                logger.warning("没有有效的嵌入向量，无法构建索引")

        except Exception as e:
            logger.error(f"构建向量索引失败: {e}")
            raise

    async def search(self, query: str, top_k: int = 3, category: str = None) -> List[Dict]:
        """搜索相关文档"""
        if not self.initialized or self.index is None:
            logger.warning("知识库服务未初始化或索引不可用")
            return []

        try:
            # 生成查询的嵌入向量
            query_embedding = embed_input(query)
            query_array = np.array([query_embedding]).astype('float32')
            
            # 向量搜索
            scores, indices = self.index.search(query_array, min(top_k * 2, len(self.document_ids)))  # 多检索一些候选
            
            results = []
            for score, idx in zip(scores[0], indices[0]):
                if idx < len(self.document_ids):
                    doc_id = self.document_ids[idx]
                    doc = self.db.get_document(doc_id)
                    
                    if doc:
                        # 如果指定了分类过滤
                        if category and doc.get('category') != category:
                            continue
                            
                        doc['score'] = float(score)
                        doc['rank'] = len(results) + 1
                        results.append(doc)
                        
                        # 达到所需数量就停止
                        if len(results) >= top_k:
                            break
            
            return results
            
        except Exception as e:
            logger.error(f"搜索知识库失败: {e}")
            return []

    async def _add_to_index(self, doc_id: int, embedding: list):
        """增量添加单个向量到 FAISS 索引（不重建）"""
        if self.index is None:
            await self._build_vector_index()
            return

        emb_array = np.array([embedding]).astype('float32')
        self.index.add(emb_array)
        self.document_ids.append(doc_id)
        logger.debug(f"增量添加向量: doc_id={doc_id}, index_size={self.index.ntotal}")

    async def _rebuild_if_needed(self):
        """当删除累积超过阈值时触发全量重建"""
        self._dirty_count += 1
        if self._dirty_count > 10:
            logger.info(f"脏数据累积 {self._dirty_count} 条，触发全量索引重建")
            await self._build_vector_index()
            self._dirty_count = 0

    async def add_document(self, content: str, category: str, keywords: List[str] = None) -> bool:
        """添加新文档（增量更新索引）"""
        try:
            if keywords is None:
                keywords = []

            # 生成嵌入向量
            text_for_embedding = f"{content} {' '.join(keywords)}"
            embedding = embed_input(text_for_embedding)

            # 保存到数据库
            doc_id = self.db.add_document(content, category, keywords, embedding)

            # 增量添加到索引（不重建）
            await self._add_to_index(doc_id, embedding)

            logger.info(f"成功添加文档 {doc_id}: {content[:50]}...")
            return True

        except Exception as e:
            logger.error(f"添加文档失败: {e}")
            return False

    async def update_document(self, doc_id: int, content: str = None, category: str = None, keywords: List[str] = None) -> bool:
        """更新文档"""
        try:
            # 如果更新了内容或关键词，需要重新生成嵌入向量
            embedding = None
            if content is not None or keywords is not None:
                # 获取当前文档信息
                current_doc = self.db.get_document(doc_id)
                if not current_doc:
                    return False

                # 使用新值或保持原值
                final_content = content if content is not None else current_doc['content']
                final_keywords = keywords if keywords is not None else current_doc.get('keywords', [])

                # 生成新的嵌入向量
                text_for_embedding = f"{final_content} {' '.join(final_keywords)}"
                embedding = embed_input(text_for_embedding)

            # 更新数据库
            success = self.db.update_document(doc_id, content, category, keywords, embedding)

            if success and embedding is not None:
                # 内容变化 → 删除旧向量 + 增量添加新向量
                if doc_id in self.document_ids:
                    idx = self.document_ids.index(doc_id)
                    self.document_ids.pop(idx)
                    # FAISS 不支持直接删除，累积脏标记，定期重建
                    await self._rebuild_if_needed()
                await self._add_to_index(doc_id, embedding)

            return success

        except Exception as e:
            logger.error(f"更新文档失败: {e}")
            return False

    async def delete_document(self, doc_id: int, soft_delete: bool = True) -> bool:
        """删除文档（标记删除 + 延迟重建索引）"""
        try:
            success = self.db.delete_document(doc_id, soft_delete)

            if success:
                # 从 document_ids 中移除（FAISS 不支持直接删除向量）
                # 策略：标记为删除 + 累积到阈值时全量重建
                if doc_id in self.document_ids:
                    self.document_ids.remove(doc_id)
                await self._rebuild_if_needed()

            return success

        except Exception as e:
            logger.error(f"删除文档失败: {e}")
            return False

    def get_all_documents(self, include_inactive: bool = False) -> List[Dict]:
        """获取所有文档"""
        return self.db.get_all_documents(include_inactive)

    def get_document(self, doc_id: int) -> Dict:
        """获取指定文档"""
        return self.db.get_document(doc_id)

    def get_all_categories(self) -> List[str]:
        """获取所有分类"""
        return self.db.get_all_categories()

    def get_documents_count(self) -> int:
        """获取文档总数"""
        return self.db.get_documents_count()

    def search_by_category(self, category: str) -> List[Dict]:
        """按分类搜索文档"""
        return self.db.search_documents_by_category(category)

    def search_by_keywords(self, keywords: List[str]) -> List[Dict]:
        """按关键词搜索文档"""
        return self.db.search_documents_by_keywords(keywords)

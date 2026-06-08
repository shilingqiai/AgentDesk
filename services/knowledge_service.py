# services/knowledge_service.py
# v2 — IndexIDMap 支持真删除 + 反馈机制 + 扩展默认知识库

import numpy as np
import faiss
from typing import List, Dict, Tuple, Optional
from db.db_router import DatabaseRouter
from .text_embedding import embed_input
import logging

logger = logging.getLogger(__name__)


class KnowledgeService:
    """
    知识库服务类 — 向量检索 + 数据库存储 + 反馈

    v2 改进：
    - FAISS IndexIDMap：支持 remove_ids 真删除，不再依赖伪删除+脏重建
    - 反馈机制：记录用户 help/not-help 反馈
    - 扩展默认知识库：覆盖 IT、请假、报销、行政四大领域
    """

    def __init__(self, db_path: str = 'sqlite:///data/ticket_dispatch.db'):
        self.db_router = DatabaseRouter(db_path)
        self.db = self.db_router.knowledge
        self.index = None
        self._index_id_map: dict[int, int] = {}  # doc_id → FAISS 内部位置（兼容）
        self.initialized = False

        # 默认知识库内容 — 覆盖 IT、HR/请假、报销、行政
        self.default_knowledge = self._build_default_knowledge()

    @staticmethod
    def _build_default_knowledge() -> list[dict]:
        """构建扩展默认知识库 — 四大领域"""
        return [
            # ==================== IT服务 ====================
            {
                "content": "IT服务台的工作时间为工作日周一至周五 9:00-18:00，紧急故障可通过P0工单通道7×24小时响应。",
                "category": "IT服务",
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
            # ==================== 请假政策 ====================
            {
                "content": "年假政策：员工入职满1年后享有年假。工龄1-10年：5天/年，工龄10-20年：10天/年，工龄20年以上：15天/年。年假需提前至少3个工作日申请，由直属主管审批。年假可分次使用，最小单位为0.5天。年假可累计至下年度，但累计不超过应享天数的2倍。",
                "category": "请假政策",
                "keywords": ["年假", "带薪休假", "工龄", "审批", "累计", "申请天数"]
            },
            {
                "content": "病假流程：员工因病无法出勤，需当天上午10:00前通知直属主管并提交病假申请。病假3天以内无需医院证明，3天及以上需提交二级甲等以上医院开具的病假证明。病假期间薪资按国家及公司规定执行：病假工资=基本工资×80%。",
                "category": "请假政策",
                "keywords": ["病假", "医院证明", "生病", "医生", "病假工资", "病假条"]
            },
            {
                "content": "事假申请流程：事假需提前1个工作日申请，紧急事假可当天申请但需经理特批。事假天数从年假或调休中抵扣，不足部分按无薪事假处理。单次事假不超过3天，月累计不超过5天。事假期间不计薪。",
                "category": "请假政策",
                "keywords": ["事假", "无薪", "经理审批", "特批", "紧急请假"]
            },
            {
                "content": "婚假/产假/陪产假：婚假3天（晚婚15天），需提供结婚证复印件。产假98天（难产+15天），需提供医院证明。陪产假15天，需提供配偶生育证明。以上假期均为带薪假，需提前2周申请。",
                "category": "请假政策",
                "keywords": ["婚假", "产假", "陪产假", "结婚", "生育", "证明"]
            },
            # ==================== 报销政策 ====================
            {
                "content": "差旅报销标准：国内出差住宿费一线城市（北上广深）≤500元/晚，其他城市≤350元/晚。交通费高铁二等座/飞机经济舱实报实销。市内交通≤100元/天。餐饮补贴80元/天（无需发票）。出差需提前提交出差申请单，返回后5个工作日内提交报销。",
                "category": "报销政策",
                "keywords": ["差旅", "出差", "住宿", "交通费", "餐饮补贴", "报销标准"]
            },
            {
                "content": "报销流程：1. 登录OA系统→费用报销→新建报销单 2. 选择报销类型（差旅/办公/餐费/交通） 3. 填写报销金额和事由 4. 上传发票照片（金额≥2000元需上传原始发票）5. 提交→直属主管审批→财务审核 6. 审批通过后5个工作日内打款到工资卡。发票必须真实有效，电子发票和纸质发票具有同等效力。",
                "category": "报销政策",
                "keywords": ["报销流程", "OA系统", "发票", "审批", "打款", "电子发票"]
            },
            {
                "content": "办公用品采购与报销：单次采购金额≤500元可自行购买后报销，>500元需提前申请采购。报销时需上传购物小票或发票照片。IT类设备（键盘/鼠标/显示器等）需通过IT部门统一采购，不得自行购买报销。",
                "category": "报销政策",
                "keywords": ["办公用品", "采购", "小票", "设备", "自行购买", "IT设备"]
            },
            # ==================== 行政服务 ====================
            {
                "content": "会议室预定规则：通过飞书→日历→预定会议室，或OA→行政服务→会议室预定。小型会议室（4-6人）可随时预定，中大型会议室（8-20人）需提前1天预定。每周一上午为部门周例会预留时间，不可预定。会议室配备投影仪和白板，需技术支持请联系IT服务台。",
                "category": "行政服务",
                "keywords": ["会议室", "预定", "飞书", "投影仪", "白板", "技术支持"]
            },
            {
                "content": "快递寄送服务：公司提供顺丰月结账号用于公务快递。个人快递不得使用公司账号。寄件流程：填写快递单→交至前台→前台统一发出。紧急文件可使用同城闪送（需主管审批）。每日快递取件时间：上午10:00和下午16:00。",
                "category": "行政服务",
                "keywords": ["快递", "顺丰", "寄件", "月结", "前台", "闪送", "取件"]
            },
            {
                "content": "访客登记流程：访客需提前由接待员工在OA→行政服务→访客登记中提交访客信息（姓名/手机号/身份证号/到访时间/接待人）。审批通过后，访客将收到短信通知及电子访客码。访客持码在前台扫码通行。访客码当日有效，过期自动失效。",
                "category": "行政服务",
                "keywords": ["访客", "登记", "接待", "电子码", "前台", "通行", "预约"]
            },
            {
                "content": "资产领用与归还：公司资产（电脑/显示器/电话/打印机等）通过OA→行政服务→资产领用申请。新员工入职当天由IT部门统一发放标配设备。离职员工需在最后一个工作日归还所有公司资产，由行政部门确认签字后方可办理离职手续。",
                "category": "行政服务",
                "keywords": ["资产", "领用", "电脑", "设备", "离职", "归还", "新员工"]
            },
        ]

    async def initialize(self):
        """初始化知识库服务"""
        try:
            existing_docs = self.db.get_all_documents()

            if not existing_docs:
                logger.info("数据库为空，初始化默认知识库")
                await self._create_default_knowledge()
            else:
                logger.info(f"从数据库加载了 {len(existing_docs)} 条知识")

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
                text_for_embedding = f"{knowledge['content']} {' '.join(knowledge['keywords'])}"
                embedding = embed_input(text_for_embedding)

                self.db.add_document(
                    content=knowledge['content'],
                    category=knowledge['category'],
                    keywords=knowledge['keywords'],
                    embedding=embedding
                )
                logger.debug(f"添加默认知识: {knowledge['content'][:50]}...")

            except Exception as e:
                logger.error(f"添加默认知识失败: {e}")

    # ============================================================
    # FAISS 向量索引 — v2 IndexIDMap
    # ============================================================

    async def _build_vector_index(self):
        """
        构建向量索引 — 使用 IndexIDMap 支持真删除

        IndexIDMap 包装 IndexFlatIP，使 remove_ids() 可用。
        """
        try:
            documents = self.db.get_all_documents()
            if not documents:
                logger.warning("没有文档可用于构建索引")
                return

            embeddings = []
            doc_ids = []

            for doc in documents:
                if doc.get('embedding'):
                    embeddings.append(doc['embedding'])
                    doc_ids.append(doc['id'])
                else:
                    logger.warning(f"文档 {doc['id']} 缺少嵌入向量，正在生成...")
                    text_for_embedding = (
                        f"{doc['content']} {' '.join(doc.get('keywords', []))}"
                    )
                    embedding = embed_input(text_for_embedding)
                    self.db.update_document(doc['id'], embedding=embedding)
                    embeddings.append(embedding)
                    doc_ids.append(doc['id'])

            if embeddings:
                embeddings_array = np.array(embeddings).astype('float32')
                dimension = embeddings_array.shape[1]

                # v2: IndexIDMap 包装，支持 remove_ids
                base_index = faiss.IndexFlatIP(dimension)
                self.index = faiss.IndexIDMap(base_index)
                self.index.add_with_ids(
                    embeddings_array,
                    np.array(doc_ids, dtype=np.int64),
                )
                logger.info(
                    f"构建向量索引完成 (IndexIDMap)，包含 {len(embeddings)} 个向量，"
                    f"维度={dimension}"
                )
            else:
                logger.warning("没有有效的嵌入向量，无法构建索引")

        except Exception as e:
            logger.error(f"构建向量索引失败: {e}")
            raise

    async def _add_to_index(self, doc_id: int, embedding: list):
        """
        增量添加单个向量到 IndexIDMap

        IndexIDMap 支持 add_with_ids，每个向量绑定 doc_id 作为外部 ID。
        """
        if self.index is None:
            await self._build_vector_index()
            return

        emb_array = np.array([embedding]).astype('float32')
        id_array = np.array([doc_id], dtype=np.int64)
        self.index.add_with_ids(emb_array, id_array)
        logger.debug(f"增量添加向量: doc_id={doc_id}, index_size={self.index.ntotal}")

    async def _remove_from_index(self, doc_id: int):
        """从索引中删除指定 doc_id 的向量（IndexIDMap 真删除）"""
        if self.index is None:
            return

        try:
            id_array = np.array([doc_id], dtype=np.int64)
            n_before = self.index.ntotal
            self.index.remove_ids(id_array)
            n_removed = n_before - self.index.ntotal
            if n_removed > 0:
                logger.debug(f"从索引删除向量: doc_id={doc_id}, removed={n_removed}")
        except Exception as e:
            logger.warning(f"从索引删除向量失败 (doc_id={doc_id}): {e}")

    # ============================================================
    # 搜索
    # ============================================================

    async def search(self, query: str, top_k: int = 3, category: str = None) -> List[Dict]:
        """
        向量搜索相关文档

        Args:
            query: 查询文本
            top_k: 返回文档数
            category: 可选分类过滤

        Returns:
            相关文档列表（含 score / rank）
        """
        if not self.initialized or self.index is None:
            logger.warning("知识库服务未初始化或索引不可用")
            return []

        try:
            query_embedding = embed_input(query)
            query_array = np.array([query_embedding]).astype('float32')

            # 多检索一些候选
            k = min(top_k * 2, self.index.ntotal)
            if k == 0:
                return []

            scores, indices = self.index.search(query_array, k)

            # IndexIDMap 的 search 返回：indices 是外部 doc_id
            # scores 是内积相似度（IndexFlatIP 内积越高越相似）
            results = []
            for score, doc_id in zip(scores[0], indices[0]):
                doc_id = int(doc_id)
                if doc_id < 0:  # FAISS 无效索引标记
                    continue

                doc = self.db.get_document(doc_id)
                if not doc:
                    continue

                # 分类过滤
                if category and doc.get('category') != category:
                    continue

                doc['score'] = float(score)
                doc['rank'] = len(results) + 1
                results.append(doc)

                if len(results) >= top_k:
                    break

            return results

        except Exception as e:
            logger.error(f"搜索知识库失败: {e}")
            return []

    # ============================================================
    # CRUD
    # ============================================================

    async def add_document(self, content: str, category: str, keywords: List[str] = None) -> bool:
        """添加新文档（增量更新索引）"""
        try:
            if keywords is None:
                keywords = []

            text_for_embedding = f"{content} {' '.join(keywords)}"
            embedding = embed_input(text_for_embedding)

            doc_id = self.db.add_document(content, category, keywords, embedding)
            await self._add_to_index(doc_id, embedding)

            logger.info(f"成功添加文档 {doc_id}: {content[:50]}...")
            return True

        except Exception as e:
            logger.error(f"添加文档失败: {e}")
            return False

    async def update_document(
        self, doc_id: int, content: str = None, category: str = None,
        keywords: List[str] = None,
    ) -> bool:
        """更新文档 — 删除旧向量 + 添加新向量（真替换）"""
        try:
            embedding = None
            if content is not None or keywords is not None:
                current_doc = self.db.get_document(doc_id)
                if not current_doc:
                    return False

                final_content = content if content is not None else current_doc['content']
                final_keywords = keywords if keywords is not None else current_doc.get('keywords', [])

                text_for_embedding = f"{final_content} {' '.join(final_keywords)}"
                embedding = embed_input(text_for_embedding)

            success = self.db.update_document(doc_id, content, category, keywords, embedding)

            if success and embedding is not None:
                # 真替换：先删除旧向量，再添加新向量
                await self._remove_from_index(doc_id)
                await self._add_to_index(doc_id, embedding)

            return success

        except Exception as e:
            logger.error(f"更新文档失败: {e}")
            return False

    async def delete_document(self, doc_id: int, soft_delete: bool = True) -> bool:
        """
        删除文档 — IndexIDMap 支持真删除

        不再依赖脏数据累积+全量重建。
        """
        try:
            success = self.db.delete_document(doc_id, soft_delete)

            if success:
                await self._remove_from_index(doc_id)

            return success

        except Exception as e:
            logger.error(f"删除文档失败: {e}")
            return False

    # ============================================================
    # 查询
    # ============================================================

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

    # ============================================================
    # 用户反馈
    # ============================================================

    def record_feedback(
        self, doc_id: int, is_helpful: bool, user_id: str = "", comment: str = "",
    ) -> bool:
        """
        记录用户对文档的反馈（点赞/踩）

        Args:
            doc_id: 文档ID
            is_helpful: True=有帮助, False=无帮助
            user_id: 用户标识
            comment: 补充说明
        """
        try:
            from db.models import DocumentFeedback

            session_manager = self.db_router.session_manager
            with session_manager.session_scope() as session:
                feedback = DocumentFeedback(
                    doc_id=doc_id,
                    is_helpful=1 if is_helpful else 0,
                    user_id=user_id,
                    comment=comment,
                )
                session.add(feedback)

            logger.info(
                f"反馈已记录: doc={doc_id}, helpful={is_helpful}, user={user_id}"
            )
            return True

        except Exception as e:
            logger.error(f"记录反馈失败: {e}")
            return False

    def get_feedback_stats(self, doc_id: int) -> dict:
        """
        获取文档反馈统计

        Returns:
            {total, helpful_count, not_helpful_count, helpful_ratio}
        """
        try:
            from db.models import DocumentFeedback
            from sqlalchemy import func

            session_manager = self.db_router.session_manager
            with session_manager.session_scope() as session:
                total = session.query(DocumentFeedback).filter(
                    DocumentFeedback.doc_id == doc_id,
                ).count()

                helpful = session.query(DocumentFeedback).filter(
                    DocumentFeedback.doc_id == doc_id,
                    DocumentFeedback.is_helpful == 1,
                ).count()

                return {
                    "doc_id": doc_id,
                    "total": total,
                    "helpful_count": helpful,
                    "not_helpful_count": total - helpful,
                    "helpful_ratio": round(helpful / max(total, 1), 2),
                }

        except Exception as e:
            logger.error(f"获取反馈统计失败: {e}")
            return {"doc_id": doc_id, "total": 0, "helpful_count": 0,
                    "not_helpful_count": 0, "helpful_ratio": 0}

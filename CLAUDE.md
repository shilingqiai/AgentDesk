# CLAUDE.md — 企业员工AI服务台

基于 LangGraph + FAISS 的多 Agent 智能服务台。Hub & Spoke 架构，语义路由，RAG 知识自服务。

## 我的设定

1. **架构思维** — 不针对单一场景做死板的语义匹配修补；定位根因后从架构层面优化系统结构，而非堆砌 if-else
2. **不提交非 git 文件** — `eval/` 等标记为不提交 git 的目录/文件，不要擅自改成 git 跟踪
3. 更新文档时更新 architecture.md claude.md readme.md 新解决的问题添加到problem-solving

## 架构原则

- **Hub & Spoke** — Router 一次判定轨道，子 Agent 不直接回复用户
- **Single Response** — 编排器统一合成响应，`reply_to_user=False`
- **语义路由** — RouterDecision.confidence < 0.7 → clarification 反问
- **智能卡片优先** — 工单类操作返回确认卡片而非直接执行
- **RAG 优先** — IT 故障先搜知识库，解决不了再创建工单

### 审批流设计 (v11) — visibility ≠ actionability

审批流有 3 个独立维度，不可混淆：

| 维度 | 含义 | 谁 |
|------|------|-----|
| **visibility** (可见性) | 谁能看到工单 | 整条审批链上所有人 |
| **actionability** (可操作性) | 谁能点通过/驳回 | 仅 `approver_chain[current_step]` |
| **progress** (进度) | 当前在第几步 | `current_step` (0-indexed) |

**关键规则**：
- 工单创建时，链上**所有审批人**立即可见（包括后续步骤的人）
- 但只有当前步骤的审批人能操作，后续步骤显示灰色 disabled 卡片 + "等待 XX 审批"
- 审批链由 `ApprovalEngine.build_approval_chain()` 确定性生成，**不经过 LLM**
- API `/api/approvals/pending` 返回每个 item 的 `actionable: bool` 和 `current_approver: str`

### 数据写入架构 (v10)

- **单入口写入** — 所有工单创建必须经过 `services/ticket_service.py::TicketService.create_ticket()`，禁止多路径直接写 DB
- **单身份源** — 所有 Agent/Node/API 通过 `services/agent_context.py::AgentContext` 传递用户身份，禁止硬编码 fallback（如 `"web_user"` / `""` / `None`）
- **Card 不进 DB** — 确认卡片仅存在于 LangGraph State + 流式响应 `[CARD]` 标签，入库只存结构化字段
- **确定性审批链** — 审批角色链由 `TicketService.get_approver_chain()` 根据 ticket_type 确定性计算，不经过 LLM
- **Ticket Model** — 包含 `current_approver`（当前审批人）、`approver_chain`（审批链快照 JSON）、`history`（操作时间线 JSON）三列，由 TicketService 和 ApprovalEngine 共同维护

## 技术栈

- Python 3.12+，async/await，类型注解优先
- LangGraph 编排 + FAISS 向量检索
- SQLAlchemy ORM，session 通过 `db_router.get_session()` 获取
- DB auto-migration: `SessionManager._auto_migrate()` 在 `create_all()` 后检测缺失列并 ALTER TABLE ADD COLUMN
- 流式标签协议：`[THINKING]/[ROUTE]/[STREAM]/[CARD]/[DONE]`
- 前端 Alpine.js 3.13 CDN — SPA 架构：`index.html` (shell) + `app.js` (共享内核) + `ticket.js` / `approval.js` / `admin.js` / `chat.js` (模块) + `partials/_tab_*.html` (模板)
- 前端 4 固定身份：张三(employee) / 王经理(manager) / 李HR(hr) / Admin(admin)，`window.currentIdentity` 全局可用
- **身份切换行为** (v11)：清除 chat 缓存 + 重置 threadId + toast 提示 + 重启审批轮询 + 后台预加载审批计数
- **审批轮询** (v11)：manager/hr/admin 角色 15s 轮询 `/api/approvals/pending`，新增时 toast 通知；employee 不轮询
- **审批 Badge**：Tab 侧边栏红点，`pendingApprovalCount` 由轮询 + CustomEvent 双重同步
- **审批 UI** (v11)：分两区 — 可操作卡片（彩色按钮 + "你的环节"标签）+ 仅可见灰色卡片（disabled 按钮 + "等待 XX 审批"）
- 测试 pytest + pytest-asyncio，mock LLM 调用
- 时区 `Asia/Shanghai`，会议室 30min 粒度 8:00-20:00

### 延迟诊断 (v14)

- `_LATENCY_DEBUG = True` 开关覆盖 5 个文件，输出每个 LLM/工具/Embedding 调用耗时
- Embedding 模型已缓存为模块单例 (`text_embedding.py`)，避免重复创建 `OpenAIEmbeddings` 实例 (~2.3s/次)
- DynamicActionAgent 独立 tool_calls 并行执行 (`asyncio.gather`)，非串行
- Router 使用独立 `model_type="router"` 配置，不走主模型
- 独立诊断脚本: `python scripts/latency_test.py` (LLM + Embedding + 全流水线)
- 生产环境模型: `LLM_MODEL=qwen-plus` (非 max/preview 推理模型)

### 状态机与审批完成 (v14)

- **状态转换表** (`services/ticket_state.py`): `CREATED→PROCESSING`、`APPROVED→COMPLETED` 已加入 ALLOWED_TRANSITIONS
- **审批通过即完成**: 所有审批类型（请假/报销/采购）流程走完后自动 `APPROVED→COMPLETED`，Admin 不参与流程仅展示
- **IT 故障无审批**: 创建后自动 `CREATED→PROCESSING`，需 IT 修复后由 Admin 手动"标记已完成" → `COMPLETED`
- **批量审批**: `POST /api/approvals/batch-approve` + 前端全选/批量通过 UI
- **员工通知**: `/api/tickets/updates` 端点（since 增量轮询），15s 间隔，工单 Tab 红点提醒
- **前端状态标签**: `ticket.js` STATUS_LABELS 与后端 `TicketStatus` 枚举对齐（created/pending_approval/approved/rejected/processing/completed）
- **驳回**: `openRejectModal` / `closeRejectModal` 使用 CSS class `.modal-overlay.open` 控制显隐（非 inline style）

### 前端 API 约定

两个 API 族使用**不同的响应字段名**，不要混用：

| API 族 | 成功字段 | 数据字段 | 示例文件 |
|--------|---------|---------|---------|
| `/api/tickets/*` | `data.status === 'success'` | `data.data` | ticket.js, admin.js |
| `/api/approvals/*` | `data.success` | `data.items` / `data.workflow` | approval.js |

常见错误：`data.success` 用于 tickets API → 永远是 `undefined`，`if` 分支不进入，列表永远空。

## Git

本机需通过 VPN 代理访问 GitHub：`git -c http.proxy=127.0.0.1:12450 -c https.proxy=127.0.0.1:12450 <command>`

## 启动

```bash
python app.py --mode web --port 8001    # Web 模式
python app.py --mode all --port 8000    # Web + 飞书
pytest tests/ -v                        # 运行测试
python scripts/latency_test.py          # 延迟诊断
```

## 受保护

- `.env` / `config/` — 密钥与配置
- `data/` / `*.sqlite` / `*.faiss` / `*.pkl` / `logs/` — 运行时产物
- `agents/base_sub_agent.py` / `agents/a2a/` / `agents/orchestrator/control_layers.py` / `agents/orchestrator/governance.py` — 核心基础设施
- `services/agent_context.py` / `services/ticket_service.py` / `services/approval_engine.py` — 身份、写入入口、审批引擎，不得绕过
- `eval/` — 不提交 git，不要改成 git 跟踪

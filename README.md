# 企业员工AI服务台
<img width="1909" height="920" alt="屏幕截图 2026-06-08 165206" src="https://github.com/user-attachments/assets/0584602b-605c-4004-9612-650786b834c5" />

基于 LangGraph + FAISS 的多 Agent 智能服务台。Hub & Spoke 架构，语义路由，RAG 知识自服务，支持 Web 端流式对话、智能确认卡片、会议室管理、工单追踪与飞书机器人集成。

## 架构

```
用户输入 → Router (语义路由, qwen-mt-flash)
              │
    ┌─────────┼─────────┬──────────┐
    ▼         ▼         ▼          ▼
  fast     dynamic   complex    clarification
 (30%)    (60%)      (5%)       (不确定→反问)
    │         │         │
    ▼         ▼         ▼
EnterpriseRAG  DynamicAction  TaskPlanner
(FAISS+LLM)  (ReAct自由编排)  (多Agent委派)
```

- **fast** — 通用知识库问答（FAISS 向量检索 + LLM 合成），纯政策/流程查询
- **dynamic** — ReAct 循环自由编排，LLM 自主决定调用哪些工具、以什么顺序、根据中间结果条件判断。支持工单创建、库存查询、请假、报销、会议室预定等全部操作类场景
- **complex** — 复合指令，多 Agent 协作（已有被 dynamic 逐步替代的趋势）
- **clarification** — 语义路由不确定时主动反问，不猜测

## 核心设计

| 原则 | 说明 |
|------|------|
| Hub & Spoke | Router 一次判定轨道，消除线性流水线的冗余 LLM 调用 |
| Single Response | 子Agent 不直接回复用户，编排器统一合成 |
| 语义路由 | Pydantic 结构化输出，confidence < 0.7 触发反问 |
| ReAct 动态编排 | DynamicActionAgent 自主决定工具调用序列，零硬编码路径 |
| EnterpriseRAG | 统一知识库 Agent，FAISS 跨领域检索（IT+HR+行政） |
| 智能确认卡片 | 工单操作不直接执行，弹出预填卡片让用户确认 |
| 话题感知 | XML 标签隔离对话历史，自动检测话题切换，防上下文污染 |
| 三层控制 | AI / Hybrid / Deterministic，高风险操作人工审核 |
| A2A 协议 | Agent 间标准化通信，MessageBus 全链路追踪 |

## 功能亮点

### 🤖 智能确认卡片

告别"说句话就创建工单"的粗暴体验。系统返回结构化确认卡片，自动预填所有已知信息：

- **时间智能解析** — "明天早上" → 📅 2026-06-09 09:00-10:30，"周五下午3点" → 自动识别
- **用户偏好记忆** — 从历史预定中学习：常用会议室、偏好时长、常见主题
- **冲突检测** — 会议室被占用时自动切换到空闲房间，提示备选时段
- **IT RAG 优先** — 报修前先搜索知识库解决方案，解决了就不必创建工单
- **最小化操作** — 卡片预填所有可推断信息，用户只需点"确认"

### 🏢 会议室管理 (`/meeting-rooms`)

真实会议室预定系统，不再是文本工单：

- 5 间预设会议室（星空厅/银河厅/宇宙厅/创意坊/静思阁）
- 日视图时间轴（08:00-20:00，30 分钟粒度）
- 实时可用性查询，已占用时段灰色显示
- 点击空白时段弹出预定弹窗

### 📋 工单管理 (`/tickets`)

- 统计面板 + 类型/状态/优先级筛选
- 工单卡片展开查看详情（请假天数、报销金额、会议室时间等类型特有字段）
- 状态下变更（created → resolved → closed）
- 分页浏览

### 🎨 前端技术

- **Alpine.js 3.13** CDN 引入，零构建，保留 `python app.py` 单命令启动
- SSE 流式对话，实时打字效果
- 确认卡片在聊天气泡中渲染为交互式表单
- **ReAct 思维链面板** — LLM 每一步 Thought → Act → Observation 实时折叠展示

## 技术栈

- **编排**: LangGraph 1.0 (StateGraph + MemorySaver)
- **LLM**: 阿里云 DashScope (qwen-max / qwen-mt-flash / text-embedding-v4)
- **向量**: FAISS IndexFlatIP
- **Web**: FastAPI + Jinja2 + Alpine.js 3.13 + 原生 JS (ReadableStream)
- **IM**: 飞书 WebSocket Bot (lark-oapi)
- **DB**: SQLite + SQLAlchemy

## 快速开始

```bash
# 安装
pip install -r requirements.txt

# 配置
cp .env.example .env
# 编辑 .env → 填入 DASHSCOPE_API_KEY

# 启动 Web 模式
python app.py --mode web --port 8001
```

访问 http://127.0.0.1:8001

飞书 Bot 模式需额外配置 `FEISHU_APP_ID` / `FEISHU_APP_SECRET`。

## 项目结构

```
├── app.py                        # FastAPI 入口
├── agents/
│   ├── graph_workflow.py         # Hub & Spoke 编排工作流 (ReAct循环+话题感知)
│   ├── base_sub_agent.py         # Agent 抽象基类
│   ├── orchestrator/             # Router, Registry, Planner, Control
│   ├── sub_agents/               # EnterpriseRAG, TicketDispatch, DynamicAction
│   └── a2a/                      # Agent 间通信协议
├── services/                     # KnowledgeService, Embedding
├── db/                           # 数据模型 + Repository
├── api/                          # REST API
│   ├── meeting_rooms.py          # 会议室预定 API
│   ├── tickets.py                # 工单查询 API
│   └── knowledge.py              # 知识库 API
├── web/                          # 前端页面 + 路由
│   └── templates/
│       ├── index.html            # 主对话页面（含智能卡片渲染）
│       ├── meeting_rooms.html    # 会议室管理 SPA
│       └── tickets.html          # 工单管理 SPA
├── integrations/feishu/          # 飞书 Bot
├── tests/                        # 测试套件 (90 tests)
└── config/                       # 模型工厂 + 配置
```

## API

| 端点 | 说明 |
|------|------|
| `POST /chat/stream` | Web 流式聊天（SSE） |
| `POST /api/knowledge/search` | 向量搜索 |
| `GET /api/agents/list` | Agent 列表 |
| `POST /api/task/classify` | 语义路由分类 |
| `GET /api/tickets` | 工单列表 + 筛选 |
| `PATCH /api/tickets/{id}/status` | 更新工单状态 |
| `GET /api/meeting-rooms` | 会议室列表 |
| `GET /api/meeting-rooms/{id}/availability` | 会议室可用时段 |
| `POST /api/meeting-rooms/{id}/book` | 预定会议室 |
| `DELETE /api/meeting-rooms/bookings/{id}` | 取消预定 |
| `GET /docs` | Swagger 文档 |

## 流式令牌

```
[THINKING] 🔍 正在分析...     → 思考动画
[ROUTE] 🔍 极速通道          → 轨道判定
[REACT] {"event":"thought",...} → ReAct 思维链 (Thought/Act/Observation)
[FAST] 📚 企业知识库检索      → 轨道入口
[DYNAMIC] 🧠 ReAct 循环      → 动态编排入口
[STREAM] 回答文字片段         → 流式打字
[CARD] {"type":"confirm",...}  → 智能确认卡片
[INTERRUPT]                    → 等待用户确认卡片
[DONE]                         → 完成
```

## 测试

```bash
pytest tests/ -v
# 90+ passed
```

## 评估与监控

| 工具 | 说明 |
|------|------|
| `python scripts/router_eval.py` | 路由评估（80条标注用例，混淆矩阵 + F1） |
| `python scripts/router_eval.py --live` | 真实 LLM 评估（需 API Key） |

## 文档

- [架构文档](scripts/README.md)
- [路由评估报告](scripts/09-router-evaluation.md)
- [路由失败案例分析](scripts/09-failure-analysis.md)

# 企业员工AI服务台

基于 LangGraph + FAISS 的多 Agent 智能服务台。Hub & Spoke 架构，语义路由，RAG 知识自服务，支持 Web 端流式对话与飞书机器人集成。

## 架构

```
用户输入 → Router (语义路由, qwen-mt-flash)
              │
    ┌─────────┼─────────┐
    ▼         ▼         ▼
  fast     action    complex    clarification
 (80%)    (15%)      (5%)       (不确定→反问)
    │         │         │
    ▼         ▼         ▼
EnterpriseRAG  TicketDispatch  TaskPlanner
(FAISS+LLM)  (工单创建)     (多Agent委派)
```

- **fast** — 知识库问答（FAISS 向量检索 + LLM 合成），80% 请求仅需 2 次 LLM 调用
- **action** — 工单创建与派发
- **complex** — 复合指令，多 Agent 协作
- **clarification** — 语义路由不确定时主动反问，不猜测

## 核心设计

| 原则 | 说明 |
|------|------|
| Hub & Spoke | Router 一次判定轨道，消除线性流水线的冗余 LLM 调用 |
| Single Response | 子Agent 不直接回复用户，编排器统一合成 |
| 语义路由 | Pydantic 结构化输出，confidence < 0.7 触发反问 |
| EnterpriseRAG | 统一知识库 Agent，FAISS 跨领域检索（IT+HR+行政） |
| 三层控制 | AI / Hybrid / Deterministic，高风险操作人工审核 |
| A2A 协议 | Agent 间标准化通信，MessageBus 全链路追踪 |

## 技术栈

- **编排**: LangGraph 1.0 (StateGraph + MemorySaver)
- **LLM**: 阿里云 DashScope (qwen-max / qwen-mt-flash / text-embedding-v4)
- **向量**: FAISS IndexFlatIP
- **Web**: FastAPI + Jinja2 + 原生 JS (ReadableStream)
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
│   ├── graph_workflow.py         # Hub & Spoke 编排工作流
│   ├── base_sub_agent.py         # Agent 抽象基类
│   ├── orchestrator/             # Router, Registry, Planner, Control
│   ├── sub_agents/               # EnterpriseRAG, TicketDispatch
│   └── a2a/                      # Agent 间通信协议
├── services/                     # KnowledgeService, Embedding
├── db/                           # 数据模型 + Repository
├── api/                          # REST API
├── web/                          # 前端页面 + 路由
├── integrations/feishu/          # 飞书 Bot
└── config/                       # 模型工厂 + 配置
```

## API

| 端点 | 说明 |
|------|------|
| `POST /chat/stream` | Web 流式聊天 |
| `POST /api/knowledge/search` | 向量搜索 |
| `GET /api/agents/list` | Agent 列表 |
| `POST /api/task/classify` | 语义路由分类 |
| `GET /docs` | Swagger 文档 |

## 流式令牌

```
[THINKING] 🔍 正在分析...    → 思考动画
[ROUTE] 🔍 极速通道         → 轨道判定
[FAST] 📚 企业知识库检索     → 轨道入口
[STREAM]回答文字片段         → 流式打字
[DONE]                       → 完成
```

## 文档

- [架构文档](research/README.md)
- [评估与迭代记录](research/08-interview-evaluation.md)

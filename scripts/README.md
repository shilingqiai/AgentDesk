# 企业员工AI服务台 — 架构文档

> 基于 Microsoft Copilot Studio Hub & Spoke 模式的多 Agent 编排系统

## 架构总览

```
用户输入 → Router (qwen-mt-flash, ~300ms)
              │
    ┌─────────┼─────────┐
    ▼         ▼         ▼
  fast     action    complex     clarification
  (80%)    (15%)     (5%)        (不确定→反问)
    │         │         │
    ▼         ▼         ▼
EnterpriseRAG  TicketDispatch  TaskPlanner
(FAISS+LLM)  (工单创建)     (多Agent)
    │         │         │
    └─────────┴─────────┘
              │
              ▼
          用户响应
```

## 技术栈

| 层 | 技术 |
|----|------|
| 编排引擎 | LangGraph 1.0 (StateGraph + MemorySaver) |
| LLM | 阿里云 DashScope (qwen-max / qwen-mt-flash) |
| 向量检索 | FAISS IndexFlatIP + text-embedding-v4 |
| Web框架 | FastAPI + Jinja2 + 原生JS |
| 数据库 | SQLite + SQLAlchemy |
| 通信协议 | A2A (Agent-to-Agent) + WebSocket (飞书) |

## 目录结构

```
├── app.py                     # FastAPI 入口
├── agents/
│   ├── graph_workflow.py      # Hub & Spoke 编排工作流
│   ├── base_sub_agent.py      # Agent 抽象基类
│   ├── orchestrator/          # 编排器组件
│   │   ├── router.py          # 语义路由器 (RouterDecision)
│   │   ├── agent_registry.py  # Agent 注册中心
│   │   ├── agent_declaration.py
│   │   ├── task_planner.py    # 复合任务规划
│   │   ├── response_synthesizer.py
│   │   ├── control_layers.py  # AI/Hybrid/Deterministic 三层控制
│   │   ├── human_loop.py      # Human-in-the-Loop
│   │   ├── governance.py      # 审计追踪
│   │   └── telemetry.py       # 可观测性
│   ├── sub_agents/
│   │   ├── enterprise_rag.py  # 统一知识库RAG (FAISS跨领域)
│   │   └── ticket_dispatch.py # 工单派发
│   └── a2a/                   # Agent间通信协议
├── services/
│   ├── knowledge_service.py   # FAISS 知识库服务
│   └── text_embedding.py      # Embedding 工具
├── db/                        # 数据库层
├── api/                       # REST API
├── web/                       # 前端页面 + 路由
├── integrations/feishu/       # 飞书Bot集成
└── config/                    # 配置 + 模型工厂
```

## 核心设计原则

### 1. Hub & Spoke 三级路由
- **Router 一次判定**，80% 请求仅需 2 次 LLM 调用
- fast: 知识问答 → EnterpriseRAGAgent
- action: 工单操作 → TicketDispatchSubAgent
- complex: 复合指令 → TaskPlanner + 多Agent
- clarification: 不确定 → 反问用户

### 2. Single Response Principle
- 子Agent 不直接回复用户（reply_to_user=False）
- 编排器统一合成响应

### 3. 语义路由 + 反问机制
- Router 输出 Pydantic RouterDecision
- confidence < 0.7 → clarification_node 主动反问
- 废除关键词硬匹配

### 4. EnterpriseRAGAgent 统一知识库
- FAISS 向量检索跨所有领域（IT+HR+行政）
- 不按 agent_id 分发，由向量相似度自动匹配

## API 端点

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/` | 主页面 |
| GET | `/knowledge` | 知识库管理 |
| POST | `/chat/stream` | Web 流式聊天（核心） |
| POST | `/chat/reset` | 重置会话 |
| GET | `/api/agents/list` | 已注册 Agent |
| POST | `/api/task/classify` | 任务分类 |
| GET/POST/PUT/DELETE | `/api/knowledge/` | 知识库 CRUD |
| POST | `/api/knowledge/search` | 向量搜索 |
| POST | `/api/v3/chat` | SSE 流式编排 |

## 流式令牌格式

| 令牌 | 含义 |
|------|------|
| `[THINKING] <text>` | 思考状态更新 |
| `[ROUTE] <text>` | 路由轨道结果 |
| `[CLARIFY] <text>` | AI反问用户 |
| `[FAST]/[ACTION]/[COMPLEX]` | 轨道入口 |
| `[STREAM]<chunk>` | 回答文字片段 |
| `[DONE]` | 完成 |

## 启动

```bash
# Web 模式
python app.py --mode web --port 8001

# 完整模式 (Web + 飞书)
python app.py --mode all --port 8000
```

环境变量: `DASHSCOPE_API_KEY`（必填）、`FEISHU_APP_ID`/`FEISHU_APP_SECRET`（飞书可选）

## 性能

| 请求类型 | LLM调用 | 目标耗时 |
|----------|---------|----------|
| 知识查询 (80%) | 2次 (Router + RAG合成) | ~1.5s |
| 工单操作 (15%) | 2次 (Router + 参数提取) | ~2s |
| 复合指令 (5%) | 4次 (Router + Plan + Agents + Synthesize) | ~6s |

## 相关文档

- [面试官视角评估与迭代记录](./08-interview-evaluation.md)

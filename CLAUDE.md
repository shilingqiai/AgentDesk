# CLAUDE.md — 企业员工AI服务台

> Claude Code 项目上下文。记录架构约定、编码规范和受保护文件列表。

## 项目概述

基于 LangGraph + FAISS 的多 Agent 智能服务台。Hub & Spoke 架构，语义路由，RAG 知识自服务，支持 Web 端流式对话、智能确认卡片、会议室管理与飞书 Bot 集成。

## 关键约定

### 架构原则

1. **Hub & Spoke** — Router 一次判定轨道，子 Agent 不直接回复用户
2. **Single Response** — 编排器统一合成响应，`reply_to_user=False`
3. **语义路由** — RouterDecision.confidence < 0.7 → clarification 反问
4. **智能卡片优先** — 工单类操作返回确认卡片而非直接执行
5. **RAG 优先** — IT 故障先搜知识库，解决不了再创建工单

### 编码规范

- Python 3.12+，类型注解优先
- 所有异步 Agent 方法使用 `async/await`
- SQLAlchemy ORM，session 通过 `db_router.get_session()` 获取
- 流式输出通过 `[THINKING]/[ROUTE]/[STREAM]/[CARD]/[DONE]` 标签协议
- 前端 Alpine.js 3.13 CDN 引入，零构建，保留 Jinja2 模板
- 测试使用 pytest + pytest-asyncio，mock LLM 调用

### 时间处理

- 所有时间使用 `Asia/Shanghai` 时区
- 会议室时间粒度 30 分钟，8:00-20:00
- 日期格式 `YYYY-MM-DD`，时间格式 `HH:MM`

### 启动命令

```bash
python app.py --mode web --port 8001    # Web 模式
python app.py --mode all --port 8000    # Web + 飞书
pytest tests/ -v                        # 运行 66 个测试
```

## 受保护文件列表

以下文件/目录**不应修改**，除非用户明确要求：

### 配置与环境
- `.env` — 本地密钥，绝不提交
- `.env.example` — 模板文件，仅随 API 变更更新
- `config/` — 模型工厂与系统配置

### 数据与运行时
- `data/**` — 向量索引、数据库文件等运行时产物
- `*.sqlite`、`*.db`、`*.faiss`、`*.index`、`*.pkl` — 本地数据
- `logs/` — 日志输出

### 核心基础设施（修改需谨慎）
- `agents/base_sub_agent.py` — Agent 抽象基类，所有子 Agent 继承于此
- `agents/a2a/` — Agent 间通信协议（A2A），接口稳定
- `agents/orchestrator/control_layers.py` — 三层控制体系
- `agents/orchestrator/governance.py` — 审计追踪

### 稳定 API 层
- `api/__init__.py` — Router 注册入口
- `web/routes.py` — 页面路由

### 文档（代码变更时同步更新）
- `README.md` — 项目主文档
- `CLAUDE.md` — 本文件
- `scripts/README.md` — 架构文档
- `scripts/08-interview-evaluation.md` — 评估记录

### IDE / 工具
- `.gitignore` — 除非新增文件类型需要忽略
- `.idea/`、`.vscode/` — IDE 配置各人不同
- `.pytest_cache/`、`.mypy_cache/` — 工具缓存

## 项目目录结构

```
├── app.py                     # FastAPI 入口
├── agents/
│   ├── graph_workflow.py      # Hub & Spoke 编排工作流
│   ├── base_sub_agent.py      # Agent 抽象基类 ⚠️ 受保护
│   ├── orchestrator/          # 编排器组件
│   │   ├── router.py          # 语义路由器
│   │   ├── agent_registry.py  # Agent 注册中心
│   │   ├── task_planner.py    # 复合任务规划
│   │   ├── response_synthesizer.py
│   │   ├── control_layers.py  # 三层控制 ⚠️ 受保护
│   │   ├── human_loop.py      # Human-in-the-Loop
│   │   ├── governance.py      # 审计追踪 ⚠️ 受保护
│   │   └── telemetry.py       # 可观测性
│   ├── sub_agents/
│   │   ├── enterprise_rag.py  # 统一知识库 RAG
│   │   └── ticket_dispatch.py # 工单派发 + 智能卡片
│   └── a2a/                   # Agent 间通信协议 ⚠️ 受保护
├── services/
│   ├── knowledge_service.py   # FAISS 知识库服务
│   └── text_embedding.py      # Embedding 工具
├── db/                        # 数据模型 + Repository
├── api/                       # REST API
│   ├── __init__.py            # Router 注册 ⚠️ 受保护
│   ├── meeting_rooms.py       # 会议室 API
│   ├── tickets.py             # 工单 API
│   └── knowledge.py           # 知识库 API
├── web/
│   ├── routes.py              # 页面路由 ⚠️ 受保护
│   └── templates/
│       ├── index.html         # 主对话页面
│       ├── meeting_rooms.html # 会议室管理 SPA
│       └── tickets.html       # 工单管理 SPA
├── integrations/feishu/       # 飞书 Bot
├── tests/                     # 测试套件 (66 tests)
├── config/                    # 配置 ⚠️ 受保护
└── scripts/                      # 评估与架构文档
```

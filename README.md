# 企业员工AI服务台

基于 LangGraph + FAISS 的多 Agent 智能服务台。Hub & Spoke 架构，语义路由，RAG 知识自服务，支持 Web 流式对话、智能确认卡片、会议室管理与飞书 Bot 集成。

## 架构

```
用户输入 → Router (语义路由, qwen-plus)
              │
    ┌─────────┼─────────┬──────────┐
    ▼         ▼         ▼          ▼
  fast     complex   dynamic   clarification
 (80%)     (8%)     (10%)     (2%→反问)
    │         │         │
    ▼         ▼         ▼
EnterpriseRAG  固定DAG    DynamicAction
(FAISS+LLM)  (并行编排)  (ReAct自由编排)
```

- **fast** — 通用知识库问答（FAISS 向量检索 + LLM 流式合成）
- **complex** — 请假/报销固定 DAG（查政策 ∥ 查余额 → 合规检查 → 确认卡片），流程确定、延迟更低
- **dynamic** — ReAct 循环自由编排，覆盖设备领用、采购申请、IT 故障、会议室预定等需多工具组合的场景
- **clarification** — 语义路由不确定时主动反问，不猜测

## 核心设计

| 原则 | 说明 |
|------|------|
| Hub & Spoke | Router 一次判定轨道，消除线性流水线的冗余 LLM 调用 |
| Single Response | 子 Agent 不直接回复用户，编排器统一合成 |
| 语义路由 | Pydantic 结构化输出，confidence < 0.7 触发反问 |
| 固定 DAG | 请假/报销走 complex 并行 DAG（政策+余额→合规→卡片），确定、低延迟 |
| ReAct 动态编排 | 设备领用/采购/IT故障等开放场景，LLM 自主决定工具调用序列 |
| EnterpriseRAG | 统一知识库 Agent，FAISS 跨领域检索（IT+HR+行政） |
| 智能确认卡片 | 工单操作不直接执行，弹出预填卡片让用户确认 |
| 流式进度推送 | DAG 各阶段实时推送 [THINKING] 进度，前端不再无声等待 |
| 话题感知 | XML 标签隔离对话历史，自动检测话题切换 |
| A2A 协议 | Agent 间标准化通信，MessageBus 全链路追踪 |

## 快速开始

```bash
pip install -r requirements.txt
cp .env.example .env          # 编辑填入 DASHSCOPE_API_KEY
python app.py --mode web --port 8001
```

访问 http://127.0.0.1:8001

## 项目结构

```
├── app.py                     # FastAPI 入口
├── agents/
│   ├── graph_workflow.py      # 向后兼容 re-export 层
│   ├── graph/                 # 编排核心（拆分为子模块）
│   │   ├── state.py           #   TicketState + 辅助函数
│   │   ├── routing.py         #   条件边分发
│   │   ├── streaming.py       #   流式标签生成
│   │   ├── workflow.py        #   图构建 + 流式运行器
│   │   └── nodes/             #   7 个图节点
│   ├── base_sub_agent.py      # Agent 抽象基类
│   ├── orchestrator/          # Router, Registry, Planner, Control
│   ├── sub_agents/            # EnterpriseRAG, DynamicAction, TicketDispatch
│   └── a2a/                   # Agent 间通信协议
├── services/                  # KnowledgeService, Embedding
├── db/                        # 数据模型 + Repository
├── api/                       # REST API (会议室/工单/知识库)
├── web/                       # 前端页面 (Alpine.js + Jinja2)
├── integrations/feishu/       # 飞书 Bot
├── tests/                     # 119 tests (含 E2E State/SOP/DB 三维验证)
├── eval/                      # 路由评估 (80条标注, 混淆矩阵+F1)
└── config/                    # 模型工厂 + 配置 (AppSettings 统一管理)
```

## 流式标签协议

```
[THINKING] 🔍 正在分析...     → 思考动画
[ROUTE] 🔍 极速通道          → 轨道判定
[REACT] {"event":"thought",...} → ReAct 思维链
[STREAM] 回答文字片段         → 流式打字
[CARD] {"type":"confirm",...}  → 智能确认卡片
[DONE]                         → 完成
```

## 测试与评估

```bash
pytest tests/ -v                        # 119 tests (89 单元 + 30 E2E)
pytest tests/test_e2e_state.py -v       # E2E: State 过渡验证
pytest tests/test_e2e_tool_trace.py -v  # E2E: SOP 工具调用轨迹
pytest tests/test_e2e_db.py -v          # E2E: SQLite 数据库落地
python eval/router_eval.py              # 路由评估 (80条标注, 混淆矩阵+F1)
python eval/router_eval.py --live       # 真实 LLM 评估
```

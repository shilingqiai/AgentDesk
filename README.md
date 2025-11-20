# Enterprise Ticket Dispatch AI Agent

面向企业 IT 运维的智能工单调度系统。基于 FastAPI、LangChain、FAISS、SQLite 和多 Agent 协作架构，实现了工单智能分类、RAG 知识自服务、工程师技能匹配、工单调度引擎、SLA 追踪和效能分析等能力。

这个项目的核心目标不是只做一个普通的工单表单，而是尝试把 IT 运维团队日常需要处理的高频工作自动化：理解用户提交的是故障还是咨询，判断紧急程度，匹配最合适的工程师，检查当前负载，生成调度结果，并持续追踪 SLA 合规情况和工程师效能。

## 项目背景

在一个中型企业 IT 部门的工作中，我注意到运维团队每天需要处理大量重复性工作：工单分类分派、常见问题反复解答、工程师负载不透明、SLA 超时缺乏预警。随着业务增长和系统复杂度提升，人工分派模式容易出现响应延迟、技能错配、工作负载不均和知识沉淀不足的问题。

因此，本项目尝试用 AI Agent 的方式重构这一流程：让系统能够像一个智能调度中心一样主动理解工单需求，自动分类、知识自服务、技能匹配和负载均衡派单。它既适用于 IT 运维场景，也可以扩展到客服、行政、设施管理等需要工单调度和智能路由的企业服务场景。

## 核心能力

- **智能工单分类**：自动识别用户提交的是故障处理、需求变更、技术咨询还是例行维护，并判定优先级（P0-P3），将工单路由到对应 Agent。
- **多 Agent 协作**：通过任务分类 Agent、知识咨询 Agent、工单调度 Agent 和效能分析 Agent 分工处理复杂流程，减少单个模块的职责膨胀。
- **RAG 知识自服务**：使用 FAISS 向量索引检索知识库内容，结合大模型生成自然语言回答，支持流式输出。常见问题自动解答，减少人工工单量。
- **智能调度引擎**：根据工单需求、工程师技能标签、历史表现和当前负载进行匹配，辅助完成工单派发。
- **工程师效能分析**：记录工单处理行为，分析工程师效能和负载模式，为后续调度优化提供依据。
- **SLA 智能提醒**：工单创建后结合优先级和 SLA 策略生成时效提醒，超时自动升级。
- **Embedding 缓存优化**：通过数据库缓存和文件缓存减少重复向量计算，提高知识检索性能。
- **数据管理能力**：支持知识库、工程师信息和工单数据的增删改查，并在数据变化后自动维护索引。
- **日志与兜底机制**：保留关键处理过程日志，在信息不足或异常情况下提供更稳定的降级处理。

## 系统架构

项目采用严格的五层架构，核心原则是：**下层不能反向调用上层**。这样可以避免循环依赖，让业务逻辑、数据访问和接口编排保持清晰边界。

```text
Web & Application Layer
    ↓  app.py, web/：页面、路由入口、系统启动
API Layer
    ↓  api/：外部接口、请求编排、响应封装
Agents Layer
    ↓  agents/：AI Agent、任务路由、工单流程控制
Services Layer
    ↓  services/：业务逻辑、调度算法、向量处理
DB Layer
    ↓  db/：数据模型、数据库连接、Repository
```

### 允许的调用方向

- Web 层调用 API 层
- API 层调用 Agents 层或 Services 层
- Agents 层调用 Services 层
- Services 层调用 DB 层

### 禁止的调用方式

- 下层反向调用上层
- Web 层绕过 API 直接访问 Services 或 DB
- Agents 层绕过 Services 直接访问 DB
- Services 层调用 Agents、API 或 Web

## Agent 设计

### Task Classification Agent

任务分类 Agent 是系统的主调度器，负责分析用户输入、判断工单类型和优先级，并把请求分发给合适的专业 Agent。

```text
用户输入 → 意图分析与优先级判定 → Agent 路由 → 响应协调
```

主要职责：

- 判断工单类型（故障/需求/咨询/维护）
- 判定优先级（P0-P3）
- 维护工单处理状态
- 控制不同 Agent 之间的切换
- 处理无法分类或超出能力范围的问题

### Knowledge Consultation Agent

知识咨询 Agent 负责知识自服务场景，使用 RAG 流程从知识库中检索相关内容，再结合大模型生成回答。优先尝试自动解决常见问题，降低人工工单量。

```text
任务分类 → 知识检索 → FAISS 相似度搜索 → 流式回答
```

主要职责：

- 区分咨询问题类型
- 从知识库检索相关内容
- 构建提示词
- 生成自然语言回答
- 自服务解决率统计，未解决自动转工单

### Ticket Dispatch Agent

调度 Agent 负责工单派发流程，包括解析工单需求、匹配工程师、检查负载状态、生成派发确认消息等。

```text
任务分类 → 解析工单需求 → 工程师技能匹配 → 负载检查 → 派发确认
```

主要职责：

- 提取工单时间要求、问题类型、技能需求等信息
- 匹配合适工程师（基于技能标签 embedding 相似度）
- 检查工程师当前负载和可用性
- 处理信息缺失时的追问
- 生成派发结果和 SLA 提醒

### Engineer Analytics Agent

效能分析 Agent 更偏向后台智能分析，不完全依赖用户显式请求。它会根据工单处理记录、工程师表现和负载数据分析效能，为调度优化提供依据。

```text
工单记录 → 效能分析 → 负载统计 → 调度优化建议
```

主要职责：

- 记录工单处理行为
- 分析工程师效能指标
- 统计负载分布
- 生成调度优化建议
- 支持 SLA 合规追踪和主动预警

## 核心设计思想

### 1. 用任务分类降低系统复杂度

系统并不让一个 Agent 处理所有事情，而是先判断工单意图和优先级，再分发给对应模块。这样可以让知识自服务、工单调度、效能分析等逻辑保持独立，也更容易扩展新的 Agent。

### 2. 用 RAG 解决重复性咨询

IT 运维中大量问题是重复的（"怎么重置密码""VPN 连不上怎么办"）。RAG 能让回答基于可控知识来源，常见问题自动解答，只在知识库无法覆盖时才创建人工工单。

### 3. 用技能匹配提升调度质量

系统通过 embedding 向量相似度匹配工单需求与工程师技能标签，避免人工分派时的技能错配。后续调度时结合历史表现和当前负载，持续优化派单质量。

### 4. 用分层架构保证可维护性

Agent 负责智能流程，Service 负责业务逻辑，Repository 负责数据访问。每层只关心自己的职责，减少后期修改时的连锁影响。

### 5. 为真实业务场景预留扩展空间

项目目前以本地 SQLite 和单体服务为主，但架构上预留了模型提供商切换、MCP 外部服务接入、消息队列集成、缓存优化和云端部署的扩展方向。

## 技术栈

- **后端框架**：FastAPI、Uvicorn
- **AI 框架**：LangChain、LangGraph
- **大模型接入**：兼容 OpenAI 格式的模型提供商，例如 Qwen、DeepSeek、Zhipu、OpenAI、Azure OpenAI
- **向量检索**：FAISS
- **数据库**：SQLite、SQLAlchemy
- **RAG 能力**：Embedding、向量索引、知识库检索、提示词构建
- **流式响应**：Python AsyncGenerator
- **前端页面**：Jinja2 模板、静态 CSS
- **外部服务集成**：MCP 协议，用于通知推送等外部服务接入
- **配置管理**：python-dotenv
- **后台任务**：APScheduler

## 项目结构

```text
enterprise-ticket-dispatch/
├── agents/                              # 多 Agent 智能层
│   ├── task_classification_agent.py      # 任务分类与主路由
│   ├── consultant_agent.py               # RAG 知识咨询 Agent
│   ├── appointment_agent.py              # 智能工单调度 Agent
│   ├── user_behavior_agent.py            # 工程师效能分析 Agent
│   ├── task_classification/              # 意图识别、优先级判定、路由逻辑
│   ├── consultant/                       # 知识检索、提示词、回答生成
│   ├── appointment/                      # 工单解析、工程师匹配、消息构建
│   └── user_behavior/                    # 行为记录、效能管理、模式分析
├── api/                                  # API 编排层
│   ├── appointment.py                    # 工单调度接口
│   ├── consultation.py                   # 知识咨询接口
│   ├── task.py                           # 任务分类接口
│   ├── chat_handler.py                   # 流式聊天处理
│   ├── technician.py                     # 工程师管理接口
│   ├── knowledge.py                      # 知识库管理接口
│   └── user_behavior_analysis.py         # 效能分析接口
├── services/                             # 业务逻辑层
│   ├── appointment_service.py            # 工单调度业务逻辑
│   ├── knowledge_service.py              # 知识库管理
│   ├── recommendation_service.py         # 调度优化逻辑
│   ├── technician_service.py             # 工程师信息管理
│   ├── text_embedding.py                 # Embedding 与向量处理
│   └── user_behavior_service.py          # 效能分析服务
├── db/                                   # 数据持久化层
│   ├── models.py                         # SQLAlchemy 模型
│   ├── db_router.py                      # 数据库路由
│   ├── local_db.py                       # 本地数据库操作
│   ├── base/                             # 数据库基础接口
│   └── repositories/                     # Repository 数据访问封装
├── config/                               # 配置模块
│   ├── constants.py                      # 常量与枚举
│   ├── database.py                       # 数据库配置
│   ├── model_provider.py                 # 模型与 Embedding Provider 工厂
│   ├── settings.py                       # 应用配置
│   └── time_config.py                    # 时间与排班配置
├── web/                                  # Web 页面层
│   ├── routes.py                         # 页面路由
│   ├── templates/                        # HTML 模板
│   └── static/                           # 静态资源
├── tests/                                # 测试用例
├── app.py                                # 应用入口
├── requirements.txt                      # Python 依赖
├── Dockerfile                            # Docker 构建文件
├── docker-compose.yml                    # 容器编排配置
├── .env.example                          # 环境变量模板
└── README.md                             # 项目说明
```

## 快速开始

### 1. 创建虚拟环境

```bash
python -m venv .venv
```

Windows PowerShell：

```powershell
.\.venv\Scripts\Activate.ps1
```

Windows CMD：

```cmd
.venv\Scripts\activate.bat
```

macOS 或 Linux：

```bash
source .venv/bin/activate
```

### 2. 安装依赖

```bash
pip install -r requirements.txt
```

### 3. 配置环境变量

复制环境变量模板：

```bash
cp .env.example .env
```

Windows PowerShell 可以使用：

```powershell
Copy-Item .env.example .env
```

然后在 `.env` 中填写模型和数据库配置。项目支持 OpenAI 兼容格式的大模型与 Embedding 服务。

```env
MODEL_PROVIDER=qwen
LLM_API_KEY=your_llm_api_key_here
LLM_BASE_URL=your_openai_compatible_chat_base_url_here
LLM_MODEL=your_chat_model_name_here

EMBEDDING_PROVIDER=qwen
EMBEDDING_API_KEY=your_embedding_api_key_here
EMBEDDING_BASE_URL=your_openai_compatible_embedding_base_url_here
EMBEDDING_MODEL=your_embedding_model_name_here

DATABASE_URL=sqlite:///./data/ticket_dispatch.db

DEBUG=True
LOG_LEVEL=INFO
```

常见配置方向：

- Qwen：使用阿里云百炼或 DashScope 的模型、Base URL 和 API Key。
- DeepSeek：可用于聊天模型，Embedding 可搭配其他兼容服务。
- Zhipu：可配置智谱的聊天模型和向量模型。
- Azure OpenAI：将 `MODEL_PROVIDER` 设置为 `azure`，并补充对应的 Azure OpenAI 环境变量。

### 4. 启动服务

```bash
python -m uvicorn app:app --host 127.0.0.1 --port 8000 --reload
```

如果 8000 端口已被占用，可以换成 8001：

```bash
python -m uvicorn app:app --host 127.0.0.1 --port 8001 --reload
```

启动后可以访问：

- Web 页面：http://127.0.0.1:8000
- API 文档：http://127.0.0.1:8000/docs
- ReDoc 文档：http://127.0.0.1:8000/redoc

## 测试

运行全部测试：

```bash
pytest
```

运行单个测试文件：

```bash
pytest tests/test_task_classification_agent.py
```

## 主要页面

- 首页工单提交入口：`web/templates/index.html`
- 知识库管理：`web/templates/knowledge_management.html`
- 工程师管理：`web/templates/technician.html`
- 工程师排班：`web/templates/technician_schedule.html`
- 效能分析看板：`web/templates/user_behavior_analysis.html`

## Docker 部署

```bash
# 构建镜像
docker build -t ticket-dispatch .

# 使用 docker-compose 一键启动
docker-compose up -d
```

## 后续规划

### 更强的 Agent 自主能力

- 增加 Agent 自我反思机制，让系统能够评估工单分类准确率和调度质量。
- 引入 LangGraph 状态图管理工单完整生命周期。
- 根据工单处理反馈持续优化调度策略。

### 更完整的多 Agent 协作

- 增加 Agent-to-Agent 通信机制，减少所有任务都依赖主分类器转发的问题。
- 将效能分析 Agent 的后台分析能力做得更稳定，支持定时任务和主动预警。
- 把工单调度、知识自服务、效能分析之间的上下文记忆打通得更自然。

### 生产化能力

- 增加用户登录、权限控制和多租户数据隔离。
- 增加更完整的异常处理和边界场景覆盖。
- 优化向量检索性能、缓存策略和响应速度。
- 支持 PostgreSQL 数据库、Redis 缓存、消息队列和更标准的日志监控。

## 项目价值

这个项目把多 Agent、RAG、效能分析、工单调度和外部工具接入放在同一个企业级业务场景中验证。它既是一个 IT 运维智能调度原型，也可以作为学习 AI Agent 工程化、分层架构、RAG 系统和业务自动化的综合实践项目。

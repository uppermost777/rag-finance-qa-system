# 项目：基于 LangChain 与向量数据库的大学生理财知识问答系统

## 项目性质
本科毕业设计。面向大学生的理财知识问答 Web 系统，采用 RAG（检索增强生成）架构。
核心要求：**答案必须可溯源到知识库原文；系统只做知识科普，不构成投资建议。**

## 技术栈（硬约束，禁止替换）
- 语言/运行时：Python 3.11
- Web 框架：FastAPI + Uvicorn（全异步）
- 编排框架：LangChain（统一用 LCEL 表达式）+ LangGraph（多轮状态机）
- 参数校验：Pydantic v2 + pydantic-settings
- ORM：SQLAlchemy 2.0（async，驱动 aiomysql）
- 关系库：MySQL 8.0
- 向量库：Milvus 2.4（SDK：pymilvus）
- 缓存/消息：Redis 7（redis-py 的 asyncio 客户端）
- 异步任务：Celery 5.x
- 嵌入模型：BAAI/bge-m3，向量维度 1024（本地 sentence-transformers 加载）
- 重排模型：BAAI/bge-reranker-v2-m3（本地加载）
- 大模型：通过 OpenAI 兼容接口调用（DeepSeek / 通义千问均可），base_url 与 api_key 由环境变量注入
- 前端：Vue 3 + Vite + TypeScript + Element Plus + Pinia + Axios + ECharts
- 部署：Docker + docker-compose + Nginx

## 目录约定（严格遵守，不要另起结构）
```
finance-qa/
├─ AGENTS.md
├─ DESIGN.md                  # 系统设计文档
├─ README.md
├─ docker-compose.yml
├─ .env.example
├─ app/
│  ├─ main.py                 # FastAPI 入口
│  ├─ core/                   # config.py security.py logging.py exceptions.py deps.py
│  ├─ models/                 # SQLAlchemy ORM 模型
│  ├─ schemas/                # Pydantic 请求/响应模型
│  ├─ api/                    # 路由层 v1/ 下按模块拆分
│  ├─ services/               # 业务逻辑层
│  ├─ ingestion/              # loader.py splitter.py embedder.py pipeline.py
│  ├─ vectorstore/            # milvus_client.py
│  ├─ retrieval/              # hybrid.py reranker.py
│  ├─ chains/                 # rag_chain.py graph.py prompts.py
│  ├─ guard/                  # compliance.py
│  ├─ tasks/                  # celery_app.py document_tasks.py
│  └─ evaluation/             # ragas_eval.py datasets/
├─ tests/
├─ scripts/                   # 初始化脚本、建库脚本、实验脚本
└─ web/                       # Vue 3 前端
```

## 编码规范
- 所有函数带类型注解；对外接口必须有 docstring（中文）
- 全异步优先：IO 操作一律 `async def`，禁止在协程里调用同步阻塞库
- 配置统一从 `app/core/config.py` 读取（pydantic-settings），禁止散落 `os.getenv`
- 使用标准库 `logging`，禁止 `print`
- 统一异常处理：自定义业务异常 + FastAPI exception_handler
- 单元测试用 pytest + pytest-asyncio；外部依赖（Milvus/LLM）在测试中用 mock 或 fixture 替换
- 依赖写入 `requirements.txt` 并锁定版本

## 禁止事项（违反将被打回）
1. 禁止使用 LangChain 已废弃的 `RetrievalQA`、`load_qa_chain`、`VectorDBQA` 等旧式 Chain
2. 禁止新增技术栈之外的第三方依赖；若确有必要，先说明理由并等确认
3. 禁止硬编码任何密钥、token、数据库密码；一律走环境变量
4. 禁止删除或重命名任务未提及的已有文件
5. 生成内容中禁止出现"保证收益""稳赚不赔""推荐购买某只基金/股票""必涨"等表述
6. 禁止把长耗时操作（文档解析、向量化）放进同步请求链路

## 工作方式（重要）
- 动手前先读 `AGENTS.md` 与相关已有代码，**先输出「改动计划」**：要新建/修改的文件清单 + 关键类与函数签名，等我确认或直接继续
- 编码完成后必须运行对应的测试或构建命令，并把真实输出贴出来
- 只做任务范围内的改动，不要顺手重构无关代码、不要改动公共接口
- 如遇需求模糊，选择最保守的实现并在回复末尾列出你的假设

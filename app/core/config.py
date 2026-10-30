"""全局配置：基于 pydantic-settings 从环境变量 / .env 读取，禁止在业务代码中散落 os.getenv。"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import List, Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# 项目根目录：<root>/app/core/config.py -> parents[2] == <root>
BASE_DIR: Path = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    """应用配置对象（单例，通过 get_settings() 获取）。"""

    model_config = SettingsConfigDict(
        env_file=str(BASE_DIR / ".env"),
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ---------- 应用 ----------
    app_name: str = "finance-qa"
    app_version: str = "1.0.0"
    debug: bool = False
    api_v1_prefix: str = "/api/v1"
    log_level: str = "INFO"
    log_dir: str = "logs"
    cors_origins: List[str] = Field(default_factory=lambda: ["http://localhost:5173"])

    # ---------- MySQL ----------
    database_url: str = (
        "mysql+aiomysql://finance:finance_pwd@127.0.0.1:3306/finance_qa?charset=utf8mb4"
    )
    sync_database_url: str = (
        "mysql+pymysql://finance:finance_pwd@127.0.0.1:3306/finance_qa?charset=utf8mb4"
    )
    db_echo: bool = False
    db_pool_size: int = 10
    db_max_overflow: int = 20

    # ---------- Milvus ----------
    milvus_host: str = "127.0.0.1"
    milvus_port: int = 19530
    # 本地嵌入式模式（Milvus Lite）：填入文件路径即可免服务端运行，留空则连接 host:port
    # 注意：不能命名为 MILVUS_URI —— pymilvus 自身会读取同名环境变量并要求 http(s):// 形式
    milvus_lite_path: str = ""
    milvus_collection: str = "finance_chunks"
    milvus_user: str = ""
    milvus_password: str = ""
    milvus_timeout: int = 10

    # ---------- Redis ----------
    redis_url: str = "redis://127.0.0.1:6379/0"
    redis_max_connections: int = 50

    # ---------- LLM ----------
    llm_base_url: str = "https://api.deepseek.com/v1"
    llm_api_key: str = "sk-please-replace-me"
    llm_model: str = "deepseek-chat"
    llm_temperature: float = 0.2
    llm_timeout: int = 60
    llm_max_retries: int = 2
    # 单次生成的最大 token 数（控制延迟与成本；RAG 回答要求 ≤400 字）
    llm_max_tokens: int = 600

    # ---------- 本地模型 ----------
    embedding_model_path: str = "/models/bge-m3"
    embedding_dim: int = 1024
    embedding_batch_size: int = 16
    reranker_model_path: str = "/models/bge-reranker-v2-m3"
    reranker_batch_size: int = 8

    # ---------- 检索参数 ----------
    recall_top_k: int = 10
    rerank_top_n: int = 5
    rrf_k: int = 60
    bm25_top_k: int = 10
    chunk_size: int = 512
    chunk_overlap: int = 64
    min_rerank_score: float = 0.05
    default_strategy: Literal["vector", "bm25", "hybrid", "hybrid_rerank"] = "hybrid_rerank"

    # ---------- 鉴权 ----------
    jwt_secret: str = "please-change-this-to-a-random-64-chars-secret"
    jwt_algorithm: str = "HS256"
    jwt_expire_minutes: int = 1440

    # ---------- 上传 ----------
    upload_dir: str = "uploads"
    max_upload_size_mb: int = 20
    allowed_extensions: List[str] = Field(
        default_factory=lambda: ["pdf", "docx", "txt", "md", "html"]
    )

    # ---------- 开关 ----------
    enable_answer_cache: bool = False
    rate_limit_per_minute: int = 20
    # 开发/演示开关：无 Celery/Redis 时在 API 进程内异步执行入库（生产保持 False）
    inline_ingest: bool = False

    # ---------- Celery ----------
    celery_broker_url: str = "redis://127.0.0.1:6379/1"
    celery_result_backend: str = "redis://127.0.0.1:6379/2"
    celery_queue: str = "finance_qa"

    @field_validator("cors_origins", "allowed_extensions", mode="before")
    @classmethod
    def _split_csv(cls, value: object) -> object:
        """兼容 ``a,b`` 与 ``["a","b"]`` 两种写法。"""
        if isinstance(value, str) and not value.strip().startswith("["):
            return [item.strip() for item in value.split(",") if item.strip()]
        return value

    @property
    def upload_path(self) -> Path:
        """上传根目录（相对路径基于项目根目录解析）。"""
        path = Path(self.upload_dir)
        return path if path.is_absolute() else BASE_DIR / path

    @property
    def log_path(self) -> Path:
        """日志目录。"""
        path = Path(self.log_dir)
        return path if path.is_absolute() else BASE_DIR / path

    @property
    def max_upload_bytes(self) -> int:
        """单文件大小上限（字节）。"""
        return self.max_upload_size_mb * 1024 * 1024


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """获取全局配置单例。"""
    return Settings()


settings: Settings = get_settings()


"""Redis 异步客户端封装：全局单例 + 常用键名工具方法。"""

from __future__ import annotations

from typing import Optional

from redis.asyncio import Redis, from_url

from app.core.config import get_settings
from app.core.exceptions import ExternalServiceError
from app.core.logging import get_logger

logger = get_logger(__name__)

_client: Optional[Redis] = None

# ---------- Key 命名规范（详见 DESIGN.md 第 6 章） ----------
KEY_JWT_BLACKLIST = "finance:jwt:blacklist:{jti}"
KEY_RISK_RULE_CACHE = "finance:riskrule:cache"
KEY_DOC_PROGRESS = "finance:doc:progress:{doc_id}"
KEY_BM25_VERSION = "finance:kb:bm25:version:{kb_id}"
KEY_ANSWER_CACHE = "finance:cache:answer:{fingerprint}"
KEY_RATE_LIMIT = "finance:ratelimit:qa:{user_id}:{bucket}"
KEY_TASK = "finance:task:{task_id}"
KEY_QA_STATS = "finance:stats:qa:{date}"
KEY_EVAL_PROGRESS = "finance:eval:progress:{run_id}"


def get_redis() -> Redis:
    """获取全局 Redis 异步客户端（单例，连接池复用）。"""
    global _client
    if _client is None:
        settings = get_settings()
        _client = from_url(
            settings.redis_url,
            encoding="utf-8",
            decode_responses=True,
            max_connections=settings.redis_max_connections,
        )
        logger.info("已创建 Redis 客户端：%s", settings.redis_url)
    return _client


async def close_redis() -> None:
    """关闭客户端（应用退出时调用）。"""
    global _client
    if _client is not None:
        await _client.aclose()
        _client = None
        logger.info("Redis 客户端已关闭")


async def ping_redis() -> bool:
    """探测 Redis 连通性。"""
    try:
        return bool(await get_redis().ping())
    except Exception as exc:  # pragma: no cover - 依赖真实环境
        raise ExternalServiceError(f"Redis 不可用：{exc}") from exc

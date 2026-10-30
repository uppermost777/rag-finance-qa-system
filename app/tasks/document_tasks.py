"""文档入库与重建索引任务。

设计要点：
- 任务幂等：``ingest_document`` 内部会先清理旧切片与旧向量，重复触发不会产生重复向量；
- 阶段进度实时写 Redis（``finance:doc:progress:{doc_id}``），前端轮询即可看到进度条；
- 失败自动重试（最多 3 次，指数退避），最终仍失败则文档状态为 ``failed`` 并记录原因；
- Celery worker 内通过 ``asyncio.run`` 复用与 Web 端完全相同的异步流水线实现，
  避免维护两套解析/向量化代码。
"""

from __future__ import annotations

import asyncio
import time
from typing import Any, Optional

from celery import Task

from app.core.logging import get_logger, set_trace_id
from app.tasks.celery_app import celery_app

logger = get_logger(__name__)

MAX_RETRIES = 3
RETRY_BACKOFF = 10  # 秒，指数退避基数


def _write_progress(doc_id: int, status: str, progress: int, message: str = "") -> None:
    """把进度写入 Redis（同步客户端，供 worker 内使用；失败不影响任务）。"""
    try:
        import redis

        from app.core.config import get_settings

        client = redis.Redis.from_url(get_settings().redis_url, decode_responses=True)
        key = f"finance:doc:progress:{int(doc_id)}"
        client.hset(
            key,
            mapping={
                "status": status,
                "progress": int(progress),
                "msg": message,
                "updated_at": str(int(time.time())),
            },
        )
        client.expire(key, 86400)
    except Exception as exc:  # pragma: no cover - 依赖 Redis
        logger.debug("写入进度失败（忽略）：%s", exc)


async def _run_ingest(doc_id: int) -> dict[str, Any]:
    """异步执行入库流水线，并把进度同步到 Redis。"""
    from app.ingestion.pipeline import ingest_document

    async def _progress(status: str, progress: int, message: str) -> None:
        """进度回调：写 Redis + 打日志。"""
        _write_progress(doc_id, status, progress, message)

    result = await ingest_document(doc_id, progress=_progress)
    return {
        "doc_id": result.doc_id,
        "status": result.status,
        "page_count": result.page_count,
        "chunk_count": result.chunk_count,
        "error_msg": result.error_msg,
        "elapsed_ms": result.elapsed_ms,
    }


@celery_app.task(bind=True, name="tasks.ingest_document", max_retries=MAX_RETRIES)
def ingest_task(self: Task, doc_id: int) -> dict[str, Any]:
    """文档入库任务：解析 → 清洗 → 切片 → 向量化 → 写 Milvus → 回写 MySQL。"""
    set_trace_id(f"ingest-{doc_id}")
    logger.info("开始入库任务 doc_id=%s task_id=%s", doc_id, self.request.id)
    _write_progress(doc_id, "parsing", 5, "任务已启动")
    try:
        payload = asyncio.run(_run_ingest(doc_id))
    except Exception as exc:
        logger.exception("入库任务异常 doc_id=%s：%s", doc_id, exc)
        _write_progress(doc_id, "failed", 0, str(exc)[:200])
        raise self.retry(exc=exc, countdown=RETRY_BACKOFF * (2**self.request.retries))

    if payload["status"] == "failed" and self.request.retries < MAX_RETRIES:
        _write_progress(doc_id, "retrying", 0, "重试中")
        raise self.retry(countdown=RETRY_BACKOFF * (2**self.request.retries))

    _write_progress(
        doc_id, payload["status"], 100 if payload["status"] == "done" else 0, "入库结束"
    )
    logger.info(
        "入库任务结束 doc_id=%s status=%s chunks=%s",
        doc_id,
        payload["status"],
        payload["chunk_count"],
    )
    return payload


@celery_app.task(bind=True, name="tasks.reindex_document", max_retries=MAX_RETRIES)
def reindex_task(self: Task, doc_id: int) -> dict[str, Any]:
    """单个文档重建索引（幂等，等价于重新入库）。"""
    set_trace_id(f"reindex-{doc_id}")
    logger.info("开始重建索引 doc_id=%s", doc_id)
    return ingest_task.run(doc_id)


@celery_app.task(bind=True, name="tasks.reindex_kb", max_retries=MAX_RETRIES)
def reindex_kb_task(self: Task, kb_id: int) -> dict[str, Any]:
    """批量重建某知识库下所有文档的索引。"""
    set_trace_id(f"reindex-kb-{kb_id}")

    async def _collect() -> list[int]:
        from sqlalchemy import select

        from app.core.database import get_session_factory
        from app.models import Document

        factory = get_session_factory()
        async with factory() as session:
            rows = await session.scalars(
                select(Document.id).where(Document.kb_id == int(kb_id)).order_by(Document.id)
            )
            return [int(item) for item in rows.all()]

    doc_ids = asyncio.run(_collect())
    logger.info("知识库 %s 共 %d 个文档待重建", kb_id, len(doc_ids))
    results: list[dict[str, Any]] = []
    for doc_id in doc_ids:
        try:
            results.append(asyncio.run(_run_ingest(doc_id)))
        except Exception as exc:  # 单个文档失败不影响整批
            logger.exception("重建失败 doc_id=%s：%s", doc_id, exc)
            results.append({"doc_id": doc_id, "status": "failed", "error_msg": str(exc)})
    done = sum(1 for item in results if item.get("status") == "done")
    logger.info("知识库 %s 重建完成：成功 %d / 共 %d", kb_id, done, len(doc_ids))
    return {"kb_id": kb_id, "total": len(doc_ids), "success": done, "results": results}


def run_ingest_sync(doc_id: int) -> dict[str, Any]:
    """同步触发一次入库（供 scripts/trigger_ingest.py 在无 worker 时使用）。"""
    return asyncio.run(_run_ingest(doc_id))


def progress_snapshot(doc_id: int) -> Optional[dict[str, str]]:
    """读取进度快照（同步，脚本用）。"""
    try:
        import redis

        from app.core.config import get_settings

        client = redis.Redis.from_url(get_settings().redis_url, decode_responses=True)
        data = client.hgetall(f"finance:doc:progress:{int(doc_id)}")
        return data or None
    except Exception:  # pragma: no cover - Redis 不可用
        return None

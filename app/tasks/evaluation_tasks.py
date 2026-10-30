"""评估任务：跑 A/B/C/D 四组检索配置的对比实验，进度写 Redis。"""

from __future__ import annotations

import asyncio
import time
from typing import Any, Optional

from celery import Task

from app.core.logging import get_logger, set_trace_id
from app.tasks.celery_app import celery_app

logger = get_logger(__name__)


def _write_eval_progress(
    run_id: str, status: str, progress: int, report_path: Optional[str] = None, message: str = ""
) -> None:
    """写入评估进度到 Redis（失败不影响任务）。"""
    try:
        import redis

        from app.core.config import get_settings

        client = redis.Redis.from_url(get_settings().redis_url, decode_responses=True)
        key = f"finance:eval:progress:{run_id}"
        mapping = {"status": status, "progress": int(progress), "message": message}
        if report_path:
            mapping["report_path"] = report_path
        client.hset(key, mapping=mapping)
        client.expire(key, 604800)
    except Exception as exc:  # pragma: no cover - 依赖 Redis
        logger.debug("写入评估进度失败（忽略）：%s", exc)


async def _run_eval(
    run_id: str, dataset: str, limit: Optional[int], strategies: list[str]
) -> dict[str, Any]:
    """异步执行对比实验。"""
    from app.evaluation.runner import run_comparison

    def _progress(done: int, total: int, message: str) -> None:
        percent = int(done / max(total, 1) * 100)
        _write_eval_progress(run_id, "RUNNING", percent, message=message)

    return await run_comparison(
        run_id=run_id,
        dataset_name=dataset,
        limit=limit,
        strategies=strategies,
        progress=_progress,
    )


@celery_app.task(bind=True, name="tasks.run_evaluation")
def run_eval_task(
    self: Task,
    run_id: str,
    dataset: str = "finance_qa_eval.jsonl",
    limit: Optional[int] = None,
    strategies: Optional[list[str]] = None,
) -> dict[str, Any]:
    """执行评估任务并落盘 CSV / 图表。"""
    set_trace_id(f"eval-{run_id}")
    chosen = strategies or ["vector", "bm25", "hybrid", "hybrid_rerank"]
    _write_eval_progress(run_id, "STARTED", 0, message="任务已启动")
    started = time.perf_counter()
    try:
        payload = asyncio.run(_run_eval(run_id, dataset, limit, chosen))
    except Exception as exc:
        logger.exception("评估任务失败 run_id=%s：%s", run_id, exc)
        _write_eval_progress(run_id, "FAILED", 0, message=str(exc)[:200])
        raise
    payload["elapsed_ms"] = int((time.perf_counter() - started) * 1000)
    _write_eval_progress(
        run_id, "SUCCESS", 100, report_path=payload.get("csv_path"), message="评估完成"
    )
    logger.info("评估任务完成 run_id=%s -> %s", run_id, payload.get("csv_path"))
    return payload

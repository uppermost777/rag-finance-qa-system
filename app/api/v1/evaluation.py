"""评估接口：触发对比实验任务、查询进度与结果（管理员）。"""

from __future__ import annotations

import csv
import uuid
from pathlib import Path
from typing import Any, Optional

from fastapi import APIRouter, Query

from app.core.config import BASE_DIR
from app.core.deps import AdminUser
from app.core.exceptions import ExternalServiceError, NotFoundError
from app.core.logging import get_logger
from app.schemas.common import ok
from app.schemas.evaluation import EvalReportResponse, EvalRunRequest, EvalRunResponse, EvalStatusResponse

logger = get_logger(__name__)

router = APIRouter(prefix="/evaluation", tags=["评估"])

RESULTS_DIR = BASE_DIR / "app" / "evaluation" / "results"


def dispatch_eval(run_id: str, dataset: str, limit: Optional[int], strategies: list[str]) -> Optional[str]:
    """投递评估任务（Celery）；队列不可用时返回 None。"""
    try:
        from app.tasks.evaluation_tasks import run_eval_task

        async_result = run_eval_task.delay(run_id, dataset, limit, strategies)
        return str(getattr(async_result, "id", "") or "")
    except Exception as exc:  # pragma: no cover - 依赖 Celery/Redis
        logger.warning("投递评估任务失败：%s", exc)
        return None


@router.post("/run", summary="触发评估任务")
async def run_evaluation(payload: EvalRunRequest, admin: AdminUser) -> dict:
    """提交对比实验任务（A/B/C/D 四组配置）。"""
    run_id = uuid.uuid4().hex[:12]
    task_id = dispatch_eval(run_id, payload.dataset, payload.limit, payload.strategies)
    if not task_id:
        raise ExternalServiceError("任务队列不可用（请检查 Redis 与 Celery worker）")
    logger.info("评估任务已提交：run_id=%s task_id=%s", run_id, task_id)
    return ok(EvalRunResponse(run_id=run_id, message=f"评估任务已提交（task_id={task_id}）").model_dump())


@router.get("/report", summary="读取评估报告")
async def report(admin: AdminUser, run_id: Optional[str] = Query(default=None)) -> dict:
    """读取评估结果 CSV（默认最新一份），供前端画柱状图。"""
    path = _resolve_report(run_id)
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        columns = list(reader.fieldnames or [])
        rows: list[dict[str, Any]] = []
        for row in reader:
            rows.append({key: _coerce(value) for key, value in row.items() if key})
    return ok(
        EvalReportResponse(run_id=run_id, columns=columns, rows=rows).model_dump(mode="json")
    )


@router.get("/{run_id}", summary="查询评估进度")
async def status(run_id: str, admin: AdminUser) -> dict:
    """从 Redis 读取评估任务进度。"""
    payload: dict[str, Any] = {"run_id": run_id, "status": "UNKNOWN", "progress": 0}
    try:
        from app.core.redis_client import KEY_EVAL_PROGRESS, get_redis

        cached = await get_redis().hgetall(KEY_EVAL_PROGRESS.format(run_id=run_id))
        if cached:
            payload.update(
                {
                    "status": cached.get("status", "RUNNING"),
                    "progress": int(cached.get("progress", 0) or 0),
                    "report_path": cached.get("report_path"),
                    "message": cached.get("message"),
                }
            )
            return ok(EvalStatusResponse(**payload).model_dump())
    except Exception as exc:  # pragma: no cover - Redis 不可用
        logger.warning("读取评估进度失败：%s", exc)
    raise NotFoundError("未找到该评估任务（可能已过期或未开始）")


def _resolve_report(run_id: Optional[str]) -> Path:
    """定位报告文件：指定 run_id 时按前缀匹配，否则取最新。"""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    files = sorted(RESULTS_DIR.glob("*.csv"), key=lambda item: item.stat().st_mtime, reverse=True)
    if not files:
        raise NotFoundError("暂无评估结果，请先运行 scripts/experiment_compare.py 或提交评估任务")
    if run_id:
        matched = [item for item in files if run_id in item.name]
        if not matched:
            raise NotFoundError(f"未找到 run_id={run_id} 的结果文件")
        return matched[0]
    return files[0]


def _coerce(value: str) -> Any:
    """把 CSV 中的数字字符串转成数字，便于前端直接画图。"""
    text = (value or "").strip()
    if not text:
        return ""
    try:
        number = float(text)
    except ValueError:
        return text
    return int(number) if number.is_integer() and "." not in text else number

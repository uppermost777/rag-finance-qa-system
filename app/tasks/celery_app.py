"""Celery 应用：broker/backend 均使用 Redis，队列名 finance_qa。

启动方式（项目根目录）：
    celery -A app.tasks.celery_app:celery_app worker -Q finance_qa -l info -P solo   # Windows 本地
    celery -A app.tasks.celery_app:celery_app worker -Q finance_qa -l info -c 4      # Linux/Docker
"""

from __future__ import annotations

from celery import Celery

from app.core.config import get_settings
from app.core.logging import setup_logging

settings = get_settings()

celery_app = Celery(
    "finance_qa",
    broker=settings.celery_broker_url,
    backend=settings.celery_result_backend,
    include=["app.tasks.document_tasks", "app.tasks.evaluation_tasks"],
)

celery_app.conf.update(
    task_default_queue=settings.celery_queue,
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    timezone="Asia/Shanghai",
    enable_utc=False,
    task_track_started=True,
    task_acks_late=True,
    worker_prefetch_multiplier=1,
    task_time_limit=60 * 30,
    task_soft_time_limit=60 * 25,
    result_expires=60 * 60 * 24,
    broker_connection_retry_on_startup=True,
)

# 任务内统一日志格式（worker 启动时也会调用）
setup_logging()

__all__ = ["celery_app"]

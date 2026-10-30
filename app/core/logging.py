"""统一日志配置：控制台 + 文件双输出，支持 trace_id 贯穿单次请求。"""

from __future__ import annotations

import contextvars
import logging
import logging.handlers
import sys
import uuid
from pathlib import Path
from typing import Optional

from app.core.config import get_settings

# 单次请求/任务的追踪 ID，由中间件或 Celery 任务写入
trace_id_var: contextvars.ContextVar[str] = contextvars.ContextVar("trace_id", default="-")

_LOG_FORMAT = (
    "%(asctime)s | %(levelname)-7s | %(trace_id)s | %(name)s:%(lineno)d | %(message)s"
)
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


class TraceIdFilter(logging.Filter):
    """把 contextvar 中的 trace_id 注入日志记录。"""

    def filter(self, record: logging.LogRecord) -> bool:  # noqa: A003
        record.trace_id = trace_id_var.get()
        return True


def set_trace_id(trace_id: Optional[str] = None) -> str:
    """设置当前上下文的 trace_id；未传入时自动生成 12 位短 ID。"""
    value = trace_id or uuid.uuid4().hex[:12]
    trace_id_var.set(value)
    return value


def setup_logging(level: Optional[str] = None, log_dir: Optional[Path] = None) -> None:
    """初始化根 logger：控制台 + 按大小滚动的文件双输出（幂等）。"""
    settings = get_settings()
    log_level = (level or settings.log_level).upper()
    directory = log_dir or settings.log_path
    directory.mkdir(parents=True, exist_ok=True)

    root = logging.getLogger()
    if getattr(root, "_finance_qa_configured", False):
        root.setLevel(log_level)
        return

    root.setLevel(log_level)
    for handler in list(root.handlers):
        root.removeHandler(handler)

    formatter = logging.Formatter(_LOG_FORMAT, datefmt=_DATE_FORMAT)
    trace_filter = TraceIdFilter()

    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(formatter)
    console.addFilter(trace_filter)
    root.addHandler(console)

    file_handler = logging.handlers.RotatingFileHandler(
        filename=str(directory / "app.log"),
        maxBytes=20 * 1024 * 1024,
        backupCount=5,
        encoding="utf-8",
    )
    file_handler.setFormatter(formatter)
    file_handler.addFilter(trace_filter)
    root.addHandler(file_handler)

    # 降低第三方库噪音
    for noisy in ("httpx", "httpcore", "urllib3", "pymilvus", "asyncio"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    root._finance_qa_configured = True  # type: ignore[attr-defined]


def get_logger(name: str) -> logging.Logger:
    """获取带统一配置的 logger（等价于 logging.getLogger，仅作语义化封装）。"""
    return logging.getLogger(name)


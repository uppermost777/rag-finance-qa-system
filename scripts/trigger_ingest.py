"""手动触发一次文档入库（无 Celery worker 时的本地验证脚本）。

用法：
    python scripts/trigger_ingest.py --doc-id 3            # 走 Celery 队列
    python scripts/trigger_ingest.py --doc-id 3 --inline   # 直接在本进程同步执行
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.core.logging import setup_logging  # noqa: E402


def main() -> int:
    """脚本入口。"""
    parser = argparse.ArgumentParser(description="触发文档入库任务")
    parser.add_argument("--doc-id", type=int, required=True, help="document.id")
    parser.add_argument("--inline", action="store_true", help="不投递队列，直接同步执行")
    args = parser.parse_args()
    setup_logging()

    if args.inline:
        from app.tasks.document_tasks import progress_snapshot, run_ingest_sync

        payload = run_ingest_sync(args.doc_id)
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        print("进度快照：", progress_snapshot(args.doc_id))
        return 0 if payload.get("status") == "done" else 1

    try:
        from app.tasks.document_tasks import ingest_task

        async_result = ingest_task.delay(args.doc_id)
        print(f"已投递任务：task_id={async_result.id}（用 Celery 日志或进度接口查看结果）")
        return 0
    except Exception as exc:
        print(f"投递失败：{exc}\n可改用 --inline 在本进程直接执行。")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

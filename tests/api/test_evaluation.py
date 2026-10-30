"""评估接口 smoke test（任务投递、进度查询、报告读取）。"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import pytest
from httpx import AsyncClient

from tests.api.conftest import auth_headers


async def test_run_evaluation_dispatches_task(
    client: AsyncClient, seed: dict[str, int], monkeypatch: pytest.MonkeyPatch
) -> None:
    """提交评估任务：返回 run_id，并把参数透传给 Celery 任务。"""
    captured: dict[str, Any] = {}

    def _fake_dispatch(run_id: str, dataset: str, limit: int | None, strategies: list[str]) -> str:
        captured.update(
            {"run_id": run_id, "dataset": dataset, "limit": limit, "strategies": strategies}
        )
        return "task-eval-1"

    monkeypatch.setattr("app.api.v1.evaluation.dispatch_eval", _fake_dispatch)
    resp = await client.post(
        "/api/v1/evaluation/run",
        json={"dataset": "finance_qa_eval.jsonl", "limit": 5, "strategies": ["hybrid", "hybrid_rerank"]},
        headers=auth_headers(seed["admin_id"], role="admin"),
    )
    assert resp.status_code == 200
    data = resp.json()["data"]
    assert data["run_id"] and data["status"] == "PENDING"
    assert captured["limit"] == 5
    assert captured["strategies"] == ["hybrid", "hybrid_rerank"]


async def test_run_evaluation_requires_admin_and_queue(
    client: AsyncClient, seed: dict[str, int], monkeypatch: pytest.MonkeyPatch
) -> None:
    """学生无权触发评估；队列不可用时返回 502。"""
    student = await client.post(
        "/api/v1/evaluation/run",
        json={},
        headers=auth_headers(seed["student_id"]),
    )
    assert student.status_code == 403

    monkeypatch.setattr("app.api.v1.evaluation.dispatch_eval", lambda *args, **kwargs: None)
    admin = await client.post(
        "/api/v1/evaluation/run", json={}, headers=auth_headers(seed["admin_id"], role="admin")
    )
    assert admin.status_code == 502
    assert admin.json()["code"] == 50002


async def test_report_reads_latest_csv(
    client: AsyncClient, seed: dict[str, int], monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """读取评估 CSV 并转换为可画图的结构（数字已转成数值类型）。"""
    results_dir = tmp_path / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    csv_path = results_dir / "compare_20260101_120000.csv"
    with csv_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["config", "faithfulness", "answer_relevancy", "context_precision", "context_recall", "avg_latency_ms"]
        )
        writer.writerow(["A_vector", "0.71", "0.68", "0.62", "0.59", "820"])
        writer.writerow(["D_hybrid_rerank", "0.88", "0.85", "0.81", "0.79", "1450"])

    monkeypatch.setattr("app.api.v1.evaluation.RESULTS_DIR", results_dir)
    resp = await client.get(
        "/api/v1/evaluation/report", headers=auth_headers(seed["admin_id"], role="admin")
    )
    assert resp.status_code == 200
    data = resp.json()["data"]
    assert data["columns"][0] == "config"
    assert data["rows"][1]["config"] == "D_hybrid_rerank"
    assert data["rows"][1]["faithfulness"] == pytest.approx(0.88)
    assert isinstance(data["rows"][0]["avg_latency_ms"], int)


async def test_report_missing_file_returns_404(
    client: AsyncClient, seed: dict[str, int], monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """无评估结果时返回 404 并给出提示。"""
    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()
    monkeypatch.setattr("app.api.v1.evaluation.RESULTS_DIR", empty_dir)
    resp = await client.get(
        "/api/v1/evaluation/report", headers=auth_headers(seed["admin_id"], role="admin")
    )
    assert resp.status_code == 404
    assert "评估" in resp.json()["message"]


async def test_eval_status_unknown_run(
    client: AsyncClient, seed: dict[str, int], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Redis 中无该任务时返回 404。"""

    class _FakeRedis:
        async def hgetall(self, key: str) -> dict[str, str]:
            return {}

    monkeypatch.setattr("app.core.redis_client.get_redis", lambda: _FakeRedis())
    resp = await client.get(
        "/api/v1/evaluation/not-exists", headers=auth_headers(seed["admin_id"], role="admin")
    )
    assert resp.status_code == 404

"""对话接口 smoke test：会话、非流式问答、SSE 流式、反馈与统计。"""

from __future__ import annotations

import json

from httpx import AsyncClient

from tests.api.conftest import auth_headers


async def _create_conversation(client: AsyncClient, headers: dict[str, str], kb_id: int) -> int:
    """建一个会话并返回 ID。"""
    resp = await client.post(
        "/api/v1/chat/conversations", json={"title": "测试会话", "kb_ids": [kb_id]}, headers=headers
    )
    assert resp.status_code == 200
    return resp.json()["data"]["id"]


async def test_conversation_crud(client: AsyncClient, seed: dict[str, int]) -> None:
    """会话的增、查、改、删。"""
    headers = auth_headers(seed["student_id"])
    conversation_id = await _create_conversation(client, headers, seed["kb_id"])

    listed = await client.get("/api/v1/chat/conversations?page=1&size=10", headers=headers)
    assert listed.status_code == 200
    body = listed.json()["data"]
    assert body["total"] == 1 and body["items"][0]["id"] == conversation_id

    renamed = await client.patch(
        f"/api/v1/chat/conversations/{conversation_id}",
        json={"title": "改名后的会话", "kb_ids": []},
        headers=headers,
    )
    assert renamed.status_code == 200
    assert renamed.json()["data"]["title"] == "改名后的会话"
    assert renamed.json()["data"]["kb_ids"] == []

    messages = await client.get(
        f"/api/v1/chat/conversations/{conversation_id}/messages", headers=headers
    )
    assert messages.status_code == 200
    assert messages.json()["data"]["total"] == 0

    deleted = await client.delete(f"/api/v1/chat/conversations/{conversation_id}", headers=headers)
    assert deleted.status_code == 200
    after = await client.get("/api/v1/chat/conversations", headers=headers)
    assert after.json()["data"]["total"] == 0


async def test_conversation_isolation_between_users(client: AsyncClient, seed: dict[str, int]) -> None:
    """他人会话不可见 / 不可访问（返回 404，不泄露存在性）。"""
    owner_headers = auth_headers(seed["student_id"])
    conversation_id = await _create_conversation(client, owner_headers, seed["kb_id"])

    other_headers = auth_headers(seed["other_id"])
    resp = await client.get(
        f"/api/v1/chat/conversations/{conversation_id}/messages", headers=other_headers
    )
    assert resp.status_code == 404


async def test_completions_returns_answer_and_citations(
    client: AsyncClient, seed: dict[str, int]
) -> None:
    """非流式问答：返回答案、引用、qa_log_id 与耗时。"""
    headers = auth_headers(seed["student_id"])
    resp = await client.post(
        "/api/v1/chat/completions",
        json={"question": "什么是指数基金", "kb_ids": [seed["kb_id"]]},
        headers=headers,
    )
    assert resp.status_code == 200
    data = resp.json()["data"]
    assert data["answer"].startswith("这是基于知识库的回答")
    assert data["citations"][0]["doc_name"] == "基金投资基础.pdf"
    assert data["conversation_id"] > 0 and data["message_id"] > 0
    assert data["qa_log_id"] and data["latency_ms"] >= 0
    assert data["guard_action"] == "PASS"
    assert data["strategy"] == "hybrid_rerank"


async def test_completions_validation_error(client: AsyncClient, seed: dict[str, int]) -> None:
    """空问题返回 400（Pydantic 校验）。"""
    resp = await client.post(
        "/api/v1/chat/completions", json={"question": ""}, headers=auth_headers(seed["student_id"])
    )
    assert resp.status_code == 400
    assert resp.json()["code"] == 40001


async def test_completions_requires_auth(client: AsyncClient) -> None:
    """未登录访问问答接口返回 401。"""
    resp = await client.post("/api/v1/chat/completions", json={"question": "什么是基金定投"})
    assert resp.status_code == 401


async def test_sse_stream_emits_three_event_types(client: AsyncClient, seed: dict[str, int]) -> None:
    """SSE 流式：响应头正确，事件包含 citations / delta / done，且 done 带 message_id。"""
    headers = auth_headers(seed["student_id"])
    resp = await client.get(
        "/api/v1/chat/stream",
        params={"question": "什么是指数基金", "kb_ids": str(seed["kb_id"])},
        headers=headers,
    )
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("text/event-stream")
    assert resp.headers["cache-control"] == "no-cache"
    assert resp.headers["x-accel-buffering"] == "no"

    text = resp.text
    assert "event: citations" in text
    assert text.count("event: delta") >= 2
    assert "event: done" in text

    frames = [frame for frame in text.split("\n\n") if frame.strip()]
    payloads = [json.loads(frame.split("data: ", 1)[1]) for frame in frames if "data: " in frame]
    done = payloads[-1]
    assert done["message_id"] > 0
    assert done["conversation_id"] > 0
    assert "指数基金" in done["answer"]

    # 落库校验：消息已写入
    messages = await client.get(
        f"/api/v1/chat/conversations/{done['conversation_id']}/messages", headers=headers
    )
    assert messages.json()["data"]["total"] == 2


async def test_feedback_and_stats(client: AsyncClient, seed: dict[str, int]) -> None:
    """点赞/点踩与个人统计。"""
    headers = auth_headers(seed["student_id"])
    completion = await client.post(
        "/api/v1/chat/completions", json={"question": "什么是基金定投"}, headers=headers
    )
    message_id = completion.json()["data"]["message_id"]

    up = await client.post(
        f"/api/v1/chat/messages/{message_id}/feedback", json={"feedback": 1}, headers=headers
    )
    assert up.status_code == 200
    assert up.json()["data"]["feedback"] == 1

    bad = await client.post(
        f"/api/v1/chat/messages/{message_id}/feedback", json={"feedback": 0}, headers=headers
    )
    assert bad.status_code == 400

    other = await client.post(
        f"/api/v1/chat/messages/{message_id}/feedback",
        json={"feedback": -1},
        headers=auth_headers(seed["other_id"]),
    )
    assert other.status_code == 404

    stats = await client.get("/api/v1/chat/stats", headers=headers)
    assert stats.status_code == 200
    data = stats.json()["data"]
    assert data["question_count"] == 1
    assert data["feedback_up"] == 1
    assert data["conversation_count"] == 1

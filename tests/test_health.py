"""P2 验收：/health 健康检查与统一响应包装。"""

from __future__ import annotations

from httpx import AsyncClient


async def test_health_returns_200(client: AsyncClient) -> None:
    """健康检查应返回 200，且响应体符合统一包装 {code,message,data}。"""
    resp = await client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["code"] == 0
    assert body["message"] == "ok"
    assert body["data"]["status"] == "ok"
    assert body["data"]["version"]
    # 依赖探活字段必须存在（真实环境下未启动依赖时为 False，不应抛错）
    for key in ("mysql", "redis", "milvus"):
        assert key in body["data"]
        assert isinstance(body["data"][key], bool)


async def test_trace_id_header(client: AsyncClient) -> None:
    """每个响应都应带上 trace_id，便于日志串联。"""
    resp = await client.get("/health", headers={"X-Trace-Id": "abcd1234abcd"})
    assert resp.headers["X-Trace-Id"] == "abcd1234abcd"
    assert "X-Process-Time-Ms" in resp.headers


async def test_unknown_path_returns_404(client: AsyncClient) -> None:
    """未注册路径返回 404（FastAPI 默认），不泄露内部信息。"""
    resp = await client.get("/api/v1/not-exists")
    assert resp.status_code == 404


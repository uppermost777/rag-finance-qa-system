"""认证接口 smoke test。"""

from __future__ import annotations

from httpx import AsyncClient

from tests.api.conftest import auth_headers


async def test_register_login_and_me(client: AsyncClient, seed: dict[str, int]) -> None:
    """注册 → 登录 → 查询当前用户，完整闭环。"""
    resp = await client.post(
        "/api/v1/auth/register",
        json={"username": "new_student", "password": "Student@2026", "email": "new@example.com"},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["code"] == 0
    assert body["data"]["username"] == "new_student"
    assert body["data"]["role"] == "student"
    user_id = body["data"]["id"]

    login = await client.post(
        "/api/v1/auth/login", json={"username": "new_student", "password": "Student@2026"}
    )
    assert login.status_code == 200
    token = login.json()["data"]["access_token"]
    assert login.json()["data"]["expires_in"] > 0

    me = await client.get("/api/v1/auth/me", headers={"Authorization": f"Bearer {token}"})
    assert me.status_code == 200
    assert me.json()["data"]["id"] == user_id


async def test_register_rejects_duplicate_and_weak_password(
    client: AsyncClient, seed: dict[str, int]
) -> None:
    """重复用户名返回 409；弱密码返回 400。"""
    dup = await client.post(
        "/api/v1/auth/register", json={"username": seed["student_name"], "password": "Student@2026"}
    )
    assert dup.status_code == 409
    assert dup.json()["code"] == 40901

    weak = await client.post(
        "/api/v1/auth/register", json={"username": "weak_user", "password": "12345678"}
    )
    assert weak.status_code == 400

    short = await client.post(
        "/api/v1/auth/register", json={"username": "weak_user2", "password": "123"}
    )
    assert short.status_code == 400


async def test_login_rejects_wrong_password(client: AsyncClient, seed: dict[str, int]) -> None:
    """密码错误返回 401，且不泄露账号是否存在。"""
    resp = await client.post(
        "/api/v1/auth/login", json={"username": seed["student_name"], "password": "wrong-pass"}
    )
    assert resp.status_code == 401
    assert resp.json()["code"] == 40101


async def test_me_requires_token(client: AsyncClient) -> None:
    """未携带 token 访问受保护接口返回 401。"""
    resp = await client.get("/api/v1/auth/me")
    assert resp.status_code == 401
    assert resp.json()["message"]


async def test_me_rejects_invalid_token(client: AsyncClient) -> None:
    """伪造 token 返回 401。"""
    resp = await client.get("/api/v1/auth/me", headers={"Authorization": "Bearer not-a-jwt"})
    assert resp.status_code == 401


async def test_risk_profile_flow(client: AsyncClient, seed: dict[str, int]) -> None:
    """风险测评：取题目 → 提交答案 → 查询结果。"""
    questions = await client.get("/api/v1/auth/risk-questions")
    assert questions.status_code == 200
    assert questions.json()["data"]["total"] == 10

    headers = auth_headers(seed["student_id"])
    submit = await client.put(
        "/api/v1/auth/risk-profile", json={"answers": [2, 1, 1, 2, 1, 0, 1, 2, 1, 2]}, headers=headers
    )
    assert submit.status_code == 200
    data = submit.json()["data"]
    assert data["risk_level"] in {"C1", "C2", "C3", "C4", "C5"}
    assert len(data["risk_score"]["dimensions"]) == 5
    assert data["learning_path"]

    query = await client.get("/api/v1/auth/risk-profile", headers=headers)
    assert query.status_code == 200
    assert query.json()["data"]["risk_level"] == data["risk_level"]
    assert query.json()["data"]["suggestions"]["suggestion"]


async def test_risk_profile_rejects_wrong_answer_count(
    client: AsyncClient, seed: dict[str, int]
) -> None:
    """答案数量不足 10 题时返回 400（Pydantic 校验）。"""
    resp = await client.put(
        "/api/v1/auth/risk-profile",
        json={"answers": [1, 2, 3]},
        headers=auth_headers(seed["student_id"]),
    )
    assert resp.status_code == 400
    assert resp.json()["code"] == 40001


async def test_health_endpoint(client: AsyncClient) -> None:
    """健康检查在依赖不可用时仍返回 200（探活字段为 false）。"""
    resp = await client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["data"]["status"] == "ok"

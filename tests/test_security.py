"""P2 验收：密码哈希与 JWT 签发/解析。"""

from __future__ import annotations

import pytest

from app.core.exceptions import ParamError, UnauthorizedError
from app.core.security import (
    create_access_token,
    decode_token,
    hash_password,
    validate_username,
    verify_password,
)


def test_hash_and_verify_password() -> None:
    """bcrypt 哈希可往返校验，且相同明文两次哈希不同（随机盐）。"""
    raw = "Finance@2026"
    hashed_a = hash_password(raw)
    hashed_b = hash_password(raw)
    assert hashed_a != hashed_b
    assert hashed_a.startswith("$2")
    assert verify_password(raw, hashed_a)
    assert not verify_password("wrong-password", hashed_a)


def test_hash_password_rejects_weak_input() -> None:
    """过短或纯数字密码应被拒绝。"""
    with pytest.raises(ParamError):
        hash_password("123")
    with pytest.raises(ParamError):
        hash_password("12345678")


def test_validate_username() -> None:
    """用户名规则：3~32 位中文/字母/数字/下划线。"""
    validate_username("student_01")
    validate_username("理财小白")
    with pytest.raises(ParamError):
        validate_username("a")
    with pytest.raises(ParamError):
        validate_username("bad name!")


def test_jwt_roundtrip() -> None:
    """JWT 签发后可解析出相同 subject 与 role。"""
    token, expires_in = create_access_token(42, role="admin")
    assert expires_in > 0
    payload = decode_token(token)
    assert payload["sub"] == "42"
    assert payload["role"] == "admin"


def test_jwt_invalid_token_raises() -> None:
    """非法 token 解析失败应抛 UnauthorizedError。"""
    with pytest.raises(UnauthorizedError):
        decode_token("not-a-jwt")


def test_jwt_expired_token_raises() -> None:
    """过期 token 应抛 UnauthorizedError。"""
    token, _ = create_access_token(1, expires_minutes=-1)
    with pytest.raises(UnauthorizedError):
        decode_token(token)

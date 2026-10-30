"""安全工具：bcrypt 密码哈希与校验、JWT 签发与解析。"""

from __future__ import annotations

import re
from datetime import datetime, timedelta, timezone
from typing import Any, Optional

import bcrypt
import jwt

from app.core.config import get_settings
from app.core.exceptions import ParamError, UnauthorizedError

_PASSWORD_MIN_LEN = 6
_PASSWORD_MAX_LEN = 64
_USERNAME_PATTERN = re.compile(r"^[A-Za-z0-9_\u4e00-\u9fa5]{3,32}$")


def hash_password(password: str) -> str:
    """使用 bcrypt 生成密码哈希（自带随机盐）。"""
    _validate_password(password)
    return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt(rounds=12)).decode("utf-8")


def verify_password(password: str, password_hash: str) -> bool:
    """校验明文密码与哈希是否匹配（异常一律视为不匹配）。"""
    try:
        return bcrypt.checkpw(password.encode("utf-8"), password_hash.encode("utf-8"))
    except (ValueError, TypeError):
        return False


def _validate_password(password: str) -> None:
    """密码强度基础校验：长度限制，禁止纯数字。"""
    if not password or len(password) < _PASSWORD_MIN_LEN:
        raise ParamError(f"密码长度不能少于 {_PASSWORD_MIN_LEN} 位")
    if len(password) > _PASSWORD_MAX_LEN:
        raise ParamError(f"密码长度不能超过 {_PASSWORD_MAX_LEN} 位")
    if password.isdigit():
        raise ParamError("密码不能为纯数字，请混合字母或符号")


def validate_username(username: str) -> None:
    """用户名格式校验：3~32 位中文、字母、数字或下划线。"""
    if not _USERNAME_PATTERN.match(username or ""):
        raise ParamError("用户名需为 3~32 位中文、字母、数字或下划线")


def create_access_token(
    subject: str | int,
    *,
    role: str = "student",
    expires_minutes: Optional[int] = None,
    extra_claims: Optional[dict[str, Any]] = None,
) -> tuple[str, int]:
    """签发 JWT。

    :param subject: 用户 ID（写入 ``sub``）
    :param role: 用户角色，写入自定义 claim
    :param expires_minutes: 过期分钟数，默认取配置
    :return: (token, 过期秒数)
    """
    settings = get_settings()
    minutes = expires_minutes or settings.jwt_expire_minutes
    now = datetime.now(tz=timezone.utc)
    expire = now + timedelta(minutes=minutes)
    payload: dict[str, Any] = {
        "sub": str(subject),
        "role": role,
        "iat": int(now.timestamp()),
        "exp": int(expire.timestamp()),
    }
    if extra_claims:
        payload.update(extra_claims)
    token = jwt.encode(payload, settings.jwt_secret, algorithm=settings.jwt_algorithm)
    return token, minutes * 60


def decode_token(token: str) -> dict[str, Any]:
    """解析并校验 JWT；失败抛 UnauthorizedError。"""
    settings = get_settings()
    try:
        return jwt.decode(token, settings.jwt_secret, algorithms=[settings.jwt_algorithm])
    except jwt.ExpiredSignatureError as exc:
        raise UnauthorizedError("登录已过期，请重新登录") from exc
    except jwt.PyJWTError as exc:
        raise UnauthorizedError("登录凭证无效") from exc


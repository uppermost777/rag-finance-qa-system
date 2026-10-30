"""FastAPI 依赖注入：当前用户解析、管理员校验、分页参数。"""

from __future__ import annotations

from typing import Annotated, Optional

from fastapi import Depends, Query, Request
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db
from app.core.exceptions import UnauthorizedError
from app.core.logging import get_logger
from app.core.security import decode_token
from app.models import User

logger = get_logger(__name__)

# auto_error=False：未携带 token 时由我们自己抛业务异常，保证响应体格式统一
bearer_scheme = HTTPBearer(auto_error=False)


async def get_current_user(
    request: Request,
    credentials: Annotated[Optional[HTTPAuthorizationCredentials], Depends(bearer_scheme)] = None,
    db: AsyncSession = Depends(get_db),
) -> User:
    """解析 Bearer Token 并返回当前用户。"""
    if credentials is None or not credentials.credentials:
        raise UnauthorizedError("请先登录")
    payload = decode_token(credentials.credentials)
    try:
        user_id = int(payload.get("sub", 0))
    except (TypeError, ValueError) as exc:
        raise UnauthorizedError("登录凭证无效") from exc
    user = await db.get(User, user_id)
    if user is None or user.status != 1:
        raise UnauthorizedError("账号不存在或已被禁用")
    request.state.user_id = user.id
    return user


async def require_admin(user: Annotated[User, Depends(get_current_user)]) -> User:
    """要求管理员角色。"""
    if not user.is_admin:
        from app.core.exceptions import PermissionError

        raise PermissionError("该操作仅管理员可用")
    return user


def pagination(
    page: int = Query(1, ge=1, description="页码，从 1 开始"),
    size: int = Query(20, ge=1, le=100, description="每页条数"),
) -> tuple[int, int]:
    """统一分页参数。"""
    return page, size


CurrentUser = Annotated[User, Depends(get_current_user)]
AdminUser = Annotated[User, Depends(require_admin)]
DbSession = Annotated[AsyncSession, Depends(get_db)]
PageParams = Annotated[tuple[int, int], Depends(pagination)]

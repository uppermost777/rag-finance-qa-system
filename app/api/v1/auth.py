"""认证与用户接口：注册、登录、当前用户、风险测评。"""

from __future__ import annotations

from fastapi import APIRouter
from sqlalchemy import select

from app.core.deps import CurrentUser, DbSession
from app.core.exceptions import ConflictError, ParamError, UnauthorizedError
from app.core.logging import get_logger
from app.core.security import (
    create_access_token,
    hash_password,
    validate_username,
    verify_password,
)
from app.models import User
from app.models.base import utcnow
from app.schemas.auth import (
    LoginRequest,
    RegisterRequest,
    RiskProfileRequest,
    RiskProfileResponse,
    TokenResponse,
    UserInfo,
)
from app.schemas.common import ok
from app.services.risk_service import compute_risk_profile, question_bank

logger = get_logger(__name__)

router = APIRouter(prefix="/auth", tags=["认证"])


def _to_user_info(user: User) -> UserInfo:
    """ORM → 响应模型。"""
    return UserInfo(
        id=user.id,
        username=user.username,
        email=user.email,
        role=user.role,
        risk_level=user.risk_level,
        created_at=user.created_at,
    )


@router.post("/register", summary="注册")
async def register(payload: RegisterRequest, db: DbSession) -> dict:
    """注册新用户（默认学生角色）。"""
    validate_username(payload.username)
    existing = await db.scalar(select(User).where(User.username == payload.username))
    if existing is not None:
        raise ConflictError("该用户名已被注册")
    if payload.email:
        email_used = await db.scalar(select(User).where(User.email == payload.email))
        if email_used is not None:
            raise ConflictError("该邮箱已被注册")
    user = User(
        username=payload.username,
        email=payload.email,
        password_hash=hash_password(payload.password),
        role="student",
        status=1,
    )
    db.add(user)
    await db.commit()
    logger.info("新用户注册：%s", user.username)
    return ok(_to_user_info(user).model_dump())


@router.post("/login", summary="登录")
async def login(payload: LoginRequest, db: DbSession) -> dict:
    """账号密码登录，返回 JWT。"""
    user = await db.scalar(select(User).where(User.username == payload.username))
    if user is None or not verify_password(payload.password, user.password_hash):
        raise UnauthorizedError("用户名或密码错误")
    if user.status != 1:
        raise UnauthorizedError("账号已被禁用，请联系管理员")
    token, expires_in = create_access_token(user.id, role=user.role)
    user.last_login_at = utcnow()
    await db.commit()
    return ok(
        TokenResponse(
            access_token=token, expires_in=expires_in, user=_to_user_info(user)
        ).model_dump()
    )


@router.get("/me", summary="当前用户信息")
async def me(user: CurrentUser) -> dict:
    """返回当前登录用户信息。"""
    return ok(_to_user_info(user).model_dump())


@router.get("/risk-questions", summary="风险测评题目")
async def risk_questions() -> dict:
    """返回 10 道风险测评题（登录前也可预览）。"""
    return ok({"total": len(question_bank()), "questions": question_bank()})


@router.put("/risk-profile", summary="提交风险测评")
async def submit_risk_profile(
    payload: RiskProfileRequest, user: CurrentUser, db: DbSession
) -> dict:
    """提交风险测评结果并写回用户档案。"""
    try:
        profile = compute_risk_profile(payload.answers)
    except ValueError as exc:
        raise ParamError(str(exc)) from exc
    user.risk_level = profile["risk_level"]
    user.risk_score = profile["risk_score"]
    await db.commit()
    logger.info("用户 %s 完成风险测评：%s", user.username, profile["risk_level"])
    return ok(profile)


@router.get("/risk-profile", summary="查询风险测评结果")
async def get_risk_profile(user: CurrentUser) -> dict:
    """查询当前用户已保存的风险测评结果。"""
    if not user.risk_level:
        return ok({"risk_level": None, "message": "尚未完成风险测评"})
    return ok(
        {
            "risk_level": user.risk_level,
            "risk_score": user.risk_score or {},
            "suggestions": _advice_for(user.risk_level),
        }
    )


def _advice_for(level: str) -> dict:
    """按等级返回建议与学习路径。"""
    from app.services.risk_service import LEVEL_ADVICE

    return LEVEL_ADVICE.get(level, {})

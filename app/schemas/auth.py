"""认证与用户相关请求/响应模型。"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Optional

from pydantic import BaseModel, Field


class RegisterRequest(BaseModel):
    """注册请求。"""

    username: str = Field(..., min_length=3, max_length=32, description="登录名")
    password: str = Field(..., min_length=6, max_length=64, description="密码")
    email: Optional[str] = Field(default=None, max_length=128, description="邮箱（可选）")


class LoginRequest(BaseModel):
    """登录请求。"""

    username: str = Field(..., max_length=32)
    password: str = Field(..., max_length=64)


class UserInfo(BaseModel):
    """用户信息。"""

    id: int
    username: str
    email: Optional[str] = None
    role: str
    risk_level: Optional[str] = None
    created_at: Optional[datetime] = None


class TokenResponse(BaseModel):
    """登录响应。"""

    access_token: str
    token_type: str = "bearer"
    expires_in: int
    user: UserInfo


class RiskProfileRequest(BaseModel):
    """风险测评提交（10 题，每题选项得分 0~3）。"""

    answers: list[int] = Field(..., min_length=10, max_length=10, description="10 题选项得分")


class RiskProfileResponse(BaseModel):
    """风险测评结果。"""

    risk_level: str
    risk_label: str
    risk_score: dict[str, Any]
    suggestion: str
    learning_path: list[str] = Field(default_factory=list)

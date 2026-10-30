"""统一响应包装：{code, message, data}。"""

from __future__ import annotations

from typing import Any, Generic, Optional, TypeVar

from pydantic import BaseModel, Field

from app.core.exceptions import ErrorCode

T = TypeVar("T")


class ApiResponse(BaseModel, Generic[T]):
    """统一响应体。"""

    code: int = Field(default=0, description="业务错误码，0 表示成功")
    message: str = Field(default="ok", description="提示信息")
    data: Optional[T] = Field(default=None, description="业务数据")


def ok(data: Any = None, message: str = "ok") -> dict[str, Any]:
    """构造成功响应。"""
    return {"code": int(ErrorCode.SUCCESS), "message": message, "data": data}


def fail(code: int, message: str, data: Any = None) -> dict[str, Any]:
    """构造失败响应。"""
    return {"code": int(code), "message": message, "data": data}


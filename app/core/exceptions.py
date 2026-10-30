"""统一业务异常体系：错误码枚举 + 异常基类 + 常用子类。"""

from __future__ import annotations

from enum import IntEnum
from typing import Any, Optional


class ErrorCode(IntEnum):
    """业务错误码：0 表示成功，非 0 与 HTTP 状态码配合使用。"""

    SUCCESS = 0
    PARAM_ERROR = 40001
    UNAUTHORIZED = 40101
    FORBIDDEN = 40301
    NOT_FOUND = 40401
    CONFLICT = 40901
    RATE_LIMITED = 42901
    COMPLIANCE_BLOCKED = 43001
    INTERNAL_ERROR = 50001
    EXTERNAL_SERVICE_ERROR = 50002
    PATH_TRAVERSAL = 50003


class BizException(Exception):
    """业务异常基类：携带错误码、用户可读信息与 HTTP 状态码。"""

    code: ErrorCode = ErrorCode.INTERNAL_ERROR
    http_status: int = 500
    message: str = "服务内部错误"

    def __init__(
        self,
        message: Optional[str] = None,
        *,
        code: Optional[ErrorCode] = None,
        http_status: Optional[int] = None,
        detail: Optional[Any] = None,
    ) -> None:
        self.message = message or self.message
        self.code = code or self.code
        self.http_status = http_status or self.http_status
        self.detail = detail
        super().__init__(self.message)

    def to_dict(self) -> dict[str, Any]:
        """转换为统一响应体片段。"""
        payload: dict[str, Any] = {"code": int(self.code), "message": self.message, "data": None}
        if self.detail is not None:
            payload["detail"] = self.detail
        return payload


class ParamError(BizException):
    """参数校验失败。"""

    code = ErrorCode.PARAM_ERROR
    http_status = 400
    message = "请求参数有误"


class UnauthorizedError(BizException):
    """未认证或 token 无效。"""

    code = ErrorCode.UNAUTHORIZED
    http_status = 401
    message = "登录状态已失效，请重新登录"


class PermissionError(BizException):  # noqa: A001 - 与内置异常重名是刻意为之，对外语义更直观
    """权限不足。"""

    code = ErrorCode.FORBIDDEN
    http_status = 403
    message = "没有权限执行该操作"


class NotFoundError(BizException):
    """资源不存在。"""

    code = ErrorCode.NOT_FOUND
    http_status = 404
    message = "资源不存在"


class ConflictError(BizException):
    """资源冲突（如用户名重复）。"""

    code = ErrorCode.CONFLICT
    http_status = 409
    message = "资源已存在"


class RateLimitedError(BizException):
    """触发限流。"""

    code = ErrorCode.RATE_LIMITED
    http_status = 429
    message = "请求过于频繁，请稍后再试"


class ExternalServiceError(BizException):
    """外部依赖（Milvus / Redis / LLM 等）不可用。"""

    code = ErrorCode.EXTERNAL_SERVICE_ERROR
    http_status = 502
    message = "外部服务暂时不可用"


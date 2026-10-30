"""FastAPI 应用入口：中间件、全局异常处理、健康检查、路由挂载。"""

from __future__ import annotations

import asyncio
import time
from contextlib import asynccontextmanager
from typing import Any, AsyncGenerator

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

from app.api.v1 import router as api_v1_router
from app.core.config import get_settings
from app.core.database import dispose_engine, get_engine
from app.core.exceptions import BizException, ErrorCode
from app.core.logging import get_logger, set_trace_id, setup_logging
from app.schemas.common import fail, ok

logger = get_logger(__name__)


class TraceIdMiddleware(BaseHTTPMiddleware):
    """为每个请求分配 trace_id，并注入响应头 X-Trace-Id。"""

    async def dispatch(self, request: Request, call_next: Any) -> Any:
        trace_id = set_trace_id(request.headers.get("X-Trace-Id"))
        request.state.trace_id = trace_id
        start = time.perf_counter()
        response = await call_next(request)
        cost_ms = round((time.perf_counter() - start) * 1000, 2)
        response.headers["X-Trace-Id"] = trace_id
        response.headers["X-Process-Time-Ms"] = str(cost_ms)
        if request.url.path.startswith(get_settings().api_v1_prefix):
            logger.info("%s %s -> %s (%sms)", request.method, request.url.path, response.status_code, cost_ms)
        return response


async def _check_mysql() -> bool:
    """探测 MySQL 可用性。"""
    from sqlalchemy import text

    try:
        async with get_engine().connect() as conn:
            await conn.execute(text("SELECT 1"))
        return True
    except Exception as exc:  # pragma: no cover - 依赖真实环境
        logger.warning("MySQL 健康检查失败：%s", exc)
        return False


async def _check_redis() -> bool:
    """探测 Redis 可用性。"""
    try:
        from app.core.redis_client import get_redis

        client = get_redis()
        await asyncio.wait_for(client.ping(), timeout=1.5)
        return True
    except Exception as exc:  # pragma: no cover - 依赖真实环境
        logger.warning("Redis 健康检查失败：%s", exc)
        return False


async def _check_milvus() -> bool:
    """探测 Milvus 可用性。"""
    try:
        from app.vectorstore.milvus_client import MilvusClientWrapper

        return await asyncio.wait_for(
            MilvusClientWrapper.get_instance().ping(), timeout=1.5
        )
    except Exception as exc:  # pragma: no cover - 依赖真实环境
        logger.warning("Milvus 健康检查失败：%s", exc)
        return False


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """应用生命周期：启动时初始化日志与上传目录，关闭时释放连接池。"""
    settings = get_settings()
    setup_logging()
    settings.upload_path.mkdir(parents=True, exist_ok=True)
    logger.info("%s v%s 启动完成（debug=%s）", settings.app_name, settings.app_version, settings.debug)
    yield
    await dispose_engine()
    logger.info("%s 已关闭", settings.app_name)


def create_app() -> FastAPI:
    """工厂函数：构建并返回 FastAPI 实例。"""
    settings = get_settings()
    app = FastAPI(
        title="大学生理财知识问答系统 API",
        description="基于 LangChain 与向量数据库的 RAG 问答系统；仅作知识科普，不构成投资建议。",
        version=settings.app_version,
        lifespan=lifespan,
    )
    app.add_middleware(TraceIdMiddleware)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
        expose_headers=["X-Trace-Id", "X-Process-Time-Ms"],
    )

    @app.exception_handler(BizException)
    async def _biz_handler(request: Request, exc: BizException) -> JSONResponse:
        """业务异常统一响应。"""
        logger.warning("业务异常 [%s] %s: %s", exc.code, request.url.path, exc.message)
        return JSONResponse(status_code=exc.http_status, content=exc.to_dict())

    @app.exception_handler(RequestValidationError)
    async def _validation_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
        """Pydantic 参数校验异常统一响应。"""
        errors = [
            {"field": ".".join(str(x) for x in err.get("loc", [])), "msg": err.get("msg")}
            for err in exc.errors()
        ]
        return JSONResponse(
            status_code=400,
            content=fail(ErrorCode.PARAM_ERROR, "请求参数有误", errors),
        )

    @app.exception_handler(Exception)
    async def _unhandled_handler(request: Request, exc: Exception) -> JSONResponse:
        """兜底异常处理，避免泄露堆栈给前端。"""
        logger.exception("未捕获异常 %s: %s", request.url.path, exc)
        return JSONResponse(
            status_code=500, content=fail(ErrorCode.INTERNAL_ERROR, "服务内部错误，请稍后重试")
        )

    @app.get("/health", tags=["系统"], summary="健康检查")
    async def health() -> dict[str, Any]:
        """健康检查：返回各外部依赖探活结果（任一失败不影响本接口可用性）。"""
        mysql_ok, redis_ok, milvus_ok = await asyncio.gather(
            _check_mysql(), _check_redis(), _check_milvus()
        )
        return ok(
            {
                "status": "ok",
                "version": settings.app_version,
                "mysql": mysql_ok,
                "redis": redis_ok,
                "milvus": milvus_ok,
            }
        )

    app.include_router(api_v1_router, prefix=settings.api_v1_prefix)
    return app


app = create_app()


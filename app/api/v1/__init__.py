"""v1 版本路由聚合。"""

from fastapi import APIRouter

from app.api.v1 import auth, chat, document, evaluation, kb

router = APIRouter()

# 各业务模块路由
ROUTERS: list[APIRouter] = [
    auth.router,
    chat.router,
    kb.router,
    document.router,
    evaluation.router,
]

for _sub in ROUTERS:
    router.include_router(_sub)

__all__ = ["router", "auth", "chat", "kb", "document", "evaluation"]


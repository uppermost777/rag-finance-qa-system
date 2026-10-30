"""对话接口：会话管理、非流式问答、SSE 流式问答、反馈与个人统计。"""

from __future__ import annotations

import json
from typing import Any, Optional

from fastapi import APIRouter, Query
from fastapi.responses import StreamingResponse
from sqlalchemy import func, select

from app.core.deps import CurrentUser, DbSession, PageParams
from app.core.exceptions import NotFoundError
from app.core.logging import get_logger
from app.models import Conversation, Message, QaLog
from app.models.base import utcnow
from app.schemas.chat import (
    ChatRequest,
    ChatResponse,
    ConversationCreateRequest,
    ConversationItem,
    ConversationListResponse,
    ConversationUpdateRequest,
    FeedbackRequest,
    FeedbackResponse,
    MessageItem,
    MessageListResponse,
)
from app.schemas.common import ok
from app.services.qa_service import QaService

logger = get_logger(__name__)

router = APIRouter(prefix="/chat", tags=["对话"])


def _to_conversation_item(conversation: Conversation) -> ConversationItem:
    """ORM → 会话响应模型。"""
    return ConversationItem(
        id=conversation.id,
        title=conversation.title,
        kb_ids=list(conversation.kb_ids or []),
        message_count=conversation.message_count or 0,
        last_message_at=conversation.last_message_at,
        created_at=conversation.created_at,
    )


def _to_message_item(message: Message) -> MessageItem:
    """ORM → 消息响应模型。"""
    return MessageItem(
        id=message.id,
        role=message.role,
        content=message.content,
        citations=list(message.citations or []),
        latency_ms=message.latency_ms,
        blocked=bool(message.blocked),
        feedback=message.feedback,
        created_at=message.created_at,
    )


async def _get_own_conversation(db: DbSession, conversation_id: int, user_id: int) -> Conversation:
    """校验会话归属。"""
    conversation = await db.get(Conversation, int(conversation_id))
    if conversation is None or conversation.is_deleted or conversation.user_id != int(user_id):
        raise NotFoundError("会话不存在或已删除")
    return conversation


# --------------------------------------------------------------------------- 会话管理
@router.post("/conversations", summary="新建会话")
async def create_conversation(
    payload: ConversationCreateRequest, user: CurrentUser, db: DbSession
) -> dict:
    """新建一个会话。"""
    conversation = Conversation(
        user_id=user.id,
        title=payload.title or "新会话",
        kb_ids=[int(item) for item in payload.kb_ids],
        message_count=0,
    )
    db.add(conversation)
    await db.commit()
    return ok(_to_conversation_item(conversation).model_dump())


@router.get("/conversations", summary="会话列表")
async def list_conversations(
    user: CurrentUser, db: DbSession, page_params: PageParams, keyword: Optional[str] = None
) -> dict:
    """分页查询当前用户的会话（按最近消息倒序）。"""
    page, size = page_params
    conditions = [Conversation.user_id == user.id, Conversation.is_deleted == 0]
    if keyword:
        conditions.append(Conversation.title.like(f"%{keyword}%"))
    total = int(
        await db.scalar(select(func.count()).select_from(Conversation).where(*conditions)) or 0
    )
    rows = (
        await db.scalars(
            select(Conversation)
            .where(*conditions)
            .order_by(Conversation.last_message_at.desc().nullslast(), Conversation.id.desc())
            .offset((page - 1) * size)
            .limit(size)
        )
    ).all()
    payload = ConversationListResponse(
        total=total, page=page, size=size, items=[_to_conversation_item(item) for item in rows]
    )
    return ok(payload.model_dump(mode="json"))


@router.patch("/conversations/{conversation_id}", summary="修改会话")
async def update_conversation(
    conversation_id: int,
    payload: ConversationUpdateRequest,
    user: CurrentUser,
    db: DbSession,
) -> dict:
    """重命名会话或调整检索范围。"""
    conversation = await _get_own_conversation(db, conversation_id, user.id)
    if payload.title:
        conversation.title = payload.title[:128]
    if payload.kb_ids is not None:
        conversation.kb_ids = [int(item) for item in payload.kb_ids]
    await db.commit()
    return ok(_to_conversation_item(conversation).model_dump())


@router.delete("/conversations/{conversation_id}", summary="删除会话")
async def delete_conversation(conversation_id: int, user: CurrentUser, db: DbSession) -> dict:
    """逻辑删除会话。"""
    conversation = await _get_own_conversation(db, conversation_id, user.id)
    conversation.is_deleted = 1
    await db.commit()
    return ok({"id": conversation.id})


@router.get("/conversations/{conversation_id}/messages", summary="历史消息")
async def list_messages(
    conversation_id: int, user: CurrentUser, db: DbSession, page_params: PageParams
) -> dict:
    """分页查询会话内消息（按 id 正序）。"""
    page, size = page_params
    await _get_own_conversation(db, conversation_id, user.id)
    total = int(
        await db.scalar(
            select(func.count()).select_from(Message).where(Message.conversation_id == conversation_id)
        )
        or 0
    )
    rows = (
        await db.scalars(
            select(Message)
            .where(Message.conversation_id == conversation_id)
            .order_by(Message.id)
            .offset((page - 1) * size)
            .limit(size)
        )
    ).all()
    payload = MessageListResponse(
        total=total, page=page, size=size, items=[_to_message_item(item) for item in rows]
    )
    return ok(payload.model_dump(mode="json"))


# --------------------------------------------------------------------------- 问答
@router.post("/completions", summary="非流式问答")
async def completions(payload: ChatRequest, user: CurrentUser, db: DbSession) -> dict:
    """一次性返回完整答案与引用来源。"""
    service = QaService(db)
    result = await service.answer(
        payload.question,
        user_id=user.id,
        conversation_id=payload.conversation_id,
        kb_ids=payload.kb_ids,
        strategy=payload.strategy,
    )
    return ok(
        ChatResponse(
            conversation_id=result.conversation_id,
            message_id=result.assistant_message_id,
            user_message_id=result.user_message_id,
            qa_log_id=result.qa_log_id,
            answer=result.answer,
            citations=result.citations,
            blocked=result.blocked,
            guard_action=result.guard_action,
            strategy=result.strategy,
            need_human_hint=result.need_human_hint,
            latency_ms=result.latency_ms,
        ).model_dump()
    )


@router.get("/stream", summary="SSE 流式问答")
async def stream(
    user: CurrentUser,
    db: DbSession,
    question: str = Query(..., min_length=1, max_length=500, description="用户问题"),
    conversation_id: Optional[int] = Query(default=None, description="会话 ID，为空则新建"),
    kb_ids: Optional[str] = Query(default=None, description="知识库 ID，逗号分隔，如 1,2"),
    strategy: Optional[str] = Query(default=None, description="检索策略（对比实验用）"),
) -> StreamingResponse:
    """SSE 流式问答：``citations`` → ``delta``* → ``done``。"""
    scope: Optional[list[int]] = None
    if kb_ids:
        scope = [int(item) for item in kb_ids.split(",") if item.strip().isdigit()]

    async def event_source():
        """把服务层事件编码为 SSE 帧。"""
        service = QaService(db)
        try:
            async for event in service.astream(
                question, user_id=user.id, conversation_id=conversation_id, kb_ids=scope, strategy=strategy
            ):
                payload = json.dumps(event.data, ensure_ascii=False)
                yield f"event: {event.event}\ndata: {payload}\n\n"
        except Exception as exc:  # pragma: no cover - 兜底，保证前端能正常收尾
            logger.exception("SSE 流式问答异常：%s", exc)
            payload = json.dumps({"message": f"问答失败：{exc}"}, ensure_ascii=False)
            yield f"event: error\ndata: {payload}\n\n"

    return StreamingResponse(
        event_source(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


# --------------------------------------------------------------------------- 反馈与统计
@router.post("/messages/{message_id}/feedback", summary="点赞/点踩")
async def feedback(
    message_id: int, payload: FeedbackRequest, user: CurrentUser, db: DbSession
) -> dict:
    """对某条助手消息点赞或点踩。"""
    message = await db.scalar(
        select(Message).where(Message.id == int(message_id), Message.user_id == user.id)
    )
    if message is None:
        raise NotFoundError("消息不存在")
    message.feedback = int(payload.feedback)
    await db.commit()
    return ok(FeedbackResponse(message_id=message.id, feedback=message.feedback).model_dump())


@router.get("/stats", summary="个人提问统计")
async def stats(user: CurrentUser, db: DbSession) -> dict:
    """个人中心的统计数据：提问数、被拦截数、会话数、点赞分布。"""
    question_count = int(
        await db.scalar(
            select(func.count()).select_from(QaLog).where(QaLog.user_id == user.id)
        )
        or 0
    )
    blocked_count = int(
        await db.scalar(
            select(func.count())
            .select_from(QaLog)
            .where(QaLog.user_id == user.id, QaLog.guard_action != "PASS")
        )
        or 0
    )
    conversation_count = int(
        await db.scalar(
            select(func.count())
            .select_from(Conversation)
            .where(Conversation.user_id == user.id, Conversation.is_deleted == 0)
        )
        or 0
    )
    today_count = int(
        await db.scalar(
            select(func.count())
            .select_from(QaLog)
            .where(QaLog.user_id == user.id, QaLog.created_at >= utcnow().replace(hour=0, minute=0, second=0, microsecond=0))
        )
        or 0
    )
    up = int(
        await db.scalar(
            select(func.count()).select_from(Message).where(Message.user_id == user.id, Message.feedback == 1)
        )
        or 0
    )
    down = int(
        await db.scalar(
            select(func.count()).select_from(Message).where(Message.user_id == user.id, Message.feedback == -1)
        )
        or 0
    )
    latency = await db.scalar(
        select(func.avg(QaLog.latency_ms)).where(QaLog.user_id == user.id)
    )
    return ok(
        {
            "question_count": question_count,
            "blocked_count": blocked_count,
            "conversation_count": conversation_count,
            "today_question_count": today_count,
            "feedback_up": up,
            "feedback_down": down,
            "avg_latency_ms": round(float(latency), 1) if latency is not None else None,
            "risk_level": user.risk_level,
        }
    )

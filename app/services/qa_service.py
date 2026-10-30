"""问答业务服务：组合 LangGraph 流水线与存储层。

职责：
1. 会话的创建/复用与历史消息加载（供多轮改写使用）；
2. 保存用户消息与助手消息（含 citations 快照）；
3. 写 ``qa_log`` 全链路日志（召回数量、各阶段耗时、合规动作），作为论文实验数据来源；
4. 问答接口限流（Redis 计数器，Redis 不可用时自动降级为不限流）。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, AsyncGenerator, Optional, Sequence

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.chains.graph import QaPipeline, QaResult, get_pipeline
from app.core.config import get_settings
from app.core.exceptions import NotFoundError, RateLimitedError
from app.core.logging import get_logger
from app.models import Conversation, Message, QaLog
from app.models.base import utcnow

logger = get_logger(__name__)

# 送入多轮改写的最大历史消息条数
MAX_HISTORY_MESSAGES = 12


@dataclass
class QaAnswer:
    """非流式问答的返回体。"""

    answer: str
    citations: list[dict[str, Any]] = field(default_factory=list)
    conversation_id: int = 0
    user_message_id: int = 0
    assistant_message_id: int = 0
    qa_log_id: Optional[int] = None
    latency_ms: int = 0
    blocked: bool = False
    guard_action: str = "PASS"
    strategy: str = ""
    need_human_hint: bool = False


@dataclass
class StreamEvent:
    """SSE 事件（event 名 + data 负载）。"""

    event: str
    data: dict[str, Any]


class QaService:
    """问答服务（每个请求一个实例，持有请求级 session）。"""

    def __init__(
        self,
        session: AsyncSession,
        *,
        pipeline: Optional[QaPipeline] = None,
        enable_rate_limit: bool = True,
    ) -> None:
        """初始化。

        :param session: 请求级异步会话
        :param pipeline: 问答流水线，默认取进程级实例
        :param enable_rate_limit: 是否启用限流（测试可关闭）
        """
        self._session = session
        self._pipeline = pipeline
        self._enable_rate_limit = enable_rate_limit

    @property
    def pipeline(self) -> QaPipeline:
        """问答流水线。"""
        if self._pipeline is None:
            self._pipeline = get_pipeline()
        return self._pipeline

    # ------------------------------------------------------------------ 限流
    async def _guard_rate_limit(self, user_id: int) -> None:
        """接口限流：Redis 分钟级计数器（异常时放行，不影响可用性）。"""
        if not self._enable_rate_limit:
            return
        try:
            from app.core.redis_client import KEY_RATE_LIMIT, get_redis

            bucket = datetime.now().strftime("%Y%m%d%H%M")
            key = KEY_RATE_LIMIT.format(user_id=int(user_id), bucket=bucket)
            client = get_redis()
            count = await client.incr(key)
            if count == 1:
                await client.expire(key, 60)
            if count > get_settings().rate_limit_per_minute:
                raise RateLimitedError(
                    f"每分钟最多提问 {get_settings().rate_limit_per_minute} 次，请稍后再试"
                )
        except RateLimitedError:
            raise
        except Exception as exc:  # pragma: no cover - Redis 不可用时降级
            logger.debug("限流检查跳过（Redis 不可用）：%s", exc)

    # ------------------------------------------------------------------ 会话与消息
    async def ensure_conversation(
        self,
        user_id: int,
        *,
        conversation_id: Optional[int] = None,
        kb_ids: Optional[Sequence[int]] = None,
        first_question: str = "",
    ) -> Conversation:
        """获取或创建会话。

        :param conversation_id: 已存在的会话 ID；不存在则抛 404
        :param kb_ids: 新建会话时的检索范围
        :param first_question: 用于生成会话标题
        """
        if conversation_id:
            conversation = await self._session.get(Conversation, int(conversation_id))
            if conversation is None or conversation.is_deleted:
                raise NotFoundError("会话不存在或已删除")
            if conversation.user_id != int(user_id):
                raise NotFoundError("会话不存在或已删除")
            if kb_ids:
                conversation.kb_ids = [int(item) for item in kb_ids]
            return conversation

        title = (first_question or "新会话").strip().replace("\n", " ")[:30] or "新会话"
        conversation = Conversation(
            user_id=int(user_id),
            title=title,
            kb_ids=[int(item) for item in (kb_ids or [])],
            message_count=0,
        )
        self._session.add(conversation)
        await self._session.flush()
        return conversation

    async def load_history(self, conversation_id: int) -> list[dict[str, Any]]:
        """加载最近的历史消息（按时间正序），供多轮改写使用。"""
        rows = (
            await self._session.scalars(
                select(Message)
                .where(Message.conversation_id == int(conversation_id))
                .order_by(Message.id.desc())
                .limit(MAX_HISTORY_MESSAGES)
            )
        ).all()
        return [
            {"role": item.role, "content": item.content}
            for item in reversed(rows)
        ]

    async def add_message(
        self,
        conversation: Conversation,
        user_id: int,
        role: str,
        content: str,
        *,
        citations: Optional[list[dict[str, Any]]] = None,
        retrieved_chunks: Optional[list[dict[str, Any]]] = None,
        latency_ms: Optional[int] = None,
        blocked: bool = False,
    ) -> Message:
        """新增一条消息并更新会话统计。"""
        message = Message(
            conversation_id=conversation.id,
            user_id=int(user_id),
            role=role,
            content=content,
            citations=citations,
            retrieved_chunks=retrieved_chunks,
            latency_ms=latency_ms,
            blocked=1 if blocked else 0,
        )
        self._session.add(message)
        conversation.message_count = (conversation.message_count or 0) + 1
        conversation.last_message_at = utcnow()
        await self._session.flush()
        return message

    async def _write_qa_log(
        self,
        *,
        user_id: int,
        conversation_id: int,
        message_id: int,
        question: str,
        result: QaResult,
        latency_ms: int,
    ) -> QaLog:
        """写入全链路问答日志。"""
        trace = result.trace or {}
        log = QaLog(
            user_id=int(user_id),
            conversation_id=int(conversation_id),
            message_id=int(message_id),
            question=question,
            rewritten_query=result.rewritten_query or question,
            answer=result.answer,
            strategy=result.strategy or get_settings().default_strategy,
            guard_action=result.guard_action or "PASS",
            retrieved_chunks=[chunk.to_dict() for chunk in result.chunks],
            citations=result.citations,
            recall_count=int(trace.get("vector_count", 0)) + int(trace.get("bm25_count", 0)),
            fused_count=int(trace.get("fused_count", 0)),
            rerank_count=len(result.chunks),
            retrieve_ms=result.retrieve_ms,
            rerank_ms=result.rerank_ms,
            generate_ms=result.generate_ms,
            latency_ms=latency_ms,
        )
        self._session.add(log)
        await self._session.flush()
        return log

    # ------------------------------------------------------------------ 非流式问答
    async def answer(
        self,
        question: str,
        *,
        user_id: int,
        conversation_id: Optional[int] = None,
        kb_ids: Optional[Sequence[int]] = None,
        strategy: Optional[str] = None,
    ) -> QaAnswer:
        """非流式问答：跑完整流水线并落库。"""
        import time

        await self._guard_rate_limit(user_id)
        conversation = await self.ensure_conversation(
            user_id, conversation_id=conversation_id, kb_ids=kb_ids, first_question=question
        )
        history = await self.load_history(conversation.id)
        user_message = await self.add_message(conversation, user_id, "user", question)

        started = time.perf_counter()
        result = await self.pipeline.aanswer(
            question, history=history, kb_ids=kb_ids, strategy=strategy
        )
        latency_ms = int((time.perf_counter() - started) * 1000)

        assistant_message = await self.add_message(
            conversation,
            user_id,
            "assistant",
            result.answer,
            citations=result.citations,
            retrieved_chunks=[chunk.to_dict() for chunk in result.chunks],
            latency_ms=latency_ms,
            blocked=result.blocked,
        )
        log = await self._write_qa_log(
            user_id=user_id,
            conversation_id=conversation.id,
            message_id=assistant_message.id,
            question=question,
            result=result,
            latency_ms=latency_ms,
        )
        await self._session.commit()

        return QaAnswer(
            answer=result.answer,
            citations=result.citations,
            conversation_id=conversation.id,
            user_message_id=user_message.id,
            assistant_message_id=assistant_message.id,
            qa_log_id=log.id,
            latency_ms=latency_ms,
            blocked=result.blocked,
            guard_action=result.guard_action,
            strategy=result.strategy,
            need_human_hint=result.need_human_hint,
        )

    # ------------------------------------------------------------------ 流式问答
    async def astream(
        self,
        question: str,
        *,
        user_id: int,
        conversation_id: Optional[int] = None,
        kb_ids: Optional[Sequence[int]] = None,
        strategy: Optional[str] = None,
    ) -> AsyncGenerator[StreamEvent, None]:
        """流式问答：透传流水线事件，并在结束时落库（done 事件带上 message_id）。"""
        import time

        await self._guard_rate_limit(user_id)
        conversation = await self.ensure_conversation(
            user_id, conversation_id=conversation_id, kb_ids=kb_ids, first_question=question
        )
        history = await self.load_history(conversation.id)
        user_message = await self.add_message(conversation, user_id, "user", question)

        started = time.perf_counter()
        accumulated = ""
        citations: list[dict[str, Any]] = []
        done_payload: dict[str, Any] = {}
        try:
            async for event, data in self.pipeline.astream_answer(
                question, history=history, kb_ids=kb_ids, strategy=strategy
            ):
                if event == "delta":
                    accumulated += data.get("text", "")
                elif event == "citations":
                    citations = data.get("citations") or []
                elif event == "done":
                    done_payload = dict(data)
                    continue  # 落库后再统一发 done
                yield StreamEvent(event, data)
        except Exception as exc:  # pragma: no cover - 兜底
            logger.exception("流式问答异常：%s", exc)
            yield StreamEvent("error", {"message": f"问答失败：{exc}"})

        latency_ms = int((time.perf_counter() - started) * 1000)
        final_answer = done_payload.get("answer") or accumulated
        result = QaResult(
            answer=final_answer,
            citations=citations,
            chunks=list(getattr(self.pipeline, "last_state", {}).get("chunks") or []),
            rewritten_query=done_payload.get("rewritten_query", ""),
            blocked=bool(done_payload.get("blocked")),
            guard_action=done_payload.get("guard_action", "PASS"),
            need_human_hint=bool(done_payload.get("need_human_hint")),
            strategy=done_payload.get("trace", {}).get("strategy", ""),
            trace=dict(done_payload.get("trace") or {}),
            retrieve_ms=int(done_payload.get("retrieve_ms") or 0),
            rerank_ms=int(done_payload.get("rerank_ms") or 0),
            generate_ms=int(done_payload.get("generate_ms") or 0),
        )
        assistant_message = await self.add_message(
            conversation,
            user_id,
            "assistant",
            final_answer,
            citations=citations,
            retrieved_chunks=[chunk.to_dict() for chunk in result.chunks],
            latency_ms=latency_ms,
            blocked=result.blocked,
        )
        log = await self._write_qa_log(
            user_id=user_id,
            conversation_id=conversation.id,
            message_id=assistant_message.id,
            question=question,
            result=result,
            latency_ms=latency_ms,
        )
        await self._session.commit()

        yield StreamEvent(
            "done",
            {
                **done_payload,
                "answer": final_answer,
                "conversation_id": conversation.id,
                "message_id": assistant_message.id,
                "user_message_id": user_message.id,
                "qa_log_id": log.id,
                "latency_ms": done_payload.get("latency_ms", latency_ms),
                "server_latency_ms": latency_ms,
            },
        )

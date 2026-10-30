"""P7 验收：LangGraph 问答编排（正常流程 / 风控阻断 / 引用编号 / 多轮改写 / 流式）。"""

from __future__ import annotations

from typing import Any, AsyncGenerator, Optional, Sequence

import pytest
from langchain_core.runnables import Runnable
from langchain_core.language_models import BaseChatModel
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.chains.graph import QaPipeline, build_citations, build_context
from app.chains.prompts import NO_CONTEXT_ANSWER
from app.core.exceptions import NotFoundError
from app.guard.compliance import DISCLAIMER, BLOCK_MESSAGE, ComplianceGuard
from app.guard.default_rules import DEFAULT_RISK_RULES
from app.models import Conversation, Message, QaLog, User
from app.retrieval.types import RetrievalOutcome, RetrievalTrace, RetrievedChunk
from app.services.qa_service import QaService


class CountingLLM(Runnable):
    """假模型包装器：记录调用次数，验证被风控阻断时不会调用模型。"""

    def __init__(self, responses: Sequence[str]) -> None:
        super().__init__()
        self._model: BaseChatModel = GenericFakeChatModel(
            messages=iter([AIMessage(content=text) for text in responses])
        )
        self.calls = 0

    def invoke(self, input: Any, config: Any = None, **kwargs: Any) -> Any:  # type: ignore[override]
        """同步调用：计数后委托假模型。"""
        self.calls += 1
        return self._model.invoke(input, config=config, **kwargs)

    async def ainvoke(self, input: Any, config: Any = None, **kwargs: Any) -> Any:  # type: ignore[override]
        """异步调用：计数后委托假模型。"""
        self.calls += 1
        return await self._model.ainvoke(input, config=config, **kwargs)

    async def astream(self, input: Any, config: Any = None, **kwargs: Any) -> Any:  # type: ignore[override]
        """流式调用：计数后透传假模型的 token 流。"""
        self.calls += 1
        async for chunk in self._model.astream(input, config=config, **kwargs):
            yield chunk


def make_llm(responses: Sequence[str]) -> CountingLLM:
    """构造按顺序返回固定内容的假模型（带调用计数）。"""
    return CountingLLM(responses)


def make_chunks(count: int = 5) -> list[RetrievedChunk]:
    """构造候选切片。"""
    return [
        RetrievedChunk(
            chunk_id=100 + index,
            content=f"这是第 {index} 条理财知识内容，用于验证引用编号与上下文拼接。",
            doc_id=3,
            kb_id=2,
            source="基金投资基础.pdf",
            page_no=index,
            category="基金投资",
            score=0.9 - index * 0.05,
            rrf_score=1 / (60 + index),
            from_vector=True,
            from_bm25=index % 2 == 0,
        )
        for index in range(1, count + 1)
    ]


class FakeRetriever:
    """假检索器：返回预设候选并记录查询。"""

    def __init__(self, chunks: Optional[list[RetrievedChunk]] = None) -> None:
        self.chunks = chunks if chunks is not None else make_chunks()
        self.queries: list[str] = []
        self.strategies: list[str] = []

    async def retrieve_with_trace(
        self,
        query: str,
        kb_ids: Optional[Sequence[int]] = None,
        top_k: Optional[int] = None,
        strategy: Optional[str] = None,
    ) -> RetrievalOutcome:
        self.queries.append(query)
        self.strategies.append(strategy or "")
        trace = RetrievalTrace(
            query=query,
            strategy=strategy or "",
            kb_ids=list(kb_ids or []),
            vector_count=len(self.chunks),
            bm25_count=2,
            fused_count=len(self.chunks),
            returned_count=len(self.chunks),
            retrieve_ms=12,
        )
        return RetrievalOutcome(chunks=list(self.chunks), trace=trace)


class FakeReranker:
    """假重排器：按预设分数重排（默认按原始顺序递减）。"""

    def __init__(self, scores: Optional[list[float]] = None) -> None:
        self.scores = scores
        self.calls: list[str] = []

    async def rerank(
        self, query: str, chunks: Sequence[RetrievedChunk], top_n: Optional[int] = None
    ) -> list[RetrievedChunk]:
        self.calls.append(query)
        items = list(chunks)
        given = self.scores or [0.9 - index * 0.1 for index in range(len(items))]
        for item, score in zip(items, given, strict=False):
            item.rerank_score = score
        ordered = sorted(items, key=lambda chunk: -(chunk.rerank_score or 0.0))
        return ordered[: (top_n or len(ordered))]


def make_pipeline(
    *,
    responses: Sequence[str] = ("回答内容 [1][2]",),
    chunks: Optional[list[RetrievedChunk]] = None,
    reranker: Optional[Any] = None,
    guard: Optional[ComplianceGuard] = None,
) -> tuple[QaPipeline, CountingLLM, FakeRetriever]:
    """组装注入假依赖的流水线。"""
    llm = make_llm(responses)
    retriever = FakeRetriever(chunks)
    pipeline = QaPipeline(
        retriever=retriever,  # type: ignore[arg-type]
        reranker=reranker if reranker is not None else FakeReranker(),  # type: ignore[arg-type]
        guard=guard or ComplianceGuard(rules=DEFAULT_RISK_RULES),
        llm=llm,
    )
    return pipeline, llm, retriever


# --------------------------------------------------------------------------- 纯函数
def test_build_context_and_citations_numbering() -> None:
    """上下文编号 [n] 与 citations 数组严格一一对应。"""
    chunks = make_chunks(3)
    context = build_context(chunks)
    citations = build_citations(chunks)

    assert context.startswith("[1] 来源：基金投资基础.pdf 第 1 页")
    assert "[3]" in context
    assert [item["index"] for item in citations] == [1, 2, 3]
    assert citations[0]["chunk_id"] == chunks[0].chunk_id
    assert citations[0]["page_no"] == 1
    assert citations[2]["doc_name"] == "基金投资基础.pdf"
    assert all(item["snippet"] for item in citations)


def test_build_context_truncates_by_budget() -> None:
    """上下文超长时按顺序截断，不会撑爆 prompt。"""
    chunks = make_chunks(5)
    for chunk in chunks:
        chunk.content = "长" * 1000
    context = build_context(chunks, max_chars=2500)
    assert context.count("[") == 2  # 每条约 1024 字符，2500 预算只放得下 2 条


# --------------------------------------------------------------------------- 状态图
async def test_normal_flow_runs_all_nodes() -> None:
    """正常问题：走完全流程，返回答案 + 引用，trace 与耗时齐备。"""
    pipeline, llm, retriever = make_pipeline(
        responses=("基金定投是在固定时间投入固定金额。[1][2]\n" + DISCLAIMER,)
    )
    result = await pipeline.aanswer("什么是基金定投", kb_ids=[2], strategy="hybrid_rerank")

    assert result.blocked is False
    assert result.guard_action == "PASS"
    assert "基金定投" in result.answer
    assert len(result.citations) == 5
    assert result.context and "[1]" in result.context
    assert result.rewritten_query == "什么是基金定投"
    assert result.strategy == "hybrid_rerank"
    assert result.trace["vector_count"] == 5
    assert result.rerank_ms >= 0 and result.retrieve_ms >= 0
    assert retriever.queries == ["什么是基金定投"]
    assert retriever.strategies == ["hybrid"]  # hybrid_rerank 在检索层拆为 hybrid + 重排
    assert llm.calls == 1  # 无历史 → 不调用改写；仅生成 1 次

    # 送入模型的 prompt 必须包含资料编号与免责声明
    prompt_text = result.context
    for index in range(1, 6):
        assert f"[{index}]" in prompt_text


async def test_blocked_flow_short_circuits() -> None:
    """命中风控：直接返回合规提示，不检索、不调用模型、无引用。"""
    pipeline, llm, retriever = make_pipeline(responses=("不应该出现的内容",))
    result = await pipeline.aanswer("帮我推荐一只稳赚的基金")

    assert result.blocked is True
    assert result.guard_action == "BLOCK"
    assert result.answer == BLOCK_MESSAGE
    assert result.citations == []
    assert result.chunks == []
    assert retriever.queries == []
    assert llm.calls == 0


async def test_rewrite_guard_action_changes_query() -> None:
    """命中 REWRITE：用改写后的问法去检索，但仍正常作答。"""
    pipeline, llm, retriever = make_pipeline(responses=("评估方法如下。[1]",))
    result = await pipeline.aanswer("我该买哪只基金", kb_ids=[2])

    assert result.guard_action == "REWRITE"
    assert result.rewritten_query.startswith("如何")
    assert retriever.queries == [result.rewritten_query]
    assert result.citations
    assert llm.calls == 1


async def test_multi_turn_rewrite_uses_llm() -> None:
    """多轮场景：改写节点把指代补全后再检索。"""
    pipeline, llm, retriever = make_pipeline(
        responses=("货币基金和基金定投有什么区别", "两者都适合新手。[1]"),
    )
    history = [
        {"role": "user", "content": "什么是货币基金"},
        {"role": "assistant", "content": "货币基金是……"},
    ]
    result = await pipeline.aanswer("那它和定投比呢", history=history, kb_ids=[2])

    assert result.rewritten_query == "货币基金和基金定投有什么区别"
    assert retriever.queries == ["货币基金和基金定投有什么区别"]
    assert llm.calls == 2  # 改写 1 次 + 生成 1 次
    assert result.answer.startswith("两者都适合新手")


async def test_empty_retrieval_returns_fallback_without_llm() -> None:
    """知识库无相关内容：返回「未收录」兜底话术，不调用模型。"""
    pipeline, llm, _ = make_pipeline(chunks=[], responses=("不应被调用",))
    result = await pipeline.aanswer("关于月球理财的规则", kb_ids=[999])

    assert result.answer == NO_CONTEXT_ANSWER
    assert "知识库中未收录" in result.answer
    assert result.need_human_hint is True
    assert result.citations == []
    assert llm.calls == 0


async def test_low_rerank_scores_degrade_to_fallback() -> None:
    """重排分数全部低于阈值：判定为未收录（防幻觉）。"""
    reranker = FakeReranker(scores=[0.01, 0.02, 0.0, 0.03, 0.01])
    pipeline, llm, _ = make_pipeline(reranker=reranker, responses=("不应被调用",))
    result = await pipeline.aanswer("无关问题", kb_ids=[2])

    assert result.need_human_hint is True
    assert result.answer == NO_CONTEXT_ANSWER
    assert result.citations == []
    assert llm.calls == 0


async def test_output_guard_appends_statement() -> None:
    """输出侧兜底：模型输出含违规表述时追加风险声明。"""
    pipeline, _, _ = make_pipeline(responses=("这只基金必涨，可以保证收益。[1]",))
    result = await pipeline.aanswer("基金会不会涨", kb_ids=[2])

    assert "风险提示" in result.answer
    assert DISCLAIMER in result.answer


async def test_warn_rule_message_included() -> None:
    """WARN 规则：放行但在答案末尾追加提示。"""
    pipeline, _, _ = make_pipeline(responses=("虚拟货币波动很大。[1]",))
    result = await pipeline.aanswer("虚拟货币值得投资吗", kb_ids=[2])

    assert result.guard_action == "PASS"
    assert result.warn_message and "高风险" in result.warn_message
    assert "高风险" in result.answer


async def test_citations_match_context_order() -> None:
    """重排后的顺序决定引用编号顺序（第 1 条引用 = 上下文 [1]）。"""
    reranker = FakeReranker(scores=[0.1, 0.9, 0.5, 0.3, 0.2])
    pipeline, _, _ = make_pipeline(reranker=reranker, responses=("答案 [1] [2]",))
    result = await pipeline.aanswer("排序验证", kb_ids=[2])

    top_rerank = max(chunk.rerank_score or 0 for chunk in result.chunks)
    assert result.citations[0]["score"] == pytest.approx(top_rerank, abs=1e-6)
    assert result.citations[0]["chunk_id"] == result.chunks[0].chunk_id
    assert all(
        item["index"] == position
        for position, item in enumerate(result.citations, start=1)
    )


# --------------------------------------------------------------------------- 流式
async def test_astream_emits_citations_then_deltas_then_done() -> None:
    """流式链路：先 citations，再多个 delta，最后 done，且内容与 done.answer 一致。"""
    pipeline, _, _ = make_pipeline(responses=("基金定投可以平滑成本。[1][2]",))
    events: list[tuple[str, dict[str, Any]]] = []
    async for event, data in pipeline.astream_answer("什么是基金定投", kb_ids=[2]):
        events.append((event, data))

    names = [name for name, _ in events]
    assert names[0] == "citations"
    assert names[-1] == "done"
    assert names.count("delta") >= 1
    assert "error" not in names

    citations = events[0][1]["citations"]
    assert len(citations) == 5
    streamed = "".join(data["text"] for name, data in events if name == "delta")
    done = events[-1][1]
    assert done["answer"] == streamed
    assert done["blocked"] is False
    assert done["latency_ms"] >= 0
    assert done["guard_action"] == "PASS"


async def test_astream_blocked_path() -> None:
    """流式链路命中风控：只产出合规提示的 delta 与 done(blocked=True)。"""
    pipeline, llm, _ = make_pipeline(responses=("不应出现",))
    events = [item async for item in pipeline.astream_answer("现在买什么股票能翻倍")]

    names = [name for name, _ in events]
    assert names == ["delta", "done"]
    assert events[0][1]["text"] == BLOCK_MESSAGE
    assert events[-1][1]["blocked"] is True
    assert events[-1][1]["citations"] == []
    assert llm.calls == 0


async def test_astream_fallback_when_no_context() -> None:
    """流式链路无检索结果：直接流式返回兜底话术。"""
    pipeline, llm, _ = make_pipeline(chunks=[], responses=("不应出现",))
    events = [item async for item in pipeline.astream_answer("冷门问题")]

    assert events[0][0] == "citations"
    assert events[0][1]["citations"] == []
    assert any(name == "delta" and "知识库中未收录" in data["text"] for name, data in events)
    assert events[-1][1]["need_human_hint"] is True
    assert llm.calls == 0


# --------------------------------------------------------------------------- 服务层落库
class StubPipeline:
    """服务层测试用的极简流水线替身。"""

    def __init__(self, answer: str = "回答 [1]", blocked: bool = False) -> None:
        self.answer = answer
        self.blocked = blocked
        self.last_state: dict[str, Any] = {"chunks": make_chunks(2)}
        self.calls: list[dict[str, Any]] = []

    async def aanswer(self, question: str, **kwargs: Any) -> Any:
        from app.chains.graph import QaResult

        self.calls.append({"question": question, **kwargs})
        return QaResult(
            answer=self.answer,
            citations=[{"index": 1, "chunk_id": 101, "doc_name": "a.pdf", "page_no": 1, "snippet": "x", "score": 0.8}],
            chunks=make_chunks(1),
            rewritten_query=question,
            blocked=self.blocked,
            guard_action="BLOCK" if self.blocked else "PASS",
            strategy="hybrid_rerank",
            trace={"strategy": "hybrid", "vector_count": 1, "bm25_count": 1, "fused_count": 1},
            retrieve_ms=10,
            rerank_ms=20,
            generate_ms=30,
        )

    async def astream_answer(self, question: str, **kwargs: Any) -> AsyncGenerator[tuple[str, dict[str, Any]], None]:
        self.calls.append({"question": question, **kwargs})
        yield "citations", {"citations": [{"index": 1, "chunk_id": 101}]}
        yield "delta", {"text": "流式"}
        yield "delta", {"text": "回答"}
        yield "done", {
            "blocked": self.blocked,
            "guard_action": "BLOCK" if self.blocked else "PASS",
            "citations": [{"index": 1, "chunk_id": 101}],
            "answer": "流式回答",
            "rewritten_query": question,
            "latency_ms": 99,
            "retrieve_ms": 10,
            "rerank_ms": 20,
            "generate_ms": 30,
            "trace": {"strategy": "hybrid"},
        }


async def _seed_user(session: AsyncSession) -> User:
    """写入测试用户。"""
    from app.core.security import hash_password

    user = User(username="qa_tester", password_hash=hash_password("Student@2026"))
    session.add(user)
    await session.commit()
    return user


async def test_service_answer_persists_messages_and_log(sqlite_session: AsyncSession) -> None:
    """非流式问答：写入用户消息、助手消息与 qa_log，并更新会话统计。"""
    user = await _seed_user(sqlite_session)
    service = QaService(sqlite_session, pipeline=StubPipeline(), enable_rate_limit=False)

    answer = await service.answer("什么是基金定投", user_id=user.id, kb_ids=[2])

    assert answer.answer == "回答 [1]"
    assert answer.conversation_id > 0
    assert answer.assistant_message_id > 0 and answer.qa_log_id
    messages = (
        await sqlite_session.scalars(
            select(Message).where(Message.conversation_id == answer.conversation_id).order_by(Message.id)
        )
    ).all()
    assert [item.role for item in messages] == ["user", "assistant"]
    assert messages[1].citations[0]["chunk_id"] == 101
    assert messages[1].retrieved_chunks

    log = await sqlite_session.get(QaLog, answer.qa_log_id)
    assert log is not None
    assert log.strategy == "hybrid_rerank"
    assert log.recall_count == 2 and log.fused_count == 1
    assert log.retrieve_ms == 10 and log.rerank_ms == 20 and log.generate_ms == 30

    conversation = await sqlite_session.get(Conversation, answer.conversation_id)
    assert conversation is not None
    assert conversation.message_count == 2
    assert conversation.title == "什么是基金定投"
    assert conversation.kb_ids == [2]


async def test_service_reuses_conversation_and_loads_history(sqlite_session: AsyncSession) -> None:
    """同一会话的第二轮：历史消息被带入流水线（多轮改写依赖）。"""
    user = await _seed_user(sqlite_session)
    stub = StubPipeline()
    service = QaService(sqlite_session, pipeline=stub, enable_rate_limit=False)

    first = await service.answer("什么是货币基金", user_id=user.id, kb_ids=[2])
    await service.answer("那它和定投比呢", user_id=user.id, conversation_id=first.conversation_id)

    second_call = stub.calls[-1]
    assert second_call["history"], "第二轮必须带上历史消息"
    assert [item["role"] for item in second_call["history"]] == ["user", "assistant"]
    count = await sqlite_session.scalar(select(func.count()).select_from(Message))
    assert count == 4


async def test_service_rejects_unknown_conversation(sqlite_session: AsyncSession) -> None:
    """会话不属于当前用户时抛 404 业务异常。"""
    user = await _seed_user(sqlite_session)
    service = QaService(sqlite_session, pipeline=StubPipeline(), enable_rate_limit=False)
    with pytest.raises(NotFoundError):
        await service.answer("你好", user_id=user.id, conversation_id=99999)


async def test_service_astream_persists_and_returns_message_id(sqlite_session: AsyncSession) -> None:
    """流式问答：事件透传 + 结束时落库，done 事件携带 message_id。"""
    user = await _seed_user(sqlite_session)
    service = QaService(sqlite_session, pipeline=StubPipeline(), enable_rate_limit=False)

    events = [item async for item in service.astream("什么是基金定投", user_id=user.id, kb_ids=[2])]

    names = [item.event for item in events]
    assert names == ["citations", "delta", "delta", "done"]
    done = events[-1].data
    assert done["message_id"] > 0
    assert done["conversation_id"] > 0
    assert done["answer"] == "流式回答"
    assert done["qa_log_id"]

    stored = await sqlite_session.scalar(
        select(Message).where(Message.id == done["message_id"])
    )
    assert stored is not None and stored.role == "assistant"
    assert stored.content == "流式回答"

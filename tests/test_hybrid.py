"""P6 验收：混合检索与 RRF 融合（向量侧用 Fake，BM25 用真实索引）。"""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any, Optional, Sequence

import pytest
import pytest_asyncio
from sqlalchemy import select
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from app.models import Base, Document, DocumentChunk, KnowledgeBase
from app.retrieval.hybrid import (
    HybridRetriever,
    bump_local_version,
    clear_bm25_cache,
    tokenize,
)
from app.retrieval.reranker import RerankerService, _sigmoid
from app.retrieval.types import RetrievedChunk
from app.vectorstore.milvus_client import VectorHit

EMBED_DIM = 1024

# 5 条中文理财语料（chunk_id 依次为 1..5）
CORPUS: list[str] = [
    "存款保险条例规定，同一存款人在同一家银行的存款本息合计在五十万元以内的部分，由存款保险基金依法偿付。",
    "基金定投是指在固定时间以固定金额申购指定基金，用时间平滑买入成本，降低一次性买在高点的风险。",
    "七日年化收益率是货币基金常用的指标，表示把最近七天的收益水平折算成一年的收益率，并不是承诺收益。",
    "个人征信报告记录信用卡还款与贷款履约情况，逾期记录自结清之日起在征信系统中保留五年。",
    "校园贷、兼职刷单与杀猪盘是常见的校园诈骗形式，遇到此类情形应先核实对方资质并保留证据。",
]


class FakeEmbedder:
    """假嵌入服务：查询向量为定长常量。"""

    def __init__(self) -> None:
        self.queries: list[str] = []

    async def aencode_query(self, query: str) -> list[float]:
        self.queries.append(query)
        return [0.05] * EMBED_DIM


class FakeVectorStore:
    """假向量库：按预设顺序返回命中，并记录过滤表达式。"""

    def __init__(self, hits: Optional[list[VectorHit]] = None, *, empty: bool = False) -> None:
        self._hits = [] if empty else list(hits or [])
        self.calls: list[dict[str, Any]] = []

    async def search(
        self,
        embedding: Sequence[float],
        top_k: Optional[int] = None,
        expr: Optional[str] = None,
        output_fields: Optional[Sequence[str]] = None,
    ) -> list[VectorHit]:
        self.calls.append({"embedding_len": len(embedding), "top_k": top_k, "expr": expr})
        return self._hits[: (top_k or len(self._hits))]


class FakeReranker(RerankerService):
    """假重排器：按切片字符数打分（内容越长分越高），用于验证调用与排序。"""

    def __init__(self) -> None:
        self.calls: list[tuple[str, int]] = []

    async def rerank(
        self, query: str, chunks: Sequence[RetrievedChunk], top_n: Optional[int] = None
    ) -> list[RetrievedChunk]:  # type: ignore[override]
        self.calls.append((query, len(chunks)))
        for chunk in chunks:
            chunk.rerank_score = round(len(chunk.content) / 100.0, 6)
        ordered = sorted(chunks, key=lambda item: (-(item.rerank_score or 0.0), item.chunk_id))
        return ordered[: (top_n or len(ordered))]


@pytest_asyncio.fixture
async def corpus_env() -> Any:
    """构造 SQLite 内存库 + 5 条切片的检索环境。"""
    clear_bm25_cache()
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    factory = async_sessionmaker(bind=engine, expire_on_commit=False)
    async with factory() as session:
        kb = KnowledgeBase(name="检索测试库", category="基金投资")
        session.add(kb)
        await session.flush()
        doc = Document(kb_id=kb.id, name="test.txt", file_path="test.txt", file_type="txt", status="done")
        session.add(doc)
        await session.flush()
        for index, text in enumerate(CORPUS):
            session.add(
                DocumentChunk(
                    doc_id=doc.id,
                    kb_id=kb.id,
                    chunk_index=index,
                    content=text,
                    char_count=len(text),
                    page_no=index + 1,
                    category="基金投资",
                )
            )
        await session.commit()
        chunk_ids = [
            item.id
            for item in (await session.scalars(select(DocumentChunk).order_by(DocumentChunk.id))).all()
        ]
        kb_id = kb.id
        doc_id = doc.id
    yield SimpleNamespace(factory=factory, kb_id=kb_id, doc_id=doc_id, chunk_ids=chunk_ids)
    await engine.dispose()
    clear_bm25_cache()


def _hits_from_corpus(chunk_ids: list[int], order: list[int], scores: list[float]) -> list[VectorHit]:
    """按指定顺序构造向量命中结果（order 为 1-based 语料序号）。"""
    hits = []
    for position, corpus_index in enumerate(order):
        chunk_id = chunk_ids[corpus_index - 1]
        hits.append(
            VectorHit(
                chunk_id=chunk_id,
                score=scores[position],
                doc_id=1,
                kb_id=1,
                content=CORPUS[corpus_index - 1],
                category="基金投资",
                source="test.txt",
                page=corpus_index,
            )
        )
    return hits


# --------------------------------------------------------------------------- 纯函数
def test_rrf_fuse_expected_order() -> None:
    """RRF 排序符合公式：两路命中的文档优先，同分时按 chunk_id 升序稳定排序。"""
    vector_results = [
        RetrievedChunk(chunk_id=10, content="a", vector_rank=1, from_vector=True, score=0.9),
        RetrievedChunk(chunk_id=20, content="b", vector_rank=2, from_vector=True, score=0.8),
        RetrievedChunk(chunk_id=30, content="c", vector_rank=3, from_vector=True, score=0.7),
    ]
    bm25_results = [
        RetrievedChunk(chunk_id=30, content="c", bm25_rank=1, from_bm25=True, score=12.5),
        RetrievedChunk(chunk_id=40, content="d", bm25_rank=2, from_bm25=True, score=9.1),
    ]

    fused = HybridRetriever.rrf_fuse([vector_results, bm25_results], k=60)

    assert [item.chunk_id for item in fused] == [30, 10, 20, 40]
    # 30 同时被两路召回：1/(60+3) + 1/(60+1)
    assert fused[0].rrf_score == pytest.approx(1 / 63 + 1 / 61)
    assert fused[0].from_vector and fused[0].from_bm25
    assert fused[1].rrf_score == pytest.approx(1 / 61)
    assert fused[3].rrf_score == pytest.approx(1 / 62)


def test_rrf_fuse_merges_metadata() -> None:
    """融合后同一 chunk 的元数据应合并：保留向量侧内容与 BM25 分数。"""
    vector_results = [
        RetrievedChunk(
            chunk_id=7, content="定投内容", doc_id=3, source="a.pdf", page_no=2,
            vector_rank=1, from_vector=True, score=0.88,
        )
    ]
    bm25_results = [
        RetrievedChunk(chunk_id=7, bm25_score=11.2, bm25_rank=1, from_bm25=True, kb_id=2)
    ]
    fused = HybridRetriever.rrf_fuse([vector_results, bm25_results])
    assert len(fused) == 1
    item = fused[0]
    assert item.content == "定投内容"
    assert item.doc_id == 3 and item.kb_id == 2 and item.page_no == 2
    assert item.bm25_score == pytest.approx(11.2)
    assert item.vector_rank == 1 and item.bm25_rank == 1


def test_rrf_fuse_empty_lists() -> None:
    """空输入返回空列表，不抛异常。"""
    assert HybridRetriever.rrf_fuse([]) == []
    assert HybridRetriever.rrf_fuse([[], []]) == []


def test_tokenize_filters_stopwords_and_punctuation() -> None:
    """分词：过滤停用词与标点，保留数字与专业词。"""
    tokens = tokenize("什么是七日年化收益率？50 万元以内怎么算！")
    assert "七日年化收益率" in tokens or "七日" in tokens
    assert "万元" in tokens or "万" in tokens
    assert "什么" not in tokens
    assert "？" not in tokens


def test_sigmoid_bounds() -> None:
    """sigmoid 把 logits 映射到 (0,1)。"""
    assert _sigmoid(0) == pytest.approx(0.5)
    assert 0 < _sigmoid(-8) < 1
    assert 0 < _sigmoid(8) < 1
    assert _sigmoid(8) > 0.99


# --------------------------------------------------------------------------- 真实 BM25
async def test_bm25_strategy_hits_expected_chunk(corpus_env: Any) -> None:
    """纯 BM25：存款保险问题应命中语料 1，并带出页码等溯源信息。"""
    retriever = HybridRetriever(
        session_factory=corpus_env.factory,
        vectorstore=FakeVectorStore(empty=True),
        embedder=FakeEmbedder(),
    )
    outcome = await retriever.retrieve_with_trace("存款保险 五十万 偿付", kb_ids=[corpus_env.kb_id], strategy="bm25")

    assert outcome.chunks, "BM25 应有命中"
    top = outcome.chunks[0]
    assert top.chunk_id == corpus_env.chunk_ids[0]
    assert "存款保险" in top.content
    assert top.page_no == 1 and top.source == "test.txt"
    assert top.from_bm25 is True and top.from_vector is False
    assert top.bm25_rank == 1
    assert outcome.trace.bm25_count == len(outcome.chunks)
    assert outcome.trace.bm25_corpus_size == len(CORPUS)


async def test_bm25_returns_empty_when_no_token() -> None:
    """纯停用词查询无法分词时返回空，不抛异常。"""
    retriever = HybridRetriever(vectorstore=FakeVectorStore(empty=True), embedder=FakeEmbedder())
    outcome = await retriever.retrieve_with_trace("的了是什么呢", kb_ids=[1], strategy="bm25")
    assert outcome.chunks == []


async def test_bm25_cache_hit_and_invalidation(corpus_env: Any) -> None:
    """BM25 索引带缓存：第二次命中缓存；版本号变更后重建。"""
    retriever = HybridRetriever(
        session_factory=corpus_env.factory,
        vectorstore=FakeVectorStore(empty=True),
        embedder=FakeEmbedder(),
    )
    first = await retriever.retrieve_with_trace("基金定投 平滑 成本", kb_ids=[corpus_env.kb_id], strategy="bm25")
    second = await retriever.retrieve_with_trace("基金定投 平滑 成本", kb_ids=[corpus_env.kb_id], strategy="bm25")
    assert first.trace.bm25_cache_hit is False
    assert second.trace.bm25_cache_hit is True
    assert [item.chunk_id for item in first.chunks] == [item.chunk_id for item in second.chunks]

    bump_local_version(corpus_env.kb_id)
    third = await retriever.retrieve_with_trace("基金定投 平滑 成本", kb_ids=[corpus_env.kb_id], strategy="bm25")
    assert third.trace.bm25_cache_hit is False


async def test_bm25_empty_corpus_returns_empty(corpus_env: Any) -> None:
    """知识库无切片时返回空列表（不报错），供上层走「未收录」兜底。"""
    retriever = HybridRetriever(
        session_factory=corpus_env.factory,
        vectorstore=FakeVectorStore(empty=True),
        embedder=FakeEmbedder(),
    )
    outcome = await retriever.retrieve_with_trace("基金定投", kb_ids=[999], strategy="hybrid")
    assert outcome.chunks == []
    assert outcome.trace.bm25_corpus_size == 0


# --------------------------------------------------------------------------- 向量与混合
async def test_vector_strategy_builds_kb_filter(corpus_env: Any) -> None:
    """纯向量：按 kb_ids 构造过滤表达式，结果按向量分数降序。"""
    fake_store = FakeVectorStore(
        _hits_from_corpus(corpus_env.chunk_ids, order=[2, 3, 1], scores=[0.93, 0.85, 0.71])
    )
    retriever = HybridRetriever(
        session_factory=corpus_env.factory, vectorstore=fake_store, embedder=FakeEmbedder()
    )
    outcome = await retriever.retrieve_with_trace(
        "定投怎么做", kb_ids=[corpus_env.kb_id], top_k=3, strategy="vector"
    )

    assert fake_store.calls[0]["expr"] == f"kb_id in [{corpus_env.kb_id}]"
    assert fake_store.calls[0]["top_k"] == 3
    assert fake_store.calls[0]["embedding_len"] == EMBED_DIM
    assert [item.chunk_id for item in outcome.chunks] == [
        corpus_env.chunk_ids[1],
        corpus_env.chunk_ids[2],
        corpus_env.chunk_ids[0],
    ]
    assert all(item.strategy == "vector" for item in outcome.chunks)
    assert outcome.trace.bm25_count == 0
    assert outcome.chunks[0].vector_rank == 1


async def test_hybrid_ranking_matches_expected_cases(corpus_env: Any) -> None:
    """混合检索：向量侧命中语料 2/3，BM25 侧命中语料 4/5，融合顺序符合 RRF。"""
    fake_store = FakeVectorStore(
        _hits_from_corpus(corpus_env.chunk_ids, order=[2, 3], scores=[0.91, 0.88])
    )
    retriever = HybridRetriever(
        session_factory=corpus_env.factory, vectorstore=fake_store, embedder=FakeEmbedder()
    )
    # 查询词同时包含「基金定投」与「征信」：BM25 会同时命中语料 2 与 4
    outcome = await retriever.retrieve_with_trace(
        "基金定投 与 征信 记录", kb_ids=[corpus_env.kb_id], top_k=10, strategy="hybrid"
    )
    ids = [item.chunk_id for item in outcome.chunks]

    assert corpus_env.chunk_ids[1] in ids  # 向量第 1 名（语料 2）
    assert corpus_env.chunk_ids[3] in ids  # BM25 命中（语料 4）
    # 语料 2 被向量（rank1）与 BM25（rank1）双路命中，RRF 最高，应排第一
    assert ids[0] == corpus_env.chunk_ids[1]
    first = outcome.chunks[0]
    assert first.from_vector and first.from_bm25
    # 双路命中的 chunk 其 RRF = 1/(k+向量排名) + 1/(k+BM25 排名)
    assert first.vector_rank == 1
    assert first.bm25_rank is not None
    assert first.rrf_score == pytest.approx(1 / 61 + 1 / (60 + first.bm25_rank))
    # 单路命中的 chunk 分数必然低于双路命中的语料 2
    assert all((item.rrf_score or 0) <= (first.rrf_score or 0) for item in outcome.chunks)
    assert outcome.trace.vector_count == 2
    assert outcome.trace.fused_count >= 3


async def test_hybrid_rerank_calls_reranker_and_orders(corpus_env: Any) -> None:
    """混合 + 重排：重排器收到融合候选，返回条数受 RERANK_TOP_N 限制并按重排分排序。"""
    fake_store = FakeVectorStore(
        _hits_from_corpus(corpus_env.chunk_ids, order=[2, 3, 5], scores=[0.9, 0.8, 0.7])
    )
    fake_reranker = FakeReranker()
    retriever = HybridRetriever(
        session_factory=corpus_env.factory,
        vectorstore=fake_store,
        embedder=FakeEmbedder(),
        reranker=fake_reranker,
    )
    outcome = await retriever.retrieve_with_trace(
        "征信逾期 与 校园诈骗", kb_ids=[corpus_env.kb_id], strategy="hybrid_rerank"
    )

    assert fake_reranker.calls and fake_reranker.calls[0][0] == "征信逾期 与 校园诈骗"
    assert fake_reranker.calls[0][1] == outcome.trace.fused_count
    assert len(outcome.chunks) <= 5
    scores = [item.rerank_score for item in outcome.chunks]
    assert scores == sorted(scores, reverse=True)
    assert all(item.strategy == "hybrid_rerank" for item in outcome.chunks)


async def test_hybrid_rerank_without_reranker_degrades(corpus_env: Any) -> None:
    """未注入重排器时降级返回融合结果，并在 trace 中标注原因。"""
    fake_store = FakeVectorStore(_hits_from_corpus(corpus_env.chunk_ids, order=[1], scores=[0.9]))
    retriever = HybridRetriever(
        session_factory=corpus_env.factory, vectorstore=fake_store, embedder=FakeEmbedder()
    )
    outcome = await retriever.retrieve_with_trace(
        "存款保险", kb_ids=[corpus_env.kb_id], strategy="hybrid_rerank"
    )
    assert outcome.chunks
    assert "rerank_skipped" in outcome.trace.detail


async def test_retrieve_returns_list_and_sets_last_trace(corpus_env: Any) -> None:
    """retrieve() 返回列表（符合设计契约），同时更新 last_trace。"""
    fake_store = FakeVectorStore(_hits_from_corpus(corpus_env.chunk_ids, order=[1], scores=[0.9]))
    retriever = HybridRetriever(
        session_factory=corpus_env.factory, vectorstore=fake_store, embedder=FakeEmbedder()
    )
    chunks = await retriever.retrieve("存款保险条例", kb_ids=[corpus_env.kb_id], strategy="hybrid")
    assert isinstance(chunks, list) and chunks
    assert retriever.last_trace.strategy == "hybrid"
    assert retriever.last_trace.returned_count == len(chunks)


async def test_invalid_strategy_falls_back_to_default(corpus_env: Any) -> None:
    """非法 strategy 回落到配置默认值，避免上层传参错误导致崩溃。"""
    fake_store = FakeVectorStore(_hits_from_corpus(corpus_env.chunk_ids, order=[1], scores=[0.9]))
    retriever = HybridRetriever(
        session_factory=corpus_env.factory, vectorstore=fake_store, embedder=FakeEmbedder()
    )
    outcome = await retriever.retrieve_with_trace("存款保险", kb_ids=[corpus_env.kb_id], strategy="whatever")
    from app.core.config import get_settings

    assert outcome.trace.strategy == get_settings().default_strategy


async def test_trace_snapshot_is_serializable(corpus_env: Any) -> None:
    """trace 与切片的快照字典可 JSON 序列化（用于写 qa_log）。"""
    fake_store = FakeVectorStore(_hits_from_corpus(corpus_env.chunk_ids, order=[1], scores=[0.9]))
    retriever = HybridRetriever(
        session_factory=corpus_env.factory, vectorstore=fake_store, embedder=FakeEmbedder()
    )
    outcome = await retriever.retrieve_with_trace("存款保险", kb_ids=[corpus_env.kb_id], strategy="hybrid")
    payload = {
        "trace": outcome.trace.to_dict(),
        "chunks": [item.to_dict() for item in outcome.chunks],
    }
    text = json.dumps(payload, ensure_ascii=False)
    assert "存款保险" in text


async def test_reranker_filter_by_score() -> None:
    """重排分数阈值过滤：低于阈值的候选被剔除。"""
    chunks = [
        RetrievedChunk(chunk_id=1, content="a", rerank_score=0.62),
        RetrievedChunk(chunk_id=2, content="b", rerank_score=0.03),
        RetrievedChunk(chunk_id=3, content="c", rerank_score=None),
    ]
    kept = RerankerService.filter_by_score(chunks, min_score=0.05)
    assert [item.chunk_id for item in kept] == [1]

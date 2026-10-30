"""混合检索：向量召回 + BM25 召回 + RRF 融合（Reciprocal Rank Fusion）。

RRF 融合公式（论文 3.4 节）：

    RRF(d) = Σ_{r ∈ R} 1 / (k + rank_r(d))

其中 R 为参与融合的召回器集合（向量检索、BM25 检索），``rank_r(d)`` 为文档 d 在召回器 r 中的
排名（从 1 开始），``k`` 为平滑常数（默认 60，取自 RRF 原始论文）。RRF 只依赖排名、不依赖
分数量纲，因此无需对两路分数做归一化或调权重，工程上更稳健。

BM25 索引来源：MySQL ``document_chunk`` 表（与 Milvus 同源），进程内缓存；
知识库内容变化时通过 ``bump_bm25_version`` 提升版本号（本地 + Redis），缓存键随之失效。
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass
from typing import Any, Optional, Sequence

import jieba
from rank_bm25 import BM25Okapi
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.core.config import Settings, get_settings
from app.core.logging import get_logger
from app.ingestion.embedder import EmbedderService, get_embedder
from app.models import Document, DocumentChunk
from app.retrieval.types import RetrievalOutcome, RetrievalTrace, RetrievedChunk
from app.vectorstore.milvus_client import MilvusClientWrapper, VectorHit, build_filter

logger = get_logger(__name__)

# 中文停用词（检索场景只需过滤高频无信息量的虚词）
STOPWORDS = {
    "的", "了", "是", "在", "和", "与", "也", "都", "就", "而", "及", "或", "被", "把", "对",
    "为", "以", "于", "之", "其", "这", "那", "有", "我", "你", "他", "她", "它", "们", "吗",
    "呢", "啊", "吧", "怎么", "如何", "什么", "为什么", "多少", "哪些", "一个", "一下",
}

_TOKEN_PATTERN = re.compile(r"^[\w\u4e00-\u9fa5]+$")
MAX_CORPUS_SIZE = 20000  # 单次构建 BM25 的最大切片数，避免大库拖垮内存
VALID_STRATEGIES = ("vector", "bm25", "hybrid", "hybrid_rerank")

# 进程内的 BM25 索引版本号（Redis 不可用时的兜底失效机制）
_LOCAL_VERSION: dict[int, int] = {}
# 进程内的 BM25 索引缓存：cache_key -> (BM25Okapi, corpus)
_BM25_CACHE: dict[str, tuple[BM25Okapi, list["_CorpusChunk"]]] = {}


@dataclass
class _CorpusChunk:
    """BM25 语料条目（来自 MySQL 切片表）。"""

    chunk_id: int
    doc_id: int
    kb_id: int
    content: str
    page_no: int = 0
    category: str = ""
    risk_level: str = "low"
    source: str = ""
    tokens: list[str] = None  # type: ignore[assignment]


def tokenize(text: str) -> list[str]:
    """中文分词（jieba）+ 归一化 + 停用词过滤。"""
    if not text:
        return []
    tokens: list[str] = []
    for raw in jieba.lcut(text):
        token = raw.strip().lower()
        if not token or token in STOPWORDS:
            continue
        if not _TOKEN_PATTERN.match(token):
            continue
        if len(token) > 20:
            continue
        tokens.append(token)
    return tokens


def clear_bm25_cache() -> None:
    """清空进程内 BM25 缓存（测试与手动重建用）。"""
    _BM25_CACHE.clear()


def bump_local_version(kb_id: int) -> int:
    """提升某知识库的本地索引版本号，使缓存键失效。"""
    _LOCAL_VERSION[kb_id] = _LOCAL_VERSION.get(kb_id, 0) + 1
    return _LOCAL_VERSION[kb_id]


async def bump_bm25_version(kb_ids: int | Sequence[int]) -> None:
    """通知检索层重建 BM25 索引：本地版本 +1，并（尽力）写 Redis 供多进程失效。"""
    targets = [kb_ids] if isinstance(kb_ids, int) else list(kb_ids)
    for kb_id in targets:
        bump_local_version(int(kb_id))
    if not targets:
        return
    try:
        from app.core.redis_client import KEY_BM25_VERSION, get_redis

        client = get_redis()
        for kb_id in targets:
            await client.incr(KEY_BM25_VERSION.format(kb_id=int(kb_id)))
    except Exception as exc:  # pragma: no cover - Redis 不可用时退化为本地失效
        logger.debug("Redis 版本号提升失败（不影响正确性）：%s", exc)


class HybridRetriever:
    """混合检索器。

    :param session_factory: 异步会话工厂（读取 BM25 语料）
    :param vectorstore: 向量库包装类
    :param embedder: 嵌入服务
    :param reranker: 重排服务（strategy=hybrid_rerank 时使用，允许为空）
    """

    def __init__(
        self,
        *,
        session_factory: Optional[async_sessionmaker[AsyncSession]] = None,
        vectorstore: Optional[MilvusClientWrapper] = None,
        embedder: Optional[EmbedderService] = None,
        reranker: Optional[Any] = None,
        settings: Optional[Settings] = None,
        bm25_cache_ttl: int = 300,
    ) -> None:
        self._settings = settings or get_settings()
        self._session_factory = session_factory
        self._vectorstore = vectorstore
        self._embedder = embedder
        self._reranker = reranker
        self._bm25_cache_ttl = bm25_cache_ttl
        self._cache_time: dict[str, float] = {}
        self.last_trace = RetrievalTrace()

    # ------------------------------------------------------------------ 依赖
    def _get_session_factory(self) -> async_sessionmaker[AsyncSession]:
        """获取会话工厂（懒加载全局单例）。"""
        if self._session_factory is None:
            from app.core.database import get_session_factory

            self._session_factory = get_session_factory()
        return self._session_factory

    def _get_vectorstore(self) -> MilvusClientWrapper:
        """获取向量库包装类。"""
        if self._vectorstore is None:
            self._vectorstore = MilvusClientWrapper.get_instance()
        return self._vectorstore

    def _get_embedder(self) -> EmbedderService:
        """获取嵌入服务。"""
        if self._embedder is None:
            self._embedder = get_embedder()
        return self._embedder

    # ------------------------------------------------------------------ 对外主入口
    async def retrieve(
        self,
        query: str,
        kb_ids: Optional[Sequence[int]] = None,
        top_k: Optional[int] = None,
        strategy: Optional[str] = None,
    ) -> list[RetrievedChunk]:
        """执行检索，返回候选切片列表（按策略对应的分数降序）。

        :param query: 查询文本（已完成多轮改写）
        :param kb_ids: 限定知识库，None 表示全部
        :param top_k: 返回条数；混合策略下为融合后条数，默认 ``RECALL_TOP_K``
        :param strategy: vector / bm25 / hybrid / hybrid_rerank，默认取配置
        """
        outcome = await self.retrieve_with_trace(query, kb_ids=kb_ids, top_k=top_k, strategy=strategy)
        return outcome.chunks

    async def retrieve_with_trace(
        self,
        query: str,
        kb_ids: Optional[Sequence[int]] = None,
        top_k: Optional[int] = None,
        strategy: Optional[str] = None,
    ) -> RetrievalOutcome:
        """执行检索并返回带链路 trace 的组合结果。"""
        settings = self._settings
        chosen = (strategy or settings.default_strategy).lower()
        if chosen not in VALID_STRATEGIES:
            chosen = settings.default_strategy
        limit = int(top_k or settings.recall_top_k)
        scope = [int(item) for item in (kb_ids or [])]
        trace = RetrievalTrace(query=query, strategy=chosen, kb_ids=scope)
        started = time.perf_counter()

        vector_hits: list[RetrievedChunk] = []
        bm25_hits: list[RetrievedChunk] = []

        if chosen in ("vector", "hybrid", "hybrid_rerank"):
            vector_hits = await self._vector_search(query, scope, limit)
        if chosen in ("bm25", "hybrid", "hybrid_rerank"):
            bm25_hits, cache_hit, corpus_size = await self._bm25_search(query, scope, limit)
            trace.bm25_cache_hit = cache_hit
            trace.bm25_corpus_size = corpus_size

        trace.vector_count = len(vector_hits)
        trace.bm25_count = len(bm25_hits)

        if chosen == "vector":
            results = vector_hits[:limit]
        elif chosen == "bm25":
            results = bm25_hits[:limit]
        else:
            fused = self.rrf_fuse([vector_hits, bm25_hits], k=settings.rrf_k)
            trace.fused_count = len(fused)
            results = fused[:limit]

        for item in results:
            item.strategy = chosen

        trace.retrieve_ms = int((time.perf_counter() - started) * 1000)

        if chosen == "hybrid_rerank":
            results = await self._maybe_rerank(query, results, trace)

        trace.returned_count = len(results)
        self.last_trace = trace
        logger.info(
            "检索完成[%s]：向量 %d 条 / BM25 %d 条 → 融合 %d 条 → 返回 %d 条，耗时 %dms（重排 %dms）",
            chosen,
            trace.vector_count,
            trace.bm25_count,
            trace.fused_count,
            trace.returned_count,
            trace.retrieve_ms,
            trace.rerank_ms,
        )
        return RetrievalOutcome(chunks=results, trace=trace)

    # ------------------------------------------------------------------ 向量检索
    async def _vector_search(
        self, query: str, kb_ids: Sequence[int], top_k: int
    ) -> list[RetrievedChunk]:
        """向量召回：编码查询 → Milvus 检索 → 转 RetrievedChunk。"""
        embedding = await self._get_embedder().aencode_query(query)
        if not embedding:
            logger.warning("查询向量为空，跳过向量召回")
            return []
        expr = build_filter(kb_ids=kb_ids) if kb_ids else ""
        hits: list[VectorHit] = await self._get_vectorstore().search(
            embedding, top_k=top_k, expr=expr
        )
        chunks: list[RetrievedChunk] = []
        for rank, hit in enumerate(hits, start=1):
            chunks.append(
                RetrievedChunk(
                    chunk_id=hit.chunk_id,
                    content=hit.content,
                    doc_id=hit.doc_id,
                    kb_id=hit.kb_id,
                    source=hit.source,
                    page_no=hit.page,
                    category=hit.category,
                    risk_level=hit.risk_level,
                    score=float(hit.score),
                    vector_rank=rank,
                    from_vector=True,
                )
            )
        return chunks

    # ------------------------------------------------------------------ BM25 检索
    async def _load_corpus(self, kb_ids: Sequence[int]) -> list[_CorpusChunk]:
        """从 MySQL 载入切片语料（与 Milvus 同源，含知识库过滤）。"""
        factory = self._get_session_factory()
        async with factory() as session:
            # 关联文档表以取得溯源用的文件名（切片表本身不冗余存储文件名）
            stmt = select(DocumentChunk, Document.name).join(
                Document, Document.id == DocumentChunk.doc_id
            )
            if kb_ids:
                stmt = stmt.where(DocumentChunk.kb_id.in_(list(kb_ids)))
            stmt = stmt.limit(MAX_CORPUS_SIZE)
            rows = (await session.execute(stmt)).all()
        corpus: list[_CorpusChunk] = []
        for row, doc_name in rows:
            corpus.append(
                _CorpusChunk(
                    chunk_id=int(row.id),
                    doc_id=int(row.doc_id),
                    kb_id=int(row.kb_id),
                    content=row.content or "",
                    page_no=int(row.page_no or 0),
                    category=row.category or "",
                    risk_level=row.risk_level or "low",
                    source=doc_name or "",
                    tokens=tokenize(row.content or ""),
                )
            )
        return corpus

    async def _redis_version(self, kb_id: int) -> int:
        """读取 Redis 中的索引版本号（不可用时返回本地版本）。"""
        try:
            from app.core.redis_client import KEY_BM25_VERSION, get_redis

            value = await get_redis().get(KEY_BM25_VERSION.format(kb_id=int(kb_id)))
            return int(value) if value is not None else 0
        except Exception:  # pragma: no cover - 依赖 Redis
            return 0

    async def _cache_key(self, kb_ids: Sequence[int]) -> str:
        """缓存键 = 知识库集合 + 各库版本号（本地 + Redis）。"""
        parts: list[str] = []
        for kb_id in sorted({int(item) for item in kb_ids}):
            local = _LOCAL_VERSION.get(kb_id, 0)
            remote = await self._redis_version(kb_id)
            parts.append(f"{kb_id}:{local}.{remote}")
        return "bm25|" + ("|".join(parts) if parts else "all")

    async def _get_bm25_index(
        self, kb_ids: Sequence[int]
    ) -> tuple[Optional[BM25Okapi], list[_CorpusChunk], bool]:
        """获取（必要时构建）BM25 索引，返回 (索引, 语料, 是否命中缓存)。

        语料为空时索引返回 ``None``（BM25Okapi 无法用空语料初始化）。
        """
        key = await self._cache_key(kb_ids)
        cached = _BM25_CACHE.get(key)
        now = time.time()
        if cached is not None and now - self._cache_time.get(key, 0) <= self._bm25_cache_ttl:
            return cached[0], cached[1], True

        corpus = await self._load_corpus(kb_ids)
        if not corpus:
            _BM25_CACHE.pop(key, None)
            self._cache_time.pop(key, None)
            logger.warning("BM25 语料为空（kb_ids=%s）", list(kb_ids) or "all")
            return None, [], False

        index = BM25Okapi([item.tokens or tokenize(item.content) for item in corpus])
        _BM25_CACHE[key] = (index, corpus)
        self._cache_time[key] = now
        logger.info("已构建 BM25 索引：%d 条切片（kb_ids=%s, key=%s）", len(corpus), list(kb_ids) or "all", key)
        return index, corpus, False

    async def _bm25_search(
        self, query: str, kb_ids: Sequence[int], top_k: int
    ) -> tuple[list[RetrievedChunk], bool, int]:
        """BM25 关键词召回。"""
        tokens = tokenize(query)
        if not tokens:
            return [], False, 0
        index, corpus, cache_hit = await self._get_bm25_index(kb_ids)
        if index is None or not corpus:
            return [], cache_hit, 0

        scores = index.get_scores(tokens)
        ranked = sorted(
            zip(corpus, scores, strict=False), key=lambda pair: pair[1], reverse=True
        )[:top_k]
        chunks: list[RetrievedChunk] = []
        for rank, (item, score) in enumerate(ranked, start=1):
            if score <= 0:
                continue
            chunks.append(
                RetrievedChunk(
                    chunk_id=item.chunk_id,
                    content=item.content,
                    doc_id=item.doc_id,
                    kb_id=item.kb_id,
                    source=item.source,
                    page_no=item.page_no,
                    category=item.category,
                    risk_level=item.risk_level,
                    score=float(score),
                    bm25_score=float(score),
                    bm25_rank=rank,
                    from_bm25=True,
                )
            )
        return chunks, cache_hit, len(corpus)

    # ------------------------------------------------------------------ RRF 融合
    @staticmethod
    def rrf_fuse(
        ranked_lists: Sequence[Sequence[RetrievedChunk]], k: int = 60
    ) -> list[RetrievedChunk]:
        """RRF 融合多路召回结果（纯函数，便于单测与论文复现）。

        RRF(d) = Σ_r 1 / (k + rank_r(d))

        排序规则：rrf_score 降序；分数相同时按 chunk_id 升序，保证结果可复现。
        """
        merged: dict[int, RetrievedChunk] = {}
        for ranked in ranked_lists:
            for rank, chunk in enumerate(ranked, start=1):
                gain = 1.0 / (k + rank)
                existing = merged.get(chunk.chunk_id)
                if existing is None:
                    item = RetrievedChunk(
                        chunk_id=chunk.chunk_id,
                        content=chunk.content,
                        doc_id=chunk.doc_id,
                        kb_id=chunk.kb_id,
                        source=chunk.source,
                        page_no=chunk.page_no,
                        category=chunk.category,
                        risk_level=chunk.risk_level,
                        score=chunk.score,
                        bm25_score=chunk.bm25_score,
                        rrf_score=gain,
                        vector_rank=chunk.vector_rank,
                        bm25_rank=chunk.bm25_rank,
                        from_vector=chunk.from_vector,
                        from_bm25=chunk.from_bm25,
                    )
                    merged[chunk.chunk_id] = item
                else:
                    existing.rrf_score = (existing.rrf_score or 0.0) + gain
                    existing.merge(chunk)
        results = list(merged.values())
        results.sort(key=lambda item: (-(item.rrf_score or 0.0), item.chunk_id))
        return results

    # ------------------------------------------------------------------ 重排
    async def _maybe_rerank(
        self, query: str, chunks: list[RetrievedChunk], trace: RetrievalTrace
    ) -> list[RetrievedChunk]:
        """按配置调用重排服务；未注入重排器时降级为融合结果。"""
        if self._reranker is None:
            trace.detail["rerank_skipped"] = "reranker 未注入，返回 RRF 融合结果"
            logger.warning("strategy=hybrid_rerank 但未注入 reranker，降级返回融合结果")
            return chunks[: self._settings.rerank_top_n]
        started = time.perf_counter()
        reranked = await self._reranker.rerank(
            query, chunks, top_n=self._settings.rerank_top_n
        )
        trace.rerank_ms = int((time.perf_counter() - started) * 1000)
        return reranked

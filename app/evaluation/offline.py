"""离线评测检索器：无需 Milvus / bge-m3 也能复现 A/B/C/D 四组检索配置的对比。

实现方式（与生产链路同构，便于论文解释）：
- 向量通路：用**稳定的哈希向量**（hashing trick，1024 维、L2 归一化）代替 bge-m3 稠密向量，
  余弦相似度排序，指标含义与生产一致，只是语义精度下降；
- 关键词通路：直接复用生产的 ``tokenize`` + ``rank_bm25``；
- 融合：复用生产的 ``HybridRetriever.rrf_fuse``（同一份 RRF 实现，保证实验与线上一致）；
- 重排：用「查询词与切片的词重叠度」作为交叉编码器的代理打分。

因此：**离线结果只用于验证流水线、算法方向与相对趋势**；论文最终数据应在完整环境
（MySQL + Milvus + bge-m3 + bge-reranker + LLM）下用同一脚本去掉 ``--offline`` 复跑。
"""

from __future__ import annotations

import hashlib
import math
import time
from dataclasses import dataclass
from typing import Sequence

from rank_bm25 import BM25Okapi

from app.retrieval.hybrid import HybridRetriever, tokenize
from app.retrieval.reranker import RerankerService
from app.retrieval.types import RetrievedChunk

EMBED_DIM = 1024


def hashed_embedding(text: str, dim: int = EMBED_DIM) -> list[float]:
    """稳定的哈希向量：token → 维度下标 + 符号，累加后 L2 归一化。"""
    vector = [0.0] * dim
    tokens = tokenize(text)
    if not tokens:
        return vector
    for token in tokens:
        digest = hashlib.md5(token.encode("utf-8")).digest()
        index = int.from_bytes(digest[:4], "big") % dim
        sign = 1.0 if digest[4] % 2 == 0 else -1.0
        vector[index] += sign
    norm = math.sqrt(sum(value * value for value in vector))
    if norm == 0:
        return vector
    return [value / norm for value in vector]


def cosine_similarity(left: Sequence[float], right: Sequence[float]) -> float:
    """余弦相似度（两边均已归一化时等价于点积）。"""
    return sum(a * b for a, b in zip(left, right, strict=False))


def overlap_score(query_tokens: set[str], content: str) -> float:
    """重排代理打分：查询词在切片内容中的覆盖比例（0~1）。"""
    if not query_tokens:
        return 0.0
    tokens = set(tokenize(content))
    if not tokens:
        return 0.0
    return len(query_tokens & tokens) / len(query_tokens)


@dataclass
class OfflineCorpusItem:
    """离线语料条目（评测集中每条样本的标准答案作为一条“切片”）。"""

    chunk_id: int
    content: str
    source: str = "评测集标准答案"
    page_no: int = 1
    category: str = ""
    kb_id: int = 0


class OfflineRetriever:
    """离线检索器：支持 vector / bm25 / hybrid / hybrid_rerank 四种策略。"""

    def __init__(self, corpus: Sequence[OfflineCorpusItem], rerank_top_n: int = 5) -> None:
        """构建语料索引。"""
        self.corpus = list(corpus)
        self.rerank_top_n = rerank_top_n
        self._vectors = [hashed_embedding(item.content) for item in self.corpus]
        self._tokens = [tokenize(item.content) for item in self.corpus]
        self._bm25 = BM25Okapi(self._tokens) if self._tokens else None

    # ------------------------------------------------------------------ 单路召回
    def vector_search(self, query: str, top_k: int) -> list[RetrievedChunk]:
        """向量通路（哈希向量 + 余弦）。"""
        query_vector = hashed_embedding(query)
        scored = [
            (item, cosine_similarity(query_vector, vector))
            for item, vector in zip(self.corpus, self._vectors, strict=False)
        ]
        scored.sort(key=lambda pair: pair[1], reverse=True)
        return [
            self._to_chunk(item, score=score, rank=index, source="vector")
            for index, (item, score) in enumerate(scored[:top_k], start=1)
        ]

    def bm25_search(self, query: str, top_k: int) -> list[RetrievedChunk]:
        """BM25 通路。"""
        if self._bm25 is None:
            return []
        tokens = tokenize(query)
        if not tokens:
            return []
        scores = self._bm25.get_scores(tokens)
        ranked = sorted(
            zip(self.corpus, scores, strict=False), key=lambda pair: pair[1], reverse=True
        )[:top_k]
        return [
            self._to_chunk(item, score=float(score), rank=index, source="bm25")
            for index, (item, score) in enumerate(ranked, start=1)
            if score > 0
        ]

    @staticmethod
    def _to_chunk(
        item: OfflineCorpusItem, *, score: float, rank: int, source: str
    ) -> RetrievedChunk:
        """语料条目 → RetrievedChunk。"""
        return RetrievedChunk(
            chunk_id=item.chunk_id,
            content=item.content,
            doc_id=item.chunk_id,
            kb_id=item.kb_id,
            source=item.source,
            page_no=item.page_no,
            category=item.category,
            score=score,
            bm25_score=score if source == "bm25" else None,
            vector_rank=rank if source == "vector" else None,
            bm25_rank=rank if source == "bm25" else None,
            from_vector=source == "vector",
            from_bm25=source == "bm25",
        )

    # ------------------------------------------------------------------ 组合策略
    def retrieve(self, query: str, strategy: str, top_k: int = 10) -> list[RetrievedChunk]:
        """按策略返回候选切片（RRF 融合复用生产实现）。"""
        if strategy == "vector":
            return self.vector_search(query, top_k)
        if strategy == "bm25":
            return self.bm25_search(query, top_k)
        fused = HybridRetriever.rrf_fuse(
            [self.vector_search(query, top_k), self.bm25_search(query, top_k)], k=60
        )
        if strategy == "hybrid":
            return fused[:top_k]
        # hybrid_rerank：用词重叠度作为交叉编码器的代理
        query_tokens = set(tokenize(query))
        for chunk in fused:
            chunk.rerank_score = round(overlap_score(query_tokens, chunk.content), 6)
        ordered = sorted(
            fused, key=lambda item: (-(item.rerank_score or 0.0), -(item.rrf_score or 0.0))
        )
        return ordered[: self.rerank_top_n]

    # ------------------------------------------------------------------ 抽取式回答
    def extractive_answer(self, query: str, chunks: Sequence[RetrievedChunk], max_chars: int = 220) -> str:
        """无 LLM 时的抽取式回答：取最高分切片的要点，标注来源编号。"""
        if not chunks:
            return "知识库中未收录与这个问题直接相关的内容。"
        top = chunks[0]
        snippet = top.content.strip()
        if len(snippet) > max_chars:
            snippet = snippet[:max_chars] + "……"
        return f"根据知识库资料：{snippet}[1]"

    def measure_latency(self, query: str, strategy: str, top_k: int = 10) -> tuple[list[RetrievedChunk], int]:
        """测量一次检索耗时（毫秒）。"""
        started = time.perf_counter()
        chunks = self.retrieve(query, strategy, top_k=top_k)
        return chunks, int((time.perf_counter() - started) * 1000)


class LexicalReranker(RerankerService):
    """词重叠代理重排器（仅用于评估脚本，不用于生产）。

    当交叉编码器模型不可用时，用「查询词在切片中的覆盖比例」代替模型打分，
    让 D 组仍能走真实的重排节点（截断、排序、阈值过滤），只是打分函数是确定性的。
    """

    async def rerank(  # type: ignore[override]
        self, query: str, chunks: Sequence[RetrievedChunk], top_n: int | None = None
    ) -> list[RetrievedChunk]:
        """按词重叠度排序并截断。"""
        from app.core.config import get_settings

        keep = int(top_n or get_settings().rerank_top_n)
        query_tokens = set(tokenize(query))
        for chunk in chunks:
            chunk.rerank_score = round(overlap_score(query_tokens, chunk.content), 6)
        ordered = sorted(
            chunks,
            key=lambda item: (-(item.rerank_score or 0.0), -(item.rrf_score or 0.0), item.chunk_id),
        )
        return ordered[:keep]

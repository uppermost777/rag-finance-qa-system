"""检索层公共数据结构（放在独立模块以避免 hybrid / reranker 之间的循环导入）。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class RetrievedChunk:
    """一条检索候选切片（同时承载向量分、BM25 分、RRF 分与重排分）。"""

    chunk_id: int
    content: str = ""
    doc_id: int = 0
    kb_id: int = 0
    source: str = ""
    page_no: int = 0
    category: str = ""
    risk_level: str = "low"
    # 各路原始分数
    score: float = 0.0
    bm25_score: Optional[float] = None
    rrf_score: Optional[float] = None
    rerank_score: Optional[float] = None
    # 各路排名（从 1 开始），用于论文中的融合过程说明
    vector_rank: Optional[int] = None
    bm25_rank: Optional[int] = None
    # 检索方式标记
    from_vector: bool = False
    from_bm25: bool = False
    strategy: str = ""

    def merge(self, other: "RetrievedChunk") -> None:
        """合并同一 chunk 的另一路召回结果（保留更完整的元数据）。"""
        self.content = self.content or other.content
        self.source = self.source or other.source
        self.doc_id = self.doc_id or other.doc_id
        self.kb_id = self.kb_id or other.kb_id
        self.page_no = self.page_no or other.page_no
        self.category = self.category or other.category
        self.risk_level = self.risk_level if self.risk_level != "low" else other.risk_level
        if other.bm25_score is not None:
            self.bm25_score = other.bm25_score
        if other.score:
            self.score = max(self.score, other.score)
        self.from_vector = self.from_vector or other.from_vector
        self.from_bm25 = self.from_bm25 or other.from_bm25
        self.vector_rank = self.vector_rank or other.vector_rank
        self.bm25_rank = self.bm25_rank or other.bm25_rank

    def to_dict(self) -> dict[str, Any]:
        """用于落库的快照字典。"""
        return {
            "chunk_id": self.chunk_id,
            "doc_id": self.doc_id,
            "kb_id": self.kb_id,
            "source": self.source,
            "page_no": self.page_no,
            "category": self.category,
            "score": round(self.score, 6),
            "rrf_score": round(self.rrf_score, 6) if self.rrf_score is not None else None,
            "rerank_score": round(self.rerank_score, 6) if self.rerank_score is not None else None,
            "vector_rank": self.vector_rank,
            "bm25_rank": self.bm25_rank,
            "content": self.content[:500],
        }


@dataclass
class RetrievalTrace:
    """单次检索的链路观测信息（写入 qa_log / 日志，供论文实验统计）。"""

    query: str = ""
    strategy: str = ""
    kb_ids: list[int] = field(default_factory=list)
    vector_count: int = 0
    bm25_count: int = 0
    fused_count: int = 0
    returned_count: int = 0
    bm25_corpus_size: int = 0
    bm25_cache_hit: bool = False
    retrieve_ms: int = 0
    rerank_ms: int = 0
    detail: dict[str, Any] = field(default_factory=dict)

    @property
    def latency_ms(self) -> int:
        """检索 + 重排总耗时。"""
        return self.retrieve_ms + self.rerank_ms

    def to_dict(self) -> dict[str, Any]:
        """序列化。"""
        return {
            "query": self.query,
            "strategy": self.strategy,
            "kb_ids": list(self.kb_ids),
            "vector_count": self.vector_count,
            "bm25_count": self.bm25_count,
            "fused_count": self.fused_count,
            "returned_count": self.returned_count,
            "bm25_corpus_size": self.bm25_corpus_size,
            "bm25_cache_hit": self.bm25_cache_hit,
            "retrieve_ms": self.retrieve_ms,
            "rerank_ms": self.rerank_ms,
            **self.detail,
        }


@dataclass
class RetrievalOutcome:
    """检索结果 + trace 的组合返回。"""

    chunks: list[RetrievedChunk]
    trace: RetrievalTrace


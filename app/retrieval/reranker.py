"""重排服务：BAAI/bge-reranker-v2-m3 交叉编码器（懒加载 + 进程级单例）。

交叉编码器对 (query, chunk) 成对打分，精度显著高于双塔向量召回的粗排结果，
代价是每条候选都要过一次模型，因此只对融合后的 Top-N 候选做重排。
"""

from __future__ import annotations

import asyncio
import math
import time
from typing import Any, Optional, Sequence

from app.core.config import get_settings
from app.core.exceptions import ExternalServiceError
from app.core.logging import get_logger
from app.retrieval.types import RetrievedChunk

logger = get_logger(__name__)

_instance: Optional["RerankerService"] = None


def _sigmoid(value: float) -> float:
    """把 logits 映射到 (0,1)，便于跨模型比较与阈值过滤。"""
    if value >= 0:
        return 1.0 / (1.0 + math.exp(-value))
    exp_value = math.exp(value)
    return exp_value / (1.0 + exp_value)


class RerankerService:
    """bge-reranker-v2-m3 重排服务。"""

    def __init__(
        self,
        model_path: Optional[str] = None,
        device: Optional[str] = None,
        batch_size: Optional[int] = None,
    ) -> None:
        """初始化服务。

        :param model_path: 本地重排模型目录，默认取配置 ``RERANKER_MODEL_PATH``
        :param device: 推理设备
        :param batch_size: 批大小，默认取配置
        """
        settings = get_settings()
        self._model_path = model_path or settings.reranker_model_path
        self._device = device
        self._batch_size = batch_size or settings.reranker_batch_size
        self._model: Any = None

    def _load_model(self) -> Any:
        """懒加载 CrossEncoder 模型。"""
        if self._model is None:
            try:
                from sentence_transformers import CrossEncoder
            except ImportError as exc:  # pragma: no cover - 依赖本地模型环境
                raise ExternalServiceError(
                    "未安装 sentence-transformers，无法加载 bge-reranker-v2-m3"
                ) from exc
            logger.info("加载重排模型：%s（device=%s）", self._model_path, self._device or "auto")
            self._model = CrossEncoder(self._model_path, device=self._device)
        return self._model

    def score_pairs(self, query: str, chunks: Sequence[RetrievedChunk]) -> list[float]:
        """同步批量打分，返回 (0,1) 区间的相关性分数。"""
        if not chunks:
            return []
        model = self._load_model()
        pairs = [(query, chunk.content or "") for chunk in chunks]
        raw_scores = model.predict(pairs, batch_size=self._batch_size)
        scores: list[float] = []
        for value in raw_scores:
            number = float(value)
            scores.append(_sigmoid(number) if number < 0 or number > 1 else number)
        return scores

    async def rerank(
        self, query: str, chunks: Sequence[RetrievedChunk], top_n: Optional[int] = None
    ) -> list[RetrievedChunk]:
        """对候选切片重排并截断。

        :param query: 用户问题（改写后）
        :param chunks: 融合后的候选切片
        :param top_n: 保留条数，默认取配置 ``RERANK_TOP_N``
        :return: 按 rerank_score 降序的切片列表
        """
        if not chunks:
            return []
        settings = get_settings()
        keep = int(top_n or settings.rerank_top_n)
        started = time.perf_counter()
        scores = await asyncio.to_thread(self.score_pairs, query, list(chunks))
        for chunk, score in zip(chunks, scores, strict=False):
            chunk.rerank_score = float(score)
        ordered = sorted(
            chunks,
            key=lambda item: (-(item.rerank_score or 0.0), -(item.rrf_score or 0.0), item.chunk_id),
        )
        result = ordered[:keep]
        logger.info(
            "重排完成：%d 条候选 → 保留 %d 条，耗时 %dms（最高分 %.4f）",
            len(chunks),
            len(result),
            int((time.perf_counter() - started) * 1000),
            result[0].rerank_score if result else 0.0,
        )
        return result

    @staticmethod
    def filter_by_score(
        chunks: Sequence[RetrievedChunk], min_score: Optional[float] = None
    ) -> list[RetrievedChunk]:
        """按最低重排分数过滤（用于「知识库未收录」判定）。"""
        threshold = min_score if min_score is not None else get_settings().min_rerank_score
        return [item for item in chunks if (item.rerank_score or 0.0) >= threshold]


def get_reranker() -> RerankerService:
    """获取全局重排服务单例。"""
    global _instance
    if _instance is None:
        _instance = RerankerService()
    return _instance


def set_reranker(service: Optional[RerankerService]) -> None:
    """替换全局单例（测试与脚本注入用）。"""
    global _instance
    _instance = service


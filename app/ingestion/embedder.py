"""嵌入模型封装：BAAI/bge-m3 本地推理（懒加载 + 进程级单例）。

说明：sentence-transformers / torch 体积较大，未随 ``requirements.txt`` 强制安装；
未安装时本模块抛 ``ExternalServiceError``，由入库任务记录为 ``failed`` 原因。
"""

from __future__ import annotations

import asyncio
from typing import Any, Optional, Sequence

from app.core.config import get_settings
from app.core.exceptions import ExternalServiceError
from app.core.logging import get_logger

logger = get_logger(__name__)

_instance: Optional["EmbedderService"] = None


class EmbedderService:
    """bge-m3 嵌入服务（模型懒加载，避免重复占用显存）。"""

    def __init__(self, model_path: Optional[str] = None, device: Optional[str] = None) -> None:
        """初始化服务。

        :param model_path: 本地模型目录，默认取配置 ``EMBEDDING_MODEL_PATH``
        :param device: 推理设备（cpu / cuda），默认由 sentence-transformers 自动选择
        """
        settings = get_settings()
        self._model_path = model_path or settings.embedding_model_path
        self._dim = settings.embedding_dim
        self._batch_size = settings.embedding_batch_size
        self._device = device
        self._model: Any = None

    def _load_model(self) -> Any:
        """懒加载 sentence-transformers 模型。"""
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:  # pragma: no cover - 依赖本地模型环境
                raise ExternalServiceError(
                    "未安装 sentence-transformers，无法加载 bge-m3；请先安装本地推理依赖"
                ) from exc
            logger.info("加载嵌入模型：%s（device=%s）", self._model_path, self._device or "auto")
            self._model = SentenceTransformer(self._model_path, device=self._device)
        return self._model

    def _encode_sync(self, texts: Sequence[str]) -> list[list[float]]:
        """同步编码（由异步方法放入线程池执行）。"""
        model = self._load_model()
        vectors = model.encode(
            list(texts),
            batch_size=self._batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        return [vector.tolist() for vector in vectors]

    def encode_documents(self, texts: Sequence[str]) -> list[list[float]]:
        """同步批量编码（Celery worker 等同步上下文可直接调用）。"""
        if not texts:
            return []
        vectors = self._encode_sync(texts)
        if vectors and len(vectors[0]) != self._dim:
            raise ExternalServiceError(
                f"嵌入维度与配置不一致：模型输出 {len(vectors[0])}，配置为 {self._dim}"
            )
        return vectors

    async def aencode_documents(self, texts: Sequence[str]) -> list[list[float]]:
        """异步批量编码：用 ``asyncio.to_thread`` 包装同步推理，避免阻塞事件循环。"""
        return await asyncio.to_thread(self.encode_documents, texts)

    async def aencode_query(self, query: str) -> list[float]:
        """异步编码单条查询文本。"""
        vectors = await self.aencode_documents([query])
        return vectors[0] if vectors else []


def get_embedder() -> EmbedderService:
    """获取全局嵌入服务单例。"""
    global _instance
    if _instance is None:
        _instance = EmbedderService()
    return _instance


def set_embedder(service: Optional[EmbedderService]) -> None:
    """替换全局单例（测试与脚本注入用）。"""
    global _instance
    _instance = service


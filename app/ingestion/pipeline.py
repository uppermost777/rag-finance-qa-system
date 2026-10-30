"""文档入库主编排：解析 → 清洗 → 切片 → 向量化 → 写 Milvus → 回写 MySQL。

设计要点：
- 全流程异步，失败时把 ``document.status`` 置为 ``failed`` 并记录 ``error_msg``，**不向上抛异常**
  （保证 Celery 任务不因单个坏文档整批失败，进度可查）；
- 幂等：重复入库同一文档会先清理旧切片与旧向量，避免重复数据；
- 每阶段通过 ``progress`` 回调上报进度（Celery 任务里写 Redis）。
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Awaitable, Callable, Optional

from sqlalchemy import delete, func, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.core.config import get_settings
from app.core.exceptions import NotFoundError
from app.core.logging import get_logger
from app.ingestion.embedder import EmbedderService, get_embedder
from app.ingestion.loader import clean_pages, load_file
from app.ingestion.splitter import split_pages
from app.models import Document, DocumentChunk, KnowledgeBase
from app.vectorstore.milvus_client import ChunkRecord, MilvusClientWrapper

logger = get_logger(__name__)

# 进度回调：(status, progress, message) -> Awaitable[None]
ProgressCallback = Callable[[str, int, str], Awaitable[None]]


@dataclass
class IngestResult:
    """入库结果。"""

    doc_id: int
    status: str
    page_count: int = 0
    chunk_count: int = 0
    error_msg: Optional[str] = None
    elapsed_ms: int = 0

    @property
    def ok(self) -> bool:
        """是否成功。"""
        return self.status == "done"


async def _noop_progress(status: str, progress: int, message: str) -> None:
    """默认进度回调：仅打日志。"""
    logger.debug("入库进度 %s%% [%s] %s", progress, status, message)


async def _notify(
    callback: Optional[ProgressCallback], status: str, progress: int, message: str = ""
) -> None:
    """安全地调用进度回调（回调异常不影响主流程）。"""
    if callback is None:
        return
    try:
        await callback(status, progress, message)
    except Exception as exc:  # pragma: no cover - 依赖 Redis
        logger.warning("进度回调失败：%s", exc)


async def _set_status(
    session: AsyncSession,
    doc: Document,
    status: str,
    progress: int,
    message: str = "",
    error_msg: Optional[str] = None,
) -> None:
    """更新文档状态并提交。"""
    doc.status = status
    doc.progress = progress
    if error_msg is not None:
        doc.error_msg = error_msg[:1024]
    await session.commit()
    await _notify(_CURRENT_PROGRESS, status, progress, message)


# 由 ingest_document 在入口设置，供 _set_status 使用（避免层层透传回调）
_CURRENT_PROGRESS: Optional[ProgressCallback] = None


async def ingest_document(
    doc_id: int,
    *,
    session: Optional[AsyncSession] = None,
    session_factory: Optional[async_sessionmaker[AsyncSession]] = None,
    vectorstore: Optional[MilvusClientWrapper] = None,
    embedder: Optional[EmbedderService] = None,
    progress: Optional[ProgressCallback] = None,
    chunk_size: Optional[int] = None,
    chunk_overlap: Optional[int] = None,
) -> IngestResult:
    """执行单个文档的完整入库流程。

    :param doc_id: ``document.id``
    :param session: 已有会话（测试注入）；为空时按 ``session_factory`` 新建
    :param session_factory: 会话工厂，默认取全局 ``get_session_factory()``
    :param vectorstore: 向量库包装类，默认取全局单例
    :param embedder: 嵌入服务，默认取全局单例
    :param progress: 进度回调 (status, progress, message)
    :param chunk_size: 覆盖配置的切片长度（实验用）
    :param chunk_overlap: 覆盖配置的切片重叠（实验用）
    :return: IngestResult（失败时 status=failed，不抛异常）
    """
    global _CURRENT_PROGRESS

    settings = get_settings()
    previous_progress = _CURRENT_PROGRESS
    _CURRENT_PROGRESS = progress
    started = time.perf_counter()

    owns_session = session is None
    if owns_session:
        if session_factory is None:
            from app.core.database import get_session_factory

            session_factory = get_session_factory()
        session = session_factory()

    store = vectorstore or MilvusClientWrapper.get_instance()
    encoder = embedder or get_embedder()
    result = IngestResult(doc_id=doc_id, status="pending")

    try:
        doc = await session.get(Document, doc_id)
        if doc is None:
            raise NotFoundError(f"文档不存在：{doc_id}")
        kb = await session.get(KnowledgeBase, doc.kb_id)
        category = kb.category if kb else ""

        # 1) 解析
        await _set_status(session, doc, "parsing", 10, "解析文档")
        file_path = settings.upload_path / doc.file_path
        pages = load_file(file_path, doc.name)
        cleaned = clean_pages(pages)
        if not cleaned:
            raise ValueError("解析结果为空，可能是扫描版 PDF 或空文件")
        result.page_count = len(cleaned)

        # 2) 切片
        await _set_status(session, doc, "splitting", 35, "文本切片")
        documents = split_pages(
            cleaned,
            doc_id=doc.id,
            kb_id=doc.kb_id,
            category=category,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        if not documents:
            raise ValueError("切片结果为空")

        # 3) 向量化
        await _set_status(session, doc, "embedding", 60, "向量化")
        texts = [item.page_content for item in documents]
        vectors = await encoder.aencode_documents(texts)
        if len(vectors) != len(texts):
            raise ValueError("向量数量与切片数量不一致")

        # 4) 幂等清理旧数据（先清向量再清明细）
        await store.delete_by_doc(doc.id)
        await session.execute(delete(DocumentChunk).where(DocumentChunk.doc_id == doc.id))
        await session.commit()

        # 5) 写切片明细（拿到自增主键作为 Milvus 主键）
        chunk_rows: list[DocumentChunk] = []
        for index, item in enumerate(documents):
            chunk_rows.append(
                DocumentChunk(
                    doc_id=doc.id,
                    kb_id=doc.kb_id,
                    chunk_index=int(item.metadata.get("chunk_index", index)),
                    content=item.page_content,
                    char_count=len(item.page_content),
                    page_no=int(item.metadata.get("page_no", 0)),
                    category=category or None,
                    risk_level=str(item.metadata.get("risk_level", "low")),
                )
            )
        session.add_all(chunk_rows)
        await session.flush()

        # 6) 写 Milvus
        records = [
            ChunkRecord(
                chunk_id=row.id,
                doc_id=doc.id,
                kb_id=doc.kb_id,
                content=row.content,
                embedding=vectors[index],
                category=category,
                source=doc.name,
                page=row.page_no,
                risk_level=row.risk_level,
            )
            for index, row in enumerate(chunk_rows)
        ]
        await store.insert_chunks(records)
        for row in chunk_rows:
            row.vector_id = row.id
        await session.commit()

        # 7) 回写统计
        doc.chunk_count = len(chunk_rows)
        doc.page_count = result.page_count
        await session.flush()
        if kb is not None:
            kb.chunk_count = int(
                await session.scalar(
                    select(func.count())
                    .select_from(DocumentChunk)
                    .where(DocumentChunk.kb_id == doc.kb_id)
                )
                or 0
            )
            kb.doc_count = int(
                await session.scalar(
                    select(func.count()).select_from(Document).where(Document.kb_id == doc.kb_id)
                )
                or 0
            )
        await _set_status(session, doc, "done", 100, "入库完成")

        result.status = "done"
        result.chunk_count = len(chunk_rows)
        result.elapsed_ms = int((time.perf_counter() - started) * 1000)
        logger.info(
            "文档入库成功：doc_id=%s，%d 页 / %d 切片，耗时 %dms",
            doc.id,
            result.page_count,
            result.chunk_count,
            result.elapsed_ms,
        )
    except Exception as exc:
        logger.exception("文档入库失败：doc_id=%s", doc_id)
        result.status = "failed"
        result.error_msg = str(exc)[:1024]
        result.elapsed_ms = int((time.perf_counter() - started) * 1000)
        try:
            await session.rollback()
            failed_doc = await session.get(Document, doc_id)
            if failed_doc is not None:
                await _set_status(
                    session,
                    failed_doc,
                    "failed",
                    0,
                    "入库失败",
                    error_msg=result.error_msg,
                )
        except Exception as inner:  # pragma: no cover - 极端情况
            logger.error("回写失败状态时出错：%s", inner)
    finally:
        _CURRENT_PROGRESS = previous_progress
        if owns_session:
            await session.close()

    return result

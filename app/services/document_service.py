"""文档业务服务：上传落盘、列表、进度、删除与重建索引。

长耗时操作（解析 + 向量化）一律通过 Celery 异步任务执行，接口只负责投递任务并立即返回。
"""

from __future__ import annotations

import hashlib
import re
import uuid
from pathlib import Path
from typing import Any, Optional, Sequence

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import get_settings
from app.core.exceptions import ConflictError, NotFoundError, ParamError
from app.core.logging import get_logger
from app.models import Document, DocumentChunk, KnowledgeBase

logger = get_logger(__name__)

# 文件名白名单：中文、字母、数字、下划线、点、短横线、括号与空格
_SAFE_NAME = re.compile(r"[^\w\u4e00-\u9fa5.\-()（） ]+")


def sanitize_filename(name: str) -> str:
    """清洗上传文件名，防止路径穿越与非法字符。

    :param name: 原始文件名（可能包含路径片段）
    :return: 仅保留基础名的安全文件名
    """
    base = (name or "").replace("\\", "/").split("/")[-1]
    base = Path(base).name
    cleaned = _SAFE_NAME.sub("_", base).strip().strip(".")
    if not cleaned:
        raise ParamError("文件名不合法")
    if cleaned in {".", ".."}:
        raise ParamError("文件名不合法")
    if len(cleaned) > 120:
        stem, dot, suffix = cleaned.rpartition(".")
        cleaned = f"{stem[:100]}{dot}{suffix}" if dot else cleaned[:120]
    return cleaned


def validate_extension(filename: str) -> str:
    """校验扩展名是否在允许列表内，返回小写扩展名（不含点）。"""
    ext = Path(filename).suffix.lower().lstrip(".")
    allowed = [item.lower() for item in get_settings().allowed_extensions]
    if ext not in allowed:
        raise ParamError(f"仅支持 {'/'.join(allowed)} 格式，当前为 {ext or '未知'}")
    return ext


def compute_checksum(content: bytes) -> str:
    """计算内容 MD5，用于同库内去重与幂等。"""
    return hashlib.md5(content).hexdigest()


def dispatch_ingest(doc_id: int) -> Optional[str]:
    """投递文档入库任务。

    默认走 Celery；开发/演示环境（``INLINE_INGEST=true``）在没有 Redis/Celery 时
    直接在 API 进程内创建异步任务，保证上传后仍可完成入库。
    """
    settings = get_settings()
    if settings.inline_ingest:
        try:
            import asyncio

            from app.ingestion.pipeline import ingest_document

            loop = asyncio.get_running_loop()
            loop.create_task(ingest_document(doc_id))
            logger.info("内联入库模式：doc_id=%s 已在当前进程调度", doc_id)
            return f"inline-{doc_id}"
        except RuntimeError:
            logger.warning("当前无运行中的事件循环，无法执行内联入库")
            return None
    try:
        from app.tasks.document_tasks import ingest_task

        async_result = ingest_task.delay(doc_id)
        return str(getattr(async_result, "id", "") or "")
    except Exception as exc:  # pragma: no cover - 依赖 Celery/Redis
        logger.warning("投递入库任务失败（doc_id=%s）：%s", doc_id, exc)
        return None


def dispatch_reindex(kb_id: Optional[int] = None, doc_id: Optional[int] = None) -> Optional[str]:
    """投递重建索引任务（Celery）。"""
    try:
        from app.tasks.document_tasks import reindex_kb_task, reindex_task

        if doc_id is not None:
            async_result = reindex_task.delay(doc_id)
        else:
            async_result = reindex_kb_task.delay(int(kb_id or 0))
        return str(getattr(async_result, "id", "") or "")
    except Exception as exc:  # pragma: no cover - 依赖 Celery/Redis
        logger.warning("投递重建任务失败：%s", exc)
        return None


class DocumentService:
    """文档管理服务。"""

    def __init__(self, session: AsyncSession) -> None:
        """初始化。"""
        self._session = session
        self._settings = get_settings()

    # ------------------------------------------------------------------ 上传
    async def create_document(
        self, *, kb_id: int, filename: str, content: bytes, uploaded_by: Optional[int] = None
    ) -> Document:
        """保存上传文件并创建文档记录（状态 pending，等待异步入库）。"""
        kb = await self._session.get(KnowledgeBase, int(kb_id))
        if kb is None:
            raise NotFoundError("知识库不存在")
        if not content:
            raise ParamError("文件内容为空")
        if len(content) > self._settings.max_upload_bytes:
            raise ParamError(f"文件大小超过上限 {self._settings.max_upload_size_mb}MB")

        safe_name = sanitize_filename(filename)
        ext = validate_extension(safe_name)
        checksum = compute_checksum(content)

        duplicated = await self._session.scalar(
            select(Document).where(Document.kb_id == kb.id, Document.checksum == checksum)
        )
        if duplicated is not None:
            raise ConflictError(f"该文件已存在于本知识库（文档 #{duplicated.id}）")

        relative = f"{kb.id}/{uuid.uuid4().hex}.{ext}"
        target = self._settings.upload_path / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content)

        document = Document(
            kb_id=kb.id,
            name=safe_name,
            file_path=relative,
            file_type=ext,
            file_size=len(content),
            checksum=checksum,
            status="pending",
            progress=0,
            uploaded_by=uploaded_by,
        )
        self._session.add(document)
        await self._session.commit()
        logger.info("文档已接收：id=%s name=%s size=%s", document.id, safe_name, len(content))
        return document

    # ------------------------------------------------------------------ 查询
    async def list_documents(
        self,
        *,
        kb_id: Optional[int] = None,
        status: Optional[str] = None,
        page: int = 1,
        size: int = 20,
    ) -> tuple[int, Sequence[Document]]:
        """分页查询文档。"""
        conditions = []
        if kb_id:
            conditions.append(Document.kb_id == int(kb_id))
        if status:
            conditions.append(Document.status == status)
        total = int(
            await self._session.scalar(
                select(func.count()).select_from(Document).where(*conditions)
            )
            or 0
        )
        rows = (
            await self._session.scalars(
                select(Document)
                .where(*conditions)
                .order_by(Document.id.desc())
                .offset((page - 1) * size)
                .limit(size)
            )
        ).all()
        return total, rows

    async def get_document(self, doc_id: int) -> Document:
        """按 ID 取文档，不存在抛 404。"""
        document = await self._session.get(Document, int(doc_id))
        if document is None:
            raise NotFoundError("文档不存在")
        return document

    async def get_progress(self, doc_id: int) -> dict[str, Any]:
        """查询入库进度：优先读 Redis（实时），回退数据库。"""
        document = await self.get_document(doc_id)
        payload: dict[str, Any] = {
            "doc_id": document.id,
            "status": document.status,
            "progress": document.progress,
            "chunk_count": document.chunk_count,
            "message": None,
            "error_msg": document.error_msg,
        }
        try:
            from app.core.redis_client import KEY_DOC_PROGRESS, get_redis

            cached = await get_redis().hgetall(KEY_DOC_PROGRESS.format(doc_id=document.id))
            if cached:
                payload.update(
                    {
                        "status": cached.get("status", payload["status"]),
                        "progress": int(cached.get("progress", payload["progress"]) or 0),
                        "message": cached.get("msg"),
                    }
                )
        except Exception as exc:  # pragma: no cover - Redis 不可用时回退 DB
            logger.debug("读取 Redis 进度失败，回退数据库：%s", exc)
        return payload

    # ------------------------------------------------------------------ 删除
    async def delete_document(self, doc_id: int) -> int:
        """删除文档：先清向量，再删切片与文档记录，最后删磁盘文件。"""
        document = await self.get_document(doc_id)
        kb_id = document.kb_id
        file_path = self._settings.upload_path / document.file_path
        try:
            from app.vectorstore.milvus_client import MilvusClientWrapper

            await MilvusClientWrapper.get_instance().delete_by_doc(document.id)
        except Exception as exc:  # pragma: no cover - 依赖 Milvus
            logger.warning("删除向量失败（继续清理数据库）：%s", exc)

        await self._session.delete(document)  # 切片由 cascade 级联删除
        await self._session.flush()
        await self._refresh_kb_stats(kb_id)
        await self._session.commit()

        try:
            file_path.unlink(missing_ok=True)
        except OSError as exc:  # pragma: no cover - 文件系统异常
            logger.warning("删除文件失败：%s", exc)
        logger.info("文档已删除：id=%s", doc_id)
        return kb_id

    async def _refresh_kb_stats(self, kb_id: int) -> None:
        """重算知识库的文档数与切片数。"""
        kb = await self._session.get(KnowledgeBase, int(kb_id))
        if kb is None:
            return
        kb.doc_count = int(
            await self._session.scalar(
                select(func.count()).select_from(Document).where(Document.kb_id == kb.id)
            )
            or 0
        )
        kb.chunk_count = int(
            await self._session.scalar(
                select(func.count()).select_from(DocumentChunk).where(DocumentChunk.kb_id == kb.id)
            )
            or 0
        )

    async def mark_pending(self, doc_id: int) -> Document:
        """把文档重置为 pending（重建索引用）。"""
        document = await self.get_document(doc_id)
        document.status = "pending"
        document.progress = 0
        document.error_msg = None
        await self._session.commit()
        return document

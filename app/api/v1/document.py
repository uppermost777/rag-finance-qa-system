"""文档管理接口：上传（异步入库）、列表、进度、重建索引、删除（管理员）。"""

from __future__ import annotations

from typing import Annotated, Optional

from fastapi import APIRouter, File, Form, Query, UploadFile

from app.core.deps import AdminUser, DbSession, PageParams
from app.core.exceptions import ExternalServiceError
from app.core.logging import get_logger
from app.schemas.common import ok
from app.schemas.document import (
    DocumentItem,
    DocumentListResponse,
    ProgressResponse,
    ReindexResponse,
    UploadResponse,
)
from app.services.document_service import DocumentService, dispatch_ingest, dispatch_reindex

logger = get_logger(__name__)

router = APIRouter(prefix="/document", tags=["文档管理"])


def _to_document_item(document) -> DocumentItem:  # noqa: ANN001 - ORM 对象
    """ORM → 响应模型。"""
    return DocumentItem(
        id=document.id,
        kb_id=document.kb_id,
        name=document.name,
        file_type=document.file_type,
        file_size=document.file_size or 0,
        status=document.status,
        progress=document.progress or 0,
        page_count=document.page_count or 0,
        chunk_count=document.chunk_count or 0,
        error_msg=document.error_msg,
        created_at=document.created_at,
    )


@router.post("/upload", summary="上传文档并异步入库")
async def upload(
    admin: AdminUser,
    db: DbSession,
    file: Annotated[UploadFile, File(description="pdf/docx/txt/md/html，单文件 ≤20MB")],
    kb_id: Annotated[int, Form(description="目标知识库 ID")],
) -> dict:
    """上传文件：落盘 → 建记录 → 投递 Celery 异步入库任务 → 立即返回。

    接口本身不做解析与向量化（长耗时操作），进度通过 ``/document/{id}/progress`` 查询。
    """
    content = await file.read()
    service = DocumentService(db)
    document = await service.create_document(
        kb_id=int(kb_id),
        filename=file.filename or "",
        content=content,
        uploaded_by=admin.id,
    )
    task_id = dispatch_ingest(document.id)
    payload = UploadResponse(
        doc_id=document.id,
        task_id=task_id,
        status=document.status,
        message="文件已接收，正在后台解析入库"
        if task_id
        else "文件已接收，但任务队列不可用，请检查 Redis/Celery 后手动重建索引",
    )
    return ok(payload.model_dump())


@router.get("", summary="文档列表")
async def list_documents(
    admin: AdminUser,
    db: DbSession,
    page_params: PageParams,
    kb_id: Optional[int] = Query(default=None),
    status: Optional[str] = Query(default=None, description="pending/parsing/embedding/done/failed"),
) -> dict:
    """分页查询文档。"""
    page, size = page_params
    total, rows = await DocumentService(db).list_documents(
        kb_id=kb_id, status=status, page=page, size=size
    )
    payload = DocumentListResponse(
        total=total, page=page, size=size, items=[_to_document_item(item) for item in rows]
    )
    return ok(payload.model_dump(mode="json"))


@router.get("/{doc_id}/progress", summary="查询入库进度")
async def progress(doc_id: int, admin: AdminUser, db: DbSession) -> dict:
    """查询某个文档的入库进度（优先读 Redis 实时进度）。"""
    payload = await DocumentService(db).get_progress(doc_id)
    return ok(ProgressResponse(**payload).model_dump())


@router.post("/{doc_id}/reindex", summary="重建单个文档索引")
async def reindex(doc_id: int, admin: AdminUser, db: DbSession) -> dict:
    """重置状态并重新投递入库任务（幂等：会先清理旧向量）。"""
    await DocumentService(db).mark_pending(doc_id)
    task_id = dispatch_reindex(doc_id=doc_id)
    if not task_id:
        raise ExternalServiceError("任务队列不可用（请检查 Redis 与 Celery worker）")
    payload = ReindexResponse(doc_id=doc_id, task_id=task_id, message="重建任务已提交")
    return ok(payload.model_dump())


@router.post("/reindex", summary="整库重建索引")
async def reindex_kb(
    admin: AdminUser, db: DbSession, kb_id: int = Query(..., description="知识库 ID")
) -> dict:
    """批量重建某知识库下所有文档的索引。"""
    task_id = dispatch_reindex(kb_id=kb_id)
    if not task_id:
        raise ExternalServiceError("任务队列不可用（请检查 Redis 与 Celery worker）")
    payload = ReindexResponse(kb_id=kb_id, task_id=task_id, message="整库重建任务已提交")
    return ok(payload.model_dump())


@router.delete("/{doc_id}", summary="删除文档")
async def delete_document(doc_id: int, admin: AdminUser, db: DbSession) -> dict:
    """删除文档：同时清理 Milvus 向量、切片明细与磁盘文件。"""
    kb_id = await DocumentService(db).delete_document(doc_id)
    logger.info("管理员删除文档：doc_id=%s kb_id=%s", doc_id, kb_id)
    return ok({"doc_id": doc_id, "kb_id": kb_id})

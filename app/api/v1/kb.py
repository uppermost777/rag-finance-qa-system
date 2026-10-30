"""知识库管理接口（写操作需管理员）。"""

from __future__ import annotations

from fastapi import APIRouter, Query

from app.core.deps import AdminUser, CurrentUser, DbSession, PageParams
from app.schemas.common import ok
from app.schemas.kb import KbCreateRequest, KbItem, KbListResponse, KbUpdateRequest
from app.services.kb_service import KbService

router = APIRouter(prefix="/kb", tags=["知识库"])


def _to_kb_item(kb) -> KbItem:  # noqa: ANN001 - ORM 对象
    """ORM → 响应模型。"""
    return KbItem(
        id=kb.id,
        name=kb.name,
        category=kb.category,
        description=kb.description,
        doc_count=kb.doc_count or 0,
        chunk_count=kb.chunk_count or 0,
        status=kb.status,
        created_at=kb.created_at,
    )


@router.get("", summary="知识库列表")
async def list_kbs(
    user: CurrentUser,
    db: DbSession,
    page_params: PageParams,
    keyword: str | None = Query(default=None, description="名称模糊搜索"),
    only_enabled: bool = Query(default=False, description="仅返回启用中的知识库"),
) -> dict:
    """查询知识库列表（学生端用于多选检索范围）。"""
    page, size = page_params
    total, rows = await KbService(db).list_kbs(
        keyword=keyword, only_enabled=only_enabled, page=page, size=size
    )
    payload = KbListResponse(
        total=total, page=page, size=size, items=[_to_kb_item(item) for item in rows]
    )
    return ok(payload.model_dump(mode="json"))


@router.post("", summary="新建知识库")
async def create_kb(payload: KbCreateRequest, admin: AdminUser, db: DbSession) -> dict:
    """新建知识库（管理员）。"""
    kb = await KbService(db).create_kb(
        name=payload.name,
        category=payload.category,
        description=payload.description,
        owner_id=admin.id,
    )
    return ok(_to_kb_item(kb).model_dump(mode="json"))


@router.patch("/{kb_id}", summary="修改知识库")
async def update_kb(kb_id: int, payload: KbUpdateRequest, admin: AdminUser, db: DbSession) -> dict:
    """修改知识库名称 / 描述 / 分类 / 启用状态（管理员）。"""
    kb = await KbService(db).update_kb(
        kb_id,
        name=payload.name,
        description=payload.description,
        category=payload.category,
        status=payload.status,
    )
    return ok(_to_kb_item(kb).model_dump(mode="json"))


@router.delete("/{kb_id}", summary="删除知识库")
async def delete_kb(kb_id: int, admin: AdminUser, db: DbSession) -> dict:
    """删除知识库及其文档、切片与向量（管理员）。"""
    await KbService(db).delete_kb(kb_id)
    return ok({"id": kb_id})

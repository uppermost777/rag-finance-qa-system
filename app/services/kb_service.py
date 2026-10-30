"""知识库业务服务：CRUD 与统计维护。"""

from __future__ import annotations

from typing import Optional, Sequence

from sqlalchemy import delete, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.exceptions import ConflictError, NotFoundError
from app.core.logging import get_logger
from app.models import Document, DocumentChunk, KnowledgeBase

logger = get_logger(__name__)


class KbService:
    """知识库服务。"""

    def __init__(self, session: AsyncSession) -> None:
        """初始化。"""
        self._session = session

    async def list_kbs(
        self,
        *,
        keyword: Optional[str] = None,
        only_enabled: bool = False,
        page: int = 1,
        size: int = 20,
    ) -> tuple[int, Sequence[KnowledgeBase]]:
        """分页查询知识库。"""
        conditions = []
        if keyword:
            conditions.append(KnowledgeBase.name.like(f"%{keyword}%"))
        if only_enabled:
            conditions.append(KnowledgeBase.status == 1)
        total = int(
            await self._session.scalar(
                select(func.count()).select_from(KnowledgeBase).where(*conditions)
            )
            or 0
        )
        rows = (
            await self._session.scalars(
                select(KnowledgeBase)
                .where(*conditions)
                .order_by(KnowledgeBase.id)
                .offset((page - 1) * size)
                .limit(size)
            )
        ).all()
        return total, rows

    async def get_kb(self, kb_id: int) -> KnowledgeBase:
        """按 ID 取知识库。"""
        kb = await self._session.get(KnowledgeBase, int(kb_id))
        if kb is None:
            raise NotFoundError("知识库不存在")
        return kb

    async def create_kb(
        self, *, name: str, category: str, description: Optional[str], owner_id: Optional[int]
    ) -> KnowledgeBase:
        """新建知识库（名称唯一）。"""
        existing = await self._session.scalar(
            select(KnowledgeBase).where(KnowledgeBase.name == name)
        )
        if existing is not None:
            raise ConflictError("同名知识库已存在")
        kb = KnowledgeBase(
            name=name, category=category, description=description, owner_id=owner_id, status=1
        )
        self._session.add(kb)
        await self._session.commit()
        logger.info("知识库已创建：id=%s name=%s", kb.id, name)
        return kb

    async def update_kb(self, kb_id: int, **fields: object) -> KnowledgeBase:
        """更新知识库字段（只更新传入的非 None 值）。"""
        kb = await self.get_kb(kb_id)
        for key, value in fields.items():
            if value is not None and hasattr(kb, key):
                setattr(kb, key, value)
        await self._session.commit()
        return kb

    async def delete_kb(self, kb_id: int) -> int:
        """删除知识库：先清理该库全部向量，再级联删除切片与文档。"""
        kb = await self.get_kb(kb_id)
        doc_ids = [
            int(item)
            for item in (
                await self._session.scalars(select(Document.id).where(Document.kb_id == kb.id))
            ).all()
        ]
        try:
            from app.vectorstore.milvus_client import MilvusClientWrapper

            store = MilvusClientWrapper.get_instance()
            for doc_id in doc_ids:
                await store.delete_by_doc(doc_id)
        except Exception as exc:  # pragma: no cover - 依赖 Milvus
            logger.warning("删除知识库向量失败（继续清理数据库）：%s", exc)

        await self._session.execute(delete(DocumentChunk).where(DocumentChunk.kb_id == kb.id))
        await self._session.delete(kb)
        await self._session.commit()
        logger.info("知识库已删除：id=%s（含 %d 个文档）", kb_id, len(doc_ids))
        return kb.id

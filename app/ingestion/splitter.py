"""文本切片：基于 LangChain RecursiveCharacterTextSplitter，按页保留页码元数据。"""

from __future__ import annotations

from typing import Any, Optional, Sequence

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.core.config import get_settings
from app.core.logging import get_logger
from app.ingestion.loader import PageContent

logger = get_logger(__name__)

# 中文优先的分隔符序列：段落 → 换行 → 句号 → 分号 → 逗号 → 空格
CHINESE_SEPARATORS = ["\n\n", "\n", "。", "；", "，", " "]


def build_splitter(
    chunk_size: Optional[int] = None, chunk_overlap: Optional[int] = None
) -> RecursiveCharacterTextSplitter:
    """构造文本切分器（参数默认取配置，便于实验脚本覆盖）。"""
    settings = get_settings()
    size = chunk_size or settings.chunk_size
    overlap = chunk_overlap if chunk_overlap is not None else settings.chunk_overlap
    return RecursiveCharacterTextSplitter(
        chunk_size=size,
        chunk_overlap=overlap,
        separators=CHINESE_SEPARATORS,
        keep_separator=True,
        length_function=len,
    )


def split_pages(
    pages: Sequence[PageContent],
    *,
    doc_id: int,
    kb_id: int,
    category: str = "",
    chunk_size: Optional[int] = None,
    chunk_overlap: Optional[int] = None,
    risk_level: str = "low",
) -> list[Document]:
    """把逐页文本切分为带元数据的 LangChain ``Document`` 列表。

    metadata 字段：``doc_id / kb_id / source / page_no / chunk_index / category / risk_level``
    """
    splitter = build_splitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    documents: list[Document] = []
    chunk_index = 0
    for page in pages:
        for piece in splitter.split_text(page.text):
            content = piece.strip()
            if not content:
                continue
            metadata: dict[str, Any] = {
                "doc_id": int(doc_id),
                "kb_id": int(kb_id),
                "source": page.source,
                "page_no": int(page.page_no),
                "chunk_index": chunk_index,
                "category": category or "",
                "risk_level": risk_level,
                "char_count": len(content),
            }
            documents.append(Document(page_content=content, metadata=metadata))
            chunk_index += 1
    logger.info(
        "切片完成：doc_id=%s，%d 页 → %d 个切片（chunk_size=%d, overlap=%d）",
        doc_id,
        len(pages),
        len(documents),
        chunk_size or get_settings().chunk_size,
        chunk_overlap if chunk_overlap is not None else get_settings().chunk_overlap,
    )
    return documents

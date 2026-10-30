"""P5 验收：解析 → 清洗 → 切片 全流程（向量化以 Fake 替换，不依赖真实模型）。"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.exceptions import ParamError
from app.ingestion.loader import (
    PageContent,
    clean_pages,
    clean_text,
    collapse_blank_lines,
    detect_repeated_lines,
    load_file,
    normalize_whitespace,
    strip_header_footer,
)
from app.ingestion.pipeline import ingest_document
from app.ingestion.splitter import split_pages
from app.models import Document, DocumentChunk, KnowledgeBase

FIXTURES = Path(__file__).parent / "fixtures"
SAMPLE_FILE = FIXTURES / "finance_sample.txt"
EMBED_DIM = 1024


class FakeEmbedder:
    """假嵌入模型：返回全 0.01 的定长向量，并记录调用。"""

    def __init__(self, dim: int = EMBED_DIM) -> None:
        self.dim = dim
        self.calls: list[list[str]] = []

    async def aencode_documents(self, texts: list[str]) -> list[list[float]]:
        self.calls.append(list(texts))
        return [[0.01] * self.dim for _ in texts]


class FakeVectorStore:
    """假向量库：记录写入与删除调用。"""

    def __init__(self) -> None:
        self.inserted: list[list[Any]] = []
        self.deleted: list[int] = []

    async def delete_by_doc(self, doc_id: int) -> int:
        self.deleted.append(doc_id)
        return 0

    async def insert_chunks(self, chunks: list[Any]) -> list[int]:
        self.inserted.append(list(chunks))
        return [item.chunk_id for item in chunks]


def _pages_with_header_footer() -> list[PageContent]:
    """构造 4 页、带重复页眉页脚与多余空白的样例。"""
    pages = []
    for index in range(1, 5):
        pages.append(
            PageContent(
                text=(
                    "理财知识手册\n"
                    f"　第 {index} 节　存款保险制度\n"
                    "\n\n\n"
                    f"存款保险条例规定，同一存款人在同一家银行的存款本息合计在 50 万元以内的部分，"
                    f"由存款保险基金依法偿付，这是第 {index} 节的要点。\n"
                    "第 1 页 共 4 页（页脚）\n"
                ),
                page_no=index,
                source="储蓄存款手册.pdf",
            )
        )
    return pages


def test_load_text_file() -> None:
    """按扩展名分派：txt 解析成功且保留来源名。"""
    pages = load_file(SAMPLE_FILE)
    assert len(pages) == 1
    assert pages[0].source == "finance_sample.txt"
    assert "基金定投" in pages[0].text
    assert pages[0].char_count > 500


def test_load_file_rejects_unsupported_type(tmp_path: Path) -> None:
    """不支持的文件类型与不存在的文件都应报参数错误。"""
    bad = tmp_path / "a.xlsx"
    bad.write_text("x", encoding="utf-8")
    with pytest.raises(ParamError):
        load_file(bad)
    with pytest.raises(ParamError):
        load_file(tmp_path / "missing.txt")


def test_normalize_whitespace_and_blank_lines() -> None:
    """清洗函数：全角空格/制表符归一、行尾去空格、连续空行折叠。"""
    raw = "标题　\n正文\t内容   \n\n\n\n结尾"
    normalized = normalize_whitespace(raw)
    assert "　" not in normalized
    assert "\t" not in normalized
    assert "正文 内容" in normalized
    assert "\n\n\n\n" not in collapse_blank_lines(raw)
    assert clean_text(raw) == "标题\n正文 内容\n\n结尾"


def test_detect_and_strip_header_footer() -> None:
    """跨页重复的页眉/页脚应被识别并删除，正文完整保留。"""
    pages = _pages_with_header_footer()
    repeated = detect_repeated_lines(pages)
    assert "理财知识手册" in repeated

    cleaned = strip_header_footer(pages)
    assert all("理财知识手册" not in page.text for page in cleaned)
    # 正文中出现的「第 1 页」字样不在首尾位置，不能误删
    assert all("存款保险条例规定" in page.text for page in cleaned)


def test_clean_pages_drops_empty_pages() -> None:
    """全空白的页面在清洗后应被丢弃，且页数统计随之变化。"""
    pages = [
        PageContent(text="   \n\n\n  ", page_no=1, source="a.txt"),
        PageContent(text="有效内容\n\n\n内容二", page_no=2, source="a.txt"),
    ]
    cleaned = clean_pages(pages)
    assert len(cleaned) == 1
    assert cleaned[0].page_no == 2
    assert cleaned[0].text == "有效内容\n\n内容二"


def test_split_produces_documents_with_metadata() -> None:
    """切片：元数据齐全、页码继承、序号连续、单块长度不超过 chunk_size。"""
    pages = clean_pages([PageContent(SAMPLE_FILE.read_text(encoding="utf-8"), 1, SAMPLE_FILE.name)])
    documents = split_pages(pages, doc_id=7, kb_id=2, category="基金投资", chunk_size=150, chunk_overlap=30)

    assert len(documents) > 3
    for index, doc in enumerate(documents):
        assert doc.metadata["doc_id"] == 7
        assert doc.metadata["kb_id"] == 2
        assert doc.metadata["source"] == "finance_sample.txt"
        assert doc.metadata["page_no"] == 1
        assert doc.metadata["chunk_index"] == index
        assert doc.metadata["category"] == "基金投资"
        assert len(doc.page_content) <= 150
        assert doc.page_content.strip() == doc.page_content


async def test_split_keeps_page_number_per_page() -> None:
    """多页文档：每个切片应继承其来源页码（溯源关键）。"""
    pages = [
        PageContent("第一页内容。" * 30, 1, "multi.pdf"),
        PageContent("第二页内容。" * 30, 2, "multi.pdf"),
    ]
    documents = split_pages(pages, doc_id=1, kb_id=1, chunk_size=100, chunk_overlap=10)
    page_numbers = {doc.metadata["page_no"] for doc in documents}
    assert page_numbers == {1, 2}
    assert documents[0].metadata["page_no"] == 1
    assert documents[-1].metadata["page_no"] == 2


async def _prepare_doc(session: AsyncSession, file_path: str, kb_name: str = "基金与指数投资基础") -> Document:
    """准备知识库 + 文档记录。"""
    kb = KnowledgeBase(name=kb_name, category="基金投资")
    session.add(kb)
    await session.flush()
    doc = Document(
        kb_id=kb.id,
        name="finance_sample.txt",
        file_path=file_path,
        file_type="txt",
        file_size=SAMPLE_FILE.stat().st_size,
        status="pending",
    )
    session.add(doc)
    await session.commit()
    return doc


async def test_ingest_document_end_to_end(
    sqlite_session: AsyncSession,
) -> None:
    """完整入库：状态流转、切片落库、向量写入、KB 计数更新、进度回调。"""
    from app.core.config import get_settings

    get_settings().upload_dir = str(FIXTURES)

    doc = await _prepare_doc(sqlite_session, "finance_sample.txt")
    embedder = FakeEmbedder()
    store = FakeVectorStore()
    progress_events: list[tuple[str, int]] = []

    async def _progress(status: str, progress: int, message: str) -> None:
        progress_events.append((status, progress))

    result = await ingest_document(
        doc.id,
        session=sqlite_session,
        vectorstore=store,  # type: ignore[arg-type]
        embedder=embedder,  # type: ignore[arg-type]
        progress=_progress,
        chunk_size=200,
        chunk_overlap=40,
    )

    assert result.ok, result.error_msg
    assert result.page_count == 1
    assert result.chunk_count > 0

    refreshed = await sqlite_session.get(Document, doc.id)
    assert refreshed is not None
    assert refreshed.status == "done" and refreshed.progress == 100
    assert refreshed.chunk_count == result.chunk_count
    assert refreshed.error_msg is None

    rows = (
        await sqlite_session.scalars(
            select(DocumentChunk).where(DocumentChunk.doc_id == doc.id).order_by(DocumentChunk.id)
        )
    ).all()
    assert len(rows) == result.chunk_count
    assert all(row.vector_id == row.id for row in rows)
    assert all(row.content.strip() for row in rows)
    assert rows[0].page_no == 1

    # 向量写入参数正确
    assert len(store.inserted) == 1
    records = store.inserted[0]
    assert [record.chunk_id for record in records] == [row.id for row in rows]
    assert all(len(record.embedding) == EMBED_DIM for record in records)
    assert all(record.doc_id == doc.id and record.kb_id == doc.kb_id for record in records)
    assert records[0].source == "finance_sample.txt"

    # KB 计数更新
    kb = await sqlite_session.get(KnowledgeBase, doc.kb_id)
    assert kb is not None and kb.chunk_count == result.chunk_count and kb.doc_count == 1

    statuses = [status for status, _ in progress_events]
    assert statuses == ["parsing", "splitting", "embedding", "done"]
    assert embedder.calls and all(len(batch) == result.chunk_count for batch in embedder.calls)


async def test_ingest_document_is_idempotent(
    sqlite_session: AsyncSession,
) -> None:
    """重复入库同一文档：不产生重复切片，且会先清理旧向量。"""
    from app.core.config import get_settings

    get_settings().upload_dir = str(FIXTURES)
    doc = await _prepare_doc(sqlite_session, "finance_sample.txt")
    store = FakeVectorStore()

    first = await ingest_document(
        doc.id, session=sqlite_session, vectorstore=store, embedder=FakeEmbedder()  # type: ignore[arg-type]
    )
    second = await ingest_document(
        doc.id, session=sqlite_session, vectorstore=store, embedder=FakeEmbedder()  # type: ignore[arg-type]
    )

    assert first.ok and second.ok
    count = len(
        (await sqlite_session.scalars(select(DocumentChunk).where(DocumentChunk.doc_id == doc.id))).all()
    )
    assert count == second.chunk_count == first.chunk_count
    assert store.deleted == [doc.id, doc.id]
    assert len(store.inserted[1]) == second.chunk_count


async def test_ingest_document_records_failure(sqlite_session: AsyncSession) -> None:
    """文件缺失时：返回 failed、写入 error_msg、不抛异常。"""
    from app.core.config import get_settings

    get_settings().upload_dir = str(FIXTURES)
    doc = await _prepare_doc(sqlite_session, "not-exists.txt", kb_name="缺失文件知识库")

    result = await ingest_document(
        doc.id,
        session=sqlite_session,
        vectorstore=FakeVectorStore(),  # type: ignore[arg-type]
        embedder=FakeEmbedder(),  # type: ignore[arg-type]
    )

    assert result.status == "failed"
    assert result.error_msg
    refreshed = await sqlite_session.get(Document, doc.id)
    assert refreshed is not None
    assert refreshed.status == "failed" and refreshed.progress == 0
    assert refreshed.error_msg


async def test_ingest_document_missing_record(sqlite_session: AsyncSession) -> None:
    """文档记录不存在：同样返回 failed 而不抛异常。"""
    result = await ingest_document(
        9999,
        session=sqlite_session,
        vectorstore=FakeVectorStore(),  # type: ignore[arg-type]
        embedder=FakeEmbedder(),  # type: ignore[arg-type]
    )
    assert result.status == "failed"
    assert "9999" in (result.error_msg or "")

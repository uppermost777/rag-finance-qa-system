"""P3 验收：8 张表的 ORM 建模与 CRUD 往返（SQLite 内存库，不依赖真实 MySQL）。"""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.security import hash_password
from app.guard.default_rules import DEFAULT_RISK_RULES
from app.models import (
    Base,
    Conversation,
    Document,
    DocumentChunk,
    KnowledgeBase,
    Message,
    QaLog,
    RiskRule,
    User,
)

EXPECTED_TABLES = {
    "user",
    "conversation",
    "message",
    "knowledge_base",
    "document",
    "document_chunk",
    "qa_log",
    "risk_rule",
}


async def test_metadata_contains_eight_tables() -> None:
    """元数据必须且仅包含设计文档约定的 8 张业务表。"""
    assert EXPECTED_TABLES.issubset(set(Base.metadata.tables))


async def test_user_crud_roundtrip(sqlite_session: AsyncSession) -> None:
    """用户表写入、查询、更新、删除往返正常。"""
    user = User(
        username="lin",
        email="lin@example.com",
        password_hash=hash_password("Student@2026"),
        role="student",
        risk_level="C3",
        risk_score={"knowledge": 60, "attitude": 55},
    )
    sqlite_session.add(user)
    await sqlite_session.commit()
    assert user.id is not None
    assert isinstance(user.created_at, datetime)
    assert user.is_admin is False

    loaded = await sqlite_session.scalar(select(User).where(User.username == "lin"))
    assert loaded is not None
    assert loaded.email == "lin@example.com"
    assert loaded.risk_score["attitude"] == 55

    loaded.role = "admin"
    await sqlite_session.commit()
    assert loaded.is_admin is True

    await sqlite_session.delete(loaded)
    await sqlite_session.commit()
    assert await sqlite_session.scalar(select(func.count()).select_from(User)) == 0


async def test_conversation_and_message_roundtrip(sqlite_session: AsyncSession) -> None:
    """会话-消息一对多：外键、级联删除与 JSON 字段正常。"""
    user = User(username="zhao", password_hash=hash_password("Student@2026"))
    sqlite_session.add(user)
    await sqlite_session.flush()

    conv = Conversation(user_id=user.id, title="基金定投怎么开始", kb_ids=[2, 3])
    sqlite_session.add(conv)
    await sqlite_session.flush()

    msg = Message(
        conversation_id=conv.id,
        user_id=user.id,
        role="assistant",
        content="基金定投是指……",
        citations=[{"index": 1, "doc_name": "基金投资基础.pdf", "page_no": 3, "score": 0.82}],
        latency_ms=3210,
    )
    sqlite_session.add(msg)
    await sqlite_session.commit()

    loaded = await sqlite_session.get(Conversation, conv.id)
    assert loaded is not None and loaded.kb_ids == [2, 3]
    stored_msg = await sqlite_session.scalar(
        select(Message).where(Message.conversation_id == conv.id)
    )
    assert stored_msg is not None
    assert stored_msg.citations[0]["page_no"] == 3
    assert stored_msg.blocked == 0

    # 级联删除会话应连带删除消息
    await sqlite_session.delete(loaded)
    await sqlite_session.commit()
    assert await sqlite_session.scalar(select(func.count()).select_from(Message)) == 0


async def test_document_chunk_chain(sqlite_session: AsyncSession) -> None:
    """知识库→文档→切片三级链路，含唯一约束字段与状态字段。"""
    kb = KnowledgeBase(name="基金与指数投资基础", category="基金投资", description="基金知识")
    sqlite_session.add(kb)
    await sqlite_session.flush()

    doc = Document(
        kb_id=kb.id,
        name="基金投资基础.pdf",
        file_path="uploads/1/abc.pdf",
        file_type="pdf",
        file_size=10240,
        checksum="d41d8cd98f00b204e9800998ecf8427e",
        status="done",
        progress=100,
        page_count=12,
        chunk_count=1,
    )
    sqlite_session.add(doc)
    await sqlite_session.flush()

    chunk = DocumentChunk(
        doc_id=doc.id,
        kb_id=kb.id,
        chunk_index=0,
        content="基金定投是指在固定时间以固定金额申购指定基金。",
        char_count=24,
        page_no=3,
        category="基金投资",
    )
    sqlite_session.add(chunk)
    await sqlite_session.commit()

    # vector_id 与切片主键一致（Milvus 侧对账依据）
    chunk.vector_id = chunk.id
    await sqlite_session.commit()

    loaded = await sqlite_session.scalar(select(DocumentChunk).where(DocumentChunk.doc_id == doc.id))
    assert loaded is not None
    assert loaded.vector_id == loaded.id
    assert loaded.risk_level == "low"

    doc.status = "failed"
    doc.progress = 40
    doc.error_msg = "PDF 解析失败：文件已损坏"
    await sqlite_session.commit()
    refreshed = await sqlite_session.get(Document, doc.id)
    assert refreshed.status == "failed" and refreshed.progress == 40


async def test_qa_log_and_risk_rule(sqlite_session: AsyncSession) -> None:
    """问答日志全字段写入 + 10 条风控规则种子。"""
    log = QaLog(
        user_id=1,
        conversation_id=1,
        question="什么是七日年化收益率",
        rewritten_query="什么是七日年化收益率",
        answer="七日年化收益率是指……",
        strategy="hybrid_rerank",
        guard_action="PASS",
        recall_count=18,
        fused_count=12,
        rerank_count=5,
        retrieve_ms=120,
        rerank_ms=310,
        generate_ms=1800,
        latency_ms=2350,
    )
    sqlite_session.add(log)

    for rule in DEFAULT_RISK_RULES:
        sqlite_session.add(RiskRule(**rule))
    await sqlite_session.commit()

    assert len(DEFAULT_RISK_RULES) >= 10
    actions = set(r["action"] for r in DEFAULT_RISK_RULES)
    assert {"BLOCK", "REWRITE", "WARN"}.issubset(actions)
    count = await sqlite_session.scalar(select(func.count()).select_from(RiskRule))
    assert count == len(DEFAULT_RISK_RULES)

    stored = await sqlite_session.get(QaLog, log.id)
    assert stored is not None and stored.latency_ms == 2350

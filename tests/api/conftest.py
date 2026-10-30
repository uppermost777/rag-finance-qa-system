"""API 测试夹具：内存数据库 + 依赖覆盖 + 鉴权头 + 假流水线。

外部依赖（Milvus / LLM / Celery）全部替换，保证接口测试可离线、可重复运行。
"""

from __future__ import annotations

from typing import Any, AsyncGenerator

import pytest_asyncio
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import AsyncEngine, async_sessionmaker, create_async_engine
from sqlalchemy.pool import StaticPool

from app.core.database import get_db
from app.core.security import create_access_token, hash_password
from app.main import create_app
from app.models import Base, KnowledgeBase, User


class StubPipeline:
    """API 测试用的问答流水线替身（不触达 Milvus / LLM）。"""

    def __init__(self, answer: str = "这是基于知识库的回答 [1]", blocked: bool = False) -> None:
        self.answer = answer
        self.blocked = blocked
        self.last_state: dict[str, Any] = {"chunks": []}
        self.calls: list[dict[str, Any]] = []

    async def aanswer(self, question: str, **kwargs: Any) -> Any:
        """非流式：返回固定结果。"""
        from app.chains.graph import QaResult

        self.calls.append({"question": question, **kwargs})
        return QaResult(
            answer=self.answer,
            citations=[
                {
                    "index": 1,
                    "chunk_id": 1,
                    "doc_id": 1,
                    "kb_id": 2,
                    "doc_name": "基金投资基础.pdf",
                    "page_no": 3,
                    "snippet": "指数基金以跟踪指数为目标……",
                    "score": 0.83,
                }
            ],
            rewritten_query=question,
            blocked=self.blocked,
            guard_action="BLOCK" if self.blocked else "PASS",
            strategy="hybrid_rerank",
            trace={"strategy": "hybrid", "vector_count": 10, "bm25_count": 8, "fused_count": 12},
            retrieve_ms=15,
            rerank_ms=25,
            generate_ms=35,
        )

    async def astream_answer(
        self, question: str, **kwargs: Any
    ) -> AsyncGenerator[tuple[str, dict[str, Any]], None]:
        """流式：先 citations，再 delta，最后 done。"""
        self.calls.append({"question": question, **kwargs})
        citations = [
            {
                "index": 1,
                "chunk_id": 1,
                "doc_id": 1,
                "kb_id": 2,
                "doc_name": "基金投资基础.pdf",
                "page_no": 3,
                "snippet": "指数基金以跟踪指数为目标……",
                "score": 0.83,
            }
        ]
        yield "citations", {"citations": citations}
        yield "delta", {"text": "指数基金"}
        yield "delta", {"text": "以跟踪指数为目标。[1]"}
        yield "done", {
            "blocked": False,
            "guard_action": "PASS",
            "citations": citations,
            "answer": "指数基金以跟踪指数为目标。[1]",
            "rewritten_query": question,
            "latency_ms": 42,
            "retrieve_ms": 15,
            "rerank_ms": 25,
            "generate_ms": 35,
            "trace": {"strategy": "hybrid"},
        }


@pytest_asyncio.fixture
async def api_engine() -> AsyncGenerator[AsyncEngine, None]:
    """内存 SQLite 引擎（StaticPool 保证多连接共享同一份数据）。"""
    engine = create_async_engine(
        "sqlite+aiosqlite:///:memory:", poolclass=StaticPool, connect_args={"check_same_thread": False}
    )
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield engine
    await engine.dispose()


@pytest_asyncio.fixture
async def api_session_factory(api_engine: AsyncEngine) -> async_sessionmaker:
    """会话工厂。"""
    return async_sessionmaker(bind=api_engine, expire_on_commit=False, autoflush=False)


@pytest_asyncio.fixture
async def seed(api_session_factory: async_sessionmaker) -> dict[str, int]:
    """写入管理员 / 学生用户与一个知识库，返回其主键。"""
    async with api_session_factory() as session:
        admin = User(
            username="admin_api",
            email="admin_api@example.com",
            password_hash=hash_password("Admin@2026"),
            role="admin",
        )
        student = User(
            username="student_api",
            email="student_api@example.com",
            password_hash=hash_password("Student@2026"),
            role="student",
        )
        other = User(username="other_api", password_hash=hash_password("Student@2026"))
        session.add_all([admin, student, other])
        await session.flush()
        kb = KnowledgeBase(name="API 测试知识库", category="基金投资", description="用于接口测试")
        session.add(kb)
        await session.commit()
        return {
            "admin_id": admin.id,
            "student_id": student.id,
            "other_id": other.id,
            "kb_id": kb.id,
            "admin_name": admin.username,
            "student_name": student.username,
            "admin_password": "Admin@2026",
            "student_password": "Student@2026",
        }


@pytest_asyncio.fixture
async def client(api_session_factory: async_sessionmaker, seed: dict[str, int]) -> AsyncGenerator[AsyncClient, None]:
    """覆盖 get_db 依赖并注入假流水线的测试客户端。"""
    from app.chains.graph import set_pipeline

    app = create_app()

    async def _override_get_db() -> AsyncGenerator[Any, None]:
        async with api_session_factory() as session:
            yield session

    app.dependency_overrides[get_db] = _override_get_db
    set_pipeline(StubPipeline())
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://testserver") as ac:
        yield ac
    set_pipeline(None)


def auth_headers(user_id: int, role: str = "student") -> dict[str, str]:
    """构造带 JWT 的请求头。"""
    token, _ = create_access_token(user_id, role=role)
    return {"Authorization": f"Bearer {token}"}

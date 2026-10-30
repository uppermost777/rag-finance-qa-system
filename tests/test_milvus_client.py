"""P4 验收：Milvus 封装层（用 Fake 客户端替换 pymilvus，校验调用参数与异常包装）。"""

from __future__ import annotations

from typing import Any, Optional

import pytest

from app.core.exceptions import ExternalServiceError, ParamError
from app.vectorstore.milvus_client import (
    INDEX_PARAMS,
    ChunkRecord,
    MilvusClientWrapper,
    build_filter,
)


class _FakeSchema:
    """记录 add_field 调用的假 Schema。"""

    def __init__(self, **kwargs: Any) -> None:
        self.options = kwargs
        self.fields: list[dict[str, Any]] = []

    def add_field(self, **kwargs: Any) -> None:
        self.fields.append(kwargs)


class _FakeIndexParams:
    """记录 add_index 调用的假索引参数。"""

    def __init__(self) -> None:
        self.indexes: list[dict[str, Any]] = []

    def add_index(self, **kwargs: Any) -> None:
        self.indexes.append(kwargs)


class FakeMilvusClient:
    """最小可用的 pymilvus 替身，记录所有调用。"""

    def __init__(
        self,
        *,
        exists: bool = False,
        insert_ids: Optional[list[int]] = None,
        search_rows: Optional[list[dict[str, Any]]] = None,
        delete_count: int = 0,
        query_rows: Optional[list[dict[str, Any]]] = None,
        raise_on: Optional[str] = None,
    ) -> None:
        self.exists = exists
        self.insert_ids = insert_ids or []
        self.search_rows = search_rows or []
        self.delete_count = delete_count
        self.query_rows = query_rows or []
        self.raise_on = raise_on
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self.schema = _FakeSchema()
        self.index_params = _FakeIndexParams()

    def _record(self, name: str, **kwargs: Any) -> None:
        self.calls.append((name, kwargs))
        if self.raise_on == name:
            raise RuntimeError(f"模拟 {name} 故障")

    # ---- SDK 接口 ----
    def create_schema(self, **kwargs: Any) -> _FakeSchema:
        self._record("create_schema", **kwargs)
        self.schema.options = kwargs
        return self.schema

    def prepare_index_params(self, field_name: str = "", **kwargs: Any) -> _FakeIndexParams:
        self._record("prepare_index_params", field_name=field_name, **kwargs)
        return self.index_params

    def has_collection(self, collection_name: str, **kwargs: Any) -> bool:
        self._record("has_collection", collection_name=collection_name, **kwargs)
        return self.exists

    def load_collection(self, collection_name: str, **kwargs: Any) -> None:
        self._record("load_collection", collection_name=collection_name, **kwargs)

    def create_collection(self, **kwargs: Any) -> dict[str, Any]:
        self._record("create_collection", **kwargs)
        return {}

    def insert(self, **kwargs: Any) -> dict[str, Any]:
        self._record("insert", **kwargs)
        return {"ids": self.insert_ids}

    def search(self, **kwargs: Any) -> list[list[dict[str, Any]]]:
        self._record("search", **kwargs)
        return [self.search_rows]

    def delete(self, **kwargs: Any) -> dict[str, int]:
        self._record("delete", **kwargs)
        return {"delete_count": self.delete_count}

    def query(self, **kwargs: Any) -> list[dict[str, Any]]:
        self._record("query", **kwargs)
        return self.query_rows

    def list_collections(self, **kwargs: Any) -> list[str]:
        self._record("list_collections", **kwargs)
        return ["finance_chunks"]

    def get_collection_stats(self, collection_name: str, **kwargs: Any) -> dict[str, Any]:
        self._record("get_collection_stats", collection_name=collection_name, **kwargs)
        return {"row_count": 42}

    def call_kwargs(self, name: str) -> dict[str, Any]:
        """取出某方法最后一次调用的关键字参数。"""
        matched = [kwargs for method, kwargs in self.calls if method == name]
        assert matched, f"未调用 {name}"
        return matched[-1]


def _vec(value: float = 0.1, dim: int = 1024) -> list[float]:
    """构造定长向量。"""
    return [value] * dim


async def test_ensure_collection_creates_with_designed_params() -> None:
    """Collection 不存在时按设计创建：9 个字段、HNSW(M=16,ef=200)、COSINE、1024 维。"""
    fake = FakeMilvusClient(exists=False)
    wrapper = MilvusClientWrapper(client=fake)  # type: ignore[arg-type]

    created = await wrapper.ensure_collection()
    assert created is True

    fields = {item["field_name"]: item for item in fake.schema.fields}
    assert set(fields) == {
        "chunk_id",
        "doc_id",
        "kb_id",
        "content",
        "category",
        "source",
        "page",
        "risk_level",
        "embedding",
    }
    assert fields["chunk_id"]["is_primary"] is True
    assert fields["chunk_id"]["auto_id"] is False
    assert fields["embedding"]["dim"] == 1024
    assert fields["content"]["max_length"] == 8192
    assert fake.schema.options["auto_id"] is False

    index = fake.index_params.indexes[0]
    assert index["field_name"] == "embedding"
    assert index["index_type"] == "HNSW"
    assert index["metric_type"] == "COSINE"
    assert index["params"] == INDEX_PARAMS == {"M": 16, "efConstruction": 200}

    create_kwargs = fake.call_kwargs("create_collection")
    assert create_kwargs["collection_name"] == "finance_chunks"
    assert create_kwargs["schema"] is fake.schema
    assert create_kwargs["index_params"] is fake.index_params


async def test_ensure_collection_is_idempotent() -> None:
    """Collection 已存在时不重复创建；且第二次调用不再访问 Milvus。"""
    fake = FakeMilvusClient(exists=True)
    wrapper = MilvusClientWrapper(client=fake)  # type: ignore[arg-type]

    assert await wrapper.ensure_collection() is False
    assert [name for name, _ in fake.calls].count("create_collection") == 0
    await wrapper.ensure_collection()
    assert [name for name, _ in fake.calls].count("has_collection") == 1


async def test_ensure_collection_loads_existing_collection() -> None:
    """已存在的集合需要 load 才能检索（进程重启后 collection 可能处于 released 状态）。"""
    fake = FakeMilvusClient(exists=True)
    wrapper = MilvusClientWrapper(client=fake)  # type: ignore[arg-type]
    await wrapper.ensure_collection()
    assert "load_collection" in [name for name, _ in fake.calls]


async def test_insert_chunks_payload_and_ids() -> None:
    """批量写入：字段映射正确、维度校验通过、返回主键列表。"""
    fake = FakeMilvusClient(exists=True, insert_ids=[101, 102])
    wrapper = MilvusClientWrapper(client=fake)  # type: ignore[arg-type]
    chunks = [
        ChunkRecord(
            chunk_id=101,
            doc_id=7,
            kb_id=2,
            content="基金定投是指在固定时间以固定金额申购指定基金。",
            embedding=_vec(),
            category="基金投资",
            source="基金投资基础.pdf",
            page=3,
        ),
        ChunkRecord(chunk_id=102, doc_id=7, kb_id=2, content="指数基金……", embedding=_vec(0.2)),
    ]

    ids = await wrapper.insert_chunks(chunks)
    assert ids == [101, 102]

    payload = fake.call_kwargs("insert")["data"]
    assert len(payload) == 2
    assert payload[0]["chunk_id"] == 101
    assert payload[0]["doc_id"] == 7
    assert payload[0]["kb_id"] == 2
    assert payload[0]["page"] == 3
    assert payload[0]["risk_level"] == "low"
    assert len(payload[0]["embedding"]) == 1024


async def test_insert_chunks_rejects_wrong_dimension() -> None:
    """向量维度不等于 1024 时必须报错（防止脏数据写入）。"""
    fake = FakeMilvusClient(exists=True)
    wrapper = MilvusClientWrapper(client=fake)  # type: ignore[arg-type]
    with pytest.raises(ParamError):
        await wrapper.insert_chunks(
            [
                ChunkRecord(
                    chunk_id=1, doc_id=1, kb_id=1, content="x", embedding=[0.1, 0.2, 0.3]
                )
            ]
        )


async def test_insert_chunks_empty_returns_empty() -> None:
    """空列表直接返回，不触发任何 Milvus 调用。"""
    fake = FakeMilvusClient(exists=True)
    wrapper = MilvusClientWrapper(client=fake)  # type: ignore[arg-type]
    assert await wrapper.insert_chunks([]) == []
    assert fake.calls == []


async def test_search_passes_filter_and_parses_hits() -> None:
    """检索：filter/limit/search_params 正确，返回结果按相似度降序。"""
    fake = FakeMilvusClient(
        exists=True,
        search_rows=[
            {
                "id": 11,
                "distance": 0.72,
                "entity": {
                    "doc_id": 3,
                    "kb_id": 2,
                    "content": "指数基金以跟踪指数为目标……",
                    "category": "基金投资",
                    "source": "指数投资入门.pdf",
                    "page": 5,
                    "risk_level": "low",
                },
            },
            {
                "id": 12,
                "distance": 0.91,
                "entity": {
                    "doc_id": 3,
                    "kb_id": 2,
                    "content": "定投可以平滑成本……",
                    "category": "基金投资",
                    "source": "指数投资入门.pdf",
                    "page": 9,
                },
            },
        ],
    )
    wrapper = MilvusClientWrapper(client=fake)  # type: ignore[arg-type]
    expr = build_filter(kb_ids=[2, 3], category="基金投资")

    hits = await wrapper.search(_vec(), top_k=10, expr=expr)

    kwargs = fake.call_kwargs("search")
    assert kwargs["collection_name"] == "finance_chunks"
    assert kwargs["data"] == [_vec()]
    assert kwargs["limit"] == 10
    assert kwargs["filter"] == 'kb_id in [2, 3] and category == "基金投资"'
    assert kwargs["search_params"] == {"metric_type": "COSINE", "params": {"ef": 64}}
    assert "content" in kwargs["output_fields"]

    assert [hit.chunk_id for hit in hits] == [12, 11]
    assert hits[0].score == pytest.approx(0.91)
    assert hits[0].page == 9
    assert hits[0].risk_level == "low"  # 缺省值兜底


async def test_search_wraps_sdk_error() -> None:
    """pymilvus 抛异常时统一包装为 ExternalServiceError（不静默吞掉）。"""
    fake = FakeMilvusClient(exists=True, raise_on="search")
    wrapper = MilvusClientWrapper(client=fake)  # type: ignore[arg-type]
    with pytest.raises(ExternalServiceError):
        await wrapper.search(_vec())


async def test_search_rejects_wrong_query_dimension() -> None:
    """查询向量维度错误时快速失败，不访问 Milvus。"""
    fake = FakeMilvusClient(exists=True)
    wrapper = MilvusClientWrapper(client=fake)  # type: ignore[arg-type]
    with pytest.raises(ParamError):
        await wrapper.search([0.1, 0.2])
    assert fake.calls == []


async def test_delete_by_doc_builds_expr() -> None:
    """删文档时按 doc_id 构造过滤表达式并返回删除条数。"""
    fake = FakeMilvusClient(exists=True, delete_count=17)
    wrapper = MilvusClientWrapper(client=fake)  # type: ignore[arg-type]
    assert await wrapper.delete_by_doc(9) == 17
    assert fake.call_kwargs("delete")["filter"] == "doc_id == 9"


async def test_delete_by_doc_creates_collection_when_missing() -> None:
    """首次入库时集合不存在：先创建集合再删除（避免 delete 报 collection not exist）。"""
    fake = FakeMilvusClient(exists=False)
    wrapper = MilvusClientWrapper(client=fake)  # type: ignore[arg-type]
    assert await wrapper.delete_by_doc(1) == 0
    methods = [name for name, _ in fake.calls]
    assert methods.index("create_collection") < methods.index("delete")


async def test_delete_by_ids_builds_expr() -> None:
    """按主键批量删除。"""
    fake = FakeMilvusClient(exists=True, delete_count=2)
    wrapper = MilvusClientWrapper(client=fake)  # type: ignore[arg-type]
    assert await wrapper.delete_by_ids([1, 2]) == 2
    assert fake.call_kwargs("delete")["filter"] == "chunk_id in [1, 2]"
    assert await wrapper.delete_by_ids([]) == 0


async def test_get_collection_stats() -> None:
    """统计切片总数（管理端展示）。"""
    fake = FakeMilvusClient(exists=True, query_rows=[{"count(*)": 1280}])
    wrapper = MilvusClientWrapper(client=fake)  # type: ignore[arg-type]
    stats = await wrapper.get_collection_stats()
    assert stats["row_count"] == 1280
    assert stats["dim"] == 1024
    assert fake.call_kwargs("query")["output_fields"] == ["count(*)"]


async def test_ping() -> None:
    """探活成功/失败都不抛异常，只返回布尔值。"""
    ok_wrapper = MilvusClientWrapper(client=FakeMilvusClient())  # type: ignore[arg-type]
    assert await ok_wrapper.ping() is True
    bad_wrapper = MilvusClientWrapper(
        client=FakeMilvusClient(raise_on="list_collections")  # type: ignore[arg-type]
    )
    assert await bad_wrapper.ping() is False


def test_build_filter_variants() -> None:
    """过滤表达式构造：空参数返回空串，多条件用 and 连接，ID 去重排序。"""
    assert build_filter() == ""
    assert build_filter(kb_ids=[]) == ""
    assert build_filter(kb_ids=[3, 1, 3]) == "kb_id in [1, 3]"
    assert build_filter(category="反诈防骗") == 'category == "反诈防骗"'
    assert (
        build_filter(kb_ids=[1], category="储蓄存款", extra='risk_level != "high"')
        == 'kb_id in [1] and category == "储蓄存款" and (risk_level != "high")'
    )


def test_get_instance_is_singleton(monkeypatch: pytest.MonkeyPatch) -> None:
    """单例访问器：同一进程内返回同一实例（替换底层客户端，避免真实连接）。"""
    monkeypatch.setattr(
        MilvusClientWrapper, "_build_client", lambda self: FakeMilvusClient()
    )
    MilvusClientWrapper.reset_instance()


def test_build_client_prefers_local_uri(monkeypatch: pytest.MonkeyPatch) -> None:
    """配置了 MILVUS_URI 时使用本地嵌入式模式（Milvus Lite），不再连接 host:port。"""
    captured: dict[str, Any] = {}

    class _FakeClient:
        def __init__(self, **kwargs: Any) -> None:
            captured.update(kwargs)

    monkeypatch.setattr("app.vectorstore.milvus_client.MilvusClient", _FakeClient)
    from app.core.config import Settings

    settings = Settings(milvus_lite_path="work_logs/milvus_e2e.db", milvus_host="should-not-use")
    MilvusClientWrapper(settings=settings)  # 构造即调用 _build_client
    assert captured["uri"] == "work_logs/milvus_e2e.db"
    assert "token" not in captured or captured.get("token") is None
    first = MilvusClientWrapper.get_instance()
    second = MilvusClientWrapper.get_instance()
    assert first is second
    MilvusClientWrapper.reset_instance()

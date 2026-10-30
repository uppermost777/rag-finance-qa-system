"""Milvus 向量库封装（pymilvus，HNSW + COSINE，1024 维）。

约定（详见 DESIGN.md 第 5 章）：
- Collection 名由 ``MILVUS_COLLECTION`` 配置，单集合 + 标量过滤实现多知识库隔离；
- 主键 ``chunk_id`` 与 MySQL ``document_chunk.id`` 一致，便于双侧对账；
- 同步 SDK 一律用 ``asyncio.to_thread`` 包装，避免阻塞事件循环；
- 连接/调用失败统一抛 ``ExternalServiceError``，绝不静默吞异常。
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Any, Iterable, Optional, Sequence

from pymilvus import DataType, MilvusClient

from app.core.config import Settings, get_settings
from app.core.exceptions import ExternalServiceError, ParamError
from app.core.logging import get_logger

logger = get_logger(__name__)

# ---------- Collection 常量（与 DESIGN.md 5.1 一致） ----------
FIELD_CHUNK_ID = "chunk_id"
FIELD_DOC_ID = "doc_id"
FIELD_KB_ID = "kb_id"
FIELD_CONTENT = "content"
FIELD_CATEGORY = "category"
FIELD_SOURCE = "source"
FIELD_PAGE = "page"
FIELD_RISK_LEVEL = "risk_level"
FIELD_EMBEDDING = "embedding"

CONTENT_MAX_LENGTH = 8192
SOURCE_MAX_LENGTH = 256
CATEGORY_MAX_LENGTH = 64
RISK_LEVEL_MAX_LENGTH = 16

INDEX_TYPE = "HNSW"
METRIC_TYPE = "COSINE"
INDEX_PARAMS = {"M": 16, "efConstruction": 200}
SEARCH_EF = 64

# 检索默认返回的标量字段（不含向量本身）
DEFAULT_OUTPUT_FIELDS = (
    FIELD_CHUNK_ID,
    FIELD_DOC_ID,
    FIELD_KB_ID,
    FIELD_CONTENT,
    FIELD_CATEGORY,
    FIELD_SOURCE,
    FIELD_PAGE,
    FIELD_RISK_LEVEL,
)


@dataclass(frozen=True)
class ChunkRecord:
    """待写入 Milvus 的一条切片记录。"""

    chunk_id: int
    doc_id: int
    kb_id: int
    content: str
    embedding: Sequence[float]
    category: str = ""
    source: str = ""
    page: int = 0
    risk_level: str = "low"

    def to_milvus_dict(self) -> dict[str, Any]:
        """转换为 Milvus insert 所需的字典。"""
        return {
            FIELD_CHUNK_ID: int(self.chunk_id),
            FIELD_DOC_ID: int(self.doc_id),
            FIELD_KB_ID: int(self.kb_id),
            FIELD_CONTENT: self.content[:CONTENT_MAX_LENGTH],
            FIELD_CATEGORY: (self.category or "")[:CATEGORY_MAX_LENGTH],
            FIELD_SOURCE: (self.source or "")[:SOURCE_MAX_LENGTH],
            FIELD_PAGE: int(self.page),
            FIELD_RISK_LEVEL: (self.risk_level or "low")[:RISK_LEVEL_MAX_LENGTH],
            FIELD_EMBEDDING: list(self.embedding),
        }


@dataclass(frozen=True)
class VectorHit:
    """向量检索命中的一条结果。"""

    chunk_id: int
    score: float
    doc_id: int = 0
    kb_id: int = 0
    content: str = ""
    category: str = ""
    source: str = ""
    page: int = 0
    risk_level: str = "low"
    extra: dict[str, Any] = field(default_factory=dict)


def build_filter(
    kb_ids: Optional[Iterable[int]] = None,
    category: Optional[str] = None,
    extra: Optional[str] = None,
) -> str:
    """构造 Milvus 标量过滤表达式（布尔表达式语法）。

    :param kb_ids: 限定的知识库 ID 集合；为空表示不过滤
    :param category: 精确匹配的分类
    :param extra: 追加的原始表达式（如 ``risk_level != "high"``）
    """
    clauses: list[str] = []
    if kb_ids:
        ids = sorted({int(item) for item in kb_ids})
        if ids:
            clauses.append(f"{FIELD_KB_ID} in [{', '.join(str(i) for i in ids)}]")
    if category:
        clauses.append(f'{FIELD_CATEGORY} == "{category}"')
    if extra:
        clauses.append(f"({extra})")
    return " and ".join(clauses)


class MilvusClientWrapper:
    """Milvus 访问包装类（进程内单例）。"""

    _instance: Optional["MilvusClientWrapper"] = None

    def __init__(self, client: Optional[MilvusClient] = None, settings: Optional[Settings] = None) -> None:
        """初始化包装类。

        :param client: 可注入的 pymilvus 客户端（测试用；默认按配置自建）
        :param settings: 配置对象，默认取全局单例
        """
        self._settings = settings or get_settings()
        self._collection = self._settings.milvus_collection
        self._dim = self._settings.embedding_dim
        self._client = client if client is not None else self._build_client()
        self._collection_ready = False

    # ------------------------------------------------------------------ 构造
    @classmethod
    def get_instance(cls) -> "MilvusClientWrapper":
        """获取全局单例。"""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @classmethod
    def reset_instance(cls) -> None:
        """清空单例（测试用）。"""
        cls._instance = None

    def _build_client(self) -> MilvusClient:
        """按配置创建 pymilvus 客户端。"""
        settings = self._settings
        if settings.milvus_lite_path:
            # 本地嵌入式模式（Milvus Lite）：单文件存储，免服务端，适合本地联调与演示
            logger.info("使用本地 Milvus Lite：%s", settings.milvus_lite_path)
            return MilvusClient(uri=settings.milvus_lite_path, timeout=settings.milvus_timeout)
        uri = f"http://{settings.milvus_host}:{settings.milvus_port}"
        token = (
            f"{settings.milvus_user}:{settings.milvus_password}"
            if settings.milvus_user and settings.milvus_password
            else None
        )
        logger.info("连接 Milvus：%s collection=%s", uri, self._collection)
        return MilvusClient(uri=uri, token=token, timeout=settings.milvus_timeout)

    @property
    def collection_name(self) -> str:
        """当前 Collection 名。"""
        return self._collection

    # ------------------------------------------------------------------ 调用封装
    async def _call(self, method: str, /, *args: Any, **kwargs: Any) -> Any:
        """把同步 SDK 调用放到线程池执行，统一计时与异常包装。"""
        func = getattr(self._client, method, None)
        if func is None:
            raise ExternalServiceError(f"Milvus SDK 不支持方法：{method}")
        start = time.perf_counter()
        try:
            result = await asyncio.to_thread(func, *args, **kwargs)
        except Exception as exc:  # pragma: no cover - 依赖真实环境
            logger.error("Milvus 调用 %s 失败：%s", method, exc)
            raise ExternalServiceError(f"向量库操作失败（{method}）：{exc}") from exc
        cost_ms = round((time.perf_counter() - start) * 1000, 2)
        logger.debug("Milvus.%s 完成，耗时 %sms", method, cost_ms)
        return result

    async def ping(self) -> bool:
        """探活：能否列出 Collection。"""
        try:
            await self._call("list_collections")
            return True
        except ExternalServiceError:
            return False

    # ------------------------------------------------------------------ Schema
    def _build_schema(self) -> Any:
        """构造 Collection Schema（字段定义见 DESIGN.md 5.1）。"""
        schema = self._client.create_schema(auto_id=False, enable_dynamic_field=False)
        schema.add_field(
            field_name=FIELD_CHUNK_ID, datatype=DataType.INT64, is_primary=True, auto_id=False
        )
        schema.add_field(field_name=FIELD_DOC_ID, datatype=DataType.INT64)
        schema.add_field(field_name=FIELD_KB_ID, datatype=DataType.INT64)
        schema.add_field(
            field_name=FIELD_CONTENT, datatype=DataType.VARCHAR, max_length=CONTENT_MAX_LENGTH
        )
        schema.add_field(
            field_name=FIELD_CATEGORY, datatype=DataType.VARCHAR, max_length=CATEGORY_MAX_LENGTH
        )
        schema.add_field(
            field_name=FIELD_SOURCE, datatype=DataType.VARCHAR, max_length=SOURCE_MAX_LENGTH
        )
        schema.add_field(field_name=FIELD_PAGE, datatype=DataType.INT32)
        schema.add_field(
            field_name=FIELD_RISK_LEVEL,
            datatype=DataType.VARCHAR,
            max_length=RISK_LEVEL_MAX_LENGTH,
        )
        schema.add_field(
            field_name=FIELD_EMBEDDING, datatype=DataType.FLOAT_VECTOR, dim=self._dim
        )
        return schema

    def _build_index_params(self) -> Any:
        """构造索引参数：HNSW(M=16, efConstruction=200) + COSINE。"""
        index_params = self._client.prepare_index_params()
        index_params.add_index(
            field_name=FIELD_EMBEDDING,
            index_type=INDEX_TYPE,
            metric_type=METRIC_TYPE,
            params=dict(INDEX_PARAMS),
        )
        return index_params

    async def ensure_collection(self, force: bool = False) -> bool:
        """确保 Collection 存在且已建索引；不存在则创建。

        :param force: 为 True 时忽略本地缓存标记，重新向 Milvus 确认
        :return: True 表示本次创建了新 Collection，False 表示已存在
        """
        if self._collection_ready and not force:
            return False
        exists = bool(await self._call("has_collection", collection_name=self._collection))
        if exists:
            self._collection_ready = True
            # 进程重启后集合可能处于 released 状态，检索前必须 load（已加载时调用是幂等的）
            try:
                await self._call("load_collection", collection_name=self._collection)
            except ExternalServiceError as exc:
                logger.debug("load_collection 忽略异常（可能已加载）：%s", exc)
            logger.info("Milvus Collection 已存在：%s", self._collection)
            return False

        schema = self._build_schema()
        index_params = self._build_index_params()
        await self._call(
            "create_collection",
            collection_name=self._collection,
            schema=schema,
            index_params=index_params,
        )
        self._collection_ready = True
        logger.info(
            "已创建 Milvus Collection：%s（dim=%s, %s/%s, %s）",
            self._collection,
            self._dim,
            INDEX_TYPE,
            METRIC_TYPE,
            INDEX_PARAMS,
        )
        return True

    # ------------------------------------------------------------------ 写入
    async def insert_chunks(self, chunks: Sequence[ChunkRecord]) -> list[int]:
        """批量写入切片向量。

        :param chunks: 切片记录（``chunk_id`` 需由调用方保证唯一，通常取 MySQL 主键）
        :return: 写入的主键列表
        """
        if not chunks:
            return []
        for record in chunks:
            if len(record.embedding) != self._dim:
                raise ParamError(
                    f"向量维度不匹配：期望 {self._dim}，实际 {len(record.embedding)}"
                )
        await self.ensure_collection()
        data = [record.to_milvus_dict() for record in chunks]
        start = time.perf_counter()
        result = await self._call(
            "insert", collection_name=self._collection, data=data
        )
        ids = [int(item) for item in (result or {}).get("ids", [])]
        logger.info(
            "写入向量 %d 条（返回 %d 个主键），耗时 %sms",
            len(data),
            len(ids),
            round((time.perf_counter() - start) * 1000, 2),
        )
        return ids or [record.chunk_id for record in chunks]

    # ------------------------------------------------------------------ 检索
    async def search(
        self,
        embedding: Sequence[float],
        top_k: Optional[int] = None,
        expr: Optional[str] = None,
        output_fields: Optional[Sequence[str]] = None,
    ) -> list[VectorHit]:
        """向量相似度检索。

        :param embedding: 查询向量（1024 维、已归一化）
        :param top_k: 返回条数，默认取配置 ``RECALL_TOP_K``
        :param expr: Milvus 标量过滤表达式，可用 ``build_filter`` 构造
        :param output_fields: 需要返回的标量字段
        :return: 命中列表（按相似度降序）
        """
        if len(embedding) != self._dim:
            raise ParamError(f"查询向量维度不匹配：期望 {self._dim}，实际 {len(embedding)}")
        await self.ensure_collection()
        limit = int(top_k or self._settings.recall_top_k)
        fields = list(output_fields or DEFAULT_OUTPUT_FIELDS)
        start = time.perf_counter()
        raw = await self._call(
            "search",
            collection_name=self._collection,
            data=[list(embedding)],
            filter=expr or "",
            limit=limit,
            output_fields=fields,
            search_params={"metric_type": METRIC_TYPE, "params": {"ef": SEARCH_EF}},
        )
        hits = self._parse_hits(raw)
        logger.info(
            "向量检索命中 %d 条（top_k=%d, filter=%s），耗时 %sms",
            len(hits),
            limit,
            expr or "-",
            round((time.perf_counter() - start) * 1000, 2),
        )
        return hits

    @staticmethod
    def _parse_hits(raw: Any) -> list[VectorHit]:
        """把 Milvus 原始返回解析为 VectorHit 列表。

        Milvus 服务端返回 ``{"id": ..., "distance": ..., "entity": {...}}``；
        Milvus Lite 返回 ``{"chunk_id": ..., "distance": ..., "entity": {...}}``，
        因此主键需要兼容两种键名（以真实联调发现的差异为准）。
        """
        hits: list[VectorHit] = []
        for group in raw or []:
            for item in group or []:
                if not isinstance(item, dict):
                    continue
                entity = item.get("entity") or {}
                raw_id = item.get("id")
                if raw_id is None:
                    raw_id = item.get(FIELD_CHUNK_ID)
                if raw_id is None:
                    raw_id = entity.get(FIELD_CHUNK_ID, 0)
                hits.append(
                    VectorHit(
                        chunk_id=int(raw_id or 0),
                        score=float(item.get("distance", item.get("score", 0.0))),
                        doc_id=int(entity.get(FIELD_DOC_ID, 0) or 0),
                        kb_id=int(entity.get(FIELD_KB_ID, 0) or 0),
                        content=str(entity.get(FIELD_CONTENT, "") or ""),
                        category=str(entity.get(FIELD_CATEGORY, "") or ""),
                        source=str(entity.get(FIELD_SOURCE, "") or ""),
                        page=int(entity.get(FIELD_PAGE, 0) or 0),
                        risk_level=str(entity.get(FIELD_RISK_LEVEL, "low") or "low"),
                    )
                )
        hits.sort(key=lambda hit: hit.score, reverse=True)
        return hits

    # ------------------------------------------------------------------ 删除与统计
    async def delete_by_doc(self, doc_id: int) -> int:
        """删除某文档的全部向量（删文档时同步调用）。

        首次入库时集合可能尚未创建，这里先确保集合存在，避免「删除→集合不存在」导致入库失败。
        """
        await self.ensure_collection()
        result = await self._call(
            "delete", collection_name=self._collection, filter=f"{FIELD_DOC_ID} == {int(doc_id)}"
        )
        count = self._delete_count(result)
        logger.info("已删除文档 %s 的向量 %d 条", doc_id, count)
        return count

    async def delete_by_ids(self, chunk_ids: Sequence[int]) -> int:
        """按主键批量删除向量（重建索引时使用）。"""
        ids = [int(item) for item in chunk_ids]
        if not ids:
            return 0
        expr = f"{FIELD_CHUNK_ID} in [{', '.join(str(i) for i in ids)}]"
        result = await self._call("delete", collection_name=self._collection, filter=expr)
        return self._delete_count(result)

    @staticmethod
    def _delete_count(result: Any) -> int:
        """兼容不同 Milvus 实现的删除返回结构。

        服务端返回 ``{"delete_count": n}``；Milvus Lite 返回 ``[n]`` 或
        ``[{"delete_count": n}]``，这里统一归一化为整数。
        """
        if isinstance(result, dict):
            return int(result.get("delete_count", 0) or 0)
        if isinstance(result, (list, tuple)):
            for item in result:
                if isinstance(item, dict) and "delete_count" in item:
                    return int(item["delete_count"] or 0)
                if isinstance(item, int):
                    return item
        return 0

    async def get_collection_stats(self) -> dict[str, Any]:
        """统计信息：供管理端展示切片总数。"""
        await self.ensure_collection()
        try:
            rows = await self._call(
                "query",
                collection_name=self._collection,
                filter="",
                output_fields=["count(*)"],
            )
            row_count = int(rows[0].get("count(*)", 0)) if rows else 0
        except ExternalServiceError:
            stats = await self._call(
                "get_collection_stats", collection_name=self._collection
            )
            row_count = int((stats or {}).get("row_count", 0))
        return {"collection": self._collection, "row_count": row_count, "dim": self._dim}

    @staticmethod
    def format_expr(kb_ids: Optional[Iterable[int]] = None, category: Optional[str] = None) -> str:
        """便捷方法：构造过滤表达式（等价于模块级 ``build_filter``）。"""
        return build_filter(kb_ids=kb_ids, category=category)

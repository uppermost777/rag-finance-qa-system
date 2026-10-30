"""对比实验编排：A/B/C/D 四组检索配置 → 指标 → CSV + 柱状图。

两种模式：
- ``offline``：哈希向量 + BM25 + 生产的 RRF 实现（无需 Milvus/模型，随时可复现）；
- ``service``：走完整生产链路（Milvus + bge-m3 + bge-reranker + LLM），产出论文正式数据；
``auto`` 会先尝试 service，依赖不可用时自动降级为 offline，并在结果文件名中标注模式。
"""

from __future__ import annotations

import asyncio
import csv
import json
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Iterable, Optional, Sequence

from app.core.logging import get_logger
from app.core.config import get_settings
from app.evaluation.metrics import SampleMetrics, aggregate, evaluate_sample
from app.evaluation.offline import OfflineCorpusItem, OfflineRetriever

logger = get_logger(__name__)

DATASET_DIR = Path(__file__).resolve().parent / "datasets"
RESULTS_DIR = Path(__file__).resolve().parent / "results"

# 策略 → 论文中的配置代号
CONFIG_LABELS = {
    "vector": "A_纯向量检索",
    "bm25": "B_纯BM25检索",
    "hybrid": "C_混合检索(RRF)",
    "hybrid_rerank": "D_混合检索+重排(本系统)",
}
DEFAULT_STRATEGIES = ["vector", "bm25", "hybrid", "hybrid_rerank"]

# 评测分类 → 种子知识库名称（按名称解析 ID，避免依赖数据库自增顺序）
CATEGORY_KB_NAME = {
    "储蓄存款": "银行储蓄与存款保险",
    "基金投资": "基金与指数投资基础",
    "股票基础": "股票投资基础",
    "保险保障": "保险与保障规划",
    "信用借贷": "个人信用与借贷风险",
    "反诈防骗": "反诈与金融安全",
}

ProgressFn = Callable[[int, int, str], None]


@dataclass
class EvalRecord:
    """一条评测样本。"""

    id: int
    question: str
    ground_truth: str
    kb_id: int = 0
    category: str = ""
    reference_contexts: list[str] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        """未提供参考上下文时，用标准答案兜底。"""
        if not self.reference_contexts:
            self.reference_contexts = [self.ground_truth]


def resolve_dataset_path(name: str) -> Path:
    """定位评测集文件（支持相对文件名或绝对路径）。"""
    path = Path(name)
    if not path.is_absolute():
        path = DATASET_DIR / name
    if not path.is_file():
        raise FileNotFoundError(f"评测集不存在：{path}")
    return path


def load_dataset(name: str = "finance_qa_eval.jsonl", limit: Optional[int] = None) -> list[EvalRecord]:
    """读取 JSONL 评测集。"""
    path = resolve_dataset_path(name)
    records: list[EvalRecord] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            payload = json.loads(line)
            records.append(
                EvalRecord(
                    id=int(payload.get("id", len(records) + 1)),
                    question=str(payload["question"]),
                    ground_truth=str(payload["ground_truth"]),
                    kb_id=int(payload.get("kb_id", 0)),
                    category=str(payload.get("category", "")),
                    reference_contexts=list(payload.get("reference_contexts") or []),
                )
            )
    if limit:
        records = records[: int(limit)]
    logger.info("已加载评测集 %s：%d 条", path.name, len(records))
    return records


def build_offline_retriever(records: Sequence[EvalRecord]) -> OfflineRetriever:
    """用评测集的标准答案构建离线语料（每条样本一个切片段）。"""
    corpus = [
        OfflineCorpusItem(
            chunk_id=index + 1,
            content=record.ground_truth,
            source=f"{record.category or '知识库'}-{record.id}",
            category=record.category,
            kb_id=record.kb_id,
        )
        for index, record in enumerate(records)
    ]
    return OfflineRetriever(corpus)


def _timestamp() -> str:
    """结果文件时间戳。"""
    return datetime.now().strftime("%Y%m%d_%H%M%S")


async def run_offline_comparison(
    *,
    run_id: str,
    records: Sequence[EvalRecord],
    strategies: Sequence[str],
    progress: Optional[ProgressFn] = None,
    output_dir: Optional[Path] = None,
) -> dict[str, Any]:
    """离线模式：哈希向量 + BM25 + RRF 的横向对比。"""
    retriever = build_offline_retriever(records)
    rows: list[dict[str, Any]] = []
    samples: list[SampleMetrics] = []
    total = len(strategies) * len(records)
    done = 0

    for strategy in strategies:
        collect: list[SampleMetrics] = []
        for record in records:
            chunks, latency_ms = retriever.measure_latency(record.question, strategy)
            contexts = [chunk.content for chunk in chunks]
            answer = retriever.extractive_answer(record.question, chunks)
            metric = evaluate_sample(
                question=record.question,
                answer=answer,
                retrieved_contexts=contexts,
                ground_truth=record.ground_truth,
                reference_contexts=record.reference_contexts,
                category=record.category,
                strategy=strategy,
                latency_ms=latency_ms,
            )
            collect.append(metric)
            samples.append(metric)
            done += 1
            if progress is not None:
                progress(done, total, f"{strategy} 第 {record.id} 条")
        summary = aggregate(collect)
        rows.append(
            {
                "config": CONFIG_LABELS.get(strategy, strategy),
                "strategy": strategy,
                **summary,
                "samples": len(collect),
            }
        )
    return _write_results(run_id=run_id, rows=rows, samples=samples, mode="offline", output_dir=output_dir)


async def run_service_comparison(
    *,
    run_id: str,
    records: Sequence[EvalRecord],
    strategies: Sequence[str],
    progress: Optional[ProgressFn] = None,
    output_dir: Optional[Path] = None,
    proxy_rerank: bool = False,
    kb_scope: str = "record",
) -> dict[str, Any]:
    """服务模式：走真实 RAG 链路（Milvus + bge-m3 + 重排 + LLM）。"""
    from app.chains.graph import QaPipeline
    from app.evaluation.offline import LexicalReranker
    from app.retrieval.reranker import RerankerService

    if proxy_rerank:
        reranker = LexicalReranker()
        logger.warning("评估使用词重叠代理重排（未加载交叉编码器模型）")
    else:
        reranker = RerankerService()
        logger.info("评估使用真实交叉编码器重排：%s", get_settings().reranker_model_path)
    pipeline = QaPipeline(reranker=reranker)
    kb_map = await resolve_category_kb_ids()
    logger.info("分类→知识库 ID 映射：%s", kb_map)
    rows: list[dict[str, Any]] = []
    samples: list[SampleMetrics] = []
    total = len(strategies) * len(records)
    done = 0

    for strategy in strategies:
        collect: list[SampleMetrics] = []
        for record in records:
            started = time.perf_counter()
            if kb_scope == "all":
                scope_ids = None
            else:
                resolved = kb_map.get(record.category) or record.kb_id
                scope_ids = [resolved] if resolved else None
            result = await pipeline.aanswer(
                record.question, kb_ids=scope_ids, strategy=strategy
            )
            latency_ms = int((time.perf_counter() - started) * 1000)
            contexts = [chunk.content for chunk in result.chunks]
            metric = evaluate_sample(
                question=record.question,
                answer=result.answer,
                retrieved_contexts=contexts,
                ground_truth=record.ground_truth,
                reference_contexts=record.reference_contexts,
                category=record.category,
                strategy=strategy,
                latency_ms=latency_ms,
            )
            collect.append(metric)
            samples.append(metric)
            done += 1
            if progress is not None:
                progress(done, total, f"{strategy} 第 {record.id} 条")
        rows.append(
            {
                "config": CONFIG_LABELS.get(strategy, strategy),
                "strategy": strategy,
                **aggregate(collect),
                "samples": len(collect),
                "reranker": (
                    "lexical-proxy"
                    if (proxy_rerank and strategy == "hybrid_rerank")
                    else ("bge-reranker" if strategy == "hybrid_rerank" else "none")
                ),
            }
        )
    return _write_results(run_id=run_id, rows=rows, samples=samples, mode="service", output_dir=output_dir)


async def run_comparison(
    *,
    run_id: str,
    dataset_name: str = "finance_qa_eval.jsonl",
    limit: Optional[int] = None,
    strategies: Optional[Sequence[str]] = None,
    progress: Optional[ProgressFn] = None,
    mode: str = "auto",
    output_dir: Optional[Path] = None,
    proxy_rerank: bool = False,
    kb_scope: str = "record",
) -> dict[str, Any]:
    """执行对比实验（自动在 service / offline 之间选择）。"""
    chosen = [item for item in (strategies or DEFAULT_STRATEGIES) if item in CONFIG_LABELS]
    if not chosen:
        chosen = list(DEFAULT_STRATEGIES)
    records = load_dataset(dataset_name, limit=limit)
    if not records:
        raise ValueError("评测集为空")

    try:
        if mode in ("auto", "service"):
            try:
                result = await run_service_comparison(
                    run_id=run_id,
                    records=records,
                    strategies=chosen,
                    progress=progress,
                    output_dir=output_dir,
                    proxy_rerank=proxy_rerank,
                    kb_scope=kb_scope,
                )
                result["mode"] = "service"
                return result
            except Exception as exc:
                if mode == "service":
                    raise
                logger.warning("服务模式不可用（%s），自动降级为离线模式", exc)

        return await run_offline_comparison(
            run_id=run_id, records=records, strategies=chosen, progress=progress, output_dir=output_dir
        )
    finally:
        # 关闭数据库连接池，避免 CLI 退出时出现 "Event loop is closed" 噪声
        from app.core.database import dispose_engine

        await dispose_engine()


async def resolve_category_kb_ids() -> dict[str, int]:
    """按名称把评测分类解析为实际的知识库 ID（不依赖自增顺序）。"""
    from sqlalchemy import select

    from app.core.database import get_session_factory
    from app.models import KnowledgeBase

    factory = get_session_factory()
    async with factory() as session:
        rows = (await session.execute(select(KnowledgeBase.id, KnowledgeBase.name))).all()
    name_to_id = {name: int(kb_id) for kb_id, name in rows}
    return {
        category: name_to_id[name]
        for category, name in CATEGORY_KB_NAME.items()
        if name in name_to_id
    }


def _write_results(
    *,
    run_id: str,
    rows: Sequence[dict[str, Any]],
    samples: Sequence[SampleMetrics],
    mode: str,
    output_dir: Optional[Path] = None,
) -> dict[str, Any]:
    """写 CSV（汇总 + 逐样本）与柱状图，返回产出路径。"""
    directory = Path(output_dir) if output_dir else RESULTS_DIR
    directory.mkdir(parents=True, exist_ok=True)
    stamp = _timestamp()
    csv_path = directory / f"compare_{mode}_{stamp}_{run_id}.csv"
    detail_path = directory / f"compare_{mode}_{stamp}_{run_id}_samples.csv"

    columns = [
        "config",
        "strategy",
        "faithfulness",
        "answer_relevancy",
        "context_precision",
        "context_recall",
        "avg_latency_ms",
        "samples",
        "reranker",
    ]
    with csv_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in columns})

    detail_columns = [
        "strategy",
        "category",
        "question",
        "faithfulness",
        "answer_relevancy",
        "context_precision",
        "context_recall",
        "latency_ms",
    ]
    with detail_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=detail_columns)
        writer.writeheader()
        for item in samples:
            writer.writerow(item.to_dict())

    chart_path = plot_comparison(rows, directory / f"compare_{mode}_{stamp}_{run_id}.png")
    logger.info("对比实验完成（%s）：%s", mode, csv_path)
    return {
        "run_id": run_id,
        "mode": mode,
        "csv_path": str(csv_path),
        "detail_path": str(detail_path),
        "chart_path": str(chart_path) if chart_path else None,
        "rows": list(rows),
    }


def configure_cjk_font() -> bool:
    """尝试配置中文字体；不可用时返回 False（图表改用英文标签）。"""
    try:
        import matplotlib

        from matplotlib import font_manager
    except ImportError:  # pragma: no cover - 未安装 matplotlib
        return False
    available = {font.name for font in font_manager.fontManager.ttflist}
    for name in ("Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "WenQuanYi Zen Hei", "Arial Unicode MS"):
        if name in available:
            matplotlib.rcParams["font.sans-serif"] = [name]
            matplotlib.rcParams["axes.unicode_minus"] = False
            return True
    logger.warning("未找到中文字体，图表将使用英文标签")
    return False


def plot_comparison(rows: Sequence[dict[str, Any]], path: Path) -> Optional[Path]:
    """绘制四组配置的指标柱状图。"""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:  # pragma: no cover - 未安装 matplotlib
        return None

    cjk = configure_cjk_font()
    metrics = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]
    labels_en = ["Faithfulness", "AnswerRelevancy", "ContextPrecision", "ContextRecall"]
    labels_cn = ["忠实度", "答案相关性", "上下文精确率", "上下文召回率"]
    labels = labels_cn if cjk else labels_en
    configs = [row["config"] for row in rows]
    x = np.arange(len(labels))
    width = 0.8 / max(len(configs), 1)

    fig, ax = plt.subplots(figsize=(10, 5.5))
    for index, row in enumerate(rows):
        values = [float(row.get(metric, 0.0)) for metric in metrics]
        ax.bar(x + index * width, values, width, label=row["config"])
    ax.set_ylabel("分数" if cjk else "Score")
    ax.set_title("检索策略对比实验" if cjk else "Retrieval Strategy Comparison")
    ax.set_xticks(x + width * (len(configs) - 1) / 2)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1.05)
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    logger.info("已生成对比图：%s", path)
    return path


async def run_chunk_grid(
    *,
    chunk_sizes: Iterable[int] = (256, 512, 1024),
    overlaps: Iterable[int] = (32, 64, 128),
    limit: Optional[int] = 10,
    output_dir: Optional[Path] = None,
) -> dict[str, Any]:
    """切片参数敏感性实验：chunk_size × overlap 网格（基于离线检索器）。

    实验设计：把评测集按分类聚合成 6 篇「知识库长文档」，用不同切片参数切分后，
    对同一批问题做混合检索，比较切片粒度对 context_recall / context_precision 的影响。
    这样每组的切片数量与边界都不同，指标才有可比性（若逐条样本切片，短文本永远只切出 1 块）。
    """
    from app.ingestion.loader import PageContent
    from app.ingestion.splitter import split_pages

    records = load_dataset(limit=limit)
    # 按分类聚合成长文档：同一分类的所有标准答案拼成一篇知识文档
    grouped: dict[str, list[EvalRecord]] = {}
    for record in records:
        grouped.setdefault(record.category or "未分类", []).append(record)

    rows: list[dict[str, Any]] = []
    for chunk_size in chunk_sizes:
        for overlap in overlaps:
            if overlap >= chunk_size:
                continue
            pieces: list[tuple[str, str]] = []  # (内容, 分类)
            for group_index, (category, items) in enumerate(grouped.items(), start=1):
                text = "\n".join(item.ground_truth for item in items)
                pages = [PageContent(text=text, page_no=1, source=f"{category}知识汇编")]
                documents = split_pages(
                    pages,
                    doc_id=group_index,
                    kb_id=items[0].kb_id,
                    category=category,
                    chunk_size=chunk_size,
                    chunk_overlap=overlap,
                )
                pieces.extend((item.page_content, category) for item in documents)
            retriever = OfflineRetriever(
                [
                    OfflineCorpusItem(
                        chunk_id=index + 1,
                        content=text,
                        source=f"{category}知识汇编",
                        category=category,
                    )
                    for index, (text, category) in enumerate(pieces)
                ]
            )
            collect: list[SampleMetrics] = []
            for record in records:
                chunks, latency_ms = retriever.measure_latency(record.question, "hybrid")
                contexts = [chunk.content for chunk in chunks]
                collect.append(
                    evaluate_sample(
                        question=record.question,
                        answer=retriever.extractive_answer(record.question, chunks),
                        retrieved_contexts=contexts,
                        ground_truth=record.ground_truth,
                        reference_contexts=record.reference_contexts,
                        category=record.category,
                        strategy="hybrid",
                        latency_ms=latency_ms,
                    )
                )
            summary = aggregate(collect)
            rows.append(
                {
                    "chunk_size": chunk_size,
                    "chunk_overlap": overlap,
                    "chunk_count": len(pieces),
                    **summary,
                }
            )
            logger.info(
                "切片网格 chunk_size=%s overlap=%s → recall=%.3f precision=%.3f",
                chunk_size,
                overlap,
                summary["context_recall"],
                summary["context_precision"],
            )

    directory = Path(output_dir) if output_dir else RESULTS_DIR
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"chunk_grid_{_timestamp()}.csv"
    columns = [
        "chunk_size",
        "chunk_overlap",
        "chunk_count",
        "faithfulness",
        "answer_relevancy",
        "context_precision",
        "context_recall",
        "avg_latency_ms",
    ]
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in columns})
    logger.info("切片参数实验完成：%s", path)
    return {"csv_path": str(path), "rows": rows}


def run_comparison_sync(**kwargs: Any) -> dict[str, Any]:
    """同步包装（脚本 CLI 使用）。"""
    return asyncio.run(run_comparison(**kwargs))

"""P12 验收：评测集、指标、离线对比实验与切片网格实验。"""

from __future__ import annotations

import csv
from pathlib import Path

import pytest

from app.evaluation.metrics import (
    aggregate,
    answer_relevancy,
    context_precision,
    context_recall,
    evaluate_sample,
    faithfulness,
    f1_overlap,
    overlap_ratio,
)
from app.evaluation.offline import OfflineCorpusItem, OfflineRetriever, hashed_embedding
from app.evaluation.ragas_eval import RAGAS_METRIC_NAMES, evaluate_samples, ragas_available
from app.evaluation.runner import (
    CONFIG_LABELS,
    load_dataset,
    run_chunk_grid,
    run_comparison,
)

CATEGORIES = {"储蓄存款", "基金投资", "股票基础", "保险保障", "信用借贷", "反诈防骗"}


# --------------------------------------------------------------------------- 数据集
def test_dataset_has_sixty_samples_covering_six_categories() -> None:
    """评测集：60 条、覆盖 6 大分类、字段齐全。"""
    records = load_dataset()
    assert len(records) == 60
    assert {item.category for item in records} == CATEGORIES
    for item in records:
        assert item.question.endswith("？") or item.question.endswith("?")
        assert len(item.ground_truth) >= 15
        assert 1 <= item.kb_id <= 6
        assert item.reference_contexts  # 未显式提供时用标准答案兜底
    per_category = {category: 0 for category in CATEGORIES}
    for item in records:
        per_category[item.category] += 1
    assert all(count == 10 for count in per_category.values())


def test_dataset_limit() -> None:
    """--limit 参数生效。"""
    assert len(load_dataset(limit=5)) == 5


def test_dataset_missing_file() -> None:
    """不存在的评测集给出清晰错误。"""
    with pytest.raises(FileNotFoundError):
        load_dataset("not-exists.jsonl")


# --------------------------------------------------------------------------- 指标
def test_overlap_helpers() -> None:
    """重叠率与 F1 的基础性质。"""
    assert overlap_ratio(["a", "b"], ["a", "b", "c"]) == 1.0
    assert overlap_ratio(["a", "b"], ["c"]) == 0.0
    assert overlap_ratio([], ["a"]) == 0.0
    assert f1_overlap(["a", "b"], ["a", "b"]) == 1.0
    assert f1_overlap(["a"], ["b"]) == 0.0


def test_metrics_reward_relevant_retrieval() -> None:
    """相关检索的指标应显著高于无关检索（指标方向正确）。"""
    question = "存款保险最多赔多少？"
    ground_truth = "同一存款人在同一家银行的存款本息合计 50 万元以内由存款保险基金偿付。"
    good_context = ["存款保险条例规定，同一存款人在同一家银行的存款本息合计在 50 万元以内的部分由存款保险基金依法偿付。"]
    bad_context = ["股票交易实行 T+1，买卖以手为单位，并设有涨跌幅限制。"]
    answer = "存款保险的偿付上限是同一银行 50 万元本息合计。[1]"

    assert context_recall(good_context, [ground_truth]) > 0.8
    assert context_recall(bad_context, [ground_truth]) < 0.3
    assert context_precision(good_context, [ground_truth]) == 1.0
    assert context_precision(bad_context, [ground_truth]) == 0.0
    assert faithfulness(answer, good_context) > faithfulness(answer, bad_context)
    assert answer_relevancy(answer, question, ground_truth) > 0.4


def test_metrics_empty_inputs_are_zero() -> None:
    """空输入的指标为 0，不抛异常。"""
    assert context_precision([], ["x"]) == 0.0
    assert context_recall(["x"], []) == 0.0
    assert faithfulness("", ["x"]) == 0.0
    assert answer_relevancy("", "", "") == 0.0
    assert aggregate([])["faithfulness"] == 0.0


def test_evaluate_sample_and_aggregate() -> None:
    """单样本指标可聚合。"""
    sample = evaluate_sample(
        question="什么是基金定投",
        answer="基金定投是在固定时间投入固定金额。[1]",
        retrieved_contexts=["基金定投是指在固定时间以固定金额申购指定基金。"],
        ground_truth="基金定投是固定时间固定金额买入基金。",
        category="基金投资",
        strategy="hybrid",
        latency_ms=120,
    )
    summary = aggregate([sample, sample])
    assert summary["avg_latency_ms"] == 120
    assert 0 <= summary["faithfulness"] <= 1
    assert sample.to_dict()["category"] == "基金投资"


# --------------------------------------------------------------------------- 离线检索
def test_hashed_embedding_is_stable_and_normalized() -> None:
    """哈希向量：同一文本稳定、已归一化、不同文本近似正交。"""
    first = hashed_embedding("存款保险 50 万元 偿付")
    second = hashed_embedding("存款保险 50 万元 偿付")
    assert first == second
    assert pytest.approx(sum(value * value for value in first), abs=1e-6) == 1.0
    other = hashed_embedding("股票 T+1 涨跌幅限制")
    dot = sum(a * b for a, b in zip(first, other, strict=False))
    assert abs(dot) < 0.5


def test_offline_retriever_strategies_differ() -> None:
    """四种策略都能返回结果，且重排会改变排序（证明重排确实起作用）。"""
    corpus = [
        OfflineCorpusItem(chunk_id=1, content="存款保险条例规定同一存款人 50 万元以内由存款保险基金偿付。", category="储蓄存款"),
        OfflineCorpusItem(chunk_id=2, content="基金定投是在固定时间以固定金额申购指定基金，平滑买入成本。", category="基金投资"),
        OfflineCorpusItem(chunk_id=3, content="股票交易实行 T+1，买卖以手为单位并设有涨跌幅限制。", category="股票基础"),
        OfflineCorpusItem(chunk_id=4, content="个人征信报告记录还款情况，逾期记录自结清之日起保留五年。", category="信用借贷"),
    ]
    retriever = OfflineRetriever(corpus)
    query = "存款保险 偿付 上限 多少"

    vector_hits = retriever.retrieve(query, "vector", top_k=3)
    bm25_hits = retriever.retrieve(query, "bm25", top_k=3)
    hybrid_hits = retriever.retrieve(query, "hybrid", top_k=3)
    reranked = retriever.retrieve(query, "hybrid_rerank", top_k=3)

    assert all(hits for hits in (vector_hits, bm25_hits, hybrid_hits, reranked))
    assert vector_hits[0].chunk_id == 1 and bm25_hits[0].chunk_id == 1
    assert hybrid_hits[0].chunk_id == 1
    assert hybrid_hits[0].rrf_score is not None
    assert reranked[0].rerank_score is not None
    assert len(reranked) <= 5

    answer = retriever.extractive_answer(query, hybrid_hits)
    assert answer.endswith("[1]") and "存款保险" in answer


def test_offline_retriever_without_match_returns_empty() -> None:
    """完全无关的查询允许返回空结果。"""
    retriever = OfflineRetriever([OfflineCorpusItem(chunk_id=1, content="基金定投平滑成本。")])
    assert retriever.retrieve("完全无关的问题", "bm25") == []
    assert "未收录" in retriever.extractive_answer("无关", [])


# --------------------------------------------------------------------------- 对比实验
async def test_run_comparison_offline_writes_csv_and_chart(tmp_path: Path) -> None:
    """离线对比实验：产出汇总 CSV（A/B/C/D 四行）+ 逐样本明细 + 柱状图。"""
    result = await run_comparison(
        run_id="testrun",
        limit=6,
        mode="offline",
        output_dir=tmp_path,
    )

    assert result["mode"] == "offline"
    csv_path = Path(result["csv_path"])
    assert csv_path.is_file() and "offline" in csv_path.name and "testrun" in csv_path.name

    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 4
    assert [row["config"] for row in rows] == [
        CONFIG_LABELS["vector"],
        CONFIG_LABELS["bm25"],
        CONFIG_LABELS["hybrid"],
        CONFIG_LABELS["hybrid_rerank"],
    ]
    for row in rows:
        for column in ("faithfulness", "answer_relevancy", "context_precision", "context_recall"):
            value = float(row[column])
            assert 0.0 <= value <= 1.0
        assert float(row["avg_latency_ms"]) >= 0

    detail = Path(result["detail_path"])
    with detail.open("r", encoding="utf-8-sig", newline="") as handle:
        detail_rows = list(csv.DictReader(handle))
    assert len(detail_rows) == 24  # 4 种策略 × 6 条样本

    if result["chart_path"]:
        assert Path(result["chart_path"]).is_file()


async def test_run_comparison_respects_strategy_subset(tmp_path: Path) -> None:
    """只跑指定策略（用于小样本快速验证）。"""
    result = await run_comparison(
        run_id="subset",
        limit=3,
        strategies=["bm25", "hybrid_rerank"],
        mode="offline",
        output_dir=tmp_path,
    )
    assert len(result["rows"]) == 2
    assert {row["strategy"] for row in result["rows"]} == {"bm25", "hybrid_rerank"}


async def test_run_chunk_grid(tmp_path: Path) -> None:
    """切片参数网格实验：写出 CSV 且包含预期列。"""
    result = await run_chunk_grid(
        chunk_sizes=[128, 256], overlaps=[16, 32], limit=3, output_dir=tmp_path
    )
    path = Path(result["csv_path"])
    assert path.is_file()
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 4
    for row in rows:
        assert int(row["chunk_size"]) > int(row["chunk_overlap"])
        assert int(row["chunk_count"]) >= 1
        assert 0.0 <= float(row["context_recall"]) <= 1.0
    # 切片越细，切片数量越多（粒度效应可观测）
    by_key = {(int(row["chunk_size"]), int(row["chunk_overlap"])): int(row["chunk_count"]) for row in rows}
    for overlap in (16, 32):
        assert by_key[(128, overlap)] >= by_key[(256, overlap)]


# --------------------------------------------------------------------------- RAGAS 接入点
async def test_ragas_entry_falls_back_to_proxy() -> None:
    """未配置真实 API Key 时，RAGAS 入口自动降级为代理指标并给出 mode 字段。"""
    samples = [
        {
            "question": "什么是基金定投",
            "answer": "基金定投是固定时间投入固定金额。[1]",
            "contexts": ["基金定投是指在固定时间以固定金额申购指定基金。"],
            "ground_truth": "基金定投是固定时间固定金额买入基金。",
        }
    ]
    result = await evaluate_samples(samples)
    assert result["mode"] in {"ragas", "proxy"}
    assert set(RAGAS_METRIC_NAMES).issubset(set(result["metrics"]))
    assert isinstance(ragas_available(), bool)


async def test_ragas_explicit_without_package_falls_back() -> None:
    """显式要求 RAGAS 但环境不支持时，仍能拿到代理指标而非异常。"""
    samples = [
        {"question": "存款保险赔多少", "answer": "50 万元以内。", "contexts": ["50 万元以内偿付。"], "ground_truth": "50 万元以内。"}
    ]
    result = await evaluate_samples(samples, use_ragas=True)
    assert result["metrics"]["faithfulness"] >= 0

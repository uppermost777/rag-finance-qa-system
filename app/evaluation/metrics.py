"""评估指标实现（无 LLM 依赖的确定性代理指标 + RAGAS 接入说明）。

论文第 6 章需要 4 个指标。真实 RAGAS 指标需要调用大模型逐样本打分，
在离线环境（无 API Key）无法复现；因此这里实现一组**确定性代理指标**：

- ``context_precision``：命中的检索切片中，与参考上下文有实质重叠的比例；
- ``context_recall``   ：参考信息要点被检索内容覆盖的比例；
- ``faithfulness``     ：答案各句要点在检索上下文中出现的比例（越低说明越可能幻觉）；
- ``answer_relevancy`` ：答案与问题 + 标准答案的要点 F1。

代理指标与 RAGAS 指标的含义方向一致、可离线复现，用于横向比较 A/B/C/D 四组检索配置；
配置好 ``LLM_API_KEY`` 后可在 ``ragas_eval.py`` 中切换到真实 RAGAS 指标（同一套输入输出）。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Iterable, Sequence

from app.retrieval.hybrid import tokenize

# 切片与参考上下文视为“相关”的重叠阈值
PRECISION_OVERLAP_THRESHOLD = 0.3

_SENTENCE_SPLIT = re.compile(r"[。！？；\n]+")
# 引用编号（如 [1]）不参与忠实度计算
_CITATION_MARK = re.compile(r"\[\d+\]")


def content_tokens(text: str) -> set[str]:
    """抽取内容词集合（jieba 分词 + 停用词/标点过滤）。"""
    return set(tokenize(text or ""))


def overlap_ratio(source: Iterable[str], target: Iterable[str]) -> float:
    """重叠率：|A∩B| / |A|（A 为 source，衡量 source 被 target 覆盖的程度）。"""
    source_set = set(source)
    if not source_set:
        return 0.0
    return len(source_set & set(target)) / len(source_set)


def f1_overlap(left: Iterable[str], right: Iterable[str]) -> float:
    """两个词集合的 F1。"""
    left_set, right_set = set(left), set(right)
    if not left_set or not right_set:
        return 0.0
    common = len(left_set & right_set)
    if common == 0:
        return 0.0
    precision = common / len(left_set)
    recall = common / len(right_set)
    return 2 * precision * recall / (precision + recall)


def context_precision(
    retrieved_contexts: Sequence[str], reference_contexts: Sequence[str]
) -> float:
    """上下文精确率：检索到的切片中真正相关的比例。"""
    if not retrieved_contexts:
        return 0.0
    reference_tokens: set[str] = set()
    for item in reference_contexts:
        reference_tokens |= content_tokens(item)
    if not reference_tokens:
        return 0.0
    relevant = 0
    for item in retrieved_contexts:
        tokens = content_tokens(item)
        if tokens and overlap_ratio(tokens, reference_tokens) >= PRECISION_OVERLAP_THRESHOLD:
            relevant += 1
    return relevant / len(retrieved_contexts)


def context_recall(retrieved_contexts: Sequence[str], reference_contexts: Sequence[str]) -> float:
    """上下文召回率：参考要点被检索内容覆盖的比例。"""
    reference_tokens: set[str] = set()
    for item in reference_contexts:
        reference_tokens |= content_tokens(item)
    if not reference_tokens:
        return 0.0
    retrieved_tokens: set[str] = set()
    for item in retrieved_contexts:
        retrieved_tokens |= content_tokens(item)
    return overlap_ratio(reference_tokens, retrieved_tokens)


def faithfulness(answer: str, retrieved_contexts: Sequence[str]) -> float:
    """忠实度：答案中每个「有实质内容」的句子，其要点能被检索上下文覆盖的比例。

    引用编号、过短片段（少于 3 个内容词）会被忽略，避免污染均值。
    """
    sentences = [
        _CITATION_MARK.sub("", item).strip()
        for item in _SENTENCE_SPLIT.split(answer or "")
    ]
    context_tokens: set[str] = set()
    for item in retrieved_contexts:
        context_tokens |= content_tokens(item)
    if not context_tokens:
        return 0.0
    scores: list[float] = []
    for sentence in sentences:
        tokens = content_tokens(sentence)
        if len(tokens) < 3:
            continue
        scores.append(min(1.0, overlap_ratio(tokens, context_tokens)))
    return sum(scores) / len(scores) if scores else 0.0


def answer_relevancy(answer: str, question: str, ground_truth: str = "") -> float:
    """答案相关性：答案与「问题 + 标准答案」的要点 F1。"""
    target = content_tokens(f"{question} {ground_truth}")
    if not target:
        return 0.0
    return f1_overlap(content_tokens(answer), target)


@dataclass
class SampleMetrics:
    """单样本指标。"""

    question: str = ""
    category: str = ""
    strategy: str = ""
    faithfulness: float = 0.0
    answer_relevancy: float = 0.0
    context_precision: float = 0.0
    context_recall: float = 0.0
    latency_ms: int = 0
    citations: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        """转为扁平字典（便于写 CSV）。"""
        return {
            "question": self.question,
            "category": self.category,
            "strategy": self.strategy,
            "faithfulness": round(self.faithfulness, 4),
            "answer_relevancy": round(self.answer_relevancy, 4),
            "context_precision": round(self.context_precision, 4),
            "context_recall": round(self.context_recall, 4),
            "latency_ms": self.latency_ms,
        }


def evaluate_sample(
    *,
    question: str,
    answer: str,
    retrieved_contexts: Sequence[str],
    ground_truth: str,
    reference_contexts: Sequence[str] | None = None,
    category: str = "",
    strategy: str = "",
    latency_ms: int = 0,
) -> SampleMetrics:
    """对单条样本计算 4 个指标。"""
    reference = list(reference_contexts or [ground_truth])
    return SampleMetrics(
        question=question,
        category=category,
        strategy=strategy,
        faithfulness=faithfulness(answer, retrieved_contexts),
        answer_relevancy=answer_relevancy(answer, question, ground_truth),
        context_precision=context_precision(retrieved_contexts, reference),
        context_recall=context_recall(retrieved_contexts, reference),
        latency_ms=latency_ms,
    )


def aggregate(samples: Sequence[SampleMetrics]) -> dict[str, float]:
    """对一组样本取平均，得到该配置的最终指标。"""
    if not samples:
        return {
            "faithfulness": 0.0,
            "answer_relevancy": 0.0,
            "context_precision": 0.0,
            "context_recall": 0.0,
            "avg_latency_ms": 0.0,
        }
    count = len(samples)
    return {
        "faithfulness": round(sum(item.faithfulness for item in samples) / count, 4),
        "answer_relevancy": round(sum(item.answer_relevancy for item in samples) / count, 4),
        "context_precision": round(sum(item.context_precision for item in samples) / count, 4),
        "context_recall": round(sum(item.context_recall for item in samples) / count, 4),
        "avg_latency_ms": round(sum(item.latency_ms for item in samples) / count, 1),
    }

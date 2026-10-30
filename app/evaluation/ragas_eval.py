"""RAGAS 评估接入点。

- 安装了 ``ragas`` 且配置了大模型时（``LLM_API_KEY``），使用 RAGAS 官方指标：
  faithfulness / answer_relevancy / context_precision / context_recall；
- 否则自动降级为 ``app.evaluation.metrics`` 中的确定性代理指标（保证离线可复现）。

两种模式输入输出结构一致，便于论文中说明「代理指标用于离线复现、RAGAS 用于正式评测」。
"""

from __future__ import annotations

from typing import Any, Optional, Sequence

from app.core.config import get_settings
from app.core.logging import get_logger
from app.evaluation.metrics import aggregate, evaluate_sample

logger = get_logger(__name__)

RAGAS_METRIC_NAMES = [
    "faithfulness",
    "answer_relevancy",
    "context_precision",
    "context_recall",
]


def ragas_available() -> bool:
    """检测 ragas 是否可导入。"""
    try:
        import ragas  # noqa: F401

        return True
    except ImportError:
        return False


def _build_ragas_metrics() -> list[Any]:
    """构造 RAGAS 指标对象（不同版本导入路径兼容）。"""
    from ragas import metrics as ragas_metrics

    names = RAGAS_METRIC_NAMES
    built: list[Any] = []
    for name in names:
        metric = getattr(ragas_metrics, name, None)
        if metric is not None:
            built.append(metric)
    if not built:
        raise RuntimeError("当前 ragas 版本未提供所需指标")
    return built


async def evaluate_with_ragas(samples: Sequence[dict[str, Any]]) -> dict[str, float]:
    """使用 RAGAS 计算四个官方指标。

    :param samples: ``[{question, answer, contexts: list[str], ground_truth}]``
    """
    from datasets import Dataset  # type: ignore[import-not-found]
    from ragas import evaluate  # type: ignore[import-not-found]

    dataset = Dataset.from_dict(
        {
            "question": [item["question"] for item in samples],
            "answer": [item["answer"] for item in samples],
            "contexts": [list(item.get("contexts") or []) for item in samples],
            "ground_truth": [item.get("ground_truth", "") for item in samples],
        }
    )
    result = evaluate(dataset, metrics=_build_ragas_metrics())
    payload = result.to_pandas().mean(numeric_only=True).to_dict()
    return {name: float(payload.get(name, 0.0)) for name in RAGAS_METRIC_NAMES}


async def evaluate_samples(
    samples: Sequence[dict[str, Any]], use_ragas: Optional[bool] = None
) -> dict[str, Any]:
    """统一入口：优先 RAGAS，失败或未安装时使用代理指标。

    :param use_ragas: 显式指定是否使用 RAGAS；None 表示自动判断
    :return: ``{"mode": "ragas"|"proxy", "metrics": {...}, "samples": [...]}``
    """
    settings = get_settings()
    auto = use_ragas is None
    want_ragas = bool(use_ragas) if not auto else ragas_available()
    if auto and want_ragas and settings.llm_api_key.startswith("sk-please"):
        logger.warning("未配置真实 LLM_API_KEY，RAGAS 指标不可用，回退代理指标")
        want_ragas = False

    if want_ragas:
        try:
            metrics = await evaluate_with_ragas(samples)
            return {"mode": "ragas", "metrics": metrics, "samples": len(samples)}
        except Exception as exc:  # pragma: no cover - 依赖外部服务
            logger.warning("RAGAS 评估失败（%s），回退代理指标", exc)

    collected = [
        evaluate_sample(
            question=item["question"],
            answer=item.get("answer", ""),
            retrieved_contexts=list(item.get("contexts") or []),
            ground_truth=item.get("ground_truth", ""),
            category=item.get("category", ""),
            strategy=item.get("strategy", ""),
            latency_ms=int(item.get("latency_ms", 0) or 0),
        )
        for item in samples
    ]
    return {"mode": "proxy", "metrics": aggregate(collected), "samples": len(samples)}

"""使用 RAGAS 官方指标评估问答系统（真实检索 + 真实生成 + 大模型打分）。

指标：faithfulness / answer_relevancy / context_precision / context_recall。
用法：
    set LLM_API_KEY=... & python scripts/run_ragas_eval.py --limit 5 --run-id ragas5
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
from langchain_core.embeddings import Embeddings
from langchain_openai import ChatOpenAI
from sentence_transformers import SentenceTransformer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.core.config import get_settings  # noqa: E402
from app.core.logging import setup_logging  # noqa: E402
from app.evaluation.runner import CATEGORY_KB_NAME, load_dataset, resolve_category_kb_ids  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


class BgeEmbeddings(Embeddings):
    """把 bge 模型适配为 LangChain Embeddings（RAGAS answer_relevancy 需要）。"""

    def __init__(self, model_path: str) -> None:
        self._model = SentenceTransformer(model_path, device="cpu")

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        vectors = self._model.encode(texts, normalize_embeddings=True, batch_size=8)
        return np.asarray(vectors).tolist()

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]


async def collect_samples(limit: int, strategy: str, stride: int = 1) -> list[dict]:
    """跑真实问答链路，收集 RAGAS 需要的四元组。"""
    from app.chains.graph import QaPipeline

    settings = get_settings()
    records = load_dataset()[::stride][:limit]
    kb_map = await resolve_category_kb_ids()
    pipeline = QaPipeline()
    samples: list[dict] = []
    for index, record in enumerate(records, start=1):
        kb_id = kb_map.get(record.category) or record.kb_id
        result = await pipeline.aanswer(
            record.question, kb_ids=[kb_id] if kb_id else None, strategy=strategy
        )
        samples.append(
            {
                "user_input": record.question,
                "response": result.answer,
                "retrieved_contexts": [chunk.content for chunk in result.chunks],
                "reference": record.ground_truth,
                "category": record.category,
                "blocked": result.blocked,
                "citations": result.citations,
            }
        )
        print(f"  [{index}/{len(records)}] {record.category} 生成完成（{len(result.chunks)} 个上下文）", flush=True)
    return samples


def run_ragas(samples: list[dict], settings) -> dict:
    """调用 RAGAS 官方指标。"""
    from ragas import evaluate
    from ragas.dataset_schema import EvaluationDataset
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from ragas.llms import LangchainLLMWrapper
    from ragas.metrics import answer_relevancy, context_precision, context_recall, faithfulness

    dataset = EvaluationDataset.from_list(
        [
            {
                "user_input": item["user_input"],
                "response": item["response"],
                "retrieved_contexts": item["retrieved_contexts"],
                "reference": item["reference"],
            }
            for item in samples
        ]
    )
    llm = LangchainLLMWrapper(
        ChatOpenAI(
            model=settings.llm_model,
            api_key=settings.llm_api_key,
            base_url=settings.llm_base_url,
            temperature=0,
            timeout=180,
            max_retries=2,
        )
    )
    embeddings = LangchainEmbeddingsWrapper(BgeEmbeddings(settings.embedding_model_path))
    result = evaluate(
        dataset,
        metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
        llm=llm,
        embeddings=embeddings,
        show_progress=False,
        raise_exceptions=False,
    )
    return result


async def main() -> int:
    """执行评估并保存 CSV / JSON。"""
    parser = argparse.ArgumentParser(description="RAGAS 官方指标评估")
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--stride", type=int, default=1, help="抽样步长（6 表示每类取一条）")
    parser.add_argument("--strategy", default="hybrid_rerank")
    parser.add_argument("--run-id", default=None)
    args = parser.parse_args()
    setup_logging()
    settings = get_settings()
    run_id = args.run_id or datetime.now().strftime("%Y%m%d_%H%M%S")

    print(f"收集样本：limit={args.limit}, strategy={args.strategy}")
    samples = await collect_samples(args.limit, args.strategy, args.stride)
    print("调用 RAGAS 官方指标（大模型打分，可能需要数分钟）")
    result = run_ragas(samples, settings)
    frame = result.to_pandas()
    out_dir = ROOT / "app" / "evaluation" / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / f"ragas_{run_id}.csv"
    detail_path = out_dir / f"ragas_{run_id}_samples.json"
    frame.to_csv(csv_path, index=False, encoding="utf-8-sig")
    detail_path.write_text(json.dumps(samples, ensure_ascii=False, indent=2), encoding="utf-8")

    metric_cols = [c for c in frame.columns if c in {"faithfulness", "answer_relevancy", "context_precision", "context_recall"}]
    means = {col: float(frame[col].mean()) for col in metric_cols}
    print("\nRAGAS 平均值：")
    for key, value in means.items():
        print(f"  {key}: {value:.4f}")
    print(f"\n明细：{csv_path}\n原始样本：{detail_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))

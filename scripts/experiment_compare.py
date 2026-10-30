"""检索策略对比实验（论文第 6 章数据来源）。

用法示例：
    python scripts/experiment_compare.py --limit 5                 # 小样本试跑（离线模式）
    python scripts/experiment_compare.py --limit 60 --mode service # 完整环境跑正式数据
    python scripts/experiment_compare.py --strategies hybrid_rerank --mode offline
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import sys
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.core.logging import setup_logging  # noqa: E402
from app.evaluation.runner import DEFAULT_STRATEGIES, RESULTS_DIR, run_comparison  # noqa: E402


async def main() -> int:
    """脚本入口。"""
    parser = argparse.ArgumentParser(description="A/B/C/D 四组检索配置对比实验")
    parser.add_argument("--dataset", default="finance_qa_eval.jsonl", help="评测集文件名")
    parser.add_argument("--limit", type=int, default=None, help="只跑前 N 条（小样本试跑）")
    parser.add_argument(
        "--strategies",
        nargs="*",
        default=DEFAULT_STRATEGIES,
        help="参与对比的策略：vector bm25 hybrid hybrid_rerank",
    )
    parser.add_argument(
        "--mode",
        choices=["auto", "offline", "service"],
        default="auto",
        help="auto：优先真实链路，依赖不可用时降级离线；offline：哈希向量离线复现",
    )
    parser.add_argument("--run-id", default=None, help="实验标识，默认随机 8 位")
    parser.add_argument("--out", default=None, help="结果目录，默认 app/evaluation/results")
    parser.add_argument(
        "--proxy-rerank",
        action="store_true",
        help="service 模式下用词重叠代理重排（未下载交叉编码器模型时使用）",
    )
    parser.add_argument(
        "--kb-scope",
        choices=["record", "all"],
        default="record",
        help="record：按样本分类检索对应知识库；all：全库检索（混合语料场景）",
    )
    args = parser.parse_args()

    setup_logging()
    run_id = args.run_id or uuid.uuid4().hex[:8]
    output_dir = Path(args.out) if args.out else None

    def _progress(done: int, total: int, message: str) -> None:
        """打印进度（每 10% 一次，避免刷屏）。"""
        if total and (done == total or done % max(1, total // 10) == 0):
            print(f"[progress] {done}/{total} - {message}", flush=True)

    result = await run_comparison(
        run_id=run_id,
        dataset_name=args.dataset,
        limit=args.limit,
        strategies=args.strategies,
        progress=_progress,
        mode=args.mode,
        output_dir=output_dir,
        proxy_rerank=args.proxy_rerank,
        kb_scope=args.kb_scope,
    )

    csv_path = Path(result["csv_path"])
    print(f"\n模式：{result['mode']}　结果文件：{csv_path}")
    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle)
        for index, row in enumerate(reader):
            if index > 5:
                break
            print(" | ".join(str(item) for item in row))
    print(f"\n逐样本明细：{result['detail_path']}")
    print(f"对比柱状图：{result.get('chart_path')}")
    print(f"（结果目录：{csv_path.parent or RESULTS_DIR}）")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))

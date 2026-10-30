"""切片参数敏感性实验：chunk_size × overlap 网格。

用法：
    python scripts/experiment_chunk.py --limit 6
    python scripts/experiment_chunk.py --limit 60 --chunk-sizes 256 512 1024 --overlaps 32 64 128
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.core.logging import setup_logging  # noqa: E402
from app.evaluation.runner import run_chunk_grid  # noqa: E402


async def main() -> int:
    """脚本入口。"""
    parser = argparse.ArgumentParser(description="切片参数对比实验")
    parser.add_argument("--limit", type=int, default=10, help="使用的评测样本数")
    parser.add_argument("--chunk-sizes", nargs="*", type=int, default=[256, 512, 1024])
    parser.add_argument("--overlaps", nargs="*", type=int, default=[32, 64, 128])
    parser.add_argument("--out", default=None, help="结果目录")
    args = parser.parse_args()

    setup_logging()
    result = await run_chunk_grid(
        chunk_sizes=args.chunk_sizes,
        overlaps=args.overlaps,
        limit=args.limit,
        output_dir=Path(args.out) if args.out else None,
    )
    path = Path(result["csv_path"])
    print(f"\n切片参数实验结果：{path}\n")
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle)
        for row in reader:
            print(" | ".join(str(item) for item in row))
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))

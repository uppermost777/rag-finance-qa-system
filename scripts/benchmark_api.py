"""接口响应时间采样脚本（性能测试用，无额外依赖）。

用法：
    python scripts/benchmark_api.py --endpoint /health --times 20
    python scripts/benchmark_api.py --endpoint /chat/completions --times 30 \
        --username student_lin --password Student@2026 \
        --payload "{\"question\": \"什么是基金定投\"}"
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import httpx  # noqa: E402


def login(base_url: str, username: str, password: str) -> str:
    """登录并返回 JWT。"""
    with httpx.Client(base_url=base_url, timeout=30) as client:
        resp = client.post(
            "/api/v1/auth/login", json={"username": username, "password": password}
        )
        resp.raise_for_status()
        return resp.json()["data"]["access_token"]


def main() -> int:
    """脚本入口：打印 min/avg/p50/p95/max 与错误数。"""
    parser = argparse.ArgumentParser(description="接口响应时间采样")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--endpoint", default="/health", help="相对路径，如 /chat/completions")
    parser.add_argument("--method", choices=["GET", "POST"], default="POST")
    parser.add_argument("--times", type=int, default=20, help="采样次数")
    parser.add_argument("--username", default=None, help="需要鉴权时提供")
    parser.add_argument("--password", default=None)
    parser.add_argument("--payload", default=None, help="POST 请求体（JSON 字符串）")
    args = parser.parse_args()

    headers = {}
    if args.username and args.password:
        headers["Authorization"] = f"Bearer {login(args.base_url, args.username, args.password)}"
    body = json.loads(args.payload) if args.payload else None

    samples: list[float] = []
    errors = 0
    with httpx.Client(base_url=args.base_url, timeout=120) as client:
        for _ in range(args.times):
            started = time.perf_counter()
            try:
                if args.method == "GET":
                    resp = client.get(args.endpoint, headers=headers)
                else:
                    resp = client.post(args.endpoint, json=body, headers=headers)
                if resp.status_code >= 400:
                    errors += 1
            except Exception as exc:  # 网络异常计入错误数
                errors += 1
                print(f"请求异常：{exc}")
            samples.append((time.perf_counter() - started) * 1000)

    samples.sort()
    index_p95 = min(len(samples) - 1, int(len(samples) * 0.95))
    print(f"接口：{args.method} {args.endpoint}　采样 {len(samples)} 次　错误 {errors} 次")
    print(f"min={samples[0]:.1f}ms  avg={statistics.mean(samples):.1f}ms  "
          f"p50={statistics.median(samples):.1f}ms  p95={samples[index_p95]:.1f}ms  "
          f"max={samples[-1]:.1f}ms")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

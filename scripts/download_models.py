"""下载 bge-m3 与 bge-reranker-v2-m3 到本地 models/（支持断点续传，仅用标准库）。

服务器（阿里云）可直接运行：
    python3 scripts/download_models.py --dest models --mirror https://hf-mirror.com

模型体积约 4.4GB，镜像站速度通常好于 huggingface.co。
"""

from __future__ import annotations

import argparse
import http.client
import time
import urllib.error
import urllib.request
from pathlib import Path

CHUNK = 4 * 1024 * 1024

MODELS: dict[str, list[str]] = {
    "bge-m3": [
        "config.json",
        "config_sentence_transformers.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "sentence_bert_config.json",
        "modules.json",
        "1_Pooling/config.json",
        "sentencepiece.bpe.model",
        "pytorch_model.bin",
    ],
    "bge-reranker-v2-m3": [
        "config.json",
        "model.safetensors",
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "sentencepiece.bpe.model",
    ],
}

REPOS = {
    "bge-m3": "BAAI/bge-m3",
    "bge-reranker-v2-m3": "BAAI/bge-reranker-v2-m3",
}


def remote_size(url: str) -> int:
    """HEAD 获取文件大小。"""
    request = urllib.request.Request(url, method="HEAD", headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(request, timeout=60) as response:
        return int(response.headers.get("Content-Length") or 0)


def download(url: str, target: Path, retries: int = 30) -> bool:
    """断点续传下载单个文件。"""
    target.parent.mkdir(parents=True, exist_ok=True)
    total = 0
    for _ in range(5):
        try:
            total = remote_size(url)
            break
        except Exception:
            time.sleep(2)

    for attempt in range(1, retries + 1):
        existing = target.stat().st_size if target.exists() else 0
        if total and existing >= total:
            return True
        headers = {"User-Agent": "Mozilla/5.0"}
        if existing:
            headers["Range"] = f"bytes={existing}-"
        try:
            with urllib.request.urlopen(
                urllib.request.Request(url, headers=headers), timeout=120
            ) as response:
                mode = "ab" if existing and response.status == 206 else "wb"
                written = existing if mode == "ab" else 0
                started = time.time()
                with target.open(mode) as handle:
                    while True:
                        chunk = response.read(CHUNK)
                        if not chunk:
                            break
                        handle.write(chunk)
                        written += len(chunk)
                        if total and written % (128 * 1024 * 1024) < CHUNK:
                            speed = written / 1048576 / max(time.time() - started, 1e-6)
                            print(
                                f"    {target.name}: {written/1048576:.0f}/{total/1048576:.0f} MB "
                                f"({speed:.2f} MB/s)",
                                flush=True,
                            )
        except (urllib.error.URLError, http.client.HTTPException, OSError) as exc:
            print(f"    第 {attempt} 次中断：{type(exc).__name__} {str(exc)[:80]}", flush=True)
        if total and target.exists() and target.stat().st_size >= total:
            print(f"    {target.name} 完成（{target.stat().st_size/1048576:.1f} MB）", flush=True)
            return True
        time.sleep(min(3 + attempt, 10))
    return False


def main() -> int:
    """入口。"""
    parser = argparse.ArgumentParser(description="下载本地嵌入与重排模型")
    parser.add_argument("--dest", default="models")
    parser.add_argument("--mirror", default="https://hf-mirror.com")
    parser.add_argument(
        "--models",
        nargs="*",
        default=list(MODELS),
        choices=list(MODELS),
        help="要下载的模型",
    )
    args = parser.parse_args()

    dest = Path(args.dest)
    ok = True
    for name in args.models:
        repo = REPOS[name]
        print(f"== {name}（{repo}）")
        for filename in MODELS[name]:
            url = f"{args.mirror}/{repo}/resolve/main/{filename}"
            print(f"  -> {filename}", flush=True)
            if not download(url, dest / name / filename):
                print(f"  !! {filename} 未完成，可重跑续传", flush=True)
                ok = False
    print("全部完成" if ok else "存在未完成文件，请重跑本脚本")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())

"""构建细粒度评测语料：把 60 条标准答案扩写成 6 张独立知识卡片。

设计目标：让同一分类内部存在多个"相似但不同"的切片，使检索策略的差异可以被观测到。
事实约束：模型只能复述要点中已有的事实与数字，不得新增数字/机构名/绝对化结论；
脚本会校验生成文本中的数字必须出现在原始要点中，不合规的卡片会被丢弃并重试。

用法：
    set LLM_API_KEY=...            # 仅环境变量，不写入文件
    python scripts/build_corpus.py --out data/corpus/cards.jsonl --concurrency 4
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import sys
from pathlib import Path
from typing import Any

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.core.logging import setup_logging  # noqa: E402
from app.evaluation.runner import load_dataset  # noqa: E402

PROMPT = (
    "你是理财知识编辑。根据给定的【主题】和【要点】，写 {n} 张彼此独立的知识卡片，"
    "分别从以下角度展开：概念解释、计算方法或操作步骤、常见误区、适用场景、风险边界、一个简短的对比说明。\n"
    "硬性要求：\n"
    "1. 只能使用【要点】中已有的事实、数字和结论，严禁新增任何数字、机构名、产品名或绝对化结论；\n"
    "2. 每张卡片正文 70~140 字，独立成文，读者不看上下文也能理解；\n"
    "3. 每张卡片有自己的标题（8~18 字），标题要包含具体概念名，便于区分；\n"
    "4. 只输出 JSON 数组：[{{\"title\": \"...\", \"body\": \"...\"}}]，不要输出任何解释或代码围栏。"
)

_DIGITS = re.compile(r"\d+(?:\.\d+)?")


def extract_digits(text: str) -> set[str]:
    """抽取文本中的数字（统一去掉小数点后的尾零）。"""
    result: set[str] = set()
    for item in _DIGITS.findall(text):
        try:
            result.add(f"{float(item):g}")
        except ValueError:
            result.add(item)
    return result


async def generate_cards(
    client: httpx.AsyncClient, base_url: str, model: str, record: Any, per_question: int
) -> list[dict[str, str]]:
    """为一条评测样本生成卡片并做事实校验。"""
    source_digits = extract_digits(record.question + record.ground_truth)
    messages = [
        {"role": "system", "content": PROMPT.format(n=per_question)},
        {
            "role": "user",
            "content": f"【主题】{record.question}\n【要点】{record.ground_truth}\n【分类】{record.category}",
        },
    ]
    for _ in range(3):
        response = await client.post(
            f"{base_url}/chat/completions",
            json={
                "model": model,
                "messages": messages,
                "temperature": 0.6,
                "max_tokens": 1800,
            },
            timeout=180,
        )
        response.raise_for_status()
        raw = response.json()["choices"][0]["message"]["content"].strip()
        raw = raw.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
        start, end = raw.find("["), raw.rfind("]")
        if start < 0 or end <= start:
            continue
        try:
            items = json.loads(raw[start : end + 1])
        except json.JSONDecodeError:
            continue
        cards: list[dict[str, str]] = []
        for item in items:
            if not isinstance(item, dict):
                continue
            title = str(item.get("title", "")).strip()
            body = str(item.get("body", "")).strip()
            if not (6 <= len(title) <= 30 and 40 <= len(body) <= 260):
                continue
            new_digits = extract_digits(body + title) - source_digits
            if new_digits:  # 出现要点中没有的数字 → 判定为可能编造
                continue
            cards.append({"title": title, "body": body})
        if len(cards) >= max(3, per_question - 2):
            return cards[:per_question]
    return []


async def main() -> int:
    """并发生成所有卡片。"""
    parser = argparse.ArgumentParser(description="生成细粒度知识卡片语料")
    parser.add_argument("--out", default="data/corpus/cards.jsonl")
    parser.add_argument("--per-question", type=int, default=6)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    setup_logging()
    key = os.environ.get("LLM_API_KEY", "")
    base_url = os.environ.get("LLM_BASE_URL", "https://api.deepseek.com/v1")
    model = os.environ.get("LLM_MODEL", "deepseek-chat")
    if not key:
        print("缺少 LLM_API_KEY 环境变量")
        return 2

    records = load_dataset(limit=args.limit)
    semaphore = asyncio.Semaphore(args.concurrency)
    results: list[dict[str, Any]] = []
    print(f"开始生成：{len(records)} 条 × {args.per_question} 张卡片")

    async with httpx.AsyncClient(headers={"Authorization": f"Bearer {key}"}) as client:
        async def worker(index: int, record: Any) -> None:
            async with semaphore:
                cards = await generate_cards(client, base_url, model, record, args.per_question)
                for position, card in enumerate(cards):
                    results.append(
                        {
                            "card_id": f"{record.id:03d}-{position + 1}",
                            "source_id": record.id,
                            "category": record.category,
                            "kb_id": record.kb_id,
                            "question": record.question,
                            "title": card["title"],
                            "body": card["body"],
                        }
                    )
                print(f"  {index + 1}/{len(records)} {record.category} -> {len(cards)} 张", flush=True)

        await asyncio.gather(*(worker(i, record) for i, record in enumerate(records)))

    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for item in sorted(results, key=lambda row: row["card_id"]):
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"完成：{len(results)} 张卡片 -> {output}")
    if len(results) < len(records) * (args.per_question - 2):
        print("警告：部分样本未生成足够卡片，可重跑补齐")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))

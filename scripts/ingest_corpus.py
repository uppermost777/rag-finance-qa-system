"""把细粒度知识卡片灌入 6 个分类知识库（一卡一文档，便于溯源与切片区分）。

用法（需先 build_corpus.py 生成 data/corpus/cards.jsonl）：
    python scripts/ingest_corpus.py --cards data/corpus/cards.jsonl
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sqlalchemy import select  # noqa: E402

from app.core.database import get_session_factory  # noqa: E402
from app.core.logging import setup_logging  # noqa: E402
from app.evaluation.runner import CATEGORY_KB_NAME  # noqa: E402
from app.ingestion.pipeline import ingest_document  # noqa: E402
from app.models import Document, KnowledgeBase  # noqa: E402
from app.services.document_service import DocumentService, sanitize_filename  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DOCS_DIR = ROOT / "data" / "corpus" / "docs"


async def main() -> int:
    """清空分类库旧文档 → 写入卡片文件 → 入库。"""
    parser = argparse.ArgumentParser(description="灌入细粒度知识卡片")
    parser.add_argument("--cards", default="data/corpus/cards.jsonl")
    parser.add_argument("--keep-existing", action="store_true", help="保留知识库中已有的文档")
    args = parser.parse_args()
    setup_logging()

    cards = [
        json.loads(line)
        for line in Path(args.cards).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    print(f"卡片数：{len(cards)}，分类分布：{dict(Counter(card['category'] for card in cards))}")

    factory = get_session_factory()
    async with factory() as session:
        kbs = {
            kb.name: kb
            for kb in (await session.scalars(select(KnowledgeBase))).all()
        }
        missing = [name for name in CATEGORY_KB_NAME.values() if name not in kbs]
        if missing:
            print(f"缺少知识库（先运行 seed_data.py）：{missing}")
            return 2

        if not args.keep_existing:
            for category, kb_name in CATEGORY_KB_NAME.items():
                kb = kbs[kb_name]
                doc_ids = [
                    int(item)
                    for item in (
                        await session.scalars(select(Document.id).where(Document.kb_id == kb.id))
                    ).all()
                ]
                service = DocumentService(session)
                for doc_id in doc_ids:
                    await service.delete_document(doc_id)
                print(f"已清空 {category}：{len(doc_ids)} 个旧文档")

        created: list[int] = []
        for index, card in enumerate(cards, start=1):
            category = card["category"]
            kb = kbs[CATEGORY_KB_NAME[category]]
            title = sanitize_filename(card["title"])[:40]
            filename = f"{card['card_id']}_{title}.md"
            content = f"# {card['title']}\n\n{card['body']}\n".encode("utf-8")
            service = DocumentService(session)
            document = await service.create_document(
                kb_id=kb.id, filename=filename, content=content
            )
            created.append(document.id)
            if index % 20 == 0:
                print(f"  已创建 {index}/{len(cards)} 个文档", flush=True)

    print(f"开始入库 {len(created)} 个文档（真实嵌入 + 向量写入）")
    done = 0
    failed = 0
    for index, doc_id in enumerate(created, start=1):
        result = await ingest_document(doc_id)
        if result.status == "done":
            done += 1
        else:
            failed += 1
            print(f"  doc {doc_id} 失败：{result.error_msg}")
        if index % 20 == 0:
            print(f"  入库 {index}/{len(created)}（成功 {done}，失败 {failed}）", flush=True)
    print(f"完成：成功 {done}，失败 {failed}")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))

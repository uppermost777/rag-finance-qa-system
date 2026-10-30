"""把评测集按分类写入 6 个种子知识库，供 service 模式的对比实验使用。

本地联调/答辩演示时，知识库需要真实文档。本脚本把 ``finance_qa_eval.jsonl`` 中每个分类的
10 条标准答案拼成一篇"知识汇编"文档，上传到同名知识库并完成入库（真实嵌入 + 向量库）。

用法：
    python scripts/prepare_eval_kb.py
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sqlalchemy import select  # noqa: E402

from app.core.database import get_session_factory  # noqa: E402
from app.core.logging import setup_logging  # noqa: E402
from app.evaluation.runner import CATEGORY_KB_NAME, load_dataset  # noqa: E402
from app.ingestion.pipeline import ingest_document  # noqa: E402
from app.models import Document, KnowledgeBase  # noqa: E402
from app.services.document_service import DocumentService  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "work_logs" / "eval_kb_docs"

async def main() -> int:
    """生成文档 → 上传 → 入库。"""
    setup_logging()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    records = load_dataset()
    grouped: dict[str, list[dict]] = {}
    for record in records:
        grouped.setdefault(record.category, []).append(
            {"question": record.question, "answer": record.ground_truth}
        )

    factory = get_session_factory()
    async with factory() as session:
        for category, items in grouped.items():
            kb_name = CATEGORY_KB_NAME.get(category)
            if not kb_name:
                print(f"跳过未知分类：{category}")
                continue
            kb = await session.scalar(select(KnowledgeBase).where(KnowledgeBase.name == kb_name))
            if kb is None:
                print(f"知识库不存在：{kb_name}（请先运行 seed_data.py）")
                continue

            text = f"{kb_name} · 知识汇编\n\n" + "\n\n".join(
                f"【问题】{item['question']}\n【要点】{item['answer']}" for item in items
            )
            filename = f"{category}_知识汇编.txt"
            path = OUT_DIR / filename
            path.write_text(text, encoding="utf-8")

            existing = await session.scalar(
                select(Document).where(Document.kb_id == kb.id, Document.name == filename)
            )
            if existing is not None:
                doc_id = existing.id
                print(f"{kb_name}：文档已存在 doc_id={doc_id}，重建索引")
            else:
                service = DocumentService(session)
                document = await service.create_document(
                    kb_id=kb.id, filename=filename, content=path.read_bytes()
                )
                doc_id = document.id
                print(f"{kb_name}：已上传 doc_id={doc_id}")

            result = await ingest_document(doc_id)
            print(
                f"  入库 {result.status}：{result.chunk_count} 个切片"
                + (f"（{result.error_msg}）" if result.error_msg else "")
            )
    print("评测知识库准备完成")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))

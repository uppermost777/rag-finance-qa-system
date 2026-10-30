"""初始化数据库：创建 ``finance_qa`` 库（若不存在）与全部业务表（幂等）。

用法：
    python scripts/init_db.py            # 建表
    python scripts/init_db.py --drop     # 先删后建（仅开发环境使用）
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sqlalchemy import text  # noqa: E402

from app.core.config import get_settings  # noqa: E402
from app.core.logging import get_logger, setup_logging  # noqa: E402
from app.models import Base  # noqa: E402  # 导入即注册全部模型

logger = get_logger("scripts.init_db")


async def _ensure_database() -> None:
    """若目标 MySQL 库不存在则创建（SQLite 跳过）。"""
    from sqlalchemy.ext.asyncio import create_async_engine

    settings = get_settings()
    if settings.database_url.startswith("sqlite"):
        return
    # 连接到 server 级（无库名）以创建数据库
    server_url = settings.database_url.rsplit("/", 1)[0] + "/?charset=utf8mb4"
    engine = create_async_engine(server_url, isolation_level="AUTOCOMMIT")
    db_name = settings.database_url.rsplit("/", 1)[-1].split("?")[0]
    async with engine.connect() as conn:
        await conn.execute(
            text(
                f"CREATE DATABASE IF NOT EXISTS `{db_name}` "
                "DEFAULT CHARACTER SET utf8mb4 DEFAULT COLLATE utf8mb4_general_ci"
            )
        )
    await engine.dispose()
    logger.info("数据库 %s 已就绪", db_name)


async def init_models(drop: bool = False) -> None:
    """创建全部表；drop=True 时先删除（幂等）。"""
    from sqlalchemy.ext.asyncio import create_async_engine

    settings = get_settings()
    engine = create_async_engine(settings.database_url, echo=settings.db_echo)
    async with engine.begin() as conn:
        if drop:
            await conn.run_sync(Base.metadata.drop_all)
            logger.warning("已删除全部表（--drop）")
        await conn.run_sync(Base.metadata.create_all)
    tables = ", ".join(sorted(Base.metadata.tables))
    logger.info("建表完成，共 %d 张：%s", len(Base.metadata.tables), tables)
    await engine.dispose()


async def main() -> None:
    """脚本入口。"""
    parser = argparse.ArgumentParser(description="初始化数据库表结构")
    parser.add_argument("--drop", action="store_true", help="先删除已有表（危险，仅开发用）")
    args = parser.parse_args()
    setup_logging()
    await _ensure_database()
    await init_models(drop=args.drop)


if __name__ == "__main__":
    asyncio.run(main())


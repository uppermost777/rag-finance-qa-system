"""Alembic 环境配置：连接信息从环境变量 / Settings 读取，支持 SQLite 与 MySQL。"""

from __future__ import annotations

import os
import sys
from logging.config import fileConfig
from pathlib import Path

from alembic import context
from sqlalchemy import engine_from_config, pool

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.core.config import get_settings  # noqa: E402
from app.models import Base  # noqa: E402  # 导入即注册全部模型

config = context.config
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

target_metadata = Base.metadata


def _database_url() -> str:
    """解析迁移使用的同步数据库 URL（优先 ALEMBIC_DATABASE_URL）。"""
    url = os.environ.get("ALEMBIC_DATABASE_URL") or get_settings().sync_database_url
    # 异步驱动无法被 Alembic 直接使用，替换为同步驱动
    return (
        url.replace("mysql+aiomysql", "mysql+pymysql")
        .replace("sqlite+aiosqlite", "sqlite")
    )


def run_migrations_offline() -> None:
    """离线模式：只生成 SQL 脚本。"""
    context.configure(
        url=_database_url(),
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        compare_type=True,
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """在线模式：连接数据库执行迁移。"""
    section = config.get_section(config.config_ini_section) or {}
    section["sqlalchemy.url"] = _database_url()
    connectable = engine_from_config(section, prefix="sqlalchemy.", poolclass=pool.NullPool)
    with connectable.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
            compare_type=True,
            render_as_batch=connection.dialect.name == "sqlite",
        )
        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()

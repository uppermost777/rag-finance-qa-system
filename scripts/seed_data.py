"""灌入种子数据：1 个管理员、2 个学生用户、6 个知识库、10 条风控规则（幂等）。

6 个知识库对应评测集的 6 大分类（储蓄存款 / 基金投资 / 股票基础 / 保险保障 / 信用借贷 / 反诈防骗），
其主键顺序即 ``app/evaluation/datasets/finance_qa_eval.jsonl`` 中 ``kb_id`` 的取值依据。

用法：python scripts/seed_data.py
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sqlalchemy import select  # noqa: E402

from app.core.config import get_settings  # noqa: E402
from app.core.logging import get_logger, setup_logging  # noqa: E402
from app.core.security import hash_password  # noqa: E402
from app.guard.default_rules import DEFAULT_RISK_RULES  # noqa: E402
from app.models import KnowledgeBase, RiskRule, User  # noqa: E402

logger = get_logger("scripts.seed_data")

SEED_USERS = [
    {
        "username": "admin",
        "email": "admin@example.com",
        "password": "Admin@2026",
        "role": "admin",
    },
    {
        "username": "student_lin",
        "email": "lin@example.com",
        "password": "Student@2026",
        "role": "student",
    },
    {
        "username": "student_zhao",
        "email": "zhao@example.com",
        "password": "Student@2026",
        "role": "student",
    },
]

SEED_KNOWLEDGE_BASES = [
    {
        "name": "银行储蓄与存款保险",
        "category": "储蓄存款",
        "description": "活期/定期/大额存单、利率换算、存款保险条例等基础知识",
    },
    {
        "name": "基金与指数投资基础",
        "category": "基金投资",
        "description": "基金分类、费用结构、定投与指数投资、净值与收益计算",
    },
    {
        "name": "个人信用与借贷风险",
        "category": "信用借贷",
        "description": "征信记录、信用卡与分期、校园贷风险、借贷成本核算",
    },
    {
        "name": "股票投资基础",
        "category": "股票基础",
        "description": "股票是什么、市盈率与估值、交易规则、波动与风险",
    },
    {
        "name": "保险与保障规划",
        "category": "保险保障",
        "description": "社保与商业保险、医疗险与意外险、投保原则与常见坑",
    },
    {
        "name": "反诈与金融安全",
        "category": "反诈防骗",
        "description": "刷单返利、校园贷、杀猪盘、虚假投资平台识别与应对",
    },
]


async def _seed_users(session) -> dict[str, User]:  # noqa: ANN001 - 脚本内部使用
    """写入用户，返回 username -> User 映射。"""
    created: dict[str, User] = {}
    for item in SEED_USERS:
        existing = await session.scalar(select(User).where(User.username == item["username"]))
        if existing is not None:
            created[item["username"]] = existing
            continue
        user = User(
            username=item["username"],
            email=item["email"],
            password_hash=hash_password(item["password"]),
            role=item["role"],
            status=1,
        )
        session.add(user)
        created[item["username"]] = user
    await session.flush()
    return created


async def _seed_kbs(session, owner_id: int) -> None:  # noqa: ANN001 - 脚本内部使用
    """写入 3 个知识库。"""
    for item in SEED_KNOWLEDGE_BASES:
        existing = await session.scalar(
            select(KnowledgeBase).where(KnowledgeBase.name == item["name"])
        )
        if existing is not None:
            continue
        session.add(
            KnowledgeBase(
                name=item["name"],
                category=item["category"],
                description=item["description"],
                owner_id=owner_id,
                status=1,
            )
        )
    await session.flush()


async def _seed_rules(session) -> None:  # noqa: ANN001 - 脚本内部使用
    """写入 10 条风控规则（按 rule_code 幂等）。"""
    for item in DEFAULT_RISK_RULES:
        existing = await session.scalar(
            select(RiskRule).where(RiskRule.rule_code == item["rule_code"])
        )
        if existing is not None:
            continue
        session.add(RiskRule(**item))
    await session.flush()


async def main() -> None:
    """脚本入口：创建表（若不存在）并灌入种子数据。"""
    setup_logging()
    from scripts.init_db import init_models

    await init_models(drop=False)

    from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

    settings = get_settings()
    engine = create_async_engine(settings.database_url)
    factory = async_sessionmaker(bind=engine, expire_on_commit=False)
    async with factory() as session:
        users = await _seed_users(session)
        await _seed_kbs(session, owner_id=users["admin"].id)
        await _seed_rules(session)
        await session.commit()
    await engine.dispose()
    logger.info(
        "种子数据完成：用户 %d 个、知识库 %d 个、风控规则 %d 条",
        len(SEED_USERS),
        len(SEED_KNOWLEDGE_BASES),
        len(DEFAULT_RISK_RULES),
    )


if __name__ == "__main__":
    asyncio.run(main())


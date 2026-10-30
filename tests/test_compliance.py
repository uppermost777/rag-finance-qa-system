"""P8 验收：合规风控（输入侧三级动作 + 输出侧声明 + 规则加载）。"""

from __future__ import annotations

from typing import Any

import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.guard.compliance import (
    BLOCK_MESSAGE,
    DISCLAIMER,
    ComplianceGuard,
    build_guard,
)
from app.guard.default_rules import DEFAULT_RISK_RULES
from app.models import Base, RiskRule


@pytest.fixture
def guard() -> ComplianceGuard:
    """内置默认规则构造的守卫（不依赖 DB / Redis）。"""
    return build_guard(DEFAULT_RISK_RULES)


# 用例表（编号 / 输入 / 期望动作）——与交付说明中的用例表一致
BLOCK_CASES = [
    (1, "帮我推荐一只稳赚的基金"),
    (2, "现在买什么股票能翻倍"),
    (3, "保证收益的产品有哪些"),
    (4, "有没有内幕消息可以透露"),
    (5, "借钱炒股划算吗"),
    (6, "我想加杠杆买股票，怎么操作"),
    (7, "介绍个年化 30% 的原始股项目"),
    (8, "哪个平台配资利率低"),
]

REWRITE_CASES = [
    (9, "我该买哪只基金"),
    (10, "哪个理财产品最好"),
    (11, "什么理财产品收益高"),
]

PASS_CASES = [
    (12, "什么是基金定投"),
    (13, "存款保险保多少"),
    (14, "七日年化收益率怎么算"),
    (15, "征信逾期记录会保留多久"),
    (16, "货币基金和债券基金有什么区别"),
]


@pytest.mark.parametrize("case_id,question", BLOCK_CASES)
def test_block_cases(guard: ComplianceGuard, case_id: int, question: str) -> None:
    """应 BLOCK：荐股荐基、承诺收益、内幕交易、杠杆配资等违规请求。"""
    result = guard.check(question)
    assert result.action == "BLOCK", f"用例 {case_id} 期望 BLOCK，实际 {result.action}"
    assert result.message == BLOCK_MESSAGE
    assert result.matched_rules
    # 拒答文案本身也必须合规：不含任何产品推荐与收益承诺
    for forbidden in ("保证收益", "稳赚不赔"):
        assert forbidden not in (result.message or "").replace("涉及投资建议或承诺收益", "")


@pytest.mark.parametrize("case_id,question", REWRITE_CASES)
def test_rewrite_cases(guard: ComplianceGuard, case_id: int, question: str) -> None:
    """应 REWRITE：把「买哪个」改写为「如何评估」，保留学习价值。"""
    result = guard.check(question)
    assert result.action == "REWRITE", f"用例 {case_id} 期望 REWRITE，实际 {result.action}"
    assert result.rewritten_query
    assert "如何" in result.rewritten_query
    assert result.effective_query == result.rewritten_query
    assert result.rewritten_query != question


@pytest.mark.parametrize("case_id,question", PASS_CASES)
def test_pass_cases(guard: ComplianceGuard, case_id: int, question: str) -> None:
    """应 PASS：纯知识型提问，不得误拦。"""
    result = guard.check(question)
    assert result.action == "PASS", f"用例 {case_id} 期望 PASS，实际 {result.action}"
    assert result.blocked is False
    assert result.effective_query == question


def test_boundary_knowledge_question_about_guarantee(guard: ComplianceGuard) -> None:
    """边界：包含敏感词但语义为知识询问 —— 降级放行并附风险提示。"""
    result = guard.check("为什么要警惕承诺收益的产品")
    assert result.action == "PASS"
    assert result.downgraded is True
    assert result.warned is True
    assert result.warn_message
    assert DISCLAIMER in (result.warn_message or "")


def test_boundary_request_still_blocked(guard: ComplianceGuard) -> None:
    """边界：同样含敏感词，但语义是索要推荐 —— 仍然拦截。"""
    result = guard.check("帮我推荐一个承诺收益的产品")
    assert result.action == "BLOCK"


def test_warn_rule_passes_with_notice(guard: ComplianceGuard) -> None:
    """WARN 规则：放行但带风险/反诈提示。"""
    crypto = guard.check("虚拟货币值得投资吗")
    assert crypto.action == "PASS"
    assert crypto.warned is True
    assert "高风险" in (crypto.warn_message or "")

    fraud = guard.check("兼职刷单赚钱吗")
    assert fraud.action == "PASS"
    assert fraud.warned is True
    assert "诈骗" in (fraud.warn_message or "")


def test_empty_question_passes(guard: ComplianceGuard) -> None:
    """空问题不触发任何规则。"""
    result = guard.check("   ")
    assert result.action == "PASS"
    assert result.matched_rules == []


def test_output_check_appends_statement(guard: ComplianceGuard) -> None:
    """输出侧：命中「保证收益」「必涨」等表述时追加风控声明。"""
    answer, matched = guard.check_output("这只基金必涨，可以保证收益。")
    assert matched
    assert "风险提示" in answer
    assert DISCLAIMER in answer


def test_output_check_keeps_compliant_answer(guard: ComplianceGuard) -> None:
    """输出侧：合规答案原样返回，不做改动。"""
    original = "基金定投是指固定时间投入固定金额，可以平滑成本。" + DISCLAIMER
    answer, matched = guard.check_output(original)
    assert matched == []
    assert answer == original


def test_disabled_rule_not_effective() -> None:
    """enabled=0 的规则不参与匹配。"""
    rules = [
        {
            "rule_code": "BLOCK_TEST",
            "name": "测试规则",
            "pattern": r"测试敏感词",
            "action": "BLOCK",
            "category": "测试",
            "enabled": 0,
        }
    ]
    guard = build_guard(rules)
    assert guard.check("测试敏感词").action == "PASS"
    assert guard.rules == [] or all(rule.enabled == 0 for rule in guard.rules)


def test_invalid_regex_rule_is_skipped(caplog: pytest.LogCaptureFixture) -> None:
    """非法正则的规则被跳过，不影响其余规则生效。"""
    rules = [
        {
            "rule_code": "BROKEN",
            "name": "坏规则",
            "pattern": r"(未闭合",
            "action": "BLOCK",
            "category": "测试",
        },
        {
            "rule_code": "OK_RULE",
            "name": "好规则",
            "pattern": r"稳赚不赔",
            "action": "BLOCK",
            "category": "承诺收益",
        },
    ]
    guard = build_guard(rules)
    assert guard.check("稳赚不赔的产品").action == "BLOCK"


def test_output_risk_patterns_registered() -> None:
    """输出侧违规词表非空且覆盖关键表述。"""
    from app.guard.compliance import OUTPUT_RISK_PATTERNS

    joined = "|".join(OUTPUT_RISK_PATTERNS)
    for keyword in ("保证收益", "稳赚不赔", "必涨", "零风险"):
        assert keyword in joined


@pytest_asyncio.fixture
async def rule_db() -> Any:
    """构造只含 risk_rule 表的 SQLite 环境。"""
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    factory = async_sessionmaker(bind=engine, expire_on_commit=False)
    yield factory
    await engine.dispose()


async def test_load_rules_from_database(rule_db: Any) -> None:
    """规则可从 risk_rule 表加载（含替代内置规则的场景）。"""
    async with rule_db() as session:
        session.add(
            RiskRule(
                rule_code="DB_ONLY",
                name="数据库规则",
                pattern=r"数据库专属敏感词",
                action="BLOCK",
                category="测试",
                priority=5,
            )
        )
        await session.commit()

    guard = ComplianceGuard(session_factory=rule_db)
    await guard.load_rules(force=True)
    assert len(guard.rules) == 1
    assert guard.check("数据库专属敏感词").action == "BLOCK"
    assert guard.check("稳赚不赔").action == "PASS"  # 内置规则已被 DB 规则替换


async def test_load_rules_falls_back_to_defaults(rule_db: Any) -> None:
    """DB 中无规则（或不可用）时回退内置默认规则，保证拦截能力不丢失。"""
    guard = ComplianceGuard(session_factory=rule_db)
    await guard.load_rules(force=True)
    assert len(guard.rules) == len(DEFAULT_RISK_RULES)
    assert guard.check("帮我推荐一只稳赚的基金").action == "BLOCK"


async def test_rule_cache_avoids_reload(rule_db: Any) -> None:
    """规则带 TTL 缓存：TTL 内不重复查库。"""
    guard = ComplianceGuard(session_factory=rule_db, cache_ttl=300)
    await guard.load_rules(force=True)
    loaded_at = guard._loaded_at
    await guard.load_rules()
    assert guard._loaded_at == loaded_at


async def test_guard_with_session_parameter(rule_db: Any) -> None:
    """支持复用请求级 session 加载规则（避免额外连接）。"""
    async with rule_db() as session:
        session.add(
            RiskRule(
                rule_code="S1",
                name="会话规则",
                pattern=r"会话级敏感词",
                action="BLOCK",
                category="测试",
            )
        )
        await session.commit()
        guard = ComplianceGuard()
        rules = await guard.load_rules(force=True, session=session)
        assert any(rule.rule_code == "S1" for rule in rules)

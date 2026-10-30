"""金融合规与安全过滤：输入侧三级动作（BLOCK / REWRITE / PASS）+ 输出侧声明追加。

设计要点：
1. 规则来自数据库 ``risk_rule`` 表（可运营调整），带 5 分钟缓存（Redis + 进程内存）；
   DB 不可用时回退到内置默认规则，保证系统始终具备拦截能力。
2. 多模式匹配：两级策略 —— 先用「组合正则」（所有规则合并为一个命名分组表达式）做单次
   快速预筛（无命中即直接放行，绝大多数正常提问走这条快路径）；预筛命中后再逐条规则
   精确匹配，保证不漏报任何一条命中的规则（组合正则单次 search 只能报告最左的一个分支）。
3. 输出侧 ``check_output`` 兜底：模型输出若出现「保证收益」「必涨」等违规表述，
   追加风控声明（Prompt 约束之外的第二道防线）。
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from typing import Any, Iterable, Optional, Sequence

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.core.logging import get_logger
from app.guard.default_rules import DEFAULT_RISK_RULES
from app.models import RiskRule

logger = get_logger(__name__)

# 全站统一免责声明（RAG Prompt、接口层、输出侧检查复用同一常量）
DISCLAIMER = (
    "以上内容仅作理财知识科普，不构成任何投资建议；"
    "理财有风险，投资需谨慎，请结合自身情况在正规持牌机构办理业务。"
)

# 输出侧违规表述（模型一旦生成即追加声明，作为最后一道防线）
OUTPUT_RISK_PATTERNS: tuple[str, ...] = (
    r"保证收益",
    r"保本保息",
    r"稳赚不赔",
    r"必涨",
    r"必定上涨",
    r"包赚",
    r"零风险",
    r"无风险高收益",
    r"推荐(购买|买入).{0,6}(基金|股票)",
)
_OUTPUT_RE = re.compile("|".join(OUTPUT_RISK_PATTERNS))

# 知识询问降级：含「什么是/为什么/区别」等学习型措辞且不含索要指令时，
# 把命中的 BLOCK 规则降级为 PASS（避免「为什么要警惕承诺收益的产品」被误拦）
_KNOWLEDGE_HINT = re.compile(
    r"(什么是|是什么|是指|含义|为什么|区别|原理|科普|如何理解|怎么理解|为什么说|怎么识别|怎么防范)"
)
_REQUEST_HINT = re.compile(r"(推荐|介绍个|介绍一|给我|帮我|该买|买入|入手|申购|买哪|哪只|哪个.*最好)")

BLOCK_MESSAGE = (
    "抱歉，这个问题涉及投资建议或承诺收益，我不能直接回答，也不提供任何具体产品推荐。"
    "如果你愿意，我可以从知识角度讲讲：如何评估一只基金或理财产品是否适合自己、"
    "怎样识别承诺收益类话术的风险、以及存款保险等基础保障制度。"
    + DISCLAIMER
)


@dataclass(frozen=True)
class _Rule:
    """内部规则表示（与 ORM 解耦，便于测试注入）。"""

    rule_code: str
    name: str
    pattern: str
    action: str
    category: str
    replacement: Optional[str] = None
    priority: int = 100
    enabled: int = 1

    @classmethod
    def from_mapping(cls, item: Any) -> "_Rule":
        """从 ORM 对象或字典构造。"""
        if isinstance(item, dict):
            data = item
        else:
            data = {
                "rule_code": item.rule_code,
                "name": item.name,
                "pattern": item.pattern,
                "action": item.action,
                "category": item.category,
                "replacement": item.replacement,
                "priority": item.priority,
                "enabled": item.enabled,
            }
        return cls(
            rule_code=str(data.get("rule_code", "")),
            name=str(data.get("name", "")),
            pattern=str(data.get("pattern", "")),
            action=str(data.get("action", "PASS")).upper(),
            category=str(data.get("category", "")),
            replacement=data.get("replacement"),
            priority=int(data.get("priority", 100)),
            enabled=int(data.get("enabled", 1)),
        )


@dataclass
class GuardResult:
    """合规检查结果。"""

    action: str = "PASS"  # BLOCK / REWRITE / PASS
    original: str = ""
    rewritten_query: Optional[str] = None
    matched_rules: list[str] = field(default_factory=list)
    message: Optional[str] = None
    warned: bool = False
    warn_message: Optional[str] = None
    downgraded: bool = False

    @property
    def blocked(self) -> bool:
        """是否直接拒答。"""
        return self.action == "BLOCK"

    @property
    def effective_query(self) -> str:
        """真正送入检索与模型的问题（REWRITE 时取改写结果）。"""
        return self.rewritten_query or self.original

    def to_dict(self) -> dict[str, Any]:
        """序列化（写 qa_log）。"""
        return {
            "action": self.action,
            "matched_rules": self.matched_rules,
            "rewritten_query": self.rewritten_query,
            "warned": self.warned,
            "downgraded": self.downgraded,
        }


class ComplianceGuard:
    """合规守卫：规则加载（带缓存）+ 输入检查 + 输出检查。"""

    def __init__(
        self,
        rules: Optional[Sequence[Any]] = None,
        *,
        session_factory: Optional[async_sessionmaker[AsyncSession]] = None,
        cache_ttl: int = 300,
    ) -> None:
        """初始化。

        :param rules: 显式规则（测试/离线场景）；为空则从 DB（或内置默认）加载
        :param session_factory: 异步会话工厂
        :param cache_ttl: 规则缓存秒数（默认 5 分钟）
        """
        self._session_factory = session_factory
        self._cache_ttl = cache_ttl
        self._rules: list[_Rule] = [_Rule.from_mapping(item) for item in (rules or DEFAULT_RISK_RULES)]
        self._loaded_at: float = 0.0
        self._bundle: Optional[re.Pattern[str]] = None
        self._bundle_map: list[_Rule] = []
        self._compiled: list[tuple[_Rule, re.Pattern[str]]] = []
        self._compile()

    # ------------------------------------------------------------------ 规则加载
    @property
    def rules(self) -> list[_Rule]:
        """当前生效规则（只读副本）。"""
        return list(self._rules)

    def _compile(self) -> None:
        """把所有规则编译成一个组合正则（命名分组 r0/r1/...）。"""
        enabled = [rule for rule in self._rules if rule.enabled]
        enabled.sort(key=lambda rule: (rule.priority, rule.rule_code))
        parts: list[str] = []
        self._bundle_map = []
        self._compiled = []
        for rule in enabled:
            try:
                compiled = re.compile(rule.pattern)
            except re.error as exc:
                logger.error("风控规则 %s 正则非法，已跳过：%s", rule.rule_code, exc)
                continue
            # 组名下标必须与 _bundle_map 的下标严格一致（跳过的规则不占位）
            parts.append(f"(?P<r{len(self._bundle_map)}>{rule.pattern})")
            self._bundle_map.append(rule)
            self._compiled.append((rule, compiled))
        self._bundle = re.compile("|".join(parts)) if parts else None

    async def load_rules(self, force: bool = False, session: Optional[AsyncSession] = None) -> list[_Rule]:
        """从缓存 / DB / 内置默认依次加载规则。

        :param force: 忽略缓存强制刷新
        """
        if not force and self._loaded_at and time.time() - self._loaded_at < self._cache_ttl:
            return self._rules

        cached = await self._read_redis_cache()
        if cached is not None and not force:
            self._apply(cached)
            self._loaded_at = time.time()
            return self._rules

        loaded: list[_Rule] = []
        try:
            if session is not None:
                loaded = await self._load_from_session(session)
            else:
                factory = self._session_factory
                if factory is None:
                    from app.core.database import get_session_factory

                    factory = get_session_factory()
                async with factory() as own_session:
                    loaded = await self._load_from_session(own_session)
        except Exception as exc:
            logger.warning("从数据库加载风控规则失败，回退内置默认规则：%s", exc)

        if not loaded:
            loaded = [_Rule.from_mapping(item) for item in DEFAULT_RISK_RULES]
        self._apply(loaded)
        self._loaded_at = time.time()
        await self._write_redis_cache(loaded)
        logger.info("风控规则已加载：%d 条（缓存 %ds）", len(self._rules), self._cache_ttl)
        return self._rules

    async def _load_from_session(self, session: AsyncSession) -> list[_Rule]:
        """从 risk_rule 表读取启用规则。"""
        rows = (
            await session.scalars(
                select(RiskRule).where(RiskRule.enabled == 1).order_by(RiskRule.priority)
            )
        ).all()
        return [_Rule.from_mapping(row) for row in rows]

    async def _read_redis_cache(self) -> Optional[list[_Rule]]:
        """读取 Redis 中的规则缓存（不可用时返回 None）。"""
        try:
            from app.core.redis_client import KEY_RISK_RULE_CACHE, get_redis

            raw = await get_redis().get(KEY_RISK_RULE_CACHE)
            if not raw:
                return None
            return [_Rule.from_mapping(item) for item in json.loads(raw)]
        except Exception:  # pragma: no cover - 依赖 Redis
            return None

    async def _write_redis_cache(self, rules: Sequence[_Rule]) -> None:
        """写入 Redis 规则缓存（失败不影响主流程）。"""
        try:
            from app.core.redis_client import KEY_RISK_RULE_CACHE, get_redis

            payload = json.dumps([rule.__dict__ for rule in rules], ensure_ascii=False)
            await get_redis().set(KEY_RISK_RULE_CACHE, payload, ex=self._cache_ttl)
        except Exception as exc:  # pragma: no cover - 依赖 Redis
            logger.debug("写入规则缓存失败：%s", exc)

    def _apply(self, rules: Sequence[_Rule]) -> None:
        """替换规则集合并重新编译。"""
        self._rules = list(rules)
        self._compile()

    # ------------------------------------------------------------------ 输入检查
    def check(self, question: str) -> GuardResult:
        """输入侧检查：返回 BLOCK / REWRITE / PASS 三级动作。

        :param question: 用户原始问题
        """
        text = (question or "").strip()
        result = GuardResult(original=text)
        if not text:
            return result

        hits: list[tuple[_Rule, str]] = []
        # 一级：组合正则快速预筛（无命中则直接放行，避免逐条匹配开销）
        if self._bundle is None or self._bundle.search(text) is None:
            return result
        # 二级：逐条精确匹配，收集全部命中规则（组合正则只报告最左分支）
        for rule, compiled in self._compiled:
            matched = compiled.search(text)
            if matched is not None:
                hits.append((rule, matched.group(0)))

        if not hits:
            return result

        result.matched_rules = [rule.rule_code for rule, _ in hits]
        blocks = [item for item in hits if item[0].action == "BLOCK"]
        rewrites = [item for item in hits if item[0].action == "REWRITE"]
        warns = [item for item in hits if item[0].action == "WARN"]

        if blocks:
            # 知识询问降级：学习型提问且不含索要指令时放行
            if _KNOWLEDGE_HINT.search(text) and not _REQUEST_HINT.search(text):
                result.action = "PASS"
                result.downgraded = True
                result.warned = True
                result.warn_message = self._warn_message(warns) or (
                    "提示：" + DISCLAIMER
                )
                logger.info("BLOCK 规则被知识询问降级：%s -> %s", result.matched_rules, text[:40])
                return result
            result.action = "BLOCK"
            result.message = BLOCK_MESSAGE
            logger.info("问题被风控拦截：%s（规则 %s）", text[:40], result.matched_rules)
            return result

        if rewrites:
            rule, _ = rewrites[0]
            template = rule.replacement or "如何评估理财产品是否适合自己"
            result.action = "REWRITE"
            result.rewritten_query = template.replace("{query}", text)
            if warns:
                result.warned = True
                result.warn_message = self._warn_message(warns)
            logger.info("问题被改写：%s -> %s", text[:40], result.rewritten_query)
            return result

        result.action = "PASS"
        result.warned = True
        result.warn_message = self._warn_message(warns)
        return result

    @staticmethod
    def _warn_message(warns: list[tuple[_Rule, str]]) -> Optional[str]:
        """构造 WARN 级别的提示语。"""
        if not warns:
            return None
        categories = {rule.category for rule, _ in warns}
        if "反诈防骗" in categories:
            return "提示：刷单返利、注销校园贷、杀猪盘等均为常见诈骗手法，请勿向陌生账户转账。"
        return "提示：该话题涉及高风险品种，历史波动可能很大，请充分了解风险后再做决定。"

    # ------------------------------------------------------------------ 输出检查
    def check_output(self, answer: str) -> tuple[str, list[str]]:
        """输出侧检查：命中违规表述时追加风控声明。

        :return: (处理后的答案, 命中的违规表述列表)
        """
        text = answer or ""
        matched = sorted({item for item in _OUTPUT_RE.findall(text) if item})
        if not matched:
            return text, []
        logger.warning("模型输出命中违规表述：%s", matched)
        statement = (
            "\n\n⚠️ 风险提示：上述内容中出现了可能被理解为承诺收益的表述，"
            "任何理财产品都不存在保本保收益的承诺。"
        )
        if DISCLAIMER not in text:
            statement += DISCLAIMER
        return text + statement, matched


def get_guard() -> ComplianceGuard:
    """获取进程级默认守卫实例（规则从 DB 加载，带缓存）。"""
    global _GUARD
    if _GUARD is None:
        _GUARD = ComplianceGuard()
    return _GUARD


_GUARD: Optional[ComplianceGuard] = None


def build_guard(rules: Iterable[Any]) -> ComplianceGuard:
    """按显式规则构造守卫（测试与离线评估用）。"""
    return ComplianceGuard(rules=list(rules))

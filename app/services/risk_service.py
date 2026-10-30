"""风险测评服务：10 题问卷 → 五维得分 → 风险等级与学习路径推荐。"""

from __future__ import annotations

from typing import Any

# 五个维度：知识水平 / 投资经验 / 风险承受 / 流动性需求 / 风险态度
DIMENSIONS: list[tuple[str, str]] = [
    ("knowledge", "理财知识"),
    ("experience", "投资经验"),
    ("capacity", "风险承受"),
    ("liquidity", "流动性需求"),
    ("attitude", "风险态度"),
]

# 10 道题，每题 4 个选项，选项得分 0~3（分值越高表示认知越成熟、承受能力越强）
QUESTIONS: list[dict[str, Any]] = [
    {
        "id": 1,
        "dimension": "knowledge",
        "text": "你对「七日年化收益率」的理解程度是？",
        "options": ["完全不了解", "听过但说不清楚", "大致明白含义", "能自己算出来"],
    },
    {
        "id": 2,
        "dimension": "knowledge",
        "text": "你读过理财产品的说明书或风险揭示书吗？",
        "options": ["从没读过", "偶尔翻过", "买之前会看关键条款", "会逐条对比不同产品"],
    },
    {
        "id": 3,
        "dimension": "experience",
        "text": "你的投资经历是？",
        "options": ["从未投资过", "只买过货币基金或存款", "买过债券或指数基金", "经历过完整的涨跌周期"],
    },
    {
        "id": 4,
        "dimension": "experience",
        "text": "持有的基金出现 10% 浮亏时，你的做法是？",
        "options": ["立刻全部卖出", "很焦虑但先观望", "按计划继续持有", "按纪律补仓并复盘原因"],
    },
    {
        "id": 5,
        "dimension": "capacity",
        "text": "每月可用于投资的钱占生活费结余的比例是？",
        "options": ["没有结余", "10% 以内", "10%~30%", "30% 以上且不影响生活"],
    },
    {
        "id": 6,
        "dimension": "capacity",
        "text": "你是否已准备好 3~6 个月生活费的应急资金？",
        "options": ["完全没有", "有一点点", "基本够用", "已备足并单独存放"],
    },
    {
        "id": 7,
        "dimension": "capacity",
        "text": "如果投资本金亏损 20%，对你生活的影响是？",
        "options": ["影响吃饭和房租", "明显影响生活质量", "有影响但能承受", "基本无影响"],
    },
    {
        "id": 8,
        "dimension": "liquidity",
        "text": "这笔投资资金预计多久不需要动用？",
        "options": ["随时可能要用（1 个月内）", "半年内可能用", "1~3 年不用", "3 年以上不用"],
    },
    {
        "id": 9,
        "dimension": "attitude",
        "text": "在收益与安全之间，你更倾向于？",
        "options": [
            "本金绝对不能亏",
            "本金优先，收益次要",
            "愿意承担一定波动换取更高收益",
            "为追求高收益可以承担较大波动",
        ],
    },
    {
        "id": 10,
        "dimension": "attitude",
        "text": "投资出现波动时，你的睡眠与情绪状态是？",
        "options": ["严重失眠", "比较影响情绪", "略有影响", "几乎不受影响"],
    },
]

# 风险等级阈值（总分 0~30）
LEVEL_RULES: list[tuple[int, str, str]] = [
    (8, "C1", "保守型"),
    (14, "C2", "稳健型"),
    (20, "C3", "平衡型"),
    (25, "C4", "成长型"),
    (30, "C5", "进取型"),
]

# 各等级的说明与学习路径推荐
LEVEL_ADVICE: dict[str, dict[str, Any]] = {
    "C1": {
        "suggestion": "你目前更适合以「保本与流动性」为优先，先积累应急资金再谈投资。",
        "learning_path": [
            "银行储蓄与存款保险：了解利率换算与 50 万元偿付上限",
            "反诈防骗：识别刷单返利、校园贷与杀猪盘话术",
            "个人信用与借贷风险：看懂征信报告与借贷成本",
        ],
    },
    "C2": {
        "suggestion": "你可以在保证本金安全的前提下，尝试理解低波动产品与分散配置的意义。",
        "learning_path": [
            "货币基金与债券基金的区别",
            "基金的费用结构：申购费、管理费与赎回费",
            "应急资金与保险保障的基本配置思路",
        ],
    },
    "C3": {
        "suggestion": "你具备一定认知，可以系统学习指数投资与定投纪律，用小额长期资金实践。",
        "learning_path": [
            "基金定投：用时间平滑成本的方法与常见误区",
            "指数基金与主动基金的差异及适用场景",
            "如何评估一只基金是否适合自己",
        ],
    },
    "C4": {
        "suggestion": "你能承受一定波动，建议在分散投资的同时补齐风险控制与复盘方法。",
        "learning_path": [
            "资产配置基本框架与相关性思维",
            "定投与再平衡的操作纪律",
            "最大回撤、夏普比率等风险指标的含义",
        ],
    },
    "C5": {
        "suggestion": "你的风险偏好较高，更需要关注杠杆、集中度与流动性风险，避免过度自信。",
        "learning_path": [
            "杠杆与配资的风险传导机制",
            "组合集中度与极端行情下的应对",
            "行为金融学中的过度自信与追涨杀跌",
        ],
    },
}


def compute_risk_profile(answers: list[int]) -> dict[str, Any]:
    """根据 10 题答案计算风险测评结果。

    :param answers: 10 个选项得分（每题 0~3）
    :return: ``{risk_level, risk_label, risk_score, suggestion, learning_path}``
    """
    if len(answers) != len(QUESTIONS):
        raise ValueError(f"需要 {len(QUESTIONS)} 题答案，实际收到 {len(answers)} 题")
    dim_totals: dict[str, int] = {key: 0 for key, _ in DIMENSIONS}
    dim_counts: dict[str, int] = {key: 0 for key, _ in DIMENSIONS}
    dim_labels = dict(DIMENSIONS)
    for question, answer in zip(QUESTIONS, answers, strict=False):
        score = max(0, min(3, int(answer)))
        dim_totals[question["dimension"]] += score
        dim_counts[question["dimension"]] += 1

    dim_percent = {
        key: round(dim_totals[key] / (dim_counts[key] * 3) * 100, 1) for key, _ in DIMENSIONS
    }
    total = sum(dim_totals.values())

    level, label = "C5", "进取型"
    for threshold, level_code, level_label in LEVEL_RULES:
        if total <= threshold:
            level, label = level_code, level_label
            break

    advice = LEVEL_ADVICE[level]
    return {
        "risk_level": level,
        "risk_label": label,
        "risk_score": {
            "total": total,
            "max_total": len(QUESTIONS) * 3,
            "dimensions": [
                {"key": key, "label": dim_labels[key], "percent": dim_percent[key]}
                for key, _ in DIMENSIONS
            ],
        },
        "suggestion": advice["suggestion"],
        "learning_path": list(advice["learning_path"]),
    }


def question_bank() -> list[dict[str, Any]]:
    """返回风险测评题目（供前端渲染）。"""
    return [dict(item) for item in QUESTIONS]

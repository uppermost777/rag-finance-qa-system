"""风控规则默认数据：``scripts/seed_data.py`` 与 ``ComplianceGuard`` 共用的唯一数据源。

action 语义：
- ``BLOCK``   ：直接拒答并返回合规提示，不进大模型
- ``REWRITE`` ：改写为合规问法后再进大模型
- ``WARN``    ：放行，但在输出末尾追加风险提示（等价 PASS + 声明）
"""

from __future__ import annotations

from typing import Any

# rule_code, name, pattern, action, category, replacement, priority, remark
DEFAULT_RISK_RULES: list[dict[str, Any]] = [
    {
        "rule_code": "BLOCK_GUARANTEE",
        "name": "承诺收益类表述",
        "pattern": r"(保证|承诺|确保)[^，。？]{0,6}(收益|盈利|赚钱|保本)|稳赚不赔|只赚不赔|必涨|必赚|包赚|包赚不赔|零风险高收益|无风险高收益|保本保息",
        "action": "BLOCK",
        "category": "承诺收益",
        "replacement": None,
        "priority": 10,
        "remark": "《证券期货投资者适当性管理办法》禁止承诺收益",
    },
    {
        "rule_code": "BLOCK_RECOMMEND",
        "name": "荐股荐基类请求",
        "pattern": r"(推荐|介绍|告诉我|给个)[^，。？]{0,8}(一只|个|支|款)?[^，。？]{0,6}(基金|股票|股票代码|理财产品|产品|币|黄金)",
        "action": "BLOCK",
        "category": "荐股荐基",
        "replacement": None,
        "priority": 20,
        "remark": "系统不具备投资顾问资质，禁止个股/个基推荐",
    },
    {
        "rule_code": "BLOCK_GET_RICH",
        "name": "暴富/翻倍类诱导",
        "pattern": r"(几天|一周|一个月|短期|快速)[^，。？]{0,6}(翻倍|翻番|暴富|赚翻)|翻(倍|番|几倍|十倍)|一夜暴富|月入百万|年化\s*\d{2,}%?\s*(以上|起)|稳定(日|月)入",
        "action": "BLOCK",
        "category": "诱导交易",
        "replacement": None,
        "priority": 30,
        "remark": "承诺高回报属于典型诈骗话术",
    },
    {
        "rule_code": "BLOCK_LEVERAGE",
        "name": "杠杆配资类请求",
        "pattern": r"配资|加杠杆[^，。？]{0,6}(炒股|买股票|买入|投资)|借钱炒股|贷款(炒股|买基金|投资股票)",
        "action": "BLOCK",
        "category": "高风险杠杆",
        "replacement": None,
        "priority": 40,
        "remark": "面向学生的杠杆/借贷投资建议风险极高",
    },
    {
        "rule_code": "BLOCK_ILLEGAL",
        "name": "非法集资类表述",
        "pattern": r"非法集资|集资平台|原始股|国家(项目|工程)[^，。？]{0,6}(高回报|高收益)|内部渠道(认购|申购)",
        "action": "BLOCK",
        "category": "非法集资",
        "replacement": None,
        "priority": 50,
        "remark": "涉非法集资话术，直接拦截",
    },
    {
        "rule_code": "BLOCK_INSIDER",
        "name": "内幕消息类请求",
        "pattern": r"内幕(消息|信息|股)|庄家[^，。？]{0,4}(消息|内幕)|跟庄|老鼠仓",
        "action": "BLOCK",
        "category": "内幕交易",
        "replacement": None,
        "priority": 60,
        "remark": "内幕交易属违法行为",
    },
    {
        "rule_code": "REWRITE_PICK_ONE",
        "name": "选品类请求改写",
        "pattern": r"(我|自己)?该(买|选|投)(哪|什么|啥)(只|个|支|款)?[^，。？]{0,4}(基金|股票|理财产品|存款)?|(买|选|投|挑)(哪只|哪个|哪支|什么|啥)[^，。？]{0,4}(股票|基金|理财产品)|哪个理财产品最好|哪个(基金|产品|理财产品)值得买|什么理财产品收益高",
        "action": "REWRITE",
        "category": "选品请求",
        "replacement": "如何评估一只基金、股票或理财产品是否适合自己",
        "priority": 200,
        "remark": "把「买哪个」转化为「如何评估」，保留学习价值",
    },
    {
        "rule_code": "REWRITE_BEST",
        "name": "最优解类提问改写",
        "pattern": r"(哪|什么|啥)(种|个|款)?[^，。？]{0,6}(最好|最赚|最划算|收益最高|回报最高|最安全)",
        "action": "REWRITE",
        "category": "选品请求",
        "replacement": "如何比较不同理财产品的风险与收益",
        "priority": 210,
        "remark": "不存在普适最优，转化为比较框架",
    },
    {
        "rule_code": "WARN_HIGH_RISK",
        "name": "高风险品种提示",
        "pattern": r"虚拟货币|比特币|炒币|期货|期权|外汇保证金|P2P|高风险(投资|投机)",
        "action": "WARN",
        "category": "高风险品种",
        "replacement": None,
        "priority": 300,
        "remark": "允许知识科普，但必须叠加风险提示",
    },
    {
        "rule_code": "WARN_ANTI_FRAUD",
        "name": "校园诈骗场景提示",
        "pattern": r"刷单|返利|杀猪盘|兼职(刷单|返利)|校园贷|裸条贷|注销校园贷",
        "action": "WARN",
        "category": "反诈防骗",
        "replacement": None,
        "priority": 310,
        "remark": "命中反诈关键词时追加防骗提醒",
    },
]


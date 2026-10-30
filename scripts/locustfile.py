"""Locust 压测脚本（并发场景）。

用法：
    pip install locust
    locust -f scripts/locustfile.py --host http://127.0.0.1:8000 --headless -u 20 -r 5 -t 60s
"""

from __future__ import annotations

import random

from locust import HttpUser, between, task  # type: ignore[import-not-found]

# 压测问题池（均为知识型提问，避免触发合规拦截影响统计）
QUESTIONS = [
    "什么是基金定投",
    "存款保险最多能赔多少钱",
    "七日年化收益率怎么算",
    "征信逾期记录会保留多久",
    "货币基金和债券基金有什么区别",
    "为什么抄底这么难",
]


class FinanceQaUser(HttpUser):
    """模拟学生用户：登录 → 反复提问（非流式）→ 偶尔查看统计。"""

    wait_time = between(1, 3)
    token: str = ""

    def on_start(self) -> None:
        """登录一次并复用 token。"""
        payload = {"username": "student_lin", "password": "Student@2026"}
        with self.client.post("/api/v1/auth/login", json=payload, catch_response=True) as resp:
            if resp.status_code == 200:
                self.token = resp.json()["data"]["access_token"]
            else:
                resp.failure(f"登录失败：{resp.status_code}")

    @property
    def headers(self) -> dict[str, str]:
        """鉴权头。"""
        return {"Authorization": f"Bearer {self.token}"}

    @task(8)
    def ask_question(self) -> None:
        """提问（非流式问答接口）。"""
        payload = {"question": random.choice(QUESTIONS), "strategy": "hybrid_rerank"}
        self.client.post("/api/v1/chat/completions", json=payload, headers=self.headers, timeout=120)

    @task(2)
    def read_stats(self) -> None:
        """查看个人统计。"""
        self.client.get("/api/v1/chat/stats", headers=self.headers)

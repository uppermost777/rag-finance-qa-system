"""本地端到端联调（无 Docker 方案）。

链路：MariaDB(MySQL 协议) + Milvus Lite + bge-m3 嵌入 + 本地 OpenAI 兼容 LLM + FastAPI + SSE。
依次完成：建表/种子 → 登录 → 上传 3 篇真实文档 → 入库 → 非流式问答 → 流式问答 → 合规拦截 → 出报告。

前置条件：
1. MariaDB 已在 127.0.0.1:3307 运行（work_logs/mariadb）；
2. models/bge-m3 已下载；
3. models/qwen2.5-0.5b-instruct 已下载（脚本会自动拉起本地 LLM 服务）。

用法：
    python scripts/e2e_local.py
    python scripts/e2e_local.py --port 8021 --skip-ingest
"""

from __future__ import annotations

import argparse
import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import httpx

ROOT = Path(__file__).resolve().parents[1]
PYTHON = sys.executable
OUTPUTS = ROOT.parent / "outputs"
DOCS_DIR = ROOT / "work_logs" / "e2e_docs"
E2E_KB_NAME = "E2E 联调知识库"


def port_open(host: str, port: int) -> bool:
    """检测端口是否可连接。"""
    with socket.socket() as sock:
        sock.settimeout(0.5)
        return sock.connect_ex((host, port)) == 0


def build_env() -> dict[str, str]:
    """构造联调环境变量（覆盖 .env，不修改仓库配置）。"""
    env = os.environ.copy()
    embedding_path, embedding_dim = pick_embedding_model()
    env.update(
        {
            "DATABASE_URL": "mysql+aiomysql://finance:finance_pwd@127.0.0.1:3307/finance_qa?charset=utf8mb4",
            "SYNC_DATABASE_URL": "mysql+pymysql://finance:finance_pwd@127.0.0.1:3307/finance_qa?charset=utf8mb4",
            "MILVUS_LITE_PATH": str(ROOT / "work_logs" / "milvus_e2e.db"),
            "MILVUS_COLLECTION": "finance_chunks",
            "EMBEDDING_MODEL_PATH": str(embedding_path),
            "EMBEDDING_DIM": str(embedding_dim),
            "EMBEDDING_BATCH_SIZE": "8",
            "LLM_BASE_URL": "http://127.0.0.1:11435/v1",
            "LLM_MODEL": "qwen2.5-0.5b-instruct",
            "LLM_API_KEY": "local-dev-key",
            "LLM_TIMEOUT": "240",
            "LLM_TEMPERATURE": "0.2",
            "JWT_SECRET": "e2e-secret-key-0123456789abcdef",
            "UPLOAD_DIR": str(ROOT / "work_logs" / "e2e_uploads"),
            "INLINE_INGEST": "true",
            "DEFAULT_STRATEGY": "hybrid",
            "MIN_RERANK_SCORE": "0.0",
            "RATE_LIMIT_PER_MINUTE": "200",
            "LOG_LEVEL": "INFO",
            "DEBUG": "false",
        }
    )
    return env


def pick_embedding_model() -> tuple[Path, int]:
    """选择可用的嵌入模型：优先 bge-m3(1024)，其次 bge-small-zh(512)。"""
    m3 = ROOT / "models" / "bge-m3" / "pytorch_model.bin"
    if m3.exists() and m3.stat().st_size > 2_000_000_000:
        return m3.parent, 1024
    small = ROOT / "models" / "bge-small-zh-v1.5" / "model.safetensors"
    if small.exists() and small.stat().st_size > 50_000_000:
        print("[e2e] 使用 bge-small-zh-v1.5（512 维）：bge-m3 尚未下载完整")
        return small.parent, 512
    raise RuntimeError("未找到可用的嵌入模型（bge-m3 或 bge-small-zh-v1.5）")


def wait_http(url: str, timeout: float = 180, interval: float = 1.0) -> bool:
    """轮询等待 HTTP 服务可用。"""
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            response = httpx.get(url, timeout=3)
            if response.status_code < 500:
                return True
        except Exception:
            pass
        time.sleep(interval)
    return False


def run_script(name: str, env: dict[str, str]) -> None:
    """运行数据初始化脚本。"""
    result = subprocess.run(
        [PYTHON, str(ROOT / "scripts" / name)],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    tail = "\n".join((result.stdout + result.stderr).strip().splitlines()[-3:])
    print(f"[e2e] {name} -> {result.returncode}\n{tail}")
    if result.returncode != 0:
        raise RuntimeError(f"{name} 执行失败")


class Server:
    """子进程服务封装（自动拉起/关闭）。"""

    def __init__(self, name: str, args: list[str], env: dict[str, str]) -> None:
        """记录启动参数。"""
        self.name = name
        self.args = args
        self.env = env
        self.process: subprocess.Popen[bytes] | None = None
        self.log_path = ROOT / "work_logs" / f"e2e_{name}.log"

    def start(self) -> None:
        """启动并把日志写入文件。"""
        log = self.log_path.open("wb")
        self.process = subprocess.Popen(
            self.args,
            cwd=ROOT,
            env=self.env,
            stdout=log,
            stderr=subprocess.STDOUT,
        )
        print(f"[e2e] 启动 {self.name} pid={self.process.pid} log={self.log_path.name}")

    def stop(self) -> None:
        """终止进程。"""
        if self.process and self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.process.kill()
            print(f"[e2e] 已停止 {self.name}")


def login(client: httpx.Client, username: str, password: str) -> str:
    """登录并返回 JWT。"""
    response = client.post("/api/v1/auth/login", json={"username": username, "password": password})
    response.raise_for_status()
    return response.json()["data"]["access_token"]


def ensure_kb(client: httpx.Client, headers: dict[str, str]) -> int:
    """确保联调知识库存在，返回 ID。"""
    response = client.get("/api/v1/kb", params={"keyword": E2E_KB_NAME, "size": 100}, headers=headers)
    for item in response.json()["data"]["items"]:
        if item["name"] == E2E_KB_NAME:
            return int(item["id"])
    created = client.post(
        "/api/v1/kb",
        json={"name": E2E_KB_NAME, "category": "综合理财", "description": "端到端联调用知识库"},
        headers=headers,
    )
    created.raise_for_status()
    return int(created.json()["data"]["id"])


def upload_and_wait(
    client: httpx.Client, headers: dict[str, str], kb_id: int, files: list[Path]
) -> list[dict[str, Any]]:
    """上传文档并轮询入库进度直到完成。"""
    uploaded: list[dict[str, Any]] = []
    for path in files:
        with path.open("rb") as handle:
            response = client.post(
                "/api/v1/document/upload",
                files={"file": (path.name, handle, "text/plain")},
                data={"kb_id": str(kb_id)},
                headers=headers,
                timeout=120,
            )
        if response.status_code == 409:
            # 重复上传（同库同 MD5）：复用已有文档，便于脚本重复执行
            listing = client.get(
                "/api/v1/document", params={"kb_id": kb_id, "size": 100}, headers=headers
            ).json()["data"]["items"]
            existing = next((item for item in listing if item["name"] == path.name), None)
            if existing is None:
                response.raise_for_status()
            payload = {"doc_id": existing["id"], "task_id": None, "status": existing["status"]}
        else:
            response.raise_for_status()
            payload = response.json()["data"]
        print(f"[e2e] 上传 {path.name} -> doc_id={payload['doc_id']} task={payload['task_id']}")
        uploaded.append(payload)

    deadline = time.time() + 600
    pending = {int(item["doc_id"]) for item in uploaded}
    results: dict[int, dict[str, Any]] = {}
    while pending and time.time() < deadline:
        for doc_id in list(pending):
            info = client.get(f"/api/v1/document/{doc_id}/progress", headers=headers).json()["data"]
            if info["status"] in ("done", "failed"):
                results[doc_id] = info
                pending.discard(doc_id)
                print(f"[e2e] doc {doc_id} 入库结束：{info['status']} chunks={info['chunk_count']}")
            else:
                print(f"[e2e] doc {doc_id} 进度 {info['progress']}% [{info['status']}]")
        time.sleep(3)
    if pending:
        raise TimeoutError(f"入库超时：{pending}")
    failed = [info for info in results.values() if info["status"] == "failed"]
    if failed:
        raise RuntimeError(f"入库失败：{failed}")
    return list(results.values())


def ask(client: httpx.Client, headers: dict[str, str], kb_id: int, question: str) -> dict[str, Any]:
    """非流式问答。"""
    started = time.time()
    response = client.post(
        "/api/v1/chat/completions",
        json={"question": question, "kb_ids": [kb_id], "strategy": "hybrid"},
        headers=headers,
        timeout=300,
    )
    response.raise_for_status()
    payload = response.json()["data"]
    payload["wall_ms"] = int((time.time() - started) * 1000)
    return payload


def ask_stream(client: httpx.Client, headers: dict[str, str], kb_id: int, question: str) -> dict[str, Any]:
    """流式问答（SSE）。"""
    events: list[tuple[str, dict[str, Any]]] = []
    with client.stream(
        "GET",
        "/api/v1/chat/stream",
        params={"question": question, "kb_ids": str(kb_id), "strategy": "hybrid"},
        headers=headers,
        timeout=300,
    ) as response:
        response.raise_for_status()
        event_name = "message"
        for line in response.iter_lines():
            if line.startswith("event:"):
                event_name = line.split(":", 1)[1].strip()
            elif line.startswith("data:"):
                data = json.loads(line.split(":", 1)[1].strip())
                events.append((event_name, data))
    text = "".join(data.get("text", "") for name, data in events if name == "delta")
    done = next((data for name, data in reversed(events) if name == "done"), {})
    return {"events": events, "text": text, "done": done}


def main() -> int:
    """执行端到端联调。"""
    parser = argparse.ArgumentParser(description="本地端到端联调")
    parser.add_argument("--port", type=int, default=8021)
    parser.add_argument("--skip-servers", action="store_true", help="服务已在运行，不再拉起")
    args = parser.parse_args()

    env = build_env()
    embedding_model, embedding_dim = pick_embedding_model()
    if not port_open("127.0.0.1", 3307):
        print("[e2e] 警告：3307 端口未监听，请先启动本地 MariaDB")
    if not (ROOT / "models" / "bge-m3" / "pytorch_model.bin").exists():
        print("[e2e] 警告：bge-m3 权重不存在，入库会失败")

    llm = Server(
        "llm",
        [PYTHON, str(ROOT / "scripts" / "local_llm_server.py"), "--model",
         str(ROOT / "models" / "qwen2.5-0.5b-instruct"), "--port", "11435"],
        env,
    )
    api = Server(
        "api",
        [PYTHON, "-m", "uvicorn", "app.main:app", "--host", "127.0.0.1", "--port", str(args.port)],
        env,
    )
    base = f"http://127.0.0.1:{args.port}"
    report_lines: list[str] = ["# 端到端联调报告", ""]
    started_at = time.strftime("%Y-%m-%d %H:%M:%S")

    try:
        run_script("init_db.py", env)
        run_script("seed_data.py", env)

        if not args.skip_servers:
            if not port_open("127.0.0.1", 11435):
                llm.start()
            api.start()

        if not wait_http("http://127.0.0.1:11435/health", timeout=240):
            raise RuntimeError("本地 LLM 服务未就绪，见 work_logs/e2e_llm.log")
        if not wait_http(f"{base}/health", timeout=120):
            raise RuntimeError("API 服务未就绪，见 work_logs/e2e_api.log")
        print("[e2e] 服务就绪")

        with httpx.Client(base_url=base, timeout=120) as client:
            health = client.get("/health").json()["data"]
            token = login(client, "admin", "Admin@2026")
            headers = {"Authorization": f"Bearer {token}"}
            kb_id = ensure_kb(client, headers)
            files = sorted(DOCS_DIR.glob("*"))
            docs = upload_and_wait(client, headers, kb_id, files)

            cases = [
                ("存款保险最多能赔多少钱？", "01_银行存款与存款保险.txt"),
                ("基金定投为什么能平滑买入成本？", "02_基金投资与定投入门.md"),
                ("征信上的逾期记录会保留多久？", "03_个人征信与校园借贷风险.txt"),
            ]
            results: list[dict[str, Any]] = []
            for question, expected_source in cases:
                payload = ask(client, headers, kb_id, question)
                top = (payload.get("citations") or [{}])[0]
                results.append(
                    {
                        "question": question,
                        "expected_source": expected_source,
                        "answer": payload.get("answer", ""),
                        "citations": payload.get("citations", []),
                        "blocked": payload.get("blocked"),
                        "guard_action": payload.get("guard_action"),
                        "latency_ms": payload.get("latency_ms"),
                        "wall_ms": payload.get("wall_ms"),
                        "top_source": top.get("doc_name"),
                        "top_page": top.get("page_no"),
                    }
                )
                print(f"[e2e] 问：{question}\n      答：{payload.get('answer','')[:120]}")
                print(f"      引用：{top.get('doc_name')} 第 {top.get('page_no')} 页")

            stream_question = "货币基金和债券基金有什么区别？"
            stream = ask_stream(client, headers, kb_id, stream_question)
            blocked = ask(client, headers, kb_id, "帮我推荐一只稳赚的基金")
            stats = client.get("/api/v1/chat/stats", headers=headers).json()["data"]

        quality = [item["top_source"] == item["expected_source"] for item in results]
        report_lines += [
            f"- 执行时间：{started_at}",
            f"- 服务健康：{health}",
            f"- 嵌入模型：{embedding_model.name}（{embedding_dim} 维）",
            f"- 知识库 ID：{kb_id}",
            f"- 文档入库：{[(d['doc_id'], d['chunk_count']) for d in docs]}",
            "",
            "## 非流式问答（真实检索 + 真实 LLM）",
            "",
            "| 问题 | 预期来源 | Top1 来源 | 页码 | 命中 | 耗时(ms) | 回答摘要 |",
            "|---|---|---|---|---|---|---|",
        ]
        for item, hit in zip(results, quality):
            report_lines.append(
                f"| {item['question']} | {item['expected_source']} | {item['top_source']} | "
                f"{item['top_page']} | {'是' if hit else '否'} | {item['wall_ms']} | "
                f"{item['answer'][:80].replace('|', ' ')} |"
            )
        report_lines += [
            "",
            f"- 来源命中率：{sum(quality)}/{len(quality)}",
            "",
            "## 流式问答（SSE）",
            "",
            f"- 问题：{stream_question}",
            f"- 事件序列：{[name for name, _ in stream['events']][:6]} … 共 {len(stream['events'])} 个事件",
            f"- 拼接文本（前 120 字）：{stream['text'][:120]}",
            f"- done 字段：message_id={stream['done'].get('message_id')} latency_ms={stream['done'].get('latency_ms')}",
            "",
            "## 合规拦截",
            "",
            f"- 问题：帮我推荐一只稳赚的基金",
            f"- blocked={blocked.get('blocked')}　guard_action={blocked.get('guard_action')}",
            f"- 回复：{blocked.get('answer', '')[:120]}",
            "",
            "## 落库统计",
            "",
            f"- {stats}",
            "",
            "## 原始回答",
            "",
        ]
        for item in results:
            report_lines += [
                f"### {item['question']}",
                "",
                item["answer"],
                "",
                "引用：",
            ]
            for citation in item["citations"]:
                report_lines.append(
                    f"- [{citation['index']}] {citation.get('doc_name')} 第 {citation.get('page_no')} 页 "
                    f"(score={citation.get('score')})：{str(citation.get('snippet'))[:80]}"
                )
            report_lines.append("")

        OUTPUTS.mkdir(parents=True, exist_ok=True)
        report_path = OUTPUTS / "端到端联调报告.md"
        report_path.write_text("\n".join(report_lines), encoding="utf-8")
        print(f"\n[e2e] 报告已写入 {report_path}")
        print(f"[e2e] 来源命中 {sum(quality)}/{len(quality)}，流式事件 {len(stream['events'])} 个，"
              f"拦截={blocked.get('blocked')}")
        return 0 if all(quality) and stream["events"] and blocked.get("blocked") else 1
    finally:
        if not args.skip_servers:
            api.stop()
            llm.stop()


if __name__ == "__main__":
    raise SystemExit(main())

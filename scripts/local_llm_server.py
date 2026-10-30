"""本地 OpenAI 兼容 LLM 服务（开发/联调用，零外部依赖）。

用途：在没有 API Key、也无法安装 Ollama 的机器上，用 transformers 加载一个小型中文模型，
对外暴露与 OpenAI 一致的 ``/v1/chat/completions``（支持 stream=True），
从而让主系统的 ``ChatOpenAI`` 客户端无需修改即可对接本地模型。

启动：
    python scripts/local_llm_server.py --model models/qwen2.5-0.5b-instruct --port 11435

说明：0.5B 级模型只用于验证链路可用性，回答质量有限；
正式演示/论文数据请改用 DeepSeek、通义千问等线上模型（修改 .env 中的 LLM_* 即可）。
"""

from __future__ import annotations

import argparse
import json
import threading
import time
import uuid
from typing import Any, Iterator

import torch
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer

app = FastAPI(title="Local OpenAI-compatible LLM")

_state: dict[str, Any] = {"model": None, "tokenizer": None, "path": ""}


def _load(model_path: str) -> None:
    """懒加载模型与分词器。"""
    if _state["model"] is not None:
        return
    print(f"[local-llm] loading {model_path} ...", flush=True)
    _state["tokenizer"] = AutoTokenizer.from_pretrained(model_path)
    _state["model"] = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.float32, low_cpu_mem_usage=True
    )
    _state["model"].eval()
    _state["path"] = model_path
    print("[local-llm] ready", flush=True)


def _prompt(messages: list[dict[str, str]]) -> tuple[Any, int]:
    """按模型 chat template 构造输入。"""
    tokenizer = _state["tokenizer"]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer([text], return_tensors="pt")
    return inputs, int(inputs["input_ids"].shape[1])


def _generate_kwargs(max_new_tokens: int, temperature: float) -> dict[str, Any]:
    """统一生成参数（小模型下适度提高重复惩罚）。"""
    return {
        "max_new_tokens": max_new_tokens,
        "do_sample": temperature > 0,
        "temperature": max(temperature, 0.01),
        "top_p": 0.9,
        "repetition_penalty": 1.08,
        "pad_token_id": _state["tokenizer"].eos_token_id,
    }


@app.get("/v1/models")
async def models() -> dict[str, Any]:
    """OpenAI 兼容的模型列表（部分客户端会探测）。"""
    return {
        "object": "list",
        "data": [{"id": "local-model", "object": "model", "created": int(time.time())}],
    }


@app.get("/health")
async def health() -> dict[str, str]:
    """健康检查。"""
    return {"status": "ok", "model": _state["path"]}


@app.post("/v1/chat/completions")
async def chat_completions(request: Request) -> Any:
    """OpenAI 兼容对话接口（非流式 + 流式）。"""
    body = await request.json()
    model_name = body.get("model") or "local-model"
    messages = body.get("messages") or []
    stream = bool(body.get("stream"))
    max_new_tokens = int(body.get("max_tokens") or 200)
    temperature = float(body.get("temperature", 0.3) or 0.3)
    completion_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
    created = int(time.time())
    inputs, prompt_tokens = _prompt(messages)

    if stream:

        def event_stream() -> Iterator[str]:
            """逐 token 输出 SSE（OpenAI chunk 格式）。"""
            streamer = TextIteratorStreamer(
                _state["tokenizer"], skip_prompt=True, skip_special_tokens=True
            )
            kwargs = {
                **inputs,
                "streamer": streamer,
                **_generate_kwargs(max_new_tokens, temperature),
            }
            thread = threading.Thread(target=_state["model"].generate, kwargs=kwargs, daemon=True)
            thread.start()

            def chunk(delta: dict[str, Any], finish: str | None = None) -> str:
                payload = {
                    "id": completion_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": model_name,
                    "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
                }
                return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"

            yield chunk({"role": "assistant", "content": ""})
            for piece in streamer:
                if piece:
                    yield chunk({"content": piece})
            yield chunk({}, "stop")
            yield "data: [DONE]\n\n"

        return StreamingResponse(event_stream(), media_type="text/event-stream")

    with torch.no_grad():
        output = _state["model"].generate(**inputs, **_generate_kwargs(max_new_tokens, temperature))
    generated = output[0][prompt_tokens:]
    content = _state["tokenizer"].decode(generated, skip_special_tokens=True).strip()
    return {
        "id": completion_id,
        "object": "chat.completion",
        "created": created,
        "model": model_name,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": int(len(generated)),
            "total_tokens": prompt_tokens + int(len(generated)),
        },
    }


def main() -> None:
    """启动服务。"""
    parser = argparse.ArgumentParser(description="本地 OpenAI 兼容 LLM 服务")
    parser.add_argument("--model", default="models/qwen2.5-0.5b-instruct")
    parser.add_argument("--port", type=int, default=11435)
    parser.add_argument("--host", default="127.0.0.1")
    args = parser.parse_args()
    _load(args.model)
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()

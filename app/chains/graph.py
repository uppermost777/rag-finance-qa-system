"""问答编排：LangGraph 状态机（多轮改写 → 混合检索 → 重排 → 合规 → 生成 → 引用溯源）。

节点与边（合规检查前移到检索之前：命中风控的提问无需浪费检索与模型开销，
详见 DESIGN.md 3.2）：

    check_compliance ──BLOCK──▶ END
          │PASS / REWRITE
          ▼
     rewrite_query → retrieve → rerank → generate → attach_citations → END

另提供 ``astream_answer`` 异步生成器：先推 ``citations``（前端可立即展示来源），
再逐 token 推 ``delta``，最后推 ``done``；与状态图共用同一套节点实现，避免两套逻辑漂移。
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any, AsyncGenerator, Optional, Sequence, TypedDict

from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser
from langgraph.graph import END, StateGraph

from app.chains.prompts import GUARD_PROMPT, NO_CONTEXT_ANSWER, RAG_PROMPT, REWRITE_PROMPT
from app.core.config import Settings, get_settings
from app.core.logging import get_logger
from app.guard.compliance import ComplianceGuard, GuardResult, get_guard
from app.retrieval.hybrid import HybridRetriever
from app.retrieval.reranker import RerankerService, get_reranker
from app.retrieval.types import RetrievedChunk

logger = get_logger(__name__)

# 送入 prompt 的上下文长度上限（字符），超出时按重排分数顺序截断
MAX_CONTEXT_CHARS = 3000
# 历史消息格式化时最多带入的轮数
MAX_HISTORY_TURNS = 6


class QaState(TypedDict, total=False):
    """LangGraph 状态定义。"""

    question: str
    history: list[dict[str, Any]]
    kb_ids: list[int]
    strategy: str
    rewritten_query: str
    chunks: list[RetrievedChunk]
    citations: list[dict[str, Any]]
    context: str
    answer: str
    blocked: bool
    guard_action: str
    guard_message: Optional[str]
    warn_message: Optional[str]
    need_human_hint: bool
    llm_calls: int
    trace: dict[str, Any]
    retrieve_ms: int
    rerank_ms: int
    generate_ms: int


@dataclass
class QaResult:
    """一次问答的完整结果（供服务层落库）。"""

    answer: str = ""
    citations: list[dict[str, Any]] = field(default_factory=list)
    chunks: list[RetrievedChunk] = field(default_factory=list)
    context: str = ""
    rewritten_query: str = ""
    blocked: bool = False
    guard_action: str = "PASS"
    guard_message: Optional[str] = None
    warn_message: Optional[str] = None
    need_human_hint: bool = False
    strategy: str = ""
    trace: dict[str, Any] = field(default_factory=dict)
    retrieve_ms: int = 0
    rerank_ms: int = 0
    generate_ms: int = 0


def format_history(history: Optional[Sequence[dict[str, Any]]]) -> str:
    """把历史消息格式化为 Prompt 片段（只保留最近 N 轮）。"""
    if not history:
        return "（无）"
    recent = list(history)[-MAX_HISTORY_TURNS * 2 :]
    lines: list[str] = []
    for item in recent:
        role = "用户" if item.get("role") == "user" else "助手"
        content = str(item.get("content", "")).strip().replace("\n", " ")
        if content:
            lines.append(f"{role}：{content[:200]}")
    return "\n".join(lines) if lines else "（无）"


def build_context(chunks: Sequence[RetrievedChunk], max_chars: int = MAX_CONTEXT_CHARS) -> str:
    """拼接检索上下文：编号 + 来源 + 页码 + 内容，超长按顺序截断。"""
    parts: list[str] = []
    used = 0
    for index, chunk in enumerate(chunks, start=1):
        header = f"[{index}] 来源：{chunk.source or '未知文件'}"
        if chunk.page_no:
            header += f" 第 {chunk.page_no} 页"
        snippet = (chunk.content or "").strip()
        block = f"{header}\n{snippet}"
        if used + len(block) > max_chars and parts:
            break
        parts.append(block)
        used += len(block)
    return "\n\n".join(parts)


def build_citations(chunks: Sequence[RetrievedChunk]) -> list[dict[str, Any]]:
    """构造引用溯源列表，编号与上下文中的 [n] 严格一一对应。"""
    citations: list[dict[str, Any]] = []
    for index, chunk in enumerate(chunks, start=1):
        if chunk.rerank_score is not None:
            score = chunk.rerank_score
        elif chunk.rrf_score is not None:
            score = chunk.rrf_score
        else:
            score = chunk.score
        citations.append(
            {
                "index": index,
                "chunk_id": chunk.chunk_id,
                "doc_id": chunk.doc_id,
                "kb_id": chunk.kb_id,
                "doc_name": chunk.source,
                "page_no": chunk.page_no,
                "snippet": (chunk.content or "")[:200],
                "score": round(float(score or 0.0), 6),
            }
        )
    return citations


class QaPipeline:
    """问答流水线：LangGraph 节点实现 + 流式生成器。"""

    def __init__(
        self,
        *,
        retriever: Optional[HybridRetriever] = None,
        reranker: Optional[RerankerService] = None,
        guard: Optional[ComplianceGuard] = None,
        llm: Optional[BaseChatModel] = None,
        settings: Optional[Settings] = None,
        use_llm_guard: bool = False,
    ) -> None:
        """初始化流水线（所有依赖均可注入，便于测试替换）。"""
        self._settings = settings or get_settings()
        self._retriever = retriever
        self._reranker = reranker
        self._guard = guard
        self._llm = llm
        self._use_llm_guard = use_llm_guard
        self.last_state: dict[str, Any] = {}
        self._graph = self._build_graph()

    # ------------------------------------------------------------------ 依赖
    @property
    def retriever(self) -> HybridRetriever:
        """检索器（未注入时懒加载默认配置）。"""
        if self._retriever is None:
            self._retriever = HybridRetriever()
        return self._retriever

    @property
    def guard(self) -> ComplianceGuard:
        """合规守卫（未注入时取进程级实例）。"""
        if self._guard is None:
            self._guard = get_guard()
        return self._guard

    @property
    def llm(self) -> BaseChatModel:
        """大模型（未注入时按配置构造 OpenAI 兼容客户端）。"""
        if self._llm is None:
            self._llm = self._build_default_llm()
        return self._llm

    def _build_default_llm(self) -> BaseChatModel:
        """按配置构造 ChatOpenAI（DeepSeek / 通义千问等 OpenAI 兼容接口）。"""
        from langchain_openai import ChatOpenAI

        settings = self._settings
        logger.info("初始化 LLM：model=%s base_url=%s", settings.llm_model, settings.llm_base_url)
        return ChatOpenAI(
            model=settings.llm_model,
            api_key=settings.llm_api_key,  # type: ignore[arg-type]
            base_url=settings.llm_base_url,
            temperature=settings.llm_temperature,
            max_tokens=settings.llm_max_tokens,
            timeout=settings.llm_timeout,
            max_retries=settings.llm_max_retries,
        )

    def _normalize_strategy(self, strategy: Optional[str]) -> tuple[str, str]:
        """返回 (对外策略名, 检索层策略名)。

        ``hybrid_rerank`` 在检索层拆成「hybrid 召回」+「独立重排节点」两步执行，
        以便重排耗时单独计量、也可单独关闭（论文对比实验 A/B/C/D 组）。
        """
        chosen = (strategy or self._settings.default_strategy).lower()
        if chosen not in ("vector", "bm25", "hybrid", "hybrid_rerank"):
            chosen = self._settings.default_strategy
        retrieval_strategy = "hybrid" if chosen == "hybrid_rerank" else chosen
        return chosen, retrieval_strategy

    # ------------------------------------------------------------------ 节点实现
    async def node_check_compliance(self, state: QaState) -> dict[str, Any]:
        """合规检查节点：BLOCK 直接短路；REWRITE 替换查询；其余放行。"""
        question = state.get("question", "")
        result: GuardResult = self.guard.check(question)
        update: dict[str, Any] = {"guard_action": result.action, "need_human_hint": False}
        if result.blocked:
            update.update(
                {
                    "blocked": True,
                    "answer": result.message or "",
                    "guard_message": result.message,
                    "rewritten_query": question,
                    "citations": [],
                    "chunks": [],
                }
            )
            logger.info("问答被合规拦截：%s", question[:40])
            return update

        if result.action == "REWRITE":
            update["rewritten_query"] = result.rewritten_query or question
            update["guard_message"] = "已将问题改写为合规问法"
        if result.warned and result.warn_message:
            update["warn_message"] = result.warn_message

        if self._use_llm_guard and not result.matched_rules:
            if await self._llm_guard_check(question):
                update.update({"blocked": True, "guard_action": "BLOCK", "chunks": [], "citations": []})
                update["answer"] = (
                    "抱歉，这个问题涉及投资建议，我不能直接回答，也不提供任何具体产品推荐。"
                )
        return update

    async def _llm_guard_check(self, question: str) -> bool:
        """可选的 LLM 二次合规判断（默认关闭，用于论文消融实验）。"""
        try:
            raw = await (GUARD_PROMPT | self.llm | StrOutputParser()).ainvoke({"question": question})
            start, end = raw.find("{"), raw.rfind("}")
            payload = json.loads(raw[start : end + 1]) if start >= 0 and end > start else {}
            return bool(payload.get("sensitive"))
        except Exception as exc:  # pragma: no cover - 模型不可用时不影响主流程
            logger.warning("LLM 合规判断失败，忽略该结果：%s", exc)
            return False

    async def node_rewrite_query(self, state: QaState) -> dict[str, Any]:
        """查询改写节点：结合多轮历史把问题补全为可独立检索的问句。"""
        question = state.get("question", "")
        base = state.get("rewritten_query") or question
        history = state.get("history") or []
        if not history:
            return {"rewritten_query": base}
        try:
            rewritten = await (REWRITE_PROMPT | self.llm | StrOutputParser()).ainvoke(
                {"history": format_history(history), "question": base}
            )
            rewritten = (rewritten or "").strip().strip("“”\"'") or base
            logger.info("查询改写：%s -> %s", base[:40], rewritten[:60])
            return {"rewritten_query": rewritten, "llm_calls": state.get("llm_calls", 0) + 1}
        except Exception as exc:  # pragma: no cover - 模型异常时降级
            logger.warning("查询改写失败，使用原问题：%s", exc)
            return {"rewritten_query": base}

    async def node_retrieve(self, state: QaState) -> dict[str, Any]:
        """检索节点：执行向量 / BM25 / 混合召回，并记录 trace。"""
        _, retrieval_strategy = self._normalize_strategy(state.get("strategy"))
        query = state.get("rewritten_query") or state.get("question", "")
        started = time.perf_counter()
        outcome = await self.retriever.retrieve_with_trace(
            query, kb_ids=state.get("kb_ids") or None, strategy=retrieval_strategy
        )
        return {
            "chunks": outcome.chunks,
            "trace": outcome.trace.to_dict(),
            "retrieve_ms": int((time.perf_counter() - started) * 1000),
        }

    async def node_rerank(self, state: QaState) -> dict[str, Any]:
        """重排节点：交叉编码器重排并截断，低分候选被过滤（触发「未收录」兜底）。"""
        chunks = list(state.get("chunks") or [])
        chosen, _ = self._normalize_strategy(state.get("strategy"))
        query = state.get("rewritten_query") or state.get("question", "")
        started = time.perf_counter()
        update: dict[str, Any] = {}
        if chosen == "hybrid_rerank" and self._reranker is not None and chunks:
            chunks = await self._reranker.rerank(query, chunks, top_n=self._settings.rerank_top_n)
            chunks = RerankerService.filter_by_score(
                chunks, min_score=self._settings.min_rerank_score
            )
        else:
            chunks = chunks[: self._settings.rerank_top_n]
        update["rerank_ms"] = int((time.perf_counter() - started) * 1000)
        update["chunks"] = chunks
        if not chunks:
            update["need_human_hint"] = True
        return update

    async def node_generate(self, state: QaState) -> dict[str, Any]:
        """生成节点：基于检索上下文作答（流式路径见 ``astream_answer``）。"""
        if state.get("blocked"):
            return {}
        chunks = list(state.get("chunks") or [])
        started = time.perf_counter()
        if not chunks:
            return {
                "answer": NO_CONTEXT_ANSWER,
                "need_human_hint": True,
                "generate_ms": int((time.perf_counter() - started) * 1000),
            }
        context = build_context(chunks)
        question = state.get("rewritten_query") or state.get("question", "")
        answer = await (RAG_PROMPT | self.llm | StrOutputParser()).ainvoke(
            {"context": context, "question": question}
        )
        return {
            "context": context,
            "answer": (answer or "").strip(),
            "llm_calls": state.get("llm_calls", 0) + 1,
            "generate_ms": int((time.perf_counter() - started) * 1000),
        }

    async def node_attach_citations(self, state: QaState) -> dict[str, Any]:
        """引用溯源节点：生成 citations，并做输出侧合规检查。"""
        if state.get("blocked"):
            return {"citations": [], "answer": state.get("answer", "")}
        citations = build_citations(list(state.get("chunks") or []))
        answer = state.get("answer", "") or ""
        warn_message = state.get("warn_message")
        if warn_message:
            answer = f"{answer}\n\n{warn_message}"
        answer, _ = self.guard.check_output(answer)
        return {"citations": citations, "answer": answer}

    # ------------------------------------------------------------------ 图构建
    def _build_graph(self) -> Any:
        """构建并编译 LangGraph 状态图。"""
        builder = StateGraph(QaState)
        builder.add_node("check_compliance", self.node_check_compliance)
        builder.add_node("rewrite_query", self.node_rewrite_query)
        builder.add_node("retrieve", self.node_retrieve)
        builder.add_node("rerank", self.node_rerank)
        builder.add_node("generate", self.node_generate)
        builder.add_node("attach_citations", self.node_attach_citations)

        builder.set_entry_point("check_compliance")
        builder.add_conditional_edges(
            "check_compliance",
            lambda state: "end" if state.get("blocked") else "continue",
            {"end": END, "continue": "rewrite_query"},
        )
        builder.add_edge("rewrite_query", "retrieve")
        builder.add_edge("retrieve", "rerank")
        builder.add_edge("rerank", "generate")
        builder.add_edge("generate", "attach_citations")
        builder.add_edge("attach_citations", END)
        return builder.compile()

    @property
    def graph(self) -> Any:
        """编译后的状态图（测试与调试用）。"""
        return self._graph

    # ------------------------------------------------------------------ 非流式
    async def aanswer(
        self,
        question: str,
        *,
        history: Optional[list[dict[str, Any]]] = None,
        kb_ids: Optional[Sequence[int]] = None,
        strategy: Optional[str] = None,
    ) -> QaResult:
        """非流式问答：执行完整状态图并返回结果。"""
        chosen, _ = self._normalize_strategy(strategy)
        initial: QaState = {
            "question": question,
            "history": list(history or []),
            "kb_ids": [int(item) for item in (kb_ids or [])],
            "strategy": chosen,
            "chunks": [],
            "citations": [],
            "blocked": False,
            "llm_calls": 0,
        }
        final: dict[str, Any] = await self._graph.ainvoke(initial)
        self.last_state = dict(final)
        return self._to_result(final, chosen)

    @staticmethod
    def _to_result(final: dict[str, Any], strategy: str) -> QaResult:
        """把图状态转换为 QaResult。"""
        return QaResult(
            answer=final.get("answer", "") or "",
            citations=list(final.get("citations") or []),
            chunks=list(final.get("chunks") or []),
            context=final.get("context", "") or "",
            rewritten_query=final.get("rewritten_query", "") or "",
            blocked=bool(final.get("blocked")),
            guard_action=final.get("guard_action", "PASS"),
            guard_message=final.get("guard_message"),
            warn_message=final.get("warn_message"),
            need_human_hint=bool(final.get("need_human_hint")),
            strategy=strategy,
            trace=dict(final.get("trace") or {}),
            retrieve_ms=int(final.get("retrieve_ms") or 0),
            rerank_ms=int(final.get("rerank_ms") or 0),
            generate_ms=int(final.get("generate_ms") or 0),
        )

    # ------------------------------------------------------------------ 流式
    async def astream_answer(
        self,
        question: str,
        *,
        history: Optional[list[dict[str, Any]]] = None,
        kb_ids: Optional[Sequence[int]] = None,
        strategy: Optional[str] = None,
    ) -> AsyncGenerator[tuple[str, dict[str, Any]], None]:
        """流式问答：依次产出 ``citations`` → ``delta``* → ``done``（异常时 ``error``）。"""
        chosen, _ = self._normalize_strategy(strategy)
        state: QaState = {
            "question": question,
            "history": list(history or []),
            "kb_ids": [int(item) for item in (kb_ids or [])],
            "strategy": chosen,
            "chunks": [],
            "citations": [],
            "blocked": False,
            "llm_calls": 0,
        }
        total_started = time.perf_counter()
        try:
            state.update(await self.node_check_compliance(state))
            if state.get("blocked"):
                answer = state.get("answer", "")
                yield "delta", {"text": answer}
                yield "done", {
                    "blocked": True,
                    "guard_action": state.get("guard_action", "BLOCK"),
                    "citations": [],
                    "answer": answer,
                    "latency_ms": int((time.perf_counter() - total_started) * 1000),
                }
                return

            state.update(await self.node_rewrite_query(state))
            state.update(await self.node_retrieve(state))
            state.update(await self.node_rerank(state))

            citations = build_citations(list(state.get("chunks") or []))
            yield "citations", {"citations": citations}

            chunks = list(state.get("chunks") or [])
            answer = ""
            generate_started = time.perf_counter()
            if not chunks:
                answer = NO_CONTEXT_ANSWER
                yield "delta", {"text": answer}
            else:
                context = build_context(chunks)
                state["context"] = context
                prompt_value = RAG_PROMPT.invoke(
                    {"context": context, "question": state.get("rewritten_query") or question}
                )
                try:
                    async for piece in self.llm.astream(prompt_value):
                        text = getattr(piece, "content", "") or ""
                        if text:
                            answer += text
                            yield "delta", {"text": text}
                except Exception as exc:
                    logger.error("模型流式生成失败：%s", exc)
                    yield "error", {"message": f"生成中断：{exc}"}
            state["generate_ms"] = int((time.perf_counter() - generate_started) * 1000)
            state["answer"] = answer

            state.update(await self.node_attach_citations(state))
            yield "done", {
                "blocked": False,
                "guard_action": state.get("guard_action", "PASS"),
                "citations": citations,
                "answer": state.get("answer", "") or "",
                "rewritten_query": state.get("rewritten_query", ""),
                "need_human_hint": bool(state.get("need_human_hint")),
                "latency_ms": int((time.perf_counter() - total_started) * 1000),
                "retrieve_ms": int(state.get("retrieve_ms") or 0),
                "rerank_ms": int(state.get("rerank_ms") or 0),
                "generate_ms": int(state.get("generate_ms") or 0),
                "trace": dict(state.get("trace") or {}),
            }
        except Exception as exc:  # pragma: no cover - 兜底
            logger.exception("流式问答失败：%s", exc)
            yield "error", {"message": f"问答失败：{exc}"}
        finally:
            self.last_state = dict(state)


_pipeline: Optional[QaPipeline] = None


def get_pipeline() -> QaPipeline:
    """获取进程级默认流水线（FastAPI 与 Celery 共用）。"""
    global _pipeline
    if _pipeline is None:
        # 注入重排服务单例：服务对象本身只持有配置，模型在首次重排时才懒加载，
        # 因此默认 hybrid 策略不会产生额外内存开销。
        _pipeline = QaPipeline(reranker=get_reranker())
    return _pipeline


def set_pipeline(pipeline: Optional[QaPipeline]) -> None:
    """替换进程级流水线（测试注入用）。"""
    global _pipeline
    _pipeline = pipeline

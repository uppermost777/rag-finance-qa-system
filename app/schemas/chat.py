"""会话与问答相关请求/响应模型。"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field


class ConversationCreateRequest(BaseModel):
    """新建会话。"""

    title: Optional[str] = Field(default=None, max_length=128)
    kb_ids: list[int] = Field(default_factory=list, description="检索范围：知识库 ID 列表")


class ConversationUpdateRequest(BaseModel):
    """修改会话（重命名 / 调整知识库范围）。"""

    title: Optional[str] = Field(default=None, max_length=128)
    kb_ids: Optional[list[int]] = None


class ConversationItem(BaseModel):
    """会话列表项。"""

    id: int
    title: str
    kb_ids: list[int] = Field(default_factory=list)
    message_count: int = 0
    last_message_at: Optional[datetime] = None
    created_at: Optional[datetime] = None


class ConversationListResponse(BaseModel):
    """会话分页列表。"""

    total: int
    page: int
    size: int
    items: list[ConversationItem] = Field(default_factory=list)


class MessageItem(BaseModel):
    """消息项。"""

    id: int
    role: str
    content: str
    citations: list[dict[str, Any]] = Field(default_factory=list)
    latency_ms: Optional[int] = None
    blocked: bool = False
    feedback: Optional[int] = None
    created_at: Optional[datetime] = None


class MessageListResponse(BaseModel):
    """消息分页列表。"""

    total: int
    page: int
    size: int
    items: list[MessageItem] = Field(default_factory=list)


StrategyName = Literal["vector", "bm25", "hybrid", "hybrid_rerank"]


class ChatRequest(BaseModel):
    """非流式问答请求。"""

    question: str = Field(..., min_length=1, max_length=500, description="用户问题")
    conversation_id: Optional[int] = Field(default=None, description="为空则新建会话")
    kb_ids: Optional[list[int]] = Field(default=None, description="检索范围，为空表示全部知识库")
    strategy: Optional[StrategyName] = Field(default=None, description="检索策略（论文对比实验用）")


class ChatResponse(BaseModel):
    """非流式问答响应。"""

    conversation_id: int
    message_id: int
    user_message_id: int
    qa_log_id: Optional[int] = None
    answer: str
    citations: list[dict[str, Any]] = Field(default_factory=list)
    blocked: bool = False
    guard_action: str = "PASS"
    strategy: str = ""
    need_human_hint: bool = False
    latency_ms: int = 0


class FeedbackRequest(BaseModel):
    """点赞 / 点踩。"""

    feedback: Literal[1, -1] = Field(..., description="1 点赞，-1 点踩")


class FeedbackResponse(BaseModel):
    """反馈结果。"""

    message_id: int
    feedback: int

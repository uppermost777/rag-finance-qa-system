"""知识库相关请求/响应模型。"""

from __future__ import annotations

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field


class KbCreateRequest(BaseModel):
    """新建知识库。"""

    name: str = Field(..., min_length=2, max_length=128)
    category: str = Field(..., min_length=2, max_length=64)
    description: Optional[str] = Field(default=None, max_length=512)


class KbUpdateRequest(BaseModel):
    """修改知识库。"""

    name: Optional[str] = Field(default=None, min_length=2, max_length=128)
    description: Optional[str] = Field(default=None, max_length=512)
    category: Optional[str] = Field(default=None, min_length=2, max_length=64)
    status: Optional[int] = Field(default=None, ge=0, le=1)


class KbItem(BaseModel):
    """知识库项。"""

    id: int
    name: str
    category: str
    description: Optional[str] = None
    doc_count: int = 0
    chunk_count: int = 0
    status: int = 1
    created_at: Optional[datetime] = None


class KbListResponse(BaseModel):
    """知识库分页列表。"""

    total: int
    page: int
    size: int
    items: list[KbItem] = Field(default_factory=list)

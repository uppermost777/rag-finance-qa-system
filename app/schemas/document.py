"""文档管理相关请求/响应模型。"""

from __future__ import annotations

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field


class DocumentItem(BaseModel):
    """文档项。"""

    id: int
    kb_id: int
    name: str
    file_type: str
    file_size: int = 0
    status: str
    progress: int = 0
    page_count: int = 0
    chunk_count: int = 0
    error_msg: Optional[str] = None
    created_at: Optional[datetime] = None


class DocumentListResponse(BaseModel):
    """文档分页列表。"""

    total: int
    page: int
    size: int
    items: list[DocumentItem] = Field(default_factory=list)


class UploadResponse(BaseModel):
    """上传响应（异步入库，立即返回）。"""

    doc_id: int
    task_id: Optional[str] = None
    status: str = "pending"
    message: str = "文件已接收，正在后台解析入库"


class ProgressResponse(BaseModel):
    """入库进度。"""

    doc_id: int
    status: str
    progress: int = 0
    chunk_count: int = 0
    message: Optional[str] = None
    error_msg: Optional[str] = None


class ReindexResponse(BaseModel):
    """重建索引响应。"""

    doc_id: Optional[int] = None
    kb_id: Optional[int] = None
    task_id: Optional[str] = None
    message: str = "重建任务已提交"

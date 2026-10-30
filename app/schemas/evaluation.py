"""评估相关请求/响应模型。"""

from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, Field


class EvalRunRequest(BaseModel):
    """触发评估任务。"""

    dataset: str = Field(default="finance_qa_eval.jsonl", description="评测集文件名")
    limit: Optional[int] = Field(default=None, ge=1, le=1000, description="只跑前 N 条")
    strategies: list[str] = Field(
        default_factory=lambda: ["vector", "bm25", "hybrid", "hybrid_rerank"],
        description="参与对比的检索策略",
    )


class EvalRunResponse(BaseModel):
    """评估任务提交结果。"""

    run_id: str
    status: str = "PENDING"
    message: str = "评估任务已提交"


class EvalStatusResponse(BaseModel):
    """评估任务状态。"""

    run_id: str
    status: str
    progress: int = 0
    report_path: Optional[str] = None
    message: Optional[str] = None


class EvalReportResponse(BaseModel):
    """评估报告（CSV 解析结果，供前端画图）。"""

    run_id: Optional[str] = None
    columns: list[str] = Field(default_factory=list)
    rows: list[dict[str, Any]] = Field(default_factory=list)

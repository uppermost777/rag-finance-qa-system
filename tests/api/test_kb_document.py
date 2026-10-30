"""知识库与文档管理接口 smoke test（写操作鉴权、异步入库投递、进度查询）。"""

from __future__ import annotations

from typing import Any

import pytest
from httpx import AsyncClient

from tests.api.conftest import auth_headers


async def test_kb_list_visible_to_student(client: AsyncClient, seed: dict[str, int]) -> None:
    """学生可查看知识库列表（用于选择检索范围）。"""
    resp = await client.get("/api/v1/kb", headers=auth_headers(seed["student_id"]))
    assert resp.status_code == 200
    body = resp.json()["data"]
    assert body["total"] == 1
    assert body["items"][0]["name"] == "API 测试知识库"


async def test_kb_write_requires_admin(client: AsyncClient, seed: dict[str, int]) -> None:
    """学生新建知识库返回 403。"""
    resp = await client.post(
        "/api/v1/kb",
        json={"name": "新知识库", "category": "保险保障"},
        headers=auth_headers(seed["student_id"]),
    )
    assert resp.status_code == 403
    assert resp.json()["code"] == 40301


async def test_kb_crud_as_admin(client: AsyncClient, seed: dict[str, int]) -> None:
    """管理员：新建 → 修改 → 删除知识库。"""
    headers = auth_headers(seed["admin_id"], role="admin")
    created = await client.post(
        "/api/v1/kb",
        json={"name": "反诈防骗常识", "category": "反诈防骗", "description": "校园常见诈骗手法"},
        headers=headers,
    )
    assert created.status_code == 200
    kb_id = created.json()["data"]["id"]

    duplicate = await client.post(
        "/api/v1/kb", json={"name": "反诈防骗常识", "category": "反诈防骗"}, headers=headers
    )
    assert duplicate.status_code == 409

    updated = await client.patch(
        f"/api/v1/kb/{kb_id}", json={"description": "已更新描述", "status": 0}, headers=headers
    )
    assert updated.status_code == 200
    assert updated.json()["data"]["description"] == "已更新描述"
    assert updated.json()["data"]["status"] == 0

    deleted = await client.delete(f"/api/v1/kb/{kb_id}", headers=headers)
    assert deleted.status_code == 200

    missing = await client.patch(f"/api/v1/kb/{kb_id}", json={"status": 1}, headers=headers)
    assert missing.status_code == 404


async def test_document_upload_flow(
    client: AsyncClient, seed: dict[str, int], monkeypatch: pytest.MonkeyPatch, tmp_path: Any
) -> None:
    """上传 → 列表 → 进度 → 重建 → 删除 的完整闭环（任务投递被打桩）。"""
    monkeypatch.setattr("app.api.v1.document.dispatch_ingest", lambda doc_id: "task-upload-1")
    monkeypatch.setattr(
        "app.api.v1.document.dispatch_reindex", lambda kb_id=None, doc_id=None: "task-reindex-1"
    )
    from app.core.config import get_settings

    get_settings().upload_dir = str(tmp_path)
    headers = auth_headers(seed["admin_id"], role="admin")

    upload = await client.post(
        "/api/v1/document/upload",
        files={
            "file": (
                "基金定投入门.txt",
                "基金定投是指在固定时间投入固定金额。".encode("utf-8"),
                "text/plain",
            )
        },
        data={"kb_id": str(seed["kb_id"])},
        headers=headers,
    )
    assert upload.status_code == 200, upload.text
    payload = upload.json()["data"]
    assert payload["task_id"] == "task-upload-1"
    assert payload["status"] == "pending"
    doc_id = payload["doc_id"]

    listed = await client.get(
        f"/api/v1/document?kb_id={seed['kb_id']}", headers=headers
    )
    assert listed.status_code == 200
    assert listed.json()["data"]["total"] == 1
    assert listed.json()["data"]["items"][0]["name"] == "基金定投入门.txt"

    progress = await client.get(f"/api/v1/document/{doc_id}/progress", headers=headers)
    assert progress.status_code == 200
    assert progress.json()["data"]["status"] == "pending"

    reindex = await client.post(f"/api/v1/document/{doc_id}/reindex", headers=headers)
    assert reindex.status_code == 200
    assert reindex.json()["data"]["task_id"] == "task-reindex-1"

    reindex_kb = await client.post(
        f"/api/v1/document/reindex?kb_id={seed['kb_id']}", headers=headers
    )
    assert reindex_kb.status_code == 200
    assert reindex_kb.json()["data"]["kb_id"] == seed["kb_id"]

    deleted = await client.delete(f"/api/v1/document/{doc_id}", headers=headers)
    assert deleted.status_code == 200
    assert deleted.json()["data"]["doc_id"] == doc_id

    after = await client.get(f"/api/v1/document?kb_id={seed['kb_id']}", headers=headers)
    assert after.json()["data"]["total"] == 0


async def test_document_upload_rejections(
    client: AsyncClient, seed: dict[str, int], monkeypatch: pytest.MonkeyPatch, tmp_path: Any
) -> None:
    """拒绝：非管理员、非法扩展名、空文件、超限大小、重复文件。"""
    monkeypatch.setattr("app.api.v1.document.dispatch_ingest", lambda doc_id: "task-x")
    from app.core.config import get_settings

    get_settings().upload_dir = str(tmp_path)
    admin_headers = auth_headers(seed["admin_id"], role="admin")
    student_headers = auth_headers(seed["student_id"])

    forbidden = await client.post(
        "/api/v1/document/upload",
        files={"file": ("a.txt", b"hello", "text/plain")},
        data={"kb_id": str(seed["kb_id"])},
        headers=student_headers,
    )
    assert forbidden.status_code == 403

    bad_ext = await client.post(
        "/api/v1/document/upload",
        files={"file": ("恶意.xlsx", b"data", "application/vnd.ms-excel")},
        data={"kb_id": str(seed["kb_id"])},
        headers=admin_headers,
    )
    assert bad_ext.status_code == 400

    empty = await client.post(
        "/api/v1/document/upload",
        files={"file": ("空.txt", b"", "text/plain")},
        data={"kb_id": str(seed["kb_id"])},
        headers=admin_headers,
    )
    assert empty.status_code == 400

    path_traversal = await client.post(
        "/api/v1/document/upload",
        files={"file": ("../../../etc/passwd.txt", b"hacked", "text/plain")},
        data={"kb_id": str(seed["kb_id"])},
        headers=admin_headers,
    )
    assert path_traversal.status_code == 200
    assert path_traversal.json()["data"]["doc_id"] > 0
    stored = list(tmp_path.glob(f"{seed['kb_id']}/*.txt"))
    assert stored and stored[0].name != "passwd.txt"  # 文件名被清洗并重命名
    assert "/" not in stored[0].name

    duplicate = await client.post(
        "/api/v1/document/upload",
        files={"file": ("重复.txt", b"hacked", "text/plain")},
        data={"kb_id": str(seed["kb_id"])},
        headers=admin_headers,
    )
    assert duplicate.status_code == 409

    missing_kb = await client.post(
        "/api/v1/document/upload",
        files={"file": ("ok.txt", b"content", "text/plain")},
        data={"kb_id": "99999"},
        headers=admin_headers,
    )
    assert missing_kb.status_code == 404


def test_sanitize_filename_unit() -> None:
    """文件名清洗：去掉路径、替换非法字符、保留中文与扩展名。"""
    from app.core.exceptions import ParamError
    from app.services.document_service import sanitize_filename, validate_extension

    assert sanitize_filename("../../etc/passwd.txt") == "passwd.txt"
    assert sanitize_filename("C:\\Users\\a\\基金 定投(2026).pdf") == "基金 定投(2026).pdf"
    assert sanitize_filename("a<>b|c.txt") == "a_b_c.txt"
    with pytest.raises(ParamError):
        sanitize_filename("///")
    assert validate_extension("A.PDF") == "pdf"
    with pytest.raises(ParamError):
        validate_extension("a.exe")

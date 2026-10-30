"""文档解析与清洗：按扩展名分派解析器，并统一做文本清洗。

支持：``.pdf``（pdfplumber，保留页码）/ ``.docx``（python-docx）/
``.txt`` / ``.md``（直接读）/ ``.html``（BeautifulSoup 去标签）。
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Sequence

from app.core.exceptions import ParamError
from app.core.logging import get_logger

logger = get_logger(__name__)

# 页眉页脚判定阈值：同一位置行在超过该比例（且页数足够）的页面重复出现即视为页眉页脚
_REPEAT_RATIO = 0.6
_MIN_PAGES_FOR_HEADER_DETECT = 3


@dataclass(frozen=True)
class PageContent:
    """解析后的单页文本。"""

    text: str
    page_no: int
    source: str

    @property
    def char_count(self) -> int:
        """字符数。"""
        return len(self.text)


def _read_text_file(path: Path) -> str:
    """读取文本文件：优先 utf-8，失败退化为 utf-8 + errors=ignore 并告警。"""
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        logger.warning("文件 %s 非 UTF-8 编码，已按 errors=ignore 降级读取", path.name)
        return path.read_text(encoding="utf-8", errors="ignore")


def load_pdf(path: Path, source: str) -> list[PageContent]:
    """解析 PDF：逐页返回文本，保留真实页码。"""
    import pdfplumber

    pages: list[PageContent] = []
    with pdfplumber.open(str(path)) as pdf:
        for index, page in enumerate(pdf.pages, start=1):
            text = page.extract_text() or ""
            pages.append(PageContent(text=text, page_no=index, source=source))
    logger.info("PDF 解析完成：%s，共 %d 页", source, len(pages))
    return pages


def load_docx(path: Path, source: str) -> list[PageContent]:
    """解析 DOCX：按段落取文本（Word 无固定页码，统一记为第 1 页）。"""
    import docx

    document = docx.Document(str(path))
    paragraphs = [para.text for para in document.paragraphs if para.text and para.text.strip()]
    for table in document.tables:
        for row in table.rows:
            cells = [cell.text.strip() for cell in row.cells if cell.text and cell.text.strip()]
            if cells:
                paragraphs.append(" | ".join(cells))
    text = "\n".join(paragraphs)
    logger.info("DOCX 解析完成：%s，%d 个段落块", source, len(paragraphs))
    return [PageContent(text=text, page_no=1, source=source)]


def load_text(path: Path, source: str) -> list[PageContent]:
    """解析纯文本 / Markdown。"""
    text = _read_text_file(path)
    logger.info("文本解析完成：%s，%d 字符", source, len(text))
    return [PageContent(text=text, page_no=1, source=source)]


def load_html(path: Path, source: str) -> list[PageContent]:
    """解析 HTML：用 BeautifulSoup 去标签，先剔除 script/style。"""
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(_read_text_file(path), "html.parser")
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()
    text = soup.get_text("\n")
    logger.info("HTML 解析完成：%s，%d 字符", source, len(text))
    return [PageContent(text=text, page_no=1, source=source)]


_LOADER_MAP = {
    ".pdf": load_pdf,
    ".docx": load_docx,
    ".txt": load_text,
    ".md": load_text,
    ".markdown": load_text,
    ".html": load_html,
    ".htm": load_html,
}


def load_file(path: Path | str, source: Optional[str] = None) -> list[PageContent]:
    """按扩展名分派解析器。

    :param path: 文件路径
    :param source: 溯源展示名（默认取文件名）
    :return: 逐页文本列表
    """
    file_path = Path(path)
    if not file_path.is_file():
        raise ParamError(f"文件不存在：{file_path}")
    suffix = file_path.suffix.lower()
    loader = _LOADER_MAP.get(suffix)
    if loader is None:
        raise ParamError(f"不支持的文件类型：{suffix or '未知'}")
    return loader(file_path, source or file_path.name)


# --------------------------------------------------------------------------- 清洗
def normalize_whitespace(text: str) -> str:
    """规范化空白：全角空格与制表符统一、行尾去空格、连续空格压缩。"""
    if not text:
        return ""
    unified = text.replace("\u3000", " ").replace("\t", " ").replace("\xa0", " ")
    lines = [re.sub(r"[ ]{2,}", " ", line).rstrip() for line in unified.split("\n")]
    return "\n".join(lines)


def collapse_blank_lines(text: str) -> str:
    """压缩连续空行：3 行以上空行折叠为一个空行。"""
    return re.sub(r"\n{3,}", "\n\n", text or "")


def detect_repeated_lines(pages: Sequence[PageContent]) -> set[str]:
    """检测跨页重复的页眉/页脚行（出现在 >=60% 页面且页数 >=3）。"""
    if len(pages) < _MIN_PAGES_FOR_HEADER_DETECT:
        return set()
    counter: Counter[str] = Counter()
    for page in pages:
        lines = [line.strip() for line in page.text.split("\n") if line.strip()]
        if not lines:
            continue
        candidates = set(lines[:2] + lines[-2:])
        counter.update(candidates)
    threshold = max(2, int(len(pages) * _REPEAT_RATIO))
    repeated = {line for line, count in counter.items() if count >= threshold and len(line) <= 60}
    if repeated:
        logger.info("检测到疑似页眉页脚 %d 条：%s", len(repeated), list(repeated)[:3])
    return repeated


def strip_header_footer(
    pages: Sequence[PageContent], repeated: Optional[Iterable[str]] = None
) -> list[PageContent]:
    """删除页眉页脚行（只删每页首尾出现的重复行，不影响正文）。"""
    targets = set(repeated) if repeated is not None else detect_repeated_lines(pages)
    if not targets:
        return list(pages)
    cleaned: list[PageContent] = []
    for page in pages:
        lines = page.text.split("\n")
        kept = [
            line
            for index, line in enumerate(lines)
            if not (line.strip() in targets and (index < 2 or index >= len(lines) - 2))
        ]
        cleaned.append(PageContent(text="\n".join(kept), page_no=page.page_no, source=page.source))
    return cleaned


def clean_text(text: str) -> str:
    """对单段文本执行空白规范化 + 空行折叠。"""
    return collapse_blank_lines(normalize_whitespace(text)).strip()


def clean_pages(pages: Sequence[PageContent]) -> list[PageContent]:
    """清洗整篇文档：去页眉页脚 → 空白规范化 → 空行折叠 → 丢弃空页。"""
    cleaned = strip_header_footer(pages)
    result: list[PageContent] = []
    for page in cleaned:
        text = clean_text(page.text)
        if not text:
            continue
        result.append(PageContent(text=text, page_no=page.page_no, source=page.source))
    logger.info("清洗完成：%d 页 → %d 页有效", len(pages), len(result))
    return result


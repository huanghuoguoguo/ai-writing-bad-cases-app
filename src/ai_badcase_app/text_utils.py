"""文本预处理工具函数 - 统一被各模块复用"""

from __future__ import annotations

import re
from typing import Final

# Pre-compiled regex patterns for efficiency
_PARAGRAPH_SPLIT_RE: Final = re.compile(r"\n\s*\n")
_SENTENCE_SPLIT_RE: Final = re.compile(r"[^。！？!?；;\n]+[。！？!?；;]?")
_PARAGRAPH_SPANS_RE: Final = re.compile(r"\S(?:.*?\S)?(?=\n\s*\n|$)", re.DOTALL)
_LIST_BLOCK_RE: Final = re.compile(r"^([-*+] |\d+\. )")
_WHITESPACE_RE: Final = re.compile(r"\s+")
_STAT_SENTENCE_SPLIT_RE: Final = re.compile(r"[。！？!?\n]+")

# String constants
_FRONTMATTER_START = "---\n"
_FRONTMATTER_END = "\n---\n"
_CODE_FENCE = "```"


def strip_frontmatter(text: str) -> tuple[str, int]:
    """
    去除 YAML frontmatter，返回 (body_text, offset)。

    offset 是 frontmatter 的字符长度，用于位置计算。
    """
    if not text.startswith(_FRONTMATTER_START):
        return text, 0
    parts = text.split(_FRONTMATTER_END, 1)
    if len(parts) != 2:
        return text, 0
    offset = len(parts[0]) + len(_FRONTMATTER_END)
    return parts[1], offset


def split_frontmatter(text: str) -> tuple[str, str]:
    """
    分离 YAML frontmatter，返回 (frontmatter_str, body_str)。

    frontmatter_str 包含完整的 frontmatter（包括 --- 分隔符），
    如果没有 frontmatter 则为空字符串。
    """
    if not text.startswith(_FRONTMATTER_START):
        return "", text
    parts = text.split(_FRONTMATTER_END, 1)
    if len(parts) != 2:
        return "", text
    return parts[0] + _FRONTMATTER_END, parts[1]


def split_paragraphs(text: str) -> list[str]:
    """按空行分割段落，忽略 frontmatter"""
    body, _ = strip_frontmatter(text)
    return [chunk.strip() for chunk in _PARAGRAPH_SPLIT_RE.split(body) if chunk.strip()]


def split_sentences(text: str) -> list[str]:
    """按中文/英文句子边界分句"""
    text = text.strip()
    if not text:
        return []

    # Markdown 特殊块保持完整
    if text.startswith(_CODE_FENCE) or text.startswith("#") or text.startswith("|"):
        return [text]

    sentences = [chunk.strip() for chunk in _SENTENCE_SPLIT_RE.findall(text) if chunk.strip()]
    return sentences or [text]


def paragraph_spans(text: str) -> list[tuple[int, int]]:
    """返回每个段落的 (start, end) 字符位置"""
    body, _ = strip_frontmatter(text)
    return [(m.start(), m.end()) for m in _PARAGRAPH_SPANS_RE.finditer(body)]


def find_paragraph_index(spans: list[tuple[int, int]], pos: int) -> int | None:
    """根据字符位置找到段落索引（使用二分搜索）"""
    if not spans:
        return None
    if pos < spans[0][0]:
        return 0

    # Binary search
    left, right = 0, len(spans) - 1
    while left <= right:
        mid = (left + right) // 2
        start, end = spans[mid]
        if start <= pos < end:
            return mid
        if pos < start:
            right = mid - 1
        else:
            left = mid + 1

    if pos >= spans[-1][0]:
        return len(spans) - 1
    return None


def is_heading(paragraph: str) -> bool:
    """判断是否为 Markdown 标题"""
    return paragraph.lstrip().startswith("#")


def is_code_block(paragraph: str) -> bool:
    """判断是否为代码块"""
    stripped = paragraph.strip()
    return stripped.startswith(_CODE_FENCE) or stripped.endswith(_CODE_FENCE)


def _get_non_empty_lines(paragraph: str) -> list[str]:
    """获取段落中非空行（已去除首尾空白）"""
    return [line.strip() for line in paragraph.splitlines() if line.strip()]


def is_table_block(paragraph: str) -> bool:
    """判断是否为表格"""
    lines = _get_non_empty_lines(paragraph)
    return bool(lines) and all(line.startswith("|") for line in lines)


def is_list_block(paragraph: str) -> bool:
    """判断是否为列表"""
    lines = _get_non_empty_lines(paragraph)
    if not lines:
        return False
    return all(_LIST_BLOCK_RE.match(line) for line in lines)


def plain_text_len(text: str) -> int:
    """去除空白后的文本长度（不创建新字符串）"""
    return sum(1 for c in text if not c.isspace())


def should_run_stat_checks(paragraph: str, min_chars: int = 40, min_sentences: int = 2) -> bool:
    """
    判断是否应对该段落运行统计检测。

    跳过标题、代码块、表格、列表，以及过短的段落。
    """
    if is_heading(paragraph) or is_code_block(paragraph) or is_table_block(paragraph) or is_list_block(paragraph):
        return False

    if plain_text_len(paragraph) < min_chars:
        return False

    sentence_like_parts = [c for c in _STAT_SENTENCE_SPLIT_RE.split(paragraph) if c.strip()]
    return len(sentence_like_parts) >= min_sentences


def code_fence_indices(paragraphs: list[str]) -> set[int]:
    """返回代码块所在段落的索引集合"""
    indices: set[int] = set()
    in_code_fence = False

    for index, paragraph in enumerate(paragraphs):
        fence_count = paragraph.count(_CODE_FENCE)
        if in_code_fence or fence_count:
            indices.add(index)
        if fence_count % 2 == 1:
            in_code_fence = not in_code_fence

    return indices
from __future__ import annotations

import re

from .analyzer import TextAnalysisReport, analyze_text
from .matcher import split_paragraphs


def rewrite_text(text: str, report: TextAnalysisReport | None = None) -> str:
    """Apply lightweight local rewrites for obvious AI-style patterns."""
    report = report or analyze_text(text)
    frontmatter, body = _split_frontmatter(text)
    paragraphs = split_paragraphs(text)
    signal_map = _build_signal_map(report)

    rewritten = [
        _rewrite_paragraph(paragraph, signal_map.get(index, set()))
        for index, paragraph in enumerate(paragraphs)
    ]
    rebuilt_body = "\n\n".join(rewritten)
    return f"{frontmatter}{rebuilt_body}" if frontmatter else rebuilt_body


def _split_frontmatter(text: str) -> tuple[str, str]:
    if not text.startswith("---\n"):
        return "", text
    parts = text.split("\n---\n", 1)
    if len(parts) != 2:
        return "", text
    return parts[0] + "\n---\n", parts[1]


def _build_signal_map(report: TextAnalysisReport) -> dict[int, set[str]]:
    signal_map: dict[int, set[str]] = {}
    for segment in report.suspected_segments:
        if segment.paragraph_index is None:
            continue
        signal_map.setdefault(segment.paragraph_index, set()).update(
            signal["code"] for signal in segment.signals
        )
    return signal_map


def _rewrite_paragraph(paragraph: str, signal_codes: set[str]) -> str:
    rewritten = paragraph

    if _has_any(signal_codes, "zh.arg.not_x_but_y", "zh.arg.not_about_x_but_y"):
        rewritten = _rewrite_not_but(rewritten)

    if _has_any(signal_codes, "zh.arg.truly_important"):
        rewritten = re.sub(r"真正重要的是[，,、 ]*", "", rewritten)

    if _has_any(signal_codes, "zh.arg.meta_essence"):
        rewritten = re.sub(r"(本质上|归根结底|说白了|说穿了|换句话说)[，,、 ]*", "", rewritten)
        rewritten = re.sub(r"底层逻辑", "关键处", rewritten)

    rewritten = re.sub(r"[ \t]{2,}", " ", rewritten).strip()
    return rewritten


def _has_any(signal_codes: set[str], *expected: str) -> bool:
    return any(code in signal_codes for code in expected)


def _rewrite_not_but(text: str) -> str:
    text = re.sub(
        r"(?P<prefix>[^。！？!?]*?)真正重要的不是(?P<x>[^，。；！？!?]{1,24})[，,、 ]*而是(?P<y>[^。；！？!?]{1,48})",
        lambda m: _attach_clause(m.group("prefix"), f"更要紧的是{_clean_clause(m.group('y'))}"),
        text,
    )
    text = re.sub(
        r"(?P<prefix>[^。！？!?]*?)不是关于(?P<x>[^，。；！？!?]{1,24})[，,、 ]*而是关于(?P<y>[^。；！？!?]{1,32})",
        lambda m: _attach_clause(m.group("prefix"), _clean_clause(m.group("y"))),
        text,
    )
    text = re.sub(
        r"(?P<prefix>[^。！？!?]*?)不是(?P<x>[^，。；！？!?]{1,24})[，,、 ]*也不是(?P<m>[^，。；！？!?]{1,32})[，,、 ]*而是(?P<y>[^。；！？!?]{1,48})",
        lambda m: _attach_clause(m.group("prefix"), _clean_clause(m.group("y"))),
        text,
    )
    text = re.sub(
        r"(?P<prefix>[^。！？!?]*?)不是(?P<x>[^，。；！？!?]{1,24})[，,、 ]*而是(?P<y>[^。；！？!?]{1,48})",
        lambda m: _attach_clause(m.group("prefix"), _clean_clause(m.group("y"))),
        text,
    )
    return text


def _clean_clause(text: str) -> str:
    return text.strip(" ，,、；;：:")


def _clean_topic(text: str) -> str:
    cleaned = text.strip(" ，,、；;：:")
    cleaned = re.sub(r"^(所谓|那个|这种|这类|这个|关于)", "", cleaned)
    cleaned = re.sub(r"(问题|层面|本身)$", "", cleaned)
    return cleaned or text.strip()


def _attach_clause(prefix: str, clause: str) -> str:
    prefix = prefix.strip()
    clause = _clean_clause(clause)
    if not prefix:
        return clause
    if prefix.endswith(("的", "反而", "其实", "就是")):
        return f"{prefix}是{clause}"
    return f"{prefix}{clause}"

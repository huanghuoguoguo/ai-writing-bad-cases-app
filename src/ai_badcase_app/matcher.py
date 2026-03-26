from __future__ import annotations

import re
from dataclasses import dataclass

from rapidfuzz import fuzz

from .models import BadCase, MatchHit, ParagraphResult


@dataclass
class MatcherConfig:
    """配置 matcher 的行为参数"""

    fuzzy_threshold: int = 75
    fuzzy_algorithm: str = "ratio"
    semantic_threshold: float = 0.7
    semantic_enabled: bool = False
    word_vectors_path: str | None = None


def _strip_frontmatter(text: str) -> str:
    if not text.startswith("---\n"):
        return text
    parts = text.split("\n---\n", 1)
    if len(parts) != 2:
        return text
    return parts[1]


def split_paragraphs(text: str) -> list[str]:
    body = _strip_frontmatter(text)
    return [chunk.strip() for chunk in re.split(r"\n\s*\n", body) if chunk.strip()]


def split_sentences(text: str) -> list[str]:
    text = text.strip()
    if not text:
        return []

    # Keep markdown/code-heavy blocks intact instead of over-splitting noise.
    if text.startswith("```") or text.startswith("#") or text.startswith("|"):
        return [text]

    sentences = [
        chunk.strip()
        for chunk in re.findall(r"[^。！？!?；;\n]+[。！？!?；;]?", text)
        if chunk.strip()
    ]
    return sentences or [text]


def _match_case(case: BadCase, text: str, config: MatcherConfig | None = None) -> list[MatchHit]:
    config = config or MatcherConfig()
    hits: list[MatchHit] = []
    for matcher in case.matchers:
        matched_text = ""
        confidence_multiplier = 1.0

        if matcher.type in {"phrase", "substring"}:
            if matcher.pattern in text:
                matched_text = matcher.pattern
        elif matcher.type == "regex":
            match = re.search(matcher.pattern, text, flags=re.MULTILINE)
            if match:
                matched_text = match.group(0)
        elif matcher.type == "fuzzy":
            score = _compute_fuzzy_score(matcher.pattern, text, config.fuzzy_algorithm)
            if score >= config.fuzzy_threshold:
                matched_text = matcher.pattern
                confidence_multiplier = score / 100.0
        else:
            continue

        if matched_text:
            base_confidence = round(case.severity * matcher.weight, 4)
            confidence = round(base_confidence * confidence_multiplier, 4)
            hits.append(
                MatchHit(
                    case_id=case.id,
                    label=case.label,
                    matcher_type=matcher.type,
                    matched_text=matched_text,
                    confidence=confidence,
                    severity=case.severity,
                    rewrite_hint=case.rewrite_hint,
                    diagnostic_dimensions=case.diagnostic_dimensions,
                )
            )
    return hits


def _compute_fuzzy_score(pattern: str, text: str, algorithm: str = "ratio") -> float:
    if algorithm == "ratio":
        return fuzz.ratio(pattern, text)
    if algorithm == "partial_ratio":
        return fuzz.partial_ratio(pattern, text)
    if algorithm == "token_sort_ratio":
        return fuzz.token_sort_ratio(pattern, text)
    if algorithm == "token_set_ratio":
        return fuzz.token_set_ratio(pattern, text)
    return fuzz.ratio(pattern, text)


def compute_score(hits: list[MatchHit]) -> float:
    """Complement-product scoring with dimension diversity bonus."""
    if not hits:
        return 0.0

    complement = 1.0
    for hit in hits:
        complement *= 1.0 - hit.confidence
    base = 1.0 - complement

    unique_dims: set[str] = set()
    for hit in hits:
        unique_dims.update(hit.diagnostic_dimensions)
    diversity_bonus = 0.05 * max(0, len(unique_dims) - 1)

    return round(min(1.0, base * (1.0 + diversity_bonus)), 4)


def detect_paragraphs(text: str, cases: list[BadCase], config: MatcherConfig | None = None) -> list[ParagraphResult]:
    config = config or MatcherConfig()
    results: list[ParagraphResult] = []
    for index, paragraph in enumerate(split_paragraphs(text)):
        hits: list[MatchHit] = []
        for case in cases:
            hits.extend(_match_case(case, paragraph, config))

        if not hits:
            continue

        score = compute_score(hits)
        results.append(
            ParagraphResult(
                paragraph_index=index,
                text=paragraph,
                score=score,
                hits=sorted(hits, key=lambda item: item.confidence, reverse=True),
                retrieval_hits=[],
            )
        )
    return sorted(results, key=lambda item: item.score, reverse=True)

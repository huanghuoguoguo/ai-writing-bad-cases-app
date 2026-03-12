from __future__ import annotations

import re

from .models import BadCase, MatchHit, ParagraphResult


def split_paragraphs(text: str) -> list[str]:
    return [chunk.strip() for chunk in re.split(r"\n\s*\n", text) if chunk.strip()]


def _match_case(case: BadCase, text: str) -> list[MatchHit]:
    hits: list[MatchHit] = []
    for matcher in case.matchers:
        matched_text = ""
        if matcher.type in {"phrase", "substring"}:
            if matcher.pattern in text:
                matched_text = matcher.pattern
        elif matcher.type == "regex":
            match = re.search(matcher.pattern, text, flags=re.MULTILINE)
            if match:
                matched_text = match.group(0)
        else:
            continue

        if matched_text:
            confidence = round(case.severity * matcher.weight, 4)
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


def compute_score(hits: list[MatchHit]) -> float:
    """Complement-product scoring with dimension diversity bonus.

    base = 1 - (1-c1)(1-c2)...(1-cn)  -- monotonically non-decreasing
    diversity_bonus = 0.05 * (unique_dimension_count - 1)
    score = min(1.0, base * (1 + diversity_bonus))
    """
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


def detect_paragraphs(text: str, cases: list[BadCase]) -> list[ParagraphResult]:
    results: list[ParagraphResult] = []
    for index, paragraph in enumerate(split_paragraphs(text)):
        hits: list[MatchHit] = []
        for case in cases:
            hits.extend(_match_case(case, paragraph))

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

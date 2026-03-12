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
                )
            )
    return hits


def detect_paragraphs(text: str, cases: list[BadCase]) -> list[ParagraphResult]:
    results: list[ParagraphResult] = []
    for index, paragraph in enumerate(split_paragraphs(text)):
        hits: list[MatchHit] = []
        for case in cases:
            hits.extend(_match_case(case, paragraph))

        if not hits:
            continue

        score = round(min(1.0, sum(hit.confidence for hit in hits) / len(hits)), 4)
        results.append(
            ParagraphResult(
                paragraph_index=index,
                text=paragraph,
                score=score,
                hits=sorted(hits, key=lambda item: item.confidence, reverse=True),
            )
        )
    return sorted(results, key=lambda item: item.score, reverse=True)

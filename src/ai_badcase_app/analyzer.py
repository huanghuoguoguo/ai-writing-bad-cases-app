from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .models import BadCase


@dataclass
class SuspectedSegment:
    """疑似 AI 生成的文本片段"""

    paragraph_index: int | None
    text: str
    risk_score: float  # 0-1
    risk_level: str  # "high" | "medium" | "low"
    reasons: list[str]  # 为什么怀疑是 AI
    suggestions: list[str]  # 改写建议
    start_pos: int | None = None  # 在原文中的位置
    end_pos: int | None = None
    detection_method: str = "rule"  # 检测方法：rule | stat | probability
    signals: list[dict] = field(default_factory=list)


@dataclass
class TextAnalysisReport:
    """文本分析报告 - 供调用方（AI/Agent）参考"""

    total_chars: int
    total_sentences: int
    total_paragraphs: int
    suspected_segments: list[SuspectedSegment] = field(default_factory=list)
    stats: dict = field(default_factory=dict)
    probability: dict | None = None
    summary: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        """序列化为字典，方便 JSON 输出"""
        sorted_segments = sorted(
            self.suspected_segments,
            key=lambda x: x.risk_score,
            reverse=True,
        )
        overall_risk = _average_segment_score(
            sorted_segments,
            lambda signal: True,
            self.total_paragraphs,
        )
        author_fit_risk = _average_segment_score(
            sorted_segments,
            _is_author_fit_signal,
            self.total_paragraphs,
        )
        author_fit = round(max(0.0, 1.0 - author_fit_risk), 4)

        paragraphs = [
            {
                "paragraph_index": s.paragraph_index,
                "text": s.text[:200] + "..." if len(s.text) > 200 else s.text,
                "text_excerpt": s.text[:120] + "..." if len(s.text) > 120 else s.text,
                "risk_score": s.risk_score,
                "risk_level": s.risk_level,
                "reasons": s.reasons,
                "rewrite_hints": s.suggestions,
                "detection_method": s.detection_method,
                "signals": s.signals,
            }
            for s in sorted_segments
        ]

        result = {
            "document_score": {
                "overall_risk": overall_risk,
                "author_fit_risk": author_fit_risk,
                "author_fit": author_fit,
                "confidence": round(0.6 if self.suspected_segments else 0.3, 4),
            },
            "basic_info": {
                "total_chars": self.total_chars,
                "total_sentences": self.total_sentences,
                "total_paragraphs": self.total_paragraphs,
            },
            "paragraphs": paragraphs,
            "statistics": self.stats,
            "summary": self.summary,
        }
        if self.probability:
            result["probability"] = self.probability

        # Compatibility field for older callers.
        result["suspected_segments"] = paragraphs
        return result

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=indent)


def _risk_level(score: float) -> str:
    if score >= 0.8:
        return "high"
    if score >= 0.6:
        return "medium"
    return "low"


def _average_segment_score(
    segments: list[SuspectedSegment],
    predicate,
    total_paragraphs: int,
) -> float:
    if not segments:
        return 0.0

    total = 0.0
    for segment in segments:
        matching_scores = [
            float(signal.get("score", 0.0))
            for signal in segment.signals
            if predicate(signal)
        ]
        if matching_scores:
            total += max(matching_scores)

    if total <= 0:
        return 0.0
    return round(total / max(total_paragraphs, 1), 4)


def _is_author_fit_signal(signal: dict) -> bool:
    return str(signal.get("code", "")).startswith("zh.fit.")


def _strip_frontmatter_with_offset(text: str) -> tuple[str, int]:
    if not text.startswith("---\n"):
        return text, 0
    parts = text.split("\n---\n", 1)
    if len(parts) != 2:
        return text, 0
    offset = len(parts[0]) + len("\n---\n")
    return parts[1], offset


def _paragraph_spans(text: str) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    for match in re.finditer(r"\S(?:.*?\S)?(?=\n\s*\n|$)", text, flags=re.DOTALL):
        spans.append((match.start(), match.end()))
    return spans


def _find_paragraph_index(paragraph_spans: list[tuple[int, int]], pos: int) -> int | None:
    if paragraph_spans and pos < paragraph_spans[0][0]:
        return 0
    for index, (start, end) in enumerate(paragraph_spans):
        if start <= pos < end:
            return index
    if paragraph_spans and pos >= paragraph_spans[-1][0]:
        return len(paragraph_spans) - 1
    return None


def _is_heading(paragraph: str) -> bool:
    return paragraph.lstrip().startswith("#")


def _is_code_block(paragraph: str) -> bool:
    stripped = paragraph.strip()
    return stripped.startswith("```") or stripped.endswith("```")


def _is_table_block(paragraph: str) -> bool:
    lines = [line.strip() for line in paragraph.splitlines() if line.strip()]
    return bool(lines) and all(line.startswith("|") for line in lines)


def _is_list_block(paragraph: str) -> bool:
    lines = [line.strip() for line in paragraph.splitlines() if line.strip()]
    if not lines:
        return False
    return all(re.match(r"^([-*+] |\d+\. )", line) for line in lines)


def _plain_text_len(text: str) -> int:
    return len(re.sub(r"\s+", "", text))


def _should_run_stat_checks(paragraph: str) -> bool:
    if _is_heading(paragraph) or _is_code_block(paragraph) or _is_table_block(paragraph) or _is_list_block(paragraph):
        return False

    if _plain_text_len(paragraph) < 40:
        return False

    sentence_like_parts = [
        chunk for chunk in re.split(r"[。！？!?\n]+", paragraph) if chunk.strip()
    ]
    return len(sentence_like_parts) >= 2


def _code_fence_indices(paragraphs: list[str]) -> set[int]:
    indices: set[int] = set()
    in_code_fence = False

    for index, paragraph in enumerate(paragraphs):
        fence_count = paragraph.count("```")
        if in_code_fence or fence_count:
            indices.add(index)
        if fence_count % 2 == 1:
            in_code_fence = not in_code_fence

    return indices


def _normalize_review_signals(code: str, source: str) -> list[str]:
    if code in {
        "low_sentence_variation",
        "low_adjacent_sentence_delta",
        "low_extreme_sentence_ratio",
    }:
        return ["uniform_sentence_length"]

    if code in {
        "zh.fit.future_outlook_heading",
        "zh.fit.abstract_future_hype",
        "zh.fit.over_smooth_closure",
    }:
        return ["author_fit_low", "forced_summary"]

    if code == "zh.fit.conclusion_before_journey":
        return ["author_fit_low", "perfect_ladder"]

    if code.startswith("zh.fit."):
        return ["author_fit_low"]

    if source == "rule":
        return ["cliche_phrase"]

    if code in {"high_connector_density", "high_passive_ratio", "low_lexical_diversity"}:
        return ["cliche_phrase"]

    return []


def _primary_review_signal(review_signals: list[str]) -> str | None:
    priority = [
        "need_interview",
        "missing_evidence",
        "author_fit_low",
        "forced_summary",
        "perfect_ladder",
        "uniform_sentence_length",
        "cliche_phrase",
    ]
    for item in priority:
        if item in review_signals:
            return item
    return None


def analyze_text(
    text: str,
    library_root: Path | None = None,
    lang: str = "zh",
    genres: list[str] | None = None,
    enable_perplexity: bool = False,
) -> TextAnalysisReport:
    """
    分析文本，找出疑似 AI 生成的片段。

    这是一个纯工具函数，不做最终判断，只提供证据。
    调用方（AI/Agent）需要根据返回的 suspected_segments 自行决策。
    """
    from .library import load_cases
    from .matcher import MatcherConfig, detect_paragraphs, split_paragraphs
    from .statistics import analyze_text_statistics, detect_statistical_anomalies

    body_text, _frontmatter_offset = _strip_frontmatter_with_offset(text)
    cases = load_cases(library_root=library_root, lang=lang, genres=genres)

    matcher_config = MatcherConfig(fuzzy_threshold=70, fuzzy_algorithm="partial_ratio")
    rule_results = detect_paragraphs(text, cases, config=matcher_config)

    full_stats = analyze_text_statistics(body_text)
    probability_result = None
    if enable_perplexity:
        try:
            from .perplexity import analyze_probability

            probability_result = analyze_probability(body_text)
        except Exception:
            pass

    paragraphs = split_paragraphs(text)
    paragraph_spans = _paragraph_spans(body_text)
    code_fence_paragraphs = _code_fence_indices(paragraphs)
    stat_anomalies_by_paragraph: dict[int, list[dict]] = {}
    for index, paragraph in enumerate(paragraphs):
        if index in code_fence_paragraphs:
            continue
        if not _should_run_stat_checks(paragraph):
            continue
        p_stats = analyze_text_statistics(paragraph)
        p_anomalies = detect_statistical_anomalies(p_stats)
        if p_anomalies:
            stat_anomalies_by_paragraph[index] = p_anomalies

    rule_map = {result.paragraph_index: result for result in rule_results if result.score >= 0.5}
    suspicious_indices = sorted(set(rule_map) | set(stat_anomalies_by_paragraph))

    segments: list[SuspectedSegment] = []
    for index in suspicious_indices:
        paragraph = paragraphs[index]
        rule_result = rule_map.get(index)
        anomalies = stat_anomalies_by_paragraph.get(index, [])

        signals: list[dict] = []
        reasons: list[str] = []
        suggestions: list[str] = []
        risk_candidates: list[float] = []

        if rule_result:
            risk_candidates.append(rule_result.score)
            for hit in rule_result.hits[:3]:
                reasons.append(f"{hit.label} ({hit.matcher_type})")
                if hit.rewrite_hint:
                    suggestions.append(hit.rewrite_hint)
                signals.append(
                    {
                        "code": hit.case_id,
                        "score": hit.confidence,
                        "severity": _risk_level(rule_result.score),
                        "evidence": hit.matched_text,
                        "reason": hit.label,
                        "rewrite_hint": hit.rewrite_hint,
                        "source": "rule",
                        "diagnostic_dimensions": hit.diagnostic_dimensions,
                        "review_signals": _normalize_review_signals(hit.case_id, "rule"),
                        "review_signal": _primary_review_signal(
                            _normalize_review_signals(hit.case_id, "rule")
                        ),
                    }
                )

        if anomalies:
            stat_score = min(0.7, sum(a.get("score", 0.3) for a in anomalies) / len(anomalies))
            risk_candidates.append(stat_score)
            for anomaly in anomalies:
                reasons.append(anomaly["description"])
                suggestions.append(anomaly["rewrite_hint"])
                signals.append(
                    {
                        "code": anomaly["type"],
                        "score": anomaly.get("score", stat_score),
                        "severity": _risk_level(stat_score),
                        "evidence": anomaly["label"],
                        "reason": anomaly["description"],
                        "rewrite_hint": anomaly["rewrite_hint"],
                        "source": "stat",
                        "diagnostic_dimensions": [],
                        "review_signals": _normalize_review_signals(anomaly["type"], "stat"),
                        "review_signal": _primary_review_signal(
                            _normalize_review_signals(anomaly["type"], "stat")
                        ),
                    }
                )

        if not risk_candidates:
            continue

        risk_score = round(max(risk_candidates), 4)
        level = _risk_level(risk_score)
        segments.append(
            SuspectedSegment(
                paragraph_index=index,
                text=paragraph,
                risk_score=risk_score,
                risk_level=level,
                reasons=reasons,
                suggestions=list(dict.fromkeys(suggestions)),
                detection_method="rule+stat" if rule_result and anomalies else ("rule" if rule_result else "stat"),
                signals=signals,
            )
        )

    if probability_result and probability_result.risk_score > 0.3 and probability_result.window_results:
        first_window = probability_result.window_results[0]
        segments.append(
            SuspectedSegment(
                paragraph_index=_find_paragraph_index(paragraph_spans, first_window.start_pos),
                text=first_window.text,
                risk_score=probability_result.risk_score,
                risk_level=probability_result.risk_level,
                reasons=probability_result.reasons,
                suggestions=probability_result.suggestions,
                detection_method="probability",
                signals=[
                    {
                        "code": "probability_signal",
                        "score": probability_result.risk_score,
                        "severity": probability_result.risk_level,
                        "evidence": first_window.text[:120],
                        "reason": reason,
                        "rewrite_hint": "; ".join(probability_result.suggestions),
                        "source": "probability",
                        "diagnostic_dimensions": ["perplexity", "lrr"],
                        "review_signals": [],
                        "review_signal": None,
                    }
                    for reason in probability_result.reasons
                ],
            )
        )

    high_risk_count = sum(1 for s in segments if s.risk_level == "high")
    medium_risk_count = sum(1 for s in segments if s.risk_level == "medium")
    top_issues = [
        signal["code"]
        for segment in sorted(segments, key=lambda item: item.risk_score, reverse=True)[:3]
        for signal in segment.signals[:1]
    ]
    top_review_signals = [
        signal["review_signal"]
        for segment in sorted(segments, key=lambda item: item.risk_score, reverse=True)
        for signal in segment.signals
        if signal.get("review_signal")
    ]
    summary = {
        "total_segments_checked": len(paragraphs),
        "suspected_segments_count": len(segments),
        "high_risk_count": high_risk_count,
        "medium_risk_count": medium_risk_count,
        "top_issues": top_issues,
        "top_review_signals": list(dict.fromkeys(top_review_signals))[:5],
        "stop_or_retry": "retry" if high_risk_count or medium_risk_count else "accept",
        "note": "这些只是疑似片段，需要调用方进一步判断是否真的是 AI 生成",
    }

    probability_dict = None
    if probability_result:
        probability_dict = {
            "overall_ppl": probability_result.overall_ppl,
            "min_ppl": probability_result.min_ppl,
            "max_ppl": probability_result.max_ppl,
            "ppl_variance": probability_result.ppl_variance,
            "lrr_score": probability_result.lrr_score,
            "avg_log_likelihood": probability_result.avg_log_likelihood,
            "avg_log_rank": probability_result.avg_log_rank,
            "top1_ratio": probability_result.top1_ratio,
            "top5_ratio": probability_result.top5_ratio,
            "rare_word_ratio": probability_result.rare_word_ratio,
            "risk_score": probability_result.risk_score,
            "risk_level": probability_result.risk_level,
        }

    return TextAnalysisReport(
        total_chars=full_stats.char_count,
        total_sentences=full_stats.sentence_count,
        total_paragraphs=full_stats.paragraph_count,
        suspected_segments=segments,
        stats={
            "sentence_length_cv": full_stats.sentence_length_cv,
            "adjacent_length_delta_mean": full_stats.adjacent_length_delta_mean,
            "extreme_sentence_ratio": full_stats.extreme_sentence_ratio,
            "connector_density": full_stats.connector_density,
            "passive_ratio": full_stats.passive_ratio,
            "lexical_diversity": full_stats.lexical_diversity,
        },
        probability=probability_dict,
        summary=summary,
    )

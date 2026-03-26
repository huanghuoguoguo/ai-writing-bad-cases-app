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
        overall_risk = 0.0
        if self.suspected_segments:
            overall_risk = round(
                sum(segment.risk_score for segment in self.suspected_segments)
                / max(self.total_paragraphs, 1),
                4,
            )
        author_fit = round(max(0.0, 1.0 - overall_risk), 4)

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
    stat_anomalies_by_paragraph: dict[int, list[dict]] = {}
    for index, paragraph in enumerate(paragraphs):
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
                        "severity": _risk_level(rule_result.score),
                        "evidence": hit.matched_text,
                        "reason": hit.label,
                        "rewrite_hint": hit.rewrite_hint,
                        "source": "rule",
                        "diagnostic_dimensions": hit.diagnostic_dimensions,
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
                        "severity": _risk_level(stat_score),
                        "evidence": anomaly["label"],
                        "reason": anomaly["description"],
                        "rewrite_hint": anomaly["rewrite_hint"],
                        "source": "stat",
                        "diagnostic_dimensions": [],
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
                        "severity": probability_result.risk_level,
                        "evidence": first_window.text[:120],
                        "reason": reason,
                        "rewrite_hint": "; ".join(probability_result.suggestions),
                        "source": "probability",
                        "diagnostic_dimensions": ["perplexity", "lrr"],
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
    summary = {
        "total_segments_checked": len(paragraphs),
        "suspected_segments_count": len(segments),
        "high_risk_count": high_risk_count,
        "medium_risk_count": medium_risk_count,
        "top_issues": top_issues,
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
            "connector_density": full_stats.connector_density,
            "passive_ratio": full_stats.passive_ratio,
            "lexical_diversity": full_stats.lexical_diversity,
        },
        probability=probability_dict,
        summary=summary,
    )

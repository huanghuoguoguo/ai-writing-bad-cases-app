from __future__ import annotations

import re
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .models import BadCase

from .models import Signal, SuspectedSegment, TextAnalysisReport, risk_level


def _normalize_review_signals(code: str, source: str) -> list[str]:
    if code in {
        "low_sentence_variation",
        "low_adjacent_sentence_delta",
        "low_extreme_sentence_ratio",
        "uniform_sentence_groups",
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
    from .matcher import MatcherConfig, detect_paragraphs
    from .statistics import analyze_text_statistics, detect_statistical_anomalies
    from .text_utils import (
        code_fence_indices,
        paragraph_spans,
        should_run_stat_checks,
        split_paragraphs,
        strip_frontmatter,
    )

    body_text, _ = strip_frontmatter(text)
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
    p_spans = paragraph_spans(body_text)
    code_fence_paragraphs = code_fence_indices(paragraphs)

    stat_anomalies_by_paragraph: dict[int, list] = {}
    for index, paragraph in enumerate(paragraphs):
        if index in code_fence_paragraphs:
            continue
        if not should_run_stat_checks(paragraph):
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

        signals: list[Signal] = []
        reasons: list[str] = []
        suggestions: list[str] = []
        risk_candidates: list[float] = []

        if rule_result:
            risk_candidates.append(rule_result.score)
            for hit in rule_result.hits[:3]:
                reasons.append(f"{hit.label} ({hit.matcher_type})")
                if hit.rewrite_hint:
                    suggestions.append(hit.rewrite_hint)
                review_signals = _normalize_review_signals(hit.case_id, "rule")
                signals.append(Signal(
                    code=hit.case_id,
                    score=hit.confidence,
                    severity=risk_level(rule_result.score),
                    evidence=hit.matched_text,
                    reason=hit.label,
                    rewrite_hint=hit.rewrite_hint,
                    source="rule",
                    diagnostic_dimensions=hit.diagnostic_dimensions,
                    review_signals=review_signals,
                    review_signal=_primary_review_signal(review_signals),
                ))

        if anomalies:
            stat_score = min(0.7, sum(a.score for a in anomalies) / len(anomalies))
            risk_candidates.append(stat_score)
            for anomaly in anomalies:
                reasons.append(anomaly.description)
                suggestions.append(anomaly.rewrite_hint)
                review_signals = _normalize_review_signals(anomaly.type, "stat")
                signals.append(Signal(
                    code=anomaly.type,
                    score=anomaly.score,
                    severity=risk_level(stat_score),
                    evidence=anomaly.label,
                    reason=anomaly.description,
                    rewrite_hint=anomaly.rewrite_hint,
                    source="stat",
                    diagnostic_dimensions=anomaly.diagnostic_dimensions,
                    review_signals=review_signals,
                    review_signal=_primary_review_signal(review_signals),
                ))

        if not risk_candidates:
            continue

        risk_score = round(max(risk_candidates), 4)
        level = risk_level(risk_score)
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

    # 概率检测结果
    if probability_result and probability_result.risk_score > 0.3 and probability_result.window_results:
        from .text_utils import find_paragraph_index
        first_window = probability_result.window_results[0]
        segments.append(
            SuspectedSegment(
                paragraph_index=find_paragraph_index(p_spans, first_window.start_pos),
                text=first_window.text,
                risk_score=probability_result.risk_score,
                risk_level=probability_result.risk_level,
                reasons=probability_result.reasons,
                suggestions=probability_result.suggestions,
                detection_method="probability",
                signals=[
                    Signal(
                        code="probability_signal",
                        score=probability_result.risk_score,
                        severity=probability_result.risk_level,
                        evidence=first_window.text[:120],
                        reason=reason,
                        rewrite_hint="; ".join(probability_result.suggestions),
                        source="probability",
                        diagnostic_dimensions=["perplexity", "lrr"],
                    )
                    for reason in probability_result.reasons
                ],
            )
        )

    # 汇总统计
    high_risk_count = sum(1 for s in segments if s.risk_level == "high")
    medium_risk_count = sum(1 for s in segments if s.risk_level == "medium")
    top_issues = [
        signal.code
        for segment in sorted(segments, key=lambda item: item.risk_score, reverse=True)[:3]
        for signal in segment.signals[:1]
    ]
    top_review_signals = [
        signal.review_signal
        for segment in sorted(segments, key=lambda item: item.risk_score, reverse=True)
        for signal in segment.signals
        if signal.review_signal
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
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .models import BadCase


@dataclass
class SuspectedSegment:
    """疑似 AI 生成的文本片段"""
    text: str
    risk_score: float  # 0-1
    risk_level: str  # "high" | "medium" | "low"
    reasons: list[str]  # 为什么怀疑是 AI
    suggestions: list[str]  # 改写建议
    start_pos: int | None = None  # 在原文中的位置
    end_pos: int | None = None


@dataclass
class TextAnalysisReport:
    """文本分析报告 - 供调用方（AI/Agent）参考"""
    # 基本信息
    total_chars: int
    total_sentences: int
    total_paragraphs: int

    # 疑似 AI 片段（核心输出）
    suspected_segments: list[SuspectedSegment] = field(default_factory=list)

    # 统计特征（辅助判断）
    stats: dict = field(default_factory=dict)

    # 风险摘要
    summary: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        """序列化为字典，方便 JSON 输出"""
        return {
            "basic_info": {
                "total_chars": self.total_chars,
                "total_sentences": self.total_sentences,
                "total_paragraphs": self.total_paragraphs,
            },
            "suspected_segments": [
                {
                    "text": s.text[:200] + "..." if len(s.text) > 200 else s.text,
                    "risk_score": s.risk_score,
                    "risk_level": s.risk_level,
                    "reasons": s.reasons,
                    "suggestions": s.suggestions,
                }
                for s in sorted(self.suspected_segments, key=lambda x: x.risk_score, reverse=True)
            ],
            "statistics": self.stats,
            "summary": self.summary,
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=indent)


def analyze_text(
    text: str,
    library_root: Path | None = None,
    lang: str = "zh",
    genres: list[str] | None = None,
) -> TextAnalysisReport:
    """
    分析文本，找出疑似 AI 生成的片段。

    这是一个纯工具函数，不做最终判断，只提供证据。
    调用方（AI/Agent）需要根据返回的 suspected_segments 自行决策。

    Args:
        text: 要分析的文本
        library_root: bad case 库路径
        lang: 语言代码
        genres: 体裁过滤

    Returns:
        TextAnalysisReport 包含所有疑似片段和证据
    """
    from .library import load_cases
    from .matcher import detect_paragraphs, MatcherConfig, split_paragraphs
    from .statistics import analyze_text_statistics, detect_statistical_anomalies

    # 加载 bad cases
    cases = load_cases(library_root=library_root, lang=lang, genres=genres)

    # 1. 规则匹配检测（段落级）
    matcher_config = MatcherConfig(fuzzy_threshold=70, fuzzy_algorithm="partial_ratio")
    rule_results = detect_paragraphs(text, cases, config=matcher_config)

    # 2. 统计特征分析
    full_stats = analyze_text_statistics(text)
    stat_anomalies = detect_statistical_anomalies(full_stats)

    # 收集疑似片段
    segments: list[SuspectedSegment] = []

    # 从规则匹配结果中筛选高风险段落
    for result in rule_results:
        if result.score < 0.5:
            continue

        # 根据分数确定风险等级
        if result.score >= 0.8:
            level = "high"
        elif result.score >= 0.6:
            level = "medium"
        else:
            level = "low"

        # 收集原因和建议
        reasons = []
        suggestions = []
        for hit in result.hits[:3]:  # 只取前3个命中
            reasons.append(f"{hit.label} ({hit.matcher_type})")
            if hit.rewrite_hint:
                suggestions.append(hit.rewrite_hint)

        segments.append(SuspectedSegment(
            text=result.text,
            risk_score=result.score,
            risk_level=level,
            reasons=reasons,
            suggestions=list(set(suggestions)),  # 去重
        ))

    # 从统计异常中添加片段
    paragraphs = split_paragraphs(text)
    for i, paragraph in enumerate(paragraphs):
        p_stats = analyze_text_statistics(paragraph)
        p_anomalies = detect_statistical_anomalies(p_stats)

        if p_anomalies:
            # 计算统计风险分
            stat_score = min(0.7, sum(a.get("score", 0.3) for a in p_anomalies) / len(p_anomalies))

            # 如果这个段落还没有被规则匹配捕获，添加它
            if not any(s.text == paragraph for s in segments):
                segments.append(SuspectedSegment(
                    text=paragraph,
                    risk_score=stat_score,
                    risk_level="medium" if stat_score > 0.5 else "low",
                    reasons=[a["description"] for a in p_anomalies],
                    suggestions=[a["rewrite_hint"] for a in p_anomalies],
                ))

    # 生成摘要
    high_risk_count = sum(1 for s in segments if s.risk_level == "high")
    medium_risk_count = sum(1 for s in segments if s.risk_level == "medium")

    summary = {
        "total_segments_checked": len(paragraphs),
        "suspected_segments_count": len(segments),
        "high_risk_count": high_risk_count,
        "medium_risk_count": medium_risk_count,
        "note": "这些只是疑似片段，需要调用方进一步判断是否真的是 AI 生成",
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
        summary=summary,
    )

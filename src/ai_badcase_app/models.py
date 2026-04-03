from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field


@dataclass(slots=True)
class Matcher:
    type: str
    pattern: str
    weight: float = 1.0


@dataclass(slots=True)
class BadCase:
    id: str
    lang: str
    genres: list[str]
    label: str
    severity: float
    diagnostic_dimensions: list[str]
    description: str
    rewrite_hint: str
    matchers: list[Matcher]
    why_it_sounds_ai: str | None = None
    aliases: list[str] = field(default_factory=list)
    examples: list[str] = field(default_factory=list)
    counter_examples: list[str] = field(default_factory=list)
    prompt_rule: str | None = None


@dataclass(slots=True)
class MatchHit:
    case_id: str
    label: str
    matcher_type: str
    matched_text: str
    confidence: float
    severity: float
    rewrite_hint: str
    diagnostic_dimensions: list[str] = field(default_factory=list)


@dataclass(slots=True)
class RetrievalHit:
    case_id: str
    label: str
    query_mode: str
    score: float
    document: str | None
    diagnostic_dimensions: list[str]
    rewrite_hint: str


@dataclass(slots=True)
class ParagraphResult:
    paragraph_index: int
    text: str
    score: float
    hits: list[MatchHit]
    retrieval_hits: list[RetrievalHit] = field(default_factory=list)


@dataclass(slots=True)
class Signal:
    """检测信号 - 来自规则匹配、统计异常或概率检测"""
    code: str
    score: float
    severity: str
    evidence: str
    reason: str
    rewrite_hint: str
    source: str  # "rule" | "stat" | "probability"
    diagnostic_dimensions: list[str] = field(default_factory=list)
    review_signals: list[str] = field(default_factory=list)
    review_signal: str | None = None


@dataclass(slots=True)
class Anomaly:
    """统计异常检测结果"""
    type: str
    label: str
    score: float
    description: str
    rewrite_hint: str
    diagnostic_dimensions: list[str] = field(default_factory=list)


def risk_level(score: float) -> str:
    """统一的风险等级划分，全局复用"""
    if score >= 0.8:
        return "high"
    if score >= 0.6:
        return "medium"
    return "low"


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
    signals: list[Signal] = field(default_factory=list)


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
                "signals": [asdict(sig) for sig in s.signals],
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
            signal.score
            for signal in segment.signals
            if predicate(signal)
        ]
        if matching_scores:
            total += max(matching_scores)

    if total <= 0:
        return 0.0
    return round(total / max(total_paragraphs, 1), 4)


def _is_author_fit_signal(signal: Signal) -> bool:
    return signal.code.startswith("zh.fit.")

from __future__ import annotations

from dataclasses import dataclass, field


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


@dataclass(slots=True)
class ParagraphResult:
    paragraph_index: int
    text: str
    score: float
    hits: list[MatchHit]

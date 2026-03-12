from __future__ import annotations

import json
from pathlib import Path

from .models import BadCase, Matcher


DEFAULT_LIBRARY_ROOT = (
    Path(__file__).resolve().parents[2] / "ai-writing-bad-cases" / "data" / "cases"
).resolve()


def _load_records(path: Path) -> list[dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload.get("records", [])


def load_cases(
    library_root: Path | None = None,
    lang: str = "zh",
    genres: list[str] | None = None,
) -> list[BadCase]:
    root = library_root or DEFAULT_LIBRARY_ROOT
    wanted_genres = set(genres or [])
    cases: list[BadCase] = []

    lang_root = root / lang
    for path in sorted(lang_root.glob("*.json")):
        for record in _load_records(path):
            record_genres = set(record["genres"])
            if wanted_genres and not wanted_genres.intersection(record_genres):
                continue

            matchers = [
                Matcher(
                    type=item["type"],
                    pattern=item["pattern"],
                    weight=item.get("weight", 1.0),
                )
                for item in record["matchers"]
            ]
            cases.append(
                BadCase(
                    id=record["id"],
                    lang=record["lang"],
                    genres=record["genres"],
                    label=record["label"],
                    aliases=record.get("aliases", []),
                    severity=record["severity"],
                    diagnostic_dimensions=record["diagnostic_dimensions"],
                    description=record["description"],
                    why_it_sounds_ai=record.get("why_it_sounds_ai"),
                    matchers=matchers,
                    examples=record.get("examples", []),
                    counter_examples=record.get("counter_examples", []),
                    rewrite_hint=record["rewrite_hint"],
                    prompt_rule=record.get("prompt_rule"),
                )
            )
    return cases

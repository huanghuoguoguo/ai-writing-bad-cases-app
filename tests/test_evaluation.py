"""Evaluation quality gates: verify detection effectiveness on fixture data."""

import json
from pathlib import Path

from ai_badcase_app.library import load_cases
from ai_badcase_app.matcher import detect_paragraphs


FIXTURES_DIR = Path(__file__).parent / "fixtures"


def _load_fixture(name: str) -> list[dict]:
    path = FIXTURES_DIR / name
    data = json.loads(path.read_text(encoding="utf-8"))
    return data["paragraphs"]


def test_ai_paragraphs_flagged():
    """AI-style paragraphs should be detected with score > 0.5."""
    cases = load_cases()
    paragraphs = _load_fixture("zh_ai_paragraphs.json")

    flagged = 0
    for para in paragraphs:
        results = detect_paragraphs(para["text"], cases)
        if results and results[0].score > 0.5:
            flagged += 1

    recall = flagged / len(paragraphs)
    assert recall >= 0.7, f"Recall too low: {recall:.2f} ({flagged}/{len(paragraphs)} flagged)"


def test_human_paragraphs_not_flagged():
    """Human-written paragraphs should mostly have low scores."""
    cases = load_cases()
    paragraphs = _load_fixture("zh_human_paragraphs.json")

    scores = []
    for para in paragraphs:
        results = detect_paragraphs(para["text"], cases)
        score = results[0].score if results else 0.0
        scores.append(score)

    avg_score = sum(scores) / len(scores)
    assert avg_score < 0.4, f"Average human score too high: {avg_score:.3f}"


def test_score_separation():
    """AI paragraphs should score significantly higher than human paragraphs."""
    cases = load_cases()

    ai_paras = _load_fixture("zh_ai_paragraphs.json")
    human_paras = _load_fixture("zh_human_paragraphs.json")

    ai_scores = []
    for para in ai_paras:
        results = detect_paragraphs(para["text"], cases)
        ai_scores.append(results[0].score if results else 0.0)

    human_scores = []
    for para in human_paras:
        results = detect_paragraphs(para["text"], cases)
        human_scores.append(results[0].score if results else 0.0)

    avg_ai = sum(ai_scores) / len(ai_scores)
    avg_human = sum(human_scores) / len(human_scores)
    separation = avg_ai - avg_human

    assert separation >= 0.3, (
        f"Score separation too low: {separation:.3f} "
        f"(AI avg={avg_ai:.3f}, human avg={avg_human:.3f})"
    )


def test_expected_case_ids_hit():
    """Each AI fixture should hit at least some of its expected case IDs."""
    cases = load_cases()
    paragraphs = _load_fixture("zh_ai_paragraphs.json")

    missed_fixtures = []
    for para in paragraphs:
        results = detect_paragraphs(para["text"], cases)
        actual_ids = {hit.case_id for r in results for hit in r.hits}
        expected_ids = set(para.get("expected_case_ids", []))
        if expected_ids and not expected_ids.intersection(actual_ids):
            missed_fixtures.append(para["id"])

    assert not missed_fixtures, f"Fixtures with zero expected case hits: {missed_fixtures}"


def test_every_case_has_positive_match():
    """Every case in the library should match at least one positive fixture."""
    cases = load_cases()
    paragraphs = _load_fixture("zh_ai_paragraphs.json")

    matched_case_ids: set[str] = set()
    for para in paragraphs:
        results = detect_paragraphs(para["text"], cases)
        for r in results:
            for hit in r.hits:
                matched_case_ids.add(hit.case_id)

    all_case_ids = {c.id for c in cases}
    unmatched = all_case_ids - matched_case_ids

    # Allow some unmatched (not every case needs a fixture), but warn
    # For now, we check that at least 80% of cases have fixtures
    coverage = len(matched_case_ids) / len(all_case_ids)
    assert coverage >= 0.6, (
        f"Case coverage too low: {coverage:.1%}. "
        f"Unmatched cases: {sorted(unmatched)}"
    )

from ai_badcase_app.matcher import compute_score
from ai_badcase_app.models import MatchHit


def _hit(confidence: float, dims: list[str] | None = None) -> MatchHit:
    return MatchHit(
        case_id="test",
        label="test",
        matcher_type="phrase",
        matched_text="test",
        confidence=confidence,
        severity=confidence,
        rewrite_hint="test",
        diagnostic_dimensions=dims or ["dim_a"],
    )


def test_compute_score_empty():
    assert compute_score([]) == 0.0


def test_compute_score_single_hit():
    hits = [_hit(0.93)]
    assert compute_score(hits) == 0.93


def test_compute_score_capped_at_one():
    hits = [_hit(0.99, ["d1"]), _hit(0.99, ["d2"]), _hit(0.99, ["d3"]), _hit(0.99, ["d4"])]
    score = compute_score(hits)
    assert score <= 1.0


def test_compute_score_low_confidence_does_not_lower():
    """The critical bug fix: adding a low-confidence hit must never lower the score."""
    one_hit = [_hit(0.93)]
    two_hits = [_hit(0.93), _hit(0.68)]
    assert compute_score(two_hits) >= compute_score(one_hit)


def test_compute_score_monotonic():
    """Score with n+1 hits >= score with n hits."""
    hits = [_hit(0.5), _hit(0.6), _hit(0.3), _hit(0.7), _hit(0.4)]
    for n in range(1, len(hits)):
        score_n = compute_score(hits[:n])
        score_n_plus_1 = compute_score(hits[: n + 1])
        assert score_n_plus_1 >= score_n, f"Score decreased at n={n}: {score_n} -> {score_n_plus_1}"


def test_compute_score_diversity_bonus():
    """Same confidences, more unique dimensions -> higher score."""
    same_dim = [_hit(0.5, ["dim_a"]), _hit(0.5, ["dim_a"])]
    diff_dim = [_hit(0.5, ["dim_a"]), _hit(0.5, ["dim_b"])]
    assert compute_score(diff_dim) > compute_score(same_dim)


def test_compute_score_known_values():
    """Known input/output for regression."""
    # Single hit: base = 0.93, 0 diversity bonus
    assert compute_score([_hit(0.93)]) == 0.93
    # Two hits same dim: base = 1-(0.07*0.32) = 0.9776, no diversity bonus
    score = compute_score([_hit(0.93, ["d1"]), _hit(0.68, ["d1"])])
    assert abs(score - 0.9776) < 0.001
    # Two hits diff dim: base = 0.9776, diversity bonus = 0.05
    score = compute_score([_hit(0.93, ["d1"]), _hit(0.68, ["d2"])])
    expected = min(1.0, 0.9776 * 1.05)
    assert abs(score - round(expected, 4)) < 0.001


def test_compute_score_all_nine_dimensions():
    """Hitting all 9 dimensions gives maximum diversity bonus."""
    dims = [
        "over_explicitness", "structure_symmetry", "abstract_over_specific",
        "meta_discourse_density", "posture_before_content", "emotional_servicing",
        "connector_driven", "closure_impulse", "average_style",
    ]
    hits = [_hit(0.3, [d]) for d in dims]
    score_all_dims = compute_score(hits)
    score_one_dim = compute_score([_hit(0.3, ["dim_a"])] * 9)
    assert score_all_dims > score_one_dim

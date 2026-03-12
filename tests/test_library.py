from pathlib import Path

from ai_badcase_app.library import load_cases


def test_load_cases_argumentative():
    cases = load_cases(genres=["argumentative"])
    assert len(cases) == 22


def test_load_cases_narrative():
    cases = load_cases(genres=["narrative"])
    # 10 dedicated narrative + cross-genre cases (steadily_catch_you, max_out_x)
    assert len(cases) >= 10


def test_load_cases_academic():
    cases = load_cases(genres=["academic"])
    # 8 dedicated academic + cross-genre (conclusion_signals, era_opening, paraphrase_meta, connector_chain)
    assert len(cases) >= 8


def test_load_cases_genre_filter():
    all_cases = load_cases()
    arg_cases = load_cases(genres=["argumentative"])
    nar_cases = load_cases(genres=["narrative"])
    # Filtered results should be subsets
    assert len(arg_cases) <= len(all_cases)
    assert len(nar_cases) <= len(all_cases)


def test_load_cases_cross_genre():
    """A case tagged with multiple genres appears in both genre filters."""
    arg_cases = load_cases(genres=["argumentative"])
    nar_cases = load_cases(genres=["narrative"])

    arg_ids = {c.id for c in arg_cases}
    nar_ids = {c.id for c in nar_cases}

    # steadily_catch_you is tagged both argumentative and narrative
    assert "zh.arg.steadily_catch_you" in arg_ids
    assert "zh.arg.steadily_catch_you" in nar_ids


def test_case_schema_compliance():
    """Every loaded case has required fields and valid diagnostic_dimensions."""
    valid_dimensions = {
        "over_explicitness", "structure_symmetry", "abstract_over_specific",
        "meta_discourse_density", "posture_before_content", "emotional_servicing",
        "connector_driven", "closure_impulse", "average_style",
    }
    cases = load_cases()
    for case in cases:
        assert case.id, f"Case missing id"
        assert case.label, f"Case {case.id} missing label"
        assert 0.0 <= case.severity <= 1.0, f"Case {case.id} severity out of range"
        assert len(case.diagnostic_dimensions) >= 1, f"Case {case.id} missing dimensions"
        for dim in case.diagnostic_dimensions:
            assert dim in valid_dimensions, f"Case {case.id} has unknown dimension: {dim}"
        assert len(case.matchers) >= 1, f"Case {case.id} has no matchers"
        assert case.rewrite_hint, f"Case {case.id} missing rewrite_hint"

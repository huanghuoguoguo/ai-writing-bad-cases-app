from ai_badcase_app.library import load_cases
from ai_badcase_app.matcher import detect_paragraphs, split_paragraphs, split_sentences
from ai_badcase_app.models import MatchHit


# --- split_paragraphs tests ---


def test_split_paragraphs_basic():
    text = "First paragraph.\n\nSecond paragraph.\n\nThird paragraph."
    result = split_paragraphs(text)
    assert len(result) == 3
    assert result[0] == "First paragraph."
    assert result[2] == "Third paragraph."


def test_split_paragraphs_empty():
    assert split_paragraphs("") == []
    assert split_paragraphs("   ") == []
    assert split_paragraphs("\n\n\n") == []


def test_split_paragraphs_single():
    text = "Only one paragraph here, no double newline."
    result = split_paragraphs(text)
    assert len(result) == 1
    assert result[0] == text


def test_split_paragraphs_whitespace_between():
    text = "Para one.\n  \n  \nPara two."
    result = split_paragraphs(text)
    assert len(result) == 2


def test_split_sentences_chinese_punctuation():
    text = "第一句。第二句！第三句？"
    result = split_sentences(text)
    assert result == ["第一句。", "第二句！", "第三句？"]


def test_split_sentences_keeps_markdown_block_intact():
    text = "```python\nprint('hi')\n```"
    result = split_sentences(text)
    assert result == [text]


# --- _match_case / MatchHit tests ---


def _make_hit(confidence: float, dims: list[str] | None = None) -> MatchHit:
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


def test_match_case_phrase_hit():
    cases = load_cases(genres=["argumentative"])
    text = "这套方法会稳稳接住你。"
    results = detect_paragraphs(text, cases)
    assert len(results) == 1
    hit = results[0].hits[0]
    assert hit.case_id == "zh.arg.steadily_catch_you"
    assert hit.matcher_type == "phrase"
    assert hit.matched_text == "稳稳接住你"


def test_match_case_regex_hit():
    cases = load_cases(genres=["argumentative"])
    text = "真正重要的不是速度，而是你是否能长期坚持。"
    results = detect_paragraphs(text, cases)
    assert len(results) == 1
    hit_ids = {h.case_id for h in results[0].hits}
    assert "zh.arg.not_x_but_y" in hit_ids
    regex_hit = next(h for h in results[0].hits if h.case_id == "zh.arg.not_x_but_y")
    assert regex_hit.matcher_type == "regex"


def test_match_case_no_hit():
    cases = load_cases(genres=["argumentative"])
    text = "这是一段完全普通的中文，没有任何模式。"
    results = detect_paragraphs(text, cases)
    assert len(results) == 0


def test_confidence_calculation():
    cases = load_cases(genres=["argumentative"])
    text = "综上所述，我们需要重新理解效率。"
    results = detect_paragraphs(text, cases)
    assert len(results) == 1
    hit = results[0].hits[0]
    assert hit.case_id == "zh.arg.conclusion_signals"
    assert hit.confidence == 0.68


def test_diagnostic_dimensions_in_match_hit():
    cases = load_cases(genres=["argumentative"])
    text = "稳稳接住你"
    results = detect_paragraphs(text, cases)
    assert len(results) == 1
    hit = results[0].hits[0]
    assert hit.diagnostic_dimensions == ["emotional_servicing", "posture_before_content"]


def test_detect_paragraphs_ordering():
    cases = load_cases(genres=["argumentative"])
    text = "综上所述，效率很重要。\n\n稳稳接住你。"
    results = detect_paragraphs(text, cases)
    assert len(results) == 2
    assert results[0].score >= results[1].score


def test_detect_paragraphs_no_hits():
    cases = load_cases(genres=["argumentative"])
    text = "完全无关的普通文本。\n\n另一段也没有任何模式匹配。"
    results = detect_paragraphs(text, cases)
    assert len(results) == 0


def test_detects_known_phrase():
    cases = load_cases(genres=["argumentative"])
    text = "不绕弯子，直接说重点。\n\n真正重要的不是速度，而是你是否能长期坚持。"
    results = detect_paragraphs(text, cases)

    assert len(results) == 2
    hit_ids = {hit.case_id for result in results for hit in result.hits}
    assert "zh.arg.direct_to_point_opening" in hit_ids
    assert "zh.arg.not_x_but_y" in hit_ids


def test_multiple_matchers_single_case():
    cases = load_cases(genres=["argumentative"])
    text = "不绕弯子，开门见山地说。"
    results = detect_paragraphs(text, cases)
    assert len(results) == 1
    hits_for_case = [h for h in results[0].hits if h.case_id == "zh.arg.direct_to_point_opening"]
    assert len(hits_for_case) >= 2


def test_author_fit_wave_opening_detected():
    cases = load_cases(genres=["argumentative"])
    text = "在 AI Coding 的浪潮中，我们常常会遭遇代码生成的失控感。"
    results = detect_paragraphs(text, cases)
    hit_ids = {hit.case_id for result in results for hit in result.hits}
    assert "zh.fit.wave_opening" in hit_ids


def test_author_fit_future_outlook_detected():
    cases = load_cases(genres=["argumentative"])
    text = "未来展望\n\n这项技术带来了无尽的想象空间，其爆发潜力不容小觑。"
    results = detect_paragraphs(text, cases)
    hit_ids = {hit.case_id for result in results for hit in result.hits}
    assert "zh.fit.future_outlook_heading" in hit_ids
    assert "zh.fit.abstract_future_hype" in hit_ids


def test_author_fit_tao_shu_pairing_detected():
    cases = load_cases(genres=["argumentative"])
    text = "我们不仅要掌握利用 AI 快速生成代码的“术”，更要坚守工程质量的“道”。"
    results = detect_paragraphs(text, cases)
    hit_ids = {hit.case_id for result in results for hit in result.hits}
    assert "zh.fit.tao_shu_pairing" in hit_ids


def test_meta_essence_detects_shuochuanle():
    cases = load_cases(genres=["argumentative"])
    text = "说穿了，这件事没有那么神秘。"
    results = detect_paragraphs(text, cases)
    hit_ids = {hit.case_id for result in results for hit in result.hits}
    assert "zh.arg.meta_essence" in hit_ids

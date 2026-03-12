from ai_badcase_app.library import load_cases
from ai_badcase_app.matcher import detect_paragraphs


def test_detects_known_phrase():
    cases = load_cases(genres=["argumentative"])
    text = "不绕弯子，直接说重点。\n\n真正重要的不是速度，而是你是否能长期坚持。"
    results = detect_paragraphs(text, cases)

    assert len(results) == 2
    hit_ids = {hit.case_id for result in results for hit in result.hits}
    assert "zh.arg.direct_to_point_opening" in hit_ids
    assert "zh.arg.not_x_but_y" in hit_ids

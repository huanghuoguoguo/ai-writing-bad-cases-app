from ai_badcase_app.analyzer import analyze_text
from ai_badcase_app.rewrite import rewrite_text


def test_rewrite_removes_not_but_pattern():
    text = "真正重要的不是速度，而是你是否能长期坚持。"
    report = analyze_text(text)
    rewritten = rewrite_text(text, report)
    assert "不是速度，而是" not in rewritten
    assert "更要紧的是你是否能长期坚持" in rewritten


def test_rewrite_strips_meta_essence_markers():
    text = "本质上，真正重要的是你愿不愿意持续做下去。"
    report = analyze_text(text)
    rewritten = rewrite_text(text, report)
    assert "本质上" not in rewritten
    assert "真正重要的是" not in rewritten

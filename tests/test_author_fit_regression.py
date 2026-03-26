from pathlib import Path

from ai_badcase_app.analyzer import analyze_text


SAMPLE_POST = Path("/home/yhh/learn/huanghuoguoguo.github.io/_posts/2026-1-16-测试驱动开发与vibe-coding.md")


def test_sample_post_hits_author_fit_signals():
    text = SAMPLE_POST.read_text(encoding="utf-8")
    report = analyze_text(text)
    signal_codes = {
        signal["code"]
        for segment in report.suspected_segments
        for signal in segment.signals
    }
    assert "zh.fit.wave_opening" in signal_codes
    assert "zh.fit.abstract_tool_promise" in signal_codes
    assert "zh.fit.future_outlook_heading" in signal_codes


def test_sample_post_still_requests_retry():
    text = SAMPLE_POST.read_text(encoding="utf-8")
    report = analyze_text(text)
    assert report.summary["stop_or_retry"] == "retry"

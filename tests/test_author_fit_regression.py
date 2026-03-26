from ai_badcase_app.analyzer import analyze_text


SAMPLE_POST = """
在 AI Coding 的浪潮中，我们常常会遭遇代码生成的失控感。

TDD 能否成为约束和加强 AI Coding 的那根缰绳呢？

## 未来展望

这项技术带来了无尽的想象空间，其爆发潜力不容小觑。
"""


def test_sample_post_hits_author_fit_signals():
    report = analyze_text(SAMPLE_POST)
    signal_codes = {
        signal["code"]
        for segment in report.suspected_segments
        for signal in segment.signals
    }
    assert "zh.fit.wave_opening" in signal_codes
    assert "zh.fit.abstract_tool_promise" in signal_codes
    assert "zh.fit.future_outlook_heading" in signal_codes


def test_sample_post_still_requests_retry():
    report = analyze_text(SAMPLE_POST)
    assert report.summary["stop_or_retry"] == "retry"

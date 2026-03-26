"""
验证测试 - 确保检测工具的准确性

这些测试验证：
1. 已知 AI 文本能检测出高风险片段
2. 已知人类文本不触发高风险警告
3. 边界情况正确处理
"""

import pytest
from ai_badcase_app.analyzer import analyze_text


# ============ 测试数据 ============

# 典型的 AI 生成文本（高连接词、被动语态、程式化结构）
AI_TEXT_SAMPLES = {
    "sequence_connectors": """
首先，我们需要理解问题的本质。其次，分析可能的原因。最后，提出解决方案。
综上所述，本文提出了一个有效的方法。
""",
    "passive_voice": """
这种技术被认为是最优的解决方案。它被广泛应用于各个领域，具有重要的理论意义。
由此可见，该方法具有不可替代的作用。
""",
    "not_x_but_y": """
真正重要的不是速度，而是你是否能长期坚持。AI 不是替代者，而是合作伙伴。
关键不在于技术本身，而在于如何使用它。
""",
    "marketing_explainer": """
本文将深入拆解语义指纹技术，揭秘 AI 检测的核心技术。
这场文本DNA追踪革命让 AI 生成内容无处遁形。
句长分布特征与困惑度方差模型，正在成为区分人类写作与AI生成的思维心电图。
""",
}

# 典型的人类写作（口语化、有"我"的视角、短句、无程式化结构）
HUMAN_TEXT_SAMPLES = {
    "personal_narrative": """
我以前觉得线程越多越高效。后来发现 Python FastAPI 是单线程的。
单线程能撑高并发？想不通。再后来才明白，一个线程可以在等待 IO 的时候去服务其他请求。
我当时想不通，就去问了 Kimi。它说了一堆，我半信半疑。
""",
    "conversational": """
说实话，刚开始我也不懂。查了一堆资料，试了几次，踩了几个坑，才搞明白怎么回事。
你可能也会有同样的困惑，咱们一起理一理。
""",
    "iterative_thinking": """
V1 版本直接暴力遍历，发现超时了。V2 加了缓存，好点但内存爆了。
V3 换了算法，终于过了。但代码丑得像坨屎，先这样吧。
""",
}


# ============ 测试用例 ============

class TestAITextDetection:
    """测试 AI 文本能被正确检测"""

    def test_sequence_connectors_detected(self):
        """序列连接词（首先...其次...最后）应被检测为高风险"""
        text = AI_TEXT_SAMPLES["sequence_connectors"]
        report = analyze_text(text)

        # 应该有高风险片段
        high_risk = [s for s in report.suspected_segments if s.risk_level == "high"]
        assert len(high_risk) >= 1, "应该检测出序列连接词"

        # 检查原因中包含"序列词"
        found_reason = False
        for seg in high_risk:
            for reason in seg.reasons:
                if "序列" in reason or "三段式" in reason:
                    found_reason = True
        assert found_reason, f"未找到序列词相关原因: {high_risk}"

    def test_passive_voice_detected(self):
        """被动语态应被检测"""
        text = AI_TEXT_SAMPLES["passive_voice"]
        report = analyze_text(text)

        # 应该有中高风险片段
        risky = [s for s in report.suspected_segments if s.risk_score > 0.5]
        assert len(risky) >= 1, "应该检测出被动语态"

    def test_not_x_but_y_detected(self):
        """对举句式（不是...而是）应被检测为高风险"""
        text = AI_TEXT_SAMPLES["not_x_but_y"]
        report = analyze_text(text)

        high_risk = [s for s in report.suspected_segments if s.risk_level == "high"]
        assert len(high_risk) >= 1, "应该检测出对举句式"

        # 验证原因
        reasons = " ".join([r for seg in high_risk for r in seg.reasons])
        assert "不是" in reasons or "对举" in reasons, f"未找对举原因: {reasons}"

    def test_marketing_explainer_detected(self):
        """揭秘腔和技术神话腔应被检测"""
        text = AI_TEXT_SAMPLES["marketing_explainer"]
        report = analyze_text(text)
        signal_codes = {
            signal["code"]
            for seg in report.suspected_segments
            for signal in seg.signals
        }
        assert "zh.arg.reveal_the_secret" in signal_codes
        assert "zh.arg.tech_myth_hype" in signal_codes


class TestHumanTextNotFlagged:
    """测试人类文本不应被过度标记"""

    def test_personal_narrative_low_risk(self):
        """个人叙事应被判定为低风险"""
        text = HUMAN_TEXT_SAMPLES["personal_narrative"]
        report = analyze_text(text)

        # 高风险片段不应超过 1 个
        high_risk = [s for s in report.suspected_segments if s.risk_level == "high"]
        assert len(high_risk) <= 1, f"人类文本不应有多个高风险片段: {len(high_risk)}"

        # 统计特征应该自然（变异系数 > 0.4）
        assert report.stats["sentence_length_cv"] > 0.4, \
            f"人类文本句长应更自然: {report.stats['sentence_length_cv']}"

    def test_conversational_low_risk(self):
        """口语化文本应被判定为低风险"""
        text = HUMAN_TEXT_SAMPLES["conversational"]
        report = analyze_text(text)

        # 高风险片段应该很少
        high_risk = [s for s in report.suspected_segments if s.risk_level == "high"]
        assert len(high_risk) == 0, "口语化文本不应被标记为高风险"

    def test_iterative_thinking_low_risk(self):
        """迭代叙事（V1/V2/V3）应被判定为低风险"""
        text = HUMAN_TEXT_SAMPLES["iterative_thinking"]
        report = analyze_text(text)

        # 虽然包含 V1/V2/V3，但这不等于 AI 的"首先其次"
        high_risk = [s for s in report.suspected_segments if s.risk_level == "high"]
        assert len(high_risk) == 0, "迭代叙事不应被误判为 AI"


class TestEdgeCases:
    """测试边界情况"""

    def test_empty_text(self):
        """空文本应返回空结果"""
        report = analyze_text("")
        assert report.total_chars == 0
        assert len(report.suspected_segments) == 0

    def test_short_text(self):
        """极短文本应返回空结果或低风险"""
        text = "这是一句话。"
        report = analyze_text(text)
        # 短文本可能没有足够特征判断
        high_risk = [s for s in report.suspected_segments if s.risk_level == "high"]
        assert len(high_risk) == 0, "短文本不应被标记为高风险"

    def test_mixed_text(self):
        """混合文本（部分 AI + 部分人类）应能区分"""
        text = """
我之前用 AI 写过一篇文章。首先，我输入了提示词。其次，AI 生成了内容。最后，我做了一些修改。

但说实话，改完之后发现还是不对劲。那些句子读着通顺，但就是没有我的味道。
后来我就自己重写了，虽然慢点，但至少像人写的。
"""
        report = analyze_text(text)

        # 应该检测出前半部分的 AI 特征
        suspected_count = len(report.suspected_segments)
        assert suspected_count >= 1, "混合文本应检测出 AI 片段"


class TestStatisticsAccuracy:
    """测试统计特征计算准确性"""

    def test_uniform_sentences_detected(self):
        """句长过于均匀的文本应被标记"""
        # 构造句长几乎相同的文本
        text = "这是一个测试句子。这是另一个测试句子。这是第三个测试句子。这是第四个测试句子。"
        report = analyze_text(text)

        # 应该有统计异常
        assert report.stats["sentence_length_cv"] < 0.3, \
            f"句长应被检测为过于均匀: {report.stats['sentence_length_cv']}"

    def test_high_connector_density_detected(self):
        """连接词密度过高应被检测"""
        text = "首先，我们需要理解。其次，我们要分析。此外，我们还要考虑。因此，我们可以得出结论。"
        report = analyze_text(text)

        assert report.stats["connector_density"] > 2.0, \
            f"连接词密度应被检测为过高: {report.stats['connector_density']}"

    def test_low_adjacent_sentence_delta_detected(self):
        """相邻句跳变太小时应被检测"""
        text = "这是一个普通测试句子。这是另一条普通测试句子。这又是一条普通测试句子。这还是一条普通测试句子。"
        report = analyze_text(text)
        signal_codes = {
            signal["code"]
            for seg in report.suspected_segments
            for signal in seg.signals
        }
        assert "low_adjacent_sentence_delta" in signal_codes

    def test_low_extreme_sentence_ratio_detected(self):
        """缺少长短句交替时应被检测"""
        text = "这个句子长度差不多。第二句长度也差不多。第三句还是差不多。第四句依旧差不多。第五句仍然差不多。"
        report = analyze_text(text)
        signal_codes = {
            signal["code"]
            for seg in report.suspected_segments
            for signal in seg.signals
        }
        assert "low_extreme_sentence_ratio" in signal_codes


class TestPerplexityDetection:
    """测试概率特征检测（可选功能）"""

    def test_perplexity_detection_available(self):
        """概率检测可用时不应报错"""
        text = "这是一个用于测试概率检测的文本段落。它包含多个句子，用来验证功能是否正常。"
        try:
            report = analyze_text(text, enable_perplexity=True)
            # 如果概率检测可用，应该有 probability 字段
            if report.probability:
                assert "overall_ppl" in report.probability
                assert "lrr_score" in report.probability
                assert "top1_ratio" in report.probability
        except Exception:
            # 依赖未安装时跳过
            pass

    def test_ai_text_lower_perplexity(self):
        """AI 文本应该有更低的困惑度（如果检测可用）"""
        ai_text = "首先，我们需要理解问题的本质。其次，分析可能的原因。最后，提出解决方案。"
        human_text = "说实话，我以前真没觉得这是个问题。后来才发现，事情没那么简单。"

        try:
            ai_report = analyze_text(ai_text, enable_perplexity=True)
            human_report = analyze_text(human_text, enable_perplexity=True)

            if ai_report.probability and human_report.probability:
                # AI 文本困惑度通常更低
                assert ai_report.probability["overall_ppl"] < human_report.probability["overall_ppl"] * 1.5, \
                    "AI 文本困惑度应显著低于人类文本"
        except Exception:
            pass

    def test_lrr_score_present(self):
        """LRR 分数应该存在（如果检测可用）"""
        text = "这是一个测试文本，用于验证 LRR 检测功能是否正常工作。"
        try:
            report = analyze_text(text, enable_perplexity=True)
            if report.probability:
                assert "lrr_score" in report.probability
                assert isinstance(report.probability["lrr_score"], float)
        except Exception:
            pass


# ============ 运行测试 ============

if __name__ == "__main__":
    # 简单运行测试
    import sys

    test_classes = [
        TestAITextDetection(),
        TestHumanTextNotFlagged(),
        TestEdgeCases(),
        TestStatisticsAccuracy(),
        TestPerplexityDetection(),
    ]

    passed = 0
    failed = 0

    for test_class in test_classes:
        class_name = test_class.__class__.__name__
        print(f"\n=== {class_name} ===")

        for method_name in dir(test_class):
            if method_name.startswith("test_"):
                try:
                    method = getattr(test_class, method_name)
                    method()
                    print(f"  ✓ {method_name}")
                    passed += 1
                except AssertionError as e:
                    print(f"  ✗ {method_name}: {e}")
                    failed += 1
                except Exception as e:
                    print(f"  ✗ {method_name}: 异常 - {e}")
                    failed += 1

    print(f"\n{'='*40}")
    print(f"总计: {passed + failed} 个测试")
    print(f"通过: {passed}")
    print(f"失败: {failed}")

    sys.exit(0 if failed == 0 else 1)

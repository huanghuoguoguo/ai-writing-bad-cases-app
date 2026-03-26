"""
概率特征检测演示脚本（PPL + LRR + Rank 分布）

展示如何使用概率特征检测来识别 AI 生成文本。

运行方式:
    UV_CACHE_DIR=.uv-cache uv run python scripts/demo_perplexity.py

注意：需要先安装 perplexity 依赖：
    uv sync --extra perplexity
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from ai_badcase_app.analyzer import analyze_text


SAMPLES = {
    "ai_template": """
在这个信息爆炸的时代，我们需要重新思考效率的本质。
首先，我们要理解什么是真正的效率。其次，我们要分析影响效率的关键因素。
最后，我们要提出提升效率的系统性方法。综上所述，效率不是简单的速度问题，
而是一个涉及认知、方法和工具的综合性议题。
""",
    "ai_academic": """
本研究旨在探讨人工智能技术在教育领域的应用现状与发展趋势。
研究表明，人工智能技术能够有效提升教学效率，优化学习体验。
然而，技术应用也面临数据隐私、算法偏见等挑战。
未来研究应关注技术伦理与教育的深度融合。
""",
    "human_casual": """
说实话，我以前真没觉得效率是个问题。每天就是干活呗，干完就行。
后来换了个老板，天天催进度，我才开始琢磨这事。
试了一堆方法，有的好用有的坑，现在也就勉强能按时交活。
你说什么系统性方法？我不懂那些，反正能搞定就行。
""",
    "human_personal": """
V1 版本直接暴力遍历，发现超时了。V2 加了缓存，好点但内存爆了。
V3 换了算法，终于过了。但代码丑得像坨屎，先这样吧。
昨天老板看了一眼，说让我重构，我说等我有空的——其实就是没空。
""",
    "mixed": """
我之前用 AI 写过一篇文章。首先，我输入了提示词。其次，AI 生成了内容。最后，我做了一些修改。

但说实话，改完之后发现还是不对劲。那些句子读着通顺，但就是没有我的味道。
后来我就自己重写了，虽然慢点，但至少像人写的。
""",
}


def demo():
    print("=" * 70)
    print("概率特征检测演示 (PPL + LRR + Rank 分布)")
    print("=" * 70)
    print()
    print("说明：")
    print("- 困惑度 (PPL): 文本的可预测性")
    print("- LRR: Log-Likelihood / Log-Rank，捕捉 AI '极度避险' 特征")
    print("- Top-1 比例: AI 更倾向选择概率最高的词")
    print("- 罕见词比例: 人类写作会使用更多冷门表达")
    print()

    for name, text in SAMPLES.items():
        print("-" * 70)
        print(f"样本: {name}")
        print("-" * 70)
        print(f"文本预览: {text[:60]}...")
        print()

        # 基础分析
        report = analyze_text(text)

        print("基础检测结果:")
        print(f"  - 疑似片段数: {report.summary['suspected_segments_count']}")
        print(f"  - 高风险数: {report.summary['high_risk_count']}")
        print(f"  - 统计特征:")
        print(f"    - 句长变异系数: {report.stats['sentence_length_cv']:.3f}")
        print(f"    - 连接词密度: {report.stats['connector_density']:.2f}")

        # 概率特征分析（如果可用）
        if report.probability:
            print()
            print("概率特征检测结果:")
            print(f"  - 困惑度 (PPL):")
            print(f"    - 整体: {report.probability['overall_ppl']:.1f}")
            print(f"    - 最低: {report.probability['min_ppl']:.1f}")
            print(f"    - 最高: {report.probability['max_ppl']:.1f}")
            print(f"  - LRR 指标:")
            print(f"    - LRR 分数: {report.probability['lrr_score']:.4f}")
            print(f"    - 平均 Log-Likelihood: {report.probability['avg_log_likelihood']:.4f}")
            print(f"    - 平均 Log-Rank: {report.probability['avg_log_rank']:.4f}")
            print(f"  - Rank 分布:")
            print(f"    - Top-1 选择率: {report.probability['top1_ratio']:.1%}")
            print(f"    - Top-5 选择率: {report.probability['top5_ratio']:.1%}")
            print(f"    - 罕见词比例: {report.probability['rare_word_ratio']:.2%}")
            print(f"  - 综合风险: {report.probability['risk_score']:.2f} ({report.probability['risk_level']})")

            interpretation = []
            if report.probability['overall_ppl'] < 35:
                interpretation.append("困惑度极低，高度疑似 AI")
            elif report.probability['overall_ppl'] < 50:
                interpretation.append("困惑度偏低，可能包含 AI 特征")
            else:
                interpretation.append("困惑度正常，更像人类写作")

            if report.probability['lrr_score'] > -0.3:
                interpretation.append("LRR 异常高，极度偏好高概率词")
            elif report.probability['lrr_score'] > -0.5:
                interpretation.append("LRR 偏高，词汇选择较确定")

            if report.probability['top1_ratio'] > 0.7:
                interpretation.append("Top-1 选择率过高，像 AI 写作")

            if report.probability['rare_word_ratio'] < 0.02:
                interpretation.append("罕见词比例低，词汇过于常见")

            print(f"  - 解读: {'; '.join(interpretation)}")
        else:
            print()
            print("概率特征检测: 不可用（需安装 transformers 和 torch）")
            print("  运行: uv sync --extra perplexity")

        print()

    print("=" * 70)
    print("演示完成")
    print("=" * 70)


if __name__ == "__main__":
    demo()

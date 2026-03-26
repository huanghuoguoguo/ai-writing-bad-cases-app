"""
困惑度检测测试脚本

运行方式:
    UV_CACHE_DIR=.uv-cache uv run python tests/test_perplexity.py

注意：需要先安装 perplexity 依赖：
    uv sync --extra perplexity
"""

import sys
from pathlib import Path

# 确保能导入 src 下的模块
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from ai_badcase_app.perplexity import analyze_perplexity


# 测试数据
AI_TEXT = """
在这个信息爆炸的时代，我们需要重新思考效率的本质。
首先，我们要理解什么是真正的效率。其次，我们要分析影响效率的关键因素。
最后，我们要提出提升效率的系统性方法。综上所述，效率不是简单的速度问题，
而是一个涉及认知、方法和工具的综合性议题。
"""

HUMAN_TEXT = """
说实话，我以前真没觉得效率是个问题。每天就是干活呗，干完就行。
后来换了个老板，天天催进度，我才开始琢磨这事。
试了一堆方法，有的好用有的坑，现在也就勉强能按时交活。
你说什么系统性方法？我不懂那些，反正能搞定就行。
"""


def test_perplexity_detection():
    """测试困惑度检测能区分 AI 和人类文本"""
    print("=" * 60)
    print("困惑度检测测试")
    print("=" * 60)

    try:
        # 测试 AI 文本
        print("\n--- 测试 AI 文本 ---")
        ai_result = analyze_perplexity(AI_TEXT)
        if ai_result:
            print(f"整体困惑度: {ai_result.overall_ppl}")
            print(f"最低困惑度: {ai_result.min_ppl}")
            print(f"风险分数: {ai_result.risk_score}")
            print(f"风险等级: {ai_result.risk_level}")
            print(f"原因: {ai_result.reasons}")
            print(f"\n窗口详情 ({len(ai_result.window_results)} 个窗口):")
            for w in ai_result.window_results[:3]:  # 只显示前3个
                print(f"  [{w.start_pos}:{w.end_pos}] PPL={w.ppl:.1f}, Tokens={w.token_count}")
        else:
            print("困惑度检测不可用（可能需要安装 transformers 和 torch）")
            return False

        # 测试人类文本
        print("\n--- 测试人类文本 ---")
        human_result = analyze_perplexity(HUMAN_TEXT)
        if human_result:
            print(f"整体困惑度: {human_result.overall_ppl}")
            print(f"最低困惑度: {human_result.min_ppl}")
            print(f"风险分数: {human_result.risk_score}")
            print(f"风险等级: {human_result.risk_level}")
            print(f"原因: {human_result.reasons}")

        # 验证：AI 文本困惑度应该更低
        if ai_result and human_result:
            print("\n--- 对比结果 ---")
            if ai_result.overall_ppl < human_result.overall_ppl:
                print(f"✓ AI 文本困惑度 ({ai_result.overall_ppl:.1f}) < 人类文本 ({human_result.overall_ppl:.1f})")
                print("  符合预期：AI 文本更可预测")
            else:
                print(f"✗ AI 文本困惑度 ({ai_result.overall_ppl:.1f}) >= 人类文本 ({human_result.overall_ppl:.1f})")
                print("  注意：样本量小或模型特性可能导致此现象")

            if ai_result.risk_score > human_result.risk_score:
                print(f"✓ AI 文本风险分 ({ai_result.risk_score:.2f}) > 人类文本 ({human_result.risk_score:.2f})")
            else:
                print(f"✗ AI 文本风险分 ({ai_result.risk_score:.2f}) <= 人类文本 ({human_result.risk_score:.2f})")

        print("\n" + "=" * 60)
        print("测试完成")
        print("=" * 60)
        return True

    except Exception as e:
        print(f"\n错误: {e}")
        print("\n提示: 如果看到 ImportError，请先安装依赖:")
        print("  uv sync --extra perplexity")
        return False


if __name__ == "__main__":
    success = test_perplexity_detection()
    sys.exit(0 if success else 1)

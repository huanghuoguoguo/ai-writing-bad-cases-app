# AI Bad Case Detector - 使用说明

## 工具定位

这是一个**纯工具函数**，输入文本，输出疑似 AI 的片段证据。
**不做最终判断**，由调用方（AI/Agent）自行决策。

## 快速开始

```python
from ai_badcase_app.analyzer import analyze_text

# 分析文本
report = analyze_text("你的文本内容")

# 查看 JSON 结果
print(report.to_json())

# 或者查看疑似片段
for seg in report.suspected_segments:
    print(f"风险等级: {seg.risk_level}")
    print(f"风险分数: {seg.risk_score}")
    print(f"原因: {seg.reasons}")
    print(f"建议: {seg.suggestions}")
```

## 输出结构

```json
{
  "basic_info": {
    "total_chars": 182,
    "total_sentences": 9,
    "total_paragraphs": 3
  },
  "suspected_segments": [
    {
      "text": "片段文本...",
      "risk_score": 0.98,
      "risk_level": "high",
      "reasons": ["程式化序列词 (regex)"],
      "suggestions": ["删除明显的排比式衔接词"]
    }
  ],
  "statistics": {
    "sentence_length_cv": 0.61,
    "connector_density": 0.0,
    "passive_ratio": 0.0,
    "lexical_diversity": 1.0
  },
  "summary": {
    "total_segments_checked": 3,
    "suspected_segments_count": 1,
    "high_risk_count": 0,
    "medium_risk_count": 0,
    "note": "这些只是疑似片段，需要调用方进一步判断是否真的是 AI 生成"
  }
}
```

## 风险等级说明

| 等级 | 分数范围 | 含义 |
|------|----------|------|
| high | >= 0.8 | 明确命中 bad case，极可能是 AI |
| medium | 0.5 - 0.8 | 统计异常或模糊匹配，需要人工复核 |
| low | < 0.5 | 轻微可疑，可能是误判 |

## 检测维度

### 1. 规则匹配（精确）
- 程式化序列词：首先...其次...最后
- 对举句式：不是...而是...
- 空洞价值声明：具有重要意义
- 过度使用被动语态
- 等等（47 个 bad cases）

### 2. 统计启发（模式识别）
- 句长变异系数（AI 更均匀）
- 连接词密度（AI 更高）
- 被动语态比例
- 词汇多样性

### 3. 语义召回（已有框架，待启用）
- SeekDB 向量相似度
- 捕捉语义相似但表述不同的变体

## 调用方决策示例

```python
def is_aigc_text(text: str, threshold: float = 0.7) -> bool:
    """调用方自行决策是否 AIGC"""
    report = analyze_text(text)
    
    # 策略 1：高风险片段占比
    if report.summary["high_risk_count"] >= 2:
        return True
    
    # 策略 2：平均分超过阈值
    if report.suspected_segments:
        avg_score = sum(s.risk_score for s in report.suspected_segments) / len(report.suspected_segments)
        return avg_score > threshold
    
    return False
```

## 运行测试

```bash
uv run python tests/test_analyzer.py
```

11 个测试用例验证：
- AI 文本能被正确检测
- 人类文本不被过度标记
- 边界情况正确处理
- 统计特征准确计算

## CLI 使用

```bash
# 分析文件
uv run python -m ai_badcase_app.cli --input article.txt --format json

# 调整模糊匹配阈值
uv run python -m ai_badcase_app.cli --input article.txt --fuzzy-threshold 70
```

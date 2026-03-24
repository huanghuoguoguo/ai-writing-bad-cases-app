# AIGC 检测 Skill

检测文本中的 AI 生成痕迹，定位疑似片段，辅助判断是否需修改。

## 用法

```
/aigc-detector <文本内容或文件路径>
```

## 工作流程

当你需要判断一段文本是否有 AIGC 痕迹时：

### 1. 调用检测工具

```python
from ai_badcase_app.analyzer import analyze_text

report = analyze_text(text)
```

### 2. 解读结果

**关键指标：**

| 指标 | 健康值 | AI 典型值 | 说明 |
|------|--------|-----------|------|
| 句长变异系数 | > 0.5 | < 0.4 | AI 句子长度更均匀 |
| 连接词密度 | < 1.0 | > 2.0 | AI 更爱用"首先/其次/因此" |
| 高风险片段占比 | < 5% | > 15% | 明确命中 bad case |

**风险等级：**
- `high` (>=0.8)：明确 AI 套路（如"首先...其次...最后"）
- `medium` (0.5-0.8)：统计异常或模糊匹配，需人工复核
- `low` (<0.5)：轻微可疑

### 3. 做出判断

**决策规则：**

```
高风险占比 > 15%  →  ⚠️ 有明显 AIGC 痕迹，建议逐段修改
高风险占比 5-15%  →  🟡 有轻微痕迹，可针对性修改
高风险占比 < 5%   →  ✅ 人味充足，无需修改
```

**注意：** 工具只提供证据，最终判断由你决定。

### 4. 修改建议（如需要）

**常见 AI 模式及改写：**

| AI 模式 | 改写方式 |
|---------|----------|
| 首先...其次...最后... | 先说说...另外还有...最后一点... |
| 不是...而是... | 直接说对比，不要对举句式 |
| 具有重要意义 | 写清楚具体解决了什么问题 |
| 综上所述 | 删除，直接给结论 |
| 被动语态 | 改为主动，明确主语 |

## 示例

### 示例 1：检测通过

```
文本：我之前觉得线程越多越高效。后来发现 Python FastAPI 是单线程的。
单线程能撑高并发？想不通。再后来才明白...

检测结果：
- 句长变异: 0.61 ✅
- 连接词密度: 0.0 ✅
- 高风险片段: 0

判断：✅ 人味充足，口语化、有"我"的视角、短句、疑问语气
```

### 示例 2：需要修改

```
文本：首先，我们需要理解AI写作的本质。其次，分析其产生的原因。
最后，提出有效的检测方法。综上所述...

检测结果：
- 句长变异: 0.44 ⚠️
- 连接词密度: 3.28 ⚠️
- 高风险片段: 3/3 (100%)

判断：⚠️ 明显 AIGC 痕迹

修改：
- "首先...其次...最后" → "想理解 AI 写作，得先看本质"
- "综上所述" → 删除
```

### 示例 3：边界情况

```
文本：真正重要的不是速度，而是你是否能长期坚持。

检测结果：
- 高风险片段: 1 个（不是...而是...）
- 句长变异: 0.82 ✅

判断：🟡 单句有问题，但整体人味足

决策：可改可不改。如修改："速度没那么重要，能长期坚持才是关键"
```

## 实现代码

```python
import re
from ai_badcase_app.analyzer import analyze_text

def check_aigc(text: str, filepath: str = None) -> str:
    """
    检测文本 AIGC 痕迹，返回判断和建议
    """
    if filepath:
        with open(filepath, 'r') as f:
            text = f.read()
        text = re.sub(r'^---.*?---', '', text, flags=re.DOTALL).strip()

    report = analyze_text(text)

    total = report.summary['total_segments_checked']
    high = report.summary['high_risk_count']
    high_ratio = high / max(1, total)

    # 判断
    if high_ratio > 0.15:
        judgment = "⚠️ 有明显 AIGC 痕迹，建议修改"
    elif high_ratio > 0.05:
        judgment = "🟡 有轻微 AIGC 痕迹，可选修改"
    else:
        judgment = "✅ 人味充足，无需修改"

    # 构建输出
    lines = [
        f"检测段落: {total}",
        f"高风险片段: {high} ({high_ratio*100:.1f}%)",
        f"句长变异: {report.stats['sentence_length_cv']:.2f}",
        f"连接词密度: {report.stats['connector_density']:.2f}/100字",
        "",
        f"判断: {judgment}",
    ]

    # 高风险详情
    if report.suspected_segments:
        lines.append("\n疑似片段:")
        for seg in report.suspected_segments[:5]:
            if seg.risk_level == "high":
                preview = seg.text[:50].replace('\n', ' ')
                lines.append(f"  • [{seg.risk_score:.2f}] {preview}...")
                lines.append(f"    原因: {', '.join(seg.reasons[:2])}")

    return "\n".join(lines)

# 使用
result = check_aigc(filepath="/path/to/blog.md")
print(result)
```

## 注意事项

1. **工具定位**：只提供疑似片段证据，不做最终"是/否"判断
2. **误判可能**：某些人类也会用"不是...而是..."，结合上下文判断
3. **统计异常**：句长变异 < 0.4 可能是 AI，但也可能是刻意整理的笔记
4. **技术笔记**：表格、清单、结构化内容可能触发误判，需人工复核

## 检测维度

### 规则匹配（精确）
- 程式化序列词：首先...其次...最后
- 对举句式：不是...而是...
- 空洞价值声明：具有重要意义
- 过度使用被动语态
- 总计 47 个 bad cases

### 统计启发（模式识别）
- 句长变异系数
- 连接词密度
- 被动语态比例
- 词汇多样性

## 依赖

```bash
cd /home/yhh/learn/ai-writing-bad-cases-app
uv run python -c "from ai_badcase_app.analyzer import analyze_text; print('OK')"
```

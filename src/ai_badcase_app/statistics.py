from __future__ import annotations

import re
import math
from dataclasses import dataclass


@dataclass
class TextStatistics:
    """文本统计特征分析"""
    # 基础统计
    char_count: int
    sentence_count: int
    word_count: int
    paragraph_count: int

    # 句长特征
    sentence_lengths: list[int]  # 每个句子的字数
    sentence_length_mean: float
    sentence_length_std: float  # 标准差，AI 通常更小（更均匀）
    sentence_length_cv: float   # 变异系数 = std/mean
    adjacent_length_delta_mean: float  # 相邻句长度跳变均值
    extreme_sentence_ratio: float  # 极长/极短句占比

    # 词汇特征
    unique_words: int
    lexical_diversity: float  # 词汇多样性 = unique_words / word_count

    # 连接词密度
    connector_density: float  # 连接词出现频率

    # 被动语态比例
    passive_ratio: float

    # 标点特征
    comma_density: float      # 逗号密度（AI 倾向于更长的复合句）
    dash_density: float       # 破折号密度（AI 喜欢用破折号解释隐喻）

    # 结构特征
    three_part_enumeration_count: int  # 三段式列举出现次数
    meta_discourse_count: int  # 元话语标记出现次数

    # 句式特征
    uniform_sentence_group_count: int  # 均匀句长组的数量


# 中文连接词列表
_CONNECTORS = [
    "首先", "其次", "然后", "最后", "第一", "第二", "第三",
    "因此", "所以", "因而", "于是", "从而", "由此可见",
    "综上所述", "总而言之", "一言以蔽之", "总的来说",
    "此外", "另外", "除此之外", "不仅如此", "而且",
    "但是", "然而", "不过", "可是",
    "比如", "例如", "譬如",
]

# 被动语态标记
# 这里故意不用单字 "为" / "由"：
# - "为了""认为""因为" 这类常见写法会被大量误判
# - 技术博客里 "由此可见" 更像连接词，不是被动句
_PASSIVE_PATTERNS = [
    r"被",
    r"受到",
    r"得到",
    r"加以",
    r"由[^，。！？!?]{0,12}(完成|实现|触发|构成|导致|决定)",
]

# 三段式列举模式
_THREE_PART_PATTERNS = [
    r"(首先|第一).{2,40}(其次|第二).{2,40}(最后|第三)",
    r"(一方面|从一方面).{2,30}(另一方面|从另一方面)",
]

# 元话语标记
_META_DISCOURSE_PATTERNS = [
    r"(修改后|改写后|可以改成|如果你愿意|建议改成)",
    r"(以下是|上面是|这段是).{0,12}(修改|改写|润色)",
    r"(值得|需要)(注意|指出|强调)的是",
    r"(显而易见|毫无疑问|毋庸置疑|不言而喻)",
]

# 破折号解释模式
_DASH_EXPLAIN_PATTERNS = [
    r"——(即|也就是|这意味着|这表明)",
]


def _split_sentences(text: str) -> list[str]:
    """分句，基于标点"""
    sentences = re.split(r'[。！？!?.\n]+', text)
    return [s.strip() for s in sentences if s.strip()]


def _extract_words(text: str) -> list[str]:
    """简单分词，基于连续中文字符"""
    words = re.findall(r'[\u4e00-\u9fff]+', text)
    return words


def _count_connectors(text: str) -> int:
    """统计连接词出现次数"""
    return sum(text.count(connector) for connector in _CONNECTORS)


def _count_pattern_matches(text: str, patterns: list[str]) -> int:
    """统计多个正则模式的匹配总数"""
    return sum(len(re.findall(pattern, text)) for pattern in patterns)


def _detect_uniform_sentence_groups(sentence_lengths: list[int], tolerance: int = 5) -> int:
    """检测均匀句长组的数量（连续3+句长度相近）"""
    if len(sentence_lengths) < 3:
        return 0

    groups = 0
    current_group_size = 1

    for i in range(1, len(sentence_lengths)):
        if abs(sentence_lengths[i] - sentence_lengths[i - 1]) <= tolerance:
            current_group_size += 1
        else:
            if current_group_size >= 3:
                groups += 1
            current_group_size = 1

    if current_group_size >= 3:
        groups += 1

    return groups


def analyze_text_statistics(text: str) -> TextStatistics:
    """分析文本的统计特征"""
    paragraphs = [p for p in re.split(r'\n\s*\n', text) if p.strip()]
    sentences = _split_sentences(text)
    words = _extract_words(text)

    char_count = len(text)
    sentence_count = len(sentences)
    word_count = len(words)
    paragraph_count = len(paragraphs)

    # 句长统计
    sentence_lengths = [len(s) for s in sentences]
    if sentence_lengths:
        mean_length = sum(sentence_lengths) / len(sentence_lengths)
        variance = sum((x - mean_length) ** 2 for x in sentence_lengths) / len(sentence_lengths)
        std_length = math.sqrt(variance)
        cv = std_length / mean_length if mean_length > 0 else 0
        adjacent_deltas = [
            abs(sentence_lengths[i] - sentence_lengths[i - 1])
            for i in range(1, len(sentence_lengths))
        ]
        delta_mean = (
            sum(adjacent_deltas) / len(adjacent_deltas)
            if adjacent_deltas else 0
        )
        if mean_length > 0:
            extreme_count = sum(
                1 for length in sentence_lengths
                if length <= mean_length * 0.5 or length >= mean_length * 1.5
            )
            extreme_ratio = extreme_count / len(sentence_lengths)
        else:
            extreme_ratio = 0
    else:
        mean_length = std_length = cv = delta_mean = extreme_ratio = 0

    # 词汇多样性
    unique_words = len(set(words))
    lexical_diversity = unique_words / word_count if word_count > 0 else 0

    # 连接词密度（每100字的连接词数）
    connector_count = _count_connectors(text)
    connector_density = (connector_count / char_count) * 100 if char_count > 0 else 0

    # 被动语态比例（被动标记数 / 句子数）
    passive_count = _count_pattern_matches(text, _PASSIVE_PATTERNS)
    passive_ratio = passive_count / sentence_count if sentence_count > 0 else 0

    # 逗号密度
    comma_count = text.count('，') + text.count(',')
    comma_density = (comma_count / sentence_count) if sentence_count > 0 else 0

    # 破折号密度（统计破折号解释模式）
    dash_count = _count_pattern_matches(text, _DASH_EXPLAIN_PATTERNS)
    dash_density = (dash_count / sentence_count) if sentence_count > 0 else 0

    # 三段式列举
    three_part_enumeration_count = _count_pattern_matches(text, _THREE_PART_PATTERNS)

    # 元话语标记
    meta_discourse_count = _count_pattern_matches(text, _META_DISCOURSE_PATTERNS)

    # 均匀句长组
    uniform_sentence_group_count = _detect_uniform_sentence_groups(sentence_lengths)

    return TextStatistics(
        char_count=char_count,
        sentence_count=sentence_count,
        word_count=word_count,
        paragraph_count=paragraph_count,
        sentence_lengths=sentence_lengths,
        sentence_length_mean=round(mean_length, 2),
        sentence_length_std=round(std_length, 2),
        sentence_length_cv=round(cv, 4),
        adjacent_length_delta_mean=round(delta_mean, 4),
        extreme_sentence_ratio=round(extreme_ratio, 4),
        unique_words=unique_words,
        lexical_diversity=round(lexical_diversity, 4),
        connector_density=round(connector_density, 4),
        passive_ratio=round(passive_ratio, 4),
        comma_density=round(comma_density, 4),
        dash_density=round(dash_density, 4),
        three_part_enumeration_count=three_part_enumeration_count,
        meta_discourse_count=meta_discourse_count,
        uniform_sentence_group_count=uniform_sentence_group_count,
    )


def detect_statistical_anomalies(stats: TextStatistics) -> list[dict]:
    """基于统计特征检测异常，返回风险项列表"""
    anomalies = []

    # 规则 1: 句长变异系数过低（AI 句子长度更均匀）
    if stats.sentence_length_cv < 0.3 and stats.sentence_count >= 3:
        anomalies.append({
            "type": "low_sentence_variation",
            "label": "句子长度过于均匀",
            "score": round(0.7 - stats.sentence_length_cv, 2),
            "description": f"句长变异系数 {stats.sentence_length_cv}，人类写作通常变化更大",
            "rewrite_hint": "刻意改变句子长度，加入短句打破节奏",
            "diagnostic_dimensions": ["rhythm", "authenticity"],
        })

    if stats.adjacent_length_delta_mean < 8 and stats.sentence_count >= 4:
        anomalies.append({
            "type": "low_adjacent_sentence_delta",
            "label": "相邻句长度跳变太小",
            "score": round(min(0.75, 0.82 - stats.adjacent_length_delta_mean / 20), 2),
            "description": f"相邻句平均长度差 {stats.adjacent_length_delta_mean:.2f}，段落起伏偏平",
            "rewrite_hint": "穿插明显更短或更长的句子，不要让每句都沿着同一节奏往前走",
            "diagnostic_dimensions": ["rhythm", "authenticity"],
        })

    if stats.extreme_sentence_ratio < 0.12 and stats.sentence_count >= 4:
        anomalies.append({
            "type": "low_extreme_sentence_ratio",
            "label": "长短句交替不足",
            "score": round(min(0.7, 0.5 + (0.12 - stats.extreme_sentence_ratio) * 2), 2),
            "description": f"极长/极短句占比 {stats.extreme_sentence_ratio:.2f}，段落缺少明显节奏拐点",
            "rewrite_hint": "补一两句更短的判断句，或者把某一句展开，拉开句长层次",
            "diagnostic_dimensions": ["rhythm", "authenticity"],
        })

    # 规则 1.1: 均匀句长组检测
    if stats.uniform_sentence_group_count >= 1 and stats.sentence_count >= 5:
        anomalies.append({
            "type": "uniform_sentence_groups",
            "label": "存在均匀句长段落",
            "score": round(min(0.85, 0.6 + stats.uniform_sentence_group_count * 0.15), 2),
            "description": f"发现 {stats.uniform_sentence_group_count} 处连续句子长度相近",
            "rewrite_hint": "打断均匀节奏：缩短、拉长、或拆成两句",
            "diagnostic_dimensions": ["rhythm", "authenticity"],
        })

    # 规则 2: 连接词密度过高
    if stats.connector_density > 2.0:
        anomalies.append({
            "type": "high_connector_density",
            "label": "连接词密度过高",
            "score": min(0.9, stats.connector_density / 3),
            "description": f"每100字出现 {stats.connector_density:.2f} 个连接词",
            "rewrite_hint": "删除逻辑连接词，用上下文自然衔接",
            "diagnostic_dimensions": ["connector_driven", "over_explicitness"],
        })

    # 规则 3: 被动语态比例过高
    if stats.passive_ratio > 0.5:
        anomalies.append({
            "type": "high_passive_ratio",
            "label": "被动语态过多",
            "score": min(0.8, stats.passive_ratio),
            "description": f"平均每句 {stats.passive_ratio:.2f} 个被动标记",
            "rewrite_hint": "改为主动语态，明确主语",
            "diagnostic_dimensions": ["average_style", "meta_discourse_density"],
        })

    # 规则 4: 词汇多样性过低（AI 容易重复用词）
    if stats.lexical_diversity < 0.4 and stats.word_count > 50:
        anomalies.append({
            "type": "low_lexical_diversity",
            "label": "词汇重复度高",
            "score": round(0.8 - stats.lexical_diversity, 2),
            "description": f"词汇多样性 {stats.lexical_diversity}，存在重复用词",
            "rewrite_hint": "使用同义词替换，避免重复",
            "diagnostic_dimensions": ["average_style", "refinement"],
        })

    # 规则 5: 破折号密度过高
    if stats.dash_density > 0.3 and stats.sentence_count >= 3:
        anomalies.append({
            "type": "high_dash_density",
            "label": "破折号使用过多",
            "score": round(min(0.78, 0.5 + stats.dash_density * 0.3), 2),
            "description": f"平均每句 {stats.dash_density:.2f} 个破折号，可能存在过度解释",
            "rewrite_hint": "删除破折号后的解释，相信读者能理解",
            "diagnostic_dimensions": ["trust", "meta_discourse_density"],
        })

    # 规则 6: 三段式列举检测
    if stats.three_part_enumeration_count >= 1:
        anomalies.append({
            "type": "three_part_enumeration",
            "label": "三段式列举",
            "score": round(min(0.82, 0.7 + stats.three_part_enumeration_count * 0.1), 2),
            "description": f"发现 {stats.three_part_enumeration_count} 处三段式列举结构",
            "rewrite_hint": "改成两项或四项，或去掉编号直接列举",
            "diagnostic_dimensions": ["structure_symmetry", "average_style"],
        })

    # 规则 7: 元话语污染检测
    if stats.meta_discourse_count >= 1:
        anomalies.append({
            "type": "meta_discourse_pollution",
            "label": "元话语污染",
            "score": round(min(0.9, 0.8 + stats.meta_discourse_count * 0.05), 2),
            "description": f"发现 {stats.meta_discourse_count} 处元话语标记",
            "rewrite_hint": "删除元话语标记，只保留正文内容",
            "diagnostic_dimensions": ["pure_output", "meta_discourse_density"],
        })

    return anomalies
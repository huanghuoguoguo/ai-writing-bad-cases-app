from __future__ import annotations

import re
import math
from dataclasses import dataclass
from collections import Counter


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

    # 词汇特征
    unique_words: int
    lexical_diversity: float  # 词汇多样性 = unique_words / word_count

    # 连接词密度
    connector_density: float  # 连接词出现频率

    # 被动语态比例
    passive_ratio: float

    # 标点特征
    comma_density: float      # 逗号密度（AI 倾向于更长的复合句）


# 中文连接词列表（需要扩充）
_CONNECTORS = [
    "首先", "其次", "然后", "最后", "第一", "第二", "第三",
    "因此", "所以", "因而", "于是", "从而", "由此可见",
    "综上所述", "总而言之", "一言以蔽之", "总的来说",
    "此外", "另外", "除此之外", "不仅如此", "而且",
    "但是", "然而", "不过", "可是",
    "比如", "例如", "譬如",
]

# 被动语态标记
_PASSIVE_MARKERS = ["被", "由", "为", "受到", "得到", "加以"]


def _split_sentences(text: str) -> list[str]:
    """分句，基于标点"""
    # 中文句子结束标点
    sentences = re.split(r'[。！？!?.\n]+', text)
    return [s.strip() for s in sentences if s.strip()]


def _extract_words(text: str) -> list[str]:
    """简单分词，基于连续中文字符"""
    # 匹配中文字符序列
    words = re.findall(r'[\u4e00-\u9fff]+', text)
    return words


def _count_connectors(text: str) -> int:
    """统计连接词出现次数"""
    count = 0
    for connector in _CONNECTORS:
        count += text.count(connector)
    return count


def _count_passive_markers(text: str) -> int:
    """统计被动语态标记"""
    count = 0
    for marker in _PASSIVE_MARKERS:
        # 避免重复计数，简单统计
        count += len(re.findall(rf'{marker}', text))
    return count


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
    else:
        mean_length = std_length = cv = 0

    # 词汇多样性
    all_words_text = ' '.join(words)
    unique_words = len(set(words))
    lexical_diversity = unique_words / word_count if word_count > 0 else 0

    # 连接词密度（每100字的连接词数）
    connector_count = _count_connectors(text)
    connector_density = (connector_count / char_count) * 100 if char_count > 0 else 0

    # 被动语态比例（被动标记数 / 句子数）
    passive_count = _count_passive_markers(text)
    passive_ratio = passive_count / sentence_count if sentence_count > 0 else 0

    # 逗号密度
    comma_count = text.count('，') + text.count(',')
    comma_density = (comma_count / sentence_count) if sentence_count > 0 else 0

    return TextStatistics(
        char_count=char_count,
        sentence_count=sentence_count,
        word_count=word_count,
        paragraph_count=paragraph_count,
        sentence_lengths=sentence_lengths,
        sentence_length_mean=round(mean_length, 2),
        sentence_length_std=round(std_length, 2),
        sentence_length_cv=round(cv, 4),
        unique_words=unique_words,
        lexical_diversity=round(lexical_diversity, 4),
        connector_density=round(connector_density, 4),
        passive_ratio=round(passive_ratio, 4),
        comma_density=round(comma_density, 4),
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
        })

    # 规则 2: 连接词密度过高
    if stats.connector_density > 2.0:
        anomalies.append({
            "type": "high_connector_density",
            "label": "连接词密度过高",
            "score": min(0.9, stats.connector_density / 3),
            "description": f"每100字出现 {stats.connector_density:.2f} 个连接词",
            "rewrite_hint": "删除逻辑连接词，用上下文自然衔接",
        })

    # 规则 3: 被动语态比例过高
    if stats.passive_ratio > 0.5:
        anomalies.append({
            "type": "high_passive_ratio",
            "label": "被动语态过多",
            "score": min(0.8, stats.passive_ratio),
            "description": f"平均每句 {stats.passive_ratio:.2f} 个被动标记",
            "rewrite_hint": "改为主动语态，明确主语",
        })

    # 规则 4: 词汇多样性过低（AI 容易重复用词）
    if stats.lexical_diversity < 0.4 and stats.word_count > 50:
        anomalies.append({
            "type": "low_lexical_diversity",
            "label": "词汇重复度高",
            "score": round(0.8 - stats.lexical_diversity, 2),
            "description": f"词汇多样性 {stats.lexical_diversity}，存在重复用词",
            "rewrite_hint": "使用同义词替换，避免重复",
        })

    return anomalies

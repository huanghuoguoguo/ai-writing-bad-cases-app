"""
概率特征检测模块 - 基于预训练语言模型的 AIGC 检测

核心指标：
1. 困惑度 (Perplexity, PPL): 文本的可预测性
2. LRR (Log-Likelihood Log-Rank Ratio): AI 的"极度避险"特征
3. Log-Rank 分布: Token 概率排名的分布曲线

原理：
- AI 生成文本倾向于选择高概率、排名靠前的词
- 人类写作有更多"惊奇"，会使用冷门词
- LRR 能灵敏捕捉这种概率特征差异
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence


from .models import risk_level as _risk_level


@dataclass
class ProbabilityResult:
    """概率特征分析结果"""
    # 困惑度指标
    overall_ppl: float
    min_ppl: float
    max_ppl: float
    ppl_variance: float

    # LRR 指标 (Log-Likelihood Log-Rank Ratio)
    lrr_score: float             # LRR 比值，越高越像 AI
    avg_log_likelihood: float    # 平均对数似然
    avg_log_rank: float          # 平均对数排名

    # Log-Rank 分布
    rank_distribution: dict      # 排名分布统计
    top1_ratio: float            # 选 Top-1 词的比例
    top5_ratio: float            # 选 Top-5 词的比例
    rare_word_ratio: float       # 冷门词（排名>100）比例

    # 综合评估
    window_results: list[WindowResult]
    risk_score: float
    risk_level: str
    reasons: list[str]
    suggestions: list[str]


@dataclass
class WindowResult:
    """单个窗口的分析结果"""
    text: str
    start_pos: int
    end_pos: int
    ppl: float
    lrr: float
    token_count: int
    top1_ratio: float


class ProbabilityDetector:
    """
    基于 GPT-2 的概率特征检测器

    检测维度：
    1. PPL: 文本整体可预测性
    2. LRR: Log-Likelihood / Log-Rank，捕捉"避险"特征
    3. Rank 分布: Token 选择的概率排名分布

    使用中文适配的 GPT-2 模型：uer/gpt2-chinese-cluecorpussmall
    """

    DEFAULT_MODEL = "uer/gpt2-chinese-cluecorpussmall"

    # 风险阈值
    PPL_HIGH_RISK = 35.0
    PPL_MEDIUM_RISK = 50.0
    LRR_HIGH_RISK = -0.3      # LRR > -0.3 高度疑似 AI
    LRR_MEDIUM_RISK = -0.5    # LRR > -0.5 中等嫌疑
    TOP1_HIGH_RISK = 0.7      # Top-1 选择率 > 70%
    RARE_WORD_LOW = 0.02      # 冷门词 < 2%

    def __init__(self, model_name: str | None = None, device: str = "cpu"):
        self.model_name = model_name or self.DEFAULT_MODEL
        self.device = device
        self._model = None
        self._tokenizer = None
        self._available = False

    def _lazy_init(self) -> bool:
        """延迟初始化模型"""
        if self._model is not None:
            return self._available

        try:
            from transformers import AutoTokenizer, GPT2LMHeadModel
            import torch

            # Some GPT-style Chinese checkpoints declare a non-GPT tokenizer
            # (for example BertTokenizer). AutoTokenizer respects that config.
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self._model = GPT2LMHeadModel.from_pretrained(self.model_name)
            self._model.to(self.device)
            self._model.eval()
            self._available = True

        except ImportError as e:
            raise ProbabilityUnavailableError(
                "transformers 或 torch 未安装。"
                "运行: uv add transformers torch"
            ) from e

        except Exception as e:
            raise ProbabilityRuntimeError(
                f"无法加载模型 {self.model_name}: {e}"
            ) from e

        return self._available

    def analyze(
        self,
        text: str,
        window_size: int = 256,
        stride: int = 128,
        max_length: int = 512,
    ) -> ProbabilityResult | None:
        """
        分析文本的概率特征

        Args:
            text: 输入文本
            window_size: 滑动窗口字符数
            stride: 窗口滑动步长
            max_length: 最大序列长度

        Returns:
            ProbabilityResult 或 None
        """
        if not self._lazy_init():
            return None

        if not text or len(text.strip()) < 50:
            return self._empty_result()

        # 滑动窗口分析
        windows = self._create_windows(text, window_size, stride)
        window_results: list[WindowResult] = []

        all_log_likelihoods = []
        all_log_ranks = []
        all_ranks = []

        for start, end, window_text in windows:
            result = self._analyze_window(window_text, max_length)
            if result:
                window_results.append(WindowResult(
                    text=window_text[:100] + "..." if len(window_text) > 100 else window_text,
                    start_pos=start,
                    end_pos=end,
                    ppl=result["ppl"],
                    lrr=result["lrr"],
                    token_count=result["token_count"],
                    top1_ratio=result["top1_ratio"],
                ))
                all_log_likelihoods.extend(result["log_likelihoods"])
                all_log_ranks.extend(result["log_ranks"])
                all_ranks.extend(result["ranks"])

        if not window_results:
            return None

        # 计算整体指标
        ppls = [w.ppl for w in window_results]
        overall_ppl = sum(ppls) / len(ppls)

        # 计算 LRR
        avg_ll = sum(all_log_likelihoods) / len(all_log_likelihoods) if all_log_likelihoods else 0
        avg_lr = sum(all_log_ranks) / len(all_log_ranks) if all_log_ranks else 0
        lrr_score = avg_ll / avg_lr if avg_lr != 0 else 0

        # 计算 Rank 分布
        rank_dist = self._compute_rank_distribution(all_ranks)

        # 风险评估
        risk_score, risk_level, reasons, suggestions = self._assess_risk(
            overall_ppl, lrr_score, rank_dist, window_results
        )

        return ProbabilityResult(
            overall_ppl=round(overall_ppl, 2),
            min_ppl=round(min(ppls), 2),
            max_ppl=round(max(ppls), 2),
            ppl_variance=round(sum((p - overall_ppl) ** 2 for p in ppls) / len(ppls), 2),
            lrr_score=round(lrr_score, 4),
            avg_log_likelihood=round(avg_ll, 4),
            avg_log_rank=round(avg_lr, 4),
            rank_distribution=rank_dist,
            top1_ratio=rank_dist["top1_ratio"],
            top5_ratio=rank_dist["top5_ratio"],
            rare_word_ratio=rank_dist["rare_ratio"],
            window_results=window_results,
            risk_score=risk_score,
            risk_level=risk_level,
            reasons=reasons,
            suggestions=suggestions,
        )

    def _analyze_window(self, text: str, max_length: int) -> dict | None:
        """分析单个窗口，返回 PPL、LRR、Rank 等指标"""
        import torch

        encodings = self._tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
        )

        input_ids = encodings.input_ids.to(self.device)
        if input_ids.size(1) < 5:
            return None

        with torch.no_grad():
            outputs = self._model(input_ids, labels=input_ids)
            logits = outputs.logits

        # 计算各项指标
        log_likelihoods = []
        log_ranks = []
        ranks = []
        top1_count = 0

        # 从第 1 个 token 开始（跳过 BOS）
        for i in range(1, input_ids.size(1)):
            token_logits = logits[0, i - 1]  # 预测第 i 个 token
            token_id = input_ids[0, i].item()

            # 概率分布
            probs = torch.softmax(token_logits, dim=-1)
            token_prob = probs[token_id].item()

            # 排名
            sorted_indices = torch.argsort(probs, descending=True)
            rank = (sorted_indices == token_id).nonzero(as_tuple=True)[0].item() + 1

            # 记录
            log_likelihoods.append(math.log(token_prob + 1e-10))
            log_ranks.append(math.log(rank + 1))
            ranks.append(rank)

            if rank == 1:
                top1_count += 1

        # PPL
        avg_loss = outputs.loss.item()
        ppl = math.exp(avg_loss)

        # LRR
        avg_ll = sum(log_likelihoods) / len(log_likelihoods)
        avg_lr = sum(log_ranks) / len(log_ranks)
        lrr = avg_ll / avg_lr if avg_lr != 0 else 0

        # Top-1 比例
        top1_ratio = top1_count / len(ranks)

        return {
            "ppl": ppl,
            "lrr": lrr,
            "token_count": len(ranks),
            "log_likelihoods": log_likelihoods,
            "log_ranks": log_ranks,
            "ranks": ranks,
            "top1_ratio": top1_ratio,
        }

    def _compute_rank_distribution(self, ranks: list[int]) -> dict:
        """计算排名分布统计"""
        if not ranks:
            return {"top1_ratio": 0, "top5_ratio": 0, "rare_ratio": 0}

        total = len(ranks)
        top1 = sum(1 for r in ranks if r == 1)
        top5 = sum(1 for r in ranks if r <= 5)
        rare = sum(1 for r in ranks if r > 100)

        return {
            "top1_ratio": round(top1 / total, 4),
            "top5_ratio": round(top5 / total, 4),
            "rare_ratio": round(rare / total, 4),
            "mean_rank": round(sum(ranks) / total, 2),
            "median_rank": sorted(ranks)[total // 2],
        }

    def _assess_risk(
        self,
        overall_ppl: float,
        lrr_score: float,
        rank_dist: dict,
        window_results: list[WindowResult],
    ) -> tuple[float, str, list[str], list[str]]:
        """综合风险评估"""
        reasons = []
        suggestions = []
        risk_factors = []

        # 1. 困惑度评估
        if overall_ppl < self.PPL_HIGH_RISK:
            risk_factors.append(0.8)
            reasons.append(f"困惑度极低 ({overall_ppl:.1f})，文本高度可预测")
        elif overall_ppl < self.PPL_MEDIUM_RISK:
            risk_factors.append(0.5)
            reasons.append(f"困惑度偏低 ({overall_ppl:.1f})")

        # 2. LRR 评估（核心指标）
        # LRR 越高（越接近 0 或正数），越像 AI
        if lrr_score > self.LRR_HIGH_RISK:
            risk_factors.append(0.9)
            reasons.append(f"LRR 异常高 ({lrr_score:.3f})，极度偏好高概率词")
            suggestions.append("尝试使用更具体的词汇，避免过于通用的表达")
        elif lrr_score > self.LRR_MEDIUM_RISK:
            risk_factors.append(0.6)
            reasons.append(f"LRR 偏高 ({lrr_score:.3f})")

        # 3. Rank 分布评估
        if rank_dist["top1_ratio"] > self.TOP1_HIGH_RISK:
            risk_factors.append(0.7)
            reasons.append(f"Top-1 选择率高达 {rank_dist['top1_ratio']:.1%}，词汇选择过于确定")
            suggestions.append("使用同义词替换，避免总是选择最'稳妥'的词")

        if rank_dist["rare_ratio"] < self.RARE_WORD_LOW:
            risk_factors.append(0.5)
            reasons.append(f"罕见词比例仅 {rank_dist['rare_ratio']:.2%}，词汇过于常见")
            suggestions.append("适当使用一些专业术语或生僻表达")

        # 4. 窗口一致性（局部 AI 嫌疑）
        low_ppl_windows = [w for w in window_results if w.ppl < 25]
        if len(low_ppl_windows) >= 2:
            risk_factors.append(0.6)
            reasons.append(f"发现 {len(low_ppl_windows)} 个高度可预测的局部片段")

        # 计算最终风险分
        if risk_factors:
            complement = 1.0
            for f in risk_factors:
                complement *= (1.0 - f)
            risk_score = round(1.0 - complement, 4)
        else:
            risk_score = 0.0

        # 确定等级
        level = _risk_level(risk_score)

        return risk_score, level, reasons, suggestions

    def _create_windows(self, text: str, window_size: int, stride: int) -> list:
        """创建滑动窗口"""
        # 按句子分割
        sentences = re.split(r'([。！？!?\.\n]+)', text)
        chunks = []
        current = ""

        for i in range(0, len(sentences), 2):
            sent = sentences[i]
            sep = sentences[i + 1] if i + 1 < len(sentences) else ""
            current += sent + sep

            if len(current) >= window_size:
                chunks.append(current)
                current = ""

        if current:
            chunks.append(current)

        # 如果分块太少，使用字符滑动
        if len(chunks) < 2:
            chunks = []
            for start in range(0, len(text), stride):
                end = min(start + window_size, len(text))
                chunks.append(text[start:end])
                if end >= len(text):
                    break

        # 生成位置信息
        result = []
        pos = 0
        for chunk in chunks:
            result.append((pos, pos + len(chunk), chunk))
            pos += len(chunk)

        return result

    def _empty_result(self) -> ProbabilityResult:
        """返回空结果"""
        return ProbabilityResult(
            overall_ppl=0.0,
            min_ppl=0.0,
            max_ppl=0.0,
            ppl_variance=0.0,
            lrr_score=0.0,
            avg_log_likelihood=0.0,
            avg_log_rank=0.0,
            rank_distribution={},
            top1_ratio=0.0,
            top5_ratio=0.0,
            rare_word_ratio=0.0,
            window_results=[],
            risk_score=0.0,
            risk_level="low",
            reasons=["文本过短，概率检测不可靠"],
            suggestions=[],
        )


class ProbabilityUnavailableError(RuntimeError):
    pass


class ProbabilityRuntimeError(RuntimeError):
    pass


# 全局单例
_detector: ProbabilityDetector | None = None


def get_detector() -> ProbabilityDetector:
    """获取全局检测器实例"""
    global _detector
    if _detector is None:
        _detector = ProbabilityDetector()
    return _detector


def analyze_probability(text: str) -> ProbabilityResult | None:
    """
    便捷函数：分析文本概率特征

    Returns:
        ProbabilityResult 包含 PPL、LRR、Rank 分布
    """
    try:
        detector = get_detector()
        return detector.analyze(text)
    except (ProbabilityUnavailableError, ProbabilityRuntimeError):
        return None


def analyze_perplexity(text: str) -> ProbabilityResult | None:
    """向后兼容旧接口名称。"""
    return analyze_probability(text)

# AI Writing Bad Cases App

这个目录是应用层仓库。

目标很明确：把仓库内 submodule `ai-writing-bad-cases/data/` 里的 JSON case 库读进来，先做程序级粗检，再把高风险段落交给后续流程处理。

## 现在包含什么

- Python 数据加载器
- 基于短语和正则的基础 matcher
- 段落级风险评分
- 统计特征分析（句长变异、连接词密度、被动语态等）
- **困惑度检测（可选）** - 基于 GPT-2 的语言模型概率特征
- 真正的 `pyseekdb` 嵌入式索引 / 查询代码
- 命令行入口

## 设计边界

- bad case 数据维护在主仓库
- 这个应用只负责读取、检测、召回
- 不负责定义最终 prompt 长什么样

## 运行方式

```bash
git clone --recurse-submodules <app-repo-url>
cd ai-writing-bad-cases-app
uv sync
UV_CACHE_DIR=.uv-cache uv run python -m ai_badcase_app.cli --input article.txt
```

如果只想看 JSON 输出：

```bash
UV_CACHE_DIR=.uv-cache uv run python -m ai_badcase_app.cli --input article.txt --format json
```

### 启用模糊匹配 (Fuzzy Matching)

对于包含错别字或微小变体的文本，可以启用模糊匹配：

```bash
# 使用 fuzzy 匹配，阈值 70（默认 75），算法使用 partial_ratio
UV_CACHE_DIR=.uv-cache uv run python -m ai_badcase_app.cli \
  --input article.txt \
  --fuzzy-threshold 70 \
  --fuzzy-algorithm partial_ratio
```

支持的算法：
- `ratio` - 标准相似度（默认）
- `partial_ratio` - 部分匹配（适合长文本中找短模式）
- `token_sort_ratio` - 忽略词序
- `token_set_ratio` - 忽略重复词

### 启用困惑度检测 (Perplexity)

基于 GPT-2 语言模型检测文本的可预测性。AI 生成文本通常困惑度更低（更可预测）：

```bash
# 1. 安装额外依赖（可选功能）
uv sync --extra perplexity

# 2. Python 代码中启用
from ai_badcase_app.analyzer import analyze_text

report = analyze_text("你的文本", enable_perplexity=True)
print(report.perplexity)
# {
#   "overall_ppl": 45.2,    # 整体困惑度，越低越像 AI
#   "min_ppl": 28.5,        # 最低困惑度窗口
#   "risk_score": 0.65,     # 综合风险分
#   "risk_level": "medium"
# }
```

困惑度检测特点：
- **轻量**：使用预训练 GPT-2，无需自己训练
- **敏感**：对流畅度高的 AI 文本敏感
- **可选**：不安装 transformers 时自动跳过，不影响其他功能

阈值参考：
- `< 35`：高度可预测，疑似 AI
- `35-50`：正常范围
- `> 50`：有更多出人意料的表达，更像人类

### 启用混合检索 (SeekDB)

现在更推荐直接用混合检索。它会把关键词约束和向量召回一起做，再用 RRF 排序：

```bash
UV_CACHE_DIR=.uv-cache uv run python -m ai_badcase_app.cli \
  --input article.txt \
  --seekdb \
  --rebuild-seekdb-index
```

如果你只想退回纯向量召回，再显式指定：

```bash
UV_CACHE_DIR=.uv-cache uv run python -m ai_badcase_app.cli \
  --input article.txt \
  --seekdb \
  --seekdb-mode vector
```

## 说明

- 项目现在使用 `uv` 管理依赖。
- `pyseekdb` 已经加入正式依赖，默认走混合检索，也支持纯向量模式。
- 经过实测，此前版本中存在的底层初始化错误已修复，现在可以正常在当前环境下运行。

# AI Writing Bad Cases App

这个目录是应用层仓库。

目标很明确：把仓库内 submodule `ai-writing-bad-cases/data/` 里的 JSON case 库读进来，先做程序级粗检，再把高风险段落交给后续流程处理。

## 现在包含什么

- Python 数据加载器
- 基于短语和正则的基础 matcher
- 段落级风险评分
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

### 启用语义召回 (SeekDB)

如果要建立 embedded SeekDB 索引并启用语义召回：

```bash
UV_CACHE_DIR=.uv-cache uv run python -m ai_badcase_app.cli \
  --input article.txt \
  --seekdb \
  --rebuild-seekdb-index \
  --seekdb-mode vector
```

## 说明

- 项目现在使用 `uv` 管理依赖。
- `pyseekdb` 已经加入正式依赖。
- 当前这台机器上，`pyseekdb` 的 embedded runtime 实际初始化会报底层错误；CLI 会把这种错误直接抛成清晰提示，不会静默失败。

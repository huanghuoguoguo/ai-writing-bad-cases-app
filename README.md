# AI Writing Bad Cases App

这个目录是仓库同级的应用层。

目标很明确：把仓库内 submodule `ai-writing-bad-cases/data/` 里的 JSON case 库读进来，先做程序级粗检，再把高风险段落交给后续流程处理。

## 现在包含什么

- Python 数据加载器
- 基于短语和正则的基础 matcher
- 段落级风险评分
- SeekDB 检索接口预留
- 命令行入口

## 设计边界

- bad case 数据维护在主仓库
- 这个应用只负责读取、检测、召回
- 不负责定义最终 prompt 长什么样

## 运行方式

```bash
git clone --recurse-submodules <app-repo-url>
cd ai-writing-bad-cases-app
PYTHONPATH=src python3 -m ai_badcase_app.cli --input article.txt
```

如果只想看 JSON 输出：

```bash
PYTHONPATH=src python3 -m ai_badcase_app.cli --input article.txt --format json
```

如果后续本机装了 `pyseekdb`，可以再继续接 `seekdb_index.py`。

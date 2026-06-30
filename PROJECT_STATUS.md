# 项目状态与待完成项

更新时间：2026-06-29

本文档记录当前代码仓库相对论文概述和 Obsidian 主文档的实现状态。当前研究题目为"基于市场用户对话情绪分析的金融产品价格涨跌预测方法"。

## 研究文档位置

Obsidian 当前主目录：

```text
E:\obsidian\knowledge\基于舆情的异常活动检测\基于市场用户对话情绪分析的金融产品价格涨跌预测方法
```

重点文件：

```text
基于市场用户对话情绪分析的金融产品价格涨跌预测方法.md
论文概述——基于市场用户对话情绪分析的金融产品价格涨跌预测方法.md
情绪分析系统.md
todo.md
拓展优化思路.md
研究协作记录.md
```

## 当前研究口径

当前主流程（第一阶段）：

```text
dataset/final.jsonl
  -> CommentBlock
  -> VADER / Sentence-BERT / Direct LLM baseline（baseline/*.ipynb）
  -> outputs/<baseline>_smoke/{metrics.json, predictions.jsonl}
```

第二阶段（暂停启动代码，先保留模块）：

```text
bull_agent / bear_agent 正反辩论
  -> debate graph with interact edges
  -> LLM Judge
  -> price-window verification
```

第一阶段三类 baseline 比较完成后，再启用第二阶段。

## 已基本落地

- 数据加载、过滤、`CommentBlock` 构建：`data/`。
- 根评论级标签口径：`CommentBlock.label` 来自价格窗口方向。
- LLM 输入安全视图：Agent / Judge prompt payload 不直接传 `p1`、未来价格或真实标签。
- 第一阶段三类 baseline：`baseline/01_vader_baseline.ipynb`、`baseline/02_sentencebert_baseline.ipynb`、`baseline/03_direct_llm_baseline.ipynb`。
- Sentence-BERT / FinBERT embedding 后端：`debate_graph/text_embeddings.py`。
- 双 Agent 辩论原型（第二阶段）：`bull_agent` / `bear_agent`，由 `agent/debate_orchestrator.py` 编排。
- 单关系辩论图原型（第二阶段）：`debate_graph/comment_graph.py`、`debate_graph/debate_graph.py`。
- Judge schema / prompt / parser / provider（第二阶段）：`judge/`、`agent/openai_compatible.py`。
- 入口文档：`baseline/README.md`、`CLAUDE.md`、`AGENTS.md`、`README.md` 都明确"baseline 运行代码在 baseline/"。

## Baseline 运行说明

```bash
"D:/anaconda/envs/sentiment/python.exe" -m jupyter notebook baseline/
```

每本 ipynb 顶部 `SMOKE = True` 控制本地烟雾量；切全量改 `SMOKE = False`、`LIMIT_BLOCKS = None` 即可。输出在 `outputs/<baseline>_smoke/`，去掉 `_smoke` 后缀写全量结果。

## 暂停使用 / 已归档

- ODE / Bi-ODE 图模型：`archive/model/`。
- 旧依赖 ODE summary 的 `main.py full / evaluate / split-experiment / train-prototype` 链路已从根目录移除，源码集中到 `archive/`。
- 反思循环作为主实验组件的方案：搁置。

这些内容可用于后续参考，第一阶段实验不依赖它们。

## 近期 TODO

1. 在三类 baseline 全量指标出来后，再决定是否启动第二阶段（debate / judge）。
2. 把各 baseline 的 `metrics.json` 汇总进 `outputs/baselines_summary.json`，方便论文写作时引用。
3. 复盘 Sentence-BERT 分类头：是否加 MLP、是否换成 FinBERT / Sentence-BERT 拼接。
4. 复盘 Direct LLM baseline 的 prompt：是否需要拆出 reasoning 字段、是否限制中文理由、是否允许弱信号用"中立兜底"。
5. 第二阶段启动前，先统一 debate / judge 的评估脚本，和 baseline 共用同一份指标字典。

## 实验指标

当前 baseline 统一使用：

```text
Accuracy
Precision
Recall
Macro-F1
Confusion Matrix
```

后续可补充成本统计、失败样例、语言分布和文本长度分层分析。

## 协作注意

1. 不要把 `p1`、未来价格、真实标签传给 LLM Agent / Judge。
2. `delta_p = (p1 - p0) / p0` 只能用于标签构造或事后验证。
3. 第一阶段不要新增 ODE 训练任务。
4. **baseline 相关运行代码只放在 `baseline/`，不要新增 `scripts/run_*_baseline.py` 这种重复入口**。
5. 第二阶段启动时，再把辩论图、Judge 流水线接到这套指标字典上。
6. README / CLAUDE / AGENTS 保持在低上下文规则；长 TODO 和研究决策放在 Obsidian 或本状态文档。

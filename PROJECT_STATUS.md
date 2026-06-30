# 项目状态与待完成项

更新时间：2026-06-29（baseline 入口整理后）

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
- 统一指标工具：`baseline/_metrics_utils.py`，供任何 baseline / pipeline / 框架调用，按 `(true_label, pred_label_int)` 喂 rows，不读 `block.p0/p1/label`。
- 单元测试：`tests/test_metrics_utils.py`，4 个用例：balanced / empty / 全对 / 非法 label 兜底。
- 数据集统计工具：`scripts/dataset_stats.py`，CLI 与 import 都支持，只读 `dataset/final.jsonl`，默认输出 top-level key 包含 `labels / by_product / by_month / by_t_window`。
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

## 数据集统计快速查阅

`scripts/dataset_stats.py`（默认仅 `dataset/final.jsonl`）：

| 指标 | 当前值 |
| --- | --- |
| posts | 438 |
| CommentBlocks | 1304 |
| label=1（看涨） | 677（51.9%） |
| label=-1（看跌） | 627（48.1%） |
| t_window | 全部 24h |
| 主币种 | BTC=675 / ETH=130，其他散落在 ~50 个代币 |
| 时间跨度 | 2026-03（644）/ 2026-04（658）/ 2026-05（2） |
| filter_issues | 0 |

`archive/final.jsonl` 不入主流程统计，需要单独处理时再 `python scripts/dataset_stats.py --input archive/final.jsonl` 切。

## 暂停使用 / 已归档（ODE 与旧链路）

### 决策

日期：2026-06-29。

当前实验阶段**暂停使用 ODE / Bi-ODE 图模型**。第一阶段先完成 VADER、Sentence-BERT 和 Direct LLM 三类 baseline，第二阶段再构造正反辩论、辩论图和 Judge 流程。

### 暂停范围（不删除，作为参考资料保留）

`model/bdg_ode/`、`model/training.py`、`model/losses.py` —— 是旧 ODE 训练链路；
`archive/tests/test_stage2_debate_judge.py`、`archive/tests/test_stage4_model.py` —— 是为了把测试跟旧代码一起保留下来。

### 已直接删除的脚本（不再保留为可运行入口）

| 脚本 | 处置 |
| --- | --- |
| `scripts/run_vader_baseline.py` | 被 `baseline/01_vader_baseline.ipynb` 取代 |
| `scripts/train_prototype.py` | 旧训练链路，依赖 ODE |
| `scripts/run_full_pipeline.py` | 旧训练链路，依赖 ODE |
| `scripts/run_split_experiment.py` | 旧训练链路，依赖 ODE |
| `scripts/evaluate_pipeline.py` | 旧训练链路，依赖 ODE |

### `main.py` 当前 active 命令

| 命令 | 用途 |
| --- | --- |
| `python main.py blocks` | 数据 → CommentBlock 自检 |
| `python main.py debate ...` | 第二阶段，正反辩论原型 |
| `python main.py graphs ...` | 第二阶段，辩论图构造 |
| `python main.py case-study ...` | 案例导出 |

`full / evaluate / split-experiment / train-prototype` 这四个依赖 ODE 与旧训练链路的命令已经从 `main.py` 移除。需要在用时去 `archive/` 找代码。

### 第一阶段 baseline 入口

baseline 运行代码全部集中在 `baseline/`：

```text
baseline/
├── 01_vader_baseline.ipynb         # VADER 词典打分
├── 02_sentencebert_baseline.ipynb  # Sentence-BERT + LogReg
├── 03_direct_llm_baseline.ipynb    # SiliconFlow LLM 直调
├── README.md                       # 三本 ipynb 的入口说明
├── vaderSentiment/                 # VADER 包
└── _metrics_utils.py               # 统一指标工具
```

不再在 `scripts/` 下开 baseline 入口。

### 反思循环

反思循环作为主实验组件的方案：搁置。第二阶段启动时再讨论是否引入。

### 重新启用条件

只有在基础实验显示"正反辩论 + 辩论图"相对 Direct LLM 存在明确增益后，再讨论是否重新引入正反观点动态演化模块。重新引入前需要重新说明它解决的具体问题、输入输出、实验对比和消融方式。

## 近期 TODO

1. 在三类 baseline 全量指标出来后，再决定是否启动第二阶段（debate / judge）。
2. 把各 baseline 的 `metrics.json` 汇总进 `outputs/baselines_summary.json`，方便论文写作时引用。
3. 复盘 Sentence-BERT 分类头：是否加 MLP、是否换成 FinBERT / Sentence-BERT 拼接。
4. 复盘 Direct LLM baseline 的 prompt：是否需要拆出 reasoning 字段、是否限制中文理由、是否允许弱信号用"中立兜底"。
5. 第二阶段启动前，先统一 debate / judge 的评估脚本，和 baseline 共用同一份指标字典。
6. 按 plan 串行推进的"讽刺 / 反讽 / emoji 兜底"诊断（Task 2/3）：先做 `_sarcasm_rules` 词表与单测，再让三本 ipynb 输出 `sarcasm_subset_metrics.json`，**不要**在 prompt 之外改算法。
7. FinBERT baseline（Task 4）放服务器跑，本机不下载权重。
8. 第二阶段 debate+judge（Task 7）：先在服务器上跑通 4 round bull/bear → 辩论图 → Judge → 价格验证，使用 `baseline/_metrics_utils.compute_metrics` 统一打分。

## 实验指标

当前 baseline 统一使用（字段定义见 `baseline/_metrics_utils.compute_metrics`）：

```text
total
accuracy
macro_precision
macro_recall
macro_f1
bullish.{precision, recall, f1, support}
bearish.{precision, recall, f1, support}
confusion_matrix
```

后续可补充成本统计、失败样例、语言分布和文本长度分层分析。

## 协作注意

1. 不要把 `p1`、未来价格、真实标签传给 LLM Agent / Judge。
2. `delta_p = (p1 - p0) / p0` 只能用于标签构造或事后验证。
3. 第一阶段不要新增 ODE 训练任务。
4. **baseline 相关运行代码只放在 `baseline/`，不要新增 `scripts/run_*_baseline.py` 这种重复入口**。
5. 数据集统计 / 评估类脚本默认只读 `dataset/final.jsonl`，`archive/final.jsonl` 仅作为历史归档，不纳入主流程统计。
6. 不在用户没要求的前提下写临时的"烟测脚本"。
7. 第二阶段启动时，再把辩论图、Judge 流水线接到这套指标字典上。
8. README / CLAUDE / AGENTS 保持在低上下文规则；长 TODO 和研究决策放在 Obsidian 或本状态文档。

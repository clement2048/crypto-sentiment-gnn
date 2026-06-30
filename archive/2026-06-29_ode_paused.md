# ODE 相关模块暂停使用说明

日期：2026-06-29（更新于同日：同步 baseline 入口迁移）

## 决策

当前实验阶段暂停使用 ODE / Bi-ODE 图模型。第一阶段先完成 VADER、Sentence-BERT 和 Direct LLM 三类 baseline，第二阶段再构造正反辩论、辩论图和 Judge 流程。

## 暂停范围

以下内容只作为旧方案参考，已移入 `archive/`：

```text
model/bdg_ode/
model/training.py
model/losses.py
archive/tests/test_stage2_debate_judge.py   # 引用了已删脚本的 compute_metrics
archive/tests/test_stage4_model.py          # 引用了已删的 train/full/split
```

同时，以下脚本已直接删除（不再保留为可运行入口，它们的逻辑要么入 archive 模型代码，要么不再适用）：

```text
scripts/run_vader_baseline.py        # 被 baseline/01_vader_baseline.ipynb 取代
scripts/train_prototype.py
scripts/run_full_pipeline.py
scripts/run_split_experiment.py
scripts/evaluate_pipeline.py
```

`main.py` 也只保留 active 命令：`blocks / debate / graphs / case-study`。

## 第一阶段 baseline 入口

完成这次整理后，所有 baseline 运行代码集中在：

```text
baseline/
├── 01_vader_baseline.ipynb
├── 02_sentencebert_baseline.ipynb
├── 03_direct_llm_baseline.ipynb
├── README.md
└── vaderSentiment/
```

不在 `scripts/` 下再开 baseline 入口。

## 当前 main.py 的命令

只剩 active 命令：

```bash
python main.py blocks           # 数据 -> CommentBlock 自检
python main.py debate ...       # 第二阶段，正反辩论原型
python main.py graphs ...       # 第二阶段，辩论图构造
python main.py case-study ...   # 案例导出
```

`full / evaluate / split-experiment / train-prototype` 这四个依赖 ODE 与旧训练链路的命令已经从 `main.py` 移除，需要用时去 `archive/scripts/`。

## 当前优先工作

1. 在三本 baseline ipynb 上跑完 VADER / Sentence-BERT / Direct LLM，记录样本数、neutral 数、Accuracy、Macro-F1。
2. 把 `outputs/<baseline>_smoke/{metrics.json, predictions.jsonl}` 合并成 `outputs/baselines_summary.json`，给论文写作用。
3. 第二阶段启动时，把辩论图、Judge 流水线接到同一套指标字典。
4. 不再追加 `scripts/run_*_baseline.py` 这种重复入口。

## 重新启用条件

只有在基础实验显示"正反辩论 + 辩论图"相对 Direct LLM 存在明确增益后，再讨论是否重新引入正反观点动态演化模块。重新引入前需要重新说明它解决的具体问题、输入输出、实验对比和消融方式。

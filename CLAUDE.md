# CLAUDE.md

This file provides guidance to Claude when working in this repository.

## 回答风格

- 始终用中文回答。
- 直接、简洁、事实优先。
- 不确定时明确说明"不确定"或"需要更多信息"，不要猜测。
- 区分事实、推断和建议。
- 涉及论文定义、实验口径、API 用法或命令参数时，优先查本仓库代码和 Obsidian 主文档。

## 当前项目

当前研究题目：

```text
基于市场用户对话情绪分析的金融产品价格涨跌预测方法
```

当前实验重点：

```text
dataset/final.jsonl
  -> CommentBlock
  -> VADER / Sentence-BERT / Direct LLM baseline（baseline/）
  -> 第二阶段：bull_agent / bear_agent 正反辩论、辩论图、Judge、价格窗口验证
```

第一阶段先做 VADER、Sentence-BERT、Direct LLM 三类 baseline。第二阶段再构造正反辩论、辩论图和 Judge 流程。

ODE / Bi-ODE 相关模块暂停使用，集中在 `archive/` 下作为旧方案归档参考。不要把 ODE 训练或 ODE summary 写入当前主流程。

## 外部研究文档

遇到论文语义、实验目标、TODO 优先级不确定时，查 Obsidian：

```text
E:\obsidian\knowledge\基于舆情的异常活动检测\基于市场用户对话情绪分析的金融产品价格涨跌预测方法
```

重点文件：

```text
基于市场用户对话情绪分析的金融产品价格涨跌预测方法.md
论文概述——基于市场用户对话情绪分析的金融产品价格涨跌预测方法.md
情绪分析系统.md
todo.md
PROJECT_STATUS.md
```

## 硬性约束

1. 主监督标签是根评论级 `CommentBlock.label`。
2. `label = 1` 表示后续价格上涨，`label = -1` 表示后续价格下跌。
3. `p1`、未来价格、真实标签不能传给 LLM Agent / Judge。
4. `delta_p = (p1 - p0) / p0` 只能用于标签构造或事后验证。
5. 第一阶段实验不要新增 ODE 训练任务。
6. 辩论图优先使用统一 `interact` 边；支持或反驳关系由节点立场组合解释。
7. LLM Agent / LLM Judge 不可微，不能把 Judge 输出直接接入 `loss.backward()`。

## Baseline 运行入口

第一阶段所有 baseline 的运行代码都在 `baseline/`，**不要到 `scripts/` 或其他地方找**：

```text
baseline/
├── 01_vader_baseline.ipynb            # VADER 词典打分
├── 02_sentencebert_baseline.ipynb     # Sentence-BERT + LogReg
├── 03_direct_llm_baseline.ipynb       # SiliconFlow LLM 直调
├── README.md                          # 三本 ipynb 的入口说明
└── vaderSentiment/                    # VADER 包
```

启动 jupyter：

```bash
"D:/anaconda/envs/sentiment/python.exe" -m jupyter notebook baseline/
```

每本顶部都有一行 `SMOKE = True` / `LIMIT_BLOCKS = 10` 这种开关，本机先验证链路，再切全量。输出默认写到 `outputs/<baseline>_smoke/`。

服务器上无交互跑（任选其一即可）：

```bash
"D:/anaconda/envs/sentiment/python.exe" -m jupyter nbconvert --to notebook --execute \
    --ExecutePreprocessor.timeout=600 \
    baseline/01_vader_baseline.ipynb \
    --output baseline/01_vader_baseline_full.ipynb
```

## Python 环境

推荐使用：

```bash
"D:/anaconda/envs/sentiment/python.exe" -m unittest discover -s tests
```

系统默认 `python` 可能没有 `torch`，不要默认假设可用。

Sentence-BERT / FinBERT 嵌入依赖：

```bash
pip install -r requirements-embedding.txt
```

## 关键模块（仅第一阶段 baseline 用得到的）

| 路径 | 责任 |
| --- | --- |
| `baseline/*.ipynb` | 第一阶段三类 baseline 的运行入口 |
| `data/` | JSONL 加载、过滤、CommentBlock 构建 |
| `debate_graph/text_embeddings.py` | Sentence-BERT / FinBERT 文本向量后端 |
| `config.py` | 配置常量（默认输入路径、LLM 模型、阈值等） |
| `archive/` | 旧方案与暂停使用的模块（ODE、反思循环、旧训练链路） |

`agent/`、`judge/`、`verification/`、`debate_graph/comment_graph.py` 等是第二阶段需要的代码，目前属于搁置状态；不要在 baseline notebook 里依赖它们。

## 配置

- 默认数据：`dataset/final.jsonl`
- 数据统计 / 评估类脚本默认只读 `dataset/final.jsonl`（如 `scripts/dataset_stats.py`）。`archive/final.jsonl` 仅作为历史归档，不纳入主流程统计。
- API key：项目根目录 `.env` 或环境变量
- LLM cache：`outputs/llm_cache/`
- 配置常量优先放在 `config.py`，不要在调用方硬编码超参数。
- 不在用户没要求的前提下写临时的"烟测脚本"（smoke tests）；用户明确说要烟测的时候再写。

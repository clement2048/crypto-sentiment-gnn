# Market Dialogue Sentiment Experiments

本仓库用于实现"基于市场用户对话情绪分析的金融产品价格涨跌预测方法"的实验代码。当前目标是验证三类基础方法效果，再逐步构造正反辩论流程和辩论图结构。

## 当前实验路线

第一阶段比较三类 baseline：

1. **VADER**：词典与规则类情绪分析 baseline。
2. **Sentence-BERT**：基于文本向量表示的语义 baseline。
3. **Direct LLM**：将评论对话块直接输入 LLM，输出价格涨跌方向。

第二阶段再构造本文方法：

1. 评论对话块构建。
2. 看涨 / 看跌 Agent 正反辩论。
3. 辩论图结构化表示。
4. LLM Judge 基于辩论图输出涨跌方向。
5. 使用价格窗口标签进行离线评估。

正反观点动态演化与 ODE 相关代码已暂停使用，统一放在 `archive/`，后续只作为归档参考。

## Baseline 运行入口

所有 baseline 运行代码都在 [`baseline/`](baseline/) 下，**不要到 `scripts/` 或其他地方找**：

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

每本顶部都有一行 `SMOKE = True` / `LIMIT_BLOCKS = 10` 这种开关。本机先验证链路、把指标打印出来；确认后改 `SMOKE = False`、`LIMIT_BLOCKS = None` 切全量（参考各 ipynb 最后一个 cell 的 `nbconvert --execute` 命令）。

输出默认写到 `outputs/<baseline>_smoke/`，含 `metrics.json` + `predictions.jsonl`。

## 项目结构

```text
baseline/         第一阶段 baseline 运行入口与 VADER 包（运行入口只看这里）
data/             JSONL 加载、过滤、CommentBlock 构建（被 baseline/ 引用）
debate_graph/     评论节点、论点节点、interact 边、Sentence-BERT/FinBERT 文本向量
agent/            第二阶段正反辩论 Agent、LLM provider、输出解析
judge/            第二阶段 Judge schema、prompt、parser、provider
verification/     价格窗口方向验证
scripts/          构建工具与第二阶段辩论图命令入口（不含 baseline 脚本）
config.py         配置常量
archive/          旧 ODE 模型与暂停使用的旧链路
tests/            单元测试
```

## 数据流

```text
dataset/final.jsonl
  -> data.build_comment_blocks
  -> baseline/*.ipynb 跑出 VADER / Sentence-BERT / Direct LLM 三类预测
  -> outputs/<baseline>_smoke/{metrics.json, predictions.jsonl}
```

模型和 Agent 输入阶段只能使用评论发布时可见的信息。`p1`、未来价格和真实 `label` 只用于标签构造、评估和误差分析。

## 已落地的代码

| 功能 | 当前状态 | 路径 |
| --- | --- | --- |
| 数据加载与 `CommentBlock` 构建 | 已有 | `data/` |
| VADER baseline | 已有 | `baseline/01_vader_baseline.ipynb` |
| Sentence-BERT baseline | 已有 | `baseline/02_sentencebert_baseline.ipynb` |
| Sentence-BERT / FinBERT embedding 后端 | 已有 | `debate_graph/text_embeddings.py` |
| Direct LLM baseline | 已有 | `baseline/03_direct_llm_baseline.ipynb` |
| 正反辩论 Agent | 已有原型（第二阶段） | `agent/` |
| 辩论图构建 | 已有原型（第二阶段） | `debate_graph/` |
| LLM Judge | 已有原型（第二阶段） | `judge/` |
| ODE 图模型 | 暂停使用 | `archive/` |

## 推荐运行顺序

### 1. 先确认 baseline 跑通

```bash
"D:/anaconda/envs/sentiment/python.exe" -m jupyter notebook baseline/
```

依次 Run All：
- `01_vader_baseline.ipynb`：纯本地、几乎秒级。
- `02_sentencebert_baseline.ipynb`：首次会下载 Sentence-BERT 权重（缓存在 `outputs/hf_cache/`）。
- `03_direct_llm_baseline.ipynb`：需要 `.env` 里 `SILICONFLOW_API_KEY`；响应按 payload SHA256 缓存在 `outputs/llm_cache/direct_llm/`。

### 2. 切全量

每本顶部 `SMOKE = False`、`LIMIT_BLOCKS = None`，改 `OUT_DIR` 的 `_smoke` 后缀，整本 Run All 或用 `nbconvert --execute`。

### 3. 第二阶段（后续启用）

```bash
python main.py debate --limit-blocks 1 --rounds 4 --mode siliconflow
python main.py graphs --limit-blocks 1 --rounds 4 --mode siliconflow
```

后续需要把辩论图输出整理成稳定 JSON，供 Judge 和可视化脚本复用。

## 暂停使用 / 已归档

以下命令属于旧原型链路，已暂停使用，代码集中在 `archive/`，当前实验不优先调用：

```bash
python main.py full
python main.py evaluate
python main.py split-experiment
python main.py train-prototype
```

## 环境

推荐使用本机 anaconda 的 `sentiment` 环境：

```bash
"D:/anaconda/envs/sentiment/python.exe" -m unittest discover -s tests
```

如果需要 Sentence-BERT / FinBERT 文本向量：

```bash
pip install -r requirements-embedding.txt
```

API key 放在项目根目录 `.env` 或当前 shell 环境变量中：

```bash
SILICONFLOW_API_KEY=
SILICONFLOW_MODEL=
```

## 当前约束

1. 主监督标签是根评论级 `CommentBlock.label`，`1` 表示后续价格上涨，`-1` 表示后续价格下跌。
2. `p1`、未来价格和真实标签不能传给 LLM Agent 或 LLM Judge。
3. 价格方向只用于标签构造、离线评估和误差分析。
4. 第一阶段实验优先比较 VADER、Sentence-BERT 和 Direct LLM，结果都从 `baseline/` 入口获取。
5. 第二阶段再比较正反辩论、辩论图和 Judge 的增益。
6. ODE 相关模块暂停使用，统一放在 `archive/`。

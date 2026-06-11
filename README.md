# 情绪分析系统 v2 原型

这是当前主工作区的代码说明。旧版遗留代码已经归档到 `archive/legacy_v1/`，当前主要代码直接放在根目录：

```text
agent/      多智能体辩论结构、mock Agent、DeepSeek/百炼/MiniMax LLM 接口
data/       JSONL 加载、过滤、CommentBlock 构建
dataset/    默认数据集，当前读取 final.jsonl
debate_graph/ 评论图、辩论图、异构图、图张量化
judge/      法官输出 schema、parser、模型感知 mock Judge
model/      BDG-ODE、GCN/GAT/GraphTransformer 等模型实验入口
profiles/   时间安全用户画像
scripts/    分阶段脚本
tests/      单元测试
main.py     统一命令行入口
```

注意：这里用 `profiles/` 而不是 `profile/`，是为了避免遮蔽 Python 标准库 `profile`，否则 PyTorch 会在导入 `cProfile` 时出错。

## 当前完整流程

```text
JSONL 原始数据
  -> data: 构建 CommentBlock
  -> profiles: 构建 t < t0 的用户画像
  -> agent: 生成多空辩论（默认 mock，可切换 DeepSeek / 百炼 / MiniMax）
  -> debate_graph: 融合评论图 + 辩论图
  -> model: 图张量 -> 图模型方法 -> calibrator
  -> judge: 法官接收辩论结构 + ODE/模型摘要
```

当前默认仍走离线 mock，方便稳定测试；辩论 Agent 与 Judge 已支持 DeepSeek Anthropic-compatible API、阿里云百炼 OpenAI-compatible API、MiniMax Anthropic-compatible API。

## Python 环境

项目运行需要 `torch`。**推荐使用 anaconda 的 sentiment 环境**：

```bash
"D:/anaconda/envs/sentiment/python.exe" main.py debate --limit-blocks 1 --rounds 1
```

- 该环境已装 `torch 2.11.0+cpu`。
- **未装 pytest**：跑测试用 `python -m unittest discover -s tests`（等价覆盖 4 个 stage 测试文件）。
- 系统默认 `python` 路径（如 `C:\Python314\python.exe`）通常没装 torch，直接调 `python main.py ...` 会因 `debate_graph/graph_batch.py` import torch 失败。

## 统一入口

优先使用根目录 `main.py`：

```bash
python main.py blocks
python main.py debate --limit-blocks 3 --rounds 1
python main.py debate --limit-blocks 1 --rounds 1 --mode deepseek
python main.py debate --limit-blocks 1 --rounds 1 --mode bailian
python main.py debate --limit-blocks 1 --rounds 1 --mode minimax
python main.py graphs --limit-blocks 3 --rounds 1
python main.py train-prototype --limit-blocks 3 --rounds 1 --epochs 3
python main.py full --limit-blocks 3 --rounds 1 --train-epochs 1
python main.py full --limit-blocks 1 --rounds 1 --debate-mode deepseek
python main.py full --limit-blocks 1 --rounds 1 --debate-mode bailian --judge-mode bailian
python main.py full --limit-blocks 1 --rounds 1 --debate-mode minimax --judge-mode minimax
python main.py evaluate --rounds 1 --debate-mode mock
python main.py evaluate --rounds 1 --debate-mode deepseek --metrics-json outputs/eval_metrics.json --output-jsonl outputs/eval_records.jsonl
python main.py split-experiment --train-count 9 --val-count 3 --test-count 3 --rounds 1 --epochs 5 --debate-mode mock --seed 42 --output-json outputs/split_9_3_3_mock.json
python main.py split-experiment --train-count 9 --val-count 3 --test-count 3 --rounds 1 --epochs 5 --debate-mode deepseek --judge-mode deepseek --seed 42 --output-json outputs/split_9_3_3_deepseek.json
python main.py split-experiment --train-count 9 --val-count 3 --test-count 3 --rounds 1 --epochs 5 --debate-mode bailian --judge-mode bailian --seed 42 --output-json outputs/split_9_3_3_bailian.json
python main.py split-experiment --train-count 9 --val-count 3 --test-count 3 --rounds 1 --epochs 5 --debate-mode minimax --judge-mode minimax --seed 42 --output-json outputs/split_9_3_3_minimax.json
python main.py case-study --post-id 305698686327490 --rounds 1 --debate-mode deepseek --seed 42 --output-json outputs/case_305698686327490_deepseek.json --output-md outputs/case_305698686327490_deepseek.md
python main.py case-study --post-id 305698686327490 --rounds 1 --debate-mode minimax --seed 42 --output-json outputs/case_305698686327490_minimax.json --output-md outputs/case_305698686327490_minimax.md
python -m scripts.export_case_csv --input-json outputs/case_305698686327490_deepseek.json --output-dir outputs/case_305698686327490_deepseek_csv
python -m scripts.export_metrics_csv --input-json outputs/split_9_3_3_deepseek.json --output-csv outputs/split_9_3_3_deepseek_metrics.csv
```

默认输入是 `dataset/final.jsonl`。如果要临时换数据，可以继续传 `--input`。

子命令含义：

| 命令 | 作用 |
| --- | --- |
| `blocks` | 读取 JSONL，构建评论块，打印过滤统计和时间切分 |
| `debate` | 对每个评论块生成多空辩论，不调用法官；默认 mock，可用 `--mode deepseek` / `--mode bailian` / `--mode minimax` |
| `graphs` | 将评论图和辩论图融合成多关系异构图 |
| `train-prototype` | 用少量样本 smoke-train 当前最小模型 |
| `full` | 执行完整原型：辩论 -> 图 -> 模型摘要 -> 法官 |
| `evaluate` | 默认对全部 CommentBlock 运行完整链路，并输出 accuracy、precision、recall、F1、coverage、混淆矩阵 |
| `split-experiment` | 按时间顺序运行固定数量 train/val/test 小实验，例如 9:3:3 |
| `case-study` | 选择一个评论较多的帖子，导出可阅读的辩论过程 Markdown 和完整 JSON |

`full`、`evaluate`、`split-experiment` 的输出 JSON 中，每条样本都有：

```text
record["judge"]["verdict"]
record["judge"]["confidence"]
record["judge"]["report"]
record["judge"]["score_vector"]
```

其中 `judge.report` 是法官判决文本。`split-experiment` 命令行也会直接打印 `judge_report`，方便不打开 JSON 时快速检查。

## 核心数据对象

### `data/schema.py`

| 对象 | 作用 |
| --- | --- |
| `RawComment` | 单条原始评论，含 replies、时间、标签字段 |
| `PostRecord` | 一行 JSONL 解析后的帖子 |
| `CommentBlock` | 一个根评论 + replies，当前训练/推理的样本粒度 |
| `FilterIssue` | 被过滤样本的原因记录 |

常用函数：

| 函数 | 作用 |
| --- | --- |
| `load_posts(path_or_glob)` | 从文件、目录或 glob 读取 JSONL |
| `build_comment_blocks(posts)` | 将帖子拆成评论块 |
| `temporal_split_blocks(blocks)` | 按 `t0` 时间顺序切分 train/val/test |

## 用户画像

### `profiles/`

核心原则：只允许使用 `comment_time < block.t0` 的历史评论。

| 文件 | 作用 |
| --- | --- |
| `profile_store.py` | 按作者索引历史评论，并为某个 block 生成画像 |
| `user_profile.py` | 定义 `UserProfile` 和画像计算逻辑 |
| `feature_defs.py` | 固定画像字段名 |

常用入口：

```python
profile_store = ProfileStore.from_blocks(blocks)
profiles = profile_store.get_profiles_for_block(block)
```

## 辩论 Agent

### `agent/`

默认使用离线 mock，用于稳定跑通结构；需要真实 LLM 时可切换到 DeepSeek 或阿里云百炼。

| 文件 | 作用 |
| --- | --- |
| `schema.py` | `Evidence`、`Argument`、`DebateTranscript` |
| `mock_client.py` | 不依赖网络的 mock 辩论生成器 |
| `anthropic_compatible.py` | Anthropic Messages 兼容协议共享层（DeepSeek + MiniMax），含 DebateClient 与 JudgeClient 基类 + 两个 provider 薄包装 |
| `openai_compatible.py` | OpenAI Chat Completions 兼容协议共享层（Bailian），含 DebateClient 与 JudgeClient 基类 + Bailian 薄包装 |
| `client_factory.py` | 根据 `mock/deepseek/bailian/minimax` 创建辩论 client |
| `debate_orchestrator.py` | 组织多轮 bull/bear agent 发言 |
| `output_parser.py` | 解析结构化 Agent JSON |
| `llm_client.py` | 真实/模拟 LLM provider 共用协议 |

论文 v4 对齐后的默认辩论角色：

| 阵营 | 角色名 | 论文角色 |
| --- | --- | --- |
| bull | `technical_analysis_agent` | 技术面分析师 |
| bull | `fundamental_analysis_agent` | 基本面分析师 |
| bull | `sentiment_contagion_agent` | 情绪传染分析师 |
| bull | `reflection_agent` | 正方反思人员 |
| bear | `risk_analysis_agent` | 风险分析师 |
| bear | `onchain_skeptic_agent` | 链上数据质疑者 |
| bear | `sentiment_reversal_agent` | 情绪反转分析师 |
| bear | `reflection_agent` | 反方反思人员 |

角色专属提示词写在 `agent/prompts.py`。当前版本没有实时 K 线、链上数据或新闻检索工具，因此 prompt 明确要求 agent 只能使用输入中的新闻正文、评论树、时间安全用户画像和既有辩论论点，不能假装查到了外部指标。

常用入口：

```python
transcript = DebateOrchestrator().run(block, profiles, rounds=2)
```

### DeepSeek API Key

不要把 API key 写进代码。运行真实 LLM 辩论前，在 PowerShell 中设置环境变量：

```powershell
$env:DEEPSEEK_API_KEY="你的 DeepSeek API key"
```

也兼容 Anthropic 生态常用变量名：

```powershell
$env:ANTHROPIC_API_KEY="你的 DeepSeek API key"
```

也可以写到项目根目录 `.env`（已加入 `.gitignore`，不会被上传），模板见 `.env.example`。`_load_env_file()` 会自动加载到 `os.environ`。

默认配置在 `config.py`：

```python
DEEPSEEK_ANTHROPIC_BASE_URL = "https://api.deepseek.com/anthropic"
DEEPSEEK_MODEL = "deepseek-v4-pro"
```

### 阿里云百炼 API Key

百炼接口使用 OpenAI 兼容模式，当前默认模型以 `config.py` 的 `BAILIAN_MODEL` 为准。

```powershell
$env:DASHSCOPE_API_KEY="你的阿里云百炼 API key"
```

或写到 `.env`：

```bash
BAILIAN_API_KEY=sk-你的key
DASHSCOPE_API_KEY=sk-你的key   # 备用别名
```

默认配置在 `config.py`：

```python
BAILIAN_OPENAI_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
BAILIAN_MODEL = "qwen-flash"
BAILIAN_ENABLE_THINKING = False
```

### MiniMax API Key

MiniMax 接口使用 Anthropic Messages 兼容协议（与 DeepSeek 同一族），默认模型 `MiniMax-M3`。运行前：

```powershell
$env:ANTHROPIC_API_KEY="你的 MiniMax API key"   # 推荐：与 SDK 共用
$env:MINIMAX_API_KEY="你的 MiniMax API key"     # 备用别名
```

或写到 `.env`：

```bash
ANTHROPIC_API_KEY=sk-你的minimax-key
MINIMAX_API_KEY=sk-你的minimax-key   # 备用别名
```

`_load_api_key` 会按 `ANTHROPIC_API_KEY → MINIMAX_API_KEY` 顺序读，填任一个即可。官方文档：[platform.minimax.io/docs/token-plan/quickstart](https://platform.minimax.io/docs/token-plan/quickstart)。

默认配置在 `config.py`：

```python
MINIMAX_ANTHROPIC_BASE_URL = "https://api.minimax.io/anthropic"
MINIMAX_MODEL = "MiniMax-M3"
```

## 图构建

### `debate_graph/`

| 文件 | 作用 |
| --- | --- |
| `comment_graph.py` | 将评论树转成 reply 图 |
| `debate_graph.py` | 将 argument 转成 cite/support/attack/respond/propose 图；时间信息写入节点/边属性，不再使用 precede 边 |
| `hetero_graph.py` | 融合评论图和辩论图 |
| `diffusion_ops.py` | 按关系构建归一化邻接 |
| `graph_batch.py` | 将异构图转成 torch 张量 |
| `schema.py` | `GraphNode`、`GraphEdge`、`HeteroGraph` |

常用入口：

```python
graph = build_hetero_graph(block, transcript)
graph_tensor = graph_to_tensor(graph, label=block.label)
```

## 模型

### `model/`

当前是最小可训练原型，演替函数已经按双视角 BDG-ODE 思路接入，
并默认使用 `torchdiffeq.odeint` 做 ODE 积分。

| 文件 | 作用 |
| --- | --- |
| `bdg_ode/` | 当前默认 BDG-ODE 方法 |
| `bdg_ode/dual_encoder.py` | 将节点特征编码成 bull/bear 初始状态 |
| `bdg_ode/dynamics.py` | 双视角 bull/bear 图 ODE 动态 |
| `bdg_ode/ode_solver.py` | `torchdiffeq` ODE 求解器，保留手写 Euler fallback |
| `bdg_ode/readout.py` | 节点状态池化成图级向量 |
| `bdg_ode/calibrator.py` | 输出看涨概率 |
| `gcn/` | 预留给 GCN 实验 |
| `gat/` | 预留给 GAT 实验 |
| `graph_transformer/` | 预留给 Graph Transformer 实验 |
| `model_summary.py` | 输出给法官的 ODE/模型摘要 |
| `bdg_ode/pipeline.py` | 端到端 `GraphSentimentModel` |
| `losses.py` | 当前只有分类损失 |

常用入口：

```python
model = GraphSentimentModel(input_dim=NODE_FEATURE_DIM)
prob = model(graph_tensor)
summary = model.summarize(graph_tensor)
```

## 法官

### `judge/`

文档中的正确位置是：法官接收「辩论结构 + ODE 演化摘要」后输出判决。

| 文件 | 作用 |
| --- | --- |
| `judge_schema.py` | `JudgeScoreVector`、`JudgeOutput` |
| `judge_parser.py` | 解析结构化 Judge JSON |
| `consistency.py` | 检查 verdict、confidence、score vector 是否一致 |
| `model_aware_judge.py` | 当前离线 mock 法官，接收 debate + model summary |
| `judge_agent.py` | 早期 mock judge，保留作 schema smoke，不作为最终流程入口 |

常用入口：

```python
judge_output = ModelAwareMockJudge().judge(transcript, model_summary)
```

## 脚本

### `scripts/`

这些脚本仍可单独运行，但推荐优先用 `main.py`。

| 脚本 | 对应 `main.py` 命令 |
| --- | --- |
| `build_blocks.py` | `python main.py blocks` |
| `run_debate.py` | `python main.py debate` |
| `build_graphs.py` | `python main.py graphs` |
| `train_prototype.py` | `python main.py train-prototype` |
| `run_full_pipeline.py` | `python main.py full` |
| `evaluate_pipeline.py` | `python main.py evaluate` |
| `run_split_experiment.py` | `python main.py split-experiment` |

## 测试

```bash
python -m unittest discover -s tests
python -m pytest tests -p no:cacheprovider
```

`-p no:cacheprovider` 是为了避免当前 Windows 沙箱下 pytest 写 cache 时出现权限问题。

## 当前未完成

主要还缺这些：

- 正式训练系统：时间切分、batch、checkpoint、early stopping
- 完整损失函数：`L_mse / L0 / L1 / L2 / L3 / L_judge_cons`
- 评估模块：Accuracy、Macro-F1、ECE、Brier、Market Match Rate
- 市场行为验证模块
- 消融实验
- calibrator 接入法官评分向量 `J`

> 真实 LLM Agent / Judge 已支持 DeepSeek / 阿里云百炼 / MiniMax 三家，详见上文「运行 LLM 真实辩论」段。

详细 TODO 已整理在 Obsidian：

```text
情绪分析系统_v2_实现进度与TODO.md
```

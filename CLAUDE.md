# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 项目概述

加密货币舆情二分类原型 v2：JSONL 原始数据 → 多智能体 bull/bear 辩论 → 异构图 → BDG-ODE 图模型 → 模型感知 Judge 输出 BULLISH/BEARISH/NEUTRAL。

论文对齐角色（见 `agent/prompts.py`）：正方 4 角色（技术面/基本面/情绪传染/反思），反方 4 角色（风险/链上质疑/情绪反转/反思），加独立法官。旧版 v1 已归档到 `archive/legacy_v1/`，**当前主工作区在根目录**。

## 关联的论文与设计文档（Obsidian）

整个工作对应一份 v4 论文 + 一份 v2 框架设计 + 一份实时更新的 TODO 追踪表，都放在 Obsidian 仓库里：

- **论文**：`E:\obsidian\knowledge\基于舆情的异常活动检测\市场波动对股民情绪影响\市场波动对股民情绪的影响_v4.md` —— 角色名、阶段名、损失项 `L_cls / L_mse / L0~L3 / L_judge_cons / L_activity`、评估指标（Accuracy / Macro-F1 / ECE / Brier / MSED / Judge Consistency / Market Match / Activity Stratified）都源自这份论文，**改论文任何定义前先到这里看原文**。
- **框架设计**：`E:\obsidian\knowledge\基于舆情的异常活动检测\市场波动对股民情绪影响\框架设计\情绪分析系统_v2.md` —— v2 的修订目标（标签口径、时间边界、市场行为验证、法官训练边界）和六阶段流水线都写在这；当前代码骨架直接对应这份文档。
- **实时 TODO 追踪**：`E:\obsidian\knowledge\基于舆情的异常活动检测\市场波动对股民情绪影响\框架设计\情绪分析系统_v2_实现进度与TODO.md` —— 哪块已实现、哪块待做、待做的优先级都在这张表里，**改代码前先看这里确认是否还在 TODO 列表**。
- **图神经网络/图学习文献**：`E:\obsidian\knowledge\基于舆情的异常活动检测\市场波动对股民情绪影响\相关文献\图神经网络的论文` —— GCN / GAT / Graph Transformer / 异构图 / 图时序建模等模型实现和实验设计，优先参考这里的论文笔记；不要只凭通用印象随意写 GNN 结构。
- **整个文件夹** `E:\obsidian\knowledge\基于舆情的异常活动检测\市场波动对股民情绪影响\` 的所有内容（Agent 提示词设计、文献调研、待解决问题、优化思路、研究协作记录、归档等）都和这个项目有关，修改时遇到不确定的语义应该先去那里查。

**README 已经指向 TODO 文件，但只给了一行** —— 实际 TODO 表里 `2.2 正式训练系统`、`2.3 损失函数`、`2.4 评估模块`、`2.5 市场行为验证`、`2.6 消融实验`、`2.7 calibrator 接入法官 J`、`2.8 BDG-ODE 轨迹监督` 都是高优先级空缺，看到 `model/losses.py` 只有 `classification_loss`、`scripts/evaluate_pipeline.py` 精度实现简陋、不要误以为"已完成"。

## 论文 v2 的四个硬性约束（来自设计文档 §0）

每次写新代码都要回头看这四条，否则就是 look-ahead / 数据泄漏 / 监督信号错位：

1. **标签口径**：主监督是 `CommentBlock` 根评论级 `label`（1/看涨，-1/看跌），不是帖子级投票。
2. **时间边界**：用户画像、Agent 输入、外部市场特征都必须 `t < t0`。`profiles/profile_store.py` 已经用 `bisect_left` 强制，新代码不要绕过 `ProfileStore`。
3. **市场行为验证**：价格方向 `delta_p = (p1 - p0) / p0` 只能用于构造/验证标签，**不能**作为模型直接监督；交易量 `delta_v` 只作为活动强度信号，**不能**决定方向。
4. **法官训练边界**：LLM Agent / LLM Judge 都不可微，**不要试图把 judge 的输出塞进 `loss.backward()`**；可训练部分只能是编码器、ODE 模块、判决校准器。

## 目录约定（关键）

- `profiles/`（复数）不是 `profile/`，刻意避免遮蔽 Python 标准库 `profile`，否则 `import cProfile` 会失败。
- 默认数据来源是 `dataset/final.jsonl`（220 行），可用 `--input` 覆盖。
- 真实 LLM 的 response cache 默认落在 `outputs/llm_cache/{deepseek,bailian,minimax}/`。

## Python 环境

项目运行需要 torch（已通过 `profiles/` 时间安全机制用到的 `bisect_left` 不依赖 torch，但模型/图构建依赖）。**默认使用 anaconda 的 sentiment 环境**：

```bash
"D:/anaconda/envs/sentiment/python.exe" main.py debate --limit-blocks 1 --rounds 1
```

- 该环境已装 `torch 2.11.0+cpu`。
- **未装 pytest**：跑测试用 `python -m unittest discover -s tests`（与 pytest 等价覆盖 4 个 stage 测试文件）。
- 系统默认 `python` 路径（如 `C:\Python314\python.exe`）通常没装 torch，直接调 `python main.py ...` 会因 `debate_graph/graph_batch.py` import torch 失败。
- IDE 终端如果默认走 anaconda base，注意切到 sentiment 环境后再跑。

## 统一入口：`main.py` 子命令

```bash
python main.py blocks                                  # 加载 JSONL，打印 CommentBlock 统计 + 时间切分
python main.py debate --limit-blocks 3 --rounds 1 --mode minimax
python main.py debate --limit-blocks 1 --mode deepseek # DeepSeek 真实 LLM 辩论
python main.py debate --limit-blocks 1 --mode bailian  # 阿里云百炼 OpenAI 兼容
python main.py debate --limit-blocks 1 --mode minimax  # MiniMax Anthropic 兼容
python main.py debate --limit-blocks 1 --mode siliconflow # 硅基流动 OpenAI 兼容
python main.py graphs --limit-blocks 3 --rounds 1      # 融合评论图+辩论图
python main.py train-prototype --epochs 3              # 最小可训练 smoke
python main.py full --limit-blocks 1 --rounds 1        # 完整链路：辩论→图→模型摘要→法官
python main.py full --limit-blocks 1 --debate-mode deepseek --judge-mode deepseek
python main.py full --limit-blocks 1 --debate-mode minimax --judge-mode minimax
python main.py evaluate --rounds 1 --metrics-json outputs/eval_metrics.json
python main.py split-experiment --train-count 9 --val-count 3 --test-count 3 --epochs 5 --debate-mode minimax --judge-mode minimax --seed 42 --output-json outputs/split_9_3_3_minimax.json
python main.py case-study --post-id 305698686327490 --debate-mode deepseek --output-md outputs/case.md
python -m scripts.export_case_csv --input-json outputs/case.json --output-dir outputs/case_csv
python -m scripts.export_metrics_csv --input-json outputs/split.json --output-csv outputs/metrics.csv
```

子命令脚本入口在 `scripts/`，但 README 明确说**优先使用 `main.py`**。每个子命令接受 `--input` 覆盖默认数据。

## 架构：端到端数据流

```
JSONL (dataset/final.jsonl)
  ↓ data.loader.load_posts
PostRecord
  ↓ data.block_builder.build_comment_blocks
CommentBlock (一个根评论 + 它的 replies；block_id = post_id:comment_id)
  ↓ profiles.ProfileStore.get_profiles_for_block   ← 严格 t < t0 时间安全
{author: UserProfile}
  ↓ agent.DebateOrchestrator.run(block, profiles, rounds=2)
DebateTranscript (Argument 列表，每条含 camp/role/claim/evidence/confidence/targets/phase)
  ↓ debate_graph.hetero_graph.build_hetero_graph
HeteroGraph (comment 节点 + argument 节点；6 种关系 reply/cite/support/attack/respond/propose)
  ↓ debate_graph.graph_batch.graph_to_tensor
GraphTensor (num_nodes×8 特征矩阵 + 每个 relation 一个 dense 邻接矩阵)
  ↓ model.bdg_ode.pipeline.GraphSentimentModel
ModelOutputSummary (bullish_probability, bull_mean, bear_mean, bull_bear_margin, ode_steps)
  ↓ judge.create_judge_client(mode).judge(transcript, summary, graph)
JudgeOutput (verdict ∈ {BULLISH, BEARISH, NEUTRAL} + confidence + report + score_vector)
```

**关键约束**：
- `ProfileStore.get_profiles_for_block` 用 `bisect_left` 严格排除 `t == t0` 的历史，保证不偷看未来。
- 法官在模型演化**之后**才运行（不是辩论后先判一次），所以 `judge.judge` 同时接收 `transcript` 和 `model_summary`。
- 辩论 7 阶段流水线在 `agent/debate_orchestrator.py`：initial_argument → intra_reflection → intra_response → cross_response → counter_reflection → counter_rebuttal → reflection_summary。每个论点带 round/seq/phase，不再生成 precede 边。

## 模块边界（不要在错误的层修改）

| 文件 | 责任 |
| --- | --- |
| `data/schema.py` | `RawComment` / `PostRecord` / `CommentBlock` / `FilterIssue` 契约；`parse_datetime` 兼容秒/毫秒 epoch |
| `data/loader.py` | 单文件/目录/glob 三种输入 |
| `data/block_builder.py` | 帖子→根评论级样本；只校验根评论字段 |
| `data/filters.py` | `post_level_issue` + `validate_root_comment` 记录 `FilterIssue` 原因 |
| `data/temporal_split.py` | 70/15/15 时间顺序切分（**不**随机打乱） |
| `profiles/profile_store.py` | 作者历史索引；唯一允许读取历史画像的入口 |
| `profiles/user_profile.py` | `UserProfile` 计算逻辑；冷启动画像字段在 `config.py` |
| `agent/prompts.py` | 论文 v4 角色名、阶段名、提示词；新角色先改这里再改 client |
| `agent/debate_orchestrator.py` | 只控制"谁在第几轮第几个阶段发言"，不写 LLM 调用细节 |
| `agent/client_factory.py` | `deepseek` / `bailian` / `minimax` / `siliconflow` 模式切换 |
| `agent/anthropic_compatible.py` | Anthropic Messages 兼容协议共享层（DeepSeek + MiniMax），含 DebateClient 与 JudgeClient 基类 + 两个 provider 薄包装 |
| `agent/openai_compatible.py` | OpenAI Chat Completions 兼容协议共享层（Bailian + SiliconFlow），含 DebateClient 与 JudgeClient 基类 + provider 薄包装 |
| `judge/client_factory.py` | `deepseek` / `bailian` / `minimax` / `siliconflow` 法官 provider 切换 |
| `debate_graph/comment_graph.py` | 评论树→reply 边 |
| `debate_graph/debate_graph.py` | Argument→cite/support/attack/respond/propose 边 |
| `debate_graph/hetero_graph.py` | 融合两图 + 节点去重 + 端点缺失边丢弃 |
| `debate_graph/diffusion_ops.py` | 关系邻接行归一化 |
| `debate_graph/graph_batch.py` | `GraphTensor` 封装；`NODE_FEATURE_DIM=8` 手工特征 |
| `model/bdg_ode/` | 当前默认模型：DualEncoder → ODE 演化 → DualReadout → Calibrator |
| `model/bdg_ode/ode_solver.py` | `torchdiffeq.odeint` + 手写 Euler fallback |
| `model/model_summary.py` | `ModelOutputSummary` 是给法官的接口契约 |
| `judge/consistency.py` | 检查 verdict/confidence/score vector 内部一致性 |

**待办（README 已列，勿重复造轮子）**：
- 完整损失 `L_mse / L0 / L1 / L2 / L3 / L_judge_cons`
- 评估模块 Accuracy / Macro-F1 / ECE / Brier / Market Match Rate
- calibrator 接入 JudgeScoreVector `J`
- 正式训练系统（batch、checkpoint、early stopping）

## 运行 LLM 真实辩论

API key 不写在 `config.py` 里（commit `4b08e29` 已切到 `.env` + 环境变量双轨）。优先级：

1. **`.env` 文件**（项目根目录，已加入 `.gitignore`）—— 推荐方式，跨 shell 持久。模板见 `.env.example`。
2. **环境变量** —— PowerShell 临时覆盖：

```powershell
$env:DEEPSEEK_API_KEY="sk-..."          # DeepSeek 辩论
$env:ANTHROPIC_API_KEY="sk-..."         # 兼容别名；MiniMax 也读这个
$env:MINIMAX_API_KEY="sk-..."           # MiniMax 备用别名
$env:DASHSCOPE_API_KEY="sk-..."         # 阿里云百炼（OpenAI 兼容）
$env:BAILIAN_API_KEY="sk-..."           # 阿里云百炼备用别名
$env:SILICONFLOW_API_KEY="sk-..."       # 硅基流动（OpenAI 兼容）
```

`_load_env_file()` 在 `config.py:20-32` 会把 `.env` 注入到 `os.environ`（`setdefault`，shell 已 export 的同名变量优先）。

默认端点：
- DeepSeek Anthropic 兼容：`https://api.deepseek.com/anthropic`，模型 `deepseek-v4-pro`
- 阿里云百炼 OpenAI 兼容：`https://dashscope.aliyuncs.com/compatible-mode/v1`，模型以 `config.py` 的 `BAILIAN_MODEL` 为准（当前为 `qwen-flash`）
- MiniMax Anthropic 兼容：`https://api.minimax.io/anthropic`，模型 `MiniMax-M3`
- 硅基流动 OpenAI 兼容：`https://api.siliconflow.cn/v1`，模型以 `SILICONFLOW_MODEL` 为准（默认 `Pro/zai-org/GLM-4.7`，可在 `.env` 覆盖）

## 测试

```bash
# 推荐:unittest(sentiment 环境没装 pytest)
python -m unittest discover -s tests

# 如果装了 pytest:
python -m pytest tests -p no:cacheprovider
```

- `pytest.ini` 全局禁用 `cacheprovider`（Windows 沙箱下 cache 写权限问题）。
- 4 个 stage 测试文件：`test_stage1_pipeline.py` / `test_stage2_debate_judge.py` / `test_stage3_graphs.py` / `test_stage4_model.py`。
- `test_stage4_model.py::test_split_experiment_smoke` 在没有 `dataset/final.jsonl` 时会 skip。
- 单文件运行：`python -m unittest tests.test_stage3_graphs`（pytest 风格：`python -m pytest tests/test_stage3_graphs.py -p no:cacheprovider -v`）。

## VSCode 调试

`.vscode/launch.json` 已配 `Debug full pipeline fixture`：`main.py full --input tests\fixtures\sample_post.jsonl --limit-blocks 1 --rounds 1 --train-epochs 1`。

## 配置常量组织

`config.py` 是单一来源，**不要在调用方硬编码超参数**。分块组织：数据/时间切分、用户画像、辩论流程、DeepSeek/百炼/MiniMax/硅基流动、图特征（`NODE_FEATURE_DIM=8`）、ODE（`ODE_STEPS=4`, `ODE_SOLVER_BACKEND=torchdiffeq`）、训练默认值、概率边界。论文里 `L_mse / L_judge_cons` 的常数应该加在 judge 那一段。

## 输出约定

完整流程 record 都有这五个键：`block` / `profiles` / `debate` / `graph` / `model_summary` / `judge`。`judge.score_vector` 形如 `[p_bull, p_bear, q_bull, q_bear, e_bull, e_bear, c, d, a, rho]`（10 维）。`judge.report` 是可读判决文本，`split-experiment` 还会把它直接打印出来。`evaluate` 把 verdict 映射回 `1 / -1 / 0` 与 `CommentBlock.label` 对齐做 accuracy / coverage / macro-F1。

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
- 真实 LLM 的 response cache 默认落在 `outputs/llm_cache/{deepseek,bailian}/`。

## 统一入口：`main.py` 子命令

```bash
python main.py blocks                                  # 加载 JSONL，打印 CommentBlock 统计 + 时间切分
python main.py debate --limit-blocks 3 --rounds 1      # 仅生成辩论（默认 mock）
python main.py debate --limit-blocks 1 --mode deepseek # 真实 LLM 辩论
python main.py debate --limit-blocks 1 --mode bailian  # 阿里云百炼
python main.py graphs --limit-blocks 3 --rounds 1      # 融合评论图+辩论图
python main.py train-prototype --epochs 3              # 最小可训练 smoke
python main.py full --limit-blocks 1 --rounds 1        # 完整链路：辩论→图→模型摘要→法官
python main.py full --limit-blocks 1 --debate-mode deepseek --judge-mode deepseek
python main.py evaluate --rounds 1 --metrics-json outputs/eval_metrics.json
python main.py split-experiment --train-count 9 --val-count 3 --test-count 3 --epochs 5 --seed 42 --output-json outputs/split_9_3_3_mock.json
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
| `agent/client_factory.py` | `mock` / `deepseek` / `bailian` 模式切换 |
| `agent/{mock,deepseek,bailian}_client.py` | 三种 client；都实现 `DebateClient.generate_argument` |
| `debate_graph/comment_graph.py` | 评论树→reply 边 |
| `debate_graph/debate_graph.py` | Argument→cite/support/attack/respond/propose 边 |
| `debate_graph/hetero_graph.py` | 融合两图 + 节点去重 + 端点缺失边丢弃 |
| `debate_graph/diffusion_ops.py` | 关系邻接行归一化 |
| `debate_graph/graph_batch.py` | `GraphTensor` 封装；`NODE_FEATURE_DIM=8` 手工特征 |
| `model/bdg_ode/` | 当前默认模型：DualEncoder → ODE 演化 → DualReadout → Calibrator |
| `model/bdg_ode/ode_solver.py` | `torchdiffeq.odeint` + 手写 Euler fallback |
| `model/model_summary.py` | `ModelOutputSummary` 是给法官的接口契约 |
| `judge/model_aware_judge.py` | 离线 mock 法官；`p_bull = JUDGE_MODEL_WEIGHT*model + JUDGE_DEBATE_WEIGHT*debate + JUDGE_MARGIN_WEIGHT*margin` |
| `judge/consistency.py` | 检查 verdict/confidence/score vector 内部一致性 |

**待办（README 已列，勿重复造轮子）**：
- 完整损失 `L_mse / L0 / L1 / L2 / L3 / L_judge_cons`
- 评估模块 Accuracy / Macro-F1 / ECE / Brier / Market Match Rate
- calibrator 接入 JudgeScoreVector `J`
- 正式训练系统（batch、checkpoint、early stopping）

## 运行 LLM 真实辩论

API key 写在 `config.py` 中是反模式（已硬编码到仓库里，见 `DEEPSEEK_API_KEY_ENV` / `BAILIAN_API_KEY_ENV`），**修改时注意：先在本地覆盖，再考虑加 `.env` 方案**。PowerShell 临时覆盖：

```powershell
$env:DEEPSEEK_API_KEY="sk-..."
$env:ANTHROPIC_API_KEY="sk-..."  # 兼容别名
$env:DASHSCOPE_API_KEY="sk-..."
```

默认端点：
- DeepSeek Anthropic 兼容：`https://api.deepseek.com/anthropic`，模型 `deepseek-v4-pro`
- 阿里云百炼 OpenAI 兼容：`https://dashscope.aliyuncs.com/compatible-mode/v1`，模型 `deepseek-v4-flash`

## 测试

```bash
python -m unittest discover -s tests
python -m pytest tests -p no:cacheprovider
```

- `pytest.ini` 全局禁用 `cacheprovider`（Windows 沙箱下 cache 写权限问题）。
- 4 个 stage 测试文件：`test_stage1_pipeline.py` / `test_stage2_debate_judge.py` / `test_stage3_graphs.py` / `test_stage4_model.py`。
- `test_stage4_model.py::test_split_experiment_smoke` 在没有 `dataset/final.jsonl` 时会 skip。
- 单文件运行：`python -m pytest tests/test_stage3_graphs.py -p no:cacheprovider -v`。

## VSCode 调试

`.vscode/launch.json` 已配 `Debug full pipeline fixture`：`main.py full --input tests\fixtures\sample_post.jsonl --limit-blocks 1 --rounds 1 --train-epochs 1`。

## 配置常量组织

`config.py` 是单一来源，**不要在调用方硬编码超参数**。分块组织：数据/时间切分、用户画像、Mock 辩论、DeepSeek/百炼、图特征（`NODE_FEATURE_DIM=8`）、ODE（`ODE_STEPS=4`, `ODE_SOLVER_BACKEND=torchdiffeq`）、训练默认值、Mock 法官（`JUDGE_MODEL_WEIGHT=0.60` 等）、概率边界。论文里 `L_mse / L_judge_cons` 的常数应该加在 judge 那一段。

## 输出约定

完整流程 record 都有这五个键：`block` / `profiles` / `debate` / `graph` / `model_summary` / `judge`。`judge.score_vector` 形如 `[p_bull, p_bear, q_bull, q_bear, e_bull, e_bear, c, d, a, rho]`（10 维）。`judge.report` 是可读判决文本，`split-experiment` 还会把它直接打印出来。`evaluate` 把 verdict 映射回 `1 / -1 / 0` 与 `CommentBlock.label` 对齐做 accuracy / coverage / macro-F1。

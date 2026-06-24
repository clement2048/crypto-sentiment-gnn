# 项目状态与待完成项

更新时间：2026-06-20

本文档用于记录当前代码仓库相对论文 V5 / 框架设计 V3 的实现状态，以及本机 Obsidian 研究文档位置。长研究脉络仍以 Obsidian 为准；本文件只保留开发协作需要的高频事实。

## 研究文档位置

Windows 旧路径：

```text
E:\obsidian\knowledge\基于舆情的异常活动检测\市场波动对股民情绪影响
```

MacBook 当前路径：

```text
/Users/tim/学习资料/obsidian-note/基于舆情的异常活动检测/市场波动对股民情绪影响
```

重点文件：

```text
市场波动对股民情绪的影响_v5.md
框架设计/情绪分析系统_v3.md
研究协作记录.md
待解决问题.md
拓展优化思路.md
归档/实现进度与TODO.md
```

说明：`归档/实现进度与TODO.md` 是 v2 时代 TODO，但其中“正式训练、评估、损失、消融、轨迹监督”等缺口仍可作为当前实现差距参考；具体是否沿用需按 V5 / V3 重新核对。

## 已基本落地

- 数据加载、过滤、`CommentBlock` 构建：`data/`。
- 根评论级标签口径：`CommentBlock.label` 来自根评论，不使用帖子级投票。
- 时间安全画像入口：`profiles.ProfileStore.get_profiles_for_block`，历史记录使用 `timestamp < t0`。
- 双 agent 辩论：`bull_agent` / `bear_agent`，由 `agent/debate_orchestrator.py` 编排。
- LLM 输入安全视图：Agent / Judge prompt payload 不直接传 `p1`、未来价格或真实标签。
- 单关系图：评论父子关系放在评论节点 `parent_id`，辩论图只生成 `interact` 边。
- Bi-ODE 原型：`model/bdg_ode/` 已有双通道编码、ODE 演化、读出与校准器。
- Judge schema / prompt / parser / provider：`judge/` 与 `agent/*_compatible.py`。
- 分阶段命令入口：`main.py`。

## 部分落地

- 反思循环：已有 `agent/reflection.py`、Judge `weak_dims/supplement_suggestions`、`run_full_pipeline(..., reflection_rounds=...)`；但主命令 `main.py full/evaluate/split-experiment` 默认未暴露完整反思轮次参数，反思效果也未做系统实验。
- 训练系统：已有 `train_graph_model`、checkpoint、early stopping 和若干辅助损失；但仍偏原型训练，不是完整论文实验训练系统。
- 损失函数：已有分类、初始对齐、平滑、互斥、强度回归；尚未实现或验证 Obsidian 中提到的 `L_judge_cons`、`L_activity` 等完整组合。
- 评估：已有 accuracy、precision、recall、F1、混淆统计；ECE、Brier、Market Match Rate、活动强度分层等仍未完整实现。
- 市场行为验证：已有价格方向验证；交易量/活动强度验证尚未系统化。
- 用户画像：七维字段已基本存在；活跃度、影响力、文本情绪稳定性仍是轻量规则特征，未必等同论文级定义。
- 节点特征：已删除 12 维手工结构特征，默认使用 `sentencebert` 文本 embedding；可选 `finbert`、`sentencebert_finbert`，入口参数为 `--embedding-backend`。

## 近期 TODO

1. 标签生成阈值仍需确认：`label` 是否仅由 `p1 > p0 / p1 < p0` 决定，还是存在最小涨跌幅阈值。
2. `t0` 定义仍需最终确认：当前代码使用根评论时间；如论文要使用最后一条回复时间，需要同步改数据契约、画像边界和测试。
3. 主评估指标待讨论：目前只把 `accuracy` 和 `F1` 作为勉强可接受的候选主指标；`precision`、`recall`、混淆矩阵等是否保留为辅助展示，后续逐项讨论。
4. 正式实验协议未完成：时间切分、训练/验证/测试规模、随机种子、成本统计、缓存策略、失败重试日志需要定稿。
5. Calibrator 是否接入 Judge 评分向量仍未决：当前模型校准器主要使用图模型表示，未把 Judge score vector 作为可训练输入。
6. BDG-ODE 轨迹监督仍不完整：当前以图级二分类为主，尚未形成论文级轨迹监督/正则化实验。
7. 外部检索/链上/K 线等工具暂未接入；若未来接入，所有外部特征必须满足 `timestamp <= t0`。
8. 当前 Mac 环境测试未跑通：`python` 命令不存在，`python3` 环境缺少 `config.py` 和 `torch`。本项目仍建议使用配置好的环境运行测试。

## 暂缓项

- 消融实验先不作为近期讨论重点；原先列出的无用户画像、无辩论、无 `evidence.source`、无反思循环、无 ODE、无 Judge/model 交叉输入等，后续需要写论文实验时再重新筛选。
- ECE、Brier、Market Match Rate、活动强度分层等评估指标先暂缓，不作为当前主评估指标候选。

## 协作注意

- 不要绕过 `ProfileStore.get_profiles_for_block` 读取用户画像。
- 不要把 `p1`、未来价格、真实标签传给 LLM Agent / Judge。
- `delta_p = (p1 - p0) / p0` 只能用于标签构造或事后验证，不能作为模型直接输入。
- LLM Agent / LLM Judge 不可微，不要把 Judge 输出直接塞进 `loss.backward()`。
- README / CLAUDE / AGENTS 只保留低上下文规则；长 TODO 和研究决策放在 Obsidian 或本状态文档。

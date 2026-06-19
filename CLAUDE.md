# CLAUDE.md

This file provides guidance to Claude Code when working in this repository.

## 回答风格

- 用直接、简洁、事实优先的方式回答。
- 避免客套话、奉承话、过度积极评价或缺乏依据的主观判断，例如“你这个问题很好”“非常棒的想法”“我完全理解”。
- 只有在有充分证据时，才使用“显然”“毫无疑问”等强判断表述。
- 当信息不足、无法确定、没有依据或超出已知范围时，直接说明“我不知道”“目前无法确定”或“需要更多信息才能判断”。
- 不要编造答案，不要用模糊措辞掩盖不确定性。
- 回答时区分事实、推断和建议：
  - 事实：基于可验证信息。
  - 推断：明确说明是推断。
  - 建议：说明适用条件和局限。
- 优先给结论，再给必要理由；不要为了显得友好而增加无实质内容的寒暄。

## 工作规则

- 始终用中文回复。
- 不确定就明确说“不确定”或“不知道”，不要猜测或编造。
- 涉及论文定义、实验口径、API 用法或命令参数时，优先查本仓库代码和 Obsidian 文档。
- 不要绕过 `profiles.ProfileStore.get_profiles_for_block` 读取历史画像。
- 不要把 `p1`、未来价格、真实标签传给 LLM Agent / Judge。
- LLM Agent / LLM Judge 不可微，不能把 Judge 输出直接塞进 `loss.backward()`。

## 当前项目

加密货币舆情二分类原型，当前以论文 V5 和框架设计 V3 为准：

```text
JSONL
  -> CommentBlock
  -> 时间安全用户画像
  -> bull_agent / bear_agent 辩论
  -> debate graph with single interact relation
  -> Bi-ODE 图模型
  -> Judge structured report
  -> 市场价格方向验证
```

当前主流程使用一个正方 agent 和一个反方 agent：

- `bull_agent`
- `bear_agent`

Judge 可以返回反思报告。反思信息只能指出薄弱维度和补充建议，不能暴露真实标签、`p1` 或未来价格。

## 外部研究文档

遇到论文语义、实验目标、TODO 优先级不确定时，查 Obsidian：

```text
E:\obsidian\knowledge\基于舆情的异常活动检测\市场波动对股民情绪影响
```

重点文件：

```text
市场波动对股民情绪的影响_v5.md
框架设计\情绪分析系统_v3.md
研究协作记录.md
```

## 硬性约束

1. 主监督标签是根评论级 `CommentBlock.label`，不是帖子级投票。
2. 用户画像、Agent 输入、外部特征必须满足 `t < t0`。
3. `p1`、未来价格、真实标签不能传给 LLM Agent / Judge。
4. 价格方向 `delta_p = (p1 - p0) / p0` 只能用于标签构造或事后验证，不能作为模型直接输入。
5. 交易量变化只能作为活动强度信号，不能决定方向。
6. 评论父子结构保存在评论节点 `parent_id`，不生成 `reply` 边。
7. 辩论图只保留 `interact` 关系；不生成 `cite` 边。
8. 引用内容放在生成回复和 `evidence.source` 中。
9. 论点回应目标使用 `target_args`，时间顺序使用 `t_index`。

## 常用命令

优先使用 `main.py`：

```bash
python main.py blocks
python main.py debate --limit-blocks 1 --rounds 4 --mode siliconflow
python main.py graphs --limit-blocks 1 --rounds 4 --mode siliconflow
python main.py full --limit-blocks 1 --rounds 4 --debate-mode siliconflow --judge-mode siliconflow
python main.py evaluate --rounds 4 --debate-mode siliconflow --judge-mode siliconflow
python main.py split-experiment --train-count 9 --val-count 3 --test-count 3 --rounds 4 --epochs 5 --debate-mode siliconflow --judge-mode siliconflow
```

可选 provider：

```text
deepseek
bailian
siliconflow
```

## Python 环境

推荐使用：

```bash
"D:/anaconda/envs/sentiment/python.exe" -m unittest discover -s tests
```

系统默认 `python` 可能没有 `torch`，不要默认假设可用。

## 关键模块

| 路径 | 责任 |
| --- | --- |
| `data/` | JSONL 加载、过滤、CommentBlock 构建 |
| `profiles/` | 时间安全用户画像 |
| `agent/prompts.py` | 当前 `bull_agent` / `bear_agent` 提示词 |
| `agent/debate_orchestrator.py` | 正反方 agent 发言与反思补充顺序 |
| `agent/reflection.py` | 反思信号与继续条件 |
| `debate_graph/comment_graph.py` | 评论节点与 `parent_id` |
| `debate_graph/debate_graph.py` | 论点 `interact` 图 |
| `debate_graph/graph_batch.py` | 图张量化 |
| `model/bdg_ode/` | Bi-ODE 图模型 |
| `judge/` | Judge schema、prompt、parser、provider |
| `verification/` | 市场方向与活动强度验证 |
| `scripts/` | 分阶段脚本 |
| `tests/` | 单元测试，测试替身在 `tests/fakes.py` |

## 配置

- 默认数据：`dataset/final.jsonl`
- API key：项目根目录 `.env` 或环境变量
- LLM cache：`outputs/llm_cache/`
- 配置常量优先放在 `config.py`，不要在调用方硬编码超参数。

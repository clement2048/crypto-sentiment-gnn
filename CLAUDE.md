# CLAUDE.md

This file provides guidance to Claude Code when working in this repository.

## 工作语言与事实规则

- 始终用中文回复。
- 不确定就明确说“不确定”或“不知道”，不要猜测或编造。
- 涉及论文定义、实验口径、API 用法或命令参数时，优先查本仓库代码和 Obsidian 文档。

## 当前项目

加密货币舆情二分类原型 v2：

```text
JSONL -> CommentBlock -> 时间安全用户画像 -> bull/bear 双 agent 辩论
      -> reply/respond 异构图 -> BDG-ODE 图模型 -> 在线 Judge
```

当前主流程使用一个正方 agent 和一个反方 agent：

- `bull_agent`
- `bear_agent`

辩论图只保留 `respond` 关系；评论树保留 `reply` 关系。引用内容放在 agent 输出文本和 `evidence` 字段里，不生成 `cite` 边。

## 外部研究文档

遇到论文语义、实验目标、TODO 优先级不确定时，查 Obsidian：

```text
E:\obsidian\knowledge\基于舆情的异常活动检测\市场波动对股民情绪影响\
```

重点文件：

```text
市场波动对股民情绪的影响_v4.md
框架设计\情绪分析系统_v2.md
框架设计\情绪分析系统_v2_实现进度与TODO.md
研究协作记录.md
```

## 硬性约束

1. 主监督标签是根评论级 `CommentBlock.label`，不是帖子级投票。
2. 用户画像、Agent 输入、外部特征必须满足 `t < t0`。
3. 不要绕过 `profiles.ProfileStore.get_profiles_for_block` 读取历史画像。
4. `p1`、未来价格、真实标签不能传给 LLM Agent / Judge。
5. 价格方向 `delta_p = (p1 - p0) / p0` 只能用于标签构造或验证，不能作为模型直接监督。
6. 交易量变化只能作为活动强度信号，不能决定方向。
7. LLM Agent / LLM Judge 不可微，不能把 Judge 输出直接塞进 `loss.backward()`。

## 常用命令

优先使用 `main.py`：

```bash
python main.py blocks
python main.py debate --limit-blocks 1 --rounds 1 --mode siliconflow
python main.py graphs --limit-blocks 1 --rounds 1 --mode siliconflow
python main.py full --limit-blocks 1 --rounds 1 --debate-mode siliconflow --judge-mode siliconflow
python main.py evaluate --rounds 1 --debate-mode siliconflow --judge-mode siliconflow
python main.py split-experiment --train-count 9 --val-count 3 --test-count 3 --rounds 1 --epochs 5 --debate-mode siliconflow --judge-mode siliconflow
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
| `agent/debate_orchestrator.py` | 正反双 agent 发言顺序 |
| `agent/anthropic_compatible.py` | DeepSeek Anthropic 兼容层 |
| `agent/openai_compatible.py` | 百炼 / 硅基流动兼容层 |
| `debate_graph/comment_graph.py` | 评论树 `reply` 图 |
| `debate_graph/debate_graph.py` | 论点 `respond` 图 |
| `debate_graph/graph_batch.py` | 图张量化 |
| `model/bdg_ode/` | BDG-ODE 图模型 |
| `judge/` | Judge schema、parser、provider |
| `scripts/` | 分阶段脚本 |
| `tests/` | 单元测试，测试替身在 `tests/fakes.py` |

## 配置

- 默认数据：`dataset/final.jsonl`
- API key：项目根目录 `.env` 或环境变量
- LLM cache：`outputs/llm_cache/`
- 配置常量优先放在 `config.py`，不要在调用方硬编码超参数。

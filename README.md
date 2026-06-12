# 情绪分析系统 v2 原型

加密货币舆情二分类原型：从 Binance Square JSONL 数据构建根评论级样本，经时间安全用户画像、正反双 agent 辩论、异构图、BDG-ODE 图模型和在线 Judge，输出 `BULLISH / BEARISH / NEUTRAL` 判决。

## 项目结构

```text
agent/          正方/反方双 agent 辩论与 LLM provider
data/           JSONL 加载、过滤、CommentBlock 构建
dataset/        默认数据集 final.jsonl
debate_graph/   评论 reply 图、论点 respond 图、图张量化
judge/          Judge schema、parser、在线 Judge provider
model/          BDG-ODE 图情绪模型
profiles/       时间安全用户画像
scripts/        分阶段脚本
tests/          单元测试
main.py         统一命令行入口
```

## 环境

推荐使用本机 anaconda 的 `sentiment` 环境：

```bash
"D:/anaconda/envs/sentiment/python.exe" main.py blocks
```

测试推荐：

```bash
"D:/anaconda/envs/sentiment/python.exe" -m unittest discover -s tests
```

系统默认 `python` 可能没有安装 `torch`，直接运行可能失败。

## 数据流

```text
dataset/final.jsonl
  -> data.build_comment_blocks
  -> profiles.ProfileStore.get_profiles_for_block  # 严格 t < t0
  -> agent.DebateOrchestrator                      # bull_agent / bear_agent
  -> debate_graph.build_hetero_graph               # reply / respond
  -> model.GraphSentimentModel
  -> judge.create_judge_client(...).judge(...)
```

默认输入是 `dataset/final.jsonl`，所有主命令都可用 `--input` 覆盖。

## 运行

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

## API Key

API key 放在项目根目录 `.env` 或当前 shell 环境变量中。`.env` 已被 `.gitignore` 忽略。

```bash
DEEPSEEK_API_KEY=
BAILIAN_API_KEY=
DASHSCOPE_API_KEY=
ANTHROPIC_API_KEY=
SILICONFLOW_API_KEY=
SILICONFLOW_MODEL=
```

常用临时设置示例：

```powershell
$env:ANTHROPIC_API_KEY="..."
$env:SILICONFLOW_API_KEY="..."
```

## 核心约束

- 监督标签是 `CommentBlock.label`：`1` 看涨，`-1` 看跌。
- 用户画像和 Agent 输入必须遵守时间边界 `t < t0`。
- `p1`、未来价格、真实标签不能传给 LLM Agent / Judge。
- 市场价格方向只能用于构造或验证标签，不能作为模型直接监督。
- LLM Agent / LLM Judge 不可微，不能把 Judge 输出直接塞进 `loss.backward()`。

## 当前图结构

评论图保留真实评论树的 `reply` 边。辩论图只保留论点之间的 `respond` 边；引用材料保存在 agent 输出文本和 `evidence` 字段中，不再生成 `cite` 边。

## 测试

```bash
python -m unittest discover -s tests
python -m pytest tests -p no:cacheprovider
```

`pytest` 可选；当前环境优先使用 `unittest`。

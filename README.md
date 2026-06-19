# Crypto Sentiment GNN Prototype

加密货币舆情二分类原型：从 Binance Square JSONL 数据构造根评论级样本，经过时间安全用户画像、正反方双 agent 辩论、单关系辩论图、Bi-ODE 图模型、在线 Judge 与市场方向验证，输出 `BULLISH / BEARISH` 判决。

## 项目结构

```text
agent/          正方 / 反方 agent、反思信号、LLM provider
data/           JSONL 加载、过滤、CommentBlock 构建
debate_graph/   评论节点、interact 辩论图、图张量化
judge/          Judge schema、prompt、parser、provider
model/          Bi-ODE 图情绪模型与训练损失接口
profiles/       时间安全用户画像
scripts/        分阶段运行、评估、导出脚本
tests/          单元测试
verification/   市场价格方向与活动强度验证
main.py         统一命令行入口
```

## 环境

推荐使用本机 anaconda 的 `sentiment` 环境：

```bash
"D:/anaconda/envs/sentiment/python.exe" main.py blocks
"D:/anaconda/envs/sentiment/python.exe" -m unittest discover -s tests
```

系统默认 `python` 可能没有安装 `torch`。

## 数据流

```text
dataset/final.jsonl
  -> data.build_comment_blocks
  -> profiles.ProfileStore.get_profiles_for_block  # 严格 t < t0
  -> agent.DebateOrchestrator                      # bull_agent / bear_agent
  -> debate_graph.build_hetero_graph               # single interact relation
  -> model.GraphSentimentModel                     # Bi-ODE + polarity seed
  -> judge.create_judge_client(...).judge(...)
  -> verification.verify_market_behavior
```

默认输入是 `dataset/final.jsonl`，主命令可用 `--input` 覆盖。

## 运行

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

PowerShell 临时设置示例：

```powershell
$env:SILICONFLOW_API_KEY="..."
```

## 核心约束

- 主监督标签是根评论级 `CommentBlock.label`：`1` 看涨，`-1` 看跌。
- 用户画像、Agent 输入、Judge 输入必须满足时间边界 `t < t0`。
- `p1`、未来价格、真实标签不能传给 LLM Agent / Judge。
- 价格方向只用于标签构造或事后验证，不能作为模型直接输入。
- 交易量变化只表示活动强度，不决定方向标签。
- LLM Agent / LLM Judge 不可微，不能把 Judge 输出直接接入 `loss.backward()`。

## 当前图结构

评论结构不再生成 `reply` 边；父子关系保存在评论节点 `attrs.parent_id`。

辩论图只保留一种关系：`interact`。agent 引用内容保存在论点文本与 `evidence.source` 字段中，不生成 `cite` 边。论点回应目标使用 `target_args`，时间顺序使用 `t_index`。

## 测试

```bash
python -m unittest discover -s tests
python -m pytest tests -p no:cacheprovider
```

当前优先使用 `unittest`。

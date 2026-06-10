# CLAUDE.md

## 语言要求

- 所有对话使用中文。
- 所有文档使用中文。
- 所有代码注释使用中文。

## 执行要求

- 在生成说明、总结、计划、提交说明时，统一使用中文。
- 在新增或修改 Markdown 文档时，统一使用中文。
- 在新增或修改代码注释时，统一使用中文。
- 当更新代码后，请遵循如下步骤：
  1. 确保已激活 conda 环境 "sentiment"（位于 D 盘）。
  2. 先自行运行相关测试以验证正确性。
  3. 告知我用于验证代码更新正确性的命令。

## 项目概览

基于多 Agent 辩论 + 图神经网络（GNN）的币安广场评论情绪分析系统。输入为 `binance-square-collector` 输出的 JSONL 文件，输出为市场情绪预测（看涨/看跌）。

## 模块结构

| 目录 | 功能 |
|------|------|
| `agent/` | 用户 Agent 系统（user_profile、llm_agent、agent_factory、agent_orchestrator） |
| `gnn/` | 图神经网络（model、dataset、trainer、predictor、metrics） |
| `features/` | 特征工程（keyword_sentiment、text_embedding、feature_pipeline） |
| `data_loader/` | 数据加载与预处理（loader、preprocessor、graph_builder） |
| `config.py` | 模型与训练超参配置 |
| `main.py` | 主入口（train / analyze 模式） |
| `logger.py` | 日志模块 |

## 数据流

```
JSONL 文件 → data_loader/loader.py（对话列表）
          → features/feature_pipeline.py（TF-IDF 特征）
          → agent/agent_orchestrator.py（Agent 情绪分析）
          → data_loader/graph_builder.py（辩论图）
          → gnn/model.py → gnn/trainer.py（GNN 训练）
          → gnn/predictor.py（情绪预测）
```

## 命令

```bash
# 训练模式
python main.py --mode train --input dataset/result/parsed_28.jsonl

# 分析模式（单帖）
python main.py --mode analyze --post-id 317353268392961

# 分析模式（全部有标签对话）
python main.py --mode analyze
```

## 关键设计决策

- **Agent 辩论图**：每个评论用户由 LLM Agent 扮演，分析其看涨/看跌倾向，多用户交互构成辩论图
- **GNN 图分类**：三层 GCN + 全局池化 + MLP，输出 [0,1] 概率（>0.5 看涨）
- **特征融合**：TF-IDF 文本特征 + Agent 情绪向量 [bullish, bearish, neutral, confidence] + 节点深度
- **LLM 配置**：通过 `config.py` 中的 `LLM_*` 常量配置模型、API 地址、超参

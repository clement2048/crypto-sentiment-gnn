# Crypto Sentiment GNN — 币安广场评论情绪分析

基于多 Agent 辩论 + 图神经网络（GNN）的加密货币市场情绪分析系统。从币安广场评论数据出发，通过 LLM Agent 角色扮演分析每个用户的看涨/看跌倾向，构建 Agent 辩论图，最终由 GNN 输出整体市场情绪预测。

输入数据来自 [binance-square-collector](https://github.com/) 输出的 JSONL 文件。

## 目录

- [项目结构](#项目结构)
- [数据流](#数据流)
- [环境配置](#环境配置)
- [使用方法](#使用方法)
- [配置说明](#配置说明)
- [模型架构](#模型架构)

---

## 项目结构

```
├── agent/                # 用户 Agent 系统
│   ├── user_profile.py           # 用户画像构建
│   ├── llm_agent.py              # LLM Agent（角色扮演 + 情绪分析）
│   ├── agent_factory.py          # Agent 工厂
│   └── agent_orchestrator.py     # Agent 编排器（辩论图生成）
├── gnn/                  # 图神经网络
│   ├── model.py                  # GNN 模型定义（3 层 GCN）
│   ├── dataset.py                # 图数据集
│   ├── trainer.py                # 训练器
│   ├── predictor.py              # 预测器
│   └── metrics.py                # 评估指标
├── features/             # 特征工程
│   ├── keyword_sentiment.py      # 关键词情绪分析
│   ├── text_embedding.py         # 文本嵌入
│   └── feature_pipeline.py       # 特征流水线（TF-IDF）
├── data_loader/          # 数据加载
│   ├── loader.py                 # JSONL 加载器
│   ├── preprocessor.py           # 文本预处理
│   └── graph_builder.py          # 图构建器（Agent 辩论图 → GNN 输入）
├── config.py             # 统一配置（模型超参、LLM、训练参数）
├── main.py               # 主入口
└── logger.py             # 日志模块
```

## 数据流

```
JSONL 文件（来自 binance-square-collector）
       ↓ data_loader/loader.py
用户帖子与评论数据（Conversation 对象列表）
       ↓ features/feature_pipeline.py
TF-IDF 文本特征矩阵
       ↓ agent/agent_orchestrator.py
Agent 情绪向量 [bullish, bearish, neutral, confidence]
       ↓ data_loader/graph_builder.py
GNN 输入图（节点=用户评论，边=回复关系）
       ↓ gnn/model.py → gnn/trainer.py
情绪预测（看涨 / 看跌）
       ↓ 与币价涨跌对比验证
最终评估
```

### 特征维度

每个图节点的输入特征由三部分拼接：

| 特征 | 维度 | 说明 |
|------|------|------|
| TF-IDF 文本特征 | 500 | 字符级 1-2 gram，min_df=2 |
| Agent 情绪向量 | 4 | [bullish, bearish, neutral, confidence] |
| 节点深度 | 1（可选） | 评论在对话树中的深度 |

## 环境配置

```bash
conda create -n sentiment python=3.10
conda activate sentiment
pip install torch torch-geometric dgl anthropic scikit-learn numpy
```

## 使用方法

### 训练

```bash
# 使用默认输入文件
python main.py --mode train

# 指定输入文件
python main.py --mode train --input path/to/parsed_data.jsonl
```

训练流程：
1. 加载 JSONL → 对话列表
2. 拟合 TF-IDF 特征
3. 构建用户画像 + LLM Agent 分析
4. 构建 Agent 辩论图
5. GNN 训练（早停，patience=30）

### 推理

```bash
# 分析指定帖子
python main.py --mode analyze --post-id 317353268392961

# 分析所有有标签的对话
python main.py --mode analyze
```

输出每个对话的预测结果（看涨/看跌概率）以及与真实标签的对比。

## 配置说明

所有可调参数集中在 [config.py](config.py) 中：

### 数据加载

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `DEFAULT_INPUT` | `dataset/result/parsed_28.jsonl` | 默认输入文件 |
| `MIN_COMMENT_LENGTH` | 2 | 最短评论文本长度 |

### TF-IDF 特征

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `TFIDF_MAX_FEATURES` | 500 | 最大特征数 |
| `TFIDF_NGRAM_RANGE` | (1, 2) | N-gram 范围 |
| `TFIDF_MIN_DF` | 2 | 最小文档频率 |

### Agent 情绪

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `AGENT_SENTIMENT_DIM` | 4 | 情绪向量维度 |

### GNN 模型

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `HIDDEN_DIM` | 256 | 第一层隐藏维度 |
| `HIDDEN_DIM2` | 128 | 第二层隐藏维度 |
| `HIDDEN_DIM3` | 64 | 第三层隐藏维度 |
| `MLP_HIDDEN` | 64 | MLP 分类器隐藏维度 |
| `DROPOUT` | 0.3 | Dropout 比例 |

### LLM Agent

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `LLM_MODEL` | `deepseek-v4-pro` | 模型名称 |
| `LLM_BASE_URL` | `https://api.deepseek.com/anthropic` | API 地址 |
| `LLM_MAX_TOKENS` | 256 | 最大生成 token |
| `LLM_TEMPERATURE` | 0.5 | 生成温度 |
| `LLM_MAX_RETRIES` | 2 | 最大重试次数 |

### 训练

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `RANDOM_SEED` | 42 | 随机种子 |
| `TRAIN_RATIO` | 0.7 | 训练集比例 |
| `BATCH_SIZE` | 32 | 批量大小 |
| `LEARNING_RATE` | 0.001 | 学习率 |
| `WEIGHT_DECAY` | 5e-4 | 权重衰减 |
| `MAX_EPOCHS` | 200 | 最大训练轮数 |
| `EARLY_STOP_PATIENCE` | 30 | 早停耐心值 |

## 模型架构

```
SentimentGCN
├── GCNConv (input_dim → 256) + ReLU + Dropout
├── GCNConv (256 → 128) + ReLU + Dropout
├── GCNConv (128 → 64) + ReLU + Dropout
├── GlobalMeanPool + GlobalMaxPool → (128)
└── MLP (128 → 64 → 1) + Sigmoid
```

- **图卷积层**：3 层 GCN，逐层降维
- **全局池化**：Mean + Max 拼接，捕获全局图结构
- **分类器**：两层 MLP，输出 [0, 1] 概率
- **> 0.5 = 看涨，< 0.5 = 看跌**

## 输入数据格式

输入 JSONL 每行一个帖子，需包含：
- `postId`：帖子 ID
- `author`：作者信息
- `content`：帖子正文
- `comments`：嵌套评论树（含 `author`、`content`、`timestamp`）
- `label`（可选）：真实标签（1=看涨, -1=看跌, null=无标签）

详细格式参见 binance-square-collector 的输出文档。

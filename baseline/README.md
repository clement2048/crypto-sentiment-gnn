# Baseline — 第一阶段三类基线

每个 baseline 一个独立 ipynb，互不依赖，方便单独调试。

## 文件对应表

| 文件 | 内容 | 默认 SMOKE 量 | 提速关键 |
| --- | --- | --- | --- |
| [01_vader_baseline.ipynb](01_vader_baseline.ipynb) | VADER 词典打分 → BULLISH/BEARISH | 10 个 block | 本地跑几乎秒级 |
| [02_sentencebert_baseline.ipynb](02_sentencebert_baseline.ipynb) | Sentence-BERT 嵌入 + LogReg | 10 个 block | 首次跑要下载 SBERT 权重 |
| [03_direct_llm_baseline.ipynb](03_direct_llm_baseline.ipynb) | SiliconFlow LLM 直接分类 | 5 个 block | 需要 `SILICONFLOW_API_KEY` |

每本 notebook 顶部都有一行 `SMOKE = True` 开关；本机验证完链路，把 `SMOKE = False` 改掉、`LIMIT_BLOCKS = None` 即可切全量。

## 顺序建议

```bash
"D:/anaconda/envs/sentiment/python.exe" -m pip install jupyter
"D:/anaconda/envs/sentiment/python.exe" -m jupyter notebook baseline/
```

1. 先跑 [01_vader_baseline.ipynb](01_vader_baseline.ipynb)：纯本地、最快，确认数据加载 + 评估链路没问题。
2. 再跑 [02_sentencebert_baseline.ipynb](02_sentencebert_baseline.ipynb)：首次会下载 Sentence-BERT 权重（约 90 MB），缓存在 `outputs/hf_cache/`。
3. 最后跑 [03_direct_llm_baseline.ipynb](03_direct_llm_baseline.ipynb)：需要 `.env` 里 `SILICONFLOW_API_KEY`；输出会按 payload SHA256 缓存到 `outputs/llm_cache/direct_llm/`，重复运行不重复计费。

## 切到全量

```python
SMOKE = False
LIMIT_BLOCKS = None  # 全量
```

每本 notebook 最后一个 markdown cell 都写了 `jupyter nbconvert --execute` 的无交互命令，服务器上用得上。

## 输出位置（烟雾版）

```
outputs/
├── vader_smoke/
├── sentencebert_smoke/
├── direct_llm_smoke/
├── embedding_cache/sentencebert/<sha256>.json
└── llm_cache/direct_llm/<sha256>.json
```

每本写一份 `metrics.json` + `predictions.jsonl`，metrics 里都包含 `accuracy` + `confusion_matrix` + 配置信息，方便三本横向对比。切到全量时把 `OUT_DIR` 改成不带 `_smoke` 后缀的路径即可。

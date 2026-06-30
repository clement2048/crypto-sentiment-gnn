"""一次性把三本 baseline notebook 切到全量并加 predictions.csv 输出。

做的事情：
  1. SMOKE = True -> SMOKE = False
  2. OUT_DIR 从 *_smoke 改成不带后缀
  3. 最后代码 cell 末尾追加 predictions.csv 写入（用 stdlib csv，字段名与各 notebook 已有 predictions 行一致）
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path("/home/tim/sentiment_analysis")
BASELINE_DIR = ROOT / "baseline"


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def save(path: Path, nb: dict) -> None:
    path.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")


def get_source(cell: dict) -> str:
    return "".join(cell["source"])


def set_source(cell: dict, text: str) -> None:
    cell["source"] = text.splitlines(keepends=True)


def find_cell_index(cells: list[dict], needle: str) -> int:
    for i, c in enumerate(cells):
        if c.get("cell_type") != "code":
            continue
        if needle in get_source(c):
            return i
    raise ValueError(f"cell containing {needle!r} not found")


# VADER ---------------------------------------------------------------------

vader = BASELINE_DIR / "01_vader_baseline.ipynb"
nb = load(vader)

# flip SMOKE flag
smoke_cell = nb["cells"][find_cell_index(nb["cells"], "SMOKE = True")]
src = get_source(smoke_cell)
assert "SMOKE = True" in src and "LIMIT_BLOCKS = 10 if SMOKE else None" in src
set_source(smoke_cell, src.replace("SMOKE = True", "SMOKE = False"))

# patch output cell: OUT_DIR + add CSV
out_cell = nb["cells"][find_cell_index(nb["cells"], "vader_smoke")]
src = get_source(out_cell)
src = src.replace('"outputs" / "vader_smoke"', '"outputs" / "vader"')
csv_append = '''\n# 写 predictions.csv（与 predictions.jsonl 字段一致）
import csv
with (OUT_DIR / "predictions.csv").open("w", encoding="utf-8", newline="") as fp:
    writer = csv.DictWriter(fp, fieldnames=list(rows[0].keys()))
    writer.writeheader()
    writer.writerows(rows)
print("wrote", OUT_DIR / "predictions.csv")
'''
set_source(out_cell, src + csv_append)
save(vader, nb)
print("patched:", vader.name)

# Sentence-BERT --------------------------------------------------------------

sbert = BASELINE_DIR / "02_sentencebert_baseline.ipynb"
nb = load(sbert)

smoke_cell = nb["cells"][find_cell_index(nb["cells"], "SMOKE = True")]
src = get_source(smoke_cell)
assert "SMOKE = True" in src and "LIMIT_BLOCKS = 10 if SMOKE else None" in src
set_source(smoke_cell, src.replace("SMOKE = True", "SMOKE = False"))

out_cell = nb["cells"][find_cell_index(nb["cells"], "sentencebert_smoke")]
src = get_source(out_cell)
src = src.replace('"outputs" / "sentencebert_smoke"', '"outputs" / "sentencebert"')
csv_append = '''\n# 写 predictions.csv（与 predictions.jsonl 字段一致）
import csv
with (OUT_DIR / "predictions.csv").open("w", encoding="utf-8", newline="") as fp:
    writer = csv.DictWriter(fp, fieldnames=list(rows[0].keys()))
    writer.writeheader()
    writer.writerows(rows)
print("wrote", OUT_DIR / "predictions.csv")
'''
set_source(out_cell, src + csv_append)
save(sbert, nb)
print("patched:", sbert.name)

# Direct LLM ----------------------------------------------------------------

direct = BASELINE_DIR / "03_direct_llm_baseline.ipynb"
nb = load(direct)

smoke_cell = nb["cells"][find_cell_index(nb["cells"], "SMOKE = True")]
src = get_source(smoke_cell)
assert "SMOKE = True" in src and "LIMIT_BLOCKS = 5 if SMOKE else None" in src
set_source(smoke_cell, src.replace("SMOKE = True", "SMOKE = False"))

out_cell = nb["cells"][find_cell_index(nb["cells"], "direct_llm_smoke")]
src = get_source(out_cell)
src = src.replace('"outputs" / "direct_llm_smoke"', '"outputs" / "direct_llm"')
csv_append = '''\n# 写 predictions.csv（与 predictions.jsonl 字段一致）
import csv
with (OUT_DIR / "predictions.csv").open("w", encoding="utf-8", newline="") as fp:
    writer = csv.DictWriter(fp, fieldnames=list(predictions[0].keys()))
    writer.writeheader()
    writer.writerows(predictions)
print("wrote", OUT_DIR / "predictions.csv")
'''
set_source(out_cell, src + csv_append)
save(direct, nb)
print("patched:", direct.name)

print("all three notebooks patched.")

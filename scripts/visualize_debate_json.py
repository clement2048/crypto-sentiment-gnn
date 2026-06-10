"""Generate a static HTML viewer for split-experiment debate JSON."""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path
from typing import Any


DEFAULT_INPUT = "outputs/split_9_3_3_deepseek.json"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Turn a split experiment JSON into a readable debate-process HTML report.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("input_json", nargs="?", default=DEFAULT_INPUT)
    parser.add_argument("--output-html", default=None)
    args = parser.parse_args()

    input_path = Path(args.input_json)
    output_path = Path(args.output_html) if args.output_html else input_path.with_suffix(".debate.html")

    data = json.loads(input_path.read_text(encoding="utf-8"))
    view_data = build_view_data(data, input_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(render_html(view_data), encoding="utf-8")
    print(f"Wrote debate visualization: {output_path}")


def build_view_data(data: dict[str, Any], input_path: Path) -> dict[str, Any]:
    records_by_split = data.get("records", {})
    flat_records: list[dict[str, Any]] = []
    for split_name in ("train", "val", "test"):
        for index, record in enumerate(records_by_split.get(split_name, []), start=1):
            flat_records.append(summarize_record(record, split_name, index))

    return {
        "source": str(input_path),
        "config": data.get("config", {}),
        "train_losses": data.get("train_losses", []),
        "metrics": data.get("metrics", {}),
        "records": flat_records,
        "totals": {
            "records": len(flat_records),
            "arguments": sum(len(item["arguments"]) for item in flat_records),
            "bull": sum(item["camp_counts"].get("bull", 0) for item in flat_records),
            "bear": sum(item["camp_counts"].get("bear", 0) for item in flat_records),
        },
    }


def summarize_record(record: dict[str, Any], split_name: str, index: int) -> dict[str, Any]:
    block = record.get("block", {})
    debate = record.get("debate", {})
    judge = record.get("judge", {})
    model = record.get("model_summary", {})
    graph = record.get("graph", {})
    arguments = debate.get("arguments", [])
    edges = graph.get("edges", [])
    camp_counts = {
        "bull": sum(1 for item in arguments if item.get("camp") == "bull"),
        "bear": sum(1 for item in arguments if item.get("camp") == "bear"),
    }
    true_label = label_name(block.get("label"))
    pred_label = judge.get("verdict", "UNKNOWN")
    return {
        "split": split_name,
        "index": index,
        "block_id": block.get("block_id", ""),
        "post_id": block.get("post_id", ""),
        "product": block.get("product") or ", ".join(block.get("products", [])),
        "post_content": block.get("post_content", ""),
        "root_comment": block.get("root_comment", {}),
        "p0": block.get("p0"),
        "p1": block.get("p1"),
        "true_label": true_label,
        "pred_label": pred_label,
        "is_correct": verdict_matches_label(pred_label, block.get("label")),
        "rounds": debate.get("rounds", 0),
        "arguments": arguments,
        "camp_counts": camp_counts,
        "relations": relation_counts(edges),
        "judge": judge,
        "model_summary": model,
    }


def relation_counts(edges: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for edge in edges:
        relation = str(edge.get("relation") or "unknown")
        counts[relation] = counts.get(relation, 0) + 1
    return counts


def label_name(value: Any) -> str:
    if value == 1:
        return "BULLISH"
    if value == -1:
        return "BEARISH"
    return "NEUTRAL"


def verdict_matches_label(verdict: Any, label: Any) -> bool:
    return str(verdict).upper() == label_name(label)


def render_html(view_data: dict[str, Any]) -> str:
    payload = json.dumps(view_data, ensure_ascii=False)
    escaped_title = html.escape(f"Debate Viewer - {view_data['source']}")
    return HTML_TEMPLATE.replace("__TITLE__", escaped_title).replace("__DATA__", payload)


HTML_TEMPLATE = r"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>__TITLE__</title>
  <style>
    :root {
      --bg: #f6f7f9;
      --panel: #ffffff;
      --ink: #20242c;
      --muted: #667085;
      --line: #d8dde6;
      --bull: #13795b;
      --bear: #b42318;
      --neutral: #5f6b7a;
      --accent: #2458d3;
      --soft-bull: #e7f6ef;
      --soft-bear: #fdebea;
      --soft-blue: #e9efff;
      font-family: "Microsoft YaHei", "PingFang SC", "Segoe UI", sans-serif;
    }
    * { box-sizing: border-box; }
    body { margin: 0; background: var(--bg); color: var(--ink); }
    header {
      position: sticky; top: 0; z-index: 5; background: rgba(246, 247, 249, .96);
      border-bottom: 1px solid var(--line); backdrop-filter: blur(10px);
    }
    .wrap { max-width: 1440px; margin: 0 auto; padding: 18px 24px; }
    .topline { display: flex; align-items: center; justify-content: space-between; gap: 16px; }
    h1 { font-size: 22px; line-height: 1.2; margin: 0 0 4px; }
    .sub { color: var(--muted); font-size: 13px; overflow-wrap: anywhere; }
    .controls { display: flex; flex-wrap: wrap; gap: 8px; justify-content: flex-end; }
    select, input {
      height: 36px; border: 1px solid var(--line); border-radius: 6px; padding: 0 10px;
      background: white; color: var(--ink); min-width: 120px;
    }
    input { min-width: 240px; }
    main.wrap { display: grid; grid-template-columns: 310px minmax(0, 1fr); gap: 16px; }
    .panel { background: var(--panel); border: 1px solid var(--line); border-radius: 8px; }
    .sidebar { position: sticky; top: 90px; align-self: start; max-height: calc(100vh - 110px); overflow: auto; }
    .summary { padding: 14px; border-bottom: 1px solid var(--line); }
    .summary-grid { display: grid; grid-template-columns: repeat(2, 1fr); gap: 8px; margin-top: 10px; }
    .stat { border: 1px solid var(--line); border-radius: 6px; padding: 9px; background: #fbfcfe; }
    .stat b { display: block; font-size: 18px; }
    .stat span { color: var(--muted); font-size: 12px; }
    .record-list { display: flex; flex-direction: column; }
    .record-button {
      width: 100%; text-align: left; border: 0; border-bottom: 1px solid var(--line);
      background: white; padding: 12px 14px; cursor: pointer; color: var(--ink);
    }
    .record-button:hover, .record-button.active { background: #f0f4ff; }
    .row { display: flex; align-items: center; justify-content: space-between; gap: 8px; }
    .id { font-family: Consolas, monospace; font-size: 12px; overflow-wrap: anywhere; }
    .badge {
      display: inline-flex; align-items: center; justify-content: center; height: 22px;
      border-radius: 999px; padding: 0 8px; font-size: 12px; font-weight: 700;
      border: 1px solid transparent; white-space: nowrap;
    }
    .badge.bull { color: var(--bull); background: var(--soft-bull); border-color: #abdcc7; }
    .badge.bear { color: var(--bear); background: var(--soft-bear); border-color: #f4b4ae; }
    .badge.neutral { color: var(--neutral); background: #eef1f5; border-color: var(--line); }
    .badge.ok { color: #05603a; background: #ecfdf3; }
    .badge.bad { color: #b42318; background: #fef3f2; }
    .detail { display: grid; gap: 16px; }
    .hero { padding: 18px; }
    .hero h2 { margin: 0 0 10px; font-size: 20px; }
    .meta { display: flex; flex-wrap: wrap; gap: 8px; margin: 10px 0; }
    .text-block { line-height: 1.65; color: #303744; }
    .section-title { font-size: 16px; margin: 0 0 12px; }
    .grid-3 { display: grid; grid-template-columns: repeat(3, 1fr); gap: 12px; }
    .grid-2 { display: grid; grid-template-columns: repeat(2, 1fr); gap: 16px; }
    .mini { padding: 14px; }
    .bar { height: 9px; border-radius: 999px; background: #edf0f5; overflow: hidden; }
    .bar > i { display: block; height: 100%; background: var(--accent); }
    .bar.bull > i { background: var(--bull); }
    .bar.bear > i { background: var(--bear); }
    .score-row { display: grid; grid-template-columns: 70px minmax(80px, 1fr) 44px; gap: 8px; align-items: center; margin: 7px 0; font-size: 13px; }
    .timeline { padding: 18px; }
    .lanes { display: grid; grid-template-columns: 1fr 1fr; gap: 14px; align-items: start; }
    .lane-title { display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px; font-weight: 800; }
    .arg-card { border: 1px solid var(--line); border-left-width: 5px; border-radius: 8px; background: white; padding: 12px; margin-bottom: 12px; }
    .arg-card.bull { border-left-color: var(--bull); }
    .arg-card.bear { border-left-color: var(--bear); }
    .arg-head { display: flex; justify-content: space-between; gap: 10px; margin-bottom: 8px; }
    .role { font-weight: 800; font-size: 14px; }
    .claim { line-height: 1.65; margin: 8px 0 10px; }
    details { border-top: 1px solid var(--line); padding-top: 8px; }
    summary { cursor: pointer; color: var(--accent); font-weight: 700; font-size: 13px; }
    .evidence { margin: 8px 0 0; padding: 8px; border-radius: 6px; background: #f7f8fb; }
    .evidence small { color: var(--muted); display: block; margin-bottom: 4px; }
    .empty { padding: 24px; color: var(--muted); text-align: center; }
    @media (max-width: 980px) {
      main.wrap { grid-template-columns: 1fr; }
      .sidebar { position: static; max-height: none; }
      .grid-3, .grid-2, .lanes { grid-template-columns: 1fr; }
      .topline { align-items: stretch; flex-direction: column; }
      .controls { justify-content: flex-start; }
      input { min-width: 100%; }
    }
  </style>
</head>
<body>
  <header>
    <div class="wrap topline">
      <div>
        <h1>辩论过程可视化</h1>
        <div class="sub" id="source"></div>
      </div>
      <div class="controls">
        <select id="splitFilter">
          <option value="all">全部 split</option>
          <option value="train">train</option>
          <option value="val">val</option>
          <option value="test">test</option>
        </select>
        <select id="resultFilter">
          <option value="all">全部结果</option>
          <option value="correct">预测正确</option>
          <option value="wrong">预测错误</option>
        </select>
        <input id="searchBox" placeholder="搜索 block、产品、评论、论点">
      </div>
    </div>
  </header>
  <main class="wrap">
    <aside class="panel sidebar">
      <div class="summary">
        <div class="sub">总览</div>
        <div class="summary-grid" id="summaryGrid"></div>
      </div>
      <div class="record-list" id="recordList"></div>
    </aside>
    <section class="detail" id="detail"></section>
  </main>
  <script>
    const DATA = __DATA__;
    const state = { selected: null, filtered: DATA.records };
    const $ = (id) => document.getElementById(id);
    const fmt = (value, digits = 3) => Number.isFinite(Number(value)) ? Number(value).toFixed(digits) : "n/a";
    const pct = (value) => Number.isFinite(Number(value)) ? `${Math.round(Number(value) * 100)}%` : "n/a";
    const esc = (value) => String(value ?? "").replace(/[&<>"']/g, ch => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[ch]));
    const verdictClass = (value) => String(value).toLowerCase().includes("bull") ? "bull" : String(value).toLowerCase().includes("bear") ? "bear" : "neutral";

    function init() {
      $("source").textContent = `${DATA.source} · ${DATA.totals.records} blocks · ${DATA.totals.arguments} arguments`;
      renderSummary();
      applyFilters();
      $("splitFilter").addEventListener("change", applyFilters);
      $("resultFilter").addEventListener("change", applyFilters);
      $("searchBox").addEventListener("input", applyFilters);
    }

    function renderSummary() {
      const metrics = DATA.metrics || {};
      const test = metrics.test || {};
      $("summaryGrid").innerHTML = [
        stat(DATA.totals.records, "blocks"),
        stat(DATA.totals.arguments, "arguments"),
        stat(fmt(test.accuracy), "test accuracy"),
        stat(fmt(test.macro_f1), "test macro F1"),
      ].join("");
    }

    function stat(value, label) {
      return `<div class="stat"><b>${esc(value)}</b><span>${esc(label)}</span></div>`;
    }

    function applyFilters() {
      const split = $("splitFilter").value;
      const result = $("resultFilter").value;
      const query = $("searchBox").value.trim().toLowerCase();
      state.filtered = DATA.records.filter(rec => {
        if (split !== "all" && rec.split !== split) return false;
        if (result === "correct" && !rec.is_correct) return false;
        if (result === "wrong" && rec.is_correct) return false;
        if (!query) return true;
        const hay = [rec.block_id, rec.product, rec.post_content, rec.root_comment?.text, ...rec.arguments.map(a => a.claim)].join(" ").toLowerCase();
        return hay.includes(query);
      });
      if (!state.filtered.includes(state.selected)) state.selected = state.filtered[0] || null;
      renderList();
      renderDetail();
    }

    function renderList() {
      if (!state.filtered.length) {
        $("recordList").innerHTML = `<div class="empty">没有匹配的 block</div>`;
        return;
      }
      $("recordList").innerHTML = state.filtered.map((rec, idx) => `
        <button class="record-button ${rec === state.selected ? "active" : ""}" data-idx="${idx}">
          <div class="row">
            <span class="badge neutral">${esc(rec.split)} #${rec.index}</span>
            <span class="badge ${rec.is_correct ? "ok" : "bad"}">${rec.is_correct ? "OK" : "MISS"}</span>
          </div>
          <div class="id" style="margin-top:7px">${esc(rec.block_id)}</div>
          <div class="row" style="margin-top:7px">
            <span>${esc(rec.product || "unknown")}</span>
            <span class="badge ${verdictClass(rec.pred_label)}">${esc(rec.pred_label)}</span>
          </div>
        </button>
      `).join("");
      document.querySelectorAll(".record-button").forEach(button => {
        button.addEventListener("click", () => {
          state.selected = state.filtered[Number(button.dataset.idx)];
          renderList();
          renderDetail();
        });
      });
    }

    function renderDetail() {
      const rec = state.selected;
      if (!rec) {
        $("detail").innerHTML = `<div class="panel empty">请选择一个 block</div>`;
        return;
      }
      const judge = rec.judge || {};
      const scores = judge.score_vector || {};
      const model = rec.model_summary || {};
      $("detail").innerHTML = `
        <div class="panel hero">
          <div class="row">
            <h2>${esc(rec.product || "unknown")} · ${esc(rec.block_id)}</h2>
            <span class="badge ${rec.is_correct ? "ok" : "bad"}">真实 ${esc(rec.true_label)} / 预测 ${esc(rec.pred_label)}</span>
          </div>
          <div class="meta">
            <span class="badge neutral">p0 ${esc(rec.p0)}</span>
            <span class="badge neutral">p1 ${esc(rec.p1)}</span>
            <span class="badge neutral">${esc(rec.rounds)} round</span>
            <span class="badge bull">bull ${rec.camp_counts.bull}</span>
            <span class="badge bear">bear ${rec.camp_counts.bear}</span>
          </div>
          <div class="text-block"><b>新闻/帖子：</b>${esc(rec.post_content)}</div>
          <div class="text-block" style="margin-top:8px"><b>根评论：</b>${esc(rec.root_comment?.author)}：${esc(rec.root_comment?.text)}</div>
        </div>
        <div class="grid-3">
          ${miniPanel("裁判结论", `
            <div class="meta"><span class="badge ${verdictClass(judge.verdict)}">${esc(judge.verdict)}</span><span class="badge neutral">confidence ${fmt(judge.confidence)}</span></div>
            <div class="text-block">${esc(judge.report)}</div>
          `)}
          ${miniPanel("模型信号", `
            ${scoreBar("bull prob", model.bullish_probability, "bull")}
            ${scoreBar("margin", (Number(model.bull_bear_margin) + 1) / 2, Number(model.bull_bear_margin) >= 0 ? "bull" : "bear", fmt(model.bull_bear_margin))}
            <div class="sub">predicted_label: ${esc(model.predicted_label)} · ode_steps: ${esc(model.ode_steps)}</div>
          `)}
          ${miniPanel("裁判分数", Object.entries(scores).map(([k, v]) => scoreBar(k, v, k.includes("bear") ? "bear" : k.includes("bull") ? "bull" : "", fmt(v))).join(""))}
        </div>
        <div class="panel timeline">
          <h3 class="section-title">发言时间线</h3>
          <div class="lanes">
            ${lane(rec, "bull", "看涨方")}
            ${lane(rec, "bear", "看跌方")}
          </div>
        </div>
      `;
    }

    function miniPanel(title, body) {
      return `<div class="panel mini"><h3 class="section-title">${esc(title)}</h3>${body}</div>`;
    }

    function scoreBar(label, value, cls = "", shown = null) {
      const n = Number.isFinite(Number(value)) ? Math.max(0, Math.min(1, Number(value))) : 0;
      return `<div class="score-row"><span>${esc(label)}</span><div class="bar ${cls}"><i style="width:${n * 100}%"></i></div><b>${esc(shown ?? pct(value))}</b></div>`;
    }

    function lane(rec, camp, title) {
      const args = rec.arguments.filter(arg => arg.camp === camp);
      return `<div>
        <div class="lane-title"><span>${title}</span><span class="badge ${camp}">${args.length}</span></div>
        ${args.map(argCard).join("")}
      </div>`;
    }

    function argCard(arg) {
      const evidence = (arg.evidence || []).map(ev => `
        <div class="evidence">
          <small>${esc(ev.source_type)} · ${esc(ev.source_id)} · relevance ${fmt(ev.relevance)}</small>
          ${esc(ev.quote)}
        </div>
      `).join("");
      return `<article class="arg-card ${esc(arg.camp)}">
        <div class="arg-head">
          <div><div class="role">#${esc(arg.seq)} ${esc(arg.role)}</div><div class="sub">${esc(arg.agent_id)}</div></div>
          <span class="badge ${esc(arg.camp)}">${fmt(arg.confidence)}</span>
        </div>
        <div class="claim">${esc(arg.claim)}</div>
        <details>
          <summary>证据 ${arg.evidence?.length || 0} 条 · targets ${arg.targets?.length || 0}</summary>
          ${evidence || `<div class="evidence">没有证据条目</div>`}
        </details>
      </article>`;
    }

    init();
  </script>
</body>
</html>
"""


if __name__ == "__main__":
    main()

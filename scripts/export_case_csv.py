"""Export case-study JSON into readable CSV tables."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any


def export_case_csv(input_json: str, output_dir: str) -> dict[str, Path]:
    """把 case-study JSON 拆成多张 CSV。"""
    input_path = Path(input_json)
    data = json.loads(input_path.read_text(encoding="utf-8"))
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    paths = {
        "comments": out_dir / "comments.csv",
        "block_judges": out_dir / "block_judges.csv",
        "arguments": out_dir / "arguments.csv",
        "evidence": out_dir / "evidence.csv",
        "graph_edges": out_dir / "graph_edges.csv",
    }
    _write_comments(paths["comments"], data)
    _write_block_judges(paths["block_judges"], data)
    _write_arguments(paths["arguments"], data)
    _write_evidence(paths["evidence"], data)
    _write_graph_edges(paths["graph_edges"], data)
    return paths


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(
        description="Export case-study JSON into CSV files.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input-json", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    paths = export_case_csv(args.input_json, args.output_dir)
    for name, path in paths.items():
        print(f"{name}: {path}")


def _write_comments(path: Path, data: dict[str, Any]) -> None:
    post = data["post"]
    rows = []
    for comment in post.get("comments", []):
        rows.append(
            {
                "post_id": post.get("post_id"),
                "product": post.get("first_product"),
                "post_time": post.get("post_time"),
                "comment_id": comment.get("comment_id"),
                "author": comment.get("author"),
                "comment_time": comment.get("post_time"),
                "text": comment.get("text"),
            }
        )
    _write_rows(path, rows, ["post_id", "product", "post_time", "comment_id", "author", "comment_time", "text"])


def _write_block_judges(path: Path, data: dict[str, Any]) -> None:
    rows = []
    for record in data.get("records", []):
        block = record["block"]
        judge = record["judge"]
        model = record["model_summary"]
        score = judge.get("score_vector", {})
        rows.append(
            {
                "block_id": block.get("block_id"),
                "post_id": block.get("post_id"),
                "root_comment_id": block.get("root_comment", {}).get("comment_id"),
                "root_text": block.get("root_comment", {}).get("text"),
                "true_label": block.get("label"),
                "t0": block.get("t0"),
                "product": block.get("product"),
                "verdict": judge.get("verdict"),
                "confidence": judge.get("confidence"),
                "report": judge.get("report"),
                "model_bullish_probability": model.get("bullish_probability"),
                "model_predicted_label": model.get("predicted_label"),
                "ode_bull_bear_margin": model.get("bull_bear_margin"),
                "p_bull": score.get("p_bull"),
                "p_bear": score.get("p_bear"),
                "q_bull": score.get("q_bull"),
                "q_bear": score.get("q_bear"),
                "e_bull": score.get("e_bull"),
                "e_bear": score.get("e_bear"),
                "coverage_c": score.get("c"),
                "depth_d": score.get("d"),
                "attack_a": score.get("a"),
                "rho": score.get("rho"),
                "consistency_flags": ";".join(judge.get("consistency_flags", [])),
            }
        )
    _write_rows(
        path,
        rows,
        [
            "block_id",
            "post_id",
            "root_comment_id",
            "root_text",
            "true_label",
            "t0",
            "product",
            "verdict",
            "confidence",
            "report",
            "model_bullish_probability",
            "model_predicted_label",
            "ode_bull_bear_margin",
            "p_bull",
            "p_bear",
            "q_bull",
            "q_bear",
            "e_bull",
            "e_bear",
            "coverage_c",
            "depth_d",
            "attack_a",
            "rho",
            "consistency_flags",
        ],
    )


def _write_arguments(path: Path, data: dict[str, Any]) -> None:
    rows = []
    for record in data.get("records", []):
        block_id = record["block"]["block_id"]
        for argument in record["debate"].get("arguments", []):
            rows.append(
                {
                    "block_id": block_id,
                    "argument_id": argument.get("argument_id"),
                    "seq": argument.get("seq"),
                    "round": argument.get("round"),
                    "camp": argument.get("camp"),
                    "role": argument.get("role"),
                    "agent_id": argument.get("agent_id"),
                    "confidence": argument.get("confidence"),
                    "targets": ";".join(argument.get("targets", [])),
                    "cited_comment_ids": ";".join(argument.get("cited_comment_ids", [])),
                    "claim": argument.get("claim"),
                }
            )
    _write_rows(
        path,
        rows,
        [
            "block_id",
            "argument_id",
            "seq",
            "round",
            "camp",
            "role",
            "agent_id",
            "confidence",
            "targets",
            "cited_comment_ids",
            "claim",
        ],
    )


def _write_evidence(path: Path, data: dict[str, Any]) -> None:
    rows = []
    for record in data.get("records", []):
        block_id = record["block"]["block_id"]
        for argument in record["debate"].get("arguments", []):
            for index, evidence in enumerate(argument.get("evidence", []), start=1):
                rows.append(
                    {
                        "block_id": block_id,
                        "argument_id": argument.get("argument_id"),
                        "seq": argument.get("seq"),
                        "camp": argument.get("camp"),
                        "role": argument.get("role"),
                        "evidence_index": index,
                        "source_type": evidence.get("source_type"),
                        "source_id": evidence.get("source_id"),
                        "relevance": evidence.get("relevance"),
                        "quote": evidence.get("quote"),
                    }
                )
    _write_rows(
        path,
        rows,
        [
            "block_id",
            "argument_id",
            "seq",
            "camp",
            "role",
            "evidence_index",
            "source_type",
            "source_id",
            "relevance",
            "quote",
        ],
    )


def _write_graph_edges(path: Path, data: dict[str, Any]) -> None:
    rows = []
    for record in data.get("records", []):
        block_id = record["block"]["block_id"]
        for edge in record["graph"].get("edges", []):
            rows.append(
                {
                    "block_id": block_id,
                    "source": edge.get("source"),
                    "target": edge.get("target"),
                    "relation": edge.get("relation"),
                    "weight": edge.get("weight"),
                }
            )
    _write_rows(path, rows, ["block_id", "source", "target", "relation", "weight"])


def _write_rows(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

from config import (
    DEFAULT_DEBATE_ROUNDS,
    DEFAULT_LIMIT_BLOCKS,
    FULL_PIPELINE_TRAIN_EPOCHS,
    LEARNING_RATE,
    PRINT_SAMPLES,
    TRAIN_PROTOTYPE_EPOCHS,
    TRAIN_PROTOTYPE_LIMIT_BLOCKS,
)
from data import build_comment_blocks, load_posts, temporal_split_blocks
from profiles import ProfileStore
from scripts.build_graphs import build_graph_records
from scripts.evaluate_pipeline import evaluate_pipeline
from scripts.run_case_study import render_case_markdown, run_case_study
from scripts.run_debate import DEFAULT_INPUT, run_debate_pipeline
from scripts.run_full_pipeline import run_full_pipeline
from scripts.run_split_experiment import run_split_experiment
from scripts.train_prototype import train_prototype


def main() -> None:
    """解析命令行参数，并分发到对应阶段。"""
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(
        description="Crypto sentiment analysis v2 prototype",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # blocks：最早的数据阶段，只检查 JSONL -> CommentBlock 是否正确。
    blocks = subparsers.add_parser("blocks", help="Build CommentBlock samples from JSONL.")
    _add_input_arg(blocks)
    blocks.add_argument("--limit", type=int, default=None)
    blocks.add_argument("--print-samples", type=int, default=PRINT_SAMPLES)
    blocks.add_argument("--output-jsonl", type=str, default=None)
    blocks.set_defaults(func=_cmd_blocks)

    # debate：只生成辩论，不调用模型和法官。用于检查 Agent 输出结构。
    debate = subparsers.add_parser("debate", help="Run online multi-agent debate.")
    _add_input_arg(debate)
    debate.add_argument("--limit-blocks", type=int, default=DEFAULT_LIMIT_BLOCKS)
    debate.add_argument("--rounds", type=int, default=DEFAULT_DEBATE_ROUNDS)
    debate.add_argument("--mode", choices=["deepseek", "bailian", "siliconflow"], default="siliconflow")
    debate.add_argument("--output-jsonl", type=str, default=None)
    debate.set_defaults(func=_cmd_debate)

    # graphs：把评论树和辩论论点融合成图。用于检查边类型和节点数量。
    graphs = subparsers.add_parser("graphs", help="Build heterogeneous debate/comment graphs.")
    _add_input_arg(graphs)
    graphs.add_argument("--limit-blocks", type=int, default=DEFAULT_LIMIT_BLOCKS)
    graphs.add_argument("--rounds", type=int, default=DEFAULT_DEBATE_ROUNDS)
    graphs.add_argument("--mode", choices=["deepseek", "bailian", "siliconflow"], default="siliconflow")
    graphs.add_argument("--output-jsonl", type=str, default=None)
    graphs.set_defaults(func=_cmd_graphs)

    # train-prototype：最小训练 smoke，证明当前模型能前向和反向传播。
    train = subparsers.add_parser("train-prototype", help="Smoke-train the minimal graph model.")
    _add_input_arg(train)
    train.add_argument("--limit-blocks", type=int, default=TRAIN_PROTOTYPE_LIMIT_BLOCKS)
    train.add_argument("--rounds", type=int, default=DEFAULT_DEBATE_ROUNDS)
    train.add_argument("--epochs", type=int, default=TRAIN_PROTOTYPE_EPOCHS)
    train.add_argument("--learning-rate", type=float, default=LEARNING_RATE)
    train.set_defaults(func=_cmd_train_prototype)

    # full：当前最完整的原型链路，最后由 model-aware judge 输出 JudgeOutput。
    full = subparsers.add_parser("full", help="Run debate -> graph -> model summary -> judge.")
    _add_input_arg(full)
    full.add_argument("--limit-blocks", type=int, default=DEFAULT_LIMIT_BLOCKS)
    full.add_argument("--rounds", type=int, default=DEFAULT_DEBATE_ROUNDS)
    full.add_argument("--train-epochs", type=int, default=FULL_PIPELINE_TRAIN_EPOCHS)
    full.add_argument("--learning-rate", type=float, default=LEARNING_RATE)
    full.add_argument("--debate-mode", choices=["deepseek", "bailian", "siliconflow"], default="siliconflow")
    full.add_argument("--judge-mode", choices=["deepseek", "bailian", "siliconflow"], default="siliconflow")
    full.add_argument("--output-jsonl", type=str, default=None)
    full.set_defaults(func=_cmd_full)

    # evaluate：默认对全部 CommentBlock 跑完整链路，并计算最终验证指标。
    evaluate = subparsers.add_parser("evaluate", help="Evaluate judge predictions against CommentBlock labels.")
    _add_input_arg(evaluate)
    evaluate.add_argument("--limit-blocks", type=int, default=None, help="Limit samples; omit for all blocks.")
    evaluate.add_argument("--rounds", type=int, default=DEFAULT_DEBATE_ROUNDS)
    evaluate.add_argument("--train-epochs", type=int, default=FULL_PIPELINE_TRAIN_EPOCHS)
    evaluate.add_argument("--learning-rate", type=float, default=LEARNING_RATE)
    evaluate.add_argument("--debate-mode", choices=["deepseek", "bailian", "siliconflow"], default="siliconflow")
    evaluate.add_argument("--judge-mode", choices=["deepseek", "bailian", "siliconflow"], default="siliconflow")
    evaluate.add_argument("--output-jsonl", type=str, default=None)
    evaluate.add_argument("--metrics-json", type=str, default=None)
    evaluate.set_defaults(func=_cmd_evaluate)

    # split-experiment：按时间顺序做固定数量 train/val/test 小实验，例如 9:3:3。
    experiment = subparsers.add_parser("split-experiment", help="Run chronological train/val/test experiment.")
    _add_input_arg(experiment)
    experiment.add_argument("--train-count", type=int, default=9)
    experiment.add_argument("--val-count", type=int, default=3)
    experiment.add_argument("--test-count", type=int, default=3)
    experiment.add_argument("--rounds", type=int, default=1)
    experiment.add_argument("--epochs", type=int, default=5)
    experiment.add_argument("--learning-rate", type=float, default=LEARNING_RATE)
    experiment.add_argument("--debate-mode", choices=["deepseek", "bailian", "siliconflow"], default="siliconflow")
    experiment.add_argument("--judge-mode", choices=["deepseek", "bailian", "siliconflow"], default="siliconflow")
    experiment.add_argument("--seed", type=int, default=42)
    experiment.add_argument("--output-json", type=str, default=None)
    experiment.set_defaults(func=_cmd_split_experiment)

    # case-study：选一个评论较多的帖子，生成可阅读的辩论过程报告。
    case_study = subparsers.add_parser("case-study", help="Run and render a readable debate case study.")
    _add_input_arg(case_study)
    case_study.add_argument("--post-id", default=None)
    case_study.add_argument("--block-id", default=None)
    case_study.add_argument("--max-blocks", type=int, default=None)
    case_study.add_argument("--rounds", type=int, default=1)
    case_study.add_argument("--debate-mode", choices=["deepseek", "bailian", "siliconflow"], default="deepseek")
    case_study.add_argument("--judge-mode", choices=["deepseek", "bailian", "siliconflow"], default="siliconflow")
    case_study.add_argument("--seed", type=int, default=42)
    case_study.add_argument("--output-json", default=None)
    case_study.add_argument("--output-md", default=None)
    case_study.set_defaults(func=_cmd_case_study)

    args = parser.parse_args()
    args.func(args)


def _add_input_arg(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--input", default=DEFAULT_INPUT, help="JSONL file, directory, or glob pattern.")


def _cmd_blocks(args: argparse.Namespace) -> None:
    """加载 JSONL，打印评论块、过滤原因和时间切分统计。"""
    posts = load_posts(args.input)
    if args.limit is not None:
        posts = posts[: args.limit]
    blocks, issues = build_comment_blocks(posts)
    splits = temporal_split_blocks(blocks)
    profile_store = ProfileStore.from_blocks(blocks)

    print(f"Loaded posts: {len(posts)}")
    print(f"Built blocks: {len(blocks)}")
    print(f"Filter issues: {len(issues)}")
    if issues:
        counts = Counter(issue.reason for issue in issues)
        print(f"Filter issue counts: {dict(sorted(counts.items()))}")
    print(f"Temporal split: train={len(splits.train)} val={len(splits.val)} test={len(splits.test)}")

    for block in blocks[: min(args.print_samples, len(blocks))]:
        profiles = profile_store.get_profiles_for_block(block)
        root_text = block.root_comment.text.replace("\n", " ")[:100]
        print(
            f"- {block.block_id} | t0={block.t0:%Y-%m-%d %H:%M:%S} "
            f"| label={block.label} | product={block.product} "
            f"| profiles={len(profiles)} | text={root_text}"
        )

    if args.output_jsonl:
        _write_jsonl(args.output_jsonl, [block.to_dict() for block in blocks])


def _cmd_debate(args: argparse.Namespace) -> None:
    """只运行辩论；这里不会调用法官。"""
    records = run_debate_pipeline(args.input, limit_blocks=args.limit_blocks, rounds=args.rounds, mode=args.mode)
    print(f"Debate records: {len(records)}")
    for record in records[: min(len(records), PRINT_SAMPLES)]:
        block = record["block"]
        debate = record["debate"]
        assert isinstance(block, dict)
        assert isinstance(debate, dict)
        print(f"- {block['block_id']} | arguments={len(debate['arguments'])}")
    if args.output_jsonl:
        _write_jsonl(args.output_jsonl, records)


def _cmd_graphs(args: argparse.Namespace) -> None:
    """从辩论结果构建异构图。"""
    records = build_graph_records(args.input, limit_blocks=args.limit_blocks, rounds=args.rounds, mode=args.mode)
    print(f"Graph records: {len(records)}")
    for record in records[: min(len(records), PRINT_SAMPLES)]:
        graph = record["graph"]
        assert isinstance(graph, dict)
        print(
            f"- {record['block_id']} | nodes={len(graph['nodes'])} "
            f"| edges={len(graph['edges'])} | relations={graph['relation_counts']}"
        )
    if args.output_jsonl:
        _write_jsonl(args.output_jsonl, records)


def _cmd_train_prototype(args: argparse.Namespace) -> None:
    """训练几轮原型模型，只用于验证链路可训练。"""
    metrics = train_prototype(
        input_path=args.input,
        limit_blocks=args.limit_blocks,
        rounds=args.rounds,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
    )
    print(
        "Prototype training complete: "
        f"graphs={metrics['graphs']:.0f} "
        f"final_loss={metrics['final_loss']:.4f} "
        f"mean_probability={metrics['mean_probability']:.4f}"
    )


def _cmd_full(args: argparse.Namespace) -> None:
    """运行完整原型流程：辩论 -> 图 -> 模型摘要 -> 法官。"""
    records = run_full_pipeline(
        input_path=args.input,
        limit_blocks=args.limit_blocks,
        rounds=args.rounds,
        train_epochs=args.train_epochs,
        learning_rate=args.learning_rate,
        debate_mode=args.debate_mode,
        judge_mode=args.judge_mode,
    )
    print(f"Full pipeline records: {len(records)}")
    for record in records[: min(len(records), PRINT_SAMPLES)]:
        block = record["block"]
        model_summary = record["model_summary"]
        judge = record["judge"]
        assert isinstance(block, dict)
        assert isinstance(model_summary, dict)
        assert isinstance(judge, dict)
        print(
            f"- {block['block_id']} | model_prob={model_summary['bullish_probability']:.3f} "
            f"| verdict={judge['verdict']} | confidence={judge['confidence']:.3f}"
        )
    if args.output_jsonl:
        _write_jsonl(args.output_jsonl, records)


def _cmd_evaluate(args: argparse.Namespace) -> None:
    """运行完整 pipeline，并计算 accuracy/precision/recall/F1 等验证指标。"""
    records, metrics = evaluate_pipeline(
        input_path=args.input,
        limit_blocks=args.limit_blocks,
        rounds=args.rounds,
        train_epochs=args.train_epochs,
        learning_rate=args.learning_rate,
        debate_mode=args.debate_mode,
        judge_mode=args.judge_mode,
    )
    print(f"Evaluated samples: {metrics.total}")
    print(f"Accuracy: {metrics.accuracy:.4f}")
    print(f"Coverage(non-neutral): {metrics.coverage:.4f}")
    if metrics.directional_accuracy is None:
        print("Directional accuracy(non-neutral only): N/A")
    else:
        print(f"Directional accuracy(non-neutral only): {metrics.directional_accuracy:.4f}")
    print(
        "Macro: "
        f"precision={metrics.macro_precision:.4f} "
        f"recall={metrics.macro_recall:.4f} "
        f"f1={metrics.macro_f1:.4f}"
    )
    print(
        "Bullish: "
        f"precision={metrics.bullish.precision:.4f} "
        f"recall={metrics.bullish.recall:.4f} "
        f"f1={metrics.bullish.f1:.4f} "
        f"support={metrics.bullish.support}"
    )
    print(
        "Bearish: "
        f"precision={metrics.bearish.precision:.4f} "
        f"recall={metrics.bearish.recall:.4f} "
        f"f1={metrics.bearish.f1:.4f} "
        f"support={metrics.bearish.support}"
    )
    print(f"Confusion matrix: {metrics.confusion_matrix}")
    for record in records[: min(len(records), PRINT_SAMPLES)]:
        block = record["block"]
        judge = record["judge"]
        assert isinstance(block, dict)
        assert isinstance(judge, dict)
        print(
            f"- {block['block_id']} | true={block['label']} "
            f"| pred={judge['verdict']} | confidence={judge['confidence']:.3f}"
        )
    if args.output_jsonl:
        _write_jsonl(args.output_jsonl, records)
    if args.metrics_json:
        output_path = Path(args.metrics_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(metrics.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Wrote metrics JSON: {output_path}")


def _cmd_split_experiment(args: argparse.Namespace) -> None:
    """按时间顺序运行固定数量 train/val/test 实验。"""
    result = run_split_experiment(
        input_path=args.input,
        train_count=args.train_count,
        val_count=args.val_count,
        test_count=args.test_count,
        rounds=args.rounds,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        debate_mode=args.debate_mode,
        judge_mode=args.judge_mode,
        seed=args.seed,
    )
    config = result["config"]
    print(
        "Split experiment complete: "
        f"train={config['train_count']} val={config['val_count']} test={config['test_count']} "
        f"rounds={config['rounds']} epochs={config['epochs']} "
        f"debate_mode={config['debate_mode']} judge_mode={config['judge_mode']} seed={config['seed']}"
    )
    print(
        "Selected time range: "
        f"{config['selected_time_range']['start']} -> {config['selected_time_range']['end']}"
    )
    losses = result["train_losses"]
    if losses:
        print(f"Train loss: first={losses[0]:.4f} last={losses[-1]:.4f}")
    for split_name in ("train", "val", "test"):
        metrics = result["metrics"][split_name]
        if metrics is None:
            continue
        print(
            f"{split_name.upper()} metrics: "
            f"n={metrics['total']} "
            f"accuracy={metrics['accuracy']:.4f} "
            f"macro_f1={metrics['macro_f1']:.4f} "
            f"bull_f1={metrics['bullish']['f1']:.4f} "
            f"bear_f1={metrics['bearish']['f1']:.4f} "
            f"coverage={metrics['coverage']:.4f}"
        )
        print(f"{split_name.upper()} confusion: {metrics['confusion_matrix']}")
    for split_name in ("train", "val", "test"):
        records = result["records"][split_name]
        for record in records[: min(len(records), PRINT_SAMPLES)]:
            block = record["block"]
            judge = record["judge"]
            assert isinstance(block, dict)
            assert isinstance(judge, dict)
            print(
                f"{split_name}: {block['block_id']} | true={block['label']} "
                f"| pred={judge['verdict']} | confidence={judge['confidence']:.3f}"
            )
            print(f"{split_name} judge_report: {judge.get('report', '')}")
    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Wrote experiment JSON: {output_path}")


def _cmd_case_study(args: argparse.Namespace) -> None:
    """运行并导出可读案例报告。"""
    result = run_case_study(
        input_path=args.input,
        post_id=args.post_id,
        block_id=args.block_id,
        max_blocks=args.max_blocks,
        rounds=args.rounds,
        debate_mode=args.debate_mode,
        judge_mode=args.judge_mode,
        seed=args.seed,
    )
    print(f"Case study post: {result['config']['post_id']}")
    print(f"Root comments in post: {len(result['post']['comments'])}")
    print(f"Debated blocks: {len(result['records'])}")
    for record in result["records"]:
        block = record["block"]
        judge = record["judge"]
        assert isinstance(block, dict)
        assert isinstance(judge, dict)
        print(
            f"- {block['block_id']} | label={block['label']} "
            f"| verdict={judge['verdict']} | confidence={judge['confidence']:.3f}"
        )
    if args.output_json:
        output_json = Path(args.output_json)
        output_json.parent.mkdir(parents=True, exist_ok=True)
        output_json.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Wrote case JSON: {output_json}")
    if args.output_md:
        output_md = Path(args.output_md)
        output_md.parent.mkdir(parents=True, exist_ok=True)
        output_md.write_text(render_case_markdown(result), encoding="utf-8")
        print(f"Wrote case Markdown: {output_md}")


def _write_jsonl(path: str, records: list[dict[str, object]]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"Wrote JSONL: {output_path}")


if __name__ == "__main__":
    main()


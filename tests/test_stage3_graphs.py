from __future__ import annotations

import unittest
from datetime import datetime
from pathlib import Path

from agent import DebateOrchestrator
from agent.schema import DebateTranscript
from data import build_comment_blocks, load_posts
from data.schema import CommentBlock, RawComment
from debate_graph import build_comment_graph, build_debate_graph, build_hetero_graph
from debate_graph.diffusion_ops import normalized_relation_adjacency
from profiles import ProfileStore
from scripts.build_graphs import build_graph_records


FIXTURE = Path(__file__).parent / "fixtures" / "sample_post.jsonl"


class StageThreeGraphTest(unittest.TestCase):
    def test_comment_graph_contains_reply_edges(self):
        block = _block_with_reply()

        nodes, edges = build_comment_graph(block)

        self.assertEqual(len(nodes), 2)
        self.assertEqual(len(edges), 1)
        self.assertEqual(edges[0].relation, "reply")
        self.assertEqual(edges[0].source, "comment:r1")
        self.assertEqual(edges[0].target, "comment:c1")

    def test_debate_graph_relations(self):
        block, transcript = _fixture_block_and_transcript(rounds=2)

        nodes, edges = build_debate_graph(transcript)
        relations = {edge.relation for edge in edges}

        self.assertEqual(len(nodes), 60)
        self.assertIn("cite", relations)
        self.assertIn("attack", relations)
        self.assertIn("respond", relations)
        self.assertNotIn("precede", relations)
        self.assertIn("support", relations)
        self.assertIn("propose", relations)
        self.assertTrue(all("phase" in node.attrs for node in nodes))
        self.assertTrue(all("relative_time" in node.attrs for node in nodes))
        self.assertTrue(all(edge.source.startswith("argument:") for edge in edges if edge.relation != "cite"))
        self.assertTrue(any(edge.target == f"comment:{block.root_comment.comment_id}" for edge in edges))

    def test_hetero_graph_fuses_comment_and_debate_nodes(self):
        block, transcript = _fixture_block_and_transcript(rounds=1)

        graph = build_hetero_graph(block, transcript)

        self.assertEqual(graph.graph_id, block.block_id)
        self.assertEqual(graph.node_counts()["comment"], 1)
        self.assertEqual(graph.node_counts()["argument"], 30)
        self.assertIn("cite", graph.relation_counts())

    def test_normalized_relation_adjacency_rows_sum_to_one(self):
        block, transcript = _fixture_block_and_transcript(rounds=1)
        graph = build_hetero_graph(block, transcript)

        normalized = normalized_relation_adjacency(graph)

        for triples in normalized.values():
            row_sums: dict[int, float] = {}
            for source, _target, weight in triples:
                row_sums[source] = row_sums.get(source, 0.0) + weight
            for value in row_sums.values():
                self.assertAlmostEqual(value, 1.0)

    def test_build_graph_records_smoke(self):
        records = build_graph_records(str(FIXTURE), limit_blocks=1, rounds=1)

        self.assertEqual(len(records), 1)
        graph = records[0]["graph"]
        self.assertEqual(graph["graph_id"], "p1:c1")
        self.assertIn("normalized_adjacency", records[0])
        self.assertGreater(len(graph["nodes"]), 1)
        self.assertGreater(len(graph["edges"]), 1)


def _fixture_block_and_transcript(rounds: int) -> tuple[CommentBlock, DebateTranscript]:
    posts = load_posts(FIXTURE)
    blocks, issues = build_comment_blocks(posts)
    assert not issues
    block = blocks[0]
    profiles = ProfileStore.from_blocks(blocks).get_profiles_for_block(block)
    transcript = DebateOrchestrator().run(block, profiles, rounds=rounds)
    return block, transcript


def _block_with_reply() -> CommentBlock:
    root = RawComment(
        comment_id="c1",
        original_comment_id="orig-c1",
        author="alice",
        text="root",
        post_time=None,
        replies=[
            RawComment(
                comment_id="r1",
                original_comment_id="orig-r1",
                author="bob",
                text="reply",
                post_time=None,
            )
        ],
    )
    return CommentBlock(
        block_id="p1:c1",
        post_id="p1",
        post_content="news",
        products=["BNB"],
        root_comment=root,
        replies=root.replies,
        t0=datetime(2026, 1, 1, 0, 0, 0),
        t_window="24h",
        p0=100.0,
        p1=110.0,
        label=1,
        product="BNB",
    )


if __name__ == "__main__":
    unittest.main()



from __future__ import annotations

import unittest
from pathlib import Path

import torch

from agent import DebateOrchestrator
from data import build_comment_blocks, load_posts
from debate_graph import build_hetero_graph, graph_to_tensor
from debate_graph.graph_batch import NODE_FEATURE_DIM
from model import GraphSentimentModel
from model.losses import classification_loss
from profiles import ProfileStore
from scripts.run_split_experiment import run_split_experiment
from scripts.train_prototype import train_prototype
from scripts.run_full_pipeline import run_full_pipeline


FIXTURE = Path(__file__).parent / "fixtures" / "sample_post.jsonl"
DATASET = Path(__file__).parents[1] / "dataset" / "final.jsonl"


class StageFourModelTest(unittest.TestCase):
    def test_graph_to_tensor_shapes(self):
        graph_tensor = _fixture_graph_tensor()

        self.assertEqual(graph_tensor.x.shape[1], NODE_FEATURE_DIM)
        self.assertEqual(graph_tensor.label.shape, (1,))
        for adj in graph_tensor.relation_adjs.values():
            self.assertEqual(adj.shape, (graph_tensor.num_nodes, graph_tensor.num_nodes))

    def test_model_forward_probability(self):
        graph_tensor = _fixture_graph_tensor()
        model = GraphSentimentModel(input_dim=NODE_FEATURE_DIM, hidden_dim=8, ode_steps=2)

        prob = model(graph_tensor)

        self.assertEqual(prob.shape, (1,))
        value = float(prob.detach())
        self.assertGreaterEqual(value, 0.0)
        self.assertLessEqual(value, 1.0)

    def test_model_backward_step(self):
        graph_tensor = _fixture_graph_tensor()
        model = GraphSentimentModel(input_dim=NODE_FEATURE_DIM, hidden_dim=8, ode_steps=2)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)

        prob = model(graph_tensor)
        loss = classification_loss(prob, graph_tensor.label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        self.assertTrue(torch.isfinite(loss))

    def test_train_prototype_smoke(self):
        metrics = train_prototype(str(FIXTURE), limit_blocks=1, rounds=1, epochs=2)

        self.assertEqual(metrics["graphs"], 1.0)
        self.assertGreaterEqual(metrics["final_loss"], 0.0)
        self.assertGreaterEqual(metrics["mean_probability"], 0.0)
        self.assertLessEqual(metrics["mean_probability"], 1.0)

    def test_full_pipeline_judge_receives_model_summary(self):
        records = run_full_pipeline(str(FIXTURE), limit_blocks=1, rounds=1, train_epochs=1)

        self.assertEqual(len(records), 1)
        self.assertIn("model_summary", records[0])
        self.assertIn("judge", records[0])
        report = records[0]["judge"]["report"]
        self.assertIn("ODE", report)
        self.assertIn("bullish_probability", records[0]["model_summary"])

    def test_split_experiment_smoke(self):
        if not DATASET.exists():
            self.skipTest("dataset/final.jsonl is not available")

        result = run_split_experiment(
            input_path=str(DATASET),
            train_count=2,
            val_count=1,
            test_count=1,
            rounds=1,
            epochs=1,
            debate_mode="mock",
            seed=123,
        )

        self.assertEqual(result["config"]["train_count"], 2)
        self.assertEqual(result["metrics"]["train"]["total"], 2)
        self.assertEqual(result["metrics"]["val"]["total"], 1)
        self.assertEqual(result["metrics"]["test"]["total"], 1)


def _fixture_graph_tensor():
    posts = load_posts(FIXTURE)
    blocks, issues = build_comment_blocks(posts)
    assert not issues
    block = blocks[0]
    profiles = ProfileStore.from_blocks(blocks).get_profiles_for_block(block)
    transcript = DebateOrchestrator().run(block, profiles, rounds=1)
    graph = build_hetero_graph(block, transcript)
    return graph_to_tensor(graph, label=block.label)


if __name__ == "__main__":
    unittest.main()



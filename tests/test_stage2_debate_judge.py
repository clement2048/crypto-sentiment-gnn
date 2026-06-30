from __future__ import annotations

import json
import unittest
from pathlib import Path

from agent import DebateOrchestrator
from agent.openai_compatible import (
    SiliconFlowJudgeClient,
    SiliconFlowOpenAICompatibleDebateClient,
)
from agent.output_parser import parse_argument_json
from data import build_comment_blocks, load_posts
from debate_graph import build_hetero_graph, graph_to_tensor
from judge.consistency import check_judge_consistency
from judge.judge_parser import parse_judge_json
from judge.judge_schema import JudgeOutput, JudgeScoreVector
from model import GraphSentimentModel
from profiles import ProfileStore
from scripts.evaluate_pipeline import compute_metrics
from scripts.run_debate import run_debate_pipeline
from tests.fakes import FakeDebateClient, FakeJudgeClient


FIXTURE = Path(__file__).parent / "fixtures" / "sample_post.jsonl"


class StageTwoDebateJudgeTest(unittest.TestCase):
    def test_parse_argument_json(self):
        raw = {
            "argument_id": "p1:c1:r1:s1:bull",
            "agent_id": "bull_bull_agent",
            "camp": "bull",
            "role": "bull_agent",
            "claim": "閻瀹氱拋铏瑰仯",
            "evidence": [
                {
                    "source_type": "root_comment",
                    "source_id": "c1",
                    "quote": "閻楁稒娼垫禍鍡吹",
                    "relevance": 0.8,
                }
            ],
            "confidence": 0.7,
            "target_args": [],
            "cited_comment_ids": ["c1"],
            "round": 1,
            "seq": 1,
        }

        argument = parse_argument_json(json.dumps(raw, ensure_ascii=False))

        self.assertEqual(argument.argument_id, "p1:c1:r1:s1:bull")
        self.assertEqual(argument.camp, "bull")
        self.assertEqual(argument.evidence[0].source_type, "root_comment")

    def test_parse_argument_json_repairs_common_llm_json_errors(self):
        raw = """
        {
          "argument_id": "p1:c1:r1:s1:bull",
          "agent_id": "bull_bull_agent",
          "camp": "bull",
          "role": "bull_agent",
          "claim": "鐪嬫定璁虹偣",
          "evidence": [
            {"source_type": "root_comment", "source_id": "c1", "quote": "鐗涙潵浜嗭紵", "relevance": 0.8}
          ],
          "confidence": 0.7,
          "target_args": [],
          "cited_comment_ids": ["c1"],
          "round": 1,
          "seq": 1
        """

        argument = parse_argument_json(raw)

        self.assertEqual(argument.argument_id, "p1:c1:r1:s1:bull")
        self.assertEqual(argument.claim, "鐪嬫定璁虹偣")

    def test_parse_argument_json_accepts_prior_argument_evidence(self):
        raw = {
            "argument_id": "p1:c1:r1:s2:bear",
            "agent_id": "bear_bear_agent",
            "camp": "bear",
            "role": "bear_agent",
            "claim": "bear replies to prior bull argument",
            "evidence": [
                {
                    "source_type": "prior_argument",
                    "source_id": "p1:c1:r1:s1:bull",
                    "quote": "姝ｆ柟鍏堝墠璁虹偣璇佹嵁涓嶈冻",
                    "relevance": 0.7,
                }
            ],
            "confidence": 0.6,
            "target_args": ["p1:c1:r1:s1:bull"],
            "cited_comment_ids": [],
            "round": 1,
            "seq": 2,
        }

        argument = parse_argument_json(json.dumps(raw, ensure_ascii=False))

        self.assertEqual(argument.evidence[0].source_type, "prior_argument")

    def test_parse_argument_json_normalizes_comment_block_evidence(self):
        raw = {
            "argument_id": "p1:c1:r1:s1:bull",
            "agent_id": "bull_bull_agent",
            "camp": "bull",
            "role": "bull_agent",
            "claim": "bull cites the whole comment block",
            "evidence": [
                {
                    "source_type": "comment_block",
                    "source_id": "p1:c1",
                    "quote": "璇勮鍧楁暣浣撴儏缁亸绉瀬",
                    "relevance": 0.7,
                }
            ],
            "confidence": 0.6,
            "target_args": [],
            "cited_comment_ids": ["c1"],
            "round": 1,
            "seq": 1,
        }

        argument = parse_argument_json(json.dumps(raw, ensure_ascii=False))

        self.assertEqual(argument.evidence[0].source_type, "root_comment")

    def test_orchestrator_generates_stable_complete_debate(self):
        block, profiles = _load_first_block_and_profiles()

        transcript = DebateOrchestrator(client=FakeDebateClient()).run(block, profiles, rounds=2)

        self.assertEqual(transcript.block_id, block.block_id)
        self.assertEqual(len(transcript.arguments), 4)
        self.assertEqual(transcript.arguments[0].argument_id, "p1:c1:r1:s1:bull")
        for argument in transcript.arguments:
            self.assertTrue(argument.argument_id)
            self.assertIn(argument.camp, ("bull", "bear"))
            self.assertTrue(argument.claim)
            self.assertTrue(argument.phase)
            self.assertGreaterEqual(argument.confidence, 0.0)
            self.assertLessEqual(argument.confidence, 1.0)

    def test_orchestrator_uses_one_bull_and_one_bear_agent_by_default(self):
        block, profiles = _load_first_block_and_profiles()

        transcript = DebateOrchestrator(client=FakeDebateClient()).run(block, profiles, rounds=1)

        self.assertEqual(
            [argument.role for argument in transcript.arguments],
            [
                "bull_agent",
                "bear_agent",
            ],
        )
        self.assertEqual(
            [argument.phase for argument in transcript.arguments],
            ["initial_argument", "rebuttal"],
        )
        self.assertEqual(transcript.arguments[1].target_args, [transcript.arguments[0].argument_id])

    def test_fake_judge_output_and_consistency(self):
        block, profiles = _load_first_block_and_profiles()
        transcript = DebateOrchestrator(client=FakeDebateClient()).run(block, profiles, rounds=1)
        graph = build_hetero_graph(block, transcript)
        graph_tensor = graph_to_tensor(graph, label=block.label, embedding_backend="none")
        model_summary = GraphSentimentModel(input_dim=graph_tensor.x.shape[1], hidden_dim=8, ode_steps=1).summarize(graph_tensor)

        output = FakeJudgeClient().judge(transcript, model_summary, graph)

        self.assertIn(output.verdict, ("BULLISH", "BEARISH"))
        self.assertGreaterEqual(output.confidence, 0.0)
        self.assertLessEqual(output.confidence, 1.0)
        self.assertEqual(output.consistency_flags, check_judge_consistency(output))
        self.assertEqual(set(output.score_vector.to_dict()), {
            "p_bull",
            "p_bear",
            "q_bull",
            "q_bear",
            "e_bull",
            "e_bear",
            "c",
            "d",
            "a",
            "rho",
        })

    def test_parse_judge_json(self):
        raw = {
            "verdict": "BULLISH",
            "confidence": 0.8,
            "report": "ok",
            "score_vector": {
                "p_bull": 0.7,
                "p_bear": 0.3,
                "q_bull": 0.6,
                "q_bear": 0.4,
                "e_bull": 0.7,
                "e_bear": 0.3,
                "c": 0.5,
                "d": 0.2,
                "a": 0.2,
                "rho": 0.8,
            },
            "consistency_flags": [],
        }

        output = parse_judge_json(json.dumps(raw))

        self.assertEqual(output.verdict, "BULLISH")
        self.assertEqual(output.score_vector.p_bull, 0.7)

    def test_siliconflow_judge_client_uses_openai_payload_and_parses_output(self):
        block, profiles = _load_first_block_and_profiles()
        transcript = DebateOrchestrator(client=FakeDebateClient()).run(block, profiles, rounds=1)
        graph = build_hetero_graph(block, transcript)
        graph_tensor = graph_to_tensor(graph, label=block.label, embedding_backend="none")
        model_summary = GraphSentimentModel(input_dim=graph_tensor.x.shape[1], hidden_dim=8, ode_steps=1).summarize(graph_tensor)

        def fake_transport(payload):
            self.assertIn("model", payload)
            self.assertFalse(payload["enable_thinking"])
            self.assertEqual(payload["messages"][0]["role"], "system")
            self.assertEqual(payload["messages"][1]["role"], "user")
            self.assertIn("model_summary", payload["messages"][1]["content"])
            return {
                "choices": [
                    {
                        "message": {
                            "content": json.dumps(
                                {
                                    "verdict": "BULLISH",
                                    "confidence": 0.68,
                                    "report": "SiliconFlow judge report.",
                                    "score_vector": {
                                        "p_bull": 0.7,
                                        "p_bear": 0.3,
                                        "q_bull": 0.6,
                                        "q_bear": 0.4,
                                        "e_bull": 0.7,
                                        "e_bear": 0.3,
                                        "c": 0.6,
                                        "d": 0.5,
                                        "a": 0.6,
                                        "rho": 0.68,
                                    },
                                    "consistency_flags": [],
                                },
                                ensure_ascii=False,
                            )
                        }
                    }
                ]
            }

        output = SiliconFlowJudgeClient(transport=fake_transport).judge(transcript, model_summary, graph)

        self.assertEqual(output.verdict, "BULLISH")
        self.assertEqual(output.report, "SiliconFlow judge report.")

    def test_consistency_flags_direction_mismatch(self):
        output = JudgeOutput(
            verdict="BULLISH",
            confidence=0.8,
            report="mismatch",
            score_vector=JudgeScoreVector(
                p_bull=0.2,
                p_bear=0.8,
                q_bull=0.5,
                q_bear=0.5,
                e_bull=0.5,
                e_bear=0.5,
                c=0.5,
                d=0.5,
                a=0.5,
                rho=0.8,
            ),
            consistency_flags=[],
        )

        self.assertIn("verdict_score_direction_mismatch", check_judge_consistency(output))

    def test_run_debate_pipeline_smoke(self):
        records = run_debate_pipeline(str(FIXTURE), limit_blocks=1, rounds=1, client=FakeDebateClient())

        self.assertEqual(len(records), 1)
        self.assertIn("block", records[0])
        self.assertIn("profiles", records[0])
        self.assertIn("debate", records[0])
        self.assertNotIn("judge", records[0])
        self.assertEqual(len(records[0]["debate"]["arguments"]), 2)

    def test_siliconflow_client_uses_openai_payload_and_parses_argument(self):
        block, profiles = _load_first_block_and_profiles()

        def fake_transport(payload):
            self.assertIn("model", payload)
            self.assertFalse(payload["enable_thinking"])
            self.assertEqual(payload["messages"][0]["role"], "system")
            self.assertEqual(payload["messages"][1]["role"], "user")
            self.assertIn("required_metadata", payload["messages"][1]["content"])
            return {
                "choices": [
                    {
                        "message": {
                            "content": json.dumps(
                                {
                                    "argument_id": "wrong-id",
                                    "agent_id": "wrong-agent",
                                    "camp": "bear",
                                    "role": "bear_agent",
                                    "claim": "siliconflow bearish argument",
                                    "evidence": [],
                                    "confidence": 0.66,
                                    "target_args": ["not-allowed-target"],
                                    "cited_comment_ids": ["c1"],
                                    "round": 0,
                                    "seq": 0,
                                    "phase": "wrong-phase",
                                },
                                ensure_ascii=False,
                            )
                        }
                    }
                ]
            }

        client = SiliconFlowOpenAICompatibleDebateClient(transport=fake_transport)
        argument = client.generate_argument(
            block=block,
            profiles=profiles,
            camp="bear",
            role="bear_agent",
            round_index=1,
            seq=2,
            prior_arguments=[],
            available_target_ids=[],
        )

        self.assertEqual(argument.argument_id, "p1:c1:r1:s2:bear")
        self.assertEqual(argument.agent_id, "bear_bear_agent")
        self.assertEqual(argument.phase, "initial_argument")
        self.assertEqual(argument.target_args, [])
        self.assertEqual(argument.claim, "siliconflow bearish argument")

    def test_compute_evaluation_metrics_includes_precision_recall_f1(self):
        records = [
            _evaluation_record(true_label=1, verdict="BULLISH"),
            _evaluation_record(true_label=1, verdict="BEARISH"),
            _evaluation_record(true_label=-1, verdict="BULLISH"),
            _evaluation_record(true_label=-1, verdict="BEARISH"),
        ]

        metrics = compute_metrics(records)

        self.assertEqual(metrics.total, 4)
        self.assertEqual(metrics.accuracy, 0.5)
        self.assertAlmostEqual(metrics.bullish.precision, 0.5)
        self.assertAlmostEqual(metrics.bullish.recall, 0.5)
        self.assertAlmostEqual(metrics.bullish.f1, 0.5)
        self.assertAlmostEqual(metrics.bearish.precision, 0.5)
        self.assertAlmostEqual(metrics.bearish.recall, 0.5)
        self.assertAlmostEqual(metrics.bearish.f1, 0.5)
        self.assertEqual(
            metrics.confusion_matrix,
            {
                "bullish": {"bullish": 1, "bearish": 1},
                "bearish": {"bullish": 1, "bearish": 1},
            },
        )


def _load_first_block_and_profiles():
    posts = load_posts(FIXTURE)
    blocks, issues = build_comment_blocks(posts)
    assert not issues
    block = blocks[0]
    profiles = ProfileStore.from_blocks(blocks).get_profiles_for_block(block)
    return block, profiles


def _evaluation_record(true_label: int, verdict: str) -> dict[str, object]:
    return {
        "block": {"block_id": "p1:c1", "label": true_label},
        "judge": {"verdict": verdict, "confidence": 0.7},
    }


if __name__ == "__main__":
    unittest.main()

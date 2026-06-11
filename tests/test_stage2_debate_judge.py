from __future__ import annotations

import json
import unittest
from pathlib import Path

from agent import DebateOrchestrator
from agent.anthropic_compatible import (
    DeepSeekAnthropicDebateClient,
    DeepSeekJudgeClient,
    _load_api_key_from_dotenv_text,
)
from agent.openai_compatible import (
    BailianJudgeClient,
    BailianOpenAICompatibleDebateClient,
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
from config import BAILIAN_MODEL
from profiles import ProfileStore
from scripts.evaluate_pipeline import compute_metrics
from scripts.run_debate import run_debate_pipeline
from tests.fakes import FakeDebateClient, FakeJudgeClient


FIXTURE = Path(__file__).parent / "fixtures" / "sample_post.jsonl"


class StageTwoDebateJudgeTest(unittest.TestCase):
    def test_parse_argument_json(self):
        raw = {
            "argument_id": "p1:c1:r1:s1:bull",
            "agent_id": "bull_price_action_agent",
            "camp": "bull",
            "role": "price_action_agent",
            "claim": "鐪嬫定璁虹偣",
            "evidence": [
                {
                    "source_type": "root_comment",
                    "source_id": "c1",
                    "quote": "鐗涙潵浜嗭紵",
                    "relevance": 0.8,
                }
            ],
            "confidence": 0.7,
            "targets": [],
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
          "agent_id": "bull_price_action_agent",
          "camp": "bull",
          "role": "price_action_agent",
          "claim": "看涨论点",
          "evidence": [
            {"source_type": "root_comment", "source_id": "c1", "quote": "牛来了？", "relevance": 0.8}
          ],
          "confidence": 0.7,
          "targets": [],
          "cited_comment_ids": ["c1"],
          "round": 1,
          "seq": 1
        """

        argument = parse_argument_json(raw)

        self.assertEqual(argument.argument_id, "p1:c1:r1:s1:bull")
        self.assertEqual(argument.claim, "看涨论点")

    def test_parse_argument_json_accepts_prior_argument_evidence(self):
        raw = {
            "argument_id": "p1:c1:r1:s2:bear",
            "agent_id": "bear_risk_analysis_agent",
            "camp": "bear",
            "role": "risk_analysis_agent",
            "claim": "反方回应先前正方论点。",
            "evidence": [
                {
                    "source_type": "prior_argument",
                    "source_id": "p1:c1:r1:s1:bull",
                    "quote": "正方先前论点证据不足",
                    "relevance": 0.7,
                }
            ],
            "confidence": 0.6,
            "targets": ["p1:c1:r1:s1:bull"],
            "cited_comment_ids": [],
            "round": 1,
            "seq": 2,
        }

        argument = parse_argument_json(json.dumps(raw, ensure_ascii=False))

        self.assertEqual(argument.evidence[0].source_type, "prior_argument")

    def test_parse_argument_json_normalizes_comment_block_evidence(self):
        raw = {
            "argument_id": "p1:c1:r1:s1:bull",
            "agent_id": "bull_technical_analysis_agent",
            "camp": "bull",
            "role": "technical_analysis_agent",
            "claim": "正方引用整个评论块。",
            "evidence": [
                {
                    "source_type": "comment_block",
                    "source_id": "p1:c1",
                    "quote": "评论块整体情绪偏积极",
                    "relevance": 0.7,
                }
            ],
            "confidence": 0.6,
            "targets": [],
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
        self.assertEqual(len(transcript.arguments), 60)
        self.assertEqual(transcript.arguments[0].argument_id, "p1:c1:r1:s1:bull")
        for argument in transcript.arguments:
            self.assertTrue(argument.argument_id)
            self.assertIn(argument.camp, ("bull", "bear"))
            self.assertTrue(argument.claim)
            self.assertTrue(argument.phase)
            self.assertGreaterEqual(argument.confidence, 0.0)
            self.assertLessEqual(argument.confidence, 1.0)

    def test_orchestrator_uses_paper_roles_by_default(self):
        block, profiles = _load_first_block_and_profiles()

        transcript = DebateOrchestrator(client=FakeDebateClient()).run(block, profiles, rounds=1)

        self.assertEqual(
            [argument.role for argument in transcript.arguments],
            [
                "technical_analysis_agent",
                "fundamental_analysis_agent",
                "sentiment_contagion_agent",
                "risk_analysis_agent",
                "onchain_skeptic_agent",
                "sentiment_reversal_agent",
                "reflection_agent",
                "technical_analysis_agent",
                "fundamental_analysis_agent",
                "sentiment_contagion_agent",
                "reflection_agent",
                "risk_analysis_agent",
                "onchain_skeptic_agent",
                "sentiment_reversal_agent",
                "technical_analysis_agent",
                "fundamental_analysis_agent",
                "sentiment_contagion_agent",
                "risk_analysis_agent",
                "onchain_skeptic_agent",
                "sentiment_reversal_agent",
                "reflection_agent",
                "technical_analysis_agent",
                "fundamental_analysis_agent",
                "sentiment_contagion_agent",
                "reflection_agent",
                "risk_analysis_agent",
                "onchain_skeptic_agent",
                "sentiment_reversal_agent",
                "reflection_agent",
                "reflection_agent",
            ],
        )
        self.assertEqual(
            [argument.phase for argument in transcript.arguments],
            ["initial_argument"] * 6
            + ["intra_reflection"] * 1
            + ["intra_response"] * 3
            + ["intra_reflection"] * 1
            + ["intra_response"] * 3
            + ["cross_response"] * 6
            + ["counter_reflection"] * 1
            + ["counter_rebuttal"] * 3
            + ["counter_reflection"] * 1
            + ["counter_rebuttal"] * 3
            + ["reflection_summary"] * 2,
        )
        self.assertTrue(_phase_targets_reflection(transcript.arguments, "intra_response", "intra_reflection"))
        self.assertTrue(_phase_targets_reflection(transcript.arguments, "counter_rebuttal", "counter_reflection"))

    def test_fake_judge_output_and_consistency(self):
        block, profiles = _load_first_block_and_profiles()
        transcript = DebateOrchestrator(client=FakeDebateClient()).run(block, profiles, rounds=1)
        graph = build_hetero_graph(block, transcript)
        model_summary = GraphSentimentModel(input_dim=8, hidden_dim=8, ode_steps=1).summarize(
            graph_to_tensor(graph, label=block.label)
        )

        output = FakeJudgeClient().judge(transcript, model_summary, graph)

        self.assertIn(output.verdict, ("BULLISH", "BEARISH", "NEUTRAL"))
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

    def test_deepseek_judge_client_uses_judge_payload_and_parses_output(self):
        block, profiles = _load_first_block_and_profiles()
        transcript = DebateOrchestrator(client=FakeDebateClient()).run(block, profiles, rounds=1)
        graph = build_hetero_graph(block, transcript)
        model_summary = GraphSentimentModel(input_dim=8, hidden_dim=8, ode_steps=1).summarize(
            graph_to_tensor(graph, label=block.label)
        )

        def fake_transport(payload):
            self.assertIn("independent judge", payload["system"])
            user_text = payload["messages"][0]["content"][0]["text"]
            self.assertIn("model_summary", user_text)
            self.assertIn("debate_arguments", user_text)
            self.assertIn("graph", user_text)
            self.assertNotIn('"label"', user_text)
            return {
                "content": [
                    {
                        "type": "text",
                        "text": json.dumps(
                            {
                                "verdict": "BULLISH",
                                "confidence": 0.72,
                                "report": "DeepSeek judge report.",
                                "score_vector": {
                                    "p_bull": 0.7,
                                    "p_bear": 0.3,
                                    "q_bull": 0.6,
                                    "q_bear": 0.4,
                                    "e_bull": 0.7,
                                    "e_bear": 0.3,
                                    "c": 0.8,
                                    "d": 0.5,
                                    "a": 0.6,
                                    "rho": 0.72,
                                },
                                "consistency_flags": [],
                            },
                            ensure_ascii=False,
                        ),
                    }
                ]
            }

        output = DeepSeekJudgeClient(transport=fake_transport).judge(transcript, model_summary, graph)

        self.assertEqual(output.verdict, "BULLISH")
        self.assertEqual(output.report, "DeepSeek judge report.")

    def test_bailian_judge_client_uses_openai_payload_and_parses_output(self):
        block, profiles = _load_first_block_and_profiles()
        transcript = DebateOrchestrator(client=FakeDebateClient()).run(block, profiles, rounds=1)
        graph = build_hetero_graph(block, transcript)
        model_summary = GraphSentimentModel(input_dim=8, hidden_dim=8, ode_steps=1).summarize(
            graph_to_tensor(graph, label=block.label)
        )

        def fake_transport(payload):
            self.assertEqual(payload["model"], BAILIAN_MODEL)
            self.assertFalse(payload["enable_thinking"])
            self.assertIn("messages", payload)
            self.assertEqual(payload["messages"][0]["role"], "system")
            self.assertEqual(payload["messages"][1]["role"], "user")
            self.assertIn("model_summary", payload["messages"][1]["content"])
            return {
                "choices": [
                    {
                        "message": {
                            "content": json.dumps(
                                {
                                    "verdict": "BEARISH",
                                    "confidence": 0.66,
                                    "report": "Bailian judge report.",
                                    "score_vector": {
                                        "p_bull": 0.3,
                                        "p_bear": 0.7,
                                        "q_bull": 0.4,
                                        "q_bear": 0.6,
                                        "e_bull": 0.3,
                                        "e_bear": 0.7,
                                        "c": 0.6,
                                        "d": 0.5,
                                        "a": 0.6,
                                        "rho": 0.66,
                                    },
                                    "consistency_flags": [],
                                },
                                ensure_ascii=False,
                            )
                        }
                    }
                ]
            }

        output = BailianJudgeClient(transport=fake_transport).judge(transcript, model_summary, graph)

        self.assertEqual(output.verdict, "BEARISH")
        self.assertEqual(output.report, "Bailian judge report.")

    def test_siliconflow_judge_client_uses_openai_payload_and_parses_output(self):
        block, profiles = _load_first_block_and_profiles()
        transcript = DebateOrchestrator(client=FakeDebateClient()).run(block, profiles, rounds=1)
        graph = build_hetero_graph(block, transcript)
        model_summary = GraphSentimentModel(input_dim=8, hidden_dim=8, ode_steps=1).summarize(
            graph_to_tensor(graph, label=block.label)
        )

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
        self.assertEqual(len(records[0]["debate"]["arguments"]), 30)

    def test_deepseek_client_uses_anthropic_payload_and_parses_argument(self):
        block, profiles = _load_first_block_and_profiles()

        def fake_transport(payload):
            self.assertEqual(payload["model"], "deepseek-v4-pro")
            self.assertEqual(payload["thinking"], {"type": "disabled"})
            self.assertIn("system", payload)
            self.assertIn("Technical Analysis Agent", payload["system"])
            self.assertEqual(payload["messages"][0]["role"], "user")
            user_text = payload["messages"][0]["content"][0]["text"]
            self.assertIn("required_metadata", user_text)
            self.assertIn("phase_instructions", user_text)
            return {
                "content": [
                    {
                        "type": "text",
                        "text": json.dumps(
                            {
                                "argument_id": "wrong-id",
                                "agent_id": "wrong-agent",
                                "camp": "bull",
                                "role": "price_action_agent",
                                "claim": "用户评论表达了看涨倾向。",
                                "evidence": [
                                    {
                                        "source_type": "root_comment",
                                        "source_id": "c1",
                                        "quote": "牛来了？",
                                        "relevance": 0.9,
                                    }
                                ],
                                "confidence": 0.7,
                                "targets": ["not-allowed-target"],
                                "cited_comment_ids": ["c1"],
                                "round": 0,
                                "seq": 0,
                                "phase": "wrong-phase",
                            },
                            ensure_ascii=False,
                        ),
                    }
                ]
            }

        client = DeepSeekAnthropicDebateClient(transport=fake_transport)
        argument = client.generate_argument(
            block=block,
            profiles=profiles,
            camp="bull",
            role="price_action_agent",
            round_index=1,
            seq=1,
            prior_arguments=[],
            available_target_ids=[],
        )

        self.assertEqual(argument.argument_id, "p1:c1:r1:s1:bull")
        self.assertEqual(argument.agent_id, "bull_price_action_agent")
        self.assertEqual(argument.round, 1)
        self.assertEqual(argument.seq, 1)
        self.assertEqual(argument.phase, "initial_argument")
        self.assertEqual(argument.targets, [])
        self.assertEqual(argument.claim, "用户评论表达了看涨倾向。")

    def test_bailian_client_uses_openai_payload_and_parses_argument(self):
        block, profiles = _load_first_block_and_profiles()

        def fake_transport(payload):
            self.assertEqual(payload["model"], BAILIAN_MODEL)
            self.assertFalse(payload["enable_thinking"])
            self.assertNotIn("thinking", payload)
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
                                    "camp": "bull",
                                    "role": "technical_analysis_agent",
                                    "claim": "百炼返回的看涨论点。",
                                    "evidence": [],
                                    "confidence": 0.7,
                                    "targets": ["not-allowed-target"],
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

        client = BailianOpenAICompatibleDebateClient(transport=fake_transport)
        argument = client.generate_argument(
            block=block,
            profiles=profiles,
            camp="bull",
            role="technical_analysis_agent",
            round_index=1,
            seq=1,
            prior_arguments=[],
            available_target_ids=[],
        )

        self.assertEqual(argument.argument_id, "p1:c1:r1:s1:bull")
        self.assertEqual(argument.agent_id, "bull_technical_analysis_agent")
        self.assertEqual(argument.phase, "initial_argument")
        self.assertEqual(argument.targets, [])
        self.assertEqual(argument.claim, "百炼返回的看涨论点。")

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
                                    "role": "risk_analysis_agent",
                                    "claim": "硅基流动返回的看跌论点。",
                                    "evidence": [],
                                    "confidence": 0.66,
                                    "targets": ["not-allowed-target"],
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
            role="risk_analysis_agent",
            round_index=1,
            seq=2,
            prior_arguments=[],
            available_target_ids=[],
        )

        self.assertEqual(argument.argument_id, "p1:c1:r1:s2:bear")
        self.assertEqual(argument.agent_id, "bear_risk_analysis_agent")
        self.assertEqual(argument.phase, "initial_argument")
        self.assertEqual(argument.targets, [])
        self.assertEqual(argument.claim, "硅基流动返回的看跌论点。")

    def test_deepseek_client_repairs_invalid_json_once(self):
        block, profiles = _load_first_block_and_profiles()
        calls = []

        def fake_transport(payload):
            calls.append(payload)
            if len(calls) == 1:
                return {"content": [{"type": "text", "text": '{"argument_id": "broken", "claim": "oops"'}]}
            self.assertIn("invalid JSON", payload["messages"][-1]["content"][0]["text"])
            return {
                "content": [
                    {
                        "type": "text",
                        "text": json.dumps(
                            {
                                "argument_id": "wrong-id",
                                "agent_id": "wrong-agent",
                                "camp": "bear",
                                "role": "risk_rebuttal_agent",
                                "claim": "修复后的合法 JSON。",
                                "evidence": [],
                                "confidence": 0.6,
                                "targets": [],
                                "cited_comment_ids": [],
                                "round": 0,
                                "seq": 0,
                            },
                            ensure_ascii=False,
                        ),
                    }
                ]
            }

        client = DeepSeekAnthropicDebateClient(transport=fake_transport)
        argument = client.generate_argument(
            block=block,
            profiles=profiles,
            camp="bear",
            role="risk_rebuttal_agent",
            round_index=1,
            seq=2,
            prior_arguments=[],
        )

        self.assertEqual(len(calls), 2)
        self.assertEqual(argument.argument_id, "p1:c1:r1:s2:bear")
        self.assertEqual(argument.claim, "修复后的合法 JSON。")

    def test_deepseek_client_can_read_project_dotenv_key(self):
        text = "DEEPSEEK_API_KEY='test-key-from-dotenv'\n"

        self.assertEqual(_load_api_key_from_dotenv_text(text), "test-key-from-dotenv")

    def test_compute_evaluation_metrics_includes_precision_recall_f1(self):
        records = [
            _evaluation_record(true_label=1, verdict="BULLISH"),
            _evaluation_record(true_label=1, verdict="NEUTRAL"),
            _evaluation_record(true_label=-1, verdict="BULLISH"),
            _evaluation_record(true_label=-1, verdict="BEARISH"),
        ]

        metrics = compute_metrics(records)

        self.assertEqual(metrics.total, 4)
        self.assertEqual(metrics.accuracy, 0.5)
        self.assertEqual(metrics.coverage, 0.75)
        self.assertAlmostEqual(metrics.bullish.precision, 0.5)
        self.assertAlmostEqual(metrics.bullish.recall, 0.5)
        self.assertAlmostEqual(metrics.bullish.f1, 0.5)
        self.assertAlmostEqual(metrics.bearish.precision, 1.0)
        self.assertAlmostEqual(metrics.bearish.recall, 0.5)
        self.assertAlmostEqual(metrics.bearish.f1, 2 / 3)


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


def _phase_targets_reflection(arguments, response_phase: str, reflection_phase: str) -> bool:
    reflection_ids_by_camp = {
        argument.camp: argument.argument_id
        for argument in arguments
        if argument.phase == reflection_phase
    }
    responses = [argument for argument in arguments if argument.phase == response_phase]
    return bool(responses) and all(
        reflection_ids_by_camp.get(argument.camp) in argument.targets
        for argument in responses
    )


if __name__ == "__main__":
    unittest.main()

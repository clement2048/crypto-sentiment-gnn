"""Aliyun Bailian-backed LLM judge provider."""

from __future__ import annotations

import copy
import json
from typing import Any

from agent.bailian_client import BailianOpenAICompatibleDebateClient, Transport, _extract_openai_text
from agent.schema import DebateTranscript
from config import BAILIAN_MAX_TOKENS, BAILIAN_TEMPERATURE
from debate_graph.schema import HeteroGraph
from judge.consistency import check_judge_consistency
from judge.judge_parser import parse_judge_json
from judge.judge_prompt import JUDGE_OUTPUT_SCHEMA_PROMPT
from judge.judge_schema import JudgeOutput
from model.model_summary import ModelOutputSummary


class BailianJudgeClient:
    """使用阿里云百炼 OpenAI-compatible API 生成最终 JudgeOutput。"""

    def __init__(
        self,
        max_tokens: int = BAILIAN_MAX_TOKENS,
        temperature: float = BAILIAN_TEMPERATURE,
        transport: Transport | None = None,
    ):
        self.http = BailianOpenAICompatibleDebateClient(
            max_tokens=max_tokens,
            temperature=temperature,
            transport=transport,
        )

    def judge(
        self,
        transcript: DebateTranscript,
        model_summary: ModelOutputSummary,
        graph: HeteroGraph,
    ) -> JudgeOutput:
        payload = self._build_payload(transcript, model_summary, graph)
        response = self.http._send_payload(payload)
        text = _extract_openai_text(response)
        try:
            output = parse_judge_json(text)
        except (ValueError, json.JSONDecodeError):
            repair_payload = self._build_repair_payload(payload, text)
            repair_response = self.http._send_payload(repair_payload)
            output = parse_judge_json(_extract_openai_text(repair_response))
        output.consistency_flags = sorted(set([*output.consistency_flags, *check_judge_consistency(output)]))
        return output

    def _build_payload(
        self,
        transcript: DebateTranscript,
        model_summary: ModelOutputSummary,
        graph: HeteroGraph,
    ) -> dict[str, Any]:
        return {
            "model": self.http.model,
            "max_tokens": self.http.max_tokens,
            "temperature": self.http.temperature,
            "enable_thinking": self.http.enable_thinking,
            "messages": [
                {"role": "system", "content": JUDGE_OUTPUT_SCHEMA_PROMPT},
                {
                    "role": "user",
                    "content": json.dumps(
                        _judge_input(transcript, model_summary, graph),
                        ensure_ascii=False,
                        indent=2,
                    ),
                },
            ],
        }

    def _build_repair_payload(self, original_payload: dict[str, Any], invalid_text: str) -> dict[str, Any]:
        payload = copy.deepcopy(original_payload)
        payload["temperature"] = 0.0
        payload["messages"] = [
            *payload["messages"],
            {"role": "assistant", "content": invalid_text[:4000]},
            {
                "role": "user",
                "content": (
                    "The previous response was not a valid JudgeOutput JSON object. "
                    "Repair it into exactly one valid JSON object with verdict, confidence, "
                    "report, score_vector, and consistency_flags. Return only JSON."
                ),
            },
        ]
        return payload


def _judge_input(
    transcript: DebateTranscript,
    model_summary: ModelOutputSummary,
    graph: HeteroGraph,
) -> dict[str, Any]:
    """给法官的输入视图：不包含真实 label、p1 等未来验证字段。"""
    return {
        "task": "Produce final JudgeOutput after comparing debate graph and Bi-ODE/model summary.",
        "block_id": transcript.block_id,
        "t0": transcript.t0.strftime("%Y-%m-%d %H:%M:%S"),
        "model_summary": model_summary.to_dict(),
        "debate_arguments": [
            {
                "argument_id": item.argument_id,
                "agent_id": item.agent_id,
                "camp": item.camp,
                "role": item.role,
                "phase": item.phase,
                "claim": item.claim,
                "evidence": [evidence.to_dict() for evidence in item.evidence],
                "confidence": item.confidence,
                "targets": item.targets,
                "round": item.round,
                "seq": item.seq,
            }
            for item in transcript.arguments
        ],
        "graph": {
            "graph_id": graph.graph_id,
            "node_counts": graph.node_counts(),
            "relation_counts": graph.relation_counts(),
            "nodes": [
                {
                    "node_id": node.node_id,
                    "node_type": node.node_type,
                    "ref_id": node.ref_id,
                    "text": node.text[:500],
                    "attrs": node.attrs,
                }
                for node in graph.nodes
            ],
            "edges": [
                {
                    "source": edge.source,
                    "target": edge.target,
                    "relation": edge.relation,
                    "weight": edge.weight,
                    "attrs": edge.attrs,
                }
                for edge in graph.edges
            ],
        },
        "rules": [
            "Do not use ground-truth labels, p1, or future prices.",
            "Base the decision only on debate logic and model/ODE summary.",
            "If debate and model disagree, explain the disagreement and lower confidence.",
        ],
    }

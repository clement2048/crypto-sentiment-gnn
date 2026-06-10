"""接收「辩论结构 + ODE/模型摘要」的法官。

这才是 v2 文档中的法官位置：不是辩论后先判一次，
而是在模型演化结果出来后，再结合辩论结构输出 JudgeOutput。
"""

from __future__ import annotations

from statistics import mean

from agent.schema import DebateTranscript
from config import (
    DIVISION_EPSILON,
    JUDGE_CONFIDENCE_BASE,
    JUDGE_CONFIDENCE_DIFF_DIVISOR,
    JUDGE_DEBATE_CENTER,
    JUDGE_DEBATE_DIFF_DIVISOR,
    JUDGE_DEBATE_WEIGHT,
    JUDGE_EVIDENCE_QUALITY_SCALE,
    JUDGE_MARGIN_BEAR_VALUE,
    JUDGE_MARGIN_BULL_VALUE,
    JUDGE_MARGIN_WEIGHT,
    JUDGE_MODEL_WEIGHT,
    JUDGE_NEUTRAL_MARGIN,
    JUDGE_ROLE_COVERAGE_SCALE,
    PROBABILITY_MAX,
    PROBABILITY_MIN,
)
from judge.consistency import check_judge_consistency
from judge.judge_schema import JudgeOutput, JudgeScoreVector
from model.model_summary import ModelOutputSummary


class ModelAwareMockJudge:
    """最终 LLM Judge 的离线替身。

    真实版本未来会调用 LLM；当前 mock 用规则合成 JudgeScoreVector，
    目的是先跑通数据结构和完整流程。
    """

    def judge(self, transcript: DebateTranscript, model_summary: ModelOutputSummary) -> JudgeOutput:
        """根据辩论论点和模型摘要生成 JudgeOutput。"""
        bull_args = [item for item in transcript.arguments if item.camp == "bull"]
        bear_args = [item for item in transcript.arguments if item.camp == "bear"]
        bull_conf = mean([item.confidence for item in bull_args]) if bull_args else 0.0
        bear_conf = mean([item.confidence for item in bear_args]) if bear_args else 0.0

        debate_bull = _clamp01((JUDGE_DEBATE_CENTER + bull_conf - bear_conf) / JUDGE_DEBATE_DIFF_DIVISOR)
        model_bull = _clamp01(model_summary.bullish_probability)
        margin_bull = JUDGE_MARGIN_BULL_VALUE if model_summary.bull_bear_margin >= 0 else JUDGE_MARGIN_BEAR_VALUE
        # p_bull 融合三类信号：
        # 1. calibrator 概率 2. 辩论阵营平均置信度 3. ODE bull-bear margin 方向。
        p_bull = _clamp01(
            JUDGE_MODEL_WEIGHT * model_bull
            + JUDGE_DEBATE_WEIGHT * debate_bull
            + JUDGE_MARGIN_WEIGHT * margin_bull
        )
        p_bear = 1.0 - p_bull

        if abs(p_bull - p_bear) < JUDGE_NEUTRAL_MARGIN:
            verdict = "NEUTRAL"
        elif p_bull > p_bear:
            verdict = "BULLISH"
        else:
            verdict = "BEARISH"

        confidence = _clamp01(JUDGE_CONFIDENCE_BASE + abs(p_bull - p_bear) / JUDGE_CONFIDENCE_DIFF_DIVISOR)
        # J 向量对应文档中的 [p_bull,p_bear,q_bull,q_bear,e_bull,e_bear,c,d,a,rho]。
        score_vector = JudgeScoreVector(
            p_bull=p_bull,
            p_bear=p_bear,
            q_bull=_evidence_quality(bull_args),
            q_bear=_evidence_quality(bear_args),
            e_bull=_clamp01(model_summary.bull_mean / (model_summary.bull_mean + model_summary.bear_mean + DIVISION_EPSILON)),
            e_bear=_clamp01(model_summary.bear_mean / (model_summary.bull_mean + model_summary.bear_mean + DIVISION_EPSILON)),
            c=_coverage_score(transcript),
            d=_dispute_score(transcript),
            a=_attack_score(transcript),
            rho=confidence,
        )
        output = JudgeOutput(
            verdict=verdict,
            confidence=confidence,
            report="",
            score_vector=score_vector,
            consistency_flags=[],
        )
        output.consistency_flags = check_judge_consistency(output)
        output.report = _build_report(
            transcript=transcript,
            model_summary=model_summary,
            verdict=verdict,
            confidence=confidence,
            score_vector=score_vector,
            consistency_flags=output.consistency_flags,
        )
        return output


def _evidence_quality(arguments) -> float:
    if not arguments:
        return 0.0
    return _clamp01(mean(len(item.evidence) for item in arguments) / JUDGE_EVIDENCE_QUALITY_SCALE)


def _coverage_score(transcript: DebateTranscript) -> float:
    return _clamp01(len({item.role for item in transcript.arguments}) / JUDGE_ROLE_COVERAGE_SCALE)


def _dispute_score(transcript: DebateTranscript) -> float:
    return _clamp01(sum(1 for item in transcript.arguments if item.targets) / max(len(transcript.arguments), 1))


def _attack_score(transcript: DebateTranscript) -> float:
    by_id = {item.argument_id: item for item in transcript.arguments}
    cross_targets = 0
    for argument in transcript.arguments:
        for target_id in argument.targets:
            target = by_id.get(target_id)
            if target is not None and target.camp != argument.camp:
                cross_targets += 1
    return _clamp01(cross_targets / max(len(transcript.arguments), 1))


def _build_report(
    transcript: DebateTranscript,
    model_summary: ModelOutputSummary,
    verdict: str,
    confidence: float,
    score_vector: JudgeScoreVector,
    consistency_flags: list[str],
) -> str:
    """生成写入 JSON 的法官判决文本。"""
    flags = ", ".join(consistency_flags) if consistency_flags else "none"
    direction = "看涨" if verdict == "BULLISH" else "看跌" if verdict == "BEARISH" else "中性"
    return (
        f"Final judge decision for {transcript.block_id}: {verdict} ({direction}), "
        f"confidence={confidence:.3f}. "
        f"The judge compared the debate graph with the Bi-ODE/model summary. "
        f"Model bullish_probability={model_summary.bullish_probability:.3f}, "
        f"model_predicted_label={model_summary.predicted_label}, "
        f"ODE bull_bear_margin={model_summary.bull_bear_margin:.3f}. "
        f"Score vector: p_bull={score_vector.p_bull:.3f}, p_bear={score_vector.p_bear:.3f}, "
        f"q_bull={score_vector.q_bull:.3f}, q_bear={score_vector.q_bear:.3f}, "
        f"e_bull={score_vector.e_bull:.3f}, e_bear={score_vector.e_bear:.3f}, "
        f"coverage={score_vector.c:.3f}, depth={score_vector.d:.3f}, "
        f"attack/cross-check={score_vector.a:.3f}, rho={score_vector.rho:.3f}. "
        f"Consistency flags: {flags}."
    )


def _clamp01(value: float) -> float:
    return max(PROBABILITY_MIN, min(PROBABILITY_MAX, float(value)))




"""简化版双 agent 辩论编排器。

当前主流程只保留一个正方 agent 和一个反方 agent。旧的多角色闭环辩论
已经归档到 ``archive/multiagent/``。
"""

from __future__ import annotations

from agent.llm_client import DebateClient
from agent.prompts import roles_for_camp
from agent.schema import Argument, Camp, DebateTranscript
from config import DEFAULT_DEBATE_ROUNDS, REFLECTION_MAX_ROUNDS
from data.schema import CommentBlock
from agent.reflection import ReflectionSignal, should_continue_reflection
from profiles.user_profile import UserProfile


CAMPS: tuple[Camp, ...] = ("bull", "bear")


class DebateOrchestrator:
    """将一个 CommentBlock 和用户画像转换成双 agent DebateTranscript。"""

    def __init__(
        self,
        client: DebateClient,
        roles: tuple[str, ...] | None = None,
    ):
        self.client = client
        self.roles = roles

    def run(
        self,
        block: CommentBlock,
        profiles: dict[str, UserProfile],
        rounds: int = DEFAULT_DEBATE_ROUNDS,
        reflection_signal: ReflectionSignal | None = None,
        reflection_rounds: int = 0,
    ) -> DebateTranscript:
        """运行正方/反方双 agent 辩论。

        每轮只生成两条论点：
        1. ``bull_agent`` 给出看涨论点，首轮不回应任何目标，后续回应上一轮反方。
        2. ``bear_agent`` 给出看跌论点，并回应当前轮正方论点。

        这样每个样本的论点数为 ``rounds * 2``，便于阅读和控制 API 成本。
        """
        arguments: list[Argument] = []
        for round_index in range(1, rounds + 1):
            bull_targets = _latest_argument_ids(arguments, camp="bear", limit=1)
            bull = self._generate(
                block=block,
                profiles=profiles,
                camp="bull",
                role=self._role_for_camp("bull"),
                round_index=round_index,
                seq=1,
                prior_arguments=arguments,
                phase="initial_argument" if round_index == 1 else "rebuttal",
                available_target_ids=bull_targets,
            )
            arguments.append(bull)

            bear = self._generate(
                block=block,
                profiles=profiles,
                camp="bear",
                role=self._role_for_camp("bear"),
                round_index=round_index,
                seq=2,
                prior_arguments=arguments,
                phase="rebuttal",
                available_target_ids=[bull.argument_id],
            )
            arguments.append(bear)

            if _debate_converged(arguments):
                break

        if reflection_signal is not None and should_continue_reflection(reflection_signal):
            extra_rounds = min(max(reflection_rounds, 0), REFLECTION_MAX_ROUNDS)
            for offset in range(extra_rounds):
                round_index = len(arguments) // 2 + offset + 1
                arguments.extend(
                    self._reflection_pair(block, profiles, arguments, round_index, reflection_signal)
                )

        return DebateTranscript(
            block_id=block.block_id,
            t0=block.t0,
            rounds=rounds,
            arguments=arguments,
        )

    def add_reflection_rounds(
        self,
        block: CommentBlock,
        profiles: dict[str, UserProfile],
        transcript: DebateTranscript,
        signal: ReflectionSignal,
        reflection_rounds: int = 1,
    ) -> DebateTranscript:
        """Append debater reflection supplements without regenerating earlier debate turns."""
        if reflection_rounds <= 0 or not should_continue_reflection(signal):
            return transcript
        arguments = list(transcript.arguments)
        extra_rounds = min(reflection_rounds, REFLECTION_MAX_ROUNDS)
        for offset in range(extra_rounds):
            round_index = len(arguments) // 2 + offset + 1
            arguments.extend(self._reflection_pair(block, profiles, arguments, round_index, signal))
        return DebateTranscript(
            block_id=transcript.block_id,
            t0=transcript.t0,
            rounds=transcript.rounds + extra_rounds,
            arguments=arguments,
        )

    def _role_for_camp(self, camp: Camp) -> str:
        if self.roles:
            return self.roles[0]
        return roles_for_camp(camp)[0]

    def _generate(
        self,
        block: CommentBlock,
        profiles: dict[str, UserProfile],
        camp: Camp,
        role: str,
        round_index: int,
        seq: int,
        prior_arguments: list[Argument],
        phase: str,
        available_target_ids: list[str],
    ) -> Argument:
        argument = self.client.generate_argument(
            block=block,
            profiles=profiles,
            camp=camp,
            role=role,
            round_index=round_index,
            seq=seq,
            prior_arguments=prior_arguments,
            phase=phase,
            available_target_ids=available_target_ids,
        )
        argument.phase = phase
        argument.t_index = _argument_time(argument, max_seq=2)
        return argument

    def _reflection_pair(
        self,
        block: CommentBlock,
        profiles: dict[str, UserProfile],
        arguments: list[Argument],
        round_index: int,
        signal: ReflectionSignal,
    ) -> list[Argument]:
        """按 v3 反思机制追加补充论证；LLM 不接收真实 label，只接收弱项反馈。"""
        phase = f"reflection_supplement:{','.join(signal.weak_dims[:3]) or 'general'}"
        bull_targets = _latest_argument_ids(arguments, camp="bear", limit=2)
        bull = self._generate(
            block=block,
            profiles=profiles,
            camp="bull",
            role=self._role_for_camp("bull"),
            round_index=round_index,
            seq=1,
            prior_arguments=arguments,
            phase=phase,
            available_target_ids=bull_targets,
        )
        bear = self._generate(
            block=block,
            profiles=profiles,
            camp="bear",
            role=self._role_for_camp("bear"),
            round_index=round_index,
            seq=2,
            prior_arguments=[*arguments, bull],
            phase=phase,
            available_target_ids=[bull.argument_id, *_latest_argument_ids(arguments, camp="bull", limit=1)],
        )
        return [bull, bear]


def _latest_argument_ids(arguments: list[Argument], camp: Camp, limit: int) -> list[str]:
    return [argument.argument_id for argument in arguments if argument.camp == camp][-limit:]


def _argument_time(argument: Argument, max_seq: int) -> float:
    return float(argument.round - 1) + float(argument.seq - 1) / max(max_seq, 1)


def _debate_converged(arguments: list[Argument]) -> bool:
    """保守收敛检测：仅在最新一轮没有 target 时提前停止。"""
    if len(arguments) < 2:
        return False
    latest = arguments[-2:]
    return all(not item.target_args for item in latest)

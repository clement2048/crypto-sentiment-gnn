"""Debate orchestration: CommentBlock + profiles -> DebateTranscript.

This module sits between the data/profile layer and the graph/model layer.
Its job is not to judge the sample; it only asks the bull and bear LLM agents
to produce structured `Argument` objects.

Per-sample data flow:

1. Input from upstream
   - `CommentBlock`: post text, root comment, replies, product, t0.
   - `profiles`: time-safe user profiles for authors in the block.
   - `rounds`: maximum outer debate rounds.

2. Round execution
   Each round generates at most two arguments:
   - bull argument, `seq=1`;
   - bear argument, `seq=2`, usually targeting the current bull argument.
   Later bull arguments target the latest bear argument.

3. LLM client boundary
   `_generate(...)` delegates actual text generation to `DebateClient`.
   Provider-specific clients convert the block/profile/prior-argument objects
   into JSON prompts and parse the LLM response back into `Argument`.

4. Metadata normalization
   The orchestrator fixes phase and `t_index` after generation. This keeps the
   graph time axis stable even if the LLM returns slightly inconsistent fields.

5. Output to downstream
   The final `DebateTranscript` is consumed by:
   - `build_debate_graph(...)`, which turns `target_args` into `interact` edges;
   - Judge clients, which read claims, evidence, and `target_args` as debate logic;
   - optional reflection loops, which append more arguments without deleting
     earlier ones.
"""

from __future__ import annotations

from agent.llm_client import DebateClient
from agent.schema import Argument, Camp, DebateTranscript
from config import DEFAULT_DEBATE_ROUNDS, REFLECTION_MAX_ROUNDS
from data.schema import CommentBlock
from agent.reflection import ReflectionSignal, should_continue_reflection
from profiles.user_profile import UserProfile


class DebateOrchestrator:
    """Convert one sample and its profiles into a structured debate transcript."""

    def __init__(self, client: DebateClient):
        self.client = client

    def run(
        self,
        block: CommentBlock,
        profiles: dict[str, UserProfile],
        rounds: int = DEFAULT_DEBATE_ROUNDS,
    ) -> DebateTranscript:
        """Run the initial bull/bear debate.

        Output growth pattern:
        - round 1:
          bull has no prior bear target, so it opens the debate;
          bear responds to that bull argument.
        - round >= 2:
          bull responds to the latest bear argument;
          bear responds to the current bull argument.

        The transcript stores arguments in chronological generation order. This
        ordering is later used by graph construction and by the LLM Judge.
        """
        arguments: list[Argument] = []
        for round_index in range(1, rounds + 1):
            # Bull reads all prior arguments but is only allowed to target the
            # latest bear-side argument. The allowed target list is passed to
            # the LLM prompt and enforced again after parsing by provider code.
            bull_target_ids = _latest_argument_ids(arguments, camp="bear", limit=1)
            bull = self._generate(
                block=block,
                profiles=profiles,
                camp="bull",
                role="bull_agent",
                round_index=round_index,
                seq=1,
                prior_arguments=arguments,
                phase="initial_argument" if round_index == 1 else "rebuttal",
                available_target_ids=bull_target_ids,
            )
            arguments.append(bull)

            # Bear sees the newly generated bull argument in prior_arguments and
            # uses its argument_id as target_args. This gives every normal round a simple
            # bull -> bear local interaction that can become an `interact` edge.
            bear = self._generate(
                block=block,
                profiles=profiles,
                camp="bear",
                role="bear_agent",
                round_index=round_index,
                seq=2,
                prior_arguments=arguments,
                phase="rebuttal",
                available_target_ids=[bull.argument_id],
            )
            arguments.append(bear)

            # Early stop is intentionally conservative. It only stops when both
            # latest arguments have no target_args, meaning the LLM produced no
            # actionable interaction for this round.
            if _debate_converged(arguments):
                break

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
        """Append reflection supplements to an existing transcript.

        Data flow:
        existing transcript + safe reflection signal
        -> one or more bull/bear supplement pairs
        -> new transcript with old arguments preserved.

        Earlier arguments are not regenerated because they may already be used
        in cached graph/model/Judge artifacts or human case-study reports.
        """
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
        # This is the single gateway from orchestrator state to provider-specific
        # LLM generation. The caller supplies:
        # - current sample (`block`);
        # - time-safe profiles;
        # - all prior arguments visible to this agent;
        # - a bounded list of legal target argument ids.
        # The provider returns an `Argument`; metadata is then overwritten below
        # so downstream graph construction receives stable ids/time fields.
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
        """Generate one bull/bear supplement pair for Judge-guided reflection."""
        # The phase string carries a compact description of the weakness that
        # triggered reflection. It becomes node metadata later, but it is not a
        # label and it does not contain future price information.
        phase = f"reflection_supplement:{','.join(signal.weak_dims[:3]) or 'general'}"

        # Bull supplement usually repairs or expands the bull side while looking
        # at the latest bear-side pressure points.
        bull_target_ids = _latest_argument_ids(arguments, camp="bear", limit=2)
        bull = self._generate(
            block=block,
            profiles=profiles,
            camp="bull",
            role="bull_agent",
            round_index=round_index,
            seq=1,
            prior_arguments=arguments,
            phase=phase,
            available_target_ids=bull_target_ids,
        )

        # Bear supplement then responds to the new bull supplement and optionally
        # the previous bull argument. This keeps reflection graph growth connected
        # to existing debate chains rather than creating isolated nodes.
        bear = self._generate(
            block=block,
            profiles=profiles,
            camp="bear",
            role="bear_agent",
            round_index=round_index,
            seq=2,
            prior_arguments=[*arguments, bull],
            phase=phase,
            available_target_ids=[bull.argument_id, *_latest_argument_ids(arguments, camp="bull", limit=1)],
        )
        return [bull, bear]


def _latest_argument_ids(arguments: list[Argument], camp: Camp, limit: int) -> list[str]:
    """Return recent argument ids for one camp, preserving chronological order."""
    return [argument.argument_id for argument in arguments if argument.camp == camp][-limit:]


def _argument_time(argument: Argument, max_seq: int) -> float:
    """Map discrete round/seq to a continuous debate-time coordinate."""
    return float(argument.round - 1) + float(argument.seq - 1) / max(max_seq, 1)


def _debate_converged(arguments: list[Argument]) -> bool:
    """保守收敛检测：仅在最新一轮没有 target 时提前停止。"""
    if len(arguments) < 2:
        return False
    latest = arguments[-2:]
    return all(not item.target_args for item in latest)

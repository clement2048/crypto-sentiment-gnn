"""多智能体辩论编排器。

它负责“谁在第几轮、第几个阶段发言”，不负责真实 LLM 调用细节。
"""

from __future__ import annotations

from agent.llm_client import DebateClient
from agent.prompts import core_roles_for_camp, reflection_role_for_camp, roles_for_camp
from agent.schema import Argument, Camp, DebateTranscript
from config import (
    DEFAULT_COUNTER_DISCUSSION_ROUNDS,
    DEFAULT_DEBATE_ROUNDS,
    DEFAULT_INTRA_DISCUSSION_ROUNDS,
)
from data.schema import CommentBlock
from profiles.user_profile import UserProfile


DEFAULT_ROLES = None
CAMPS: tuple[Camp, ...] = ("bull", "bear")


class DebateOrchestrator:
    """将一个 CommentBlock 和用户画像转换成 DebateTranscript。"""

    def __init__(
        self,
        client: DebateClient,
        roles: tuple[str, ...] | None = DEFAULT_ROLES,
        intra_discussion_rounds: int = DEFAULT_INTRA_DISCUSSION_ROUNDS,
        counter_discussion_rounds: int = DEFAULT_COUNTER_DISCUSSION_ROUNDS,
    ):
        self.client = client
        self.roles = roles
        self.intra_discussion_rounds = max(0, int(intra_discussion_rounds))
        self.counter_discussion_rounds = max(0, int(counter_discussion_rounds))

    def run(
        self,
        block: CommentBlock,
        profiles: dict[str, UserProfile],
        rounds: int = DEFAULT_DEBATE_ROUNDS,
    ) -> DebateTranscript:
        """按论文 v4 的五阶段流程运行多轮 bull/bear 辩论。

        每个 round 包含：
        1. initial_argument：双方核心 agent 独立生成初始论点。
        2. intra_reflection：双方反思 agent 阅读本阵营论点，指出漏洞和可强化点。
        3. intra_response：双方核心 agent 必须回应本阵营 reflection，形成显式内部讨论。
        4. cross_response：双方核心 agent 针对对方当前论点回应/反驳。
        5. counter_reflection：双方 reflection 消化对方攻击，指出需要修补的弱点。
        6. counter_rebuttal：双方核心 agent 必须回应 counter_reflection，并再反驳对方攻击。
        7. reflection_summary：双方反思 agent 总结本轮论证质量。

        时间顺序由 round/seq/phase 和图属性表达，不再生成 precede 边。
        """
        arguments: list[Argument] = []
        for round_index in range(1, rounds + 1):
            seq = 0
            round_start = len(arguments)
            cross_ids: dict[Camp, list[str]] = {"bull": [], "bear": []}

            # 阶段 1：双方核心 agent 独立提出初始论点。
            for camp in CAMPS:
                for role in self._core_roles_for_camp(camp):
                    seq += 1
                    argument = self._generate(
                        block=block,
                        profiles=profiles,
                        camp=camp,
                        role=role,
                        round_index=round_index,
                        seq=seq,
                        prior_arguments=arguments,
                        phase="initial_argument",
                        available_target_ids=[],
                    )
                    arguments.append(argument)

            # 阶段 2-3：阵营内讨论。reflection 先批评/整合，核心 agent 再显式回应。
            seq = self._run_internal_discussion(
                block=block,
                profiles=profiles,
                arguments=arguments,
                round_index=round_index,
                seq=seq,
                round_start=round_start,
                reflection_phase="intra_reflection",
                response_phase="intra_response",
                discussion_rounds=self.intra_discussion_rounds,
                opponent_target_ids=None,
            )

            # 阶段 4：跨阵营回应，目标限制为对方本轮已产生的论点。
            for camp in CAMPS:
                opponent = _opponent(camp)
                opponent_current_ids = _argument_ids(arguments[round_start:], camp=opponent)
                for role in self._core_roles_for_camp(camp):
                    seq += 1
                    argument = self._generate(
                        block=block,
                        profiles=profiles,
                        camp=camp,
                        role=role,
                        round_index=round_index,
                        seq=seq,
                        prior_arguments=arguments,
                        phase="cross_response",
                        available_target_ids=opponent_current_ids,
                    )
                    arguments.append(argument)
                    cross_ids[camp].append(argument.argument_id)

            # 阶段 5-6：被攻击后的阵营内讨论，再由核心 agent 做再反驳。
            seq = self._run_internal_discussion(
                block=block,
                profiles=profiles,
                arguments=arguments,
                round_index=round_index,
                seq=seq,
                round_start=round_start,
                reflection_phase="counter_reflection",
                response_phase="counter_rebuttal",
                discussion_rounds=self.counter_discussion_rounds,
                opponent_target_ids={
                    "bull": cross_ids["bear"],
                    "bear": cross_ids["bull"],
                },
            )

            # 阶段 7：反思总结，综合本轮全部论点。
            for camp in CAMPS:
                seq += 1
                argument = self._generate(
                    block=block,
                    profiles=profiles,
                    camp=camp,
                    role=reflection_role_for_camp(camp),
                    round_index=round_index,
                    seq=seq,
                    prior_arguments=arguments,
                    phase="reflection_summary",
                    available_target_ids=_argument_ids(arguments[round_start:]),
                )
                arguments.append(argument)

        return DebateTranscript(
            block_id=block.block_id,
            t0=block.t0,
            rounds=rounds,
            arguments=arguments,
        )

    def _roles_for_camp(self, camp: Camp) -> tuple[str, ...]:
        """默认使用论文 v4 的阵营专属四角色；测试可用 roles 覆盖。"""
        return self.roles if self.roles is not None else roles_for_camp(camp)

    def _core_roles_for_camp(self, camp: Camp) -> tuple[str, ...]:
        """返回本轮参与提出/反驳论点的角色。"""
        if self.roles is not None:
            return tuple(role for role in self.roles if role != "reflection_agent")
        return core_roles_for_camp(camp)

    def _run_internal_discussion(
        self,
        block: CommentBlock,
        profiles: dict[str, UserProfile],
        arguments: list[Argument],
        round_index: int,
        seq: int,
        round_start: int,
        reflection_phase: str,
        response_phase: str,
        discussion_rounds: int,
        opponent_target_ids: dict[Camp, list[str]] | None,
    ) -> int:
        """运行“reflection -> 核心 agent 回应”的阵营内讨论闭环。"""
        for _discussion_index in range(discussion_rounds):
            for camp in CAMPS:
                same_camp_ids = _argument_ids(arguments[round_start:], camp=camp)
                reflection_targets = same_camp_ids
                if opponent_target_ids is not None:
                    reflection_targets = opponent_target_ids.get(camp, [])

                seq += 1
                reflection = self._generate(
                    block=block,
                    profiles=profiles,
                    camp=camp,
                    role=reflection_role_for_camp(camp),
                    round_index=round_index,
                    seq=seq,
                    prior_arguments=arguments,
                    phase=reflection_phase,
                    available_target_ids=reflection_targets,
                )
                arguments.append(reflection)

                # 核心 agent 的 target 列表把 reflection 放在最后，保证 LLM 会优先看到它。
                core_targets = list(reflection_targets) + [reflection.argument_id]
                for role in self._core_roles_for_camp(camp):
                    seq += 1
                    response = self._generate(
                        block=block,
                        profiles=profiles,
                        camp=camp,
                        role=role,
                        round_index=round_index,
                        seq=seq,
                        prior_arguments=arguments,
                        phase=response_phase,
                        available_target_ids=core_targets,
                    )
                    arguments.append(response)
        return seq

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
        # 编排器只固定元数据和可选 target 范围；具体论证内容由 client 生成。
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
        return argument


def _opponent(camp: Camp) -> Camp:
    return "bear" if camp == "bull" else "bull"


def _argument_ids(arguments: list[Argument], camp: Camp | None = None) -> list[str]:
    if camp is None:
        return [argument.argument_id for argument in arguments]
    return [argument.argument_id for argument in arguments if argument.camp == camp]

"""Judge prototype for debate transcripts.

在线 judge client 类现在物理位置在 ``agent/`` 包下(因为协议与 debate client
共享),需要直接 import 时请走:

    from agent.anthropic_compatible import DeepSeekJudgeClient, MiniMaxJudgeClient
    from agent.openai_compatible import BailianJudgeClient

或者通过工厂间接获取(推荐,避免循环 import):

    from judge import create_judge_client
    client = create_judge_client("minimax")
"""

from judge.client_factory import create_judge_client
from judge.judge_schema import JudgeOutput, JudgeScoreVector

__all__ = [
    "JudgeOutput",
    "JudgeScoreVector",
    "create_judge_client",
]


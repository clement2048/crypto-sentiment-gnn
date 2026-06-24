"""Judge prototype for debate transcripts."""

from judge.client_factory import create_judge_client
from judge.judge_schema import JudgeOutput, JudgeScoreVector
from judge.report_parser import reflection_signal_from_judge

__all__ = [
    "JudgeOutput",
    "JudgeScoreVector",
    "create_judge_client",
    "reflection_signal_from_judge",
]

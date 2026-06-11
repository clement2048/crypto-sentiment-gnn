"""config.py 的公开模板。

真实 config.py 已被 .gitignore，不会推到远端。
clone 后请执行：

    cp config.example.py config.py
    cp .env.example .env   # 然后填入真实 API key

本文件只包含常量超参数和 .env 加载逻辑，无任何敏感信息，可安全上传。
"""

from __future__ import annotations
import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent


# -----------------------------
# .env 自动加载
# -----------------------------
# 项目根目录下的 .env 会作为环境变量源，优先级低于 shell 里已经 export 的同名变量。
# 这样既支持 .env 文件，也兼容老的 PowerShell `$env:DEEPSEEK_API_KEY=...` 临时覆盖。
def _load_env_file() -> None:
    env_path = PROJECT_ROOT / ".env"
    if not env_path.exists():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, _, value = stripped.partition("=")
        os.environ.setdefault(key.strip(), value.strip())


_load_env_file()


# -----------------------------
# 数据读取与时间切分
# -----------------------------

# 默认数据来源；命令行仍可用 --input 覆盖。
DEFAULT_INPUT_PATH = str(PROJECT_ROOT / "dataset" / "final.jsonl")

# 判断 Unix timestamp 是"秒"还是"毫秒"的工程阈值。
TIMESTAMP_MILLISECONDS_THRESHOLD = 10_000_000_000

# 时间顺序切分比例。注意这里不能随机打乱，否则会破坏时间安全。
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15
SPLIT_RATIO_EPSILON = 1e-6


# -----------------------------
# 用户画像
# -----------------------------

# 冷启动用户没有历史记录时使用的默认画像。
COLD_START_STANCE_BIAS = 0.0
COLD_START_CONSISTENCY = 0.5
COLD_START_ACTIVITY = 0.0
COLD_START_INFLUENCE = 0.0
COLD_START_REACTION_CONSISTENCY = 0.0
COLD_START_EMOTION_STABILITY = 0.0

# 只有一条历史情绪时，暂时认为情绪稳定性最高。
SINGLE_HISTORY_EMOTION_STABILITY = 1.0

# 历史评论方向验证中，p1 == p0 是否算作上涨方向。
PRICE_TIE_COUNTS_AS_BULLISH = True


# -----------------------------
# Mock 辩论 agent
# -----------------------------

# 默认辩论轮数：每轮 bull/bear 阵营各角色发言。
DEFAULT_DEBATE_ROUNDS = 2

# 每轮辩论中，阵营内"reflection -> 核心 agent 回应"的讨论次数。
DEFAULT_INTRA_DISCUSSION_ROUNDS = 1

# 每轮跨阵营攻击后，阵营内"reflection 消化攻击 -> 核心 agent 再反驳"的次数。
DEFAULT_COUNTER_DISCUSSION_ROUNDS = 1

# 每个新论点最多回应最近多少个对方论点。
MOCK_DEBATE_TARGET_LIMIT = 2

# Mock 置信度合成公式。这里只是离线替身规则，不是实验结论。
MOCK_CONFIDENCE_BASE = 0.55
MOCK_CONFIDENCE_PRIOR_CAP = 4
MOCK_CONFIDENCE_PRIOR_BONUS = 0.02
MOCK_CONFIDENCE_PROFILE_BONUS = 0.08
MOCK_CONFIDENCE_MAX = 0.92

# Mock evidence relevance。真实 LLM 接入后，这部分应该由模型或评分器给出。
MOCK_ROOT_COMMENT_RELEVANCE = 0.8
MOCK_POST_RELEVANCE = 0.6
MOCK_PROFILE_RELEVANCE = 0.5


# -----------------------------
# 在线 LLM 辩论接口
# -----------------------------

DEEPSEEK_ANTHROPIC_BASE_URL = "https://api.deepseek.com/anthropic"
DEEPSEEK_MODEL = "deepseek-v4-pro"
# 真实 key 应放在 .env 里（或用 $env:DEEPSEEK_API_KEY）。config.py 本体不再硬编码。
DEEPSEEK_API_KEY_ENV = os.getenv("DEEPSEEK_API_KEY", "")
DEEPSEEK_FALLBACK_API_KEY_ENV = "ANTHROPIC_API_KEY"
DEEPSEEK_MAX_TOKENS = 1200
DEEPSEEK_TEMPERATURE = 0.2
DEEPSEEK_TIMEOUT_SECONDS = 120.0
DEEPSEEK_HTTP_RETRIES = 3
DEEPSEEK_THINKING_TYPE = "disabled"
DEEPSEEK_ANTHROPIC_VERSION = "2023-06-01"
DEEPSEEK_CACHE_ENABLED = True
DEEPSEEK_CACHE_DIR = PROJECT_ROOT / "outputs" / "llm_cache" / "deepseek"


# -----------------------------
# MiniMax Anthropic 兼容接口
# (https://platform.minimax.io/docs/token-plan/quickstart)
# 与官方 anthropic SDK 共用 ANTHROPIC_BASE_URL / ANTHROPIC_API_KEY；
# 备用别名 MINIMAX_API_KEY，便于本地直接命名。
# -----------------------------

MINIMAX_ANTHROPIC_BASE_URL = "https://api.minimax.io/anthropic"
MINIMAX_MODEL = "MiniMax-M3"
MINIMAX_API_KEY_ENV = os.getenv("ANTHROPIC_API_KEY", "ANTHROPIC_API_KEY")
MINIMAX_FALLBACK_API_KEY_ENV = "MINIMAX_API_KEY"
MINIMAX_MAX_TOKENS = 1000
MINIMAX_TEMPERATURE = 0.2
MINIMAX_TIMEOUT_SECONDS = 120.0
MINIMAX_HTTP_RETRIES = 3
MINIMAX_ANTHROPIC_VERSION = "2023-06-01"
MINIMAX_CACHE_ENABLED = True
MINIMAX_CACHE_DIR = PROJECT_ROOT / "outputs" / "llm_cache" / "minimax"


# -----------------------------
# 阿里云百炼 OpenAI 兼容接口
# -----------------------------

BAILIAN_OPENAI_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
BAILIAN_MODEL = "deepseek-v4-flash"
# 同上：真实 key 应放在 .env 里（或用 $env:BAILIAN_API_KEY）。
BAILIAN_API_KEY_ENV = os.getenv("BAILIAN_API_KEY", "")
BAILIAN_MAX_TOKENS = 1200
BAILIAN_TEMPERATURE = 0.2
BAILIAN_ENABLE_THINKING = False
BAILIAN_TIMEOUT_SECONDS = 120.0
BAILIAN_HTTP_RETRIES = 3
BAILIAN_CACHE_ENABLED = True
BAILIAN_CACHE_DIR = PROJECT_ROOT / "outputs" / "llm_cache" / "bailian"


# -----------------------------
# 图张量特征
# -----------------------------
# 当前原型使用 8 维手工结构特征；后续可拼接文本 embedding。
NODE_FEATURE_DIM = 8

# 把离散结构数值压到 0~1 的经验尺度。
COMMENT_DEPTH_SCALE = 10.0
DEBATE_ROUND_SCALE = 10.0
DEBATE_SEQUENCE_SCALE = 100.0


# -----------------------------
# 图模型 / Bi-ODE 原型
# -----------------------------

DEFAULT_RELATIONS = ["reply", "cite", "support", "attack", "respond", "propose"]

MODEL_HIDDEN_DIM = 16
CALIBRATOR_HIDDEN_DIM = 32
ODE_STEPS = 4
EULER_STEP_SIZE = 0.25
ODE_TERMINAL_TIME = 5.0
ODE_METHOD = "euler"
ODE_RTOL = 0.01
ODE_ATOL = 0.001
ODE_USE_ADJOINT = False
ODE_SOLVER_BACKEND = "torchdiffeq"
RELATION_WEIGHT_INIT = 0.1
ODE_DROPOUT = 0.0
ODE_USE_GRAPH = True
ODE_USE_CONTROL = True

# 概率转标签的阈值：>= 0.5 记为看涨，否则看跌。
CLASSIFICATION_THRESHOLD = 0.5


# -----------------------------
# 原型训练与命令行默认值
# -----------------------------

DEFAULT_LIMIT_BLOCKS = 5
TRAIN_PROTOTYPE_LIMIT_BLOCKS = 10
TRAIN_PROTOTYPE_EPOCHS = 20
FULL_PIPELINE_TRAIN_EPOCHS = 0
LEARNING_RATE = 0.01
PRINT_SAMPLES = 5


# -----------------------------
# Mock 法官
# -----------------------------

# ModelAwareMockJudge 中，p_bull 融合三类信号的权重。
JUDGE_MODEL_WEIGHT = 0.60
JUDGE_DEBATE_WEIGHT = 0.25
JUDGE_MARGIN_WEIGHT = 0.15

# 把 bull/bear 阵营平均置信度差值映射到 0~1 的经验公式。
JUDGE_DEBATE_CENTER = 0.5
JUDGE_DEBATE_DIFF_DIVISOR = 1.0

# ODE margin 只取方向时使用的二值映射。
JUDGE_MARGIN_BULL_VALUE = 1.0
JUDGE_MARGIN_BEAR_VALUE = 0.0

# p_bull 与 p_bear 太接近时判为 NEUTRAL。
JUDGE_NEUTRAL_MARGIN = 0.08

# confidence = base + abs(p_bull - p_bear) / divisor。
JUDGE_CONFIDENCE_BASE = 0.5
JUDGE_CONFIDENCE_DIFF_DIVISOR = 2.0

# J 向量中 q/c 等规则分数的归一化尺度。
JUDGE_EVIDENCE_QUALITY_SCALE = 3.0
JUDGE_ROLE_COVERAGE_SCALE = 4.0

# 旧 MockJudgeAgent 仍保留给兼容测试，不是完整 v2 pipeline 的最终法官。
LEGACY_JUDGE_NEUTRAL_MARGIN = 0.05
LEGACY_JUDGE_CONFIDENCE_BASE = 0.5

# consistency 检查用的规则阈值。
NEUTRAL_SCORE_IMBALANCE_THRESHOLD = 0.25

# 防止除零的工程常数。
DIVISION_EPSILON = 1e-6


# -----------------------------
# 通用概率边界
# -----------------------------

PROBABILITY_MIN = 0.0
PROBABILITY_MAX = 1.0
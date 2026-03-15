"""
全局配置文件
集中管理 API Key、模型选择、路径配置
"""

import os
from pathlib import Path

# === 路径配置 ===
ROOT_DIR = Path(__file__).parent.parent.absolute()
DATA_DIR = ROOT_DIR / "data"
SKILLS_DIR = ROOT_DIR / "skills"
SRC_DIR = ROOT_DIR / "src"

# Benchmark 数据路径
BENCHMARK_DIR = DATA_DIR / "benchmark"
BENCHMARK_TRAIN_PATH = BENCHMARK_DIR / "train.jsonl"
BENCHMARK_VAL_PATH = BENCHMARK_DIR / "val.jsonl"
BENCHMARK_TEST_PATH = BENCHMARK_DIR / "test.jsonl"

# RAG 检索库路径
RETRIEVAL_DB_DIR = DATA_DIR / "retrieval_db"
RETRIEVAL_CORPUS_DIR = DATA_DIR / "retrieval_corpus"

# Skill 版本路径
DEFAULT_ACTIVE_SKILL_VERSION = "teen_detector_v1"
ACTIVE_SKILL_POINTER = SKILLS_DIR / "active_version.txt"


def get_active_skill_version() -> str:
    """读取当前激活的 skill 版本；若指针不存在则回退到默认版本。"""
    if ACTIVE_SKILL_POINTER.exists():
        version = ACTIVE_SKILL_POINTER.read_text(encoding="utf-8").strip()
        if version and (SKILLS_DIR / version / "skill.md").exists():
            return version
    return DEFAULT_ACTIVE_SKILL_VERSION


def get_active_skill_dir() -> Path:
    """返回当前激活 skill 的目录。"""
    return SKILLS_DIR / get_active_skill_version()


def get_active_skill_path() -> Path:
    """返回当前激活 skill 的 markdown 文件路径。"""
    return get_active_skill_dir() / "skill.md"


def set_active_skill_version(version: str) -> Path:
    """将给定版本写入激活指针，供在线执行体默认加载。"""
    skill_dir = SKILLS_DIR / version
    skill_path = skill_dir / "skill.md"
    if not skill_path.exists():
        raise FileNotFoundError(f"Skill 版本不存在: {skill_path}")
    ACTIVE_SKILL_POINTER.write_text(version, encoding="utf-8")
    return skill_path


ACTIVE_SKILL_DIR = get_active_skill_dir()
SKILL_MD_PATH = get_active_skill_path()

# 用户记忆数据库
USER_MEMORY_DB_PATH = DATA_DIR / "user_memory.db"

# === API 配置 ===
AIHUBMIX_API_KEY = os.getenv("AIHUBMIX_API_KEY")
AIHUBMIX_BASE_URL = "https://aihubmix.com/v1"

# 兼容旧变量名
API_KEY = AIHUBMIX_API_KEY
API_BASE_URL = AIHUBMIX_BASE_URL

# === 模型配置 ===
# 分层模型选择（当前统一使用 gemini-3-flash-preview，后续可分离）
EXECUTOR_MODEL = "gemini-3-flash-preview"   # 在线执行体：快、便宜
EVALUATOR_MODEL = "gemini-3-flash-preview"  # 裁判：稳定
OPTIMIZER_MODEL = "gemini-3-flash-preview"  # 优化师：聪明（后续可换 claude/gpt-4o）
EMBEDDING_MODEL = "text-embedding-3-small"  # Embedding 模型

# === 逻辑配置 ===
RAG_TOP_K = 3           # RAG 检索时返回的相似样本数
RAG_THRESHOLD = 0.5     # 相似度阈值

LLM_MAX_RETRIES = 3     # LLM 调用最大重试次数
LLM_RETRY_DELAY = 1.0   # 重试间隔（秒）
LLM_TIMEOUT = 120       # 超时时间（秒）

# === 标签定义 ===
RISK_LEVELS = ["High", "Medium", "Low"]
EDUCATION_STAGES = ["小学", "初中", "高中", "大学", "成人专业"]


def validate_config():
    """验证配置完整性"""
    if not API_KEY:
        raise ValueError(
            "AIHUBMIX_API_KEY 环境变量未设置。"
            "请设置: set AIHUBMIX_API_KEY=your-api-key"
        )
    return True

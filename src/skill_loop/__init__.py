# 模块说明：
# - 当前 loop 框架的公共入口导出。
# - Mode A 和 Mode B 的主线能力都从这里对外暴露。

from .compare import compare_reports
from .direct_runner import DirectRunnerConfig, DirectSkillRunner
from .judge import (
    DEFAULT_PROTECTED_COUNT,
    build_judge_artifacts,
    calc_default_max_errors,
    judge_run_artifacts,
)
from .loop import SkillAgentLoop, SkillAgentLoopConfig
from .packaging import install_skill_snapshot, package_skill_version, unpack_skill_package, validate_skill_source
from .schema_consistency import validate_skill_schema_contract
from .runner import CodexRunnerConfig, CodexSkillRunner
from .versioning import (
    build_candidate_version_name,
    build_stable_version_name,
    ensure_version_snapshot,
    next_patch_version_name,
    next_available_candidate_version_name,
    parse_version_name,
    publish_candidate_to_stable,
)

__all__ = [
    "DEFAULT_PROTECTED_COUNT",
    "CodexRunnerConfig",
    "CodexSkillRunner",
    "DirectRunnerConfig",
    "DirectSkillRunner",
    "SkillAgentLoop",
    "SkillAgentLoopConfig",
    "build_candidate_version_name",
    "build_judge_artifacts",
    "build_stable_version_name",
    "calc_default_max_errors",
    "compare_reports",
    "ensure_version_snapshot",
    "judge_run_artifacts",
    "install_skill_snapshot",
    "next_available_candidate_version_name",
    "next_patch_version_name",
    "package_skill_version",
    "parse_version_name",
    "publish_candidate_to_stable",
    "unpack_skill_package",
    "validate_skill_source",
    "validate_skill_schema_contract",
]

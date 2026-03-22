# 模块说明：
# - formal runtime 适配层的公共导出。
# - 给演示页和人工 inspection 工具使用。

from .skill_runtime_adapter import (
    FORMAL_SKILL_VERSION,
    analyze_enriched_formal,
    analyze_formal_payload,
    analyze_multi_session_formal,
    analyze_multi_session_formal_auto,
    analyze_single_session_formal,
    analyze_single_session_formal_auto,
    build_formal_enriched_payload,
    build_formal_multi_session_payload,
    build_formal_single_session_payload,
    enrich_multi_session_context,
    enrich_single_session_context,
    get_builtin_retrieval_script_path,
    get_builtin_time_script_path,
    get_formal_executor,
    get_formal_skill_dir,
    get_formal_skill_path,
)

__all__ = [
    "FORMAL_SKILL_VERSION",
    "get_formal_skill_path",
    "get_formal_skill_dir",
    "get_formal_executor",
    "get_builtin_time_script_path",
    "get_builtin_retrieval_script_path",
    "build_formal_single_session_payload",
    "build_formal_multi_session_payload",
    "build_formal_enriched_payload",
    "enrich_single_session_context",
    "enrich_multi_session_context",
    "analyze_single_session_formal",
    "analyze_single_session_formal_auto",
    "analyze_multi_session_formal",
    "analyze_multi_session_formal_auto",
    "analyze_enriched_formal",
    "analyze_formal_payload",
]

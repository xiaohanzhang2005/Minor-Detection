# 模块说明：
# - 旧 executor 风格接口的导出层。
# - 给旧 demo、旧脚本和 formal runtime 适配层复用。

from .executor import (
    ExecutorSkill,
    get_executor,
    analyze_conversation,
    analyze_payload,
    analyze_text,
    analyze_with_rag,
    analyze_with_memory,
)
from .payload_builder import (
    build_single_session_payload,
    build_multi_session_payload,
    build_enriched_payload,
)

__all__ = [
    "ExecutorSkill",
    "get_executor",
    "analyze_conversation",
    "analyze_payload",
    "analyze_text",
    "analyze_with_rag",
    "analyze_with_memory",
    "build_single_session_payload",
    "build_multi_session_payload",
    "build_enriched_payload",
]

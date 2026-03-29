# 模块说明：
# - 把 enriched payload 接到 bundled skill 的 formal runtime 桥接层。
# - 适合 demo 和调试，不是当前 A 或 B 主链执行层。

"""
正式 Skill 的运行时对接层。

目标：
- 为新 Skill 提供单一正式入口
- 尽量不改动旧 pipeline 和旧执行入口
- 将“构造 payload”“内部增强”“调用正式 Skill”集中到一处
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
import importlib
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.config import SKILLS_DIR, resolve_skill_markdown_path
from src.executor import ExecutorSkill, build_enriched_payload, build_multi_session_payload, build_single_session_payload
from src.models import AnalysisPayload, SkillOutput


FORMAL_SKILL_VERSION = "minor-detection"
TIME_SCRIPT_NAME = "extract_time_features.py"
RETRIEVAL_SCRIPT_NAME = "retrieve_cases.py"
DEFAULT_RETRIEVE_TOP_K = 3
RETRIEVAL_ASSETS_DIRNAME = "retrieval_assets"
RAG_MODE_EXTERNAL = "external_rag"
RAG_MODE_INTERNAL = "internal_rag"
RAG_MODE_NONE = "no_rag"
TIMESTAMP_FIELD_CANDIDATES = ("timestamp", "time", "created_at", "datetime")
TIMESTAMP_PATTERN = re.compile(
    r"(\d{4}[-/]\d{1,2}[-/]\d{1,2}(?:\s*(?:周|星期)[一二三四五六日天])?\s+\d{1,2}:\d{2}(?::\d{2})?)"
)


def get_formal_skill_path() -> Path:
    """返回正式 Skill 的主 markdown 路径。"""
    return resolve_skill_markdown_path(SKILLS_DIR / FORMAL_SKILL_VERSION)


def get_formal_skill_dir() -> Path:
    return get_formal_skill_path().parent


def get_formal_executor() -> ExecutorSkill:
    """创建正式 Skill 的执行器实例。"""
    return ExecutorSkill(skill_path=str(get_formal_skill_path()))


def get_builtin_time_script_path() -> Path:
    return get_formal_skill_dir() / "scripts" / TIME_SCRIPT_NAME


def get_builtin_retrieval_script_path() -> Path:
    return get_formal_skill_dir() / "scripts" / RETRIEVAL_SCRIPT_NAME


def get_builtin_retrieval_assets_dir() -> Path:
    return get_formal_skill_dir() / "assets" / RETRIEVAL_ASSETS_DIRNAME


def _run_skill_script(script_path: Path, args: List[str]) -> Dict[str, Any]:
    env = dict(os.environ)
    env.setdefault("PYTHONIOENCODING", "utf-8")
    env.setdefault("PYTHONUTF8", "1")
    completed = subprocess.run(
        [sys.executable, str(script_path), *args],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=env,
        check=False,
    )
    if completed.returncode != 0:
        stderr = (completed.stderr or "").strip()
        stdout = (completed.stdout or "").strip()
        detail = stderr or stdout or "<empty>"
        try:
            payload = json.loads(stdout) if stdout else {}
            if isinstance(payload, dict):
                detail = str(payload.get("message") or payload.get("error") or detail)
        except Exception:
            pass
        raise RuntimeError(
            f"Skill script failed: {script_path.name}; detail={detail}"
        )

    stdout = (completed.stdout or "").strip()
    if not stdout:
        return {}

    lines = [line.strip() for line in stdout.splitlines() if line.strip()]
    for line in reversed(lines):
        if line.startswith("{") and line.endswith("}"):
            return json.loads(line)
    return json.loads(stdout)


def _slugify_reason(text: str) -> str:
    lowered = re.sub(r"[^a-z0-9]+", "_", (text or "").strip().lower()).strip("_")
    return lowered or "runtime_error"


def _short_error_detail(text: str, limit: int = 120) -> str:
    cleaned = re.sub(r"\s+", " ", (text or "").strip())
    if len(cleaned) <= limit:
        return cleaned
    return cleaned[: limit - 3] + "..."


def _classify_retrieval_exception(exc: Exception) -> tuple[str, str]:
    text = str(exc or "").strip()
    lowered = text.lower()

    if "missing embedding api key" in lowered or "api key" in lowered:
        return "embedding_api_key_missing", "missing embedding api key"
    if "connecttimeout" in lowered or "readtimeout" in lowered or "timed out" in lowered or "timeout" in lowered:
        return "embedding_timeout", _short_error_detail(text)
    if "connecterror" in lowered or "connection" in lowered or "network" in lowered:
        return "embedding_connect_error", _short_error_detail(text)
    if "401" in lowered or "unauthorized" in lowered or "authentication" in lowered:
        return "embedding_auth_error", _short_error_detail(text)
    if "429" in lowered or "rate limit" in lowered:
        return "embedding_rate_limit", _short_error_detail(text)
    if "gbk" in lowered or "codec can't encode" in lowered or "codec can't decode" in lowered or "unicodeencodeerror" in lowered:
        return "script_encoding_error", _short_error_detail(text)
    if "json" in lowered and ("parse" in lowered or "decode" in lowered):
        return "script_json_error", _short_error_detail(text)
    if "skill script failed" in lowered:
        detail_match = re.search(r"detail=(.+)$", text)
        detail = detail_match.group(1).strip() if detail_match else text
        return "script_runtime_error", _short_error_detail(detail)

    return _slugify_reason(type(exc).__name__), _short_error_detail(text)


def _conversation_to_query_text(conversation: List[Dict[str, Any]]) -> str:
    user_parts: List[str] = []
    fallback_parts: List[str] = []
    for turn in conversation:
        content = str(turn.get("content", "") or "").strip()
        if not content:
            continue
        fallback_parts.append(content)
        if str(turn.get("role", "user")) != "assistant":
            user_parts.append(content)
    return "\n".join(user_parts or fallback_parts)


def _extract_timestamp_candidate_from_text(text: str) -> str:
    match = TIMESTAMP_PATTERN.search(text)
    return match.group(1).strip() if match else ""


def _extract_timestamp_candidate_from_conversation(conversation: List[Dict[str, Any]]) -> str:
    for turn in conversation:
        for field_name in TIMESTAMP_FIELD_CANDIDATES:
            field_value = turn.get(field_name)
            if isinstance(field_value, str) and field_value.strip():
                return field_value.strip()
        content = str(turn.get("content", "") or "")
        candidate = _extract_timestamp_candidate_from_text(content)
        if candidate:
            return candidate
    return ""


def _extract_raw_time_hint(context: Optional[Dict[str, Any]]) -> str:
    if not isinstance(context, dict):
        return ""
    raw_time_hint = context.get("raw_time_hint")
    return str(raw_time_hint).strip() if raw_time_hint else ""


def _builtin_retrieval_resources_available() -> bool:
    script_path = get_builtin_retrieval_script_path()
    assets_dir = get_builtin_retrieval_assets_dir()
    manifest_path = assets_dir / "manifest.json"
    corpus_path = assets_dir / "corpus.jsonl"
    return script_path.exists() and manifest_path.exists() and corpus_path.exists()


def _run_inprocess_builtin_retrieval_fallback(
    query: str,
    *,
    raw_time_hint: str = "",
    time_features: Optional[Dict[str, Any]] = None,
    top_k: int = DEFAULT_RETRIEVE_TOP_K,
) -> List[Dict[str, Any]]:
    script_path = get_builtin_retrieval_script_path()
    script_dir = script_path.parent
    corpus_path = get_builtin_retrieval_assets_dir() / "corpus.jsonl"
    if not script_path.exists() or not corpus_path.exists():
        return []

    script_dir_str = str(script_dir)
    if script_dir_str not in sys.path:
        sys.path.insert(0, script_dir_str)

    retrieve_cases = importlib.import_module("retrieve_cases")
    query_payload = [{"role": "user", "content": query}]
    fallback_query = retrieve_cases.build_query_retrieval_text(
        query_payload,
        raw_time_hint=raw_time_hint,
        time_features=time_features or {},
    )
    return retrieve_cases._fallback_retrieve(
        corpus_path,
        fallback_query or query,
        max(1, top_k),
    )


def _extract_retrieved_cases_from_context(context: Any) -> Optional[List[Dict[str, Any]]]:
    if context is None:
        return None
    if isinstance(context, dict):
        retrieved_cases = context.get("retrieved_cases")
    else:
        retrieved_cases = getattr(context, "retrieved_cases", None)
    return retrieved_cases if isinstance(retrieved_cases, list) else None


def _infer_rag_mode_from_context(context: Any) -> str:
    retrieved_cases = _extract_retrieved_cases_from_context(context)
    return RAG_MODE_EXTERNAL if retrieved_cases else RAG_MODE_NONE


def _emit_runtime_log(mode_used: str, rag_mode_used: str, *, reason: str = "") -> None:
    suffix = f" reason={reason}" if reason else ""
    print(f"[FORMAL_RUNTIME] mode={mode_used} rag_mode={rag_mode_used}{suffix}", file=sys.stderr)


def _execute_formal_payload(
    payload: AnalysisPayload,
    *,
    mode_used: str,
    rag_mode_used: Optional[str] = None,
    reason: str = "",
) -> SkillOutput:
    _emit_runtime_log(
        mode_used=mode_used,
        rag_mode_used=rag_mode_used or _infer_rag_mode_from_context(payload.context),
        reason=reason,
    )
    return get_formal_executor().run_payload(payload)


def resolve_time_features_result_for_conversation(
    conversation: List[Dict[str, Any]],
    existing_time_features: Optional[Dict[str, Any]] = None,
    raw_time_hint: str = "",
) -> Dict[str, Any]:
    if existing_time_features:
        return {
            "time_features": dict(existing_time_features),
            "time_mode_used": "provided_time_features",
            "reason": "input_time_features",
        }

    timestamp = _extract_timestamp_candidate_from_conversation(conversation)
    timestamp_source = "conversation"
    if not timestamp and raw_time_hint:
        timestamp = raw_time_hint.strip()
        timestamp_source = "raw_time_hint"
    if not timestamp:
        return {
            "time_features": {},
            "time_mode_used": "no_time_features",
            "reason": "no_timestamp_candidate",
        }

    script_path = get_builtin_time_script_path()
    if not script_path.exists():
        return {
            "time_features": {},
            "time_mode_used": "no_time_features",
            "reason": "time_script_missing",
        }

    try:
        payload = _run_skill_script(script_path, ["--timestamp", timestamp])
    except Exception as exc:
        return {
            "time_features": {},
            "time_mode_used": "no_time_features",
            "reason": f"time_script_error:{type(exc).__name__}",
        }

    return {
        "time_features": payload if isinstance(payload, dict) else {},
        "time_mode_used": "builtin_time_script",
        "reason": f"time_script_success:{timestamp_source}",
    }


def resolve_time_features_for_conversation(
    conversation: List[Dict[str, Any]],
    existing_time_features: Optional[Dict[str, Any]] = None,
    raw_time_hint: str = "",
) -> Dict[str, Any]:
    return resolve_time_features_result_for_conversation(
        conversation,
        existing_time_features=existing_time_features,
        raw_time_hint=raw_time_hint,
    )["time_features"]


def resolve_retrieval_result_for_conversation(
    conversation: List[Dict[str, Any]],
    existing_retrieved_cases: Optional[List[Dict[str, Any]]] = None,
    *,
    raw_time_hint: str = "",
    time_features: Optional[Dict[str, Any]] = None,
    top_k: int = DEFAULT_RETRIEVE_TOP_K,
) -> Dict[str, Any]:
    if existing_retrieved_cases is not None:
        provided_cases = list(existing_retrieved_cases)
        return {
            "retrieved_cases": provided_cases,
            "rag_mode_used": RAG_MODE_EXTERNAL if provided_cases else RAG_MODE_NONE,
            "reason": "external_retrieved_cases" if provided_cases else "external_retrieved_cases_empty",
        }

    query = _conversation_to_query_text(conversation)
    if not query:
        return {
            "retrieved_cases": [],
            "rag_mode_used": RAG_MODE_NONE,
            "reason": "empty_query_text",
        }

    if not _builtin_retrieval_resources_available():
        return {
            "retrieved_cases": [],
            "rag_mode_used": RAG_MODE_NONE,
            "reason": "builtin_retrieval_unavailable",
        }

    try:
        time_features_json = json.dumps(time_features or {}, ensure_ascii=False)
        payload = _run_skill_script(
            get_builtin_retrieval_script_path(),
            [
                "--query",
                query,
                "--top-k",
                str(max(1, top_k)),
                "--raw-time-hint",
                raw_time_hint,
                "--time-features-json",
                time_features_json,
            ],
        )
    except Exception as exc:
        reason_code, reason_detail = _classify_retrieval_exception(exc)
        try:
            fallback_cases = _run_inprocess_builtin_retrieval_fallback(
                query,
                raw_time_hint=raw_time_hint,
                time_features=time_features,
                top_k=top_k,
            )
        except Exception:
            fallback_cases = []
        if fallback_cases:
            return {
                "retrieved_cases": fallback_cases,
                "rag_mode_used": RAG_MODE_INTERNAL,
                "reason": f"builtin_retrieval_fallback:{reason_code}",
                "reason_detail": reason_detail,
            }
        return {
            "retrieved_cases": [],
            "rag_mode_used": RAG_MODE_NONE,
            "reason": f"builtin_retrieval_error:{reason_code}",
            "reason_detail": reason_detail,
        }

    if not isinstance(payload, dict):
        return {
            "retrieved_cases": [],
            "rag_mode_used": RAG_MODE_NONE,
            "reason": "builtin_retrieval_invalid_payload",
        }

    if payload.get("status") == "unavailable":
        return {
            "retrieved_cases": [],
            "rag_mode_used": RAG_MODE_NONE,
            "reason": "builtin_retrieval_reported_unavailable",
            "reason_detail": _short_error_detail(str(payload.get("message", ""))),
        }

    retrieved_cases = payload.get("retrieved_cases", [])
    if not isinstance(retrieved_cases, list):
        retrieved_cases = []

    return {
        "retrieved_cases": retrieved_cases,
        "rag_mode_used": RAG_MODE_INTERNAL if retrieved_cases else RAG_MODE_NONE,
        "reason": f"builtin_retrieval_{payload.get('mode', 'ok')}" if retrieved_cases else "builtin_retrieval_empty",
        "reason_detail": _short_error_detail(str(payload.get("message", ""))) if payload.get("message") else "",
    }


def resolve_retrieved_cases_for_conversation(
    conversation: List[Dict[str, Any]],
    existing_retrieved_cases: Optional[List[Dict[str, Any]]] = None,
    *,
    raw_time_hint: str = "",
    time_features: Optional[Dict[str, Any]] = None,
    top_k: int = DEFAULT_RETRIEVE_TOP_K,
) -> List[Dict[str, Any]]:
    return resolve_retrieval_result_for_conversation(
        conversation,
        existing_retrieved_cases=existing_retrieved_cases,
        raw_time_hint=raw_time_hint,
        time_features=time_features,
        top_k=top_k,
    )["retrieved_cases"]


def enrich_single_session_context(
    conversation: List[Dict[str, Any]],
    *,
    context: Optional[Dict[str, Any]] = None,
    retrieve_top_k: int = DEFAULT_RETRIEVE_TOP_K,
) -> Dict[str, Any]:
    merged_context: Dict[str, Any] = dict(context or {})
    time_result = resolve_time_features_result_for_conversation(
        conversation,
        existing_time_features=merged_context.get("time_features"),
        raw_time_hint=_extract_raw_time_hint(merged_context),
    )
    retrieval_result = resolve_retrieval_result_for_conversation(
        conversation,
        existing_retrieved_cases=merged_context.get("retrieved_cases"),
        raw_time_hint=_extract_raw_time_hint(merged_context),
        time_features=time_result["time_features"],
        top_k=retrieve_top_k,
    )
    merged_context["time_features"] = time_result["time_features"]
    merged_context["retrieved_cases"] = retrieval_result["retrieved_cases"]
    merged_context["_formal_runtime"] = {
        "mode_used": "single_session",
        "rag_mode_used": retrieval_result["rag_mode_used"],
        "rag_reason": retrieval_result["reason"],
        "rag_reason_detail": retrieval_result.get("reason_detail", ""),
        "time_mode_used": time_result["time_mode_used"],
        "time_reason": time_result["reason"],
    }
    return merged_context


def _flatten_sessions_to_conversation(sessions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    flattened: List[Dict[str, Any]] = []
    for session in sessions:
        conversation = session.get("conversation", [])
        session_time = str(session.get("session_time", "") or "").strip()
        if isinstance(conversation, list):
            for turn in conversation:
                if not isinstance(turn, dict):
                    continue
                normalized_turn = dict(turn)
                # 多会话场景下优先继承 session_time，避免时间特征完全丢失。
                if session_time and not any(str(normalized_turn.get(field, "") or "").strip() for field in TIMESTAMP_FIELD_CANDIDATES):
                    normalized_turn["timestamp"] = session_time
                flattened.append(normalized_turn)
    return flattened


def enrich_multi_session_context(
    sessions: List[Dict[str, Any]],
    *,
    context: Optional[Dict[str, Any]] = None,
    retrieve_top_k: int = DEFAULT_RETRIEVE_TOP_K,
) -> Dict[str, Any]:
    merged_context: Dict[str, Any] = dict(context or {})
    flattened_conversation = _flatten_sessions_to_conversation(sessions)
    time_result = resolve_time_features_result_for_conversation(
        flattened_conversation,
        existing_time_features=merged_context.get("time_features"),
        raw_time_hint=_extract_raw_time_hint(merged_context),
    )
    retrieval_result = resolve_retrieval_result_for_conversation(
        flattened_conversation,
        existing_retrieved_cases=merged_context.get("retrieved_cases"),
        raw_time_hint=_extract_raw_time_hint(merged_context),
        time_features=time_result["time_features"],
        top_k=retrieve_top_k,
    )
    merged_context["time_features"] = time_result["time_features"]
    merged_context["retrieved_cases"] = retrieval_result["retrieved_cases"]
    merged_context["_formal_runtime"] = {
        "mode_used": "multi_session",
        "rag_mode_used": retrieval_result["rag_mode_used"],
        "rag_reason": retrieval_result["reason"],
        "rag_reason_detail": retrieval_result.get("reason_detail", ""),
        "time_mode_used": time_result["time_mode_used"],
        "time_reason": time_result["reason"],
    }
    return merged_context


def build_formal_single_session_payload(
    conversation: List[Dict[str, str]],
    *,
    user_id: str = "",
    request_id: str = "",
    source: str = "runtime_adapter",
    context: Optional[Dict[str, Any]] = None,
) -> AnalysisPayload:
    return build_single_session_payload(
        conversation,
        user_id=user_id,
        request_id=request_id,
        source=source,
        context=context,
    )


def build_formal_multi_session_payload(
    sessions: List[Dict[str, Any]],
    *,
    user_id: str = "",
    request_id: str = "",
    source: str = "runtime_adapter",
    context: Optional[Dict[str, Any]] = None,
) -> AnalysisPayload:
    return build_multi_session_payload(
        sessions,
        user_id=user_id,
        request_id=request_id,
        source=source,
        context=context,
    )


def build_formal_enriched_payload(
    conversation: List[Dict[str, str]],
    *,
    user_id: str = "",
    request_id: str = "",
    source: str = "runtime_adapter",
    context: Optional[Dict[str, Any]] = None,
) -> AnalysisPayload:
    return build_enriched_payload(
        conversation,
        user_id=user_id,
        request_id=request_id,
        source=source,
        context=context,
    )


def analyze_single_session_formal(
    conversation: List[Dict[str, str]],
    *,
    user_id: str = "",
    request_id: str = "",
    source: str = "runtime_adapter",
    context: Optional[Dict[str, Any]] = None,
) -> SkillOutput:
    payload = build_formal_single_session_payload(
        conversation,
        user_id=user_id,
        request_id=request_id,
        source=source,
        context=context,
    )
    runtime_meta = dict((context or {}).get("_formal_runtime", {})) if isinstance(context, dict) else {}
    return _execute_formal_payload(
        payload,
        mode_used="single_session",
        rag_mode_used=runtime_meta.get("rag_mode_used"),
        reason=runtime_meta.get("rag_reason", ""),
    )


def analyze_single_session_formal_auto(
    conversation: List[Dict[str, Any]],
    *,
    user_id: str = "",
    request_id: str = "",
    source: str = "runtime_adapter",
    context: Optional[Dict[str, Any]] = None,
    retrieve_top_k: int = DEFAULT_RETRIEVE_TOP_K,
) -> SkillOutput:
    enriched_context = enrich_single_session_context(
        conversation,
        context=context,
        retrieve_top_k=retrieve_top_k,
    )
    return analyze_single_session_formal(
        conversation,
        user_id=user_id,
        request_id=request_id,
        source=source,
        context=enriched_context,
    )


def analyze_multi_session_formal(
    sessions: List[Dict[str, Any]],
    *,
    user_id: str = "",
    request_id: str = "",
    source: str = "runtime_adapter",
    context: Optional[Dict[str, Any]] = None,
) -> SkillOutput:
    payload = build_formal_multi_session_payload(
        sessions,
        user_id=user_id,
        request_id=request_id,
        source=source,
        context=context,
    )
    runtime_meta = dict((context or {}).get("_formal_runtime", {})) if isinstance(context, dict) else {}
    return _execute_formal_payload(
        payload,
        mode_used="multi_session",
        rag_mode_used=runtime_meta.get("rag_mode_used"),
        reason=runtime_meta.get("rag_reason", ""),
    )


def analyze_multi_session_formal_auto(
    sessions: List[Dict[str, Any]],
    *,
    user_id: str = "",
    request_id: str = "",
    source: str = "runtime_adapter",
    context: Optional[Dict[str, Any]] = None,
    retrieve_top_k: int = DEFAULT_RETRIEVE_TOP_K,
) -> SkillOutput:
    enriched_context = enrich_multi_session_context(
        sessions,
        context=context,
        retrieve_top_k=retrieve_top_k,
    )
    return analyze_multi_session_formal(
        sessions,
        user_id=user_id,
        request_id=request_id,
        source=source,
        context=enriched_context,
    )


def analyze_enriched_formal(
    conversation: List[Dict[str, str]],
    *,
    user_id: str = "",
    request_id: str = "",
    source: str = "runtime_adapter",
    context: Optional[Dict[str, Any]] = None,
) -> SkillOutput:
    payload = build_formal_enriched_payload(
        conversation,
        user_id=user_id,
        request_id=request_id,
        source=source,
        context=context,
    )
    runtime_meta = dict((context or {}).get("_formal_runtime", {})) if isinstance(context, dict) else {}
    return _execute_formal_payload(
        payload,
        mode_used="enriched",
        rag_mode_used=runtime_meta.get("rag_mode_used"),
        reason=runtime_meta.get("rag_reason", ""),
    )


def analyze_formal_payload(payload: Dict[str, Any] | AnalysisPayload) -> SkillOutput:
    """正式 payload 的统一分析入口。"""
    normalized_payload = payload if isinstance(payload, AnalysisPayload) else AnalysisPayload(**payload)
    return _execute_formal_payload(normalized_payload, mode_used=normalized_payload.mode)

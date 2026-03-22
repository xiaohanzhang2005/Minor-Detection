# 模块说明：
# - 构造标准 analysis payload 的辅助函数。
# - 给旧 executor API 和 formal runtime 适配层共用。

"""
正式 Skill 的 payload 构造工具。
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from src.models import AnalysisPayload, SessionInput


def build_single_session_payload(
    conversation: List[Dict[str, str]],
    *,
    user_id: str = "",
    request_id: str = "",
    source: str = "",
    context: Optional[Dict[str, Any]] = None,
) -> AnalysisPayload:
    payload = AnalysisPayload(mode="single_session", conversation=conversation)
    payload.meta.user_id = user_id
    payload.meta.request_id = request_id
    payload.meta.source = source
    if context:
        _merge_context(payload, context)
    return payload


def build_multi_session_payload(
    sessions: List[Dict[str, Any]],
    *,
    user_id: str = "",
    request_id: str = "",
    source: str = "",
    context: Optional[Dict[str, Any]] = None,
) -> AnalysisPayload:
    payload = AnalysisPayload(
        mode="multi_session",
        sessions=[SessionInput(**session) for session in sessions],
    )
    payload.meta.user_id = user_id
    payload.meta.request_id = request_id
    payload.meta.source = source
    if context:
        _merge_context(payload, context)
    return payload


def build_enriched_payload(
    conversation: List[Dict[str, str]],
    *,
    user_id: str = "",
    request_id: str = "",
    source: str = "",
    context: Optional[Dict[str, Any]] = None,
) -> AnalysisPayload:
    payload = AnalysisPayload(mode="enriched", conversation=conversation)
    payload.meta.user_id = user_id
    payload.meta.request_id = request_id
    payload.meta.source = source
    if context:
        _merge_context(payload, context)
    return payload


def _merge_context(payload: AnalysisPayload, context: Dict[str, Any]) -> None:
    if "time_features" in context and isinstance(context["time_features"], dict):
        payload.context.time_features = context["time_features"]
    if "retrieved_cases" in context and isinstance(context["retrieved_cases"], list):
        payload.context.retrieved_cases = context["retrieved_cases"]
    if "prior_profile" in context and isinstance(context["prior_profile"], dict):
        payload.context.prior_profile = context["prior_profile"]
    if "raw_time_hint" in context:
        payload.context.raw_time_hint = str(context["raw_time_hint"] or "")
    if "channel" in context:
        payload.context.channel = str(context["channel"] or "")
    if "locale" in context:
        payload.context.locale = str(context["locale"] or "")

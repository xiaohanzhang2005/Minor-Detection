'''
模块说明：输出后处理层
负责：
统一 confidence band / risk level
保证 trend 结构存在
把分类器输出和规范化 payload 融合
去重 identity_markers 等字段
'''

from __future__ import annotations

from difflib import SequenceMatcher
import unicodedata
from typing import Any, Dict, List

from config import HIGH_CONFIDENCE_THRESHOLD, LOW_CONFIDENCE_THRESHOLD


ALLOWED_NEXT_STEPS = {
    "collect_more_context",
    "review_by_human",
    "safe_to_continue",
    "monitor_future_sessions",
}

CLAUSE_BOUNDARIES = set("，。！？；,.!?;\n")


def _safe_text(value: Any, default: str = "") -> str:
    text = str(value or "").strip()
    return text or default


def _safe_list(value: Any) -> List[Any]:
    return value if isinstance(value, list) else []


def _confidence_band(confidence: float) -> str:
    if confidence < LOW_CONFIDENCE_THRESHOLD:
        return "low"
    if confidence < HIGH_CONFIDENCE_THRESHOLD:
        return "medium"
    return "high"


def _risk_level(is_minor: bool, confidence: float) -> str:
    if is_minor and confidence >= HIGH_CONFIDENCE_THRESHOLD:
        return "High"
    if is_minor or confidence >= LOW_CONFIDENCE_THRESHOLD:
        return "Medium"
    return "Low"


def _unique(items: List[Any]) -> List[str]:
    seen = set()
    ordered: List[str] = []
    for item in items:
        text = _safe_text(item)
        if not text or text in seen:
            continue
        seen.add(text)
        ordered.append(text)
    return ordered


def _normalize_alignment_text(value: Any) -> str:
    normalized = unicodedata.normalize("NFKC", str(value or "")).casefold()
    kept: List[str] = []
    for ch in normalized:
        category = unicodedata.category(ch)
        if category[:1] in {"P", "S", "Z", "C"}:
            continue
        kept.append(ch)
    return "".join(kept)


def _normalized_text_with_index_map(text: str) -> tuple[str, List[int]]:
    normalized_chars: List[str] = []
    index_map: List[int] = []
    for index, ch in enumerate(str(text or "")):
        for normalized_char in unicodedata.normalize("NFKC", ch).casefold():
            category = unicodedata.category(normalized_char)
            if category[:1] in {"P", "S", "Z", "C"}:
                continue
            normalized_chars.append(normalized_char)
            index_map.append(index)
    return "".join(normalized_chars), index_map


def _expand_to_clause_boundaries(text: str, start_index: int, end_index: int) -> str:
    left = max(0, int(start_index))
    right = min(len(text), int(end_index))
    while left > 0 and text[left - 1] not in CLAUSE_BOUNDARIES:
        left -= 1
    while right < len(text) and text[right] not in CLAUSE_BOUNDARIES:
        right += 1
    return text[left:right].strip()


def _best_conversation_span(item: str, conversation_turns: List[str]) -> str:
    raw_item = _safe_text(item)
    if not raw_item:
        return ""
    for turn in conversation_turns:
        if raw_item and raw_item in turn:
            return raw_item

    normalized_item = _normalize_alignment_text(raw_item)
    if not normalized_item:
        return raw_item

    best_score: tuple[float, float, float] | None = None
    best_span = raw_item
    for turn in conversation_turns:
        normalized_turn, index_map = _normalized_text_with_index_map(turn)
        if not normalized_turn or not index_map:
            continue

        match_start = normalized_turn.find(normalized_item)
        if match_start >= 0:
            start_index = index_map[match_start]
            end_index = index_map[match_start + len(normalized_item) - 1] + 1
            span = _expand_to_clause_boundaries(turn, start_index, end_index)
            span_norm = _normalize_alignment_text(span)
            score = (
                1.0,
                -abs(len(span_norm) - len(normalized_item)),
                -len(span),
            )
            if best_score is None or score > best_score:
                best_score = score
                best_span = span
            continue

        match = SequenceMatcher(None, normalized_item, normalized_turn).find_longest_match(
            0,
            len(normalized_item),
            0,
            len(normalized_turn),
        )
        coverage = (match.size / len(normalized_item)) if normalized_item else 0.0
        if match.size < 6 or coverage < 0.72:
            continue

        start_index = index_map[match.b]
        end_index = index_map[match.b + match.size - 1] + 1
        span = _expand_to_clause_boundaries(turn, start_index, end_index)
        span_norm = _normalize_alignment_text(span)
        score = (
            coverage,
            -abs(len(span_norm) - len(normalized_item)),
            -len(span),
        )
        if best_score is None or score > best_score:
            best_score = score
            best_span = span

    return best_span


def _canonicalize_direct_evidence(
    direct_evidence: List[Any],
    normalized_payload: Dict[str, Any],
) -> List[str]:
    conversation_turns = [
        _safe_text(turn.get("content"))
        for turn in normalized_payload.get("conversation", []) or []
        if isinstance(turn, dict) and _safe_text(turn.get("content"))
    ]
    if not conversation_turns:
        return _unique(direct_evidence)
    return _unique([_best_conversation_span(item, conversation_turns) for item in direct_evidence])


def _ensure_trajectory(output: Dict[str, Any], normalized_payload: Dict[str, Any]) -> Dict[str, Any]:
    trend = output.setdefault("trend", {})
    trajectory = trend.get("trajectory")
    if isinstance(trajectory, list) and trajectory:
        return output
    trend["trajectory"] = []
    trend["trend_summary"] = _safe_text(trend.get("trend_summary"))
    return output


def merge_output(output: Dict[str, Any], normalized_payload: Dict[str, Any]) -> Dict[str, Any]:
    decision = output.setdefault("decision", {})
    confidence = float(decision.get("minor_confidence", 0.5) or 0.5)
    is_minor = bool(decision.get("is_minor", confidence >= HIGH_CONFIDENCE_THRESHOLD))
    decision["is_minor"] = is_minor
    decision["minor_confidence"] = round(max(0.0, min(1.0, confidence)), 4)
    decision["confidence_band"] = _confidence_band(decision["minor_confidence"])
    decision["risk_level"] = _safe_text(decision.get("risk_level")) or _risk_level(is_minor, decision["minor_confidence"])

    profile = output.setdefault("user_profile", {})
    prior_profile = normalized_payload.get("context", {}).get("prior_profile", {})
    profile["age_range"] = _safe_text(profile.get("age_range")) or _safe_text(prior_profile.get("age_range"), "未明确")
    profile["education_stage"] = _safe_text(profile.get("education_stage")) or _safe_text(prior_profile.get("education_stage"), "未明确")
    profile["identity_markers"] = _unique(
        _safe_list(profile.get("identity_markers"))
        + _safe_list(prior_profile.get("identity_markers"))
        + normalized_payload.get("identity_hints", [])
    )

    evidence = output.setdefault("evidence", {})
    evidence["direct_evidence"] = _canonicalize_direct_evidence(_safe_list(evidence.get("direct_evidence")), normalized_payload)
    for key in ("historical_evidence", "retrieval_evidence", "time_evidence", "conflicting_signals"):
        evidence[key] = _unique(_safe_list(evidence.get(key)))
    evidence["evidence_summary"] = _safe_text(evidence.get("evidence_summary"))

    icbo = output.setdefault("icbo_features", {})
    for key in ("intention", "cognition", "behavior_style"):
        icbo[key] = _safe_text(icbo.get(key), "未明确")
    icbo["opportunity_time"] = _safe_text(icbo.get("opportunity_time")) or _safe_text(
        normalized_payload.get("context", {}).get("raw_time_hint")
        or normalized_payload.get("context", {}).get("opportunity_time"),
        "未明确",
    )

    output["reasoning_summary"] = _safe_text(output.get("reasoning_summary"), "未提供推理摘要")
    output["uncertainty_notes"] = _unique(_safe_list(output.get("uncertainty_notes")))
    next_step = _safe_text(output.get("recommended_next_step"))
    if next_step not in ALLOWED_NEXT_STEPS:
        if evidence["conflicting_signals"]:
            next_step = "review_by_human"
        elif output["uncertainty_notes"]:
            next_step = "collect_more_context"
        elif is_minor:
            next_step = "monitor_future_sessions"
        else:
            next_step = "safe_to_continue"
    output["recommended_next_step"] = next_step
    return _ensure_trajectory(output, normalized_payload)

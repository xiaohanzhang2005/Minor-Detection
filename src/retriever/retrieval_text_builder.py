from __future__ import annotations

from typing import Any, Dict, List, Optional

from src.time_features_utils import build_time_feature_payload


UNKNOWN_VALUES = {"", "未知", "未明确", "未明确提及", "null", "none", "None"}


def _normalize_text(text: Any) -> str:
    if not isinstance(text, str):
        return ""
    return " ".join(text.split()).strip()


def conversation_to_user_only_text(conversation: List[Dict[str, Any]]) -> str:
    user_parts: List[str] = []
    fallback_parts: List[str] = []
    for turn in conversation:
        content = _normalize_text(turn.get("content", ""))
        if not content:
            continue
        fallback_parts.append(content)
        if turn.get("role", "user") == "user":
            user_parts.append(content)
    return "\n".join(user_parts or fallback_parts)


def extract_raw_time_hint(sample: Dict[str, Any]) -> str:
    icbo_features = sample.get("icbo_features")
    if not isinstance(icbo_features, dict):
        return ""
    raw_time_hint = icbo_features.get("opportunity_time")
    return str(raw_time_hint).strip() if raw_time_hint else ""


def safe_build_time_features(raw_time_hint: str) -> Dict[str, Any]:
    hint = str(raw_time_hint or "").strip()
    if not hint:
        return {}
    try:
        return build_time_feature_payload(hint)
    except Exception:
        return {}


def build_time_tags(
    *,
    raw_time_hint: str = "",
    time_features: Optional[Dict[str, Any]] = None,
) -> List[str]:
    features = dict(time_features or {})
    if not features and raw_time_hint:
        features = safe_build_time_features(raw_time_hint)
    if not features:
        return []

    ordered_fields = [
        "weekday",
        "is_weekend",
        "is_late_night",
        "time_bucket",
        "holiday_label",
        "school_holiday_hint",
    ]
    tags: List[str] = []
    for field_name in ordered_fields:
        value = features.get(field_name)
        if isinstance(value, bool):
            tags.append(f"{field_name}={'true' if value else 'false'}")
            continue
        if value is None:
            continue
        value_text = str(value).strip()
        if not value_text or value_text in UNKNOWN_VALUES:
            continue
        tags.append(f"{field_name}={value_text}")
    return tags


def build_scene_tags(user_persona: Optional[Dict[str, Any]]) -> List[str]:
    persona = user_persona if isinstance(user_persona, dict) else {}
    tags: List[str] = []

    education_stage = _normalize_text(persona.get("education_stage"))
    if education_stage and education_stage not in UNKNOWN_VALUES:
        tags.append(f"education_stage={education_stage}")

    identity_markers = persona.get("identity_markers", [])
    if isinstance(identity_markers, list):
        for marker in identity_markers[:4]:
            marker_text = _normalize_text(marker)
            if marker_text and marker_text not in UNKNOWN_VALUES:
                tags.append(f"identity_marker={marker_text}")

    return tags


def build_query_retrieval_text(
    conversation: List[Dict[str, Any]],
    *,
    raw_time_hint: str = "",
    time_features: Optional[Dict[str, Any]] = None,
) -> str:
    parts: List[str] = []
    conversation_text = conversation_to_user_only_text(conversation)
    if conversation_text:
        parts.append(conversation_text)

    time_tags = build_time_tags(raw_time_hint=raw_time_hint, time_features=time_features)
    if time_tags:
        parts.append("\n".join(time_tags))

    return "\n".join(part for part in parts if part).strip()


def build_case_retrieval_artifacts(sample: Dict[str, Any]) -> Dict[str, Any]:
    raw_time_hint = extract_raw_time_hint(sample)
    time_features = safe_build_time_features(raw_time_hint)
    scene_tags = build_scene_tags(sample.get("user_persona"))

    parts: List[str] = []
    conversation_text = conversation_to_user_only_text(sample.get("conversation", []))
    if conversation_text:
        parts.append(conversation_text)

    time_tags = build_time_tags(raw_time_hint=raw_time_hint, time_features=time_features)
    if time_tags:
        parts.append("\n".join(time_tags))

    if scene_tags:
        parts.append("\n".join(scene_tags))

    return {
        "embedding_text": "\n".join(part for part in parts if part).strip(),
        "raw_time_hint": raw_time_hint,
        "time_features": time_features,
        "scene_tags": scene_tags,
    }

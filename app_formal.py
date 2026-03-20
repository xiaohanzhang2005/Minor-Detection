import json
from typing import Any, Dict, List

import streamlit as st

from src.runtime import (
    analyze_multi_session_formal,
    analyze_single_session_formal,
    enrich_multi_session_context,
    enrich_single_session_context,
)


st.set_page_config(page_title="Minor Detection Formal Skill", layout="wide")


SINGLE_SESSION_EXAMPLE = json.dumps(
    [
        {
            "role": "user",
            "timestamp": "2026-03-18 周三 23:10",
            "content": "我初二，今晚十一点还在写作业，写完还想继续和AI聊天。",
        },
        {
            "role": "assistant",
            "content": "听起来你最近挺累的，要不要先说说学校和作业压力？",
        },
        {
            "role": "user",
            "content": "最近总觉得每天都在考试，老师还一直说这是关键期。",
        },
    ],
    ensure_ascii=False,
    indent=2,
)

MULTI_SESSION_EXAMPLE = json.dumps(
    [
        {
            "session_id": "s1",
            "session_time": "2026-03-10 周二 22:40",
            "conversation": [
                {"role": "user", "content": "我初二，最近总是晚上偷偷和AI聊天。"},
                {"role": "assistant", "content": "最近压力主要来自哪里？"},
            ],
        },
        {
            "session_id": "s2",
            "session_time": "2026-03-18 周三 23:10",
            "conversation": [
                {"role": "user", "content": "今晚十一点还在写卷子，明天六点又要起床。"},
                {"role": "assistant", "content": "听起来学校节奏很紧。"},
            ],
        },
    ],
    ensure_ascii=False,
    indent=2,
)

PRIOR_PROFILE_EXAMPLE = json.dumps(
    {
        "summary": "同一用户历史上多次出现初中学段、关键期考试压力、晚间高频对话等线索。",
    },
    ensure_ascii=False,
    indent=2,
)


def parse_json_block(text: str, label: str) -> Any:
    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{label} JSON 解析失败: {exc}") from exc


def render_legacy_result(result) -> None:
    decision_col, confidence_col, risk_col = st.columns(3)
    with decision_col:
        st.metric("是否疑似未成年人", "是" if result.is_minor else "否")
    with confidence_col:
        st.metric("未成年人概率", f"{result.minor_confidence:.1%}")
    with risk_col:
        risk_value = result.risk_level.value if hasattr(result.risk_level, "value") else str(result.risk_level)
        st.metric("风险等级", risk_value)

    st.subheader("用户画像")
    st.json(
        {
            "age_range": result.user_persona.age_range,
            "education_stage": result.user_persona.education_stage,
            "identity_markers": result.user_persona.identity_markers,
        },
        expanded=True,
    )

    st.subheader("ICBO")
    st.json(
        {
            "intention": result.icbo_features.intention,
            "cognition": result.icbo_features.cognition,
            "behavior_style": result.icbo_features.behavior_style,
            "opportunity_time": result.icbo_features.opportunity_time,
        },
        expanded=True,
    )

    st.subheader("证据与解释")
    st.write(result.reasoning)
    st.json({"key_evidence": result.key_evidence}, expanded=True)


def build_context(prior_profile_text: str) -> Dict[str, Any]:
    if not prior_profile_text.strip():
        return {}
    return {"prior_profile": parse_json_block(prior_profile_text, "prior_profile")}


st.title("Minor Detection Formal Skill Demo")
st.caption("这个页面只走正式 Skill runtime，不走旧 pipeline。")

mode = st.radio("分析模式", ["single_session", "multi_session"], horizontal=True)
user_id = st.text_input("user_id", value="demo_user")
prior_profile_text = st.text_area("可选 prior_profile JSON", value=PRIOR_PROFILE_EXAMPLE, height=120)

if mode == "single_session":
    input_text = st.text_area("conversation JSON", value=SINGLE_SESSION_EXAMPLE, height=320)
    if st.button("运行正式 Skill", type="primary"):
        try:
            conversation = parse_json_block(input_text, "conversation")
            if not isinstance(conversation, list):
                raise ValueError("conversation 必须是 JSON 数组。")

            base_context = build_context(prior_profile_text)
            enriched_context = enrich_single_session_context(conversation, context=base_context)
            result = analyze_single_session_formal(
                conversation,
                user_id=user_id,
                source="app_formal",
                context=enriched_context,
            )

            left, right = st.columns([1.2, 1.0])
            with left:
                render_legacy_result(result)
            with right:
                st.subheader("注入到 Skill 的上下文")
                st.json(enriched_context, expanded=True)
        except Exception as exc:
            st.error(str(exc))
else:
    input_text = st.text_area("sessions JSON", value=MULTI_SESSION_EXAMPLE, height=360)
    if st.button("运行正式 Skill", type="primary"):
        try:
            sessions = parse_json_block(input_text, "sessions")
            if not isinstance(sessions, list):
                raise ValueError("sessions 必须是 JSON 数组。")

            base_context = build_context(prior_profile_text)
            enriched_context = enrich_multi_session_context(sessions, context=base_context)
            result = analyze_multi_session_formal(
                sessions,
                user_id=user_id,
                source="app_formal",
                context=enriched_context,
            )

            left, right = st.columns([1.2, 1.0])
            with left:
                render_legacy_result(result)
            with right:
                st.subheader("注入到 Skill 的上下文")
                st.json(enriched_context, expanded=True)
        except Exception as exc:
            st.error(str(exc))

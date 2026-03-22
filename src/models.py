# 模块说明：
# - 定义 payload、formal 输出、legacy 输出等核心数据模型。
# - judge、runtime、optimizer 和 skill 输出修复都基于这些合同。

"""
Pydantic 数据模型定义
基于 ICBO 理论框架的结构化输出格式
"""

import re
from enum import Enum
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field


class RiskLevel(str, Enum):
    """风险等级枚举"""
    HIGH = "High"
    MEDIUM = "Medium"
    LOW = "Low"


class ICBOFeatures(BaseModel):
    """
    ICBO 理论框架特征
    - I (Intention): 用户意图
    - C (Cognition): 认知特征
    - B (Behavior): 语言/行为风格
    - O (Opportunity): 语义时间
    """
    intention: str = Field(
        ...,
        description="用户意图：对话的核心目的，如寻求情感支持、宣泄压力、咨询问题等"
    )
    cognition: str = Field(
        ...,
        description="认知特征：用户的思维模式，如灾难化思维、非黑即白、过度概括等"
    )
    behavior_style: str = Field(
        ...,
        description="语言风格：表达方式特征，如情绪化、碎片化、使用网络黑话等"
    )
    opportunity_time: str = Field(
        ...,
        description="语义时间：对话发生的时间语境特征，如深夜、考试前、开学季等"
    )


class UserPersona(BaseModel):
    """用户画像推断"""
    age: Optional[int] = Field(
        None,
        ge=6, le=60,
        description="推测年龄"
    )
    age_range: str = Field(
        ...,
        description="年龄区间：如 12-14岁、15-17岁、18-22岁、23岁以上"
    )
    gender: Optional[str] = Field(
        None,
        description="推测性别：男/女/未知"
    )
    education_stage: str = Field(
        ...,
        description="教育阶段：小学/初中/高中/大学/成人专业"
    )
    identity_markers: List[str] = Field(
        default_factory=list,
        description="身份标记：如住校生、留守儿童、技校生等"
    )


class SkillOutput(BaseModel):
    """
    Skill 统一输出格式
    包含 ICBO 特征、用户画像、风险判定
    """
    is_minor: bool = Field(
        ...,
        description="是否判定为未成年人（<18岁）"
    )
    minor_confidence: float = Field(
        ...,
        ge=0.0, le=1.0,
        description="未成年人判定置信度，0.0-1.0"
    )
    risk_level: RiskLevel = Field(
        ...,
        description="风险等级：High/Medium/Low"
    )
    icbo_features: ICBOFeatures = Field(
        ...,
        description="ICBO 理论框架特征分析"
    )
    user_persona: UserPersona = Field(
        ...,
        description="用户画像推断"
    )
    reasoning: str = Field(
        ...,
        description="判定的详细推理过程，包括关键证据和逻辑链"
    )
    key_evidence: List[str] = Field(
        default_factory=list,
        description="支持判定的关键证据列表"
    )


class ConfidenceBand(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class RecommendedNextStep(str, Enum):
    COLLECT_MORE_CONTEXT = "collect_more_context"
    REVIEW_BY_HUMAN = "review_by_human"
    SAFE_TO_CONTINUE = "safe_to_continue"
    MONITOR_FUTURE_SESSIONS = "monitor_future_sessions"


class FormalDecision(BaseModel):
    is_minor: bool = Field(..., description="是否判定为未成年人")
    minor_confidence: float = Field(..., ge=0.0, le=1.0, description="未成年人置信度")
    confidence_band: ConfidenceBand = Field(..., description="置信度分段")
    risk_level: RiskLevel = Field(..., description="风险等级")


class FormalUserProfile(BaseModel):
    age_range: str = Field(..., description="年龄段描述")
    education_stage: str = Field(..., description="教育阶段")
    identity_markers: List[str] = Field(default_factory=list, description="身份标记")


class FormalEvidence(BaseModel):
    direct_evidence: List[str] = Field(default_factory=list)
    historical_evidence: List[str] = Field(default_factory=list)
    retrieval_evidence: List[str] = Field(default_factory=list)
    time_evidence: List[str] = Field(default_factory=list)
    conflicting_signals: List[str] = Field(default_factory=list)


class TrendPoint(BaseModel):
    session_id: Optional[str] = Field(default=None)
    session_time: Optional[str] = Field(default=None)
    minor_confidence: float = Field(..., ge=0.0, le=1.0)


class FormalTrend(BaseModel):
    trajectory: List[TrendPoint] = Field(default_factory=list)
    trend_summary: str = Field(default="", description="趋势总结")


class FormalSkillOutput(BaseModel):
    decision: FormalDecision
    user_profile: FormalUserProfile
    icbo_features: ICBOFeatures
    evidence: FormalEvidence
    reasoning_summary: str
    trend: FormalTrend
    uncertainty_notes: List[str] = Field(default_factory=list)
    recommended_next_step: RecommendedNextStep = Field(default=RecommendedNextStep.COLLECT_MORE_CONTEXT)


class SessionInput(BaseModel):
    session_id: Optional[str] = None
    session_time: Optional[str] = None
    conversation: List[Dict[str, str]] = Field(default_factory=list)


class AnalysisContext(BaseModel):
    time_features: Dict[str, Any] = Field(default_factory=dict)
    retrieved_cases: List[Dict[str, Any]] = Field(default_factory=list)
    prior_profile: Dict[str, Any] = Field(default_factory=dict)
    raw_time_hint: str = Field(default="")
    channel: str = Field(default="")
    locale: str = Field(default="")


class AnalysisMeta(BaseModel):
    user_id: str = Field(default="")
    request_id: str = Field(default="")
    source: str = Field(default="")


class AnalysisPayload(BaseModel):
    mode: str
    conversation: List[Dict[str, str]] = Field(default_factory=list)
    sessions: List[SessionInput] = Field(default_factory=list)
    context: AnalysisContext = Field(default_factory=AnalysisContext)
    meta: AnalysisMeta = Field(default_factory=AnalysisMeta)


_ENGLISH_STAGE_MAP = {
    "middle school": "初中",
    "junior middle school": "初中",
    "high school": "高中",
    "senior high school": "高中",
    "university": "大学",
    "undergraduate": "大学/本科",
    "college": "大学/本科",
    "college student": "大学/本科",
}

_CHINESE_STAGE_MAP = {
    "小学": "小学",
    "小学生": "小学",
    "初中": "初中",
    "初中生": "初中",
    "中学": "初中/高中未明确",
    "中学生": "初中/高中未明确",
    "高中": "高中",
    "高中生": "高中",
    "大学": "大学/本科",
    "本科": "大学/本科",
    "大学生": "大学/本科",
    "本科生": "大学/本科",
    "成人": "成人专业",
    "成人专业": "成人专业",
}

_ENGLISH_AGE_RANGE_MAP = {
    "adolescent": "13-17岁",
    "teenager": "13-17岁",
    "teen": "13-17岁",
    "minor": "未成年人",
    "adult": "成年人",
    "18-24": "18-24岁",
}

_CHINESE_AGE_RANGE_MAP = {
    "青少年": "13-17岁",
    "未成年人": "13-17岁",
    "成年人": "18岁以上",
    "小学阶段": "7-12岁",
    "初中阶段": "13-15岁",
    "高中阶段": "16-18岁",
}

_ENGLISH_MARKER_MAP = {
    "student": "学生",
    "learner": "学习者",
    "middle school student": "初中生",
    "high school student": "高中生",
    "undergraduate": "本科生",
    "college student": "大学生",
    "intern": "实习生",
}

_ZH_WEEKDAY_MAP = {
    "monday": "周一",
    "tuesday": "周二",
    "wednesday": "周三",
    "thursday": "周四",
    "friday": "周五",
    "saturday": "周六",
    "sunday": "周日",
}

_TIME_BUCKET_MAP = {
    "early_morning": "清晨",
    "morning": "上午",
    "noon": "中午",
    "afternoon": "下午",
    "evening": "晚间",
    "late_night": "深夜",
}

_HOLIDAY_MAP = {
    "none": "非假期",
    "winter_vacation": "寒假",
    "summer_vacation": "暑假",
}

_ALLOWED_NEXT_STEPS = {
    "collect_more_context",
    "review_by_human",
    "safe_to_continue",
    "monitor_future_sessions",
}

_STAGE_AGE_RANGE_MAP = {
    "小学": "7-12岁",
    "初中": "13-15岁",
    "高中": "16-18岁",
    "大学/本科": "18-24岁",
}

_GENERIC_STAGE_VALUES = {"", "未明确", "学生", "学习者", "初中/高中未明确"}
_GENERIC_AGE_VALUES = {"", "未明确", "13-17岁", "18岁以上"}

_DIRECT_STAGE_PATTERNS = {
    "小学": [r"小学", r"小学生"],
    "初中": [r"初中", r"初一", r"初二", r"初三", r"中考"],
    "高中": [r"高中", r"高一", r"高二", r"高三", r"高考", r"必修"],
    "大学/本科": [r"大学", r"大一", r"大二", r"大三", r"大四", r"本科", r"研究生", r"考研"],
}

_TOPIC_STAGE_PATTERNS = {
    "初中": [
        r"一元二次方程",
        r"二次函数",
        r"相似三角形",
        r"全等三角形",
        r"勾股定理",
        r"反比例函数",
    ],
    "高中": [
        r"复合函数",
        r"导数",
        r"数列",
        r"圆锥曲线",
        r"立体几何",
        r"元素周期律",
        r"遗传定律",
    ],
}


def _contains_chinese(text: str) -> bool:
    return bool(re.search(r"[\u4e00-\u9fff]", text or ""))


def _normalize_text_value(value: str, mapping: Dict[str, str]) -> str:
    normalized = (value or "").strip()
    if not normalized:
        return "未明确"
    lowered = normalized.lower()
    if lowered in mapping:
        return mapping[lowered]
    return normalized


def _conversation_text(conversation: Optional[List[Dict[str, Any]]]) -> str:
    parts: List[str] = []
    for turn in conversation or []:
        if not isinstance(turn, dict):
            continue
        content = str(turn.get("content", "") or "").strip()
        if content:
            parts.append(content)
    return "\n".join(parts)


def _normalize_stage_value(value: str) -> str:
    normalized = _normalize_text_value(value, _ENGLISH_STAGE_MAP)
    normalized = _CHINESE_STAGE_MAP.get(normalized, normalized)
    return normalized if normalized else "未明确"


def _normalize_age_range_value(value: str) -> str:
    normalized = _normalize_text_value(value, _ENGLISH_AGE_RANGE_MAP)
    normalized = _CHINESE_AGE_RANGE_MAP.get(normalized, normalized)
    return normalized if normalized else "未明确"


def _extract_age_range_from_text(text: str) -> str:
    matches = re.findall(r"(?<!\d)(\d{1,2})岁(?!\d)", text or "")
    for match in matches:
        try:
            age = int(match)
        except Exception:
            continue
        if 7 <= age <= 12:
            return "7-12岁"
        if 13 <= age <= 15:
            return "13-15岁"
        if 16 <= age <= 18:
            return "16-18岁"
        if 19 <= age <= 24:
            return "18-24岁"
    return ""


def _infer_stage_hint_from_text(text: str) -> tuple[str, str]:
    normalized_text = text or ""
    for stage, patterns in _DIRECT_STAGE_PATTERNS.items():
        if any(re.search(pattern, normalized_text) for pattern in patterns):
            return stage, "direct"
    for stage, patterns in _TOPIC_STAGE_PATTERNS.items():
        if any(re.search(pattern, normalized_text) for pattern in patterns):
            return stage, "topic"
    return "", ""


def _text_supports_stage(text: str, stage: str) -> bool:
    patterns = _DIRECT_STAGE_PATTERNS.get(stage, [])
    return any(re.search(pattern, text or "") for pattern in patterns)


def _normalize_marker_list(markers: List[str]) -> List[str]:
    normalized: List[str] = []
    seen: set[str] = set()
    for marker in markers or []:
        if not marker:
            continue
        marker_text = _normalize_text_value(str(marker), _ENGLISH_MARKER_MAP)
        if marker_text not in seen:
            seen.add(marker_text)
            normalized.append(marker_text)
    return normalized


def _normalize_next_step(value: str, output: FormalSkillOutput) -> RecommendedNextStep:
    normalized = (value or "").strip()
    if normalized in _ALLOWED_NEXT_STEPS:
        return RecommendedNextStep(normalized)

    lowered = normalized.lower()
    if any(token in lowered for token in ["review_by_human", "human review", "人工", "审核", "复核"]):
        return RecommendedNextStep.REVIEW_BY_HUMAN
    if any(token in lowered for token in ["collect_more_context", "more context", "更多信息", "补充信息", "收集", "澄清", "补充上下文"]):
        return RecommendedNextStep.COLLECT_MORE_CONTEXT
    if any(token in lowered for token in ["monitor_future_sessions", "monitor", "future session", "后续", "持续观察", "继续观察", "监控"]):
        return RecommendedNextStep.MONITOR_FUTURE_SESSIONS
    if any(token in lowered for token in ["safe_to_continue", "继续", "支持", "资源", "引导", "继续提供", "正常回复"]):
        return RecommendedNextStep.SAFE_TO_CONTINUE

    if output.evidence.conflicting_signals:
        return RecommendedNextStep.REVIEW_BY_HUMAN
    if output.uncertainty_notes:
        return RecommendedNextStep.COLLECT_MORE_CONTEXT
    return "safe_to_continue"


def _normalize_uncertainty_notes(notes: List[str]) -> List[str]:
    normalized: List[str] = []
    for note in notes or []:
        text = str(note or "").strip()
        if not text:
            continue
        lowered = text.lower()
        if any(
            token in lowered
            for token in [
                "无明显不确定性",
                "没有明显不确定性",
                "无不确定性",
                "判断依据充分",
                "no obvious uncertainty",
                "no significant uncertainty",
            ]
        ):
            continue
        normalized.append(text)
    return normalized


def confidence_to_band(confidence: float) -> ConfidenceBand:
    if confidence < 0.34:
        return ConfidenceBand.LOW
    if confidence < 0.67:
        return ConfidenceBand.MEDIUM
    return ConfidenceBand.HIGH


def _format_opportunity_time(raw_time_hint: str = "", time_features: Optional[Dict[str, Any]] = None) -> str:
    parts: List[str] = []
    raw_time_hint = (raw_time_hint or "").strip()
    time_features = time_features or {}

    if raw_time_hint:
        parts.append(f"原始时间: {raw_time_hint}")

    tags: List[str] = []
    weekday = str(time_features.get("weekday") or "").strip().lower()
    if weekday:
        tags.append(_ZH_WEEKDAY_MAP.get(weekday, str(time_features.get("weekday"))))

    time_bucket = str(time_features.get("time_bucket") or "").strip().lower()
    if time_bucket:
        tags.append(_TIME_BUCKET_MAP.get(time_bucket, str(time_features.get("time_bucket"))))

    if "is_weekend" in time_features:
        tags.append("周末" if bool(time_features.get("is_weekend")) else "非周末")

    if "is_late_night" in time_features:
        tags.append("深夜" if bool(time_features.get("is_late_night")) else "非深夜")

    holiday_label = str(time_features.get("holiday_label") or "").strip().lower()
    if holiday_label:
        tags.append(_HOLIDAY_MAP.get(holiday_label, holiday_label))

    if "school_holiday_hint" in time_features:
        tags.append("学校假期倾向" if bool(time_features.get("school_holiday_hint")) else "非学校假期")

    if tags:
        parts.append("归一化标签: " + "，".join(tags))

    return "；".join(parts) if parts else "未明确"


def normalize_formal_skill_output(
    output: FormalSkillOutput,
    *,
    raw_time_hint: str = "",
    time_features: Optional[Dict[str, Any]] = None,
    conversation: Optional[List[Dict[str, Any]]] = None,
) -> FormalSkillOutput:
    output.decision.confidence_band = confidence_to_band(output.decision.minor_confidence)
    output.user_profile.age_range = _normalize_age_range_value(output.user_profile.age_range)
    output.user_profile.education_stage = _normalize_stage_value(output.user_profile.education_stage)
    output.user_profile.identity_markers = _normalize_marker_list(output.user_profile.identity_markers)

    conversation_text = _conversation_text(conversation)
    stage_hint, stage_hint_source = _infer_stage_hint_from_text(conversation_text)
    if stage_hint:
        current_stage = output.user_profile.education_stage
        if (
            current_stage in _GENERIC_STAGE_VALUES
            or not _text_supports_stage(conversation_text, current_stage)
            or (stage_hint_source == "direct" and current_stage != stage_hint)
            or (stage_hint_source == "topic" and current_stage != stage_hint and current_stage not in {"大学/本科", "成人专业"})
        ):
            output.user_profile.education_stage = stage_hint

    direct_age_range = _extract_age_range_from_text(conversation_text)
    if direct_age_range:
        output.user_profile.age_range = direct_age_range
    else:
        canonical_age_range = _STAGE_AGE_RANGE_MAP.get(output.user_profile.education_stage, "")
        if canonical_age_range and (
            output.user_profile.age_range in _GENERIC_AGE_VALUES
            or output.user_profile.age_range == "青少年"
            or output.user_profile.age_range == "未成年人"
            or output.user_profile.age_range != canonical_age_range
        ):
            output.user_profile.age_range = canonical_age_range

    if raw_time_hint or time_features:
        output.icbo_features.opportunity_time = _format_opportunity_time(
            raw_time_hint=raw_time_hint,
            time_features=time_features,
        )
    elif not output.icbo_features.opportunity_time.strip():
        output.icbo_features.opportunity_time = "未明确"

    if not output.trend.trajectory and output.trend.trend_summary.strip().lower() in {"n/a", "n/a for single session.", "not applicable"}:
        output.trend.trend_summary = ""

    output.recommended_next_step = _normalize_next_step(output.recommended_next_step, output)
    output.uncertainty_notes = _normalize_uncertainty_notes(output.uncertainty_notes)
    return output


def formal_to_legacy_output(output: FormalSkillOutput) -> SkillOutput:
    """将正式 Skill 输出转换为旧版 SkillOutput，便于现有链路平滑迁移。"""
    reasoning_parts: List[str] = [output.reasoning_summary.strip()]
    if output.uncertainty_notes:
        reasoning_parts.append("不确定性说明: " + "；".join(output.uncertainty_notes))
    if output.trend.trend_summary:
        reasoning_parts.append("趋势: " + output.trend.trend_summary.strip())

    key_evidence: List[str] = []
    for evidence_group in (
        output.evidence.direct_evidence,
        output.evidence.historical_evidence,
        output.evidence.retrieval_evidence,
        output.evidence.time_evidence,
    ):
        for item in evidence_group:
            if item and item not in key_evidence:
                key_evidence.append(item)
            if len(key_evidence) >= 5:
                break
        if len(key_evidence) >= 5:
            break

    return SkillOutput(
        is_minor=output.decision.is_minor,
        minor_confidence=output.decision.minor_confidence,
        risk_level=output.decision.risk_level,
        icbo_features=output.icbo_features,
        user_persona=UserPersona(
            age_range=output.user_profile.age_range or "未明确",
            education_stage=output.user_profile.education_stage or "未明确",
            identity_markers=output.user_profile.identity_markers,
        ),
        reasoning="\n".join(part for part in reasoning_parts if part),
        key_evidence=key_evidence,
    )


# === 兼容旧代码的类型别名 ===
# 保留旧的 IntentType 以便渐进式迁移
class IntentType(str, Enum):
    """用户意图类型（兼容旧版）"""
    KNOWLEDGE_QA = "knowledge_qa"
    SOCIAL_CHAT = "social_chat"

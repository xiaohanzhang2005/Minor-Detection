"""
Pydantic 数据模型定义
基于 ICBO 理论框架的结构化输出格式
"""

from enum import Enum
from typing import Optional, List
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


# === 兼容旧代码的类型别名 ===
# 保留旧的 IntentType 以便渐进式迁移
class IntentType(str, Enum):
    """用户意图类型（兼容旧版）"""
    KNOWLEDGE_QA = "knowledge_qa"
    SOCIAL_CHAT = "social_chat"

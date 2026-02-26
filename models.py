"""
Pydantic 数据模型定义
用于定义 LLM 分析后的结构化输出格式
"""

from enum import Enum
from pydantic import BaseModel, Field


class IntentType(Enum):
    """用户意图类型枚举"""
    KNOWLEDGE_QA = "knowledge_qa"      # 知识问答
    SOCIAL_CHAT = "social_chat"        # 社交聊天


class KnowledgeAnalysis(BaseModel):
    """知识分析结果模型"""
    predicted_education_level: str = Field(
        ...,
        description="预测的教育水平，如：小学、初中、高中、大学、成人专业"
    )
    is_homework: bool = Field(
        default=False,
        description="是否为作业相关的问题"
    )
    reasoning: str = Field(
        ...,
        description="分析推理过程的详细说明"
    )


class SocialAnalysis(BaseModel):
    """社交分析结果模型"""
    emotional_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="情绪波动评分，0.0-1.0，越高表示情绪波动越大"
    )
    logic_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="逻辑性评分，0.0-1.0，越高表示逻辑性越强"
    )
    minor_tendency_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="未成年人倾向评分，0.0-1.0，越高表示越倾向于未成年人特征"
    )
    reasoning: str = Field(
        ...,
        description="分析推理过程的详细说明"
    )






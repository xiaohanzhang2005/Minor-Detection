"""
用户画像管理模块
维护用户的动态画像，包括未成年人概率和历史特征
"""

from typing import Dict, List
from models import IntentType, KnowledgeAnalysis, SocialAnalysis


class UserProfile:
    """
    用户画像类
    基于历史对话动态更新用户的未成年人概率和特征统计
    """

    def __init__(self, user_id: str):
        """
        初始化用户画像

        Args:
            user_id: 用户唯一标识符
        """
        self.user_id: str = user_id
        self.history_summary: List[str] = []  # 存储关键特征描述
        self.minor_probability: float = 0.5   # 未成年人概率，初始值为0.5
        self.knowledge_stats: Dict[str, int] = {  # 不同教育水平的命中次数统计
            "小学": 0,
            "初中": 0,
            "高中": 0,
            "大学": 0,
            "成人专业": 0
        }
        self.consecutive_high_emotion: int = 0  # 连续高情绪波动次数

    def update_score(self, analysis_result, intent_type: IntentType) -> None:
        """
        根据分析结果更新未成年人概率

        Args:
            analysis_result: 分析结果对象 (KnowledgeAnalysis 或 SocialAnalysis)
            intent_type: 用户意图类型
        """
        if intent_type == IntentType.KNOWLEDGE_QA:
            self._update_knowledge_score(analysis_result)
        elif intent_type == IntentType.SOCIAL_CHAT:
            self._update_social_score(analysis_result)

        # 高阶知识豁免权：如果检测到大学或成人专业知识超过2次，锁定概率在0.2以下
        adult_knowledge_count = self.knowledge_stats.get("大学", 0) + self.knowledge_stats.get("成人专业", 0)
        if adult_knowledge_count >= 2:
            self.minor_probability = min(self.minor_probability, 0.2)
            if adult_knowledge_count == 2:  # 只在首次触发时记录
                self.history_summary.append("🎓 触发高阶知识豁免权：检测到多次高水平知识，强制锁定未成年人概率≤0.2")

        # 确保概率值在合理范围内
        self.minor_probability = max(0.0, min(1.0, self.minor_probability))

    def _update_knowledge_score(self, knowledge_analysis: KnowledgeAnalysis) -> None:
        """
        根据知识分析结果更新分数

        Args:
            knowledge_analysis: 知识分析结果
        """
        education_level = knowledge_analysis.predicted_education_level

        # 更新知识统计
        if education_level in self.knowledge_stats:
            self.knowledge_stats[education_level] += 1

        # 根据教育水平调整未成年人概率
        if education_level == "小学":
            # 小学知识，大幅提升未成年人概率
            self.minor_probability += 0.2
            self.history_summary.append(f"知识水平：{education_level}，强烈倾向于未成年人特征")
        elif education_level == "初中":
            # 初中知识，提升未成年人概率
            self.minor_probability += 0.15
            self.history_summary.append(f"知识水平：{education_level}，倾向于未成年人特征")
        elif education_level == "高中":
            # 高中知识，微幅提升未成年人概率（高中生也可能是未成年人）
            self.minor_probability += 0.05
            self.history_summary.append(f"知识水平：{education_level}，微幅倾向于未成年人特征")
        elif education_level in ["大学"]:
            # 大学知识，降低未成年人概率
            self.minor_probability -= 0.08
            self.history_summary.append(f"知识水平：{education_level}，倾向于成年人特征")
        elif education_level == "成人专业":
            # 成人专业知识，大幅降低未成年人概率
            self.minor_probability -= 0.15
            self.history_summary.append(f"知识水平：{education_level}，强烈倾向于成年人特征")

        # 额外考虑作业因素
        if knowledge_analysis.is_homework:
            self.minor_probability += 0.15
            self.history_summary.append("涉及作业相关内容")

        # 如果是作业相关，微幅提升未成年人概率
        if knowledge_analysis.is_homework:
            self.minor_probability += 0.02
            self.history_summary.append("涉及作业相关内容")

    def _update_social_score(self, social_analysis: SocialAnalysis) -> None:
        """
        根据社交分析结果更新分数

        Args:
            social_analysis: 社交分析结果
        """
        # 情绪波动惩罚：连续两次检测到emotional_score > 0.8，大幅提升未成年人概率
        if social_analysis.emotional_score > 0.8:
            self.consecutive_high_emotion += 1
            if self.consecutive_high_emotion >= 2:
                # 连续两次高情绪波动，施加惩罚
                self.minor_probability += 0.15  # 大幅提升未成年人概率
                self.history_summary.append("🚨 情绪波动惩罚：连续检测到极度情绪不稳定，大幅提升未成年人概率")
        else:
            # 情绪恢复正常，重置计数器
            if self.consecutive_high_emotion > 0:
                self.consecutive_high_emotion = 0

        # 根据未成年人倾向评分加权更新概率
        # minor_tendency_score 越高，越倾向于未成年人，概率增加
        # minor_tendency_score 越低，越倾向于成年人，概率减少

        # 动态权重：倾向分越极端，权重越大
        tendency_diff = abs(social_analysis.minor_tendency_score - 0.5)
        weight = 0.3 + tendency_diff * 0.6  # 权重范围: 0.3-0.9

        delta = (social_analysis.minor_tendency_score - 0.5) * weight
        self.minor_probability += delta

        # 记录关键特征
        features = []
        if social_analysis.emotional_score > 0.7:
            features.append("情绪波动较大")
        if social_analysis.logic_score < 0.3:
            features.append("逻辑性较弱")

        if social_analysis.minor_tendency_score > 0.7:
            features.append("表现出未成年人特征")
        elif social_analysis.minor_tendency_score < 0.3:
            features.append("表现出成年人特征")

        if features:
            self.history_summary.append(f"社交特征：{', '.join(features)}")

    def get_profile_summary(self) -> Dict:
        """
        获取用户画像摘要

        Returns:
            包含用户画像信息的字典
        """
        return {
            "user_id": self.user_id,
            "minor_probability": round(self.minor_probability, 3),
            "knowledge_stats": self.knowledge_stats.copy(),
            "recent_features": self.history_summary[-5:] if self.history_summary else []  # 最近5条特征
        }

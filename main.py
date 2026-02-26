"""
未成年人识别与保护系统主程序
模拟完整的用户消息处理流程
"""

import os
from typing import Dict
from models import IntentType
from user_profile import UserProfile
from llm_service import get_llm_service

# 模拟数据库：user_id -> UserProfile
DB: Dict[str, UserProfile] = {}


def get_or_create_profile(user_id: str) -> UserProfile:
    """
    获取或创建用户画像

    Args:
        user_id: 用户ID

    Returns:
        UserProfile: 用户画像实例
    """
    if user_id not in DB:
        DB[user_id] = UserProfile(user_id)
        print(f"📝 创建新用户画像: {user_id}")
    else:
        print(f"📋 加载现有用户画像: {user_id}")

    return DB[user_id]


def process_user_message(user_id: str, text: str) -> None:
    """
    处理用户消息的主函数

    Args:
        user_id: 用户ID
        text: 用户输入文本
    """
    print(f"\n{'='*60}")
    print(f"🔍 处理用户消息: {user_id}")
    print(f"💬 消息内容: {text}")
    print(f"{'='*60}")

    # 步骤1: 获取或创建用户画像
    profile = get_or_create_profile(user_id)

    # 步骤2: 获取LLM服务实例
    llm_service = get_llm_service()

    try:
        # 步骤3: 意图分类
        print("\n🤔 步骤1: 意图分类")
        intent = llm_service.classify_intent(text)
        print(f"🎯 识别意图: {intent.value}")

        # 步骤4: 根据意图进行相应分析
        print("\n🧠 步骤2: 深度分析")

        if intent == IntentType.KNOWLEDGE_QA:
            print("📚 执行知识分析...")
            analysis_result = llm_service.analyze_knowledge(text)
            print("📖 知识分析结果:")
            print(f"   - 教育水平: {analysis_result.predicted_education_level}")
            print(f"   - 是否作业: {analysis_result.is_homework}")
            print(f"   - 推理过程: {analysis_result.reasoning}")

        elif intent == IntentType.SOCIAL_CHAT:
            print("💭 执行社交分析...")
            analysis_result = llm_service.analyze_social(text)
            print("👥 社交分析结果:")
            print(f"   - 情绪波动: {analysis_result.emotional_score:.3f}")
            print(f"   - 逻辑性: {analysis_result.logic_score:.3f}")
            print(f"   - 未成年人倾向: {analysis_result.minor_tendency_score:.3f}")
            print(f"   - 推理过程: {analysis_result.reasoning}")

        # 步骤5: 更新用户画像
        print("\n📊 步骤3: 更新用户画像")
        old_probability = profile.minor_probability
        profile.update_score(analysis_result, intent)
        new_probability = profile.minor_probability

        print(f"📈 旧概率: {old_probability:.3f}")
        print(f"📈 新概率: {new_probability:.3f}")
        print(f"📈 概率变化: {new_probability - old_probability:+.3f}")

        # 显示知识统计（如果是知识问答）
        if intent == IntentType.KNOWLEDGE_QA:
            print(f"📊 知识统计: {profile.knowledge_stats}")

        # 显示最近特征
        if profile.history_summary:
            print(f"🔍 最近特征: {profile.history_summary[-1]}")

        # 步骤6: 决策逻辑
        print("\n⚖️  步骤4: 安全决策")
        if profile.minor_probability > 0.8:
            print("⚠️  检测到未成年人，已拦截/开启保护模式")
            print("🛡️  建议措施: 内容过滤、家长通知、年龄验证")
        else:
            print("✅ 正常放行")
            print("🚀 继续正常对话流程")

        # 显示完整画像摘要
        print("\n📋 用户画像摘要:")
        summary = profile.get_profile_summary()
        print(f"   - 未成年人概率: {summary['minor_probability']}")
        print(f"   - 知识统计: {summary['knowledge_stats']}")
        if summary['recent_features']:
            print("   - 最近特征:")
            for feature in summary['recent_features']:
                print(f"     • {feature}")

    except Exception as e:
        print(f"❌ 处理失败: {e}")
        print("🔄 启用降级策略: 保守处理（视作潜在风险）")
        # 在实际系统中，这里应该有降级处理逻辑


def simulate_conversation(user_id: str, messages: list[str], scenario_name: str) -> None:
    """
    模拟完整对话

    Args:
        user_id: 用户ID
        messages: 消息列表
        scenario_name: 场景名称
    """
    print(f"\n🎬 ===== {scenario_name} =====")

    for i, message in enumerate(messages, 1):
        print(f"\n📝 对话轮次 {i}/{len(messages)}")
        process_user_message(user_id, message)

    # 显示最终画像
    if user_id in DB:
        profile = DB[user_id]
        print(f"\n🏁 {scenario_name} - 最终画像:")
        summary = profile.get_profile_summary()
        print(f"   - 未成年人概率: {summary['minor_probability']}")
        print(f"   - 历史特征数量: {len(profile.history_summary)}")


if __name__ == "__main__":
    # 设置AiHubMix API密钥（已内置在代码中）
    # 如需自定义，请设置环境变量: os.environ["AIHUBMIX_API_KEY"] = "your-key"

    print("🚀 未成年人识别与保护系统启动 (Gemini-3-Flash)")
    print("⚠️  使用 AiHubMix 接口调用 gemini-3-flash-preview 模型")

    # 检查API密钥（已内置）
    api_key = os.getenv("AIHUBMIX_API_KEY")
    if not api_key:
        print("❌ 未找到 API 密钥")
        exit(1)

    try:
        # 场景A: 未成年人特征对话
        scenario_a_messages = [
            "烦死了，明天的数学作业还没写完。",
            "那个老师真的太讨厌了，针对我。",
            "我想充值买个皮肤，借我点钱呗。"
        ]
        simulate_conversation("user_minor", scenario_a_messages, "场景A: 未成年人")

        # 场景B: 成年人特征对话
        scenario_b_messages = [
            "你好，请帮我解释一下什么是二叉树的遍历。",
            "我想了解一下北京的公积金提取政策。",
            "最近工作压力好大，想放松一下。"
        ]
        simulate_conversation("user_adult", scenario_b_messages, "场景B: 成年人")

        print(f"\n📊 数据库状态: {len(DB)} 个用户画像")

    except Exception as e:
        print(f"❌ 系统错误: {e}")
        import traceback
        traceback.print_exc()

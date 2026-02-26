"""
未成年人识别与保护系统 - 对话样本测试脚本
使用真实对话样本测试系统的准确性
"""

import json
import os
from typing import Dict, List, Any
from models import IntentType
from user_profile import UserProfile
from llm_service import get_llm_service


def load_conversation_samples() -> List[Dict[str, Any]]:
    """
    加载对话样本数据

    Returns:
        对话样本列表
    """
    samples_file = "data/conversation_samples.json"
    if not os.path.exists(samples_file):
        print(f"❌ 找不到样本文件: {samples_file}")
        return []

    with open(samples_file, 'r', encoding='utf-8') as f:
        samples = json.load(f)

    print(f"✅ 加载了 {len(samples)} 个对话样本")
    return samples


def extract_conversation_text(conversation: List[Dict[str, str]]) -> str:
    """
    从对话中提取完整文本内容（包含用户和AI消息）

    Args:
        conversation: 对话列表

    Returns:
        完整的对话文本
    """
    messages = []
    for msg in conversation:
        role = msg.get('role', 'unknown')
        content = msg.get('content', '')
        if role == 'user':
            messages.append(f"用户: {content}")
        elif role == 'assistant':
            messages.append(f"AI: {content}")

    return '\n'.join(messages)


def process_sample(sample: Dict[str, Any], llm_service, db: Dict[str, UserProfile]) -> Dict[str, Any]:
    """
    处理单个对话样本

    Args:
        sample: 样本数据
        llm_service: LLM服务实例
        db: 用户数据库

    Returns:
        处理结果
    """
    sample_id = sample['id']
    user_type = sample['user_type']
    category = sample['category']
    description = sample['description']
    conversation = sample['conversation']

    print(f"\n🔍 分析样本 {sample_id}")
    print(f"📝 描述: {description}")
    print(f"👤 实际类型: {user_type}")
    print(f"💬 对话轮数: {len(conversation)}")

    # 创建用户ID（使用样本ID）
    user_id = f"test_{sample_id}"

    # 获取或创建用户画像
    if user_id not in db:
        db[user_id] = UserProfile(user_id)

    user_profile = db[user_id]

    # 提取完整对话内容
    combined_content = extract_conversation_text(conversation)

    try:
        # 第一步：意图分类
        print("🤖 调用LLM分析中...")
        intent = llm_service.classify_intent(combined_content)

        # 第二步：根据意图进行相应分析
        print(f"🎭 意图分类结果: {intent.value}")

        if intent == IntentType.KNOWLEDGE_QA:
            analysis = llm_service.analyze_knowledge(combined_content)
            print("📚 知识塔分析结果:")
            print(f"   📚 教育水平: {analysis.predicted_education_level}")
            print(f"   📚 是否作业: {analysis.is_homework}")
            print(f"   💭 推理过程: {analysis.reasoning}")
        else:
            analysis = llm_service.analyze_social(combined_content)
            print("👥 社交塔分析结果:")
            print(f"   👥 情绪波动: {analysis.emotional_score:.3f}")
            print(f"   👥 逻辑性: {analysis.logic_score:.3f}")
            print(f"   👥 未成年人倾向: {analysis.minor_tendency_score:.3f}")
            print(f"   💭 推理过程: {analysis.reasoning}")

        # 第三步：更新用户画像
        user_profile.update_score(analysis, intent)

        # 获取更新后的概率
        minor_prob = user_profile.minor_probability
        print(f"🎯 未成年人概率: {minor_prob:.3f}")

        # 决策逻辑 - 进一步降低阈值以提高对未成年人的敏感度
        is_predicted_minor = minor_prob > 0.65
        is_actual_minor = user_type == "minor"

        is_correct = is_predicted_minor == is_actual_minor
        status = "✅ 正确" if is_correct else "❌ 错误"
        print(f"🎯 预测结果: {status}")

        # 显示知识统计（如果有）
        if user_profile.knowledge_stats:
            print(f"📊 知识统计: {user_profile.knowledge_stats}")

        # 显示历史特征
        if user_profile.history_summary:
            print("🔍 关键特征:")
            for feature in user_profile.history_summary[-3:]:  # 只显示最近3个
                print(f"   - {feature}")

        return {
            'sample_id': sample_id,
            'actual_type': user_type,
            'predicted_minor': is_predicted_minor,
            'minor_probability': minor_prob,
            'is_correct': is_correct,
            'intent': intent.value,
            'analysis': analysis.dict() if hasattr(analysis, 'dict') else str(analysis)
        }

    except Exception as e:
        print(f"❌ 处理样本 {sample_id} 时出错: {e}")
        return {
            'sample_id': sample_id,
            'actual_type': user_type,
            'error': str(e),
            'is_correct': False
        }


def main():
    """主函数"""
    print("🧪 未成年人识别与保护系统 - 对话样本测试")
    print("=" * 50)

    # 加载对话样本
    samples = load_conversation_samples()
    if not samples:
        return

    # 初始化LLM服务
    try:
        llm_service = get_llm_service()
        print("✅ LLM服务初始化成功")
    except Exception as e:
        print(f"❌ LLM服务初始化失败: {e}")
        return

    # 初始化用户数据库（内存模拟）
    db: Dict[str, UserProfile] = {}

    # 处理结果统计
    results = []
    correct_predictions = 0
    total_samples = len(samples)
    minor_samples = 0
    adult_samples = 0

    # 处理每个样本
    for i, sample in enumerate(samples, 1):
        print(f"\n📊 处理进度: {i}/{total_samples}")

        result = process_sample(sample, llm_service, db)
        results.append(result)

        if result.get('actual_type') == 'minor':
            minor_samples += 1
        else:
            adult_samples += 1

        if result.get('is_correct', False):
            correct_predictions += 1

    # 输出统计结果
    print("\n" + "=" * 60)
    print("📊 测试结果统计")
    print("=" * 60)

    accuracy = correct_predictions / total_samples if total_samples > 0 else 0
    print(f"🎯 总体准确率: {accuracy:.3f} ({correct_predictions}/{total_samples})")

    if minor_samples > 0:
        minor_correct = sum(1 for r in results if r.get('actual_type') == 'minor' and r.get('is_correct', False))
        minor_accuracy = minor_correct / minor_samples
        print(f"👶 未成年人识别准确率: {minor_accuracy:.3f} ({minor_correct}/{minor_samples})")

    if adult_samples > 0:
        adult_correct = sum(1 for r in results if r.get('actual_type') == 'adult' and r.get('is_correct', False))
        adult_accuracy = adult_correct / adult_samples
        print(f"👨 成年人识别准确率: {adult_accuracy:.3f} ({adult_correct}/{adult_samples})")

    # 显示详细结果
    print("\n📋 详细结果:")
    print("-" * 40)

    for result in results:
        sample_id = result['sample_id']
        actual = result['actual_type']
        predicted_prob = result.get('minor_probability', 0)
        is_correct = result.get('is_correct', False)
        status = "✅" if is_correct else "❌"

        if 'error' in result:
            print(f"{status} {sample_id}: {actual} -> 错误 ({result['error']})")
        else:
            print(f"{status} {sample_id}: {actual} -> {predicted_prob:.3f}")

    print("\n🎉 测试完成！")


if __name__ == "__main__":
    main()

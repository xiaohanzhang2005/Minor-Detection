# 模块说明：
# - 旧 executor demo 的 smoke test。
# - 属于历史测试，不是当前主链关键回归。

"""
MVP Demo 测试脚本
验证 ICBO 核心识别流程
"""

import json
import sys
from pathlib import Path

# 添加项目根目录到 Python 路径
ROOT_DIR = Path(__file__).parent.absolute()
sys.path.insert(0, str(ROOT_DIR))

from src.executor import ExecutorSkill
from src.models import SkillOutput


def test_single_text():
    """测试单条文本分析"""
    print("=" * 60)
    print("🧪 测试 1: 单条文本分析")
    print("=" * 60)

    executor = ExecutorSkill()

    # 测试样本：典型未成年人表达
    test_text = "在吗……我真的快疯了，明天就要月考了，一点都看不进去书。我妈天天在我耳边念叨，说我不努力，可是她根本不知道我有多累。我真的好想消失啊……"

    print(f"\n📝 输入文本:\n{test_text}\n")

    result = executor.run_text(test_text)

    print("📊 分析结果:")
    print(f"  - 是否未成年人: {result.is_minor}")
    print(f"  - 置信度: {result.minor_confidence:.2f}")
    print(f"  - 风险等级: {result.risk_level.value}")
    print(f"\n🔍 ICBO 特征:")
    print(f"  - 意图 (I): {result.icbo_features.intention}")
    print(f"  - 认知 (C): {result.icbo_features.cognition}")
    print(f"  - 风格 (B): {result.icbo_features.behavior_style}")
    print(f"  - 时间 (O): {result.icbo_features.opportunity_time}")
    print(f"\n👤 用户画像:")
    print(f"  - 年龄: {result.user_persona.age}")
    print(f"  - 年龄区间: {result.user_persona.age_range}")
    print(f"  - 教育阶段: {result.user_persona.education_stage}")
    print(f"  - 身份标记: {result.user_persona.identity_markers}")
    print(f"\n💡 推理过程:\n{result.reasoning}")
    print(f"\n📌 关键证据: {result.key_evidence}")

    return result


def test_multi_turn_conversation():
    """测试多轮对话分析"""
    print("\n" + "=" * 60)
    print("🧪 测试 2: 多轮对话分析")
    print("=" * 60)

    executor = ExecutorSkill()

    # 从现有数据中提取一个样本
    conversation = [
        {"role": "user", "content": "在吗……我真的要疯了，现在两点半了，一点困意都没有。"},
        {"role": "assistant", "content": "这么晚了还没睡着，听起来你现在心情很烦躁，是发生什么事让你这么困扰吗？"},
        {"role": "user", "content": "还能有啥，要开学了啊！！看到班群里发那个返校通知，我心跳一下子就快得不行，好几天都这样了。"},
        {"role": "assistant", "content": "原来是开学带来的压力。这种'开学综合征'确实会让很多人感到焦虑，你是特别担心学校里的某件事吗？"},
        {"role": "user", "content": "我不知道……反正就是想到要回那个地方就喘不过气，作业还没补完，老师肯定又要找我。"},
    ]

    print("\n📝 输入对话:")
    for msg in conversation:
        role = "用户" if msg["role"] == "user" else "AI"
        print(f"  [{role}]: {msg['content']}")

    result = executor.run(conversation)

    print(f"\n📊 分析结果:")
    print(f"  - 是否未成年人: {result.is_minor}")
    print(f"  - 置信度: {result.minor_confidence:.2f}")
    print(f"  - 风险等级: {result.risk_level.value}")
    print(f"  - 年龄推测: {result.user_persona.age_range}")
    print(f"  - 教育阶段: {result.user_persona.education_stage}")

    return result


def test_adult_sample():
    """测试成年人样本（边界案例）"""
    print("\n" + "=" * 60)
    print("🧪 测试 3: 成年人样本（边界验证）")
    print("=" * 60)

    executor = ExecutorSkill()

    # 模拟一个高迷惑性的成年人样本
    test_text = "唉，考研真的太难了，二战压力好大。每天在图书馆待到十一点，回到出租屋还要继续看视频课。我导师说我的研究方向太窄了，让我重新选题，可是时间根本来不及啊……"

    print(f"\n📝 输入文本:\n{test_text}\n")

    result = executor.run_text(test_text)

    print("📊 分析结果:")
    print(f"  - 是否未成年人: {result.is_minor}")
    print(f"  - 置信度: {result.minor_confidence:.2f}")
    print(f"  - 年龄推测: {result.user_persona.age_range}")
    print(f"  - 教育阶段: {result.user_persona.education_stage}")
    print(f"\n💡 推理过程:\n{result.reasoning}")

    return result


def main():
    """运行所有测试"""
    print("\n" + "🚀" * 20)
    print("      MVP Demo - ICBO 青少年识别系统")
    print("🚀" * 20 + "\n")

    try:
        # 测试 1: 单条文本
        result1 = test_single_text()

        # 测试 2: 多轮对话
        result2 = test_multi_turn_conversation()

        # 测试 3: 成年人边界
        result3 = test_adult_sample()

        # 汇总
        print("\n" + "=" * 60)
        print("📈 测试汇总")
        print("=" * 60)
        print(f"测试1 (未成年人文本): is_minor={result1.is_minor}, confidence={result1.minor_confidence:.2f}")
        print(f"测试2 (多轮对话): is_minor={result2.is_minor}, confidence={result2.minor_confidence:.2f}")
        print(f"测试3 (成年人边界): is_minor={result3.is_minor}, confidence={result3.minor_confidence:.2f}")

        print("\n✅ MVP Demo 测试完成!")

    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

"""
未成年人识别与保护系统 - 知识问答数据库测试脚本

用途：
- 从 data/知识问答数据库 中抽取测试文本（支持单个 JSONL 或目录内多个 JSON）
- 明确避开在线校准器使用的那部分样本（校准器只用每条记录前 3 条用户话语的拼接）
- 通过主流程调用 LLM，观察知识塔分析与在线校准效果
"""

import json
import os
from typing import Any, Dict, List, Optional

from llm_service import get_llm_service
from models import IntentType
from user_profile import UserProfile


def get_qa_data_dir() -> str:
    """返回知识问答数据库目录。"""
    root_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(root_dir, "data", "知识问答数据库")


def get_qa_data_file() -> str:
    """返回知识问答数据库 JSONL 文件路径（优先）。"""
    data_dir = get_qa_data_dir()
    return os.path.join(
        data_dir, "knowledge_qa_semantic_v2_like.jsonl"
    )


def load_qa_records() -> List[Dict[str, Any]]:
    """读取知识问答数据库记录：JSONL 优先，回退目录多 JSON。"""
    data_file = get_qa_data_file()
    data_dir = get_qa_data_dir()

    records: List[Dict[str, Any]] = []

    # 1) 优先 JSONL
    if os.path.isfile(data_file):
        with open(data_file, "r", encoding="utf-8") as f:
            for i, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue

                try:
                    obj = json.loads(line)
                except Exception:
                    print(f"⚠️ JSONL 第 {i} 行解析失败，已跳过")
                    continue

                if isinstance(obj, dict):
                    records.append(obj)

            print(f"ℹ️ 加载源: jsonl -> {data_file}")
        print(f"✅ 在 {data_file} 中读取到 {len(records)} 条记录")
        return records

    # 2) 回退目录多 JSON
    if not os.path.isdir(data_dir):
        print(f"❌ 找不到目录: {data_dir}")
        return []

    files = [
        os.path.join(data_dir, f)
        for f in os.listdir(data_dir)
        if f.endswith(".json")
    ]
    if not files:
        print(f"❌ 未找到 JSONL，也未找到 .json 文件: {data_dir}")
        return []

    for path in sorted(files):
        try:
            with open(path, "r", encoding="utf-8") as f:
                obj = json.load(f)
            if isinstance(obj, dict):
                records.append(obj)
        except Exception:
            print(f"⚠️ 解析失败，已跳过: {path}")

    print(f"ℹ️ 加载源: json-dir -> {data_dir}")
    print(f"✅ 在 {data_dir} 中回退读取到 {len(records)} 条 JSON 记录")
    return records


def extract_test_text_from_conversation(
    conversation: List[Dict[str, Any]],
) -> Optional[str]:
    """
    从单个对话样本中抽取测试文本。

    关键约束：避免与在线校准器使用的文本“完全相同”。
    - 在线校准器只用：每条记录中前几轮 user 消息（最多 3 条），纯文本拼接，中间用换行分隔
    - 这里优先选择：第 4 条及之后的 user 消息（以及后续 AI 回复），并添加「用户: / AI:」前缀
    - 如果对话中 user 消息少于 4 条，则使用整段对话的带前缀拼接文本（同样不会与校准器文本完全一致）
    """
    if not conversation:
        return None

    # 先统计 user 消息数量
    user_count = 0
    segments_after_third_user: List[str] = []

    for msg in conversation:
        role = msg.get("role", "")
        content = (msg.get("content") or "").strip()
        if not content:
            continue

        if role == "user":
            user_count += 1

        # 第 4 条及之后的 user 消息以及它之后的所有消息都可以作为测试区段
        if user_count >= 4:
            if role == "user":
                segments_after_third_user.append(f"用户: {content}")
            elif role == "assistant":
                segments_after_third_user.append(f"AI: {content}")

    # 优先使用“第 4 条 user 之后”的内容
    if segments_after_third_user:
        return "\n".join(segments_after_third_user)

    # 回退方案：使用整段对话的带前缀文本（与校准器原始 user 拼接文本仍然不同）
    all_segments: List[str] = []
    for msg in conversation:
        role = msg.get("role", "")
        content = (msg.get("content") or "").strip()
        if not content:
            continue
        if role == "user":
            all_segments.append(f"用户: {content}")
        elif role == "assistant":
            all_segments.append(f"AI: {content}")

    return "\n".join(all_segments) if all_segments else None


def iter_qa_test_samples() -> List[Dict[str, Any]]:
    """
    将知识问答数据库中的每条记录转成一个“测试样本”：
    - id: dataset_id（缺失则使用行号）
    - stage / subject: 来自 _meta 元信息（如果有）
    - text: 按上面的规则抽取的测试文本
    """
    records = load_qa_records()
    samples: List[Dict[str, Any]] = []

    for idx, data in enumerate(records, start=1):
        meta = data.get("_meta", {})
        if not isinstance(meta, dict):
            meta = {}

        stage = meta.get("_stage")
        subject = meta.get("_subject")
        conversation = data.get("conversation") or []
        if not isinstance(conversation, list):
            continue

        text = extract_test_text_from_conversation(conversation)
        if not text:
            print(f"⚠️ 第 {idx} 条记录未找到可用对话文本，已跳过")
            continue

        sample_id = str(data.get("dataset_id") or f"line_{idx}")
        samples.append(
            {
                "id": sample_id,
                "stage": stage,
                "subject": subject,
                "text": text,
            }
        )

    print(f"✅ 成功构造 {len(samples)} 个测试样本")
    return samples


def run_single_sample(
    sample: Dict[str, Any],
    llm_service,
    db: Dict[str, UserProfile],
) -> None:
    """对单个样本运行系统主流程并打印结果。"""
    sample_id = sample["id"]
    stage = sample.get("stage") or "未知学段"
    subject = sample.get("subject") or "未知学科"
    text = sample["text"]

    print("\n" + "=" * 60)
    print(f"🧪 测试样本: {sample_id}")
    print(f"📚 学段: {stage} | 学科: {subject}")
    print("-" * 60)
    print("📝 输入文本（截断预览，前 300 字）：")
    print(text[:300] + ("..." if len(text) > 300 else ""))
    print("-" * 60)

    # 使用文件名作为“虚拟用户 ID”
    user_id = f"qa_{sample_id}"
    if user_id not in db:
        db[user_id] = UserProfile(user_id)
    user_profile = db[user_id]

    # 1) 意图分类
    print("🤖 调用 LLM 进行意图分类...")
    intent = llm_service.classify_intent(text)
    print(f"🎭 意图分类结果: {intent.value}")

    # 2) 分塔分析（这里重点关注知识塔）
    if intent == IntentType.KNOWLEDGE_QA:
        analysis = llm_service.analyze_knowledge(text)
        print("📚 知识塔分析结果（含在线校准）:")
        print(f"   📚 预测教育水平: {analysis.predicted_education_level}")
        print(f"   📚 是否作业: {analysis.is_homework}")
        print("   💭 推理与校准说明（截断至前 400 字）:")
        reasoning = analysis.reasoning or ""
        print("   " + reasoning[:400].replace("\n", "\n   "))
        if len(reasoning) > 400:
            print("   ...")
    else:
        analysis = llm_service.analyze_social(text)
        print("👥 社交塔分析结果:")
        print(f"   👥 情绪波动: {analysis.emotional_score:.3f}")
        print(f"   👥 逻辑性: {analysis.logic_score:.3f}")
        print(f"   👥 未成年人倾向: {analysis.minor_tendency_score:.3f}")
        print("   💭 推理说明（截断至前 400 字）:")
        reasoning = analysis.reasoning or ""
        print("   " + reasoning[:400].replace("\n", "\n   "))
        if len(reasoning) > 400:
            print("   ...")

    # 3) 更新用户画像并查看未成年人概率
    user_profile.update_score(analysis, intent)
    minor_prob = user_profile.minor_probability
    print(f"🎯 当前用户未成年人概率估计: {minor_prob:.3f}")

    # 4) 如有知识统计与历史特征则打印
    if user_profile.knowledge_stats:
        print(f"📊 知识统计: {user_profile.knowledge_stats}")
    if user_profile.history_summary:
        print("🔍 关键特征（最近若干条）:")
        for feature in user_profile.history_summary[-3:]:
            print(f"   - {feature}")


def main() -> None:
    """主入口：批量跑一遍知识问答数据库中的样本。"""
    print("🧪 未成年人识别与保护系统 - 知识问答数据库测试")
    print("=" * 60)

    samples = iter_qa_test_samples()
    if not samples:
        return

    # 初始化 LLM 服务
    try:
        llm_service = get_llm_service()
        print("✅ LLM 服务初始化成功")
    except Exception as e:
        print(f"❌ LLM 服务初始化失败: {e}")
        return

    # 简单的内存用户数据库
    db: Dict[str, UserProfile] = {}

    # 逐个样本测试
    for i, sample in enumerate(samples, 1):
        print(f"\n📊 处理进度: {i}/{len(samples)}")
        run_single_sample(sample, llm_service, db)

    print("\n🎉 知识问答数据库测试完成！")


if __name__ == "__main__":
    main()


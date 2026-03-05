"""
未成年人识别与保护系统 - 社交对话校准器最小测试脚本

用途：
- 从 data/社交问答/semantic_data_v2.jsonl 中抽取社交对话样本
- 对比“LLM 原始社交分析”与“社交库校准后的分析”
- 重点可视化展示 reasoning 字段中的校准增量说明
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
from typing import Any, Dict, List, Optional

# 兼容通过 VS Code 启动器执行脚本时的模块导入路径
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

from llm_service import get_llm_service
from models import SocialAnalysis
from social_chat_calibrator import calibrate_social_analysis


def get_social_data_file() -> str:
    """返回社交问答 JSONL 数据文件路径。"""
    root_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(root_dir, "data", "社交问答", "semantic_data_v2.jsonl")


def load_social_records(max_records: int = 20) -> List[Dict[str, Any]]:
    """读取 JSONL 数据，返回前 max_records 条可用记录。"""
    path = get_social_data_file()
    if not os.path.isfile(path):
        print(f"❌ 找不到文件: {path}")
        return []

    records: List[Dict[str, Any]] = []
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if len(records) >= max_records:
                    break
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                except Exception:
                    continue
                if isinstance(data, dict):
                    records.append(data)
    except Exception as e:
        print(f"❌ 读取 JSONL 失败: {e}")
        return []

    print(f"✅ 已加载 {len(records)} 条社交样本")
    return records


def extract_test_text(record: Dict[str, Any]) -> Optional[str]:
    """
    从单条社交样本构造测试文本。

    关键点：避免与校准器样本“完全相同”的拼接方式。
    - 校准器内部样本：前 4 条 user 文本直接拼接
    - 这里优先采用：从第 3 条 user 消息开始，且加入“用户:/AI:”前缀
    - 若不足则回退到整段带前缀拼接
    """
    conversation = record.get("conversation") or []
    if not isinstance(conversation, list):
        return None

    user_count = 0
    preferred_segments: List[str] = []

    for item in conversation:
        if not isinstance(item, dict):
            continue
        role = item.get("role", "")
        content = (item.get("content") or "").strip()
        if not content:
            continue

        if role == "user":
            user_count += 1

        if user_count >= 3:
            if role == "user":
                preferred_segments.append(f"用户: {content}")
            elif role == "assistant":
                preferred_segments.append(f"AI: {content}")

    if preferred_segments:
        return "\n".join(preferred_segments)

    fallback_segments: List[str] = []
    for item in conversation:
        if not isinstance(item, dict):
            continue
        role = item.get("role", "")
        content = (item.get("content") or "").strip()
        if not content:
            continue
        if role == "user":
            fallback_segments.append(f"用户: {content}")
        elif role == "assistant":
            fallback_segments.append(f"AI: {content}")

    return "\n".join(fallback_segments) if fallback_segments else None


def run_raw_social_analysis(llm_service, text: str) -> SocialAnalysis:
    """调用 LLMService 内部通用接口，获取“未校准”的原始社交分析结果。"""
    system_prompt = """你是一个心理专家。请分析用户对话的情绪波动、逻辑性和语言风格，特别关注对话的连贯性和思维逻辑。

分析维度：
1. 情绪波动 (emotional_score): 0.0-1.0，评估情绪表达的激烈程度
   - 低分：平静、理性、情绪稳定
   - 高分：激动、焦虑、强烈情感、前言不搭后语

2. 逻辑性 (logic_score): 0.0-1.0，评估表达的条理性
   - 低分：跳跃性思维、缺乏逻辑连接、对AI回复理解不充分、回答偏题
   - 高分：结构清晰、有条理、对AI问题回答切题、思维连贯

3. 未成年人倾向 (minor_tendency_score): 0.0-1.0，基于以下特征评估：
   - 未成年人特征：情绪极端化、逻辑简单、使用校园/网络黑话、易受暗示、表达不成熟、对复杂问题理解困难
   - 成年人特征：情绪稳定、逻辑严谨、专业术语、理性分析、成熟表达、思维深入

重点分析对话特征：
- 是否能理解AI的问题并做出相关回答
- 回答是否逻辑连贯、前后一致
- 是否使用年龄特征明显的语言和思维方式
- 对复杂概念的理解和表达能力

请以JSON格式返回结果：
{
  "emotional_score": 0.0-1.0,
  "logic_score": 0.0-1.0,
  "minor_tendency_score": 0.0-1.0,
  "reasoning": "详细的分析推理过程，包括对话连贯性、逻辑思维和年龄特征评估"
}"""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": text},
    ]

    # 这里调用内部统一解析逻辑，得到“原始（未校准）”结果
    return llm_service._call_with_retry(messages, SocialAnalysis)


def print_compare_result(
    sample_id: str,
    text: str,
    raw_analysis: SocialAnalysis,
    calibrated_analysis: SocialAnalysis,
) -> None:
    """打印原始分析 vs 校准后分析。"""
    print("\n" + "=" * 70)
    print(f"🧪 样本: {sample_id}")
    print("-" * 70)
    print("📝 输入文本（前 260 字）：")
    print(text[:260] + ("..." if len(text) > 260 else ""))
    print("-" * 70)

    print("📊 原始分析分数:")
    print(f"   emotional_score: {raw_analysis.emotional_score:.3f}")
    print(f"   logic_score: {raw_analysis.logic_score:.3f}")
    print(f"   minor_tendency_score: {raw_analysis.minor_tendency_score:.3f}")

    print("📊 校准后分析分数:")
    print(f"   emotional_score: {calibrated_analysis.emotional_score:.3f}")
    print(f"   logic_score: {calibrated_analysis.logic_score:.3f}")
    print(f"   minor_tendency_score: {calibrated_analysis.minor_tendency_score:.3f}")

    raw_reasoning = raw_analysis.reasoning or ""
    calibrated_reasoning = calibrated_analysis.reasoning or ""

    print("\n💭 原始 reasoning（前 320 字）：")
    print(raw_reasoning[:320].replace("\n", "\n   "))
    if len(raw_reasoning) > 320:
        print("...")

    print("\n🧭 校准后 reasoning（前 520 字）：")
    print(calibrated_reasoning[:520].replace("\n", "\n   "))
    if len(calibrated_reasoning) > 520:
        print("...")

    if calibrated_reasoning != raw_reasoning:
        added_part = calibrated_reasoning[len(raw_reasoning):].strip()
        if added_part:
            print("\n✅ 校准增量片段：")
            print(added_part[:260].replace("\n", "\n   "))
        else:
            print("\n✅ 检测到 reasoning 有变化（可能发生在中间位置）。")
    else:
        print("\nℹ️ 本样本未触发校准阈值，reasoning 无新增。")


def main() -> None:
    """主入口：最小化对比测试。"""
    parser = argparse.ArgumentParser(
        description="社交校准器最小测试：对比原始社交分析与校准后 reasoning"
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=3,
        help="最多处理的样本数量（默认: 3）",
    )
    args = parser.parse_args()

    max_samples = max(1, args.max_samples)

    print("🧪 社交校准器最小测试：原始社交分析 vs 校准后 reasoning")
    print("=" * 70)
    print(f"⚙️ 参数: max_samples={max_samples}")

    records = load_social_records(max_records=max_samples)
    if not records:
        return

    try:
        llm_service = get_llm_service()
        print("✅ LLM 服务初始化成功")
    except Exception as e:
        print(f"❌ LLM 服务初始化失败: {e}")
        return

    processed = 0
    for i, record in enumerate(records, 1):
        sample_id = str(record.get("dataset_id") or f"sample_{i}")
        text = extract_test_text(record)
        if not text:
            continue

        print(f"\n📊 处理进度: {i}/{len(records)}")

        try:
            raw_analysis = run_raw_social_analysis(llm_service, text)
            calibrated_analysis = calibrate_social_analysis(copy.deepcopy(raw_analysis), text)
            print_compare_result(sample_id, text, raw_analysis, calibrated_analysis)
            processed += 1
        except Exception as e:
            print(f"⚠️ 样本 {sample_id} 处理失败，已跳过: {e}")

    print("\n" + "=" * 70)
    print(f"🎉 测试完成，共成功处理 {processed} 条样本")


if __name__ == "__main__":
    main()

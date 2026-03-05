"""
社交对话数据库在线校准模块

作用：
- 利用 data/社交问答/semantic_data_v2.jsonl 中的标注对话，作为“参考样本库”
  为 LLM 的社交分析结果做轻量级在线校准

设计原则：
- 只做“辅助判定”，不替代 LLM 主判断
- 数据集缺失或解析失败时静默降级，不影响主流程
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import List, Optional

from models import SocialAnalysis


@dataclass
class SocialExample:
    """从社交问答数据库抽取的简化样本"""

    text: str
    age: Optional[int]


@dataclass
class RetrievedCase:
    """检索得到的候选样本"""

    example: SocialExample
    score: float


class SocialChatCalibrator:
    """
    基于社交问答数据库的轻量级校准器

    - 预加载样本到内存
    - 使用字符 Jaccard 相似度做近邻检索
    - 对 LLM 的 SocialAnalysis 结果补充年龄/年级暗示
    """

    def __init__(
        self,
        data_file: Optional[str] = None,
        max_examples: int = 3000,
        top_k: int = 3,
        similarity_threshold: float = 0.40,
    ) -> None:
        root_dir = os.path.dirname(os.path.abspath(__file__))
        default_file = os.path.join(root_dir, "data", "社交问答", "semantic_data_v2.jsonl")

        self.data_file = data_file or default_file
        self.max_examples = max_examples
        self.top_k = top_k
        self.similarity_threshold = similarity_threshold

        self.examples: List[SocialExample] = []
        self._loaded = False
        self._load_examples_if_available()

    # === 公共入口 ===

    def calibrate_social_analysis(
        self, analysis: SocialAnalysis, text: str
    ) -> SocialAnalysis:
        """
        对 LLM 的社交分析结果进行温和校准。
        - 当检索到高相似样本时，在 reasoning 中补充年龄/年级暗示
        """
        if not self._loaded or not text:
            return analysis

        retrieved = self._retrieve_top_k(text, self.top_k)
        if not retrieved:
            return analysis

        # 取 top-k 的平均相似度做触发判定
        avg_similarity = sum(r.score for r in retrieved) / len(retrieved)
        if avg_similarity < self.similarity_threshold:
            return analysis

        ages = [r.example.age for r in retrieved if isinstance(r.example.age, int)]
        if not ages:
            analysis.reasoning += (
                f"\n[社交库参考] 检索到高相似社交对话样本（Top-{len(retrieved)}，"
                f"平均相似度≈{avg_similarity:.2f}），但样本年龄缺失，未提供年龄校准建议。"
            )
            return analysis

        avg_age = round(sum(ages) / len(ages), 1)
        grade_hint = self._age_to_grade_hint(avg_age)

        analysis.reasoning += (
            f"\n[社交库校准] 当前表达与数据库中社交样本高度相似"
            f"（Top-{len(retrieved)}，平均相似度≈{avg_similarity:.2f}）。"
            f"相似样本平均年龄≈{avg_age}岁，{grade_hint}，可作为未成年人倾向判定的辅助证据。"
        )

        return analysis

    # === 内部：加载与检索 ===

    def _load_examples_if_available(self) -> None:
        """尝试从 JSONL 文件加载样本，失败时静默降级。"""
        if self._loaded:
            return

        try:
            if not os.path.isfile(self.data_file):
                return

            with open(self.data_file, "r", encoding="utf-8") as f:
                for line in f:
                    if len(self.examples) >= self.max_examples:
                        break

                    line = line.strip()
                    if not line:
                        continue

                    example = self._try_parse_line(line)
                    if example is not None:
                        self.examples.append(example)

            self._loaded = len(self.examples) > 0
        except Exception:
            self._loaded = False

    def _try_parse_line(self, line: str) -> Optional[SocialExample]:
        """解析单行 JSONL，失败静默跳过。"""
        try:
            data = json.loads(line)
        except Exception:
            return None

        conversation = data.get("conversation", [])
        user_texts: List[str] = []
        for item in conversation:
            if isinstance(item, dict) and item.get("role") == "user":
                content = item.get("content")
                if isinstance(content, str):
                    content = content.strip()
                    if content:
                        user_texts.append(content)

        if not user_texts:
            return None

        combined_text = "\n".join(user_texts[:4])
        if not combined_text:
            return None

        persona = data.get("user_persona", {})
        age_raw = persona.get("age") if isinstance(persona, dict) else None
        age = self._normalize_age(age_raw)

        return SocialExample(text=combined_text, age=age)

    def _retrieve_top_k(self, text: str, top_k: int) -> List[RetrievedCase]:
        """基于字符 Jaccard 相似度检索 Top-K 样本。"""
        candidates: List[RetrievedCase] = []
        for ex in self.examples:
            score = self._char_jaccard_similarity(text, ex.text)
            if score <= 0.0:
                continue
            candidates.append(RetrievedCase(example=ex, score=score))

        if not candidates:
            return []

        candidates.sort(key=lambda c: c.score, reverse=True)
        return candidates[: max(1, top_k)]

    @staticmethod
    def _normalize_age(age_raw: object) -> Optional[int]:
        """将年龄字段规范化为 int，无法解析则返回 None。"""
        if isinstance(age_raw, int):
            return age_raw if 6 <= age_raw <= 80 else None

        if isinstance(age_raw, str):
            age_raw = age_raw.strip()
            if not age_raw:
                return None
            if age_raw.isdigit():
                value = int(age_raw)
                return value if 6 <= value <= 80 else None

        return None

    @staticmethod
    def _char_jaccard_similarity(a: str, b: str) -> float:
        """基于字符集合的 Jaccard 相似度，简单快速，适合中文短文本粗略对比。"""
        set_a = set(a)
        set_b = set(b)
        if not set_a or not set_b:
            return 0.0
        inter = set_a & set_b
        union = set_a | set_b
        if not union:
            return 0.0
        return len(inter) / len(union)

    @staticmethod
    def _age_to_grade_hint(age: float) -> str:
        """将平均年龄转换为粗粒度年级暗示。"""
        if age < 12:
            return "大致对应小学阶段"
        if age < 15:
            return "大致对应初中阶段"
        if age < 18:
            return "大致对应高中阶段"
        return "可能已超过典型中学年龄段"


_global_social_calibrator: Optional[SocialChatCalibrator] = None


def get_global_social_calibrator() -> SocialChatCalibrator:
    """获取全局社交校准器实例（懒加载，单例）。"""
    global _global_social_calibrator
    if _global_social_calibrator is None:
        _global_social_calibrator = SocialChatCalibrator()
    return _global_social_calibrator


def calibrate_social_analysis(analysis: SocialAnalysis, text: str) -> SocialAnalysis:
    """对 SocialAnalysis 进行全局校准的便捷函数。"""
    calibrator = get_global_social_calibrator()
    return calibrator.calibrate_social_analysis(analysis, text)

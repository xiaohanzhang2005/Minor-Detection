"""
知识问答数据库在线校准模块

作用：
- 利用 data/知识问答数据库 下的标注对话（支持单个 JSONL 或目录内多个 JSON），作为“刻度尺”为 LLM 的
    教育阶段预测和作业识别做轻量级在线校准

设计原则：
- 只做“辅助判定”，不替代 LLM 主判断
- 数据集缺失或解析失败时静默降级，不影响主流程
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Dict, List, Optional

from models import KnowledgeAnalysis


@dataclass
class QAExample:
    """从知识问答数据库抽取的简化样本"""

    text: str
    stage: str  # 小学 / 初中 / 高中
    subject: str


class QAKnowledgeCalibrator:
    """
    基于知识问答数据库的轻量级校准器

    - 预加载少量样本到内存（按学段分桶）
    - 使用简单的字符交集相似度估算“最像哪个学段”
    - 对 LLM 的 KnowledgeAnalysis 结果做温和校正
    """

    SUPPORTED_STAGES = {"小学", "初中", "高中"}

    def __init__(
        self,
        data_dir: Optional[str] = None,
        data_file: Optional[str] = None,
        max_examples_per_stage: int = 80,
    ) -> None:
        # 默认数据路径：优先 JSONL，回退目录多 JSON
        root_dir = os.path.dirname(os.path.abspath(__file__))
        default_dir = os.path.join(root_dir, "data", "知识问答数据库")
        default_file = os.path.join(default_dir, "knowledge_qa_semantic_v2_like.jsonl")

        # 兼容参数：
        # - data_file: 显式指定文件
        # - data_dir: 目录或文件路径
        # 运行时会优先尝试 JSONL，再回退到目录内多个 JSON
        self.data_dir = data_dir or default_dir
        if data_file:
            self.data_file = data_file
        elif data_dir:
            self.data_file = (
                os.path.join(data_dir, "knowledge_qa_semantic_v2_like.jsonl")
                if os.path.isdir(data_dir)
                else data_dir
            )
        else:
            self.data_file = default_file

        self.max_examples_per_stage = max_examples_per_stage

        self.examples_by_stage: Dict[str, List[QAExample]] = {
            stage: [] for stage in self.SUPPORTED_STAGES
        }

        self._loaded = False
        self._load_examples_if_available()

    # === 公共入口 ===

    def calibrate_knowledge(
        self, analysis: KnowledgeAnalysis, text: str
    ) -> KnowledgeAnalysis:
        """
        对 LLM 的知识分析结果进行温和校准。
        - 可能会微调 predicted_education_level
        - 可能会将 is_homework 从 False → True
        """
        # 没有加载到任何样本时，直接返回原结果
        if not self._loaded:
            return analysis

        # 1) 基于样本估算学段
        stage, score = self._estimate_stage(text)

        # 2) 根据相似度和冲突情况，决定是否调整 predicted_education_level
        if stage and score >= 0.25:
            analysis = self._maybe_adjust_education_level(analysis, stage, score)

        # 3) 基于简单规则和语料特征增强作业识别
        if not analysis.is_homework:
            if self._looks_like_homework(text):
                analysis.is_homework = True
                analysis.reasoning += "\n[规则增强] 文本包含典型作业/考试表达，增强 is_homework=True 判定。"

        return analysis

    # === 内部：加载与估计 ===

    def _load_examples_if_available(self) -> None:
        """尝试加载样本：优先 JSONL，回退目录多 JSON。"""
        if self._loaded:
            return

        try:
            loaded_any = False

            # 1) 优先 JSONL
            if os.path.isfile(self.data_file) and self.data_file.endswith(".jsonl"):
                loaded_any = self._load_from_jsonl(self.data_file)
                if loaded_any:
                    print(f"[QAKnowledgeCalibrator] 加载源: jsonl -> {self.data_file}")

            # 2) 回退目录多 JSON
            if not loaded_any:
                candidate_dir = self.data_dir if os.path.isdir(self.data_dir) else os.path.dirname(self.data_file)
                if os.path.isdir(candidate_dir):
                    loaded_any = self._load_from_json_dir(candidate_dir)
                    if loaded_any:
                        print(f"[QAKnowledgeCalibrator] 加载源: json-dir -> {candidate_dir}")

            # 至少加载到一个样本才算成功
            if loaded_any and any(self.examples_by_stage.values()):
                self._loaded = True
        except Exception:
            # 任何异常都不影响主流程，直接视为未加载
            self._loaded = False

    def _load_from_jsonl(self, path: str) -> bool:
        """从 JSONL 加载样本。"""
        loaded_any = False
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if all(
                    len(v) >= self.max_examples_per_stage
                    for v in self.examples_by_stage.values()
                ):
                    break

                line = line.strip()
                if not line:
                    continue

                try:
                    data = json.loads(line)
                except Exception:
                    continue

                if isinstance(data, dict):
                    before = sum(len(v) for v in self.examples_by_stage.values())
                    self._try_load_record(data)
                    after = sum(len(v) for v in self.examples_by_stage.values())
                    if after > before:
                        loaded_any = True
        return loaded_any

    def _load_from_json_dir(self, directory: str) -> bool:
        """从目录内多个 JSON 加载样本。"""
        loaded_any = False
        files = [f for f in os.listdir(directory) if f.endswith(".json")]
        for filename in files:
            if all(
                len(v) >= self.max_examples_per_stage
                for v in self.examples_by_stage.values()
            ):
                break

            path = os.path.join(directory, filename)
            try:
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
            except Exception:
                continue

            if isinstance(data, dict):
                before = sum(len(v) for v in self.examples_by_stage.values())
                self._try_load_record(data)
                after = sum(len(v) for v in self.examples_by_stage.values())
                if after > before:
                    loaded_any = True
        return loaded_any

    def _try_load_record(self, data: Dict[str, object]) -> None:
        """从单条 JSON 记录中抽取样本。失败静默跳过。"""
        meta = data.get("_meta", {})
        if not isinstance(meta, dict):
            return

        stage = meta.get("_stage")
        subject = meta.get("_subject", "")
        if stage not in self.SUPPORTED_STAGES:
            return

        # 已达到该学段配额
        if len(self.examples_by_stage[stage]) >= self.max_examples_per_stage:
            return

        conversation = data.get("conversation", [])
        if not isinstance(conversation, list):
            return

        user_texts: List[str] = []
        for item in conversation:
            if isinstance(item, dict) and item.get("role") == "user":
                content = item.get("content")
                if isinstance(content, str):
                    user_texts.append(content.strip())

        if not user_texts:
            return

        # 取前几轮用户话语拼接为一个样本
        combined_text = "\n".join(user_texts[:3])
        if not combined_text:
            return

        example = QAExample(text=combined_text, stage=stage, subject=subject)
        self.examples_by_stage[stage].append(example)

    def _estimate_stage(self, text: str) -> tuple[Optional[str], float]:
        """
        使用简单字符交集相似度，估算最接近的学段。

        Returns:
            (stage, score) 如果无有效样本则返回 (None, 0.0)
        """
        if not text:
            return None, 0.0

        best_stage: Optional[str] = None
        best_score = 0.0

        for stage, examples in self.examples_by_stage.items():
            for ex in examples:
                score = self._char_jaccard_similarity(text, ex.text)
                if score > best_score:
                    best_score = score
                    best_stage = stage

        return best_stage, best_score

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

    # === 内部：校准规则 ===

    def _maybe_adjust_education_level(
        self, analysis: KnowledgeAnalysis, stage: str, score: float
    ) -> KnowledgeAnalysis:
        """
        在相似度足够高时，基于学段对 predicted_education_level 做温和校准。
        只做“贴近真实教材”的微调，不做剧烈跳变。
        """
        original = analysis.predicted_education_level

        # 映射：学段 -> 目标教育水平标签
        stage_to_level = {
            "小学": "小学",
            "初中": "初中",
            "高中": "高中",
        }
        target_level = stage_to_level.get(stage)
        if not target_level:
            return analysis

        # 如果 LLM 已经和样本学段一致，则仅在 reasoning 里记录信息
        if original == target_level:
            analysis.reasoning += (
                f"\n[知识库参考] 与「{stage}」阶段教材问答样本高度相似"
                f"（相似度≈{score:.2f}），增强当前判断置信度。"
            )
            return analysis

        # 当相似度较高且 LLM 判断在相邻层级（例如 初中 vs 高中）时，可以做温和纠偏
        neighbor_pairs = {
            ("小学", "初中"),
            ("初中", "小学"),
            ("初中", "高中"),
            ("高中", "初中"),
        }
        if (original, target_level) in neighbor_pairs and score >= 0.30:
            analysis.reasoning += (
                f"\n[知识库校准] LLM 预测为「{original}」，"
                f"但与「{stage}」阶段样本相似度较高（≈{score:.2f}），"
                f"将教育水平微调为「{target_level}」。"
            )
            analysis.predicted_education_level = target_level
            return analysis

        # 其余情况只做参考提示，不直接改标签
        analysis.reasoning += (
            f"\n[知识库参考] 与「{stage}」阶段样本有一定相似度（≈{score:.2f}），"
            f"暂不覆盖原预测「{original}」，仅作为后续决策参考。"
        )
        return analysis

    @staticmethod
    def _looks_like_homework(text: str) -> bool:
        """
        使用简单关键词规则判断是否“很像作业相关文本”。
        只在 LLM 判定为 False 时用作 recall 增强。
        """
        if not text:
            return False

        # 一些常见的作业 / 考试语境关键词
        homework_keywords = [
            "作业",
            "家庭作业",
            "练习册",
            "卷子",
            "试卷",
            "考试",
            "期末",
            "月考",
            "周测",
            "单元测试",
            "试题",
            "题目",
            "第几题",
            "老师布置",
            "布置了一道",
        ]
        return any(k in text for k in homework_keywords)


_global_calibrator: Optional[QAKnowledgeCalibrator] = None


def get_global_calibrator() -> QAKnowledgeCalibrator:
    """获取全局校准器实例（懒加载，单例）。"""
    global _global_calibrator
    if _global_calibrator is None:
        _global_calibrator = QAKnowledgeCalibrator()
    return _global_calibrator


def calibrate_knowledge_analysis(
    analysis: KnowledgeAnalysis, text: str
) -> KnowledgeAnalysis:
    """
    对 KnowledgeAnalysis 进行全局校准的便捷函数。
    """
    calibrator = get_global_calibrator()
    return calibrator.calibrate_knowledge(analysis, text)


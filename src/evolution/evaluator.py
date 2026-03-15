"""
Skill 评估器
在 benchmark 数据集上评估 skill 的性能，生成评估报告
"""

import json
import time
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple
from dataclasses import dataclass, field, asdict
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.config import BENCHMARK_TEST_PATH, BENCHMARK_VAL_PATH
from src.executor import ExecutorSkill
from src.models import SkillOutput


@dataclass
class EvaluationMetrics:
    """评估指标"""
    # 基础指标
    total_samples: int = 0
    correct: int = 0
    accuracy: float = 0.0
    
    # 分类指标
    true_positive: int = 0   # 正确识别为未成年
    true_negative: int = 0   # 正确识别为成年人
    false_positive: int = 0  # 误判为未成年
    false_negative: int = 0  # 漏判未成年
    
    precision: float = 0.0   # 精确率
    recall: float = 0.0      # 召回率
    f1_score: float = 0.0    # F1分数
    
    # 置信度分析
    avg_confidence: float = 0.0
    confidence_std: float = 0.0
    
    # 耗时
    total_time: float = 0.0
    avg_time_per_sample: float = 0.0
    
    def compute_derived_metrics(self):
        """计算衍生指标"""
        # 准确率
        if self.total_samples > 0:
            self.accuracy = self.correct / self.total_samples
        
        # 精确率
        if self.true_positive + self.false_positive > 0:
            self.precision = self.true_positive / (self.true_positive + self.false_positive)
        
        # 召回率
        if self.true_positive + self.false_negative > 0:
            self.recall = self.true_positive / (self.true_positive + self.false_negative)
        
        # F1
        if self.precision + self.recall > 0:
            self.f1_score = 2 * self.precision * self.recall / (self.precision + self.recall)
        
        # 平均耗时
        if self.total_samples > 0:
            self.avg_time_per_sample = self.total_time / self.total_samples


@dataclass
class EvaluationResult:
    """单个样本的评估结果"""
    sample_id: str
    ground_truth: bool       # 真实标签 (is_minor)
    predicted: bool          # 预测结果
    confidence: float        # 置信度
    is_correct: bool         # 是否正确
    reasoning: str           # 推理过程
    latency: float           # 耗时(秒)
    error: Optional[str] = None  # 错误信息


@dataclass
class EvaluationReport:
    """评估报告"""
    skill_version: str
    eval_time: str
    dataset: str
    metrics: EvaluationMetrics
    results: List[EvaluationResult] = field(default_factory=list)
    errors: List[Dict[str, Any]] = field(default_factory=list)  # 错误样本
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "skill_version": self.skill_version,
            "eval_time": self.eval_time,
            "dataset": self.dataset,
            "metrics": asdict(self.metrics),
            "results_summary": {
                "total": len(self.results),
                "correct": sum(1 for r in self.results if r.is_correct),
                "errors": len(self.errors),
            },
            "error_samples": self.errors[:10],  # 只保留前10个错误
        }
    
    def summary(self) -> str:
        """生成摘要文本"""
        m = self.metrics
        lines = [
            f"[Report] 评估报告 - {self.skill_version}",
            f"   数据集: {self.dataset}",
            f"   评估时间: {self.eval_time}",
            "",
            f"   准确率: {m.accuracy:.2%}",
            f"   精确率: {m.precision:.2%}",
            f"   召回率: {m.recall:.2%}",
            f"   F1 分数: {m.f1_score:.4f}",
            "",
            f"   TP/TN/FP/FN: {m.true_positive}/{m.true_negative}/{m.false_positive}/{m.false_negative}",
            f"   平均置信度: {m.avg_confidence:.2f}",
            f"   平均耗时: {m.avg_time_per_sample:.2f}s",
        ]
        return "\n".join(lines)


class SkillEvaluator:
    """
    Skill 评估器
    在 benchmark 数据集上评估 skill 的性能
    """
    
    def __init__(
        self,
        executor: Optional[ExecutorSkill] = None,
        skill_version: str = "unknown",
        retriever: Any = None,
        rag_top_k: int = 3,
    ):
        """
        初始化评估器
        
        Args:
            executor: ExecutorSkill 实例
            skill_version: skill 版本标识
            retriever: SemanticRetriever 实例（可选），传入则评估时带 RAG
            rag_top_k: RAG 检索的 top-k 数量
        """
        self.executor = executor or ExecutorSkill()
        self.skill_version = skill_version
        self.retriever = retriever
        self.rag_top_k = rag_top_k
    
    def evaluate(
        self,
        dataset_path: Optional[str] = None,
        max_samples: Optional[int] = None,
        verbose: bool = True,
        use_test_set: bool = False,
    ) -> EvaluationReport:
        """
        执行评估
        
        Args:
            dataset_path: 数据集路径，默认使用验证集
            max_samples: 最大评估样本数
            verbose: 是否输出详细信息
            use_test_set: 是否使用测试集（仅用于最终评估，迭代优化时应使用验证集）
            
        Returns:
            EvaluationReport
        """
        # 确定数据集路径（默认使用验证集，保护测试集不被污染）
        if dataset_path is None:
            dataset_path = str(BENCHMARK_TEST_PATH if use_test_set else BENCHMARK_VAL_PATH)
        
        dataset_path = Path(dataset_path)
        if not dataset_path.exists():
            raise FileNotFoundError(f"数据集不存在: {dataset_path}")
        
        if verbose:
            print(f"[INFO] 加载数据集: {dataset_path}")
        
        # 加载数据
        samples = []
        with open(dataset_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    samples.append(json.loads(line))
        
        if max_samples:
            samples = samples[:max_samples]
        
        if verbose:
            print(f"[OK] 加载 {len(samples)} 条样本")
            rag_status = "启用" if self.retriever is not None else "关闭"
            print(f"[INFO] 开始评估... (RAG: {rag_status})")
        
        # 执行评估
        metrics = EvaluationMetrics(total_samples=len(samples))
        results = []
        errors = []
        confidences = []
        
        start_time = time.time()
        
        for i, sample in enumerate(samples):
            sample_id = sample.get("sample_id", f"sample_{i}")
            ground_truth = sample.get("is_minor", True)
            conversation = sample.get("conversation", [])
            
            if verbose and (i + 1) % 5 == 0:
                print(f"  评估进度: {i + 1}/{len(samples)}")
            
            try:
                sample_start = time.time()
                # 如果有 retriever 则走 RAG 校准路径
                if self.retriever is not None:
                    try:
                        from src.config import RAG_THRESHOLD
                        rag_results = self.retriever.retrieve(
                            conversation, top_k=self.rag_top_k, threshold=RAG_THRESHOLD,
                        )
                        context = self.retriever.format_for_prompt(rag_results) if rag_results else None
                    except Exception:
                        context = None
                    output = self.executor.run(conversation, context=context)
                else:
                    output = self.executor.run(conversation)
                sample_latency = time.time() - sample_start
                
                predicted = output.is_minor
                confidence = output.minor_confidence
                is_correct = (predicted == ground_truth)
                
                # 更新指标
                if is_correct:
                    metrics.correct += 1
                
                if ground_truth:  # 真实为未成年
                    if predicted:
                        metrics.true_positive += 1
                    else:
                        metrics.false_negative += 1
                else:  # 真实为成年人
                    if predicted:
                        metrics.false_positive += 1
                    else:
                        metrics.true_negative += 1
                
                confidences.append(confidence)
                
                result = EvaluationResult(
                    sample_id=sample_id,
                    ground_truth=ground_truth,
                    predicted=predicted,
                    confidence=confidence,
                    is_correct=is_correct,
                    reasoning=output.reasoning[:200],
                    latency=sample_latency,
                )
                results.append(result)
                
                # 记录错误样本
                if not is_correct:
                    errors.append({
                        "sample_id": sample_id,
                        "ground_truth": "minor" if ground_truth else "adult",
                        "predicted": "minor" if predicted else "adult",
                        "confidence": confidence,
                        "reasoning": output.reasoning[:300],
                    })
                
            except Exception as e:
                if verbose:
                    print(f"  [WARN] 样本 {sample_id} 评估失败: {e}")
                
                predicted = True  # 保守判定
                is_correct = (predicted == ground_truth)
                
                # 更新混淆矩阵（异常样本也必须计入）
                if is_correct:
                    metrics.correct += 1
                if ground_truth:
                    if predicted:
                        metrics.true_positive += 1
                    else:
                        metrics.false_negative += 1
                else:
                    if predicted:
                        metrics.false_positive += 1
                    else:
                        metrics.true_negative += 1
                
                result = EvaluationResult(
                    sample_id=sample_id,
                    ground_truth=ground_truth,
                    predicted=predicted,
                    confidence=0.5,
                    is_correct=is_correct,
                    reasoning="",
                    latency=0,
                    error=str(e),
                )
                results.append(result)
        
        # 计算总耗时
        metrics.total_time = time.time() - start_time
        
        # 计算置信度统计
        if confidences:
            import statistics
            metrics.avg_confidence = statistics.mean(confidences)
            if len(confidences) > 1:
                metrics.confidence_std = statistics.stdev(confidences)
        
        # 计算衍生指标
        metrics.compute_derived_metrics()
        
        # 生成报告
        report = EvaluationReport(
            skill_version=self.skill_version,
            eval_time=datetime.now().isoformat(),
            dataset=str(dataset_path),
            metrics=metrics,
            results=results,
            errors=errors,
        )
        
        if verbose:
            print("")
            print(report.summary())
        
        return report
    
    def evaluate_on_validation(
        self,
        max_samples: Optional[int] = None,
        verbose: bool = True,
    ) -> EvaluationReport:
        """在验证集上评估"""
        return self.evaluate(
            dataset_path=str(BENCHMARK_VAL_PATH),
            max_samples=max_samples,
            verbose=verbose,
        )
    
    def compare_skills(
        self,
        other_executor: ExecutorSkill,
        other_version: str,
        dataset_path: Optional[str] = None,
        max_samples: Optional[int] = None,
    ) -> Tuple[EvaluationReport, EvaluationReport]:
        """
        对比两个 skill 版本的性能
        
        Args:
            other_executor: 另一个 ExecutorSkill 实例
            other_version: 另一个版本标识
            dataset_path: 数据集路径
            max_samples: 最大样本数
            
        Returns:
            (当前版本报告, 对比版本报告)
        """
        print(f"[INFO] 对比评估: {self.skill_version} vs {other_version}")
        
        # 评估当前版本
        print(f"\n[INFO] 评估 {self.skill_version}...")
        report_current = self.evaluate(dataset_path, max_samples, verbose=False)
        
        # 评估对比版本
        print(f"\n[INFO] 评估 {other_version}...")
        other_evaluator = SkillEvaluator(other_executor, other_version)
        report_other = other_evaluator.evaluate(dataset_path, max_samples, verbose=False)
        
        # 输出对比结果
        print("\n[INFO] 对比结果:")
        print(f"  {'指标':<15} {self.skill_version:>12} {other_version:>12} {'Δ':>10}")
        print(f"  {'-'*50}")
        
        m1, m2 = report_current.metrics, report_other.metrics
        for metric_name, v1, v2 in [
            ("准确率", m1.accuracy, m2.accuracy),
            ("精确率", m1.precision, m2.precision),
            ("召回率", m1.recall, m2.recall),
            ("F1", m1.f1_score, m2.f1_score),
        ]:
            delta = v1 - v2
            delta_str = f"+{delta:.2%}" if delta > 0 else f"{delta:.2%}"
            print(f"  {metric_name:<15} {v1:>11.2%} {v2:>11.2%} {delta_str:>10}")
        
        return report_current, report_other


def run_evaluation(
    skill_version: str = "teen_detector_v1",
    dataset: str = "val",
    max_samples: Optional[int] = None,
) -> EvaluationReport:
    """
    快速运行评估
    
    Args:
        skill_version: skill 版本
        dataset: "test" 或 "val"
        max_samples: 最大样本数
        
    Returns:
        EvaluationReport
    """
    evaluator = SkillEvaluator(skill_version=skill_version)
    
    if dataset == "val":
        return evaluator.evaluate_on_validation(max_samples=max_samples)
    else:
        return evaluator.evaluate(max_samples=max_samples)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Skill 评估器")
    parser.add_argument("--dataset", default="val", choices=["test", "val"], help="数据集")
    parser.add_argument("--max-samples", type=int, default=None, help="最大样本数")
    parser.add_argument("--version", default="teen_detector_v1", help="skill 版本")
    
    args = parser.parse_args()
    
    report = run_evaluation(
        skill_version=args.version,
        dataset=args.dataset,
        max_samples=args.max_samples,
    )
    
    # 保存报告
    report_path = Path(__file__).parent.parent.parent / "data" / "eval_reports"
    report_path.mkdir(parents=True, exist_ok=True)
    
    report_file = report_path / f"eval_{args.version}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(report_file, "w", encoding="utf-8") as f:
        json.dump(report.to_dict(), f, ensure_ascii=False, indent=2)
    
    print(f"\n[OK] 报告已保存: {report_file}")

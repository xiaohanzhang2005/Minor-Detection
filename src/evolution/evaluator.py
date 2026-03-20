"""
Skill evaluator for validation and test benchmarks.
"""

from __future__ import annotations

import json
import math
import random
import tempfile
import time
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.config import BENCHMARK_TEST_PATH, BENCHMARK_VAL_PATH
from src.executor import ExecutorSkill
from src.memory import UserMemory
from src.runtime import FORMAL_SKILL_VERSION, analyze_single_session_formal_auto


@dataclass
class EvaluationMetrics:
    total_samples: int = 0
    correct: int = 0
    accuracy: float = 0.0
    true_positive: int = 0
    true_negative: int = 0
    false_positive: int = 0
    false_negative: int = 0
    precision: float = 0.0
    recall: float = 0.0
    f1_score: float = 0.0
    avg_confidence: float = 0.0
    confidence_std: float = 0.0
    total_time: float = 0.0
    avg_time_per_sample: float = 0.0

    def compute_derived_metrics(self) -> None:
        if self.total_samples > 0:
            self.accuracy = self.correct / self.total_samples
            self.avg_time_per_sample = self.total_time / self.total_samples

        if self.true_positive + self.false_positive > 0:
            self.precision = self.true_positive / (self.true_positive + self.false_positive)

        if self.true_positive + self.false_negative > 0:
            self.recall = self.true_positive / (self.true_positive + self.false_negative)

        if self.precision + self.recall > 0:
            self.f1_score = 2 * self.precision * self.recall / (self.precision + self.recall)


@dataclass
class EvaluationResult:
    sample_id: str
    ground_truth: bool
    predicted: bool
    confidence: float
    is_correct: bool
    reasoning: str
    latency: float
    error: Optional[str] = None


@dataclass
class EvaluationReport:
    skill_version: str
    eval_time: str
    dataset: str
    metrics: EvaluationMetrics
    sampling: Dict[str, Any] = field(default_factory=dict)
    results: List[EvaluationResult] = field(default_factory=list)
    errors: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "skill_version": self.skill_version,
            "eval_time": self.eval_time,
            "dataset": self.dataset,
            "metrics": asdict(self.metrics),
            "sampling": self.sampling,
            "results_summary": {
                "total": len(self.results),
                "correct": sum(1 for result in self.results if result.is_correct),
                "errors": len(self.errors),
            },
            "error_samples": self.errors[:10],
        }

    def summary(self) -> str:
        metrics = self.metrics
        lines = [
            f"[Report] Evaluation report - {self.skill_version}",
            f"   Dataset: {self.dataset}",
            f"   Eval time: {self.eval_time}",
            "",
            f"   Accuracy: {metrics.accuracy:.2%}",
            f"   Precision: {metrics.precision:.2%}",
            f"   Recall: {metrics.recall:.2%}",
            f"   F1: {metrics.f1_score:.4f}",
            "",
            (
                "   Sampling: "
                f"{self.sampling.get('strategy', 'all')} "
                f"({self.sampling.get('selected', metrics.total_samples)}/"
                f"{self.sampling.get('available', metrics.total_samples)})"
            ),
            (
                "   TP/TN/FP/FN: "
                f"{metrics.true_positive}/{metrics.true_negative}/"
                f"{metrics.false_positive}/{metrics.false_negative}"
            ),
            f"   Avg confidence: {metrics.avg_confidence:.2f}",
            f"   Avg latency: {metrics.avg_time_per_sample:.2f}s",
        ]
        return "\n".join(lines)


class SkillEvaluator:
    def __init__(
        self,
        executor: Optional[ExecutorSkill] = None,
        skill_version: str = "unknown",
        retriever: Any = None,
        rag_top_k: int = 3,
        use_memory: bool = False,
        memory_db_path: Optional[str] = None,
    ):
        self.executor = executor or ExecutorSkill()
        self.skill_version = skill_version
        self.retriever = retriever
        self.rag_top_k = rag_top_k
        self.use_memory = use_memory
        self.memory_db_path = memory_db_path

    def _uses_formal_skill_runtime(self) -> bool:
        if self.skill_version == FORMAL_SKILL_VERSION:
            return True

        skill_name = str(getattr(self.executor, "skill_name", "") or "").lower()
        skill_dir_name = str(getattr(self.executor, "skill_dir_name", "") or "").lower()
        if FORMAL_SKILL_VERSION in {skill_name, skill_dir_name}:
            return True

        skill_path = getattr(self.executor, "skill_path", None)
        if not skill_path:
            return False

        path = Path(str(skill_path))
        if path.name.lower() != "skill.md":
            return False

        return (path.parent / "references" / "output-schema.md").exists()

    @staticmethod
    def _resolve_age_bucket(sample: Dict[str, Any]) -> str:
        age_bucket = sample.get("age_bucket")
        if age_bucket:
            return str(age_bucket)

        age = sample.get("age")
        is_minor = bool(sample.get("is_minor", False))
        if isinstance(age, (int, float)):
            age = int(age)
            if is_minor:
                if age <= 12:
                    return "minor_07_12"
                if age <= 15:
                    return "minor_13_15"
                return "minor_16_17"
            if age <= 22:
                return "adult_18_22"
            if age <= 26:
                return "adult_23_26"
            return "adult_27_plus"

        return "minor_unknown" if is_minor else "adult_unknown"

    @classmethod
    def _resolve_stratum_key(cls, sample: Dict[str, Any]) -> str:
        source = str(sample.get("source", "unknown"))
        age_bucket = cls._resolve_age_bucket(sample)
        return f"{source}::{age_bucket}"

    @staticmethod
    def _allocate_stratified_counts(group_sizes: Dict[str, int], target_total: int) -> Dict[str, int]:
        if target_total <= 0:
            return {key: 0 for key in group_sizes}

        keys = sorted(group_sizes.keys())
        allocations = {key: 0 for key in keys}
        if not keys:
            return allocations

        if target_total >= len(keys):
            for key in keys:
                allocations[key] = 1
            target_total -= len(keys)

        if target_total <= 0:
            return allocations

        remaining_capacity = {key: group_sizes[key] - allocations[key] for key in keys}
        total_capacity = sum(max(0, value) for value in remaining_capacity.values())
        if total_capacity <= 0:
            return allocations

        base_additions: Dict[str, int] = {}
        remainders: List[Tuple[float, int, str]] = []
        assigned = 0
        for key in keys:
            capacity = max(0, remaining_capacity[key])
            quota = (capacity / total_capacity) * target_total if total_capacity else 0.0
            floor_value = min(capacity, math.floor(quota))
            base_additions[key] = floor_value
            assigned += floor_value
            remainders.append((quota - floor_value, capacity - floor_value, key))

        for key, floor_value in base_additions.items():
            allocations[key] += floor_value

        remaining = target_total - assigned
        remainders.sort(key=lambda item: (-item[0], -item[1], item[2]))
        for _, spare_capacity, key in remainders:
            if remaining <= 0:
                break
            if spare_capacity <= 0:
                continue
            allocations[key] += 1
            remaining -= 1

        return allocations

    @classmethod
    def _select_samples(
        cls,
        samples: List[Dict[str, Any]],
        max_samples: Optional[int],
        strategy: str,
        sample_seed: int,
    ) -> tuple[List[Dict[str, Any]], Dict[str, Any]]:
        available = len(samples)
        if max_samples is None or max_samples >= available:
            return list(samples), {
                "strategy": "all",
                "seed": sample_seed,
                "requested": max_samples,
                "selected": available,
                "available": available,
                "by_stratum": {},
            }

        rng = random.Random(sample_seed)
        if strategy == "sequential":
            selected = list(samples[:max_samples])
        elif strategy == "random":
            selected = list(samples)
            rng.shuffle(selected)
            selected = selected[:max_samples]
        elif strategy == "stratified":
            grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
            for sample in samples:
                grouped[cls._resolve_stratum_key(sample)].append(sample)

            allocations = cls._allocate_stratified_counts(
                {key: len(group) for key, group in grouped.items()},
                max_samples,
            )

            selected = []
            for key in sorted(grouped.keys()):
                group = list(grouped[key])
                rng.shuffle(group)
                selected.extend(group[: allocations.get(key, 0)])
            rng.shuffle(selected)
        else:
            raise ValueError(f"Unsupported sample strategy: {strategy}")

        by_stratum: Dict[str, int] = {}
        for sample in selected:
            key = cls._resolve_stratum_key(sample)
            by_stratum[key] = by_stratum.get(key, 0) + 1

        return selected, {
            "strategy": strategy,
            "seed": sample_seed,
            "requested": max_samples,
            "selected": len(selected),
            "available": available,
            "by_stratum": dict(sorted(by_stratum.items())),
        }

    def _build_eval_user_id(self, sample: Dict[str, Any], index: int) -> Optional[str]:
        candidate = sample.get("user_id")
        if candidate:
            return str(candidate)

        extra_info = sample.get("extra_info")
        if isinstance(extra_info, dict) and extra_info.get("user_id"):
            return str(extra_info["user_id"])

        return None

    def _create_memory(self) -> tuple[Optional[UserMemory], Optional[tempfile.TemporaryDirectory[str]]]:
        if not self.use_memory:
            return None, None

        memory_db_path = self.memory_db_path
        temp_memory_dir = None
        if memory_db_path is None:
            temp_memory_dir = tempfile.TemporaryDirectory(prefix="minor_protection_eval_memory_")
            memory_db_path = str(Path(temp_memory_dir.name) / "user_memory.db")
        return UserMemory(db_path=memory_db_path), temp_memory_dir

    def _build_context(
        self,
        conversation: List[Dict[str, str]],
        user_id: Optional[str],
        memory: Optional[UserMemory],
        raw_time_hint: str = "",
    ) -> Optional[str]:
        context_parts: List[str] = []

        if memory is not None and user_id is not None:
            profile = memory.get_profile(user_id)
            if profile is not None:
                context_parts.append(profile.to_context_string())

        if self.retriever is not None:
            try:
                from src.config import RAG_THRESHOLD

                rag_results = self.retriever.retrieve(
                    conversation,
                    top_k=self.rag_top_k,
                    threshold=RAG_THRESHOLD,
                    raw_time_hint=raw_time_hint,
                )
                if rag_results:
                    context_parts.append(self.retriever.format_for_prompt(rag_results))
            except Exception:
                pass

        return "\n\n".join(context_parts) if context_parts else None

    @staticmethod
    def _serialize_retrieval_result(result: Any) -> Dict[str, Any]:
        if isinstance(result, dict):
            return dict(result)

        payload: Dict[str, Any] = {}
        for field_name in (
            "sample_id",
            "score",
            "conversation",
            "is_minor",
            "icbo_features",
            "user_persona",
            "source",
        ):
            if hasattr(result, field_name):
                payload[field_name] = getattr(result, field_name)
        return payload

    def _build_formal_context(
        self,
        sample: Dict[str, Any],
        conversation: List[Dict[str, str]],
        user_id: Optional[str],
        memory: Optional[UserMemory],
    ) -> Optional[Dict[str, Any]]:
        context: Dict[str, Any] = {}

        icbo_features = sample.get("icbo_features")
        if isinstance(icbo_features, dict):
            raw_time_hint = icbo_features.get("opportunity_time")
            if raw_time_hint:
                context["raw_time_hint"] = str(raw_time_hint)

        if memory is not None and user_id is not None:
            profile = memory.get_profile(user_id)
            if profile is not None:
                context["prior_profile"] = {
                    "summary": profile.to_context_string(),
                    "total_sessions": profile.total_sessions,
                    "estimated_age_range": profile.estimated_age_range,
                    "education_stage": profile.education_stage,
                    "identity_markers": list(profile.identity_markers),
                }

        if self.retriever is not None:
            retrieved_cases: List[Dict[str, Any]] = []
            try:
                from src.config import RAG_THRESHOLD

                raw_time_hint = str(context.get("raw_time_hint", "") or "")
                rag_results = self.retriever.retrieve(
                    conversation,
                    top_k=self.rag_top_k,
                    threshold=RAG_THRESHOLD,
                    raw_time_hint=raw_time_hint,
                )
                retrieved_cases = [
                    self._serialize_retrieval_result(result)
                    for result in rag_results
                ]
            except Exception:
                retrieved_cases = []

            # Preserve the evaluator's explicit external-retrieval decision.
            context["retrieved_cases"] = retrieved_cases

        return context or None

    def _run_sample(
        self,
        *,
        sample: Dict[str, Any],
        sample_id: str,
        conversation: List[Dict[str, str]],
        user_id: Optional[str],
        memory: Optional[UserMemory],
    ) -> Any:
        if self._uses_formal_skill_runtime():
            formal_context = self._build_formal_context(sample, conversation, user_id, memory)
            return analyze_single_session_formal_auto(
                conversation,
                user_id=user_id or "",
                request_id=sample_id,
                source="benchmark_evaluator",
                context=formal_context,
                retrieve_top_k=self.rag_top_k,
            )

        raw_time_hint = ""
        icbo_features = sample.get("icbo_features")
        if isinstance(icbo_features, dict):
            raw_time_hint = str(icbo_features.get("opportunity_time") or "")
        context = self._build_context(conversation, user_id, memory, raw_time_hint=raw_time_hint)
        return self.executor.run(conversation, user_id=user_id, context=context)

    def evaluate(
        self,
        dataset_path: Optional[str] = None,
        max_samples: Optional[int] = None,
        sample_strategy: str = "sequential",
        sample_seed: int = 42,
        verbose: bool = True,
        use_test_set: bool = False,
    ) -> EvaluationReport:
        if dataset_path is None:
            dataset_path = str(BENCHMARK_TEST_PATH if use_test_set else BENCHMARK_VAL_PATH)

        dataset = Path(dataset_path)
        if not dataset.exists():
            raise FileNotFoundError(f"Dataset not found: {dataset}")

        if verbose:
            print(f"[INFO] Loading dataset: {dataset}")

        samples: List[Dict[str, Any]] = []
        with open(dataset, "r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if line:
                    samples.append(json.loads(line))

        samples, sampling_info = self._select_samples(
            samples=samples,
            max_samples=max_samples,
            strategy=sample_strategy,
            sample_seed=sample_seed,
        )

        if verbose:
            print(f"[OK] Loaded {len(samples)} samples")
            if sampling_info.get("strategy") != "all":
                print(
                    "[INFO] Sampling subset: "
                    f"strategy={sampling_info['strategy']} "
                    f"seed={sampling_info['seed']} "
                    f"selected={sampling_info['selected']}/{sampling_info['available']}"
                )
            rag_status = "on" if self.retriever is not None else "off"
            memory_status = "on" if self.use_memory else "off"
            print(f"[INFO] Starting evaluation... (RAG: {rag_status}, Memory: {memory_status})")

        metrics = EvaluationMetrics(total_samples=len(samples))
        results: List[EvaluationResult] = []
        errors: List[Dict[str, Any]] = []
        confidences: List[float] = []
        memory, temp_memory_dir = self._create_memory()

        start_time = time.time()
        try:
            for index, sample in enumerate(samples):
                sample_id = sample.get("sample_id", f"sample_{index}")
                ground_truth = sample.get("is_minor", True)
                conversation = sample.get("conversation", [])

                if verbose and (index + 1) % 5 == 0:
                    print(f"  Progress: {index + 1}/{len(samples)}")

                try:
                    sample_start = time.time()
                    user_id = self._build_eval_user_id(sample, index) if memory is not None else None
                    output = self._run_sample(
                        sample=sample,
                        sample_id=sample_id,
                        conversation=conversation,
                        user_id=user_id,
                        memory=memory,
                    )
                    if memory is not None and user_id is not None:
                        memory.update_profile(user_id, output)
                    sample_latency = time.time() - sample_start

                    predicted = output.is_minor
                    confidence = output.minor_confidence
                    is_correct = predicted == ground_truth

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

                    confidences.append(confidence)
                    results.append(
                        EvaluationResult(
                            sample_id=sample_id,
                            ground_truth=ground_truth,
                            predicted=predicted,
                            confidence=confidence,
                            is_correct=is_correct,
                            reasoning=output.reasoning[:200],
                            latency=sample_latency,
                        )
                    )

                    if not is_correct:
                        errors.append(
                            {
                                "sample_id": sample_id,
                                "ground_truth": "minor" if ground_truth else "adult",
                                "predicted": "minor" if predicted else "adult",
                                "confidence": confidence,
                                "reasoning": output.reasoning[:300],
                            }
                        )
                except Exception as exc:
                    if verbose:
                        print(f"  [WARN] Sample {sample_id} evaluation failed: {exc}")

                    predicted = True
                    is_correct = predicted == ground_truth
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

                    results.append(
                        EvaluationResult(
                            sample_id=sample_id,
                            ground_truth=ground_truth,
                            predicted=predicted,
                            confidence=0.5,
                            is_correct=is_correct,
                            reasoning="",
                            latency=0,
                            error=str(exc),
                        )
                    )
        finally:
            if temp_memory_dir is not None:
                temp_memory_dir.cleanup()

        metrics.total_time = time.time() - start_time

        if confidences:
            import statistics

            metrics.avg_confidence = statistics.mean(confidences)
            if len(confidences) > 1:
                metrics.confidence_std = statistics.stdev(confidences)

        metrics.compute_derived_metrics()

        report = EvaluationReport(
            skill_version=self.skill_version,
            eval_time=datetime.now().isoformat(),
            dataset=str(dataset),
            metrics=metrics,
            sampling=sampling_info,
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
        sample_strategy: str = "sequential",
        sample_seed: int = 42,
        verbose: bool = True,
    ) -> EvaluationReport:
        return self.evaluate(
            dataset_path=str(BENCHMARK_VAL_PATH),
            max_samples=max_samples,
            sample_strategy=sample_strategy,
            sample_seed=sample_seed,
            verbose=verbose,
        )

    def compare_skills(
        self,
        other_executor: ExecutorSkill,
        other_version: str,
        dataset_path: Optional[str] = None,
        max_samples: Optional[int] = None,
        sample_strategy: str = "sequential",
        sample_seed: int = 42,
    ) -> Tuple[EvaluationReport, EvaluationReport]:
        print(f"[INFO] Comparing: {self.skill_version} vs {other_version}")

        print(f"\n[INFO] Evaluating {self.skill_version}...")
        report_current = self.evaluate(
            dataset_path,
            max_samples,
            sample_strategy=sample_strategy,
            sample_seed=sample_seed,
            verbose=False,
        )

        print(f"\n[INFO] Evaluating {other_version}...")
        other_evaluator = SkillEvaluator(
            other_executor,
            other_version,
            retriever=self.retriever,
            rag_top_k=self.rag_top_k,
            use_memory=self.use_memory,
        )
        report_other = other_evaluator.evaluate(
            dataset_path,
            max_samples,
            sample_strategy=sample_strategy,
            sample_seed=sample_seed,
            verbose=False,
        )

        print("\n[INFO] Comparison result:")
        print(f"  {'Metric':<15} {self.skill_version:>12} {other_version:>12} {'Delta':>10}")
        print(f"  {'-' * 50}")

        metrics_current = report_current.metrics
        metrics_other = report_other.metrics
        for metric_name, value_current, value_other in [
            ("Accuracy", metrics_current.accuracy, metrics_other.accuracy),
            ("Precision", metrics_current.precision, metrics_other.precision),
            ("Recall", metrics_current.recall, metrics_other.recall),
            ("F1", metrics_current.f1_score, metrics_other.f1_score),
        ]:
            delta = value_current - value_other
            delta_str = f"+{delta:.2%}" if delta > 0 else f"{delta:.2%}"
            print(f"  {metric_name:<15} {value_current:>11.2%} {value_other:>11.2%} {delta_str:>10}")

        return report_current, report_other


def run_evaluation(
    skill_version: str = "teen_detector_v1",
    dataset: str = "val",
    max_samples: Optional[int] = None,
    sample_strategy: str = "sequential",
    sample_seed: int = 42,
) -> EvaluationReport:
    evaluator = SkillEvaluator(skill_version=skill_version)
    if dataset == "val":
        return evaluator.evaluate_on_validation(
            max_samples=max_samples,
            sample_strategy=sample_strategy,
            sample_seed=sample_seed,
        )
    return evaluator.evaluate(
        max_samples=max_samples,
        sample_strategy=sample_strategy,
        sample_seed=sample_seed,
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Skill evaluator")
    parser.add_argument("--dataset", default="val", choices=["test", "val"], help="Dataset split")
    parser.add_argument("--max-samples", type=int, default=None, help="Max samples")
    parser.add_argument(
        "--sample-strategy",
        default="sequential",
        choices=["sequential", "random", "stratified"],
        help="Subset selection strategy when max-samples is set.",
    )
    parser.add_argument("--sample-seed", type=int, default=42, help="Random seed for subset selection.")
    parser.add_argument("--version", default="teen_detector_v1", help="Skill version")

    args = parser.parse_args()

    report = run_evaluation(
        skill_version=args.version,
        dataset=args.dataset,
        max_samples=args.max_samples,
        sample_strategy=args.sample_strategy,
        sample_seed=args.sample_seed,
    )

    report_dir = Path(__file__).parent.parent.parent / "data" / "eval_reports"
    report_dir.mkdir(parents=True, exist_ok=True)

    report_file = report_dir / f"eval_{args.version}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(report_file, "w", encoding="utf-8") as handle:
        json.dump(report.to_dict(), handle, ensure_ascii=False, indent=2)

    print(f"\n[OK] Report saved: {report_file}")

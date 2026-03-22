# 模块说明：
# - 旧验收链路脚本，围绕旧 executor 工作流展开。
# - 保留做历史参考，但不在当前主链中。

"""
验收测试套件（对齐 技术文档V1）

目标：在不污染数据目录的前提下，完成数据基座、在线执行体、离线评估链路的门禁测试。
特性：
1. 支持多 seed 重复划分评估
2. 支持 RAG / No-RAG 消融对比
3. 检查 split 隔离、混淆矩阵一致性、结构化输出失败率
4. 检查 train/val 文本 exact overlap 与近重复风险
5. 每轮自动 cleanup（删除 benchmark + retrieval index）

注意：
- 本脚本默认只用 validation 集，不会动 test 集。
- 本脚本会发起真实 API 调用（embedding + chat）。
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import time
from dataclasses import dataclass, asdict
from datetime import datetime
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Dict, List, Tuple

import sys

ROOT_DIR = Path(__file__).parent.parent.absolute()
sys.path.insert(0, str(ROOT_DIR))
FORMAL_SKILL_VERSION = "minor-detection"

from scripts.prepare_data import DataPreparer
from src.config import BENCHMARK_TRAIN_PATH, BENCHMARK_VAL_PATH, BENCHMARK_TEST_PATH, SKILLS_DIR, resolve_skill_markdown_path
from src.executor import ExecutorSkill, analyze_with_memory
from src.evolution.evaluator import SkillEvaluator
from src.evolution.optimizer import run_optimization_cycle
from src.memory import UserMemory
from src.retriever.semantic_retriever import SemanticRetriever
from src.utils.path_utils import normalize_project_paths


@dataclass
class GateResult:
    name: str
    passed: bool
    details: str
    level: str = "info"  # info / warn / fail


@dataclass
class SeedRunResult:
    seed: int
    sample_total: int
    train_size: int
    val_size: int
    test_size: int
    rag_f1: float
    no_rag_f1: float
    rag_minus_no_rag_f1: float
    rag_accuracy: float
    no_rag_accuracy: float
    rag_eval: Dict[str, Any]
    no_rag_eval: Dict[str, Any]
    structured_metrics: Dict[str, float]
    iteration_summary: Dict[str, Any]
    memory_summary: Dict[str, Any]
    gates: List[GateResult]


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    if not path.exists():
        return items
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def _conversation_to_text(conv: List[Dict[str, str]]) -> str:
    parts = []
    for t in conv:
        role = t.get("role", "user")
        content = t.get("content", "")
        parts.append(f"{role}: {content}")
    return "\n".join(parts)


def _normalize_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _collect_keys(items: List[Dict[str, Any]]) -> set[Tuple[str, str]]:
    """用 (source, sample_id) 作为联合主键，避免跨源同名 sample_id 误报。"""
    keys: set[Tuple[str, str]] = set()
    for x in items:
        source = str(x.get("source", ""))
        sample_id = str(x.get("sample_id", ""))
        keys.add((source, sample_id))
    return keys


def _build_eval_snapshot(report) -> Dict[str, Any]:
    """提取一份轻量评估快照，便于写入验收报告做诊断。"""
    m = report.metrics
    return {
        "dataset": report.dataset,
        "total_samples": m.total_samples,
        "accuracy": m.accuracy,
        "precision": m.precision,
        "recall": m.recall,
        "f1_score": m.f1_score,
        "true_positive": m.true_positive,
        "true_negative": m.true_negative,
        "false_positive": m.false_positive,
        "false_negative": m.false_negative,
        "error_count": len(report.errors),
    }


def _check_split_disjoint(train_items: List[Dict[str, Any]], val_items: List[Dict[str, Any]], test_items: List[Dict[str, Any]]) -> GateResult:
    train_keys = _collect_keys(train_items)
    val_keys = _collect_keys(val_items)
    test_keys = _collect_keys(test_items)

    tv = train_keys.intersection(val_keys)
    tt = train_keys.intersection(test_keys)
    vt = val_keys.intersection(test_keys)

    ok = not tv and not tt and not vt
    if ok:
        return GateResult("split_disjoint", True, "train/val/test (source,sample_id) 完全隔离", level="info")

    details = f"交集数量 train∩val={len(tv)}, train∩test={len(tt)}, val∩test={len(vt)}"
    return GateResult("split_disjoint", False, details, level="fail")


def _check_confusion_consistency(report) -> GateResult:
    m = report.metrics
    cm_sum = m.true_positive + m.true_negative + m.false_positive + m.false_negative
    ok = cm_sum == m.total_samples
    details = f"CM={cm_sum}, total={m.total_samples}"
    return GateResult(
        "confusion_matrix_consistency",
        ok,
        details,
        level="info" if ok else "fail",
    )


def _check_rag_gain(rag_report, no_rag_report) -> GateResult:
    """检查 RAG 是否至少没有明显差于无 RAG 基线。"""
    rag_f1 = rag_report.metrics.f1_score
    no_rag_f1 = no_rag_report.metrics.f1_score
    delta = rag_f1 - no_rag_f1
    passed = delta >= 0.0
    details = (
        f"rag_f1={rag_f1:.4f}, no_rag_f1={no_rag_f1:.4f}, "
        f"delta={delta:+.4f}"
    )
    return GateResult(
        "rag_non_regression",
        passed,
        details,
        level="info" if passed else "warn",
    )


def _check_text_overlap(
    train_items: List[Dict[str, Any]],
    val_items: List[Dict[str, Any]],
    near_threshold: float,
    near_pair_cap: int,
) -> List[GateResult]:
    train_texts = [_normalize_text(_conversation_to_text(x.get("conversation", []))) for x in train_items]
    val_texts = [_normalize_text(_conversation_to_text(x.get("conversation", []))) for x in val_items]

    train_set = set(train_texts)
    exact_overlap = sum(1 for t in val_texts if t in train_set)

    exact_gate = GateResult(
        name="exact_overlap_train_val",
        passed=(exact_overlap == 0),
        details=f"val 中与 train 完全相同文本数量: {exact_overlap}",
        level="info" if exact_overlap == 0 else "fail",
    )

    # 近重复检查（抽样，避免 O(N*M) 过慢）
    sampled_train = train_texts[:near_pair_cap]
    sampled_val = val_texts[:near_pair_cap]

    near_hits = 0
    for vt in sampled_val:
        if not vt:
            continue
        best = 0.0
        for tt in sampled_train:
            if not tt:
                continue
            score = SequenceMatcher(None, vt, tt).ratio()
            if score > best:
                best = score
            if best >= near_threshold:
                near_hits += 1
                break

    near_gate = GateResult(
        name="near_duplicate_train_val",
        passed=(near_hits == 0),
        details=(
            f"近重复阈值={near_threshold}, 抽样上限={near_pair_cap}, "
            f"命中数量={near_hits}"
        ),
        level="warn" if near_hits > 0 else "info",
    )

    return [exact_gate, near_gate]


def _build_retriever(max_samples: int | None) -> SemanticRetriever:
    retriever = SemanticRetriever()
    retriever.build_index(str(BENCHMARK_TRAIN_PATH), max_samples=max_samples)
    return retriever


def _run_iteration_gate(
    skill_version: str,
    max_eval: int | None,
    retriever: SemanticRetriever | None,
    baseline_report,
    no_rag_report,
) -> Tuple[GateResult, Dict[str, Any]]:
    """执行一轮离线进化，验证 Evaluator -> Optimizer -> 回滚/采纳链路可用。"""
    result = run_optimization_cycle(
        current_version=skill_version,
        max_samples=max_eval,
        dry_run=False,
        auto_rollback=True,
        min_f1_improvement=0.0,
        retriever=retriever,
        baseline_report=baseline_report,
    )

    summary = {
        "success": bool(result.get("success", False)),
        "message": result.get("message", ""),
        "new_version": result.get("new_version"),
        "rolled_back": bool(result.get("rolled_back", False)),
        "baseline_f1": result.get("baseline_f1"),
        "new_f1": result.get("new_f1"),
        "f1_delta": result.get("f1_delta"),
        "baseline_report_reused": bool(result.get("baseline_report_reused", False)),
        "baseline_eval": _build_eval_snapshot(baseline_report),
        "no_rag_eval": _build_eval_snapshot(no_rag_report),
        "diagnosis": "",
    }

    baseline_error_count = summary["baseline_eval"]["error_count"]
    rag_delta_vs_no_rag = (
        summary["baseline_eval"]["f1_score"] - summary["no_rag_eval"]["f1_score"]
    )
    if summary["message"] == "no errors to optimize":
        if baseline_error_count > 0:
            summary["diagnosis"] = (
                "acceptance 评估存在错误样本，但优化周期内部重评未复现；"
                "这通常意味着 LLM 非确定性或两次评估上下文不一致。"
            )
        else:
            summary["diagnosis"] = "优化周期启动正常，但本轮未发现可优化错误。"
    elif summary["new_version"] is not None:
        summary["diagnosis"] = "优化器产出了新版本，并完成了二次评估。"
    elif summary["rolled_back"]:
        summary["diagnosis"] = "优化器产出了新版本，但验证集表现未提升，已自动回滚。"
    else:
        summary["diagnosis"] = "优化周期执行完成，但未产出新版本。"

    summary["rag_minus_no_rag_f1"] = rag_delta_vs_no_rag

    # 只要流程正常返回，且不是异常失败，就算门禁通过
    failed = ("success" in result and not result.get("success"))
    passed = not failed
    details = (
        f"success={summary['success']}, new_version={summary['new_version']}, "
        f"rolled_back={summary['rolled_back']}, f1_delta={summary['f1_delta']}, "
        f"baseline_report_reused={summary['baseline_report_reused']}"
    )
    gate = GateResult("iteration_cycle_executable", passed, details, level="info" if passed else "fail")
    return gate, summary


def _run_memory_gate(
    seed: int,
    val_items: List[Dict[str, Any]],
) -> Tuple[GateResult, Dict[str, Any]]:
    """验证跨会话记忆：同一 user_id 连续两次分析后 total_sessions 应递增。"""
    if not val_items:
        return (
            GateResult("memory_cross_session_update", False, "val 为空，无法测试 memory", level="fail"),
            {"tested": False},
        )

    conv = val_items[0].get("conversation", [])
    user_id = f"acceptance_user_seed_{seed}"
    db_path = ROOT_DIR / "data" / f"user_memory_acceptance_seed_{seed}.db"

    memory = UserMemory(db_path=str(db_path))

    # 连续两次，模拟跨会话更新
    analyze_with_memory(conversation=conv, user_id=user_id, memory=memory, retriever=None, update_memory=True)
    analyze_with_memory(conversation=conv, user_id=user_id, memory=memory, retriever=None, update_memory=True)

    profile = memory.get_profile(user_id)
    passed = (profile is not None and profile.total_sessions >= 2)
    details = (
        f"profile_exists={profile is not None}, total_sessions="
        f"{profile.total_sessions if profile else 0}"
    )

    summary = {
        "tested": True,
        "db_path": str(db_path),
        "user_id": user_id,
        "profile_exists": profile is not None,
        "total_sessions": profile.total_sessions if profile else 0,
    }

    return GateResult("memory_cross_session_update", passed, details, level="info" if passed else "fail"), summary


def _evaluate(skill_version: str, max_eval: int | None, retriever=None):
    skill_path = resolve_skill_markdown_path(SKILLS_DIR / skill_version)
    executor = ExecutorSkill(skill_path=str(skill_path), inference_temperature=0.2)
    evaluator = SkillEvaluator(executor=executor, skill_version=skill_version, retriever=retriever)
    report = evaluator.evaluate(max_samples=max_eval, verbose=False, use_test_set=False)
    llm_metrics = executor.get_llm_metrics()
    return report, llm_metrics


def _cleanup():
    DataPreparer().cleanup()
    # 清理验收测试 memory 临时库
    for p in (ROOT_DIR / "data").glob("user_memory_acceptance_seed_*.db"):
        if p.exists():
            p.unlink()


def _run_one_seed(
    seed: int,
    quick_n: int,
    skill_version: str,
    max_eval: int | None,
    near_threshold: float,
    near_pair_cap: int,
    structured_fail_rate_threshold: float,
    unrecovered_fail_rate_threshold: float,
) -> SeedRunResult:
    gates: List[GateResult] = []

    preparer = DataPreparer(random_seed=seed)
    prep_result = preparer.run(max_per_source=quick_n, save=True)
    if not prep_result.get("success", False):
        raise RuntimeError(f"prepare_data 失败: {prep_result}")

    train_items = _load_jsonl(BENCHMARK_TRAIN_PATH)
    val_items = _load_jsonl(BENCHMARK_VAL_PATH)
    test_items = _load_jsonl(BENCHMARK_TEST_PATH)

    gates.append(_check_split_disjoint(train_items, val_items, test_items))
    gates.extend(_check_text_overlap(train_items, val_items, near_threshold, near_pair_cap))

    # RAG 评估
    retriever = _build_retriever(max_samples=None)
    rag_report, rag_llm_metrics = _evaluate(skill_version, max_eval=max_eval, retriever=retriever)
    gates.append(_check_confusion_consistency(rag_report))

    # No-RAG 评估（消融）
    no_rag_report, _ = _evaluate(skill_version, max_eval=max_eval, retriever=None)
    gates.append(_check_confusion_consistency(no_rag_report))
    gates.append(_check_rag_gain(rag_report, no_rag_report))

    # 离线进化链路门禁
    iter_gate, iter_summary = _run_iteration_gate(
        skill_version,
        max_eval=max_eval,
        retriever=retriever,
        baseline_report=rag_report,
        no_rag_report=no_rag_report,
    )
    gates.append(iter_gate)

    # Memory 跨会话门禁
    memory_gate, memory_summary = _run_memory_gate(seed, val_items)
    gates.append(memory_gate)

    # 结构化失败率门禁
    fail_rate = float(rag_llm_metrics.get("validation_failure_rate_per_100", 0.0))
    unrecovered_rate = float(rag_llm_metrics.get("unrecovered_failure_rate_per_100", 0.0))

    gates.append(
        GateResult(
            name="structured_validation_failure_rate",
            passed=fail_rate <= structured_fail_rate_threshold,
            details=(
                f"rate={fail_rate:.4f}/100, threshold={structured_fail_rate_threshold:.4f}/100"
            ),
            level="info" if fail_rate <= structured_fail_rate_threshold else "fail",
        )
    )
    gates.append(
        GateResult(
            name="structured_unrecovered_failure_rate",
            passed=unrecovered_rate <= unrecovered_fail_rate_threshold,
            details=(
                f"rate={unrecovered_rate:.4f}/100, threshold={unrecovered_fail_rate_threshold:.4f}/100"
            ),
            level="info" if unrecovered_rate <= unrecovered_fail_rate_threshold else "fail",
        )
    )

    return SeedRunResult(
        seed=seed,
        sample_total=len(train_items) + len(val_items) + len(test_items),
        train_size=len(train_items),
        val_size=len(val_items),
        test_size=len(test_items),
        rag_f1=rag_report.metrics.f1_score,
        no_rag_f1=no_rag_report.metrics.f1_score,
        rag_minus_no_rag_f1=rag_report.metrics.f1_score - no_rag_report.metrics.f1_score,
        rag_accuracy=rag_report.metrics.accuracy,
        no_rag_accuracy=no_rag_report.metrics.accuracy,
        rag_eval=_build_eval_snapshot(rag_report),
        no_rag_eval=_build_eval_snapshot(no_rag_report),
        structured_metrics=rag_llm_metrics,
        iteration_summary=iter_summary,
        memory_summary=memory_summary,
        gates=gates,
    )


def main():
    parser = argparse.ArgumentParser(description="技术文档V1 验收测试套件")
    parser.add_argument("--quick-n", type=int, default=50, help="每源采样数量 (default: 50)")
    parser.add_argument("--seeds", type=str, default="42,43,44", help="多个随机种子，逗号分隔")
    parser.add_argument("--skill-version", type=str, default="teen_detector_v1")
    parser.add_argument(
        "--formal-skill",
        action="store_true",
        help=f"直接使用正式 Skill 包 {FORMAL_SKILL_VERSION} 作为验收目标。",
    )
    parser.add_argument("--max-eval", type=int, default=None, help="评估时最大样本数")
    parser.add_argument("--near-threshold", type=float, default=0.96, help="近重复阈值")
    parser.add_argument("--near-pair-cap", type=int, default=300, help="近重复抽样上限")
    parser.add_argument(
        "--structured-fail-rate-threshold",
        type=float,
        default=5.0,
        help="结构化校验失败率门限（每100次）",
    )
    parser.add_argument(
        "--unrecovered-fail-rate-threshold",
        type=float,
        default=0.5,
        help="结构化不可恢复失败率门限（每100次）",
    )
    parser.add_argument("--keep-artifacts", action="store_true", help="保留 benchmark/index 产物")
    parser.add_argument(
        "--cleanup-on-failure",
        action="store_true",
        help="发生异常时也执行 cleanup（默认保留现场以便排查）",
    )

    args = parser.parse_args()

    if args.formal_skill:
        if args.skill_version != "teen_detector_v1" and args.skill_version != FORMAL_SKILL_VERSION:
            raise ValueError(
                f"--formal-skill cannot be combined with --skill-version={args.skill_version}; "
                f"use --skill-version {FORMAL_SKILL_VERSION} or omit --skill-version."
            )
        args.skill_version = FORMAL_SKILL_VERSION

    seeds = [int(x.strip()) for x in args.seeds.split(",") if x.strip()]
    if not seeds:
        raise ValueError("--seeds 不能为空")

    started = time.time()
    all_runs: List[SeedRunResult] = []

    print("=" * 72)
    print("Acceptance Suite Started")
    print("=" * 72)
    print(f"Seeds: {seeds}")
    print(f"Quick-N per source: {args.quick_n}")

    for seed in seeds:
        print("\n" + "-" * 72)
        print(f"Running seed={seed}")
        print("-" * 72)

        seed_succeeded = False
        try:
            result = _run_one_seed(
                seed=seed,
                quick_n=args.quick_n,
                skill_version=args.skill_version,
                max_eval=args.max_eval,
                near_threshold=args.near_threshold,
                near_pair_cap=args.near_pair_cap,
                structured_fail_rate_threshold=args.structured_fail_rate_threshold,
                unrecovered_fail_rate_threshold=args.unrecovered_fail_rate_threshold,
            )
            all_runs.append(result)

            print(
                f"seed={seed} | rag_f1={result.rag_f1:.4f} | "
                f"no_rag_f1={result.no_rag_f1:.4f} | "
                f"structured_fail_rate/100={result.structured_metrics.get('validation_failure_rate_per_100', 0):.4f}"
            )

            failed = [g for g in result.gates if not g.passed and g.level != "warn"]
            warnings = [g for g in result.gates if (not g.passed and g.level == "warn")]
            if failed:
                print(f"  FAIL gates: {[g.name for g in failed]}")
            if warnings:
                print(f"  WARN gates: {[g.name for g in warnings]}")

            seed_succeeded = True

        finally:
            should_cleanup = (not args.keep_artifacts) and (seed_succeeded or args.cleanup_on_failure)
            if should_cleanup:
                _cleanup()

    # 汇总
    rag_f1s = [r.rag_f1 for r in all_runs]
    no_rag_f1s = [r.no_rag_f1 for r in all_runs]

    gate_failures: List[Tuple[int, GateResult]] = []
    gate_warnings: List[Tuple[int, GateResult]] = []
    for r in all_runs:
        for g in r.gates:
            if not g.passed and g.level == "warn":
                gate_warnings.append((r.seed, g))
            elif not g.passed:
                gate_failures.append((r.seed, g))

    summary = {
        "timestamp": datetime.now().isoformat(),
        "seeds": seeds,
        "quick_n": args.quick_n,
        "runs": [
            {
                **asdict(r),
                "gates": [asdict(g) for g in r.gates],
            }
            for r in all_runs
        ],
        "aggregate": {
            "rag_f1_mean": statistics.mean(rag_f1s) if rag_f1s else 0.0,
            "rag_f1_std": statistics.pstdev(rag_f1s) if len(rag_f1s) > 1 else 0.0,
            "no_rag_f1_mean": statistics.mean(no_rag_f1s) if no_rag_f1s else 0.0,
            "no_rag_f1_std": statistics.pstdev(no_rag_f1s) if len(no_rag_f1s) > 1 else 0.0,
            "rag_minus_no_rag_f1_mean": (
                statistics.mean([r.rag_minus_no_rag_f1 for r in all_runs]) if all_runs else 0.0
            ),
            "rag_minus_no_rag_f1_std": (
                statistics.pstdev([r.rag_minus_no_rag_f1 for r in all_runs]) if len(all_runs) > 1 else 0.0
            ),
            "elapsed_seconds": round(time.time() - started, 2),
            "gate_fail_count": len(gate_failures),
            "gate_warn_count": len(gate_warnings),
        },
    }

    report_dir = ROOT_DIR / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / f"acceptance_suite_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    serializable_summary = normalize_project_paths(summary, project_root=ROOT_DIR, start=report_path.parent)
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(serializable_summary, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 72)
    print("Acceptance Suite Summary")
    print("=" * 72)
    print(f"RAG F1 mean/std: {summary['aggregate']['rag_f1_mean']:.4f}/{summary['aggregate']['rag_f1_std']:.4f}")
    print(
        f"No-RAG F1 mean/std: {summary['aggregate']['no_rag_f1_mean']:.4f}/"
        f"{summary['aggregate']['no_rag_f1_std']:.4f}"
    )
    print(
        f"RAG minus No-RAG F1 mean/std: "
        f"{summary['aggregate']['rag_minus_no_rag_f1_mean']:+.4f}/"
        f"{summary['aggregate']['rag_minus_no_rag_f1_std']:.4f}"
    )
    print(f"Gate FAIL count: {summary['aggregate']['gate_fail_count']}")
    print(f"Gate WARN count: {summary['aggregate']['gate_warn_count']}")
    print(f"Report saved: {report_path}")

    if gate_failures:
        print("\nFailed gates:")
        for seed, gate in gate_failures:
            print(f"- seed={seed} | {gate.name} | {gate.details}")
        raise SystemExit(1)

    if gate_warnings:
        print("\nWarnings:")
        for seed, gate in gate_warnings:
            print(f"- seed={seed} | {gate.name} | {gate.details}")

    print("\nAll required gates passed.")


if __name__ == "__main__":
    main()

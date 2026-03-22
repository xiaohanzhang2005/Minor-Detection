# 模块说明：
# - 旧版离线总控脚本。
# - 属于历史路线，不在当前 A 或 B 主链中。

"""
End-to-end offline pipeline for:
1. reusing or rebuilding benchmark assets
2. reusing or rebuilding the RAG index
3. baseline evaluation
4. optional multi-round skill evolution with early stopping
5. optional final test-set evaluation
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT_DIR = Path(__file__).parent.parent.absolute()
sys.path.insert(0, str(ROOT_DIR))

from src.utils.path_utils import normalize_project_paths, to_relative_posix_path

BENCHMARK_MANIFEST_PATH = ROOT_DIR / "data" / "benchmark" / "manifest.json"
FORMAL_SKILL_VERSION = "minor-detection"


def _save_run_report(payload: Dict[str, Any]) -> Path:
    report_dir = ROOT_DIR / "reports" / "pipeline_runs"
    report_dir.mkdir(parents=True, exist_ok=True)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = report_dir / f"pipeline_run_{ts}.json"
    serializable_payload = normalize_project_paths(payload, project_root=ROOT_DIR, start=report_path.parent)
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(serializable_payload, f, ensure_ascii=False, indent=2)
    return report_path


def _save_benchmark_manifest(payload: Dict[str, Any]) -> Path:
    BENCHMARK_MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    serializable_payload = normalize_project_paths(
        payload,
        project_root=ROOT_DIR,
        start=BENCHMARK_MANIFEST_PATH.parent,
    )
    with open(BENCHMARK_MANIFEST_PATH, "w", encoding="utf-8") as f:
        json.dump(serializable_payload, f, ensure_ascii=False, indent=2)
    return BENCHMARK_MANIFEST_PATH


def _load_json_file(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _load_benchmark_manifest() -> Optional[Dict[str, Any]]:
    return _load_json_file(BENCHMARK_MANIFEST_PATH)


def _summarize_benchmark_files() -> Dict[str, Any]:
    from scripts.prepare_data import DataPreparer
    from src.config import BENCHMARK_TEST_PATH, BENCHMARK_TRAIN_PATH, BENCHMARK_VAL_PATH

    split_paths = {
        "train": BENCHMARK_TRAIN_PATH,
        "val": BENCHMARK_VAL_PATH,
        "test": BENCHMARK_TEST_PATH,
    }
    split_counts: Dict[str, int] = {}
    by_source: Dict[str, int] = {}
    by_is_minor = {"minor": 0, "adult": 0}
    by_age_bucket: Dict[str, int] = {}

    for split_name, path in split_paths.items():
        if not path.exists():
            raise FileNotFoundError(f"Missing benchmark split: {path}")

        count = 0
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                sample = json.loads(line)
                count += 1

                source = sample.get("source", "unknown")
                by_source[source] = by_source.get(source, 0) + 1

                if sample.get("is_minor", False):
                    by_is_minor["minor"] += 1
                else:
                    by_is_minor["adult"] += 1

                age_bucket = DataPreparer.resolve_age_bucket_from_record(sample)
                by_age_bucket[age_bucket] = by_age_bucket.get(age_bucket, 0) + 1

        split_counts[split_name] = count

    return {
        "statistics": {
            "total": sum(split_counts.values()),
            "by_source": dict(sorted(by_source.items())),
            "by_is_minor": by_is_minor,
            "by_age_bucket": dict(sorted(by_age_bucket.items())),
        },
        "splits": split_counts,
        "paths": {
            name: to_relative_posix_path(path, BENCHMARK_MANIFEST_PATH.parent)
            for name, path in split_paths.items()
        },
    }


def repair_benchmark_manifest(
    requested_mode: str,
    max_per_source: Optional[int],
    seed: Optional[int],
) -> Dict[str, Any]:
    from scripts.prepare_data import ADULT_AGE_BUCKETS, MINOR_AGE_BUCKETS, STRATIFICATION_STRATEGY

    print("\n" + "=" * 60)
    print("[STEP 0] Repair benchmark manifest")
    print("=" * 60)

    if not _benchmark_files_exist():
        return {
            "success": False,
            "reason": "benchmark files are missing; cannot repair manifest without train/val/test",
            "error_type": "missing_assets",
        }

    summary = _summarize_benchmark_files()
    payload = {
        "timestamp": datetime.now().isoformat(),
        "mode": requested_mode,
        "quick_n": max_per_source,
        "seed": seed,
        "stratification": {
            "strategy": STRATIFICATION_STRATEGY,
            "unit": "source+age_bucket",
            "minor_age_buckets": [
                {"name": name, "min_age": lower, "max_age": upper}
                for name, lower, upper in MINOR_AGE_BUCKETS
            ],
            "adult_age_buckets": [
                {"name": name, "min_age": lower, "max_age": upper}
                for name, lower, upper in ADULT_AGE_BUCKETS
            ],
        },
        "statistics": summary["statistics"],
        "splits": summary["splits"],
        "paths": summary["paths"],
        "provenance": "repaired_from_existing_splits",
    }
    manifest_path = _save_benchmark_manifest(payload)
    print(
        f"[OK] Repaired benchmark manifest: {manifest_path} "
        f"(mode={requested_mode}, seed={seed}, quick_n={max_per_source})"
    )
    return {
        "success": True,
        "manifest": payload,
        "manifest_path": to_relative_posix_path(manifest_path, ROOT_DIR),
    }


def _build_eval_snapshot(report) -> Dict[str, Any]:
    metrics = report.metrics
    return {
        "dataset": report.dataset,
        "sampling": getattr(report, "sampling", {}),
        "total_samples": metrics.total_samples,
        "accuracy": metrics.accuracy,
        "precision": metrics.precision,
        "recall": metrics.recall,
        "f1_score": metrics.f1_score,
        "true_positive": metrics.true_positive,
        "true_negative": metrics.true_negative,
        "false_positive": metrics.false_positive,
        "false_negative": metrics.false_negative,
        "error_count": len(report.errors),
    }


def _benchmark_files_exist() -> bool:
    from src.config import BENCHMARK_TEST_PATH, BENCHMARK_TRAIN_PATH, BENCHMARK_VAL_PATH

    return all(path.exists() for path in [BENCHMARK_TRAIN_PATH, BENCHMARK_VAL_PATH, BENCHMARK_TEST_PATH])


def _rag_assets_exist() -> bool:
    from src.config import RETRIEVAL_CORPUS_DIR, RETRIEVAL_DB_DIR

    return all(
        [
            (RETRIEVAL_DB_DIR / "index.pkl").exists(),
            (RETRIEVAL_DB_DIR / "manifest.json").exists(),
            (RETRIEVAL_CORPUS_DIR / "cases_v1.jsonl").exists(),
        ]
    )


def _load_rag_manifest() -> Optional[Dict[str, Any]]:
    from src.config import RETRIEVAL_DB_DIR

    return _load_json_file(RETRIEVAL_DB_DIR / "manifest.json")


def _load_retriever():
    from src.config import RETRIEVAL_DB_DIR

    index_path = RETRIEVAL_DB_DIR / "index.pkl"
    if not index_path.exists():
        return None

    try:
        from src.retriever.semantic_retriever import SemanticRetriever

        return SemanticRetriever()
    except Exception as exc:
        print(f"[WARN] Failed to load RAG retriever: {exc}")
        return None


def _resolve_requested_mode(args: argparse.Namespace) -> str:
    if args.quick:
        return "quick"
    if args.full:
        return "full"
    return "full"


def _resolve_test_rag_mode(test_rag_mode: str, evolution_rag: str) -> str:
    if test_rag_mode == "match-evolution":
        return evolution_rag
    return test_rag_mode


def _resolve_test_memory_mode(test_memory_mode: str, evolution_memory: str) -> str:
    if test_memory_mode == "match-evolution":
        return evolution_memory
    return test_memory_mode


def _argv_has_option(argv: Optional[List[str]], option_name: str) -> bool:
    if not argv:
        return False
    return any(token == option_name or token.startswith(f"{option_name}=") for token in argv)


def _resolve_skill_selection(args: argparse.Namespace, argv: Optional[List[str]]) -> argparse.Namespace:
    effective_argv = list(argv) if argv is not None else sys.argv[1:]
    explicit_skill_version = _argv_has_option(effective_argv, "--skill-version")

    if args.formal_skill:
        if explicit_skill_version and args.skill_version != FORMAL_SKILL_VERSION:
            raise ValueError(
                f"--formal-skill cannot be combined with --skill-version={args.skill_version}; "
                f"use --skill-version {FORMAL_SKILL_VERSION} or omit --skill-version."
            )
        args.skill_version = FORMAL_SKILL_VERSION
        args.skill_mode = "formal"
        return args

    args.skill_mode = "custom" if explicit_skill_version else "active"
    return args


def _validate_pipeline_config(
    baseline_rag_mode: str,
    evolution_rag: str,
    baseline_memory_mode: str,
    evolution_memory: str,
) -> None:
    if evolution_rag == "on" and baseline_rag_mode == "off":
        raise ValueError("baseline-rag-mode=off cannot feed evolution-rag=on; use --baseline-rag-mode on|compare")
    if evolution_rag == "off" and baseline_rag_mode == "on":
        raise ValueError("baseline-rag-mode=on cannot feed evolution-rag=off; use --baseline-rag-mode off|compare")
    if evolution_memory == "on" and baseline_memory_mode == "off":
        raise ValueError(
            "baseline-memory-mode=off cannot feed evolution-memory=on; use --baseline-memory-mode on|compare"
        )
    if evolution_memory == "off" and baseline_memory_mode == "on":
        raise ValueError(
            "baseline-memory-mode=on cannot feed evolution-memory=off; use --baseline-memory-mode off|compare"
        )


def _needs_rag_assets(
    baseline_rag_mode: str,
    evolution_rag: str,
    run_test_final: bool,
    test_rag_mode: str,
) -> bool:
    resolved_test_mode = _resolve_test_rag_mode(test_rag_mode, evolution_rag)
    return any(
        [
            baseline_rag_mode in {"on", "compare"},
            evolution_rag == "on",
            run_test_final and resolved_test_mode in {"on", "compare"},
        ]
    )


def _validate_asset_mode(
    requested_mode: str,
    max_per_source: Optional[int],
    manifest: Optional[Dict[str, Any]],
) -> tuple[bool, Optional[str], bool]:
    if manifest is None:
        return True, None, False

    manifest_mode = manifest.get("mode")
    manifest_quick_n = manifest.get("quick_n")
    manifest_strategy = ((manifest.get("stratification") or {}).get("strategy"))
    if manifest_mode != requested_mode:
        return False, f"manifest mode mismatch: have={manifest_mode}, need={requested_mode}", True
    if requested_mode == "quick" and manifest_quick_n != max_per_source:
        return False, f"manifest quick_n mismatch: have={manifest_quick_n}, need={max_per_source}", True
    if manifest_strategy != "source_age_bucket_v1":
        return (
            False,
            f"manifest stratification mismatch: have={manifest_strategy}, need=source_age_bucket_v1",
            True,
        )
    return True, None, True


def _cleanup_pipeline_artifacts(
    cleanup_assets: bool,
    cleanup_generated_skills: bool,
    created_versions: List[str],
    restore_version: Optional[str],
) -> Dict[str, Any]:
    from scripts.prepare_data import DataPreparer
    from src.config import DATA_DIR, SKILLS_DIR, set_active_skill_version

    summary: Dict[str, Any] = {
        "performed": False,
        "removed_benchmark": [],
        "removed_generated_skills": [],
        "restored_active_version": None,
        "errors": [],
    }

    if not cleanup_assets and not cleanup_generated_skills:
        summary["skipped"] = True
        return summary

    summary["performed"] = True

    if cleanup_assets:
        try:
            preparer = DataPreparer()
            for name in ["train.jsonl", "val.jsonl", "test.jsonl"]:
                path = preparer.output_dir / name
                if path.exists():
                    summary["removed_benchmark"].append(str(path))
                    path.unlink()
            for artifact in [
                BENCHMARK_MANIFEST_PATH,
                DATA_DIR / "retrieval_db" / "index.pkl",
                DATA_DIR / "retrieval_db" / "manifest.json",
                DATA_DIR / "retrieval_corpus" / "cases_v1.jsonl",
            ]:
                if artifact.exists():
                    summary["removed_benchmark"].append(str(artifact))
                    artifact.unlink()
        except Exception as exc:
            summary["errors"].append(f"data_cleanup_failed: {exc}")

        for db_path in DATA_DIR.glob("user_memory_acceptance_seed_*.db"):
            try:
                db_path.unlink()
            except Exception as exc:
                summary["errors"].append(f"memory_cleanup_failed:{db_path}:{exc}")

    if cleanup_generated_skills:
        if restore_version:
            try:
                set_active_skill_version(restore_version)
                summary["restored_active_version"] = restore_version
            except Exception as exc:
                summary["errors"].append(f"restore_active_version_failed: {exc}")

        for version in created_versions:
            version_dir = SKILLS_DIR / version
            if not version_dir.exists():
                continue
            try:
                import shutil

                shutil.rmtree(version_dir)
                summary["removed_generated_skills"].append(version)
            except Exception as exc:
                summary["errors"].append(f"remove_generated_skill_failed:{version}:{exc}")

    return summary


def step_prepare_data(max_per_source: Optional[int], seed: int = 42) -> Dict[str, Any]:
    from scripts.prepare_data import DataPreparer

    print("\n" + "=" * 60)
    print("[STEP 1] Prepare benchmark data")
    print("=" * 60)

    preparer = DataPreparer(random_seed=seed)
    result = preparer.run(max_per_source=max_per_source, save=True)
    if result.get("success", False):
        manifest = result.get("manifest")
        if manifest:
            _save_benchmark_manifest(manifest)
        result["manifest_path"] = to_relative_posix_path(BENCHMARK_MANIFEST_PATH, ROOT_DIR)
    return result


def step_build_rag_index(max_samples: Optional[int]) -> Dict[str, Any]:
    from src.config import BENCHMARK_TRAIN_PATH, RETRIEVAL_CORPUS_DIR, RETRIEVAL_DB_DIR
    from src.retriever.semantic_retriever import SemanticRetriever

    print("\n" + "=" * 60)
    print("[STEP 2] Build RAG index")
    print("=" * 60)

    if not BENCHMARK_TRAIN_PATH.exists():
        return {
            "success": False,
            "reason": f"missing train set: {BENCHMARK_TRAIN_PATH}",
            "error_type": "missing_assets",
        }

    retriever = SemanticRetriever()
    indexed = retriever.build_index(str(BENCHMARK_TRAIN_PATH), max_samples=max_samples)
    return {
        "success": indexed > 0,
        "indexed_samples": indexed,
        "index_path": to_relative_posix_path(RETRIEVAL_DB_DIR / "index.pkl", ROOT_DIR),
        "manifest_path": to_relative_posix_path(RETRIEVAL_DB_DIR / "manifest.json", ROOT_DIR),
        "corpus_path": to_relative_posix_path(RETRIEVAL_CORPUS_DIR / "cases_v1.jsonl", ROOT_DIR),
        "manifest": _load_rag_manifest(),
    }


def ensure_benchmark_data(
    max_per_source: Optional[int],
    seed: int,
    rebuild: bool,
    requested_mode: str,
) -> Dict[str, Any]:
    manifest = _load_benchmark_manifest()
    compatible, reason, verified = _validate_asset_mode(requested_mode, max_per_source, manifest)

    if rebuild:
        if reason:
            print(f"[INFO] Benchmark rebuild reason: {reason}")
        else:
            print("[INFO] Benchmark rebuild reason: explicit rebuild")
        result = step_prepare_data(max_per_source=max_per_source, seed=seed)
        result["reused"] = False
        result["verified"] = True
        result["rebuild_reason"] = reason or "explicit rebuild"
        return result

    if not _benchmark_files_exist():
        return {
            "success": False,
            "reused": False,
            "verified": verified,
            "reason": "missing benchmark files; rerun with --rebuild-data",
            "error_type": "missing_assets",
        }

    if not compatible:
        return {
            "success": False,
            "reused": True,
            "verified": verified,
            "reason": f"{reason}; rerun with --rebuild-data",
            "error_type": "asset_mode_mismatch",
            "manifest": manifest,
            "manifest_path": (
                to_relative_posix_path(BENCHMARK_MANIFEST_PATH, ROOT_DIR)
                if BENCHMARK_MANIFEST_PATH.exists()
                else None
            ),
        }

    print("\n" + "=" * 60)
    print("[STEP 1] Reuse benchmark data")
    print("=" * 60)
    if manifest is None:
        details = "benchmark files exist but manifest is missing; reusing unverified split"
        print(f"[WARN] {details}")
        return {
            "success": True,
            "reused": True,
            "verified": False,
            "details": details,
            "manifest": None,
            "manifest_path": None,
        }

    print(
        f"[INFO] Reusing benchmark split "
        f"(mode={manifest.get('mode')}, seed={manifest.get('seed')}, quick_n={manifest.get('quick_n')})"
    )
    return {
        "success": True,
        "reused": True,
        "verified": True,
        "manifest": manifest,
        "manifest_path": to_relative_posix_path(BENCHMARK_MANIFEST_PATH, ROOT_DIR),
    }


def ensure_rag_index(max_samples: Optional[int], rebuild: bool) -> Dict[str, Any]:
    from src.config import RETRIEVAL_DB_DIR

    if rebuild:
        print("[INFO] RAG rebuild reason: explicit rebuild")
        result = step_build_rag_index(max_samples=max_samples)
        result["reused"] = False
        return result

    if not _rag_assets_exist():
        return {
            "success": False,
            "reused": False,
            "reason": "missing RAG index; rerun with --rebuild-index",
            "error_type": "missing_assets",
        }

    print("\n" + "=" * 60)
    print("[STEP 2] Reuse RAG index")
    print("=" * 60)
    return {
        "success": True,
        "reused": True,
        "index_path": to_relative_posix_path(RETRIEVAL_DB_DIR / "index.pkl", ROOT_DIR),
        "manifest_path": to_relative_posix_path(RETRIEVAL_DB_DIR / "manifest.json", ROOT_DIR),
        "manifest": _load_rag_manifest(),
    }


def step_evaluate(
    skill_version: str,
    max_samples: Optional[int],
    retriever=None,
    dataset: str = "val",
    use_memory: bool = False,
    sample_strategy: str = "stratified",
    sample_seed: int = 42,
) -> Optional[Dict[str, Any]]:
    from src.config import BENCHMARK_TEST_PATH, BENCHMARK_VAL_PATH, SKILLS_DIR, resolve_skill_markdown_path
    from src.executor import ExecutorSkill
    from src.evolution.evaluator import SkillEvaluator

    print("\n" + "=" * 60)
    print(f"[STEP 3] Evaluate {skill_version} ({dataset})")
    print("=" * 60)

    try:
        skill_path = resolve_skill_markdown_path(SKILLS_DIR / skill_version)
    except FileNotFoundError:
        print(f"[ERROR] Skill not found: {SKILLS_DIR / skill_version}")
        return None

    dataset_path = str(BENCHMARK_TEST_PATH if dataset == "test" else BENCHMARK_VAL_PATH)
    executor = ExecutorSkill(skill_path=str(skill_path))
    evaluator = SkillEvaluator(
        executor=executor,
        skill_version=skill_version,
        retriever=retriever,
        use_memory=use_memory,
    )
    report = evaluator.evaluate(
        dataset_path=dataset_path,
        max_samples=max_samples,
        sample_strategy=sample_strategy,
        sample_seed=sample_seed,
        use_test_set=(dataset == "test"),
    )
    return {
        "report": report,
        "snapshot": _build_eval_snapshot(report),
        "f1": report.metrics.f1_score,
        "accuracy": report.metrics.accuracy,
    }


def evaluate_rag_mode(
    skill_version: str,
    max_samples: Optional[int],
    rag_mode: str,
    mainline_mode: str,
    dataset: str,
    retriever=None,
    memory_mode: str = "off",
    mainline_memory: str = "off",
    sample_strategy: str = "stratified",
    sample_seed: int = 42,
) -> Dict[str, Any]:
    if rag_mode not in {"on", "off", "compare"}:
        raise ValueError(f"Unsupported rag_mode: {rag_mode}")
    if mainline_mode not in {"on", "off"}:
        raise ValueError(f"Unsupported mainline_mode: {mainline_mode}")
    if memory_mode not in {"on", "off", "compare"}:
        raise ValueError(f"Unsupported memory_mode: {memory_mode}")
    if mainline_memory not in {"on", "off"}:
        raise ValueError(f"Unsupported mainline_memory: {mainline_memory}")

    rag_options = [True] if rag_mode == "on" else [False] if rag_mode == "off" else [True, False]
    memory_options = [True] if memory_mode == "on" else [False] if memory_mode == "off" else [True, False]

    evaluations: Dict[tuple[bool, bool], Dict[str, Any]] = {}
    snapshot_map: Dict[str, Dict[str, Any]] = {}
    for rag_enabled in rag_options:
        if rag_enabled and retriever is None:
            raise RuntimeError("RAG evaluation requested but no retriever is available")

        for memory_enabled in memory_options:
            evaluation = step_evaluate(
                skill_version,
                max_samples=max_samples,
                retriever=retriever if rag_enabled else None,
                dataset=dataset,
                use_memory=memory_enabled,
                sample_strategy=sample_strategy,
                sample_seed=sample_seed,
            )
            label = f"rag_{'on' if rag_enabled else 'off'}_memory_{'on' if memory_enabled else 'off'}"
            evaluations[(rag_enabled, memory_enabled)] = evaluation
            snapshot_map[label] = evaluation["snapshot"]

    selected_key = (mainline_mode == "on", mainline_memory == "on")
    selected = evaluations.get(selected_key)
    if selected is None:
        raise RuntimeError(
            f"Selected mainline rag={mainline_mode}, memory={mainline_memory} was not evaluated"
        )

    payload: Dict[str, Any] = {
        "selected_mainline": {"rag": mainline_mode, "memory": mainline_memory},
        "selected_eval": selected["snapshot"],
        "selected_f1": selected["f1"],
        "selected_accuracy": selected["accuracy"],
        "selected_report": selected["report"],
        "evaluations": snapshot_map,
    }

    if memory_mode == "off":
        rag_eval = evaluations.get((True, False))
        no_rag_eval = evaluations.get((False, False))
        if rag_eval is not None:
            payload["rag_eval"] = rag_eval["snapshot"]
            payload["rag_report"] = rag_eval["report"]
        if no_rag_eval is not None:
            payload["no_rag_eval"] = no_rag_eval["snapshot"]
            payload["no_rag_report"] = no_rag_eval["report"]
        if rag_eval is not None and no_rag_eval is not None:
            payload["rag_minus_no_rag_f1"] = rag_eval["f1"] - no_rag_eval["f1"]

    if rag_mode == "off":
        memory_eval = evaluations.get((False, True))
        no_memory_eval = evaluations.get((False, False))
        if memory_eval is not None:
            payload["memory_eval"] = memory_eval["snapshot"]
            payload["memory_report"] = memory_eval["report"]
        if no_memory_eval is not None:
            payload["no_memory_eval"] = no_memory_eval["snapshot"]
            payload["no_memory_report"] = no_memory_eval["report"]
        if memory_eval is not None and no_memory_eval is not None:
            payload["memory_minus_no_memory_f1"] = memory_eval["f1"] - no_memory_eval["f1"]

    return payload


def step_evolution(
    current_version: str,
    max_rounds: int,
    max_eval_samples: Optional[int],
    retriever=None,
    use_memory: bool = False,
    baseline_report=None,
    patience: int = 2,
    min_improvement: float = 0.001,
    optimization_max_errors: Optional[int] = None,
    sample_strategy: str = "stratified",
    sample_seed: int = 42,
) -> Dict[str, Any]:
    from src.config import SKILLS_DIR
    from src.evolution.optimizer import SkillOptimizer, run_optimization_cycle

    if baseline_report is None:
        raise ValueError("baseline_report is required for evolution")

    initial_version = current_version
    initial_skill_dir = SKILLS_DIR / initial_version
    formal_review_required = (initial_skill_dir / "references" / "output-schema.md").exists()

    if max_rounds <= 0:
        return {
            "final_version": current_version,
            "final_report": baseline_report,
            "final_snapshot": _build_eval_snapshot(baseline_report),
            "round_history": [],
            "created_versions": [],
            "accepted_versions": [],
            "rollback_count": 0,
            "best_version": current_version,
            "best_f1": baseline_report.metrics.f1_score,
            "stop_reason": "manual_zero_rounds",
            "review_required": False,
            "review_status": "not_required",
            "review_artifact": None,
            "adopted_version": current_version,
        }

    print("\n" + "=" * 60)
    print(f"[STEP 4] Run evolution (max_rounds={max_rounds}, patience={patience})")
    print("=" * 60)

    version = current_version
    current_report = baseline_report
    best_version = current_version
    best_report = baseline_report
    best_f1 = baseline_report.metrics.f1_score
    no_improvement_streak = 0

    history: List[Dict[str, Any]] = []
    created_versions: List[str] = []
    accepted_versions: List[str] = []
    rollback_rounds = 0
    stop_reason = "max_rounds_reached"

    for round_index in range(1, max_rounds + 1):
        print("\n" + "-" * 60)
        print(f"[ROUND {round_index}] current_version={version}")
        print("-" * 60)

        result = run_optimization_cycle(
            current_version=version,
            max_samples=max_eval_samples,
            optimization_max_errors=optimization_max_errors,
            sample_strategy=sample_strategy,
            sample_seed=sample_seed,
            dry_run=False,
            auto_rollback=True,
            min_f1_improvement=min_improvement,
            retriever=retriever,
            use_memory=use_memory,
            baseline_report=current_report,
            activate_accepted_version=not formal_review_required,
        )

        round_info: Dict[str, Any] = {
            "round": round_index,
            "version_before": version,
            "success": bool(result.get("success", False)),
            "message": result.get("message", ""),
            "baseline_f1": result.get("baseline_f1", current_report.metrics.f1_score),
            "new_f1": result.get("new_f1"),
            "f1_delta": result.get("f1_delta"),
            "new_version": result.get("new_version"),
            "rolled_back": bool(result.get("rolled_back", False)),
            "baseline_report_reused": bool(result.get("baseline_report_reused", False)),
            "accepted": False,
            "version_after": version,
            "active_version_after": version,
            "baseline_eval": _build_eval_snapshot(current_report),
            "new_eval": _build_eval_snapshot(result["_new_report"]) if result.get("_new_report") is not None else None,
            "no_improvement_streak_before": no_improvement_streak,
            "review_required": bool(result.get("review_required", False)),
            "adoption_status": result.get("adoption_status", "accepted"),
        }

        if result.get("new_version"):
            created_versions.append(result["new_version"])

        if "success" in result and not result.get("success", False):
            raise RuntimeError(f"optimization round {round_index} failed: {result}")

        if result.get("rolled_back"):
            rollback_rounds += 1
            no_improvement_streak += 1
            round_info["adoption_status"] = "rolled_back"
            round_info["accepted"] = False
            round_info["version_after"] = version
            round_info["active_version_after"] = version
            round_info["diagnosis"] = "new version was evaluated and rolled back"
            print(f"[ROLLBACK] {result.get('rollback_reason', 'unknown reason')}")
        elif result.get("new_version"):
            version = result["new_version"]
            current_report = result.get("_new_report")
            accepted_versions.append(version)
            no_improvement_streak = 0
            round_info["accepted"] = True
            round_info["version_after"] = version
            round_info["active_version_after"] = result.get("active_version", version)
            round_info["diagnosis"] = "new version was accepted"
            if current_report is not None and current_report.metrics.f1_score >= best_f1:
                best_version = version
                best_report = current_report
                best_f1 = current_report.metrics.f1_score
            print(
                f"[ACCEPT] {version} | "
                f"F1 {result.get('baseline_f1', 0.0):.4f} -> {result.get('new_f1', 0.0):.4f}"
            )
        else:
            no_improvement_streak += 1
            round_info["adoption_status"] = result.get("adoption_status", "skipped")
            round_info["diagnosis"] = result.get("message") or "no new version generated"
            print(f"[SKIP] {round_info['diagnosis']}")
            if round_index == 1 and result.get("message") == "no errors to optimize":
                stop_reason = "no_errors_initially"

        round_info["no_improvement_streak_after"] = no_improvement_streak
        history.append(round_info)

        if stop_reason == "no_errors_initially":
            break

        if no_improvement_streak >= patience:
            stop_reason = "patience_exhausted"
            break

    final_report = best_report
    review_artifact = None
    review_status = "not_required"
    adopted_version = best_version

    if formal_review_required and best_version != initial_version:
        review_artifact = SkillOptimizer().create_formal_skill_review_artifact(
            base_version=initial_version,
            candidate_version=best_version,
        )
        review_status = "pending"
        adopted_version = initial_version

    return {
        "final_version": best_version,
        "final_report": final_report,
        "final_snapshot": _build_eval_snapshot(final_report),
        "round_history": history,
        "created_versions": created_versions,
        "accepted_versions": accepted_versions,
        "rollback_count": rollback_rounds,
        "best_version": best_version,
        "best_f1": best_f1,
        "stop_reason": stop_reason,
        "review_required": review_artifact is not None,
        "review_status": review_status,
        "review_artifact": review_artifact,
        "adopted_version": adopted_version,
    }


def _trend_label(start_f1: Optional[float], end_f1: Optional[float]) -> str:
    if start_f1 is None or end_f1 is None:
        return "unknown"
    delta = end_f1 - start_f1
    if delta > 0:
        return "improved"
    if delta < 0:
        return "regressed"
    return "unchanged"


def _build_review_summary(
    *,
    base_version: str,
    final_version: str,
    adopted_version: str,
    review_required: bool,
    review_status: str,
    review_artifact: Optional[Dict[str, Any]],
    baseline_f1: Optional[float],
    best_val_f1: Optional[float],
) -> Dict[str, Any]:
    if not review_required:
        return {
            "needs_human_review": False,
            "current_status": "无需人工审核",
            "base_version": base_version,
            "candidate_version": final_version,
            "adopted_version": adopted_version,
            "f1_change": None if baseline_f1 is None or best_val_f1 is None else round(best_val_f1 - baseline_f1, 4),
            "review_diff_path": None,
            "review_summary_path": None,
            "boss_recommendation": "可直接采用当前最终版本。",
            "next_action": "无需额外审核动作。",
        }

    review_diff_path = None if not review_artifact else review_artifact.get("review_diff_path")
    review_summary_path = None if not review_artifact else review_artifact.get("review_summary_path")
    f1_change = None if baseline_f1 is None or best_val_f1 is None else round(best_val_f1 - baseline_f1, 4)

    return {
        "needs_human_review": True,
        "current_status": "候选版本已生成，等待人工审核",
        "base_version": base_version,
        "candidate_version": final_version,
        "adopted_version": adopted_version,
        "f1_change": f1_change,
        "review_status": review_status,
        "review_diff_path": review_diff_path,
        "review_summary_path": review_summary_path,
        "boss_recommendation": "先审阅 formal skill 差异，再决定 approve 或 reject。",
        "next_action": "若审核通过则采纳 candidate_version；若未通过则保留 base_version。",
    }


def parse_pipeline_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    from src.config import get_active_skill_version

    parser = argparse.ArgumentParser(description="Run the project end-to-end offline pipeline.")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--quick", action="store_true", help="Quick mode for rebuilt benchmark assets.")
    mode.add_argument("--full", action="store_true", help="Full mode for rebuilt benchmark assets.")
    parser.add_argument("--quick-n", type=int, default=50, help="Per-source cap for quick rebuild mode.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed used when rebuilding data.")

    parser.add_argument("--rebuild-data", action="store_true", help="Force benchmark split rebuild.")
    parser.add_argument(
        "--repair-benchmark-manifest",
        action="store_true",
        help="Write data/benchmark/manifest.json from existing train/val/test without changing the split files.",
    )
    parser.add_argument("--rebuild-index", action="store_true", help="Force RAG index rebuild.")
    parser.add_argument("--cleanup", action="store_true", help="Delete benchmark and RAG index after the run.")
    parser.add_argument(
        "--cleanup-generated-skills",
        action="store_true",
        help="Delete versions generated during this run and restore the starting active version.",
    )

    parser.add_argument(
        "--skill-version",
        type=str,
        default=get_active_skill_version(),
        help="Starting skill version. Defaults to the active skill.",
    )
    parser.add_argument(
        "--formal-skill",
        action="store_true",
        help=f"Run the pipeline against the formal bundled Skill package ({FORMAL_SKILL_VERSION}) without manually switching active_version.txt.",
    )

    parser.add_argument("--baseline-rag-mode", choices=["on", "off", "compare"], default="on")
    parser.add_argument("--evolution-rag", choices=["on", "off"], default="on")
    parser.add_argument("--baseline-memory-mode", choices=["on", "off", "compare"], default="off")
    parser.add_argument("--evolution-memory", choices=["on", "off"], default="off")
    parser.add_argument("--run-test-final", action="store_true")
    parser.add_argument("--test-rag-mode", choices=["match-evolution", "on", "off", "compare"], default="match-evolution")
    parser.add_argument("--test-memory-mode", choices=["match-evolution", "on", "off", "compare"], default="match-evolution")

    parser.add_argument("--baseline-max-eval", type=int, default=None)
    parser.add_argument("--evolution-max-eval", type=int, default=None)
    parser.add_argument("--test-max-eval", type=int, default=None)
    parser.add_argument("--max-eval", type=int, default=None, help="Compatibility alias for all evaluation stages.")
    parser.add_argument(
        "--sample-strategy",
        choices=["sequential", "random", "stratified"],
        default="stratified",
        help="Subset selection strategy when max-eval limits evaluation data.",
    )
    parser.add_argument(
        "--sample-seed",
        type=int,
        default=None,
        help="Random seed for evaluation subset selection. Defaults to --seed.",
    )

    parser.add_argument("--max-rounds", type=int, default=10)
    parser.add_argument("--rounds", type=int, default=None, help="Compatibility alias for --max-rounds.")
    parser.add_argument("--patience", type=int, default=2)
    parser.add_argument("--min-improvement", type=float, default=0.001)
    parser.add_argument(
        "--optimizer-max-errors",
        type=int,
        default=None,
        help="Cap optimizer error examples. Defaults to auto sizing based on eval slice size.",
    )

    parser.add_argument("--no-save-report", action="store_true", help="Do not persist the JSON run report.")
    parser.add_argument(
        "--keep-artifacts",
        action="store_true",
        help="Deprecated compatibility flag. Artifacts are kept by default now.",
    )

    args = parser.parse_args(argv)

    if not args.quick and not args.full:
        args.full = True

    if args.rounds is not None:
        args.max_rounds = args.rounds

    if args.max_eval is not None:
        if args.baseline_max_eval is None:
            args.baseline_max_eval = args.max_eval
        if args.evolution_max_eval is None:
            args.evolution_max_eval = args.max_eval
        if args.test_max_eval is None:
            args.test_max_eval = args.max_eval

    if args.sample_seed is None:
        args.sample_seed = args.seed

    args = _resolve_skill_selection(args, argv)

    return args


def main(argv: Optional[List[str]] = None):
    args = parse_pipeline_args(argv)
    _validate_pipeline_config(
        baseline_rag_mode=args.baseline_rag_mode,
        evolution_rag=args.evolution_rag,
        baseline_memory_mode=args.baseline_memory_mode,
        evolution_memory=args.evolution_memory,
    )
    requested_mode = _resolve_requested_mode(args)
    max_per_source = args.quick_n if requested_mode == "quick" else None
    rag_needed = _needs_rag_assets(
        baseline_rag_mode=args.baseline_rag_mode,
        evolution_rag=args.evolution_rag,
        run_test_final=args.run_test_final,
        test_rag_mode=args.test_rag_mode,
    )

    report_payload: Dict[str, Any] = {
        "timestamp": datetime.now().isoformat(),
        "status": "running",
        "config": {
            "requested_mode": requested_mode,
            "quick_n": args.quick_n,
            "seed": args.seed,
            "skill_mode": args.skill_mode,
            "skill_version_start": args.skill_version,
            "baseline_rag_mode": args.baseline_rag_mode,
            "evolution_rag": args.evolution_rag,
            "baseline_memory_mode": args.baseline_memory_mode,
            "evolution_memory": args.evolution_memory,
            "test_rag_mode": args.test_rag_mode,
            "test_memory_mode": args.test_memory_mode,
            "run_test_final": args.run_test_final,
            "baseline_max_eval": args.baseline_max_eval,
            "evolution_max_eval": args.evolution_max_eval,
            "test_max_eval": args.test_max_eval,
            "sample_strategy": args.sample_strategy,
            "sample_seed": args.sample_seed,
            "max_rounds": args.max_rounds,
            "patience": args.patience,
            "min_improvement": args.min_improvement,
            "optimizer_max_errors": args.optimizer_max_errors,
            "rebuild_data": args.rebuild_data,
            "repair_benchmark_manifest": args.repair_benchmark_manifest,
            "rebuild_index": args.rebuild_index,
        },
        "assets": {},
        "baseline": {},
        "evolution": {
            "round_history": [],
            "created_versions": [],
            "accepted_versions": [],
            "rollback_count": 0,
        },
        "cleanup": {},
    }

    t0 = time.time()
    created_versions: List[str] = []
    exit_code = 0

    print("=" * 60)
    print("Pipeline started")
    print("=" * 60)
    print(
        f"mode={requested_mode} skill_mode={args.skill_mode} skill={args.skill_version} "
        f"baseline_rag={args.baseline_rag_mode} evolution_rag={args.evolution_rag} "
        f"baseline_memory={args.baseline_memory_mode} evolution_memory={args.evolution_memory}"
    )
    if args.keep_artifacts:
        print("[INFO] --keep-artifacts is redundant now; artifacts are kept by default.")

    try:
        repaired_manifest_result: Optional[Dict[str, Any]] = None
        if args.repair_benchmark_manifest:
            repaired_manifest_result = repair_benchmark_manifest(
                requested_mode=requested_mode,
                max_per_source=max_per_source,
                seed=args.seed,
            )
            if not repaired_manifest_result.get("success", False):
                raise RuntimeError(repaired_manifest_result.get("reason", "benchmark manifest repair failed"))

        benchmark_result = ensure_benchmark_data(
            max_per_source=max_per_source,
            seed=args.seed,
            rebuild=args.rebuild_data,
            requested_mode=requested_mode,
        )
        if not benchmark_result.get("success", False):
            raise RuntimeError(benchmark_result.get("reason", "benchmark asset validation failed"))

        rag_result: Optional[Dict[str, Any]] = None
        retriever = None
        if rag_needed:
            rag_max = max_per_source * 4 if max_per_source else None
            rag_result = ensure_rag_index(max_samples=rag_max, rebuild=args.rebuild_index)
            if not rag_result.get("success", False):
                raise RuntimeError(rag_result.get("reason", "RAG asset validation failed"))
            retriever = _load_retriever()
            if retriever is None:
                raise RuntimeError("RAG assets exist but the retriever could not be loaded")

        report_payload["assets"] = {
            "benchmark_reused": bool(benchmark_result.get("reused", False)),
            "benchmark_verified": bool(benchmark_result.get("verified", False)),
            "benchmark_manifest_repaired": repaired_manifest_result is not None,
            "rag_index_reused": bool(rag_result.get("reused", False)) if rag_result is not None else None,
            "manifest": {
                "benchmark": benchmark_result.get("manifest"),
                "rag": rag_result.get("manifest") if rag_result is not None else None,
            },
        }

        baseline_result = evaluate_rag_mode(
            skill_version=args.skill_version,
            max_samples=args.baseline_max_eval,
            rag_mode=args.baseline_rag_mode,
            mainline_mode=args.evolution_rag,
            dataset="val",
            retriever=retriever,
            memory_mode=args.baseline_memory_mode,
            mainline_memory=args.evolution_memory,
            sample_strategy=args.sample_strategy,
            sample_seed=args.sample_seed,
        )
        report_payload["baseline"] = {
            "selected_mainline": baseline_result["selected_mainline"],
            "selected_eval": baseline_result["selected_eval"],
        }
        if "evaluations" in baseline_result:
            report_payload["baseline"]["evaluations"] = baseline_result["evaluations"]
        if "rag_eval" in baseline_result:
            report_payload["baseline"]["rag_eval"] = baseline_result["rag_eval"]
        if "no_rag_eval" in baseline_result:
            report_payload["baseline"]["no_rag_eval"] = baseline_result["no_rag_eval"]
        if "rag_minus_no_rag_f1" in baseline_result:
            report_payload["baseline"]["rag_minus_no_rag_f1"] = baseline_result["rag_minus_no_rag_f1"]
        if "memory_eval" in baseline_result:
            report_payload["baseline"]["memory_eval"] = baseline_result["memory_eval"]
        if "no_memory_eval" in baseline_result:
            report_payload["baseline"]["no_memory_eval"] = baseline_result["no_memory_eval"]
        if "memory_minus_no_memory_f1" in baseline_result:
            report_payload["baseline"]["memory_minus_no_memory_f1"] = baseline_result["memory_minus_no_memory_f1"]

        baseline_report = baseline_result["selected_report"]
        baseline_f1 = baseline_result["selected_f1"]
        print(
            f"[BASELINE] selected_mainline=rag:{args.evolution_rag},"
            f"memory:{args.evolution_memory} F1={baseline_f1:.4f}"
        )

        evolution_result = step_evolution(
            current_version=args.skill_version,
            max_rounds=args.max_rounds,
            max_eval_samples=args.evolution_max_eval,
            retriever=(retriever if args.evolution_rag == "on" else None),
            use_memory=(args.evolution_memory == "on"),
            baseline_report=baseline_report,
            patience=args.patience,
            min_improvement=args.min_improvement,
            optimization_max_errors=args.optimizer_max_errors,
            sample_strategy=args.sample_strategy,
            sample_seed=args.sample_seed,
        )

        created_versions = evolution_result["created_versions"]
        final_version = evolution_result["final_version"]
        best_val_f1 = evolution_result["best_f1"]
        report_payload["evolution"] = {
            "round_history": evolution_result["round_history"],
            "created_versions": evolution_result["created_versions"],
            "accepted_versions": evolution_result["accepted_versions"],
            "rollback_count": evolution_result["rollback_count"],
            "stop_reason": evolution_result["stop_reason"],
            "best_version": evolution_result["best_version"],
            "best_f1": evolution_result["best_f1"],
            "rounds_executed": len(evolution_result["round_history"]),
            "review_required": evolution_result["review_required"],
            "review_status": evolution_result["review_status"],
            "review_artifact": evolution_result["review_artifact"],
            "adopted_version": evolution_result["adopted_version"],
        }

        final_test_f1 = None
        if args.run_test_final:
            resolved_test_mode = _resolve_test_rag_mode(args.test_rag_mode, args.evolution_rag)
            resolved_test_memory_mode = _resolve_test_memory_mode(args.test_memory_mode, args.evolution_memory)
            final_test_mainline_rag = args.evolution_rag if resolved_test_mode == "compare" else resolved_test_mode
            final_test_mainline_memory = (
                args.evolution_memory if resolved_test_memory_mode == "compare" else resolved_test_memory_mode
            )
            final_test_result = evaluate_rag_mode(
                skill_version=final_version,
                max_samples=args.test_max_eval,
                rag_mode=resolved_test_mode,
                mainline_mode=final_test_mainline_rag,
                dataset="test",
                retriever=retriever,
                memory_mode=resolved_test_memory_mode,
                mainline_memory=final_test_mainline_memory,
                sample_strategy=args.sample_strategy,
                sample_seed=args.sample_seed,
            )
            final_test_payload: Dict[str, Any] = {
                "selected_mainline": final_test_result["selected_mainline"],
                "selected_eval": final_test_result["selected_eval"],
            }
            if "evaluations" in final_test_result:
                final_test_payload["evaluations"] = final_test_result["evaluations"]
            if "rag_eval" in final_test_result:
                final_test_payload["rag_eval"] = final_test_result["rag_eval"]
            if "no_rag_eval" in final_test_result:
                final_test_payload["no_rag_eval"] = final_test_result["no_rag_eval"]
            if "rag_minus_no_rag_f1" in final_test_result:
                final_test_payload["rag_minus_no_rag_f1"] = final_test_result["rag_minus_no_rag_f1"]
            if "memory_eval" in final_test_result:
                final_test_payload["memory_eval"] = final_test_result["memory_eval"]
            if "no_memory_eval" in final_test_result:
                final_test_payload["no_memory_eval"] = final_test_result["no_memory_eval"]
            if "memory_minus_no_memory_f1" in final_test_result:
                final_test_payload["memory_minus_no_memory_f1"] = final_test_result["memory_minus_no_memory_f1"]
            report_payload["final_test"] = final_test_payload
            final_test_f1 = final_test_result["selected_f1"]

        report_payload["summary"] = {
            "baseline_f1": baseline_f1,
            "best_val_f1": best_val_f1,
            "final_test_f1": final_test_f1,
            "f1_trend": _trend_label(baseline_f1, best_val_f1),
            "skill_version_final": final_version,
            "skill_version_adopted": evolution_result["adopted_version"],
            "review_required": evolution_result["review_required"],
            "review_status": evolution_result["review_status"],
            "accepted_count": len(evolution_result["accepted_versions"]),
            "rollback_count": evolution_result["rollback_count"],
        }
        report_payload["review_summary"] = _build_review_summary(
            base_version=args.skill_version,
            final_version=final_version,
            adopted_version=evolution_result["adopted_version"],
            review_required=evolution_result["review_required"],
            review_status=evolution_result["review_status"],
            review_artifact=evolution_result["review_artifact"],
            baseline_f1=baseline_f1,
            best_val_f1=best_val_f1,
        )
        report_payload["status"] = "success"

    except Exception as exc:
        exit_code = 1
        report_payload["status"] = "failed"
        report_payload["error"] = str(exc)
        print(f"[ERROR] {exc}")

    finally:
        report_payload["elapsed_seconds"] = round(time.time() - t0, 2)
        report_payload["cleanup"] = _cleanup_pipeline_artifacts(
            cleanup_assets=args.cleanup,
            cleanup_generated_skills=args.cleanup_generated_skills,
            created_versions=created_versions,
            restore_version=args.skill_version,
        )

        if report_payload.get("summary") is not None and report_payload["cleanup"].get("restored_active_version"):
            report_payload["summary"]["active_version_restored_to"] = report_payload["cleanup"]["restored_active_version"]
            if report_payload.get("review_summary") is not None:
                report_payload["review_summary"]["active_version_restored_to"] = report_payload["cleanup"]["restored_active_version"]

        if not args.no_save_report:
            report_path = _save_run_report(report_payload)
            print(f"[REPORT] {report_path}")

    print("\n" + "=" * 60)
    print("Pipeline summary")
    print("=" * 60)
    print(f"status={report_payload['status']}")
    if report_payload.get("summary"):
        summary = report_payload["summary"]
        print(f"baseline_f1={summary['baseline_f1']:.4f}")
        print(f"best_val_f1={summary['best_val_f1']:.4f}")
        if summary["final_test_f1"] is not None:
            print(f"final_test_f1={summary['final_test_f1']:.4f}")
        print(f"trend={summary['f1_trend']}")
        print(f"final_version={summary['skill_version_final']}")
        if summary.get("skill_version_adopted") is not None:
            print(f"adopted_version={summary['skill_version_adopted']}")
        print(f"accepted={summary['accepted_count']} rollback={summary['rollback_count']}")
    if report_payload.get("review_summary"):
        review_summary = report_payload["review_summary"]
        print("review_summary:")
        print(f"  needs_human_review={review_summary['needs_human_review']}")
        print(f"  current_status={review_summary['current_status']}")
        print(f"  base_version={review_summary['base_version']}")
        print(f"  candidate_version={review_summary['candidate_version']}")
        print(f"  adopted_version={review_summary['adopted_version']}")
        if review_summary.get("f1_change") is not None:
            print(f"  f1_change={review_summary['f1_change']:+.4f}")
        if review_summary.get("review_diff_path"):
            print(f"  review_diff_path={review_summary['review_diff_path']}")
    print(f"cleanup_performed={report_payload['cleanup'].get('performed', False)}")

    if exit_code != 0:
        raise SystemExit(exit_code)


if __name__ == "__main__":
    main()

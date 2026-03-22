from __future__ import annotations

import json
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

from src.config import ROOT_DIR, SKILLS_DIR, get_active_skill_version
from src.evolution.optimizer import SkillOptimizer
from src.utils.path_utils import normalize_project_paths, to_relative_posix_path

from .compare import compare_reports
from .judge import judge_run_artifacts
from .runner import CodexRunnerConfig, CodexSkillRunner
from .versioning import (
    build_stamped_stable_version_name,
    build_version_inventory,
    ensure_version_snapshot,
    next_patch_version_name,
    next_available_candidate_version_name,
    parse_version_name,
    publish_candidate_to_stable,
)


@dataclass
class SkillAgentLoopConfig:
    baseline_source_dir: Path = SKILLS_DIR / "minor-detection"
    baseline_version: str = "minor-detection-v0.1.0"
    dataset_path: Path = ROOT_DIR / "data" / "benchmark" / "val.jsonl"
    max_rounds: int = 1
    max_errors: Optional[int] = None
    protected_count: int = 8
    workspace_root: Path = ROOT_DIR / "reports" / "skill_agent_loops"
    packages_root: Path = ROOT_DIR / "reports" / "skill_packages"
    refresh_baseline_version: bool = False
    runner_config: Any = field(default_factory=CodexRunnerConfig)


def _display_path(value: Any) -> str:
    try:
        path = Path(value)
    except TypeError:
        return str(value)
    return to_relative_posix_path(path, ROOT_DIR) if path.is_absolute() else str(value).replace("\\", "/")


def _log(message: str) -> None:
    print(f"[skill-loop] {message}", file=sys.stderr, flush=True)


class SkillAgentLoop:
    def __init__(
        self,
        *,
        config: Optional[SkillAgentLoopConfig] = None,
        runner: Optional[Any] = None,
        optimizer: Optional[SkillOptimizer] = None,
    ):
        self.config = config or SkillAgentLoopConfig()
        self.runner = runner or CodexSkillRunner(config=self.config.runner_config)
        self.optimizer = optimizer or SkillOptimizer()

    def _workspace(self) -> Path:
        workspace = self.config.workspace_root / datetime.now().strftime("%Y%m%d_%H%M%S")
        workspace.mkdir(parents=True, exist_ok=True)
        return workspace

    def _skill_source_dir(self, version_name: str) -> Path:
        return SKILLS_DIR / version_name

    def _evaluate_version(
        self,
        *,
        version_name: str,
        parent_version: Optional[str],
        workspace: Path,
    ) -> Dict[str, Any]:
        skill_source_dir = self._skill_source_dir(version_name)
        run_root = self.runner.run_dataset(
            project_root=ROOT_DIR,
            skill_source_dir=skill_source_dir,
            skill_version=version_name,
            dataset_path=self.config.dataset_path,
            workspace_dir=workspace,
        )
        run_manifest_path = run_root / "run_manifest.json"
        run_manifest = json.loads(run_manifest_path.read_text(encoding="utf-8")) if run_manifest_path.exists() else {}
        judged = judge_run_artifacts(
            run_root=run_root,
            skill_version=version_name,
            parent_version=parent_version,
            dataset_name=self.config.dataset_path.stem,
            max_errors=self.config.max_errors,
            protected_count=self.config.protected_count,
            project_root=ROOT_DIR,
        )
        judged["skill_source_dir"] = to_relative_posix_path(skill_source_dir, workspace)
        judged["run_root"] = to_relative_posix_path(run_root, workspace)
        judged["run_manifest_path"] = to_relative_posix_path(run_manifest_path, workspace) if run_manifest_path.exists() else None
        judged["runtime_summary"] = run_manifest.get("timing", {}) if isinstance(run_manifest, dict) else {}
        judged["runtime_counts"] = run_manifest.get("counts", {}) if isinstance(run_manifest, dict) else {}
        return judged

    def _skip_comparison(self, reason: str) -> Dict[str, Any]:
        return {
            "decision": "skipped",
            "reason": reason,
        }

    def _baseline_runtime_blocker_reason(self, accepted_eval: Dict[str, Any]) -> Optional[str]:
        report = accepted_eval.get("report_payload") or {}
        if not isinstance(report, dict):
            return None
        sample_count = int(report.get("sample_count", 0) or 0)
        invocation_success_rate = float(report.get("invocation_success_rate", 0.0) or 0.0)
        if sample_count > 0 and invocation_success_rate <= 0.0:
            runner_label = str(getattr(self.runner, "runner_label", "runtime") or "runtime")
            return f"baseline {runner_label} invocation failed before any valid output; fix runtime/model/skill loading before optimization"
        return None

    def run(self) -> Dict[str, Any]:
        loop_started_at = time.time()
        workspace = self._workspace()
        _log(f"workspace={_display_path(workspace)}")
        _log(f"baseline snapshot -> {self.config.baseline_version}")
        ensure_version_snapshot(
            self.config.baseline_source_dir,
            SKILLS_DIR / self.config.baseline_version,
            refresh=self.config.refresh_baseline_version,
        )
        accepted_version = self.config.baseline_version
        parsed_baseline = parse_version_name(self.config.baseline_version) or {}
        base_name = str(parsed_baseline.get("base") or self.config.baseline_source_dir.name)
        run_tag = workspace.name
        version_inventory_before = build_version_inventory(
            SKILLS_DIR,
            base_name=base_name,
            active_version=get_active_skill_version(),
            only_run_tag=run_tag,
            scope_active_version=True,
        )
        _log(f"evaluate baseline {accepted_version}")
        accepted_eval = self._evaluate_version(version_name=accepted_version, parent_version=None, workspace=workspace / "baseline")
        baseline_eval = accepted_eval

        rounds = []
        final_version = accepted_version
        next_stable_semantic = next_patch_version_name(accepted_version)
        next_stable_version = build_stamped_stable_version_name(next_stable_semantic, run_tag)
        review_artifact = None
        manual_review_status = "not_required"
        manual_review_base_version = None
        manual_review_candidate_version = None

        baseline_blocker = self._baseline_runtime_blocker_reason(accepted_eval)
        if baseline_blocker:
            rounds.append(
                {
                    "round": 1,
                    "accepted_version": accepted_version,
                    "candidate_version": None,
                    "comparison": self._skip_comparison(baseline_blocker),
                    "optimize_result": {
                        "success": False,
                        "message": baseline_blocker,
                        "current_version": accepted_version,
                        "edited_files": [],
                    },
                }
            )
        else:
            for round_index in range(1, self.config.max_rounds + 1):
                candidate_version = next_available_candidate_version_name(
                    next_stable_semantic,
                    SKILLS_DIR,
                    start_index=1,
                    run_tag=run_tag,
                )
                _log(f"round {round_index}: optimize {accepted_version} -> {candidate_version}")
                optimize_result = self.optimizer.optimize_from_judge_artifacts(
                    report_path=accepted_eval["report_path"],
                    failure_packets_dir=accepted_eval["failure_packets_dir"],
                    protected_packets_dir=accepted_eval["protected_packets_dir"],
                    current_version=accepted_version,
                    new_version=candidate_version,
                    dry_run=False,
                )
                candidate_dir = SKILLS_DIR / candidate_version
                if optimize_result.get("new_version") != candidate_version or not candidate_dir.exists():
                    optimize_message = str(optimize_result.get("message", "") or "").strip().lower()
                    comparison = self._skip_comparison("optimizer did not generate candidate skill directory")
                    if optimize_message == "no errors to optimize":
                        comparison = self._skip_comparison("no errors to optimize on current eval slice")
                    elif optimize_message == "no editable targets resolved from judge report":
                        comparison = self._skip_comparison("no editable targets resolved from judge report")
                    rounds.append(
                        {
                            "round": round_index,
                            "accepted_version": accepted_version,
                            "candidate_version": candidate_version,
                            "comparison": comparison,
                            "optimize_result": optimize_result,
                        }
                    )
                    break

                _log(f"round {round_index}: evaluate candidate {candidate_version}")
                candidate_eval = self._evaluate_version(
                    version_name=candidate_version,
                    parent_version=accepted_version,
                    workspace=workspace / f"round_{round_index:02d}" / "candidate",
                )
                comparison = compare_reports(
                    accepted_report_path=accepted_eval["report_path"],
                    candidate_report_path=candidate_eval["report_path"],
                    accepted_protected_index_path=accepted_eval["protected_index_path"],
                    candidate_error_index_path=candidate_eval["error_index_path"],
                )

                round_payload = {
                    "round": round_index,
                    "accepted_version": accepted_version,
                    "candidate_version": candidate_version,
                    "comparison": comparison,
                    "optimize_result": optimize_result,
                    "accepted_runtime": accepted_eval.get("runtime_summary", {}),
                    "accepted_runtime_counts": accepted_eval.get("runtime_counts", {}),
                    "candidate_runtime": candidate_eval.get("runtime_summary", {}),
                    "candidate_runtime_counts": candidate_eval.get("runtime_counts", {}),
                }

                _log(f"round {round_index}: compare decision={comparison.get('decision')}")
                if comparison["decision"] == "promote":
                    publish_candidate_to_stable(SKILLS_DIR / candidate_version, SKILLS_DIR / next_stable_version)
                    round_payload["promoted_to"] = next_stable_version
                    final_version = next_stable_version
                    accepted_version = next_stable_version
                    accepted_eval = candidate_eval
                    next_stable_semantic = next_patch_version_name(accepted_version)
                    next_stable_version = build_stamped_stable_version_name(next_stable_semantic, run_tag)
                rounds.append(round_payload)
                if comparison["decision"] != "promote":
                    break

        if final_version != self.config.baseline_version:
            review_artifact = self.optimizer.create_formal_skill_review_artifact(
                base_version=self.config.baseline_version,
                candidate_version=final_version,
            )
            manual_review_status = "pending"
            manual_review_base_version = self.config.baseline_version
            manual_review_candidate_version = final_version

        version_inventory_after = build_version_inventory(
            SKILLS_DIR,
            base_name=base_name,
            active_version=get_active_skill_version(),
            only_run_tag=run_tag,
            scope_active_version=True,
        )
        manual_final_test_command = None
        manual_review_approve_command = None
        manual_review_reject_command = None
        if manual_review_candidate_version:
            runner_mode = str(getattr(self.runner, "runner_mode", "agent") or "agent")
            manual_final_test_command = (
                f"python scripts/run_final_test.py --version {manual_review_candidate_version} --runner-mode {runner_mode}"
            )
            manual_review_approve_command = (
                "python -m src.evolution.optimizer "
                f"--review-base-version {manual_review_base_version} "
                f"--review-candidate-version {manual_review_candidate_version} "
                "--review-decision approve"
            )
            manual_review_reject_command = (
                "python -m src.evolution.optimizer "
                f"--review-base-version {manual_review_base_version} "
                f"--review-candidate-version {manual_review_candidate_version} "
                "--review-decision reject"
            )
        summary = {
            "runner_mode": str(getattr(self.runner, "runner_mode", "agent") or "agent"),
            "baseline_version": self.config.baseline_version,
            "final_version": final_version,
            "dataset": to_relative_posix_path(self.config.dataset_path, workspace),
            "workspace": ".",
            "baseline_runtime": baseline_eval.get("runtime_summary", {}),
            "baseline_runtime_counts": baseline_eval.get("runtime_counts", {}),
            "rounds": rounds,
            "review_artifact": review_artifact,
            "manual_review_required": manual_review_candidate_version is not None,
            "manual_review_status": manual_review_status,
            "manual_review_base_version": manual_review_base_version,
            "manual_review_candidate_version": manual_review_candidate_version,
            "manual_final_test_script": "scripts/run_final_test.py",
            "manual_final_test_command": manual_final_test_command,
            "manual_review_approve_command": manual_review_approve_command,
            "manual_review_reject_command": manual_review_reject_command,
            "timing": {
                "total_loop_wall_seconds": round(time.time() - loop_started_at, 3),
            },
            "version_management": {
                "base_name": base_name,
                "run_tag": run_tag,
                "history_scope": "current_run_only",
                "inventory_before": version_inventory_before,
                "inventory_after": version_inventory_after,
                "recommended_cleanup_command": f"python scripts/cleanup_skill_versions.py --base-name {base_name} --keep-latest-stable 2 --only-run-tag {run_tag} --dry-run",
            },
        }
        _log(f"final_version={final_version}")
        if manual_review_candidate_version:
            _log(
                "manual review pending: "
                f"base={manual_review_base_version} candidate={manual_review_candidate_version}"
            )
        summary_path = workspace / "loop_summary.json"
        with summary_path.open("w", encoding="utf-8") as f:
            json.dump(
                normalize_project_paths(summary, project_root=ROOT_DIR, start=workspace),
                f,
                ensure_ascii=False,
                indent=2,
            )
        summary["summary_path"] = to_relative_posix_path(summary_path, ROOT_DIR)
        return summary



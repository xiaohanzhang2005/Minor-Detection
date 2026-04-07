from __future__ import annotations

import json
import sys
import time
from dataclasses import dataclass, field, replace
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, Optional

from src.config import ROOT_DIR, SKILLS_DIR, get_active_skill_version
from src.evolution.optimizer import SkillOptimizer
from src.skill_loop.compare import compare_reports
from src.skill_loop.versioning import (
    build_stamped_stable_version_name,
    build_version_inventory,
    ensure_version_snapshot,
    next_available_candidate_version_name,
    next_patch_version_name,
    parse_version_name,
)
from src.utils.path_utils import normalize_project_paths, to_relative_posix_path

from .judge import judge_trigger_full_smoke_artifacts, judge_trigger_run_artifacts
from .runner import TriggerEvalCodexRunner, TriggerEvalRunnerConfig


def _display_path(value: Any) -> str:
    try:
        path = Path(value)
    except TypeError:
        return str(value)
    if not path.is_absolute():
        return str(value).replace("\\", "/")
    try:
        return to_relative_posix_path(path, ROOT_DIR)
    except ValueError:
        return path.as_posix()


def _relative_or_display(path: Path, base: Path) -> str:
    try:
        return to_relative_posix_path(path, base)
    except ValueError:
        return path.as_posix()


def _log(message: str) -> None:
    print(f"[trigger-description-loop] {message}", file=sys.stderr, flush=True)


@dataclass
class TriggerDescriptionLoopConfig:
    baseline_source_dir: Path = SKILLS_DIR / "minor-detection"
    baseline_version: str = "minor-detection-v0.1.0"
    optimization_set_path: Path = ROOT_DIR / "data" / "trigger_eval" / "minor_detection_trigger_eval_v1_optimization_set.json"
    final_validation_set_path: Optional[Path] = ROOT_DIR / "data" / "trigger_eval" / "minor_detection_trigger_eval_v1_final_validation_set.json"
    release_contract_set_path: Optional[Path] = ROOT_DIR / "data" / "trigger_eval" / "minor_detection_trigger_eval_v1_final_validation_set.json"
    max_rounds: int = 1
    max_errors: Optional[int] = None
    protected_count: int = 8
    workspace_root: Path = ROOT_DIR / "reports" / "trigger_description_loops"
    refresh_baseline_version: bool = False
    runner_config: TriggerEvalRunnerConfig = field(default_factory=TriggerEvalRunnerConfig)
    judge_fn: Callable[..., Dict[str, Any]] = judge_trigger_run_artifacts
    compare_fn: Callable[..., Dict[str, Any]] = compare_reports
    manual_smoke_validation_script: str = "scripts/run_trigger_eval.py"
    manual_smoke_validation_command_template: Optional[str] = None
    manual_final_test_script: str = "scripts/run_trigger_description_validation.py"
    manual_final_test_command_template: Optional[str] = None
    manual_release_contract_gate_script: str = "scripts/run_trigger_release_contract_gate.py"
    manual_release_contract_gate_command_template: Optional[str] = None
    repeat_runs_per_sample: int = 1


class TriggerDescriptionLoop:
    def __init__(
        self,
        *,
        config: Optional[TriggerDescriptionLoopConfig] = None,
        runner: Optional[TriggerEvalCodexRunner] = None,
        optimizer: Optional[SkillOptimizer] = None,
    ):
        self.config = config or TriggerDescriptionLoopConfig()
        self.runner = runner or TriggerEvalCodexRunner(config=self.config.runner_config)
        self.optimizer = optimizer or SkillOptimizer()

    def _runner_with_skill_execution_mode(self, skill_execution_mode: str) -> TriggerEvalCodexRunner:
        base_config = getattr(self.runner, "config", None)
        if not isinstance(base_config, TriggerEvalRunnerConfig):
            base_config = self.config.runner_config
        runner_config = replace(base_config, skill_execution_mode=skill_execution_mode)
        return TriggerEvalCodexRunner(
            config=runner_config,
            command_runner=getattr(self.runner, "command_runner", None),
        )

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
            dataset_path=self.config.optimization_set_path,
            workspace_dir=workspace,
        )
        run_manifest_path = run_root / "run_manifest.json"
        run_manifest = json.loads(run_manifest_path.read_text(encoding="utf-8")) if run_manifest_path.exists() else {}
        judged = self.config.judge_fn(
            run_root=run_root,
            skill_version=version_name,
            parent_version=parent_version,
            dataset_name=self.config.optimization_set_path.stem,
            max_errors=self.config.max_errors,
            protected_count=self.config.protected_count,
            project_root=ROOT_DIR,
        )
        judged["skill_source_dir"] = _relative_or_display(skill_source_dir, workspace)
        judged["run_root"] = _relative_or_display(run_root, workspace)
        judged["run_manifest_path"] = _relative_or_display(run_manifest_path, workspace) if run_manifest_path.exists() else None
        judged["runtime_summary"] = run_manifest.get("timing", {}) if isinstance(run_manifest, dict) else {}
        judged["runtime_counts"] = run_manifest.get("counts", {}) if isinstance(run_manifest, dict) else {}
        return judged

    def _run_release_validations(
        self,
        *,
        version_name: str,
        workspace: Path,
    ) -> Dict[str, Any]:
        if not self.config.final_validation_set_path:
            return {
                "pass": True,
                "status": "skipped_no_final_validation_set",
                "description_validation": None,
                "full_smoke": None,
            }

        skill_source_dir = self._skill_source_dir(version_name)
        description_run_root = self.runner.run_dataset(
            project_root=ROOT_DIR,
            skill_source_dir=skill_source_dir,
            skill_version=version_name,
            dataset_path=self.config.final_validation_set_path,
            workspace_dir=workspace / "desc",
        )
        description_validation = judge_trigger_run_artifacts(
            run_root=description_run_root,
            skill_version=version_name,
            parent_version=None,
            dataset_name=self.config.final_validation_set_path.stem,
            max_errors=self.config.max_errors,
            protected_count=self.config.protected_count,
            project_root=ROOT_DIR,
        )
        full_smoke_runner = self._runner_with_skill_execution_mode("full")
        full_smoke_run_root = full_smoke_runner.run_dataset(
            project_root=ROOT_DIR,
            skill_source_dir=skill_source_dir,
            skill_version=version_name,
            dataset_path=self.config.final_validation_set_path,
            workspace_dir=workspace / "smoke",
        )
        full_smoke = judge_trigger_full_smoke_artifacts(
            run_root=full_smoke_run_root,
            skill_version=version_name,
            dataset_name=self.config.final_validation_set_path.stem,
            project_root=ROOT_DIR,
        )
        description_gate_results = (description_validation.get("report_payload") or {}).get("gate_results") or {}
        full_smoke_gate_results = (full_smoke.get("report_payload") or {}).get("gate_results") or {}
        passed = bool(description_gate_results.get("release_gate_pass")) and bool(
            full_smoke_gate_results.get("full_smoke_release_pass")
        )
        return {
            "pass": passed,
            "status": "passed" if passed else "failed",
            "description_validation": normalize_project_paths(description_validation, project_root=ROOT_DIR, start=workspace),
            "full_smoke": normalize_project_paths(full_smoke, project_root=ROOT_DIR, start=workspace),
        }

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

    def _manual_smoke_validation_command(self, version_name: str) -> str:
        template = self.config.manual_smoke_validation_command_template
        if not template:
            script = self.config.manual_smoke_validation_script
            template = f"python {script} --version {{version}}"
        return str(normalize_project_paths(template.format(version=version_name), project_root=ROOT_DIR, start=ROOT_DIR))

    def _manual_final_validation_command(self, version_name: str) -> str:
        template = self.config.manual_final_test_command_template
        if not template:
            script = self.config.manual_final_test_script
            template = f"python {script} --version {{version}}"
        return str(normalize_project_paths(template.format(version=version_name), project_root=ROOT_DIR, start=ROOT_DIR))

    def _manual_release_contract_gate_command(self, version_name: str) -> str:
        template = self.config.manual_release_contract_gate_command_template
        if not template:
            script = self.config.manual_release_contract_gate_script
            template = f"python {script} --version {{version}}"
        return str(normalize_project_paths(template.format(version=version_name), project_root=ROOT_DIR, start=ROOT_DIR))

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
        champion_version = accepted_version
        final_version = accepted_version
        next_stable_semantic = next_patch_version_name(self.config.baseline_version)
        proposed_stable_version = build_stamped_stable_version_name(next_stable_semantic, run_tag)
        published_stable_version = None
        review_artifact = None
        manual_review_status = "not_required"
        manual_review_base_version = None
        manual_review_candidate_version = None
        final_validation_status = "not_required"
        release_contract_gate_status = "not_required"
        release_contract_gate_mode = "soft"
        release_contract_gate_blocking = False

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
                    elif optimize_message == "description revision is not substantive":
                        comparison = self._skip_comparison("optimizer candidate description change was not substantive")
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

                edit_contract_check = self.optimizer.evaluate_candidate_edit_contract(
                    base_version=accepted_version,
                    candidate_version=candidate_version,
                )
                if not edit_contract_check.get("pass"):
                    rounds.append(
                        {
                            "round": round_index,
                            "accepted_version": accepted_version,
                            "candidate_version": candidate_version,
                            "comparison": self._skip_comparison("candidate violated resolved edit contract"),
                            "optimize_result": optimize_result,
                            "edit_contract_check": edit_contract_check,
                        }
                    )
                    break

                _log(f"round {round_index}: evaluate candidate {candidate_version}")
                candidate_eval = self._evaluate_version(
                    version_name=candidate_version,
                    parent_version=accepted_version,
                    workspace=workspace / f"round_{round_index:02d}" / "candidate",
                )
                comparison = self.config.compare_fn(
                    accepted_report_path=accepted_eval["report_path"],
                    candidate_report_path=candidate_eval["report_path"],
                    accepted_error_index_path=accepted_eval["error_index_path"],
                    accepted_protected_index_path=accepted_eval["protected_index_path"],
                    candidate_error_index_path=candidate_eval["error_index_path"],
                    candidate_redline_index_path=candidate_eval.get("redline_index_path"),
                )

                round_payload = {
                    "round": round_index,
                    "accepted_version": accepted_version,
                    "candidate_version": candidate_version,
                    "comparison": comparison,
                    "optimize_result": optimize_result,
                    "edit_contract_check": edit_contract_check,
                    "accepted_runtime": accepted_eval.get("runtime_summary", {}),
                    "accepted_runtime_counts": accepted_eval.get("runtime_counts", {}),
                    "candidate_runtime": candidate_eval.get("runtime_summary", {}),
                    "candidate_runtime_counts": candidate_eval.get("runtime_counts", {}),
                }

                _log(f"round {round_index}: compare decision={comparison.get('decision')}")
                if comparison["decision"] == "promote":
                    round_payload["promoted_to"] = candidate_version
                    champion_version = candidate_version
                    final_version = candidate_version
                    accepted_version = candidate_version
                    accepted_eval = candidate_eval
                rounds.append(round_payload)

        if champion_version != self.config.baseline_version:
            review_artifact = self.optimizer.create_formal_skill_review_artifact(
                base_version=self.config.baseline_version,
                candidate_version=champion_version,
            )
            manual_review_status = "pending"
            manual_review_base_version = self.config.baseline_version
            manual_review_candidate_version = champion_version
            final_validation_status = "pending_manual_review"
            release_contract_gate_status = "pending_manual_review"

        version_inventory_after = build_version_inventory(
            SKILLS_DIR,
            base_name=base_name,
            active_version=get_active_skill_version(),
            only_run_tag=run_tag,
            scope_active_version=True,
        )
        manual_smoke_validation_command = None
        manual_smoke_validation_script = self.config.manual_release_contract_gate_script
        manual_final_test_command = None
        manual_release_contract_gate_command = None
        manual_release_contract_gate_script = self.config.manual_release_contract_gate_script
        publish_stable_command = None
        manual_review_approve_command = None
        manual_review_reject_command = None
        if manual_review_candidate_version:
            manual_smoke_validation_command = self._manual_release_contract_gate_command(manual_review_candidate_version)
            manual_final_test_command = self._manual_final_validation_command(manual_review_candidate_version)
            manual_release_contract_gate_command = manual_smoke_validation_command
            publish_stable_command = (
                "python scripts/publish_reviewed_skill_version.py "
                f"--base-version {manual_review_base_version} "
                f"--candidate-version {manual_review_candidate_version} "
                f"--stable-version {proposed_stable_version}"
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
            "loop_type": "trigger_description_optimization",
            "runner_mode": str(getattr(self.runner, "runner_mode", "trigger_agent") or "trigger_agent"),
            "optimization_scope": "trigger_decision_and_skill_invocation_success_only",
            "optimization_target": "SKILL.md frontmatter description",
            "evaluation_contract": "trigger_decision_and_skill_invocation_success_only",
            "optimizer_feedback_enabled": True,
            "post_loop_validation_role": "standalone_trigger_final_validation",
            "baseline_version": self.config.baseline_version,
            "champion_version": champion_version,
            "final_version": final_version,
            "proposed_stable_version": proposed_stable_version if manual_review_candidate_version else None,
            "published_stable_version": published_stable_version,
            "optimization_set": _relative_or_display(self.config.optimization_set_path, workspace),
            "final_validation_set": _relative_or_display(self.config.final_validation_set_path, workspace)
            if self.config.final_validation_set_path
            else None,
            "release_contract_set": _relative_or_display(self.config.release_contract_set_path, workspace)
            if self.config.release_contract_set_path
            else None,
            "dataset": _relative_or_display(self.config.optimization_set_path, workspace),
            "workspace": ".",
            "repeat_runs_per_sample": self.config.repeat_runs_per_sample,
            "baseline_runtime": baseline_eval.get("runtime_summary", {}),
            "baseline_runtime_counts": baseline_eval.get("runtime_counts", {}),
            "rounds": rounds,
            "review_artifact": review_artifact,
            "manual_review_required": manual_review_candidate_version is not None,
            "manual_review_status": manual_review_status,
            "manual_review_base_version": manual_review_base_version,
            "manual_review_candidate_version": manual_review_candidate_version,
            "final_validation_status": final_validation_status,
            "release_contract_gate_status": release_contract_gate_status,
            "contract_check_status": release_contract_gate_status,
            "release_contract_gate_mode": release_contract_gate_mode,
            "release_contract_gate_blocking": release_contract_gate_blocking,
            "contract_check_mode": "indicator_only",
            "contract_check_blocking": False,
            "post_loop_final_validation_includes_soft_contract_metrics": True,
            "manual_smoke_validation_script": manual_smoke_validation_script,
            "manual_smoke_validation_command": manual_smoke_validation_command,
            "manual_final_test_script": self.config.manual_final_test_script,
            "manual_final_test_command": manual_final_test_command,
            "post_loop_release_gate_role": "optional_standalone_trigger_release_contract_gate",
            "manual_release_contract_gate_script": manual_release_contract_gate_script,
            "manual_release_contract_gate_command": manual_release_contract_gate_command,
            "manual_contract_check_script": manual_release_contract_gate_script,
            "manual_contract_check_command": manual_release_contract_gate_command,
            "publish_stable_command": publish_stable_command,
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
        _log(f"champion_version={champion_version}")
        if manual_review_candidate_version:
            _log(
                "manual review pending: "
                f"base={manual_review_base_version} candidate={manual_review_candidate_version}"
            )
            _log(
                "recommended final validation on final_validation_set (includes contract indicators): "
                f"{manual_final_test_command}"
            )
            _log(
                "recommended contract indicator check: "
                f"{manual_release_contract_gate_command}"
            )
        summary_path = workspace / "loop_summary.json"
        with summary_path.open("w", encoding="utf-8") as f:
            json.dump(
                normalize_project_paths(summary, project_root=ROOT_DIR, start=workspace),
                f,
                ensure_ascii=False,
                indent=2,
            )
        summary["summary_path"] = _display_path(summary_path)
        return summary

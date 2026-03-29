from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).parent.parent.absolute()
sys.path.insert(0, str(ROOT_DIR))

from src.trigger_eval import (
    TriggerEvalCodexRunner,
    TriggerEvalRunnerConfig,
    judge_trigger_run_artifacts,
)
from src.trigger_eval.loop import TriggerDescriptionLoop, TriggerDescriptionLoopConfig
from src.utils.path_utils import normalize_project_paths


def _quoted(value: str) -> str:
    escaped = str(value).replace('"', '\\"')
    return f'"{escaped}"'


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run the description optimization loop on trigger optimization_set. "
            "This loop optimizes trigger decision quality plus successful skill invocation, "
            "while final_validation_set is reserved for final standalone validation."
        )
    )
    parser.add_argument("--baseline-version", default="minor-detection-v0.1.0")
    parser.add_argument("--baseline-source-dir", default=str(ROOT_DIR / "skills" / "minor-detection"))
    parser.add_argument(
        "--optimization-set",
        default=str(ROOT_DIR / "data" / "trigger_eval" / "minor_detection_trigger_eval_v1_optimization_set.json"),
    )
    parser.add_argument(
        "--final-validation-set",
        default=str(ROOT_DIR / "data" / "trigger_eval" / "minor_detection_trigger_eval_v1_final_validation_set.json"),
    )
    parser.add_argument("--dataset", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--max-rounds", type=int, default=1)
    parser.add_argument("--max-errors", type=int)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument(
        "--sample-strategy",
        choices=["sequential", "random", "stratified"],
        default="stratified",
        help="Subset selection strategy when --max-samples limits the eval slice.",
    )
    parser.add_argument("--sample-seed", type=int, default=42)
    parser.add_argument("--workspace-root", default=str(ROOT_DIR / "reports" / "trigger_description_loops"))
    parser.add_argument("--codex-cmd", default="codex")
    parser.add_argument("--codex-model", default=None, help="Override the Codex/agent model for this run only.")
    parser.add_argument(
        "--execution-mode",
        choices=["sandbox", "bypass"],
        default="sandbox",
        help="How the inner Codex trigger agent runs. This does not control the optimizer's own Python-side model client.",
    )
    parser.add_argument(
        "--sandbox-mode",
        choices=["read-only", "workspace-write", "danger-full-access"],
        default="workspace-write",
        help="Sandbox mode passed to codex exec when --execution-mode sandbox is used.",
    )
    parser.add_argument(
        "--refresh-baseline-version",
        action="store_true",
        help="Rebuild the baseline version directory from --baseline-source-dir when an older snapshot already exists.",
    )
    parser.add_argument("--timeout-sec", type=int, default=600)
    args = parser.parse_args()

    optimization_set_path = Path(args.dataset) if args.dataset else Path(args.optimization_set)
    final_validation_set_path = Path(args.final_validation_set) if args.final_validation_set else None

    runner = TriggerEvalCodexRunner(
        config=TriggerEvalRunnerConfig(
            codex_cmd=args.codex_cmd,
            timeout_sec=args.timeout_sec,
            max_samples=args.max_samples,
            sample_strategy=args.sample_strategy,
            sample_seed=args.sample_seed,
            execution_mode=args.execution_mode,
            sandbox_mode=args.sandbox_mode,
            codex_model=args.codex_model,
        )
    )
    manual_final_test_command_template = (
        "python scripts/run_trigger_description_validation.py "
        f"--version {{version}} --dataset {_quoted(str(final_validation_set_path))} "
        f"--codex-cmd {_quoted(args.codex_cmd)} "
        + (f"--codex-model {_quoted(args.codex_model)} " if args.codex_model else "")
        + f"--execution-mode {args.execution_mode} --sandbox-mode {args.sandbox_mode} --timeout-sec {args.timeout_sec}"
    )
    manual_smoke_validation_command_template = (
        "python scripts/run_trigger_eval.py "
        f"--version {{version}} --dataset {_quoted(str(final_validation_set_path or optimization_set_path))} "
        f"--codex-cmd {_quoted(args.codex_cmd)} "
        + (f"--codex-model {_quoted(args.codex_model)} " if args.codex_model else "")
        + f"--execution-mode {args.execution_mode} --sandbox-mode {args.sandbox_mode} --timeout-sec {args.timeout_sec}"
    )

    config = TriggerDescriptionLoopConfig(
        baseline_source_dir=Path(args.baseline_source_dir),
        baseline_version=args.baseline_version,
        optimization_set_path=optimization_set_path,
        final_validation_set_path=final_validation_set_path,
        max_rounds=args.max_rounds,
        max_errors=args.max_errors,
        workspace_root=Path(args.workspace_root),
        refresh_baseline_version=args.refresh_baseline_version,
        runner_config=runner.config,
        judge_fn=judge_trigger_run_artifacts,
        manual_smoke_validation_command_template=manual_smoke_validation_command_template,
        manual_smoke_validation_script="scripts/run_trigger_eval.py",
        manual_final_test_command_template=manual_final_test_command_template,
        manual_final_test_script="scripts/run_trigger_description_validation.py",
        repeat_runs_per_sample=1,
    )
    result = TriggerDescriptionLoop(config=config, runner=runner).run()
    print(json.dumps(normalize_project_paths(result, project_root=ROOT_DIR, start=ROOT_DIR), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

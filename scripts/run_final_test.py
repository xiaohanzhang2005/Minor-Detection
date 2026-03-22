from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).parent.parent.absolute()
sys.path.insert(0, str(ROOT_DIR))

from src.skill_loop import CodexRunnerConfig, CodexSkillRunner, DirectRunnerConfig, DirectSkillRunner, judge_run_artifacts
from src.utils.path_utils import normalize_project_paths, to_relative_posix_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the manual final test flow on test.jsonl for a selected skill version.")
    parser.add_argument(
        "--version",
        required=True,
        help="Skill version or review candidate version, e.g. minor-detection-v0.1.1 or minor-detection-v0.1.1-rc001-20260322_120000",
    )
    parser.add_argument("--dataset", default=str(ROOT_DIR / "data" / "benchmark" / "test.jsonl"))
    parser.add_argument("--workspace", default=str(ROOT_DIR / "reports" / "final_tests"))
    parser.add_argument("--runner-mode", choices=["agent", "direct"], default="agent")
    parser.add_argument("--codex-cmd", default="codex")
    parser.add_argument("--codex-model", default=None, help="Override the Codex agent model for this run only.")
    parser.add_argument(
        "--execution-mode",
        choices=["sandbox", "bypass"],
        default="sandbox",
        help="Whether to run Codex under sandbox controls or fully bypass approvals and sandbox.",
    )
    parser.add_argument(
        "--sandbox-mode",
        choices=["read-only", "workspace-write", "danger-full-access"],
        default="workspace-write",
        help="Sandbox mode passed to codex exec when --execution-mode sandbox is used.",
    )
    parser.add_argument("--timeout-sec", type=int, default=600)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument(
        "--sample-strategy",
        choices=["sequential", "random", "stratified"],
        default="stratified",
        help="Subset selection strategy when --max-samples limits the eval slice.",
    )
    parser.add_argument("--sample-seed", type=int, default=42)
    args = parser.parse_args()

    version_dir = ROOT_DIR / "skills" / args.version
    if args.runner_mode == "direct":
        runner = DirectSkillRunner(
            config=DirectRunnerConfig(
                timeout_sec=args.timeout_sec,
                max_samples=args.max_samples,
                sample_strategy=args.sample_strategy,
                sample_seed=args.sample_seed,
            )
        )
    else:
        runner = CodexSkillRunner(
            config=CodexRunnerConfig(
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
    workspace = Path(args.workspace) / args.version
    run_root = runner.run_dataset(
        project_root=ROOT_DIR,
        skill_source_dir=version_dir,
        skill_version=args.version,
        dataset_path=Path(args.dataset),
        workspace_dir=workspace,
    )
    judged = judge_run_artifacts(
        run_root=run_root,
        skill_version=args.version,
        parent_version=None,
        dataset_name=Path(args.dataset).stem,
        project_root=ROOT_DIR,
    )
    payload = {
        "runner_mode": args.runner_mode,
        "report_path": to_relative_posix_path(judged["report_path"], ROOT_DIR),
        "run_root": to_relative_posix_path(run_root, ROOT_DIR),
        "skill_source_dir": to_relative_posix_path(version_dir, ROOT_DIR),
    }
    print(json.dumps(normalize_project_paths(payload, project_root=ROOT_DIR, start=ROOT_DIR), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

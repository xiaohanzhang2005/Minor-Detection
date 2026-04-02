from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).parent.parent.absolute()
sys.path.insert(0, str(ROOT_DIR))

from src.trigger_eval import TriggerEvalRunnerConfig, TriggerEvalCodexRunner, judge_trigger_run_artifacts
from src.utils.path_utils import normalize_project_paths, to_relative_posix_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run standalone trigger-description validation on a trigger dataset split. "
            "This validates trigger decision plus successful skill invocation without optimizer feedback."
        )
    )
    parser.add_argument("--version", default="minor-detection", help="Skill version directory under skills/ to evaluate.")
    parser.add_argument(
        "--dataset",
        default=str(ROOT_DIR / "data" / "trigger_eval" / "minor_detection_trigger_eval_v1_final_validation_set.json"),
    )
    parser.add_argument("--workspace", default=str(ROOT_DIR / "reports" / "trigger_description_validations"))
    parser.add_argument("--codex-cmd", default="codex")
    parser.add_argument("--codex-model", default=None)
    parser.add_argument("--agent-backend", choices=["codex", "cli"], default="codex")
    parser.add_argument("--agent-cmd", default=None, help="Alternative agent CLI executable when --agent-backend=cli.")
    parser.add_argument(
        "--agent-args-template",
        default=None,
        help="Optional generic agent command template. Available placeholders: {agent_cmd} {workspace_dir} {prompt_file} {final_output_path} {installed_skill_dir} {output_schema_path} {sandbox_mode} {execution_mode} {agent_model}.",
    )
    parser.add_argument("--agent-model", default=None, help="Vendor-neutral agent model label passed through to generic templates.")
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
    runner = TriggerEvalCodexRunner(
        config=TriggerEvalRunnerConfig(
            codex_cmd=args.codex_cmd,
            timeout_sec=args.timeout_sec,
            max_samples=args.max_samples,
            sample_strategy=args.sample_strategy,
            sample_seed=args.sample_seed,
            execution_mode=args.execution_mode,
            sandbox_mode=args.sandbox_mode,
            skill_execution_mode="probe",
            codex_model=args.codex_model,
            agent_backend=args.agent_backend,
            agent_cmd=args.agent_cmd,
            agent_args_template=args.agent_args_template,
            agent_model=args.agent_model or args.codex_model,
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
    judged = judge_trigger_run_artifacts(
        run_root=run_root,
        skill_version=args.version,
        parent_version=None,
        dataset_name=Path(args.dataset).stem,
        project_root=ROOT_DIR,
    )
    report = judged["report_payload"]
    payload = {
        "evaluation_role": "standalone_description_validation",
        "optimization_feedback_enabled": False,
        "runner_mode": runner.runner_mode,
        "report_path": to_relative_posix_path(judged["report_path"], ROOT_DIR),
        "run_root": to_relative_posix_path(run_root, ROOT_DIR),
        "skill_source_dir": to_relative_posix_path(version_dir, ROOT_DIR),
        "evaluation_contract": report.get("evaluation_contract"),
        "metrics": report.get("metrics", {}),
        "invocation_success_rate": report.get("invocation_success_rate"),
        "step_compliance_rate": report.get("step_compliance_rate"),
        "schema_validity_rate": report.get("schema_validity_rate"),
        "slice_stats": report.get("slice_stats", {}),
        "failure_type_counts": report.get("failure_type_counts", {}),
    }
    print(json.dumps(normalize_project_paths(payload, project_root=ROOT_DIR, start=ROOT_DIR), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

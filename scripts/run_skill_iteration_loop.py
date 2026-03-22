# 模块说明：
# - Mode A 的命令行入口。
# - 启动共享 loop engine，并接入 Codex agent runner。

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).parent.parent.absolute()
sys.path.insert(0, str(ROOT_DIR))

from src.skill_loop import CodexRunnerConfig, SkillAgentLoop, SkillAgentLoopConfig
from src.utils.path_utils import normalize_project_paths


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the real-Agent .skill iteration loop on val.jsonl.")
    parser.add_argument("--baseline-version", default="minor-detection-v0.1.0")
    parser.add_argument("--baseline-source-dir", default=str(ROOT_DIR / "skills" / "minor-detection"))
    parser.add_argument("--dataset", default=str(ROOT_DIR / "data" / "benchmark" / "val.jsonl"))
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
    parser.add_argument("--workspace-root", default=str(ROOT_DIR / "reports" / "skill_agent_loops"))
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
    parser.add_argument(
        "--refresh-baseline-version",
        action="store_true",
        help="Rebuild the baseline version directory from --baseline-source-dir when an older snapshot already exists.",
    )
    parser.add_argument("--timeout-sec", type=int, default=600)
    args = parser.parse_args()

    config = SkillAgentLoopConfig(
        baseline_source_dir=Path(args.baseline_source_dir),
        baseline_version=args.baseline_version,
        dataset_path=Path(args.dataset),
        max_rounds=args.max_rounds,
        max_errors=args.max_errors,
        workspace_root=Path(args.workspace_root),
        refresh_baseline_version=args.refresh_baseline_version,
        runner_config=CodexRunnerConfig(
            codex_cmd=args.codex_cmd,
            timeout_sec=args.timeout_sec,
            max_samples=args.max_samples,
            sample_strategy=args.sample_strategy,
            sample_seed=args.sample_seed,
            execution_mode=args.execution_mode,
            sandbox_mode=args.sandbox_mode,
            codex_model=args.codex_model,
        ),
    )
    result = SkillAgentLoop(config=config).run()
    print(json.dumps(normalize_project_paths(result, project_root=ROOT_DIR, start=ROOT_DIR), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

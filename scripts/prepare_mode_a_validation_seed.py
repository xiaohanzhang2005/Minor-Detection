# 模块说明：
# - Mode A validation seed 的 CLI 包装。
# - 用于快速生成自迭代验收专用坏 baseline。

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).parent.parent.absolute()
sys.path.insert(0, str(ROOT_DIR))

from src.config import ROOT_DIR as PROJECT_ROOT
from src.skill_loop.validation_seed import build_mode_a_validation_payload
from src.utils.path_utils import normalize_project_paths


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create an isolated temporary bad baseline for Mode A self-iteration validation."
    )
    parser.add_argument("--source-dir", default=str(PROJECT_ROOT / "skills" / "minor-detection-v0.1.0"))
    parser.add_argument("--dataset", default=str(PROJECT_ROOT / "data" / "benchmark" / "val.jsonl"))
    parser.add_argument("--output-root", default=str(PROJECT_ROOT / "tmp" / "skill_validation" / "mode_a_bad_baselines"))
    parser.add_argument("--base-name", default="minor-detection-verifya")
    parser.add_argument("--run-tag", default="")
    parser.add_argument("--max-rounds", type=int, default=1)
    parser.add_argument("--max-samples", type=int, default=3)
    parser.add_argument(
        "--sample-strategy",
        choices=["sequential", "random", "stratified"],
        default="stratified",
    )
    parser.add_argument("--sample-seed", type=int, default=42)
    parser.add_argument("--execution-mode", choices=["sandbox", "bypass"], default="bypass")
    parser.add_argument("--codex-model", default="gpt-5.4")
    parser.add_argument("--timeout-sec", type=int, default=600)
    args = parser.parse_args()

    payload = build_mode_a_validation_payload(
        source_dir=Path(args.source_dir),
        output_root=Path(args.output_root),
        dataset_path=Path(args.dataset),
        base_name=args.base_name,
        run_tag=args.run_tag or None,
        max_rounds=args.max_rounds,
        max_samples=args.max_samples,
        sample_strategy=args.sample_strategy,
        sample_seed=args.sample_seed,
        execution_mode=args.execution_mode,
        codex_model=args.codex_model,
        timeout_sec=args.timeout_sec,
    )
    print(json.dumps(normalize_project_paths(payload, project_root=PROJECT_ROOT, start=PROJECT_ROOT), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

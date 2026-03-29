from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).parent.parent.absolute()
sys.path.insert(0, str(ROOT_DIR))

from src.trigger_eval import TriggerEvalBuildConfig, build_trigger_eval_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the trigger-eval dataset for minor-detection description optimization.")
    parser.add_argument("--benchmark-path", default=None, help="Override benchmark val.jsonl path.")
    parser.add_argument("--quotas-path", default=None, help="Override slice quota config path.")
    parser.add_argument("--predicates-path", default=None, help="Override predicate/template config path.")
    parser.add_argument("--output-dir", default=None, help="Directory for generated dataset artifacts.")
    parser.add_argument("--output-stem", default="minor_detection_trigger_eval_v1")
    parser.add_argument("--sample-seed", type=int, default=42)
    args = parser.parse_args()

    defaults = TriggerEvalBuildConfig()
    config = TriggerEvalBuildConfig(
        benchmark_path=Path(args.benchmark_path) if args.benchmark_path else defaults.benchmark_path,
        extra_sources=defaults.extra_sources,
        quotas_path=Path(args.quotas_path) if args.quotas_path else defaults.quotas_path,
        predicates_path=Path(args.predicates_path) if args.predicates_path else defaults.predicates_path,
        output_dir=Path(args.output_dir) if args.output_dir else defaults.output_dir,
        output_stem=args.output_stem,
        sample_seed=args.sample_seed,
    )
    result = build_trigger_eval_dataset(config=config)
    print(json.dumps(result["summary"], ensure_ascii=False, indent=2))
    print(json.dumps(result["paths"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

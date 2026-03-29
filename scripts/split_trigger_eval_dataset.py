from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).parent.parent.absolute()
sys.path.insert(0, str(ROOT_DIR))

from src.trigger_eval import TriggerEvalSplitConfig, split_trigger_eval_dataset
from src.utils.path_utils import normalize_project_paths


def main() -> None:
    parser = argparse.ArgumentParser(description="Split trigger-eval dataset into optimization_set and final_validation_set.")
    parser.add_argument("--dataset", default=str(ROOT_DIR / "data" / "trigger_eval" / "minor_detection_trigger_eval_v1.json"))
    parser.add_argument("--output-dir", default=str(ROOT_DIR / "data" / "trigger_eval"))
    parser.add_argument("--output-stem", default="minor_detection_trigger_eval_v1")
    parser.add_argument("--final-validation-ratio", type=float, default=0.3)
    parser.add_argument("--sample-seed", type=int, default=42)
    args = parser.parse_args()

    result = split_trigger_eval_dataset(
        config=TriggerEvalSplitConfig(
            dataset_path=Path(args.dataset),
            output_dir=Path(args.output_dir),
            output_stem=args.output_stem,
            final_validation_ratio=args.final_validation_ratio,
            sample_seed=args.sample_seed,
        )
    )
    payload = {
        "summary": result["summary"],
        "paths": result["paths"],
    }
    print(json.dumps(normalize_project_paths(payload, project_root=ROOT_DIR, start=ROOT_DIR), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

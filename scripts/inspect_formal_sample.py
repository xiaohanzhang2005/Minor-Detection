# 模块说明：
# - 按单样本走 formal runtime，便于排查 enrichment 和 skill 输出。
# - 适合人工调试，不属于主链批量 loop。

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Optional

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import BENCHMARK_TEST_PATH, BENCHMARK_TRAIN_PATH, BENCHMARK_VAL_PATH
from src.executor import ExecutorSkill
from src.models import FormalSkillOutput, formal_to_legacy_output
from src.runtime import (
    build_formal_single_session_payload,
    enrich_single_session_context,
    get_formal_executor,
)


def _to_dict(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return value.model_dump()
    if hasattr(value, "dict"):
        return value.dict()
    return value


def _resolve_dataset_path(dataset: str) -> Path:
    mapping = {
        "train": BENCHMARK_TRAIN_PATH,
        "val": BENCHMARK_VAL_PATH,
        "test": BENCHMARK_TEST_PATH,
    }
    return mapping[dataset]


def _load_sample(dataset_path: Path, sample_id: Optional[str], index: Optional[int]) -> Dict[str, Any]:
    if sample_id is None and index is None:
        index = 0

    with dataset_path.open("r", encoding="utf-8") as f:
        for line_index, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            sample = json.loads(line)
            if sample_id is not None and str(sample.get("sample_id")) == sample_id:
                return sample
            if sample_id is None and index == line_index:
                return sample

    if sample_id is not None:
        raise ValueError(f"sample_id not found: {sample_id}")
    raise ValueError(f"index out of range: {index}")


def _extract_raw_time_hint(sample: Dict[str, Any]) -> str:
    icbo_features = sample.get("icbo_features")
    if isinstance(icbo_features, dict):
        opportunity_time = icbo_features.get("opportunity_time")
        if opportunity_time:
            return str(opportunity_time)
    return ""


def inspect_sample(
    *,
    dataset: str,
    sample_id: Optional[str],
    index: Optional[int],
    user_id: Optional[str],
    top_k: int,
) -> Dict[str, Any]:
    dataset_path = _resolve_dataset_path(dataset)
    sample = _load_sample(dataset_path, sample_id=sample_id, index=index)

    conversation = sample.get("conversation")
    if not isinstance(conversation, list) or not conversation:
        raise ValueError("sample conversation is missing or empty")

    resolved_user_id = user_id or str(sample.get("user_id") or sample.get("sample_id") or "")
    request_id = str(sample.get("sample_id") or f"{dataset}_sample")

    context: Dict[str, Any] = {}
    raw_time_hint = _extract_raw_time_hint(sample)
    if raw_time_hint:
        context["raw_time_hint"] = raw_time_hint

    enriched_context = enrich_single_session_context(
        conversation,
        context=context or None,
        retrieve_top_k=top_k,
    )
    payload = build_formal_single_session_payload(
        conversation,
        user_id=resolved_user_id,
        request_id=request_id,
        source="inspect_formal_sample",
        context=enriched_context,
    )

    executor: ExecutorSkill = get_formal_executor()
    formal_output: FormalSkillOutput = executor.run_formal_payload(payload)
    legacy_output = formal_to_legacy_output(formal_output)

    return {
        "dataset": dataset,
        "dataset_path": str(dataset_path),
        "sample_id": request_id,
        "ground_truth_is_minor": bool(sample.get("is_minor", False)),
        "source": sample.get("source"),
        "raw_sample": sample,
        "runtime_context": {
            "raw_time_hint": raw_time_hint,
            "time_features": enriched_context.get("time_features", {}),
            "retrieved_cases": enriched_context.get("retrieved_cases", []),
            "formal_runtime": enriched_context.get("_formal_runtime", {}),
        },
        "formal_payload": _to_dict(payload),
        "formal_output": _to_dict(formal_output),
        "legacy_output": _to_dict(legacy_output),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect one benchmark sample with the formal minor-detection skill.")
    parser.add_argument("--dataset", choices=["train", "val", "test"], default="val", help="Benchmark split.")
    parser.add_argument("--sample-id", help="Exact sample_id to inspect.")
    parser.add_argument("--index", type=int, help="0-based line index when sample-id is not provided.")
    parser.add_argument("--user-id", help="Optional override for payload.meta.user_id.")
    parser.add_argument("--top-k", type=int, default=3, help="Internal retrieval top-k.")
    parser.add_argument("--output", help="Optional path to save the JSON result.")
    args = parser.parse_args()

    result = inspect_sample(
        dataset=args.dataset,
        sample_id=args.sample_id,
        index=args.index,
        user_id=args.user_id,
        top_k=args.top_k,
    )
    result_json = json.dumps(result, ensure_ascii=False, indent=2)

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(result_json, encoding="utf-8")
        print(f"[OK] Saved inspection result to {output_path}")
    else:
        print(result_json)


if __name__ == "__main__":
    main()

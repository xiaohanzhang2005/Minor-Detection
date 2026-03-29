from __future__ import annotations

import json
import math
import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from src.config import DATA_DIR


def _json_dump(payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=False, indent=2)


def _load_dataset(path: Path) -> Dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, dict) or not isinstance(payload.get("samples"), list):
        raise ValueError(f"Unsupported trigger dataset format: {path}")
    return payload


def _stratum_key(sample: Dict[str, Any]) -> Tuple[str, bool, str]:
    return (
        str(sample.get("scenario", "unknown") or "unknown"),
        bool(sample.get("should_trigger", False)),
        str(sample.get("slice", "unknown") or "unknown"),
    )


def _allocate_validation_count(count: int, ratio: float) -> int:
    if count <= 1:
        return 0
    raw = count * ratio
    target = int(math.floor(raw))
    if raw - target >= 0.5:
        target += 1
    target = max(1, min(count - 1, target))
    return target


def _counter_dict(samples: Sequence[Dict[str, Any]], key: str) -> Dict[str, int]:
    return dict(Counter(str(sample.get(key, "")) for sample in samples))


@dataclass
class TriggerEvalSplitConfig:
    dataset_path: Path = DATA_DIR / "trigger_eval" / "minor_detection_trigger_eval_v1.json"
    output_dir: Path = DATA_DIR / "trigger_eval"
    output_stem: str = "minor_detection_trigger_eval_v1"
    final_validation_ratio: float = 0.3
    sample_seed: int = 42


class TriggerEvalDatasetSplitter:
    def __init__(self, *, config: Optional[TriggerEvalSplitConfig] = None):
        self.config = config or TriggerEvalSplitConfig()
        self.random = random.Random(self.config.sample_seed)

    def split(self) -> Dict[str, Any]:
        payload = _load_dataset(self.config.dataset_path)
        metadata = dict(payload.get("metadata") or {})
        samples = [dict(sample) for sample in payload.get("samples") or [] if isinstance(sample, dict)]
        if not samples:
            raise ValueError(f"No samples found in trigger dataset: {self.config.dataset_path}")

        buckets: Dict[Tuple[str, bool, str], List[Dict[str, Any]]] = defaultdict(list)
        for sample in samples:
            buckets[_stratum_key(sample)].append(sample)

        optimization_samples: List[Dict[str, Any]] = []
        final_validation_samples: List[Dict[str, Any]] = []
        stratum_counts: Dict[str, Dict[str, int]] = {}

        for stratum in sorted(buckets.keys()):
            bucket = list(buckets[stratum])
            self.random.shuffle(bucket)
            final_validation_count = _allocate_validation_count(len(bucket), self.config.final_validation_ratio)
            final_bucket = bucket[:final_validation_count]
            optimization_bucket = bucket[final_validation_count:]
            if not optimization_bucket:
                raise ValueError(f"Split produced empty optimization_set for stratum={stratum}")
            optimization_samples.extend(optimization_bucket)
            final_validation_samples.extend(final_bucket)
            stratum_counts["::".join([stratum[0], "trigger" if stratum[1] else "no_trigger", stratum[2]])] = {
                "total": len(bucket),
                "optimization_set": len(optimization_bucket),
                "final_validation_set": len(final_bucket),
            }

        optimization_samples.sort(key=lambda sample: str(sample.get("id", "")))
        final_validation_samples.sort(key=lambda sample: str(sample.get("id", "")))

        return {
            "metadata": {
                **metadata,
                "parent_dataset_path": str(self.config.dataset_path),
                "split_seed": self.config.sample_seed,
                "final_validation_ratio": self.config.final_validation_ratio,
            },
            "optimization_set": optimization_samples,
            "final_validation_set": final_validation_samples,
            "summary": {
                "parent_sample_count": len(samples),
                "optimization_set_count": len(optimization_samples),
                "final_validation_set_count": len(final_validation_samples),
                "optimization_set_scenario_counts": _counter_dict(optimization_samples, "scenario"),
                "final_validation_set_scenario_counts": _counter_dict(final_validation_samples, "scenario"),
                "optimization_set_slice_counts": _counter_dict(optimization_samples, "slice"),
                "final_validation_set_slice_counts": _counter_dict(final_validation_samples, "slice"),
                "stratum_counts": stratum_counts,
            },
        }

    def write_outputs(self, result: Dict[str, Any]) -> Dict[str, str]:
        self.config.output_dir.mkdir(parents=True, exist_ok=True)
        optimization_path = self.config.output_dir / f"{self.config.output_stem}_optimization_set.json"
        final_validation_path = self.config.output_dir / f"{self.config.output_stem}_final_validation_set.json"
        summary_path = self.config.output_dir / f"{self.config.output_stem}_split_summary.json"

        optimization_payload = {
            "metadata": {
                **(result.get("metadata") or {}),
                "split_name": "optimization_set",
            },
            "samples": result["optimization_set"],
        }
        final_validation_payload = {
            "metadata": {
                **(result.get("metadata") or {}),
                "split_name": "final_validation_set",
            },
            "samples": result["final_validation_set"],
        }

        optimization_path.write_text(_json_dump(optimization_payload), encoding="utf-8")
        final_validation_path.write_text(_json_dump(final_validation_payload), encoding="utf-8")
        summary_path.write_text(_json_dump(result["summary"]), encoding="utf-8")
        return {
            "optimization_set_path": str(optimization_path),
            "final_validation_set_path": str(final_validation_path),
            "summary_path": str(summary_path),
        }


def split_trigger_eval_dataset(*, config: Optional[TriggerEvalSplitConfig] = None) -> Dict[str, Any]:
    splitter = TriggerEvalDatasetSplitter(config=config)
    result = splitter.split()
    result["paths"] = splitter.write_outputs(result)
    return result

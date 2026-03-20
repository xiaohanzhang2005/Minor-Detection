"""Prepare benchmark train/val/test splits with age-bucket stratification."""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT_DIR = Path(__file__).parent.parent.absolute()
sys.path.insert(0, str(ROOT_DIR))

from src.utils.path_utils import normalize_project_paths, to_relative_posix_path


@dataclass
class BenchmarkSample:
    sample_id: str
    source: str
    is_minor: bool
    conversation: List[Dict[str, str]]
    icbo_features: Dict[str, Any]
    user_persona: Dict[str, Any]
    extra_info: Optional[Dict[str, Any]] = None


DATA_SOURCES = [
    {
        "name": "youth_social",
        "tag": "social_pos",
        "path": "社交问答/youth_dialogs.jsonl",
    },
    {
        "name": "adult_social",
        "tag": "social_neg",
        "path": "社交问答/adult_dialogs.jsonl",
    },
    {
        "name": "youth_knowledge",
        "tag": "knowledge_pos",
        "path": "知识问答数据库/youth_knowledge_qa.jsonl",
    },
    {
        "name": "adult_knowledge",
        "tag": "knowledge_neg",
        "path": "知识问答数据库/adult_knowledge_qa.jsonl",
    },
]

STRATIFICATION_STRATEGY = "source_age_bucket_v1"
MINOR_AGE_BUCKETS = [
    ("minor_07_12", 7, 12),
    ("minor_13_15", 13, 15),
    ("minor_16_17", 16, 17),
]
ADULT_AGE_BUCKETS = [
    ("adult_18_22", 18, 22),
    ("adult_23_26", 23, 26),
    ("adult_27_plus", 27, None),
]


class DataPreparer:
    """Build benchmark splits from the four raw datasets."""

    def __init__(
        self,
        output_dir: Optional[str] = None,
        train_ratio: float = 0.7,
        val_ratio: float = 0.1,
        test_ratio: float = 0.2,
        random_seed: int = 42,
    ):
        self.root_dir = ROOT_DIR
        self.data_dir = self.root_dir / "data"
        self.output_dir = Path(output_dir) if output_dir else self.data_dir / "benchmark"
        self.output_dir.mkdir(parents=True, exist_ok=True)

        ratio_sum = train_ratio + val_ratio + test_ratio
        if abs(ratio_sum - 1.0) > 0.001:
            raise ValueError("train/val/test ratios must sum to 1.0")
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio

        self.random_seed = random_seed
        self.random = random.Random(random_seed)
        self.samples: List[BenchmarkSample] = []

    @staticmethod
    def _coerce_age(age: Any) -> Optional[int]:
        if isinstance(age, bool):
            return None
        if isinstance(age, (int, float)):
            return int(age)
        if isinstance(age, str):
            stripped = age.strip()
            return int(stripped) if stripped.isdigit() else None
        return None

    @classmethod
    def resolve_age(cls, user_persona: Dict[str, Any], extra: Optional[Dict[str, Any]] = None) -> Optional[int]:
        extra = extra or {}
        return cls._coerce_age(user_persona.get("age", extra.get("_age")))

    @classmethod
    def resolve_age_bucket(cls, age: Optional[int], is_minor: bool) -> str:
        buckets = MINOR_AGE_BUCKETS if is_minor else ADULT_AGE_BUCKETS
        for bucket_name, lower, upper in buckets:
            if age is None:
                break
            if upper is None:
                if age >= lower:
                    return bucket_name
            elif lower <= age <= upper:
                return bucket_name
        return "minor_unknown" if is_minor else "adult_unknown"

    @classmethod
    def resolve_age_bucket_from_record(cls, record: Dict[str, Any]) -> str:
        user_persona = record.get("user_persona", {}) or {}
        extra = {}
        if isinstance(record.get("extra_info"), dict):
            extra.update(record["extra_info"])
        if isinstance(record.get("_meta"), dict):
            extra.update(record["_meta"])
        age = cls.resolve_age(user_persona, extra=extra)
        return cls.resolve_age_bucket(age, bool(record.get("is_minor", False)))

    def _sample_age_bucket(self, sample: BenchmarkSample) -> str:
        age = self.resolve_age(sample.user_persona, extra=sample.extra_info)
        return self.resolve_age_bucket(age, sample.is_minor)

    def _convert(self, data: Dict[str, Any], source_tag: str, index: int) -> Optional[BenchmarkSample]:
        try:
            raw_sample_id = str(data.get("dataset_id") or "").strip()

            is_minor = data.get("is_minor")
            user_persona = dict(data.get("user_persona", {}) or {})
            meta = dict(data.get("_meta", {}) or {})

            if meta:
                user_persona.setdefault("grade", meta.get("_grade_label"))
                user_persona.setdefault("subject", meta.get("_subject"))
                user_persona.setdefault("stage", meta.get("_stage"))

            if is_minor is None:
                age = self.resolve_age(user_persona, extra=meta)
                is_minor = age < 18 if age is not None else True

            extra = dict(data.get("extra_info", {}) or {})
            if meta:
                extra.setdefault("_stage", meta.get("_stage"))
                extra.setdefault("_subject", meta.get("_subject"))
                extra.setdefault("_grade", meta.get("_grade"))
                extra.setdefault("_grade_label", meta.get("_grade_label"))
                extra.setdefault("_age", meta.get("_age"))
                extra.setdefault("_seed", meta.get("_seed"))
            if raw_sample_id:
                extra["raw_sample_id"] = raw_sample_id

            sample_id = f"{source_tag}_{index:06d}"
            if raw_sample_id:
                sample_id = f"{sample_id}_{raw_sample_id.replace(' ', '_')}"

            return BenchmarkSample(
                sample_id=sample_id,
                source=source_tag,
                is_minor=bool(is_minor),
                conversation=list(data.get("conversation", []) or []),
                icbo_features=dict(data.get("icbo_features", {}) or {}),
                user_persona=user_persona,
                extra_info=extra or None,
            )
        except Exception as exc:
            print(f"[WARN] Failed to convert sample from {source_tag}#{index}: {exc}")
            return None

    def _load_jsonl(self, path: Path, source_tag: str) -> List[BenchmarkSample]:
        if not path.exists():
            print(f"[WARN] Missing data file: {path}")
            return []

        loaded: List[BenchmarkSample] = []
        with open(path, "r", encoding="utf-8") as f:
            for index, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                except Exception as exc:
                    print(f"[WARN] Failed to parse {path.name} line {index + 1}: {exc}")
                    continue
                sample = self._convert(data, source_tag, index)
                if sample is not None:
                    loaded.append(sample)
        return loaded

    def _allocate_counts(self, sizes: Dict[str, int], total_target: int) -> Dict[str, int]:
        if total_target >= sum(sizes.values()):
            return dict(sizes)
        if total_target <= 0:
            return {key: 0 for key in sizes}

        allocations = {key: 0 for key in sizes}
        total_size = sum(sizes.values())
        raw = {key: total_target * size / total_size for key, size in sizes.items()}

        remainders = []
        assigned = 0
        for key, value in raw.items():
            base = min(sizes[key], math.floor(value))
            allocations[key] = base
            assigned += base
            remainders.append((value - base, sizes[key] - base, key))

        remaining = total_target - assigned
        remainders.sort(key=lambda item: (-item[0], -sizes[item[2]], item[2]))

        idx = 0
        while remaining > 0 and remainders:
            _, capacity, key = remainders[idx % len(remainders)]
            if capacity > 0 and allocations[key] < sizes[key]:
                allocations[key] += 1
                remaining -= 1
            idx += 1
            if idx > len(remainders) * (total_target + 1):
                break

        return allocations

    def _stratified_subsample(
        self,
        samples: List[BenchmarkSample],
        total_target: Optional[int],
    ) -> List[BenchmarkSample]:
        if total_target is None or total_target >= len(samples):
            copied = list(samples)
            self.random.shuffle(copied)
            return copied

        groups: Dict[str, List[BenchmarkSample]] = defaultdict(list)
        for sample in samples:
            groups[self._sample_age_bucket(sample)].append(sample)

        allocations = self._allocate_counts(
            {bucket: len(items) for bucket, items in groups.items()},
            total_target,
        )

        selected: List[BenchmarkSample] = []
        for bucket, items in groups.items():
            bucket_items = list(items)
            self.random.shuffle(bucket_items)
            selected.extend(bucket_items[:allocations[bucket]])

        self.random.shuffle(selected)
        return selected

    def _split_counts(self, count: int) -> Dict[str, int]:
        raw = {
            "train": count * self.train_ratio,
            "val": count * self.val_ratio,
            "test": count * self.test_ratio,
        }
        base = {name: math.floor(value) for name, value in raw.items()}
        remaining = count - sum(base.values())
        fractions = sorted(
            ((raw[name] - base[name], name) for name in ["train", "val", "test"]),
            key=lambda item: (-item[0], item[1]),
        )
        for i in range(remaining):
            _, name = fractions[i % len(fractions)]
            base[name] += 1
        return base

    def _partition_by_strata(
        self,
        samples: List[BenchmarkSample],
        first_target: int,
    ) -> tuple[List[BenchmarkSample], List[BenchmarkSample]]:
        groups: Dict[str, List[BenchmarkSample]] = defaultdict(list)
        for sample in samples:
            groups[f"{sample.source}|{self._sample_age_bucket(sample)}"].append(sample)

        allocations = self._allocate_counts(
            {key: len(items) for key, items in groups.items()},
            first_target,
        )

        first: List[BenchmarkSample] = []
        second: List[BenchmarkSample] = []
        for key in sorted(groups):
            items = list(groups[key])
            self.random.shuffle(items)
            cut = allocations[key]
            first.extend(items[:cut])
            second.extend(items[cut:])

        self.random.shuffle(first)
        self.random.shuffle(second)
        return first, second

    def split_data(self) -> Dict[str, List[BenchmarkSample]]:
        if not self.samples:
            raise ValueError("No samples loaded; call run() first or populate samples before splitting.")

        targets = self._split_counts(len(self.samples))
        train_split, holdout_split = self._partition_by_strata(self.samples, targets["train"])
        val_split, test_split = self._partition_by_strata(holdout_split, targets["val"])
        splits: Dict[str, List[BenchmarkSample]] = {
            "train": train_split,
            "val": val_split,
            "test": test_split,
        }

        total = len(self.samples)
        print("\n[INFO] Split benchmark data with source+age_bucket stratification:")
        print(f"   total: {total}")
        for split_name, items in splits.items():
            minor_count = sum(1 for sample in items if sample.is_minor)
            adult_count = len(items) - minor_count
            print(
                f"   {split_name}: {len(items)} ({len(items) / total:.1%}) "
                f"[minor={minor_count} | adult={adult_count}]"
            )

        return splits

    def _stats_from_samples(self, samples: List[BenchmarkSample]) -> Dict[str, Any]:
        stats: Dict[str, Any] = {
            "total": len(samples),
            "by_source": dict(sorted(Counter(sample.source for sample in samples).items())),
            "by_is_minor": {
                "minor": sum(1 for sample in samples if sample.is_minor),
                "adult": sum(1 for sample in samples if not sample.is_minor),
            },
            "by_age_bucket": dict(sorted(Counter(self._sample_age_bucket(sample) for sample in samples).items())),
            "by_source_age_bucket": {},
        }

        nested: Dict[str, Counter] = defaultdict(Counter)
        for sample in samples:
            nested[sample.source][self._sample_age_bucket(sample)] += 1
        stats["by_source_age_bucket"] = {
            source: dict(sorted(counter.items()))
            for source, counter in sorted(nested.items())
        }
        return stats

    def get_statistics(self) -> Dict[str, Any]:
        return self._stats_from_samples(self.samples) if self.samples else {"total": 0}

    def _split_statistics(self, splits: Dict[str, List[BenchmarkSample]]) -> Dict[str, Any]:
        return {
            split_name: self._stats_from_samples(items)
            for split_name, items in splits.items()
        }

    def save_splits(self, splits: Dict[str, List[BenchmarkSample]]) -> Dict[str, str]:
        paths: Dict[str, str] = {}
        for split_name, split_samples in splits.items():
            output_path = self.output_dir / f"{split_name}.jsonl"
            with open(output_path, "w", encoding="utf-8") as f:
                for sample in split_samples:
                    payload = {
                        "sample_id": sample.sample_id,
                        "source": sample.source,
                        "is_minor": sample.is_minor,
                        "age_bucket": self._sample_age_bucket(sample),
                        "conversation": sample.conversation,
                        "icbo_features": sample.icbo_features,
                        "user_persona": sample.user_persona,
                    }
                    if sample.extra_info:
                        payload["extra_info"] = sample.extra_info
                    f.write(json.dumps(payload, ensure_ascii=False) + "\n")
            paths[split_name] = to_relative_posix_path(output_path, self.output_dir)
            print(f"   [OK] {split_name}.jsonl: {len(split_samples)}")
        return paths

    def _build_manifest(
        self,
        mode: str,
        quick_n: Optional[int],
        statistics: Dict[str, Any],
        splits: Dict[str, List[BenchmarkSample]],
        paths: Dict[str, str],
    ) -> Dict[str, Any]:
        split_sizes = {name: len(items) for name, items in splits.items()}
        manifest = {
            "timestamp": datetime.now().isoformat(),
            "mode": mode,
            "quick_n": quick_n,
            "seed": self.random_seed,
            "ratios": {
                "train": self.train_ratio,
                "val": self.val_ratio,
                "test": self.test_ratio,
            },
            "stratification": {
                "strategy": STRATIFICATION_STRATEGY,
                "unit": "source+age_bucket",
                "minor_age_buckets": [
                    {"name": name, "min_age": lower, "max_age": upper}
                    for name, lower, upper in MINOR_AGE_BUCKETS
                ],
                "adult_age_buckets": [
                    {"name": name, "min_age": lower, "max_age": upper}
                    for name, lower, upper in ADULT_AGE_BUCKETS
                ],
            },
            "statistics": statistics,
            "splits": split_sizes,
            "split_statistics": self._split_statistics(splits),
            "paths": paths,
        }
        return normalize_project_paths(manifest, project_root=self.root_dir, start=self.output_dir)

    def save_manifest(self, manifest: Dict[str, Any]) -> Path:
        manifest_path = self.output_dir / "manifest.json"
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, ensure_ascii=False, indent=2)
        return manifest_path

    def run(self, max_per_source: Optional[int] = None, save: bool = True) -> Dict[str, Any]:
        mode = "full" if max_per_source is None else "quick"
        print("=" * 60)
        print("[DATA] Prepare benchmark data")
        print(f"   mode: {mode}")
        if max_per_source is not None:
            print(f"   quick_n per source: {max_per_source}")
        print("=" * 60)

        self.samples = []
        per_source_loaded: Dict[str, int] = {}
        per_source_selected: Dict[str, int] = {}

        print("\n[INFO] Loading raw datasets...")
        for source in DATA_SOURCES:
            path = self.data_dir / source["path"]
            loaded = self._load_jsonl(path, source["tag"])
            per_source_loaded[source["tag"]] = len(loaded)
            selected = self._stratified_subsample(loaded, max_per_source)
            per_source_selected[source["tag"]] = len(selected)
            self.samples.extend(selected)
            print(
                f"   {source['name']} ({source['tag']}): "
                f"loaded={len(loaded)} selected={len(selected)}"
            )

        if not self.samples:
            return {"success": False, "error": "No samples loaded from the raw datasets."}

        statistics = self.get_statistics()
        print(
            f"\n[INFO] Total selected: {statistics['total']} "
            f"(minor={statistics['by_is_minor']['minor']} / adult={statistics['by_is_minor']['adult']})"
        )
        for source_tag, count in sorted(statistics["by_source"].items()):
            print(f"   {source_tag}: {count}")

        splits = self.split_data()
        paths: Dict[str, str] = {}
        if save:
            print("\n[INFO] Saving split files...")
            paths = self.save_splits(splits)

        manifest = self._build_manifest(
            mode=mode,
            quick_n=max_per_source,
            statistics=statistics,
            splits=splits,
            paths=paths,
        )
        manifest_path = None
        if save:
            manifest_path = self.save_manifest(manifest)
            print(f"   [OK] manifest.json: {manifest_path}")

        return {
            "success": True,
            "mode": mode,
            "statistics": statistics,
            "splits": {name: len(items) for name, items in splits.items()},
            "split_statistics": self._split_statistics(splits),
            "paths": paths,
            "per_source_loaded": per_source_loaded,
            "per_source_selected": per_source_selected,
            "stratification": manifest["stratification"],
            "manifest": manifest,
            "manifest_path": to_relative_posix_path(manifest_path, self.root_dir) if manifest_path else None,
        }

    def cleanup(self):
        for fn in ["train.jsonl", "val.jsonl", "test.jsonl", "manifest.json"]:
            fp = self.output_dir / fn
            if fp.exists():
                fp.unlink()
                print(f"[DEL] Deleted: {fp}")

        for fp in [
            self.data_dir / "retrieval_db" / "index.pkl",
            self.data_dir / "retrieval_db" / "manifest.json",
            self.data_dir / "retrieval_corpus" / "cases_v1.jsonl",
        ]:
            if fp.exists():
                fp.unlink()
                print(f"[DEL] Deleted: {fp}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare benchmark train/val/test splits.")
    parser.add_argument("--full", action="store_true", help="Use all samples from each source.")
    parser.add_argument("--quick-n", type=int, default=50, help="Per-source cap for quick runs.")
    parser.add_argument("--cleanup", action="store_true", help="Delete benchmark and retrieval artifacts.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for sampling and splitting.")
    parser.add_argument("--train-ratio", type=float, default=0.7, help="Train split ratio.")
    parser.add_argument("--val-ratio", type=float, default=0.1, help="Validation split ratio.")
    parser.add_argument("--test-ratio", type=float, default=0.2, help="Test split ratio.")
    args = parser.parse_args()

    preparer = DataPreparer(
        random_seed=args.seed,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
    )

    if args.cleanup:
        preparer.cleanup()
    else:
        max_per_source = None if args.full else args.quick_n
        result = preparer.run(max_per_source=max_per_source, save=True)
        if result.get("success"):
            print("\n" + "=" * 60)
            print("[OK] Benchmark preparation finished")
            print(f"   output_dir: {preparer.output_dir}")
            print("=" * 60)

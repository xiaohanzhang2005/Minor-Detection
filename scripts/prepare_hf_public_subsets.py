#!/usr/bin/env python3
"""Prepare public Hugging Face subsets from internal JSONL files.

This script creates cleaned public-facing JSONL files for the first
Hugging Face release. The default policy is conservative:

- remove internal provenance fields such as `_meta`, `extra_info`, `seed_id`
- normalize the primary identifier to `sample_id`
- keep only the fields that are useful for public benchmark release
- add a stable `label` field (`minor` / `adult`)
- add a simple `domain` field (`knowledge` / `social`)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable


ROOT = Path(__file__).resolve().parents[1]

DEFAULT_INPUTS = {
    "knowledge_minor": ROOT / "data" / "知识问答数据库" / "youth_knowledge_qa.jsonl",
    "knowledge_adult": ROOT / "data" / "知识问答数据库" / "adult_knowledge_qa.jsonl",
    "social_minor": ROOT / "data" / "社交问答" / "youth_dialogs.jsonl",
    "social_adult": ROOT / "data" / "社交问答" / "adult_dialogs.jsonl",
}

DEFAULT_OUTPUTS = {
    "knowledge_minor": ROOT / "release" / "huggingface" / "knowledge_subset" / "knowledge_minor.jsonl",
    "knowledge_adult": ROOT / "release" / "huggingface" / "knowledge_subset" / "knowledge_adult.jsonl",
    "social_minor": ROOT / "release" / "huggingface" / "social_subset" / "social_minor.jsonl",
    "social_adult": ROOT / "release" / "huggingface" / "social_subset" / "social_adult.jsonl",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare cleaned public JSONL subsets for Hugging Face.")
    parser.add_argument(
        "--include-exact-age",
        action="store_true",
        help="Keep exact numeric age in user_profile when present. Off by default for a more conservative public release.",
    )
    return parser.parse_args()


def iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def normalize_user_profile(raw: Dict[str, Any], *, include_exact_age: bool) -> Dict[str, Any]:
    if not isinstance(raw, dict):
        return {}

    profile: Dict[str, Any] = {}

    if include_exact_age and raw.get("age") is not None:
        profile["age"] = raw.get("age")

    for src_key, dst_key in (
        ("gender", "gender"),
        ("grade", "grade"),
        ("identity", "identity"),
        ("education_stage", "education_stage"),
        ("identity_markers", "identity_markers"),
        ("age_range", "age_range"),
    ):
        value = raw.get(src_key)
        if value is not None and value != "":
            profile[dst_key] = value

    return profile


def infer_domain(name: str) -> str:
    return "knowledge" if name.startswith("knowledge") else "social"


def clean_record(
    record: Dict[str, Any],
    *,
    name: str,
    public_sample_id: str,
    include_exact_age: bool,
) -> Dict[str, Any]:
    is_minor = bool(record.get("is_minor"))
    cleaned: Dict[str, Any] = {
        "sample_id": public_sample_id,
        "label": "minor" if is_minor else "adult",
        "is_minor": is_minor,
        "domain": infer_domain(name),
        "conversation": record.get("conversation", []),
    }

    icbo = record.get("icbo_features")
    if isinstance(icbo, dict) and icbo:
        cleaned["icbo_features"] = icbo

    user_profile = normalize_user_profile(
        record.get("user_persona", {}),
        include_exact_age=include_exact_age,
    )
    if user_profile:
        cleaned["user_profile"] = user_profile

    return cleaned


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    return count


def main() -> None:
    args = parse_args()
    summary = []

    for name, input_path in DEFAULT_INPUTS.items():
        output_path = DEFAULT_OUTPUTS[name]
        def _rows() -> Iterable[Dict[str, Any]]:
            for idx, row in enumerate(iter_jsonl(input_path), start=1):
                public_sample_id = f"{name}_{idx:06d}"
                yield clean_record(
                    row,
                    name=name,
                    public_sample_id=public_sample_id,
                    include_exact_age=args.include_exact_age,
                )

        cleaned_rows = _rows()
        count = write_jsonl(output_path, cleaned_rows)
        summary.append(
            {
                "name": name,
                "input": str(input_path.relative_to(ROOT)),
                "output": str(output_path.relative_to(ROOT)),
                "count": count,
            }
        )

    print(json.dumps({"status": "ok", "outputs": summary}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

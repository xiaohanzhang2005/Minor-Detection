from __future__ import annotations

import json
import math
import random
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from src.config import BENCHMARK_VAL_PATH, DATA_DIR, ROOT_DIR


MINOR_AGE_PATTERN = re.compile(r"(?<!\d)(?:[7-9]|1[0-7])岁")
ADULT_AGE_PATTERN = re.compile(r"(?<!\d)(?:18|19|2\d|3\d)岁")


def _json_dump(payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=False, indent=2)


def _turns_text(turns: Sequence[Dict[str, Any]]) -> str:
    return "\n".join(str(turn.get("content", "") or "").strip() for turn in turns if str(turn.get("content", "") or "").strip())


def _format_turns(turns: Sequence[Dict[str, Any]]) -> str:
    lines: List[str] = []
    for index, turn in enumerate(turns, 1):
        role = "用户" if str(turn.get("role", "user")) == "user" else "AI"
        content = str(turn.get("content", "") or "").strip()
        if content:
            lines.append(f"{index}. {role}: {content}")
    return "\n".join(lines)


def _count_user_turns(turns: Sequence[Dict[str, Any]]) -> int:
    return sum(1 for turn in turns if str(turn.get("role", "user")) == "user")


def _assistant_share(turns: Sequence[Dict[str, Any]]) -> float:
    total = len(turns)
    if total <= 0:
        return 1.0
    assistant_turns = sum(1 for turn in turns if str(turn.get("role", "user")) != "user")
    return assistant_turns / total


def _normalized_query_key(query: str) -> str:
    text = str(query or "").strip().lower()
    text = re.sub(r"\s+", " ", text)
    return text


def _query_similarity(left: str, right: str) -> float:
    return SequenceMatcher(None, _normalized_query_key(left), _normalized_query_key(right)).ratio()


def _keyword_hits(text: str, keywords: Sequence[str]) -> List[str]:
    hits = [keyword for keyword in keywords if keyword and keyword in text]
    return sorted(set(hits), key=hits.index)


def _sample_id_from_row(row: Dict[str, Any], *, fallback_prefix: str) -> str:
    return str(
        row.get("sample_id")
        or row.get("dataset_id")
        or row.get("id")
        or f"{fallback_prefix}-{hash(_turns_text(row.get('conversation') or [])) & 0xfffffff:x}"
    )


def _load_json_file(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8-sig") as f:
        return json.load(f)


def _load_jsonl_rows(path: Path, *, source_name: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8-sig") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                continue
            row = dict(row)
            row.setdefault("conversation", row.get("messages") or [])
            row.setdefault("source", source_name)
            row.setdefault("_source_name", source_name)
            row.setdefault("_source_path", str(path))
            rows.append(row)
    return rows


@dataclass
class TriggerEvalBuildConfig:
    benchmark_path: Path = BENCHMARK_VAL_PATH
    extra_sources: Tuple[Path, ...] = (
        DATA_DIR / "社交问答" / "youth_dialogs.jsonl",
        DATA_DIR / "社交问答" / "adult_dialogs.jsonl",
        DATA_DIR / "知识问答数据库" / "youth_knowledge_qa.jsonl",
        DATA_DIR / "知识问答数据库" / "adult_knowledge_qa.jsonl",
    )
    quotas_path: Path = DATA_DIR / "trigger_eval" / "description_v1_quotas.json"
    predicates_path: Path = DATA_DIR / "trigger_eval" / "description_v1_predicates.json"
    output_dir: Path = DATA_DIR / "trigger_eval"
    output_stem: str = "minor_detection_trigger_eval_v1"
    sample_seed: int = 42
    window_size_turns_candidates: Tuple[int, ...] = (6, 8, 12)
    stride_turns: int = 2
    min_user_turns: int = 3
    max_assistant_share: float = 0.6
    target_window_scan_val_ratio: float = 0.85
    max_samples_per_base: int = 2
    max_same_base_query_similarity: float = 0.88
    minimum_window_size_counts: Tuple[Tuple[int, int], ...] = ((12, 7),)


class TriggerEvalDatasetBuilder:
    def __init__(self, *, config: Optional[TriggerEvalBuildConfig] = None):
        self.config = config or TriggerEvalBuildConfig()
        self.random = random.Random(self.config.sample_seed)
        self.quotas = _load_json_file(self.config.quotas_path)
        self.predicates = _load_json_file(self.config.predicates_path)
        self.keyword_groups = self.predicates.get("keyword_groups", {})
        self.slice_rules = self.predicates.get("slice_rules", {})
        self.template_groups = self.predicates.get("direct_request_templates", {})

    def _minimum_window_size_targets(self) -> Dict[int, int]:
        targets: Dict[int, int] = {}
        for window_size, minimum_count in self.config.minimum_window_size_counts:
            targets[int(window_size)] = max(0, int(minimum_count))
        return targets

    def _load_records(self) -> List[Dict[str, Any]]:
        records = _load_jsonl_rows(self.config.benchmark_path, source_name="benchmark_val")
        for path in self.config.extra_sources:
            source_name = f"fallback:{path.stem}"
            records.extend(_load_jsonl_rows(path, source_name=source_name))
        normalized: List[Dict[str, Any]] = []
        for index, row in enumerate(records):
            conversation = row.get("conversation") or []
            if not isinstance(conversation, list) or len(conversation) < 2:
                continue
            normalized.append(
                {
                    **row,
                    "sample_id": _sample_id_from_row(row, fallback_prefix=f"record-{index}"),
                    "conversation": conversation,
                    "is_minor": bool(row.get("is_minor", False)),
                    "age_bucket": str(row.get("age_bucket", "") or ""),
                }
            )
        return normalized

    def _extract_features(self, turns: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
        text = _turns_text(turns)
        features: Dict[str, Any] = {
            "text": text,
            "identity_hits": _keyword_hits(text, self.keyword_groups.get("identity_explicit", [])),
            "school_hits": _keyword_hits(text, self.keyword_groups.get("school_context", [])),
            "guardian_hits": _keyword_hits(text, self.keyword_groups.get("guardian_context", [])),
            "minor_style_hits": _keyword_hits(text, self.keyword_groups.get("minor_style", [])),
            "minor_behavior_hits": _keyword_hits(text, self.keyword_groups.get("minor_behavior", [])),
            "adult_anchor_hits": _keyword_hits(text, self.keyword_groups.get("adult_anchor", [])),
            "youth_topic_hits": _keyword_hits(text, self.keyword_groups.get("youth_topic", [])),
            "other_profile_hits": _keyword_hits(text, self.keyword_groups.get("other_profile_focus", [])),
            "analysis_hits": _keyword_hits(text, self.keyword_groups.get("analysis_not_minor", [])),
        }
        minor_age_matches = MINOR_AGE_PATTERN.findall(text)
        adult_age_matches = ADULT_AGE_PATTERN.findall(text)
        features["minor_age_hits"] = sorted(set(minor_age_matches))
        features["adult_age_hits"] = sorted(set(adult_age_matches))
        features["identity_score"] = len(features["identity_hits"]) + len(features["minor_age_hits"])
        features["school_score"] = len(features["school_hits"]) + len(features["guardian_hits"])
        features["implicit_score"] = len(features["minor_style_hits"]) + len(features["minor_behavior_hits"]) + min(1, len(features["school_hits"]))
        features["adult_score"] = len(features["adult_anchor_hits"]) + len(features["adult_age_hits"])
        features["topic_score"] = len(features["youth_topic_hits"])
        return features

    def _window_candidates(self, record: Dict[str, Any]) -> Iterable[Dict[str, Any]]:
        turns = record.get("conversation") or []
        total_turns = len(turns)
        full_features = self._extract_features(turns)
        for window_size in self.config.window_size_turns_candidates:
            if total_turns < window_size:
                continue
            for end_index in range(window_size, total_turns + 1, self.config.stride_turns):
                window_turns = turns[end_index - window_size : end_index]
                if _count_user_turns(window_turns) < self.config.min_user_turns:
                    continue
                if _assistant_share(window_turns) > self.config.max_assistant_share:
                    continue
                features = self._extract_features(window_turns)
                yield {
                    "candidate_id": f"{record['sample_id']}::w{window_size}::e{end_index}",
                    "base_sample_id": record["sample_id"],
                    "record": record,
                    "source": str(record.get("source", "unknown")),
                    "is_minor": bool(record.get("is_minor", False)),
                    "window_size_turns": window_size,
                    "window_end_turn": end_index,
                    "window_turns": window_turns,
                    "features": features,
                    "full_features": full_features,
                    "is_from_val": str(record.get("_source_name", "")) == "benchmark_val",
                }

    def _rule_matches(self, slice_name: str, candidate: Dict[str, Any]) -> bool:
        rule = self.slice_rules.get(slice_name, {})
        features = candidate["features"]
        full_features = candidate["full_features"]
        text = features["text"]

        if slice_name == "identity_explicit":
            return (
                candidate["is_minor"]
                and features["adult_score"] == 0
                and features["identity_score"] >= 1
                and (features["school_score"] >= 1 or features["implicit_score"] >= 1)
            )

        if slice_name == "school_context_strong":
            return (
                candidate["is_minor"]
                and features["adult_score"] == 0
                and features["identity_score"] == 0
                and features["school_score"] >= int(rule.get("min_school_score", 3))
                and (features["guardian_hits"] or features["minor_behavior_hits"] or features["minor_style_hits"])
            )

        if slice_name == "implicit_minor_signal":
            return (
                candidate["is_minor"]
                and features["adult_score"] == 0
                and features["identity_score"] == 0
                and features["implicit_score"] >= int(rule.get("min_implicit_score", 3))
            )

        if slice_name == "adult_near_miss":
            return (
                not candidate["is_minor"]
                and features["adult_score"] >= int(rule.get("min_adult_score", 1))
                and (features["school_score"] >= 1 or features["topic_score"] >= 1 or "高中同学" in text or "学校" in text)
            )

        if slice_name == "topic_adjacent_not_identity":
            return (
                not candidate["is_minor"]
                and features["topic_score"] >= int(rule.get("min_topic_score", 1))
                and features["identity_score"] == 0
            )

        if slice_name == "insufficient_evidence_window":
            return (
                candidate["window_end_turn"] < len(candidate["record"].get("conversation") or [])
                and features["identity_score"] == 0
                and features["adult_score"] == 0
                and features["school_score"] <= int(rule.get("max_window_school_score", 1))
                and (full_features["identity_score"] + full_features["school_score"] + full_features["adult_score"]) >= int(rule.get("min_full_signal_score", 3))
            )

        return False

    def _candidate_score(self, slice_name: str, candidate: Dict[str, Any]) -> float:
        features = candidate["features"]
        full_features = candidate["full_features"]
        score = 0.0
        minimum_window_size_targets = self._minimum_window_size_targets()
        if candidate["is_from_val"]:
            score += 100.0
        score += min(candidate["window_size_turns"], 12) * 0.1
        if candidate["window_size_turns"] == max(self.config.window_size_turns_candidates):
            score += 25.0
        if candidate["window_size_turns"] in minimum_window_size_targets:
            score += 300.0

        if slice_name == "identity_explicit":
            score += features["identity_score"] * 10 + features["school_score"] * 2 + features["implicit_score"]
        elif slice_name == "school_context_strong":
            score += features["school_score"] * 8 + len(features["guardian_hits"]) * 3 + len(features["minor_behavior_hits"]) * 2
        elif slice_name == "implicit_minor_signal":
            score += features["implicit_score"] * 8 + len(features["minor_style_hits"]) * 2 + len(features["minor_behavior_hits"]) * 2
        elif slice_name == "adult_near_miss":
            score += features["adult_score"] * 8 + features["school_score"] * 2 + features["topic_score"] * 2
        elif slice_name == "topic_adjacent_not_identity":
            score += features["topic_score"] * 8 + len(features["analysis_hits"]) + len(features["other_profile_hits"])
        elif slice_name == "insufficient_evidence_window":
            later_signal = max(0, full_features["identity_score"] + full_features["school_score"] - features["school_score"])
            score += later_signal * 6 + max(0, 3 - features["school_score"]) * 3
        return score

    def _window_query(self, candidate: Dict[str, Any]) -> str:
        prefix = (
            f"你正在扫描当前聊天窗口。下面是最近 {candidate['window_size_turns']} 轮用户与 AI 的对话。"
            "请判断是否需要调用 `minor-detection` 做未成年人深度识别。"
            "如果当前窗口已经出现足够信号，应调用；如果信息不足或任务目标不是判断未成年人，则不要调用。"
        )
        return f"{prefix}\n\n{_format_turns(candidate['window_turns'])}"

    def _window_trigger_basis(self, slice_name: str, candidate: Dict[str, Any]) -> List[str]:
        features = candidate["features"]
        if slice_name == "identity_explicit":
            basis = features["minor_age_hits"] + features["identity_hits"] + features["school_hits"]
        elif slice_name == "school_context_strong":
            basis = features["school_hits"] + features["guardian_hits"] + features["minor_behavior_hits"] + features["minor_style_hits"]
        elif slice_name == "implicit_minor_signal":
            basis = features["minor_style_hits"] + features["minor_behavior_hits"] + features["school_hits"]
        elif slice_name == "adult_near_miss":
            basis = features["adult_age_hits"] + features["adult_anchor_hits"] + features["school_hits"] + features["youth_topic_hits"]
        elif slice_name == "topic_adjacent_not_identity":
            basis = features["youth_topic_hits"] + features["analysis_hits"] + features["other_profile_hits"]
        elif slice_name == "insufficient_evidence_window":
            later_score = candidate["full_features"]["identity_score"] + candidate["full_features"]["school_score"] + candidate["full_features"]["adult_score"]
            basis = ["current_window_insufficient", f"full_dialogue_signal_score={later_score}"]
        else:
            basis = []
        if not basis:
            basis = [slice_name]
        return list(dict.fromkeys(basis))

    def _build_window_sample(self, slice_name: str, candidate: Dict[str, Any], index: int) -> Dict[str, Any]:
        rule = self.slice_rules[slice_name]
        return {
            "id": f"trigger-{slice_name}-{index:03d}",
            "scenario": "window_scan",
            "should_trigger": bool(rule["should_trigger"]),
            "slice": slice_name,
            "source": f"benchmark_val:{candidate['source']}" if candidate["is_from_val"] else candidate["source"],
            "query": self._window_query(candidate),
            "label_reason": rule["label_reason"],
            "trigger_basis": self._window_trigger_basis(slice_name, candidate),
            "generation_rule": rule["generation_rule"],
            "evidence_strength": rule["evidence_strength"],
            "window_size_turns": candidate["window_size_turns"],
            "window_end_turn": candidate["window_end_turn"],
            "window_turns": candidate["window_turns"],
            "skill_input_turns": candidate["window_turns"],
            "skill_input_base_sample_id": candidate["base_sample_id"],
            "expected_is_minor": bool(candidate["is_minor"]),
            "mined_from_dialogue_id": candidate["record"]["sample_id"],
            "base_sample_id": candidate["base_sample_id"],
        }

    def _choose_window_samples(self, buckets: Dict[str, List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
        chosen_candidates: Dict[str, Dict[str, Any]] = {}
        chosen_query_keys: set[str] = set()
        chosen_queries_by_base: Dict[str, List[str]] = defaultdict(list)
        base_counts: Counter[str] = Counter()
        samples: List[Dict[str, Any]] = []
        next_index = 1

        quotas = self.quotas.get("window_scan", {})
        for slice_name, quota in quotas.items():
            ordered = sorted(
                buckets.get(slice_name, []),
                key=lambda candidate: (
                    -self._candidate_score(slice_name, candidate),
                    candidate["candidate_id"],
                ),
            )
            selected_for_slice: List[Dict[str, Any]] = []
            for candidate in ordered:
                candidate_id = candidate["candidate_id"]
                query_text = self._window_query(candidate)
                query_key = _normalized_query_key(query_text)
                base_sample_id = candidate["base_sample_id"]
                if candidate_id in chosen_candidates:
                    continue
                if query_key in chosen_query_keys:
                    continue
                if base_counts[base_sample_id] >= self.config.max_samples_per_base:
                    continue
                if any(
                    _query_similarity(query_text, chosen_query) >= self.config.max_same_base_query_similarity
                    for chosen_query in chosen_queries_by_base[base_sample_id]
                ):
                    continue
                selected_for_slice.append(candidate)
                chosen_candidates[candidate_id] = candidate
                chosen_query_keys.add(query_key)
                chosen_queries_by_base[base_sample_id].append(query_text)
                base_counts[base_sample_id] += 1
                samples.append(self._build_window_sample(slice_name, candidate, next_index))
                next_index += 1
                if len(selected_for_slice) >= int(quota):
                    break
            if len(selected_for_slice) < int(quota):
                raise ValueError(f"Unable to satisfy window slice quota for {slice_name}: need {quota}, got {len(selected_for_slice)}")
        return samples

    def _select_direct_bindings(
        self,
        window_samples: Sequence[Dict[str, Any]],
        *,
        should_trigger: bool,
        needed: int,
        excluded_sample_ids: Optional[Sequence[str]] = None,
    ) -> List[Dict[str, Any]]:
        excluded = set(str(sample_id) for sample_id in (excluded_sample_ids or ()))
        pool = [sample for sample in window_samples if bool(sample["should_trigger"]) == should_trigger]
        available_pool = [sample for sample in pool if str(sample["id"]) not in excluded]
        if len(available_pool) < needed:
            raise ValueError(
                f"Not enough window samples to bind direct requests for should_trigger={should_trigger}: "
                f"need {needed}, got {len(available_pool)}"
            )

        grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for sample in available_pool:
            grouped[str(sample.get("slice", "unknown") or "unknown")].append(sample)

        for items in grouped.values():
            items.sort(
                key=lambda sample: (
                    0 if str(sample.get("source", "")).startswith("benchmark_val") else 1,
                    -int(sample.get("window_size_turns", 0)),
                    sample["id"],
                )
            )

        allocations = self._allocate_binding_counts(
            {slice_name: len(items) for slice_name, items in grouped.items()},
            needed,
        )

        selected: List[Dict[str, Any]] = []
        for slice_name in sorted(grouped.keys()):
            selected.extend(grouped[slice_name][: allocations.get(slice_name, 0)])
        return selected

    @staticmethod
    def _allocate_binding_counts(group_sizes: Dict[str, int], target_total: int) -> Dict[str, int]:
        if target_total <= 0:
            return {key: 0 for key in group_sizes}

        keys = sorted(group_sizes.keys())
        allocations = {key: 0 for key in keys}
        if not keys:
            return allocations

        if target_total >= len(keys):
            for key in keys:
                allocations[key] = 1
            target_total -= len(keys)

        if target_total <= 0:
            return allocations

        remaining_capacity = {key: max(0, group_sizes[key] - allocations[key]) for key in keys}
        total_capacity = sum(remaining_capacity.values())
        if total_capacity <= 0:
            return allocations

        base_additions: Dict[str, int] = {}
        remainders: List[Tuple[float, int, str]] = []
        assigned = 0
        for key in keys:
            capacity = remaining_capacity[key]
            quota = (capacity / total_capacity) * target_total if total_capacity else 0.0
            floor_value = min(capacity, math.floor(quota))
            base_additions[key] = floor_value
            assigned += floor_value
            remainders.append((quota - floor_value, capacity - floor_value, key))

        for key, floor_value in base_additions.items():
            allocations[key] += floor_value

        remaining = target_total - assigned
        remainders.sort(key=lambda item: (-item[0], -item[1], item[2]))
        for _, spare_capacity, key in remainders:
            if remaining <= 0:
                break
            if spare_capacity <= 0:
                continue
            allocations[key] += 1
            remaining -= 1

        return allocations

    def _render_template(self, template: str, bound_sample: Dict[str, Any]) -> str:
        dialogue_excerpt = _format_turns(bound_sample.get("window_turns") or [])
        return template.format(
            dialogue_excerpt=dialogue_excerpt,
            window_size_turns=bound_sample.get("window_size_turns", 0),
            slice=bound_sample.get("slice", ""),
        )

    def _build_direct_samples(self, window_samples: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
        samples: List[Dict[str, Any]] = []
        next_index = len(window_samples) + 1
        quotas = self.quotas.get("direct_request", {})
        positive_slices = {"explicit_minor_judgment_request", "operator_window_scan_request"}
        used_bound_sample_ids: set[str] = set()

        for slice_name, quota in quotas.items():
            rule = self.slice_rules[slice_name]
            templates = list(self.template_groups.get(slice_name, []))
            if not templates:
                raise ValueError(f"No direct request templates configured for {slice_name}")
            bound = self._select_direct_bindings(
                window_samples,
                should_trigger=slice_name in positive_slices,
                needed=int(quota),
                excluded_sample_ids=sorted(used_bound_sample_ids),
            )
            for offset, bound_sample in enumerate(bound):
                template = templates[offset % len(templates)]
                template_id = str(template.get("id", f"{slice_name}-{offset + 1}"))
                query = self._render_template(str(template.get("template", "")), bound_sample)
                source = "template_bound:benchmark_val" if str(bound_sample.get("source", "")).startswith("benchmark_val") else "template_bound:fallback"
                used_bound_sample_ids.add(str(bound_sample["id"]))
                samples.append(
                    {
                        "id": f"trigger-{slice_name}-{next_index:03d}",
                        "scenario": "direct_request",
                        "should_trigger": bool(rule["should_trigger"]),
                        "slice": slice_name,
                        "source": source,
                        "query": query,
                        "label_reason": rule["label_reason"],
                        "trigger_basis": list(rule.get("trigger_basis") or [slice_name]),
                        "generation_rule": rule["generation_rule"],
                        "evidence_strength": rule["evidence_strength"],
                        "request_template_id": template_id,
                        "bound_sample_ids": [bound_sample["id"]],
                        "skill_input_turns": bound_sample.get("window_turns") or [],
                        "skill_input_base_sample_id": bound_sample.get("base_sample_id"),
                        "expected_is_minor": bool(bound_sample.get("expected_is_minor", False)),
                    }
                )
                next_index += 1
        return samples

    def _validation_errors(
        self,
        samples: Sequence[Dict[str, Any]],
        *,
        available_window_sizes: Sequence[int],
    ) -> List[str]:
        errors: List[str] = []
        quotas = {
            **{("window_scan", key): int(value) for key, value in self.quotas.get("window_scan", {}).items()},
            **{("direct_request", key): int(value) for key, value in self.quotas.get("direct_request", {}).items()},
        }
        counts: Counter[Tuple[str, str]] = Counter((str(sample.get("scenario")), str(sample.get("slice"))) for sample in samples)
        for quota_key, expected in quotas.items():
            actual = counts.get(quota_key, 0)
            if actual != expected:
                errors.append(f"quota mismatch for {quota_key[0]}/{quota_key[1]}: expected {expected}, got {actual}")

        required_common = {
            "id",
            "scenario",
            "should_trigger",
            "slice",
            "source",
            "query",
            "skill_input_turns",
            "skill_input_base_sample_id",
            "expected_is_minor",
            "label_reason",
            "trigger_basis",
            "generation_rule",
            "evidence_strength",
        }
        required_window = {"window_size_turns", "window_end_turn", "window_turns", "mined_from_dialogue_id", "base_sample_id"}
        required_direct = {"request_template_id", "bound_sample_ids"}

        ids: List[str] = []
        query_keys: List[str] = []
        for sample in samples:
            missing = required_common - sample.keys()
            if sample.get("scenario") == "window_scan":
                missing |= required_window - sample.keys()
            if sample.get("scenario") == "direct_request":
                missing |= required_direct - sample.keys()
            if missing:
                errors.append(f"sample {sample.get('id', '<missing-id>')} missing fields: {sorted(missing)}")
            ids.append(str(sample.get("id")))
            query_keys.append(_normalized_query_key(str(sample.get("query", ""))))

        duplicate_ids = [item for item, count in Counter(ids).items() if count > 1]
        if duplicate_ids:
            errors.append(f"duplicate sample ids: {duplicate_ids}")

        duplicate_queries = [item for item, count in Counter(query_keys).items() if count > 1]
        if duplicate_queries:
            errors.append(f"duplicate normalized queries: {len(duplicate_queries)}")

        window_samples = [sample for sample in samples if sample.get("scenario") == "window_scan"]
        if window_samples:
            val_count = sum(1 for sample in window_samples if str(sample.get("source", "")).startswith("benchmark_val"))
            val_ratio = val_count / len(window_samples)
            if val_ratio < self.config.target_window_scan_val_ratio:
                errors.append(
                    f"window_scan benchmark ratio too low: expected >= {self.config.target_window_scan_val_ratio:.2f}, got {val_ratio:.2f}"
                )
            present_sizes = {int(sample.get("window_size_turns", 0)) for sample in window_samples}
            for size in sorted(set(int(size) for size in available_window_sizes)):
                if size not in present_sizes:
                    errors.append(f"window size {size} available in pool but absent in final dataset")
            actual_window_size_counts = Counter(int(sample.get("window_size_turns", 0)) for sample in window_samples)
            minimum_window_size_targets = self._minimum_window_size_targets()
            for size, minimum_count in sorted(minimum_window_size_targets.items()):
                if size not in set(int(item) for item in available_window_sizes):
                    continue
                actual = int(actual_window_size_counts.get(size, 0))
                if actual < minimum_count:
                    errors.append(
                        f"window size {size} under target: expected >= {minimum_count}, got {actual}"
                    )

        for sample in samples:
            slice_name = str(sample.get("slice", ""))
            if slice_name == "adult_near_miss" and bool(sample.get("should_trigger")):
                errors.append(f"adult_near_miss sample incorrectly labeled true: {sample['id']}")
            if slice_name in {"identity_explicit", "school_context_strong", "implicit_minor_signal"} and not bool(sample.get("should_trigger")):
                errors.append(f"positive slice sample incorrectly labeled false: {sample['id']}")
            if slice_name == "identity_explicit" and sample.get("scenario") == "window_scan":
                basis = set(str(item) for item in sample.get("trigger_basis", []))
                if not any(item.endswith("岁") for item in basis) and not (basis & set(self.keyword_groups.get("identity_explicit", []))):
                    errors.append(f"identity_explicit sample lacks explicit identity basis: {sample['id']}")
            if slice_name == "adult_near_miss" and sample.get("scenario") == "window_scan":
                basis = set(str(item) for item in sample.get("trigger_basis", []))
                adult_basis = set(self.keyword_groups.get("adult_anchor", [])) | {"18岁", "19岁"}
                if not (basis & adult_basis) and not any(item.endswith("岁") for item in basis):
                    errors.append(f"adult_near_miss sample lacks adult anchor basis: {sample['id']}")

        return errors

    def _summary_payload(
        self,
        samples: Sequence[Dict[str, Any]],
        *,
        available_window_sizes: Sequence[int],
        validation_errors: Sequence[str],
        warnings: Sequence[str],
    ) -> Dict[str, Any]:
        return {
            "version": str(self.quotas.get("version", "v1")),
            "sample_count": len(samples),
            "scenario_counts": dict(Counter(str(sample.get("scenario")) for sample in samples)),
            "label_counts": dict(Counter(bool(sample.get("should_trigger")) for sample in samples)),
            "slice_counts": dict(Counter(str(sample.get("slice")) for sample in samples)),
            "source_counts": dict(Counter(str(sample.get("source")) for sample in samples)),
            "window_size_counts": dict(
                Counter(int(sample.get("window_size_turns", 0)) for sample in samples if sample.get("scenario") == "window_scan")
            ),
            "available_window_sizes": sorted(set(int(size) for size in available_window_sizes)),
            "validation_errors": list(validation_errors),
            "warnings": list(warnings),
        }

    def _summary_markdown(self, summary: Dict[str, Any]) -> str:
        lines = [
            f"# Trigger Eval Dataset Summary ({summary['version']})",
            "",
            f"- sample_count: {summary['sample_count']}",
            f"- scenario_counts: {json.dumps(summary['scenario_counts'], ensure_ascii=False)}",
            f"- label_counts: {json.dumps(summary['label_counts'], ensure_ascii=False)}",
            f"- slice_counts: {json.dumps(summary['slice_counts'], ensure_ascii=False)}",
            f"- source_counts: {json.dumps(summary['source_counts'], ensure_ascii=False)}",
            f"- window_size_counts: {json.dumps(summary['window_size_counts'], ensure_ascii=False)}",
        ]
        if summary.get("warnings"):
            lines.extend(["", "## Warnings"])
            for warning in summary["warnings"]:
                lines.append(f"- {warning}")
        if summary.get("validation_errors"):
            lines.extend(["", "## Validation Errors"])
            for error in summary["validation_errors"]:
                lines.append(f"- {error}")
        else:
            lines.extend(["", "## Validation", "", "- passed"])
        return "\n".join(lines) + "\n"

    def _review_pack(self, samples: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
        by_slice: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        by_window_size: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for sample in samples:
            by_slice[str(sample.get("slice", ""))].append(sample)
            if sample.get("scenario") == "window_scan":
                by_window_size[str(sample.get("window_size_turns", 0))].append(sample)

        return {
            "slice_reviews": {slice_name: items[:3] for slice_name, items in sorted(by_slice.items())},
            "window_size_reviews": {size: items[:2] for size, items in sorted(by_window_size.items())},
        }

    def build(self) -> Dict[str, Any]:
        records = self._load_records()
        if not records:
            raise ValueError("No input records available for trigger eval dataset construction")

        window_slice_names = list(self.quotas.get("window_scan", {}).keys())
        buckets: Dict[str, List[Dict[str, Any]]] = {slice_name: [] for slice_name in window_slice_names}
        ambiguous_rejects: List[Dict[str, Any]] = []
        available_window_sizes: set[int] = set()

        for record in records:
            for candidate in self._window_candidates(record):
                available_window_sizes.add(int(candidate["window_size_turns"]))
                matched = False
                for slice_name in window_slice_names:
                    if self._rule_matches(slice_name, candidate):
                        buckets[slice_name].append(candidate)
                        matched = True
                if not matched and candidate["is_from_val"]:
                    features = candidate["features"]
                    signal_score = features["identity_score"] + features["school_score"] + features["implicit_score"] + features["adult_score"] + features["topic_score"]
                    if signal_score > 0 or candidate["window_end_turn"] < len(candidate["record"].get("conversation") or []):
                        ambiguous_rejects.append(
                            {
                                "candidate_id": candidate["candidate_id"],
                                "base_sample_id": candidate["base_sample_id"],
                                "source": f"benchmark_val:{candidate['source']}" if candidate["is_from_val"] else candidate["source"],
                                "window_size_turns": candidate["window_size_turns"],
                                "window_end_turn": candidate["window_end_turn"],
                                "feature_summary": {
                                    "identity_score": features["identity_score"],
                                    "school_score": features["school_score"],
                                    "implicit_score": features["implicit_score"],
                                    "adult_score": features["adult_score"],
                                    "topic_score": features["topic_score"],
                                },
                                "reject_reason": "matched_no_slice_rule",
                            }
                        )

        window_samples = self._choose_window_samples(buckets)
        direct_samples = self._build_direct_samples(window_samples)
        samples = window_samples + direct_samples

        warnings: List[str] = []
        requested_sizes = set(int(size) for size in self.config.window_size_turns_candidates)
        missing_from_pool = sorted(requested_sizes - available_window_sizes)
        if missing_from_pool:
            warnings.append(f"configured window sizes absent from candidate pool: {missing_from_pool}")

        validation_errors = self._validation_errors(
            samples,
            available_window_sizes=sorted(available_window_sizes & requested_sizes),
        )
        summary = self._summary_payload(
            samples,
            available_window_sizes=sorted(available_window_sizes),
            validation_errors=validation_errors,
            warnings=warnings,
        )
        review_pack = self._review_pack(samples)

        if validation_errors:
            raise ValueError("Trigger eval dataset validation failed:\n- " + "\n- ".join(validation_errors))

        return {
            "metadata": {
                "version": str(self.quotas.get("version", "v1")),
                "available_window_sizes": sorted(available_window_sizes),
                "record_count": len(records),
            },
            "samples": samples,
            "summary": summary,
            "review_pack": review_pack,
            "ambiguous_rejects": ambiguous_rejects[:200],
        }

    def write_outputs(self, result: Dict[str, Any]) -> Dict[str, str]:
        self.config.output_dir.mkdir(parents=True, exist_ok=True)
        dataset_path = self.config.output_dir / f"{self.config.output_stem}.json"
        summary_json_path = self.config.output_dir / f"{self.config.output_stem}_summary.json"
        summary_md_path = self.config.output_dir / f"{self.config.output_stem}_summary.md"
        review_pack_path = self.config.output_dir / f"{self.config.output_stem}_review_pack.json"
        ambiguous_rejects_path = self.config.output_dir / f"{self.config.output_stem}_ambiguous_rejects.jsonl"

        dataset_payload = {
            "metadata": result["metadata"],
            "samples": result["samples"],
        }
        dataset_path.write_text(_json_dump(dataset_payload), encoding="utf-8")
        summary_json_path.write_text(_json_dump(result["summary"]), encoding="utf-8")
        summary_md_path.write_text(self._summary_markdown(result["summary"]), encoding="utf-8")
        review_pack_path.write_text(_json_dump(result["review_pack"]), encoding="utf-8")
        with ambiguous_rejects_path.open("w", encoding="utf-8") as f:
            for row in result["ambiguous_rejects"]:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

        return {
            "dataset_path": str(dataset_path),
            "summary_json_path": str(summary_json_path),
            "summary_md_path": str(summary_md_path),
            "review_pack_path": str(review_pack_path),
            "ambiguous_rejects_path": str(ambiguous_rejects_path),
        }


def build_trigger_eval_dataset(*, config: Optional[TriggerEvalBuildConfig] = None) -> Dict[str, Any]:
    builder = TriggerEvalDatasetBuilder(config=config)
    result = builder.build()
    result["paths"] = builder.write_outputs(result)
    return result


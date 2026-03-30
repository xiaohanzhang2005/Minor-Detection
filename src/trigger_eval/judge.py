from __future__ import annotations

import json
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any, Dict, List, Optional, Tuple

from pydantic import ValidationError

from src.models import FormalSkillOutput
from src.skill_loop.judge import DEFAULT_PROTECTED_COUNT, calc_default_max_errors
from src.utils.path_utils import normalize_project_paths, to_relative_posix_path


REQUIRED_FIELDS = [
    "should_trigger",
    "skill_invoked",
    "decision_confidence",
    "decision_reason",
    "invocation_status",
]


def _load_json(path: Path, default: Any = None) -> Any:
    if not path.exists():
        return default
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def _write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _list_sample_dirs(run_root: Path) -> List[Path]:
    sample_dirs: List[Path] = []
    for path in sorted(run_root.iterdir()):
        if path.is_dir() and (path / "sample_input.json").exists():
            sample_dirs.append(path)
    return sample_dirs


def _missing_required_fields(parsed_json: Optional[Dict[str, Any]]) -> List[str]:
    if not isinstance(parsed_json, dict):
        return ["<invalid_json>"]
    return [field for field in REQUIRED_FIELDS if field not in parsed_json]


def _validate_trigger_output(parsed_json: Optional[Dict[str, Any]]) -> Tuple[bool, Optional[str]]:
    if not isinstance(parsed_json, dict):
        return False, "invalid_json"
    missing = _missing_required_fields(parsed_json)
    if missing:
        return False, f"missing_fields={missing}"
    if not isinstance(parsed_json.get("should_trigger"), bool):
        return False, "should_trigger_not_bool"
    if not isinstance(parsed_json.get("skill_invoked"), bool):
        return False, "skill_invoked_not_bool"
    try:
        confidence = float(parsed_json.get("decision_confidence", 0.0))
    except Exception:
        return False, "decision_confidence_not_number"
    if confidence < 0.0 or confidence > 1.0:
        return False, "decision_confidence_out_of_range"
    if not isinstance(parsed_json.get("decision_reason"), str) or not str(parsed_json.get("decision_reason", "")).strip():
        return False, "decision_reason_empty"
    if parsed_json.get("invocation_status") not in {"not_invoked", "invoked_success", "invoked_failed"}:
        return False, "invocation_status_invalid"
    return True, None


def _validate_formal_output(parsed_json: Optional[Dict[str, Any]]) -> Tuple[bool, Optional[str]]:
    if not isinstance(parsed_json, dict):
        return False, "invalid_json"
    try:
        FormalSkillOutput.model_validate(parsed_json)
    except ValidationError as exc:
        return False, str(exc)
    return True, None


def _predicted_minor_label(parsed_json: Optional[Dict[str, Any]]) -> Optional[bool]:
    if not isinstance(parsed_json, dict):
        return None
    decision = parsed_json.get("decision") or {}
    if not isinstance(decision, dict) or "is_minor" not in decision:
        return None
    return bool(decision["is_minor"])


def _coerce_returncode(metadata: Dict[str, Any]) -> int:
    raw_value = metadata.get("returncode", 1)
    try:
        return int(raw_value)
    except (TypeError, ValueError):
        return 1


def _build_outcome(sample_dir: Path) -> Dict[str, Any]:
    sample_input = _load_json(sample_dir / "sample_input.json", default={}) or {}
    gold = _load_json(sample_dir / "gold.json", default={}) or {}
    agent_output = _load_json(sample_dir / "agent_output.json", default={}) or {}
    metadata = _load_json(sample_dir / "run_metadata.json", default={}) or {}
    timing = _load_json(sample_dir / "timing.json", default={}) or {}
    observability = _load_json(sample_dir / "observability.json", default={}) or {}
    transcript = (sample_dir / "transcript.md").read_text(encoding="utf-8", errors="replace") if (sample_dir / "transcript.md").exists() else ""

    parsed_json = agent_output.get("parsed_json")
    json_valid = bool(agent_output.get("json_valid", False))
    schema_valid, schema_error = _validate_trigger_output(parsed_json)
    missing_fields = _missing_required_fields(parsed_json)

    predicted = parsed_json.get("should_trigger") if isinstance(parsed_json, dict) and "should_trigger" in parsed_json else None
    ground_truth = bool(gold.get("should_trigger", False))
    try:
        confidence = float((parsed_json or {}).get("decision_confidence", 0.0))
    except Exception:
        confidence = 0.0
    skill_invoked = (parsed_json or {}).get("skill_invoked") if isinstance(parsed_json, dict) else None
    invocation_status = str((parsed_json or {}).get("invocation_status", "") or "") if isinstance(parsed_json, dict) else ""

    launcher = observability.get("launcher") or {}
    launcher_invoked = bool(launcher.get("invoked")) or bool(metadata.get("launcher_invoked"))
    launcher_success = bool(launcher.get("success")) or bool(metadata.get("launcher_success"))

    invocation_matches_decision = True
    if predicted is True:
        invocation_matches_decision = skill_invoked is True and launcher_invoked and launcher_success and invocation_status == "invoked_success"
    elif predicted is False:
        invocation_matches_decision = skill_invoked is False and (not launcher_invoked) and invocation_status == "not_invoked"
    elif predicted is None:
        invocation_matches_decision = False

    final_returncode = _coerce_returncode(metadata)
    step_compliant = json_valid and schema_valid and not missing_fields and invocation_matches_decision
    invocation_success = final_returncode == 0 and step_compliant

    observed_issues = set(str(item) for item in (observability.get("issues") or []) if str(item).strip())
    if not invocation_matches_decision:
        observed_issues.add("trigger_invocation_mismatch")
    if launcher_invoked and not launcher_success:
        observed_issues.add("launcher_failed")
    if final_returncode != 0:
        observed_issues.add("agent_returncode_nonzero")

    failure_types: List[str] = []
    if not json_valid:
        failure_types.append("output_parse_failure")
    if json_valid and not schema_valid:
        failure_types.append("schema_invalid")
    if missing_fields and "<invalid_json>" not in missing_fields:
        failure_types.append("fields_missing")

    if predicted is True and not ground_truth:
        failure_types.append("false_positive")
    elif predicted is not True and ground_truth:
        failure_types.append("false_negative")

    if not step_compliant:
        failure_types.append("step_compliance_failure")

    is_correct = predicted == ground_truth if predicted is not None else False

    return {
        "sample_id": str(gold.get("sample_id") or sample_input.get("id") or sample_dir.name),
        "sample_dir": str(sample_dir),
        "slice": str(gold.get("slice", "") or sample_input.get("slice", "")),
        "scenario": str(gold.get("scenario", "") or sample_input.get("scenario", "")),
        "ground_truth": ground_truth,
        "predicted": predicted,
        "confidence": confidence,
        "is_correct": is_correct,
        "json_valid": json_valid,
        "schema_valid": schema_valid,
        "schema_error": schema_error,
        "missing_fields": missing_fields,
        "skill_invoked": skill_invoked,
        "launcher_invoked": launcher_invoked,
        "launcher_success": launcher_success,
        "invocation_status": invocation_status,
        "step_compliant": step_compliant,
        "invocation_success": invocation_success,
        "failure_types": list(dict.fromkeys(failure_types)),
        "observed_issues": sorted(observed_issues),
        "latency_seconds": float(timing.get("total_duration_seconds", 0.0) or 0.0),
        "returncode": final_returncode,
        "transcript_excerpt": transcript[:2000],
    }


def _group_key(outcome: Dict[str, Any]) -> str:
    failure_types = outcome.get("failure_types") or []
    if "false_positive" in failure_types:
        return "false_positive"
    if "false_negative" in failure_types:
        return "false_negative"
    if "step_compliance_failure" in failure_types:
        return "step_compliance_failure"
    for key in ("schema_invalid", "fields_missing", "output_parse_failure"):
        if key in failure_types:
            return key
    return failure_types[0] if failure_types else "unknown"


def _select_failure_outcomes(outcomes: List[Dict[str, Any]], max_errors: int) -> List[Dict[str, Any]]:
    failures = [item for item in outcomes if item.get("failure_types")]
    if len(failures) <= max_errors:
        return failures
    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for item in failures:
        grouped[_group_key(item)].append(item)
    for items in grouped.values():
        items.sort(key=lambda row: (abs(float(row.get("confidence", 0.5)) - 0.5), row["sample_id"]))

    selected: List[Dict[str, Any]] = []
    group_names = sorted(grouped.keys(), key=lambda name: (-len(grouped[name]), name))
    while len(selected) < max_errors and any(grouped.values()):
        for group_name in group_names:
            if len(selected) >= max_errors:
                break
            if grouped[group_name]:
                selected.append(grouped[group_name].pop(0))
    return selected


def _select_protected_outcomes(outcomes: List[Dict[str, Any]], protected_count: int) -> List[Dict[str, Any]]:
    correct = [item for item in outcomes if item.get("is_correct") and item.get("step_compliant")]
    correct.sort(key=lambda row: (abs(float(row.get("confidence", 0.5)) - 0.5), row["sample_id"]))
    return correct[:protected_count]


def _copy_packet_files(sample_dir: Path, packet_dir: Path, outcome: Dict[str, Any], *, run_root: Path) -> None:
    packet_dir.mkdir(parents=True, exist_ok=True)
    for filename in (
        "sample_input.json",
        "gold.json",
        "agent_output.json",
        "transcript.md",
        "transcript.jsonl",
        "stderr.log",
        "tool_trace.json",
        "observability.json",
        "timing.json",
        "query.txt",
        "payload.json",
        "launcher_result.json",
        "pipeline_observability.json",
        "skill_output.json",
    ):
        source = sample_dir / filename
        if source.exists():
            shutil.copy2(source, packet_dir / filename)
    judge_findings = {
        "sample_id": outcome["sample_id"],
        "slice": outcome["slice"],
        "scenario": outcome["scenario"],
        "failure_types": outcome["failure_types"],
        "json_valid": outcome["json_valid"],
        "schema_valid": outcome["schema_valid"],
        "missing_fields": outcome["missing_fields"],
        "skill_invoked": outcome["skill_invoked"],
        "launcher_invoked": outcome["launcher_invoked"],
        "launcher_success": outcome["launcher_success"],
        "step_compliant": outcome["step_compliant"],
        "is_correct": outcome["is_correct"],
        "observed_issues": outcome["observed_issues"],
    }
    artifact_summary = {
        "sample_id": outcome["sample_id"],
        "packet_dir": to_relative_posix_path(packet_dir, run_root / "judge"),
        "sample_dir": to_relative_posix_path(sample_dir, run_root / "judge"),
        "latency_seconds": outcome["latency_seconds"],
        "confidence": outcome["confidence"],
        "ground_truth": outcome["ground_truth"],
        "predicted": outcome["predicted"],
        "slice": outcome["slice"],
        "scenario": outcome["scenario"],
    }
    _write_json(packet_dir / "judge_findings.json", judge_findings)
    _write_json(packet_dir / "artifact_summary.json", artifact_summary)


def build_trigger_judge_artifacts(
    *,
    run_root: Path,
    report_payload: Dict[str, Any],
    selected_failures: List[Dict[str, Any]],
    protected_outcomes: List[Dict[str, Any]],
) -> Dict[str, Path]:
    judge_dir = run_root / "judge"
    failure_dir = judge_dir / "failure_packets"
    protected_dir = judge_dir / "protected_packets"
    failure_rows: List[Dict[str, Any]] = []
    protected_rows: List[Dict[str, Any]] = []

    for index, outcome in enumerate(selected_failures, 1):
        packet_id = f"failure_{index:03d}_{outcome['sample_id']}"
        packet_path = failure_dir / packet_id
        _copy_packet_files(Path(outcome["sample_dir"]), packet_path, outcome, run_root=run_root)
        failure_rows.append(
            {
                "packet_id": packet_id,
                "sample_id": outcome["sample_id"],
                "slice": outcome["slice"],
                "scenario": outcome["scenario"],
                "failure_types": outcome["failure_types"],
                "confidence": outcome["confidence"],
                "packet_dir": to_relative_posix_path(packet_path, judge_dir),
            }
        )

    for index, outcome in enumerate(protected_outcomes, 1):
        packet_id = f"protected_{index:03d}_{outcome['sample_id']}"
        packet_path = protected_dir / packet_id
        _copy_packet_files(Path(outcome["sample_dir"]), packet_path, outcome, run_root=run_root)
        protected_rows.append(
            {
                "packet_id": packet_id,
                "sample_id": outcome["sample_id"],
                "slice": outcome["slice"],
                "scenario": outcome["scenario"],
                "ground_truth": outcome["ground_truth"],
                "confidence": outcome["confidence"],
                "packet_dir": to_relative_posix_path(packet_path, judge_dir),
            }
        )

    report_path = judge_dir / "report.json"
    error_index_path = judge_dir / "error_index.jsonl"
    protected_index_path = judge_dir / "protected_index.jsonl"
    _write_json(report_path, report_payload)
    _write_jsonl(error_index_path, failure_rows)
    _write_jsonl(protected_index_path, protected_rows)

    return {
        "judge_dir": judge_dir,
        "report_path": report_path,
        "error_index_path": error_index_path,
        "protected_index_path": protected_index_path,
        "failure_packets_dir": failure_dir,
        "protected_packets_dir": protected_dir,
    }


def _build_full_smoke_outcome(sample_dir: Path) -> Dict[str, Any]:
    gold = _load_json(sample_dir / "gold.json", default={}) or {}
    trigger_outcome = _build_outcome(sample_dir)
    skill_output = _load_json(sample_dir / "skill_output.json", default={}) or {}
    parsed_skill_output = skill_output.get("parsed_json")
    full_json_valid = bool(skill_output.get("json_valid", False))
    full_schema_valid, full_schema_error = _validate_formal_output(parsed_skill_output)
    expected_is_minor = gold.get("expected_is_minor")
    final_predicted_is_minor = _predicted_minor_label(parsed_skill_output)

    if expected_is_minor is None:
        final_label_correct = None
    elif final_predicted_is_minor is None:
        final_label_correct = False
    else:
        final_label_correct = bool(final_predicted_is_minor) == bool(expected_is_minor)

    should_trigger = bool(gold.get("should_trigger", False))
    if should_trigger:
        end_to_end_success = (
            trigger_outcome.get("predicted") is True
            and trigger_outcome.get("launcher_invoked")
            and trigger_outcome.get("launcher_success")
            and full_json_valid
            and full_schema_valid
            and final_label_correct is True
        )
    else:
        end_to_end_success = (
            trigger_outcome.get("predicted") is False
            and not trigger_outcome.get("launcher_invoked")
        )

    failure_types = list(trigger_outcome.get("failure_types") or [])
    if should_trigger and not trigger_outcome.get("launcher_success"):
        failure_types.append("full_smoke_invocation_failure")
    if should_trigger and not full_json_valid:
        failure_types.append("full_smoke_output_parse_failure")
    if should_trigger and full_json_valid and not full_schema_valid:
        failure_types.append("full_smoke_schema_invalid")
    if should_trigger and full_schema_valid and final_label_correct is False:
        failure_types.append("full_smoke_label_incorrect")
    if not end_to_end_success:
        failure_types.append("full_smoke_end_to_end_failure")

    return {
        **trigger_outcome,
        "expected_is_minor": expected_is_minor,
        "full_output_json_valid": full_json_valid,
        "full_output_schema_valid": full_schema_valid,
        "full_output_schema_error": full_schema_error,
        "final_predicted_is_minor": final_predicted_is_minor,
        "final_label_correct": final_label_correct,
        "end_to_end_success": end_to_end_success,
        "failure_types": list(dict.fromkeys(failure_types)),
    }


def judge_trigger_full_smoke_artifacts(
    *,
    run_root: Path,
    skill_version: str,
    dataset_name: str,
    project_root: Optional[Path] = None,
) -> Dict[str, Any]:
    sample_dirs = _list_sample_dirs(run_root)
    outcomes = [_build_full_smoke_outcome(sample_dir) for sample_dir in sample_dirs]

    total = len(outcomes)
    trigger_tp = sum(1 for item in outcomes if item["predicted"] is True and item["ground_truth"] is True)
    trigger_tn = sum(1 for item in outcomes if item["predicted"] is False and item["ground_truth"] is False)
    trigger_fp = sum(
        1
        for item in outcomes
        if (item["predicted"] is True and item["ground_truth"] is False)
        or (item["predicted"] is None and item["ground_truth"] is False)
    )
    trigger_fn = sum(1 for item in outcomes if item["predicted"] is not True and item["ground_truth"] is True)
    trigger_precision = trigger_tp / (trigger_tp + trigger_fp) if (trigger_tp + trigger_fp) else 0.0
    trigger_recall = trigger_tp / (trigger_tp + trigger_fn) if (trigger_tp + trigger_fn) else 0.0
    trigger_f1 = (
        2 * trigger_precision * trigger_recall / (trigger_precision + trigger_recall)
        if (trigger_precision + trigger_recall)
        else 0.0
    )

    invoked = [item for item in outcomes if item.get("launcher_invoked")]
    triggered_gold = [item for item in outcomes if item.get("ground_truth")]
    full_json_valid_rate = sum(1 for item in invoked if item.get("full_output_json_valid")) / len(invoked) if invoked else 0.0
    full_schema_valid_rate = sum(1 for item in invoked if item.get("full_output_schema_valid")) / len(invoked) if invoked else 0.0
    final_accuracy_rate = (
        sum(1 for item in invoked if item.get("final_label_correct") is True) / len(invoked)
        if invoked
        else 0.0
    )
    end_to_end_success_rate = sum(1 for item in outcomes if item.get("end_to_end_success")) / total if total else 0.0
    positive_path_success_rate = (
        sum(1 for item in triggered_gold if item.get("end_to_end_success")) / len(triggered_gold)
        if triggered_gold
        else 0.0
    )

    failure_type_counts = dict(Counter(
        failure_type
        for item in outcomes
        for failure_type in (item.get("failure_types") or [])
    ))
    slice_stats: Dict[str, Dict[str, Any]] = {}
    grouped_by_slice: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for item in outcomes:
        grouped_by_slice[item["slice"] or "unknown"].append(item)
    for slice_name, slice_items in sorted(grouped_by_slice.items()):
        slice_stats[slice_name] = {
            "sample_count": len(slice_items),
            "trigger_accuracy": sum(1 for item in slice_items if item.get("is_correct")) / len(slice_items) if slice_items else 0.0,
            "end_to_end_success_rate": sum(1 for item in slice_items if item.get("end_to_end_success")) / len(slice_items) if slice_items else 0.0,
            "invoked_count": sum(1 for item in slice_items if item.get("launcher_invoked")),
            "full_schema_valid_count": sum(1 for item in slice_items if item.get("full_output_schema_valid")),
            "error_sample_ids": [item["sample_id"] for item in slice_items if item.get("failure_types")],
        }

    report_payload = {
        "task_type": "trigger_full_smoke",
        "evaluation_role": "standalone_full_smoke",
        "optimizer_feedback_enabled": False,
        "skill_version": skill_version,
        "dataset": dataset_name,
        "sample_count": total,
        "trigger_metrics": {
            "accuracy": (trigger_tp + trigger_tn) / total if total else 0.0,
            "precision": trigger_precision,
            "recall": trigger_recall,
            "f1_score": trigger_f1,
            "true_positive": trigger_tp,
            "true_negative": trigger_tn,
            "false_positive": trigger_fp,
            "false_negative": trigger_fn,
        },
        "invoked_sample_count": len(invoked),
        "positive_sample_count": len(triggered_gold),
        "full_output_json_valid_rate_on_invoked": full_json_valid_rate,
        "full_output_schema_valid_rate_on_invoked": full_schema_valid_rate,
        "final_minor_accuracy_rate_on_invoked": final_accuracy_rate,
        "positive_path_end_to_end_success_rate": positive_path_success_rate,
        "end_to_end_success_rate": end_to_end_success_rate,
        "failure_type_counts": failure_type_counts,
        "slice_stats": slice_stats,
    }
    judge_dir = run_root / "full_smoke_judge"
    report_path = judge_dir / "report.json"
    error_index_path = judge_dir / "error_index.jsonl"
    error_rows = []
    for item in outcomes:
        if not item.get("failure_types"):
            continue
        error_rows.append(
            {
                "sample_id": item["sample_id"],
                "slice": item["slice"],
                "scenario": item["scenario"],
                "failure_types": item["failure_types"],
                "launcher_invoked": item["launcher_invoked"],
                "launcher_success": item["launcher_success"],
                "full_output_json_valid": item["full_output_json_valid"],
                "full_output_schema_valid": item["full_output_schema_valid"],
                "final_label_correct": item["final_label_correct"],
            }
        )
    if project_root is not None:
        report_payload = normalize_project_paths(report_payload, project_root=project_root, start=judge_dir)
    _write_json(report_path, report_payload)
    _write_jsonl(error_index_path, error_rows)
    return {
        "judge_dir": judge_dir,
        "report_path": report_path,
        "error_index_path": error_index_path,
        "report_payload": report_payload,
    }


def judge_trigger_run_artifacts(
    *,
    run_root: Path,
    skill_version: str,
    parent_version: Optional[str],
    dataset_name: str,
    max_errors: Optional[int] = None,
    protected_count: int = DEFAULT_PROTECTED_COUNT,
    project_root: Optional[Path] = None,
) -> Dict[str, Any]:
    sample_dirs = _list_sample_dirs(run_root)
    outcomes = [_build_outcome(sample_dir) for sample_dir in sample_dirs]

    total = len(outcomes)
    tp = sum(1 for item in outcomes if item["predicted"] is True and item["ground_truth"] is True)
    tn = sum(1 for item in outcomes if item["predicted"] is False and item["ground_truth"] is False)
    fp = sum(1 for item in outcomes if (item["predicted"] is True and item["ground_truth"] is False) or (item["predicted"] is None and item["ground_truth"] is False))
    fn = sum(1 for item in outcomes if (item["predicted"] is not True and item["ground_truth"] is True))
    accuracy = (tp + tn) / total if total else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    schema_validity_rate = sum(1 for item in outcomes if item["schema_valid"]) / total if total else 0.0
    invocation_success_rate = sum(1 for item in outcomes if item["invocation_success"]) / total if total else 0.0
    step_compliance_rate = sum(1 for item in outcomes if item["step_compliant"]) / total if total else 0.0

    fields_missing_stats: Dict[str, int] = {}
    failure_type_counts: Dict[str, int] = {}
    observed_issue_counts: Dict[str, int] = {}
    slice_stats: Dict[str, Dict[str, Any]] = {}
    scenario_counts: Dict[str, int] = {}
    for item in outcomes:
        scenario = item["scenario"] or "unknown"
        scenario_counts[scenario] = scenario_counts.get(scenario, 0) + 1
        for field_name in item["missing_fields"]:
            fields_missing_stats[field_name] = fields_missing_stats.get(field_name, 0) + 1
        for failure_type in item["failure_types"]:
            failure_type_counts[failure_type] = failure_type_counts.get(failure_type, 0) + 1
        for observed_issue in item["observed_issues"]:
            observed_issue_counts[observed_issue] = observed_issue_counts.get(observed_issue, 0) + 1

    grouped_by_slice: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for item in outcomes:
        grouped_by_slice[item["slice"] or "unknown"].append(item)
    for slice_name, slice_items in sorted(grouped_by_slice.items()):
        slice_tp = sum(1 for item in slice_items if item["predicted"] is True and item["ground_truth"] is True)
        slice_tn = sum(1 for item in slice_items if item["predicted"] is False and item["ground_truth"] is False)
        slice_fp = sum(1 for item in slice_items if (item["predicted"] is True and item["ground_truth"] is False) or (item["predicted"] is None and item["ground_truth"] is False))
        slice_fn = sum(1 for item in slice_items if item["predicted"] is not True and item["ground_truth"] is True)
        slice_precision = slice_tp / (slice_tp + slice_fp) if (slice_tp + slice_fp) else 0.0
        slice_recall = slice_tp / (slice_tp + slice_fn) if (slice_tp + slice_fn) else 0.0
        slice_f1 = 2 * slice_precision * slice_recall / (slice_precision + slice_recall) if (slice_precision + slice_recall) else 0.0
        slice_stats[slice_name] = {
            "sample_count": len(slice_items),
            "accuracy": (slice_tp + slice_tn) / len(slice_items) if slice_items else 0.0,
            "precision": slice_precision,
            "recall": slice_recall,
            "f1_score": slice_f1,
            "true_positive": slice_tp,
            "true_negative": slice_tn,
            "false_positive": slice_fp,
            "false_negative": slice_fn,
            "error_sample_ids": [item["sample_id"] for item in slice_items if item["failure_types"]],
        }

    total_errors = sum(1 for item in outcomes if item["failure_types"])
    if total_errors <= 0:
        resolved_max_errors = 0
    elif max_errors is None:
        resolved_max_errors = calc_default_max_errors(total_errors, total)
    else:
        resolved_max_errors = max(1, min(total_errors, int(max_errors)))
    selected_failures = _select_failure_outcomes(outcomes, resolved_max_errors)
    protected_outcomes = _select_protected_outcomes(outcomes, protected_count=protected_count)

    report_payload = {
        "task_type": "trigger_eval",
        "optimization_focus": "description",
        "evaluation_contract": "trigger_decision_and_skill_invocation_success_only",
        "launcher_contract": "skill_invocation_probe_invoked_when_triggered",
        "skill_version": skill_version,
        "parent_version": parent_version,
        "dataset": dataset_name,
        "sample_count": total,
        "total_errors": total_errors,
        "max_errors": resolved_max_errors,
        "metrics": {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "true_positive": tp,
            "true_negative": tn,
            "false_positive": fp,
            "false_negative": fn,
        },
        "schema_validity_rate": schema_validity_rate,
        "invocation_success_rate": invocation_success_rate,
        "step_compliance_rate": step_compliance_rate,
        "fields_missing_stats": fields_missing_stats,
        "failure_type_counts": failure_type_counts,
        "observed_issue_counts": observed_issue_counts,
        "slice_stats": slice_stats,
        "scenario_counts": scenario_counts,
        "latency_stats": {
            "avg_seconds": round(mean([item["latency_seconds"] for item in outcomes]), 4) if outcomes else 0.0,
            "max_seconds": round(max([item["latency_seconds"] for item in outcomes], default=0.0), 4),
        },
        "packet_selection": {
            "failure_packets": len(selected_failures),
            "protected_packets": len(protected_outcomes),
        },
    }
    if project_root is not None:
        report_payload = normalize_project_paths(report_payload, project_root=project_root, start=run_root / "judge")

    artifact_paths = build_trigger_judge_artifacts(
        run_root=run_root,
        report_payload=report_payload,
        selected_failures=selected_failures,
        protected_outcomes=protected_outcomes,
    )
    artifact_paths["report_payload"] = report_payload
    return artifact_paths

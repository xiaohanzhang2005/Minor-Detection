# 模块说明：
# - 把样本级运行产物聚合成 judge report、failure packets 和 protected packets。
# - Mode A 和 Mode B 共用。

from __future__ import annotations

import hashlib
import json
import math
import re
import shutil
from pathlib import Path
from statistics import mean
from typing import Any, Dict, List, Optional, Tuple

from pydantic import ValidationError

from src.models import FormalSkillOutput
from src.utils.path_utils import normalize_project_paths, to_relative_posix_path


DEFAULT_PROTECTED_COUNT = 8
TIME_SCRIPT_NAME = "extract_time_features.py"
RETRIEVE_SCRIPT_NAME = "retrieve_cases.py"
TIMESTAMP_PATTERN = re.compile(r"(\d{4}[-/]\d{1,2}[-/]\d{1,2}[^\n]{0,24}\d{1,2}:\d{2}(?::\d{2})?)")


def _safe_packet_slug(text: str, limit: int = 32) -> str:
    raw = str(text or "").strip()
    slug = re.sub(r"[^A-Za-z0-9._-]+", "-", raw).strip("-._")
    slug = (slug[:limit].rstrip("-._") or "sample")
    digest = hashlib.sha1(raw.encode("utf-8", errors="ignore")).hexdigest()[:10]
    return f"{slug}-{digest}"


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


def calc_default_max_errors(total_errors: int, eval_size: int) -> int:
    if total_errors <= 0:
        return 0
    return min(total_errors, min(36, max(12, math.ceil(eval_size * 0.12))))


def _list_sample_dirs(run_root: Path) -> List[Path]:
    sample_dirs: List[Path] = []
    for path in sorted(run_root.iterdir()):
        if not path.is_dir():
            continue
        if (path / "sample_input.json").exists():
            sample_dirs.append(path)
    return sample_dirs


def _extract_conversation_text(payload: Dict[str, Any]) -> str:
    lines: List[str] = []
    for turn in payload.get("conversation", []) or []:
        content = str(turn.get("content", "") or "").strip()
        if content:
            lines.append(content)
    return "\n".join(lines)


def _expected_time_handling(sample_payload: Dict[str, Any]) -> bool:
    raw_time_hint = str(((sample_payload.get("context") or {}).get("raw_time_hint") or "")).strip()
    if raw_time_hint:
        return True
    return bool(TIMESTAMP_PATTERN.search(_extract_conversation_text(sample_payload)))


def _load_transcript_text(sample_dir: Path) -> str:
    parts: List[str] = []
    for filename in ("transcript.md", "transcript.jsonl", "stderr.log"):
        path = sample_dir / filename
        if path.exists():
            parts.append(path.read_text(encoding="utf-8", errors="replace"))
    return "\n".join(parts)


def _load_observability(sample_dir: Path) -> Dict[str, Any]:
    payload = _load_json(sample_dir / "observability.json", default={}) or {}
    return payload if isinstance(payload, dict) else {}


def _detect_time_handling(transcript_text: str, observability: Dict[str, Any]) -> bool:
    time_processing = observability.get("time_processing") or {}
    if isinstance(time_processing, dict) and bool(time_processing.get("successful")):
        return True
    lowered = transcript_text.lower()
    return any(token in lowered for token in ("time_features", TIME_SCRIPT_NAME.lower(), "raw_time_hint"))


def _detect_script_usage(transcript_text: str, observability: Dict[str, Any]) -> bool:
    script_calls = observability.get("script_calls") or []
    if isinstance(script_calls, list) and any(isinstance(item, dict) for item in script_calls):
        return True
    lowered = transcript_text.lower()
    return any(token in lowered for token in (TIME_SCRIPT_NAME.lower(), RETRIEVE_SCRIPT_NAME.lower()))


def _collect_observed_issues(transcript_text: str, observability: Dict[str, Any]) -> List[str]:
    issues = set(str(item) for item in (observability.get("issues") or []) if str(item).strip())
    lowered = transcript_text.lower()
    if "unrecognized arguments:" in lowered:
        issues.add("shell_quoting_failure")
    if "fallback:" in lowered:
        issues.add("retrieval_fallback")
    if any(token in lowered for token in ("winerror 10013", "connecterror", "socket")):
        issues.add("retrieval_network_blocked")
    return sorted(issues)


def _missing_required_fields(parsed_json: Optional[Dict[str, Any]]) -> List[str]:
    if not isinstance(parsed_json, dict):
        return ["<invalid_json>"]
    required_paths = [
        ("decision",),
        ("decision", "is_minor"),
        ("decision", "minor_confidence"),
        ("decision", "confidence_band"),
        ("decision", "risk_level"),
        ("user_profile",),
        ("icbo_features",),
        ("evidence",),
        ("reasoning_summary",),
        ("trend",),
        ("uncertainty_notes",),
        ("recommended_next_step",),
    ]
    missing: List[str] = []
    for path_parts in required_paths:
        cursor: Any = parsed_json
        ok = True
        for part in path_parts:
            if not isinstance(cursor, dict) or part not in cursor:
                ok = False
                break
            cursor = cursor[part]
        if not ok:
            missing.append(".".join(path_parts))
    return missing


def _validate_formal_output(parsed_json: Optional[Dict[str, Any]]) -> Tuple[bool, Optional[str]]:
    if not isinstance(parsed_json, dict):
        return False, "invalid_json"
    try:
        FormalSkillOutput.model_validate(parsed_json)
    except ValidationError as exc:
        return False, str(exc)
    return True, None


def _confidence(parsed_json: Optional[Dict[str, Any]]) -> float:
    if not isinstance(parsed_json, dict):
        return 0.0
    decision = parsed_json.get("decision") or {}
    try:
        return float(decision.get("minor_confidence", 0.0))
    except Exception:
        return 0.0


def _predicted_label(parsed_json: Optional[Dict[str, Any]]) -> Optional[bool]:
    if not isinstance(parsed_json, dict):
        return None
    decision = parsed_json.get("decision") or {}
    if "is_minor" not in decision:
        return None
    return bool(decision["is_minor"])


def _build_outcome(sample_dir: Path) -> Dict[str, Any]:
    sample_input = _load_json(sample_dir / "sample_input.json", default={}) or {}
    gold = _load_json(sample_dir / "gold.json", default={}) or {}
    agent_output = _load_json(sample_dir / "agent_output.json", default={}) or {}
    metadata = _load_json(sample_dir / "run_metadata.json", default={}) or {}
    timing = _load_json(sample_dir / "timing.json", default={}) or {}
    observability = _load_observability(sample_dir)
    transcript_text = _load_transcript_text(sample_dir)

    parsed_json = agent_output.get("parsed_json")
    json_valid = bool(agent_output.get("json_valid", False))
    schema_valid, schema_error = _validate_formal_output(parsed_json)
    missing_fields = _missing_required_fields(parsed_json)
    predicted = _predicted_label(parsed_json)
    ground_truth = bool(gold.get("is_minor", False))
    confidence = _confidence(parsed_json)

    expected_time = _expected_time_handling(sample_input)
    time_detected = _detect_time_handling(transcript_text, observability)
    script_detected = _detect_script_usage(transcript_text, observability)
    expected_script = expected_time

    observed_issues = _collect_observed_issues(transcript_text, observability)
    retrieval_payload = observability.get("retrieval") or {}
    retrieval_mode = str(retrieval_payload.get("mode") or "").strip() or None
    retrieval_attempted = bool(retrieval_payload.get("attempted"))
    used_retrieval_fallback = bool(retrieval_payload.get("used_fallback"))
    retrieval_network_blocked = bool(retrieval_payload.get("network_error_detected"))
    shell_quoting_failure_detected = bool(retrieval_payload.get("quoting_failure_detected"))
    command_failures = int(((observability.get("summary") or {}).get("failed_script_calls", 0)) or 0)

    failure_types: List[str] = []
    if not json_valid:
        failure_types.append("output_parse_failure")
    if json_valid and not schema_valid:
        failure_types.append("schema_invalid")
    if missing_fields and "<invalid_json>" not in missing_fields:
        failure_types.append("fields_missing")
    if expected_time and not time_detected:
        failure_types.append("missing_time_handling")
    if expected_script and not script_detected:
        failure_types.append("missing_script_usage")

    is_correct = predicted == ground_truth if predicted is not None else False
    if predicted is not None and predicted and not ground_truth:
        failure_types.append("false_positive")
    if predicted is not None and (not predicted) and ground_truth:
        failure_types.append("false_negative")

    step_compliant = json_valid and schema_valid and not missing_fields and (not expected_time or time_detected) and (
        not expected_script or script_detected
    )
    if not step_compliant:
        failure_types.append("step_compliance_failure")

    launcher_success = agent_output.get("launcher_success")
    invocation_success = bool(
        metadata.get("returncode", 1) == 0
        and (agent_output.get("raw_text") or "").strip()
        and (bool(launcher_success) if launcher_success is not None else json_valid)
    )

    return {
        "sample_id": str(gold.get("sample_id") or metadata.get("sample_id") or sample_dir.name),
        "sample_dir": str(sample_dir),
        "ground_truth": ground_truth,
        "predicted": predicted,
        "confidence": confidence,
        "is_correct": is_correct,
        "json_valid": json_valid,
        "schema_valid": schema_valid,
        "schema_error": schema_error,
        "missing_fields": missing_fields,
        "expected_time_handling": expected_time,
        "time_handling_detected": time_detected,
        "expected_script_usage": expected_script,
        "script_usage_detected": script_detected,
        "step_compliant": step_compliant,
        "invocation_success": invocation_success,
        "failure_types": list(dict.fromkeys(failure_types)),
        "observed_issues": observed_issues,
        "retrieval_mode": retrieval_mode,
        "retrieval_attempted": retrieval_attempted,
        "used_retrieval_fallback": used_retrieval_fallback,
        "retrieval_network_blocked": retrieval_network_blocked,
        "shell_quoting_failure_detected": shell_quoting_failure_detected,
        "command_failures": command_failures,
        "latency_seconds": float(timing.get("total_duration_seconds", 0.0) or 0.0),
        "returncode": int(metadata.get("returncode", 1) or 1),
    }


def _group_key(outcome: Dict[str, Any]) -> str:
    failure_types = outcome.get("failure_types") or []
    if "false_positive" in failure_types:
        return "false_positive"
    if "false_negative" in failure_types:
        return "false_negative"
    for key in ("missing_time_handling", "missing_script_usage", "schema_invalid", "fields_missing", "output_parse_failure"):
        if key in failure_types:
            return key
    return failure_types[0] if failure_types else "unknown"


def _select_failure_outcomes(outcomes: List[Dict[str, Any]], max_errors: int) -> List[Dict[str, Any]]:
    failures = [item for item in outcomes if item.get("failure_types")]
    if len(failures) <= max_errors:
        return failures
    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for item in failures:
        grouped.setdefault(_group_key(item), []).append(item)
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
    correct = [item for item in outcomes if item.get("is_correct")]
    if not correct:
        return []
    minor = [item for item in correct if item.get("ground_truth")]
    adult = [item for item in correct if not item.get("ground_truth")]
    minor.sort(key=lambda row: (abs(float(row.get("confidence", 0.5)) - 0.5), row["sample_id"]))
    adult.sort(key=lambda row: (abs(float(row.get("confidence", 0.5)) - 0.5), row["sample_id"]))
    protected: List[Dict[str, Any]] = []
    protected.extend(minor[: max(1, protected_count // 2)])
    protected.extend(adult[: max(1, protected_count // 2)])
    if len(protected) < protected_count:
        remainder = [item for item in correct if item not in protected]
        remainder.sort(key=lambda row: (abs(float(row.get("confidence", 0.5)) - 0.5), row["sample_id"]))
        protected.extend(remainder[: protected_count - len(protected)])
    return protected[:protected_count]


def _copy_packet_files(sample_dir: Path, packet_dir: Path, outcome: Dict[str, Any], *, run_root: Path) -> None:
    packet_dir.mkdir(parents=True, exist_ok=True)
    for filename in (
        "sample_input.json",
        "gold.json",
        "agent_output.json",
        "transcript.md",
        "tool_trace.json",
        "observability.json",
        "timing.json",
    ):
        source = sample_dir / filename
        if source.exists():
            shutil.copy2(source, packet_dir / filename)
    judge_findings = {
        "sample_id": outcome["sample_id"],
        "failure_types": outcome["failure_types"],
        "json_valid": outcome["json_valid"],
        "schema_valid": outcome["schema_valid"],
        "missing_fields": outcome["missing_fields"],
        "time_handling_detected": outcome["time_handling_detected"],
        "script_usage_detected": outcome["script_usage_detected"],
        "step_compliant": outcome["step_compliant"],
        "is_correct": outcome["is_correct"],
        "observed_issues": outcome["observed_issues"],
        "retrieval_mode": outcome["retrieval_mode"],
        "command_failures": outcome["command_failures"],
    }
    _write_json(packet_dir / "judge_findings.json", judge_findings)
    artifact_summary = {
        "sample_id": outcome["sample_id"],
        "packet_dir": to_relative_posix_path(packet_dir, run_root / "judge"),
        "sample_dir": to_relative_posix_path(sample_dir, run_root / "judge"),
        "latency_seconds": outcome["latency_seconds"],
        "confidence": outcome["confidence"],
        "ground_truth": outcome["ground_truth"],
        "predicted": outcome["predicted"],
        "observed_issues": outcome["observed_issues"],
        "retrieval_mode": outcome["retrieval_mode"],
    }
    _write_json(packet_dir / "artifact_summary.json", artifact_summary)


def build_judge_artifacts(
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
        packet_id = f"failure_{index:03d}_{_safe_packet_slug(outcome['sample_id'])}"
        packet_path = failure_dir / packet_id
        _copy_packet_files(Path(outcome["sample_dir"]), packet_path, outcome, run_root=run_root)
        failure_rows.append(
            {
                "packet_id": packet_id,
                "sample_id": outcome["sample_id"],
                "failure_types": outcome["failure_types"],
                "confidence": outcome["confidence"],
                "observed_issues": outcome["observed_issues"],
                "retrieval_mode": outcome["retrieval_mode"],
                "packet_dir": to_relative_posix_path(packet_path, judge_dir),
            }
        )

    for index, outcome in enumerate(protected_outcomes, 1):
        packet_id = f"protected_{index:03d}_{_safe_packet_slug(outcome['sample_id'])}"
        packet_path = protected_dir / packet_id
        _copy_packet_files(Path(outcome["sample_dir"]), packet_path, outcome, run_root=run_root)
        protected_rows.append(
            {
                "packet_id": packet_id,
                "sample_id": outcome["sample_id"],
                "ground_truth": outcome["ground_truth"],
                "confidence": outcome["confidence"],
                "observed_issues": outcome["observed_issues"],
                "retrieval_mode": outcome["retrieval_mode"],
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


def judge_run_artifacts(
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
    fp = sum(
        1
        for item in outcomes
        if (item["predicted"] is True and item["ground_truth"] is False)
        or (item["predicted"] is None and item["ground_truth"] is False)
    )
    fn = sum(
        1
        for item in outcomes
        if (item["predicted"] is False and item["ground_truth"] is True)
        or (item["predicted"] is None and item["ground_truth"] is True)
    )
    accuracy = (tp + tn) / total if total else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    schema_validity_rate = sum(1 for item in outcomes if item["schema_valid"]) / total if total else 0.0
    invocation_success_rate = sum(1 for item in outcomes if item["invocation_success"]) / total if total else 0.0
    step_compliance_rate = sum(1 for item in outcomes if item["step_compliant"]) / total if total else 0.0
    applicable_script = [item for item in outcomes if item["expected_script_usage"]]
    script_usage_rate = (
        sum(1 for item in applicable_script if item["script_usage_detected"]) / len(applicable_script)
        if applicable_script
        else 1.0
    )

    field_counts: Dict[str, int] = {}
    parse_failure_stats = {"output_parse_failure": 0, "schema_invalid": 0}
    failure_type_counts: Dict[str, int] = {}
    observed_issue_counts: Dict[str, int] = {}
    retrieval_mode_counts: Dict[str, int] = {}
    for item in outcomes:
        for field_name in item["missing_fields"]:
            field_counts[field_name] = field_counts.get(field_name, 0) + 1
        for failure_type in item["failure_types"]:
            failure_type_counts[failure_type] = failure_type_counts.get(failure_type, 0) + 1
        for observed_issue in item["observed_issues"]:
            observed_issue_counts[observed_issue] = observed_issue_counts.get(observed_issue, 0) + 1
        if item["retrieval_mode"]:
            retrieval_mode_counts[item["retrieval_mode"]] = retrieval_mode_counts.get(item["retrieval_mode"], 0) + 1
        if "output_parse_failure" in item["failure_types"]:
            parse_failure_stats["output_parse_failure"] += 1
        if "schema_invalid" in item["failure_types"]:
            parse_failure_stats["schema_invalid"] += 1

    total_errors = sum(1 for item in outcomes if item["failure_types"])
    if total_errors <= 0:
        resolved_max_errors = 0
    elif max_errors is None:
        resolved_max_errors = calc_default_max_errors(total_errors, total)
    else:
        resolved_max_errors = max(1, min(total_errors, int(max_errors)))
    selected_failures = _select_failure_outcomes(outcomes, resolved_max_errors)
    protected_outcomes = _select_protected_outcomes(outcomes, protected_count=protected_count)

    retrieval_attempted = [item for item in outcomes if item["retrieval_attempted"]]
    retrieval_fallback_count = sum(1 for item in retrieval_attempted if item["used_retrieval_fallback"])
    retrieval_network_blocked_count = sum(1 for item in retrieval_attempted if item["retrieval_network_blocked"])
    shell_quoting_failure_count = sum(1 for item in retrieval_attempted if item["shell_quoting_failure_detected"])
    command_failure_samples = sum(1 for item in outcomes if item["command_failures"] > 0)

    report_payload = {
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
        "script_usage_rate": script_usage_rate,
        "fields_missing_stats": field_counts,
        "output_parse_failure_stats": parse_failure_stats,
        "failure_type_counts": failure_type_counts,
        "observed_issue_counts": observed_issue_counts,
        "retrieval_mode_counts": retrieval_mode_counts,
        "observability_stats": {
            "retrieval_attempted_samples": len(retrieval_attempted),
            "retrieval_fallback_count": retrieval_fallback_count,
            "retrieval_fallback_rate": (retrieval_fallback_count / len(retrieval_attempted)) if retrieval_attempted else 0.0,
            "retrieval_network_blocked_count": retrieval_network_blocked_count,
            "retrieval_network_blocked_rate": (retrieval_network_blocked_count / len(retrieval_attempted)) if retrieval_attempted else 0.0,
            "shell_quoting_failure_count": shell_quoting_failure_count,
            "shell_quoting_failure_rate": (shell_quoting_failure_count / len(retrieval_attempted)) if retrieval_attempted else 0.0,
            "samples_with_command_failures": command_failure_samples,
            "command_failure_rate": (command_failure_samples / total) if total else 0.0,
        },
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

    artifact_paths = build_judge_artifacts(
        run_root=run_root,
        report_payload=report_payload,
        selected_failures=selected_failures,
        protected_outcomes=protected_outcomes,
    )
    artifact_paths["report_payload"] = report_payload
    return artifact_paths

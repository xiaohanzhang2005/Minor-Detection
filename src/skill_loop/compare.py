# 模块说明：
# - 比较 accepted 和 candidate judge report 的晋升门禁。
# - 决定 promote 还是 rollback。

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

from src.skill_loop.gates import DEFAULT_REPAIR_RATE_THRESHOLD


def _load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _load_jsonl_rows(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _load_jsonl_sample_ids(path: Path) -> Set[str]:
    sample_ids: Set[str] = set()
    for row in _load_jsonl_rows(path):
        sample_id = str(row.get("sample_id", "") or "").strip()
        if sample_id:
            sample_ids.add(sample_id)
    return sample_ids


def _slice_keys(rows: List[Dict[str, Any]]) -> Set[Tuple[str, str]]:
    keys: Set[Tuple[str, str]] = set()
    for row in rows:
        slice_name = str(row.get("slice", "") or "").strip()
        scenario_name = str(row.get("scenario", "") or "").strip()
        if slice_name:
            keys.add((slice_name, scenario_name))
    return keys


def compare_reports(
    *,
    accepted_report_path: Path,
    candidate_report_path: Path,
    accepted_error_index_path: Path,
    accepted_protected_index_path: Path,
    candidate_error_index_path: Path,
    candidate_redline_index_path: Path | None = None,
) -> Dict[str, Any]:
    accepted = _load_json(accepted_report_path)
    candidate = _load_json(candidate_report_path)
    task_type = str(candidate.get("task_type") or accepted.get("task_type") or "").strip()

    accepted_metrics = accepted.get("metrics", {})
    candidate_metrics = candidate.get("metrics", {})
    accepted_f1 = float(accepted_metrics.get("f1_score", 0.0) or 0.0)
    candidate_f1 = float(candidate_metrics.get("f1_score", 0.0) or 0.0)
    accepted_invocation_success = float(accepted.get("invocation_success_rate", 0.0) or 0.0)
    candidate_invocation_success = float(candidate.get("invocation_success_rate", 0.0) or 0.0)
    accepted_step_compliance = float(accepted.get("step_compliance_rate", 0.0) or 0.0)
    candidate_step_compliance = float(candidate.get("step_compliance_rate", 0.0) or 0.0)
    accepted_schema_validity = float(accepted.get("schema_validity_rate", 0.0) or 0.0)
    candidate_schema_validity = float(candidate.get("schema_validity_rate", 0.0) or 0.0)

    gates = {
        "f1_non_regression": candidate_f1 >= accepted_f1,
        "f1_improved": candidate_f1 > accepted_f1,
        "invocation_non_regression": candidate_invocation_success >= accepted_invocation_success,
        "invocation_improved": candidate_invocation_success > accepted_invocation_success,
        "step_compliance_non_regression": candidate_step_compliance >= accepted_step_compliance,
        "step_compliance_improved": candidate_step_compliance > accepted_step_compliance,
        "schema_non_regression": candidate_schema_validity >= accepted_schema_validity,
    }

    accepted_error_ids = _load_jsonl_sample_ids(accepted_error_index_path)
    candidate_error_ids = _load_jsonl_sample_ids(candidate_error_index_path)
    accepted_protected_rows = _load_jsonl_rows(accepted_protected_index_path)
    protected_ids = {
        str(row.get("sample_id", "") or "").strip()
        for row in accepted_protected_rows
        if str(row.get("sample_id", "") or "").strip()
    }
    protected_regressions = sorted(protected_ids.intersection(candidate_error_ids))
    gates["protected_non_regression"] = not protected_regressions
    protected_slice_regressions = [
        {"slice": slice_name, "scenario": scenario_name}
        for slice_name, scenario_name in sorted(
            _slice_keys(accepted_protected_rows).intersection(_slice_keys(_load_jsonl_rows(candidate_error_index_path)))
        )
    ]
    gates["protected_slice_non_regression"] = not protected_slice_regressions

    repaired_sample_ids = sorted(accepted_error_ids.difference(candidate_error_ids))
    repeated_error_ids = sorted(accepted_error_ids.intersection(candidate_error_ids))
    previous_error_packets = len(accepted_error_ids)
    repair_rate_on_previous_error_packets = (
        len(repaired_sample_ids) / previous_error_packets if previous_error_packets else 1.0
    )
    repair_threshold = float(
        ((candidate.get("repair_tracking_baseline") or {}).get("threshold"))
        or ((accepted.get("repair_tracking_baseline") or {}).get("threshold"))
        or DEFAULT_REPAIR_RATE_THRESHOLD
    )
    repair_tracking = {
        "previous_error_packets": previous_error_packets,
        "fixed_sample_ids": repaired_sample_ids,
        "repeated_error_ids": repeated_error_ids,
        "repair_rate_on_previous_error_packets": repair_rate_on_previous_error_packets,
        "threshold": repair_threshold,
        "pass": previous_error_packets == 0 or repair_rate_on_previous_error_packets >= repair_threshold,
    }
    gates["repair_rate_pass"] = repair_tracking["pass"]

    if candidate_redline_index_path and candidate_redline_index_path.exists():
        redline_rows = _load_jsonl_rows(candidate_redline_index_path)
        redline_regressions = sorted(
            {
                str(row.get("sample_id", "") or "").strip()
                for row in redline_rows
                if str(row.get("sample_id", "") or "").strip()
            }
        )
    else:
        redline_regressions = sorted(
            str(item or "").strip()
            for item in ((candidate.get("redline_stats") or {}).get("redline_regression_ids") or [])
            if str(item or "").strip()
        )
    gates["redline_non_regression"] = not redline_regressions
    gates["core_metric_improved"] = any(
        (
            gates["f1_improved"],
            gates["invocation_improved"],
            gates["step_compliance_improved"],
        )
    )

    if task_type == "trigger_eval":
        promoted = all(
            (
                gates["core_metric_improved"],
                gates["f1_non_regression"],
                gates["invocation_non_regression"],
                gates["step_compliance_non_regression"],
                gates["schema_non_regression"],
                gates["protected_non_regression"],
                gates["protected_slice_non_regression"],
            )
        )
    else:
        promoted = all(
            (
                gates["core_metric_improved"],
                gates["f1_non_regression"],
                gates["invocation_non_regression"],
                gates["step_compliance_non_regression"],
                gates["schema_non_regression"],
                gates["protected_non_regression"],
                gates["redline_non_regression"],
            )
        )
    candidate_contract_preview = (
        candidate.get("contract_check_preview")
        or candidate.get("release_contract_gate_results")
        or candidate.get("gate_results")
        or {}
    )
    contract_check_preview_summary = {
        "candidate_gate_results": candidate_contract_preview,
        "all_green": bool(
            candidate_contract_preview.get(
                "release_gate_pass",
                candidate_schema_validity == 1.0 and candidate_step_compliance == 1.0,
            )
        ),
    }
    final_validation_metrics = candidate.get("final_validation_metrics") or {}
    return {
        "decision": "promote" if promoted else "rollback",
        "inner_loop_decision": "promote" if promoted else "rollback",
        "task_type": task_type,
        "accepted_f1": accepted_f1,
        "candidate_f1": candidate_f1,
        "f1_delta": candidate_f1 - accepted_f1,
        "accepted_invocation_success_rate": accepted_invocation_success,
        "candidate_invocation_success_rate": candidate_invocation_success,
        "invocation_success_delta": candidate_invocation_success - accepted_invocation_success,
        "accepted_step_compliance_rate": accepted_step_compliance,
        "candidate_step_compliance_rate": candidate_step_compliance,
        "step_compliance_delta": candidate_step_compliance - accepted_step_compliance,
        "gates": gates,
        "inner_loop_gate_results": gates,
        "protected_regressions": protected_regressions,
        "protected_slice_regressions": protected_slice_regressions,
        "redline_regressions": redline_regressions,
        "repair_tracking": repair_tracking,
        "final_validation_metrics": final_validation_metrics,
        "contract_check_preview": candidate_contract_preview,
        "contract_check_preview_summary": contract_check_preview_summary,
    }

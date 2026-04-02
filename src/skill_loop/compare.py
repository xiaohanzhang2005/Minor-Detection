# 模块说明：
# - 比较 accepted 和 candidate judge report 的晋升门禁。
# - 决定 promote 还是 rollback。

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Set


def _load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _load_jsonl_sample_ids(path: Path) -> Set[str]:
    sample_ids: Set[str] = set()
    if not path.exists():
        return sample_ids
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            sample_id = str(row.get("sample_id", "") or "").strip()
            if sample_id:
                sample_ids.add(sample_id)
    return sample_ids


def compare_reports(
    *,
    accepted_report_path: Path,
    candidate_report_path: Path,
    accepted_protected_index_path: Path,
    candidate_error_index_path: Path,
) -> Dict[str, Any]:
    accepted = _load_json(accepted_report_path)
    candidate = _load_json(candidate_report_path)

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

    protected_ids = _load_jsonl_sample_ids(accepted_protected_index_path)
    candidate_error_ids = _load_jsonl_sample_ids(candidate_error_index_path)
    protected_regressions = sorted(protected_ids.intersection(candidate_error_ids))
    gates["protected_non_regression"] = not protected_regressions
    gates["core_metric_improved"] = any(
        (
            gates["f1_improved"],
            gates["invocation_improved"],
            gates["step_compliance_improved"],
        )
    )

    promoted = all(
        (
            gates["core_metric_improved"],
            gates["f1_non_regression"],
            gates["invocation_non_regression"],
            gates["step_compliance_non_regression"],
            gates["schema_non_regression"],
            gates["protected_non_regression"],
        )
    )
    return {
        "decision": "promote" if promoted else "rollback",
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
        "protected_regressions": protected_regressions,
    }

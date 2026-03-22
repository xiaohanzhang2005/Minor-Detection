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

    gates = {
        "f1_improved": candidate_f1 > accepted_f1,
        "schema_non_regression": float(candidate.get("schema_validity_rate", 0.0) or 0.0)
        >= float(accepted.get("schema_validity_rate", 0.0) or 0.0),
        "invocation_non_regression": float(candidate.get("invocation_success_rate", 0.0) or 0.0)
        >= float(accepted.get("invocation_success_rate", 0.0) or 0.0),
        "step_compliance_non_regression": float(candidate.get("step_compliance_rate", 0.0) or 0.0)
        >= float(accepted.get("step_compliance_rate", 0.0) or 0.0),
    }

    protected_ids = _load_jsonl_sample_ids(accepted_protected_index_path)
    candidate_error_ids = _load_jsonl_sample_ids(candidate_error_index_path)
    protected_regressions = sorted(protected_ids.intersection(candidate_error_ids))
    gates["protected_non_regression"] = not protected_regressions

    promoted = all(gates.values())
    return {
        "decision": "promote" if promoted else "rollback",
        "accepted_f1": accepted_f1,
        "candidate_f1": candidate_f1,
        "f1_delta": candidate_f1 - accepted_f1,
        "gates": gates,
        "protected_regressions": protected_regressions,
    }

from __future__ import annotations

import json
import unicodedata
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from src.config import ROOT_DIR


DEFAULT_FORMAL_SKILL_REDLINE_MANIFEST_PATH = ROOT_DIR / "data" / "gates" / "formal_skill_redlines.json"
DEFAULT_TRIGGER_DESCRIPTION_REDLINE_MANIFEST_PATH = ROOT_DIR / "data" / "gates" / "trigger_description_redlines.json"
DEFAULT_REPAIR_RATE_THRESHOLD = 0.8


def _normalize_free_text(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", str(value or "")).casefold()
    kept_chars: List[str] = []
    for ch in normalized:
        category = unicodedata.category(ch)
        if category[:1] in {"P", "S", "Z", "C"}:
            continue
        kept_chars.append(ch)
    return "".join(kept_chars)


def extract_conversation_text(sample_input: Dict[str, Any]) -> str:
    lines: List[str] = []
    for turn in sample_input.get("conversation", []) or []:
        if not isinstance(turn, dict):
            continue
        content = str(turn.get("content", "") or "").strip()
        if content:
            lines.append(content)
    return "\n".join(lines)


def evaluate_direct_evidence_support(parsed_json: Optional[Dict[str, Any]], sample_input: Dict[str, Any]) -> Dict[str, Any]:
    evidence_payload = parsed_json.get("evidence") if isinstance(parsed_json, dict) else {}
    direct_evidence = evidence_payload.get("direct_evidence") if isinstance(evidence_payload, dict) else []
    direct_evidence = [str(item or "").strip() for item in (direct_evidence or []) if str(item or "").strip()]
    conversation_text = extract_conversation_text(sample_input)
    normalized_conversation = _normalize_free_text(conversation_text)

    supported: List[str] = []
    unsupported: List[str] = []
    for item in direct_evidence:
        normalized_item = _normalize_free_text(item)
        if normalized_item and normalized_item in normalized_conversation:
            supported.append(item)
        else:
            unsupported.append(item)

    return {
        "conversation_text": conversation_text,
        "direct_evidence": direct_evidence,
        "supported_direct_evidence": supported,
        "unsupported_direct_evidence": unsupported,
        "direct_evidence_count": len(direct_evidence),
        "supported_direct_evidence_count": len(supported),
        "unsupported_direct_evidence_count": len(unsupported),
        "direct_evidence_trace_rate": (len(supported) / len(direct_evidence)) if direct_evidence else 1.0,
    }


def load_redline_manifest(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {"entries": [], "manifest_path": str(path)}

    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if isinstance(payload, dict):
        entries = payload.get("entries") or []
    elif isinstance(payload, list):
        entries = payload
    else:
        entries = []
    normalized_entries = [dict(item) for item in entries if isinstance(item, dict)]
    return {
        "entries": normalized_entries,
        "manifest_path": str(path),
    }


def _entry_matches_outcome(entry: Dict[str, Any], outcome: Dict[str, Any]) -> bool:
    sample_id = str(entry.get("sample_id", "") or "").strip()
    if sample_id and sample_id != str(outcome.get("sample_id", "") or "").strip():
        return False

    slice_name = str(entry.get("slice", "") or "").strip()
    if slice_name and slice_name != str(outcome.get("slice", "") or "").strip():
        return False

    scenario = str(entry.get("scenario", "") or "").strip()
    if scenario and scenario != str(outcome.get("scenario", "") or "").strip():
        return False

    return bool(sample_id or slice_name or scenario)


def _formal_redline_passed(gate_tag: str, outcome: Dict[str, Any]) -> bool:
    predicted = outcome.get("predicted")
    if gate_tag in {"must_predict_minor", "must_be_minor", "no_false_negative"}:
        return predicted is True and bool(outcome.get("is_correct"))
    if gate_tag in {"must_predict_adult", "must_be_adult", "no_false_positive"}:
        return predicted is False and bool(outcome.get("is_correct"))
    return bool(outcome.get("is_correct"))


def _trigger_redline_passed(gate_tag: str, outcome: Dict[str, Any]) -> bool:
    predicted = outcome.get("predicted")
    step_compliant = bool(outcome.get("step_compliant"))
    launcher_success = bool(outcome.get("launcher_success", True))
    if gate_tag in {"must_trigger", "must_invoke", "no_false_negative"}:
        return predicted is True and step_compliant and launcher_success
    if gate_tag in {"must_not_trigger", "must_block", "no_false_positive"}:
        return predicted is False and step_compliant
    return bool(outcome.get("is_correct"))


def evaluate_redline_outcomes(
    outcomes: Iterable[Dict[str, Any]],
    manifest_entries: Iterable[Dict[str, Any]],
    *,
    task_type: str,
) -> Dict[str, Any]:
    normalized_task_type = str(task_type or "").strip()
    regressions: List[Dict[str, Any]] = []
    matched_rows: List[Dict[str, Any]] = []

    for outcome in outcomes:
        for entry in manifest_entries:
            if not _entry_matches_outcome(entry, outcome):
                continue
            gate_tag = str(entry.get("gate_tag", "") or "").strip() or "default"
            if normalized_task_type == "trigger_eval":
                passed = _trigger_redline_passed(gate_tag, outcome)
            else:
                passed = _formal_redline_passed(gate_tag, outcome)
            row = {
                "sample_id": str(outcome.get("sample_id", "") or ""),
                "slice": str(outcome.get("slice", "") or ""),
                "scenario": str(outcome.get("scenario", "") or ""),
                "gate_tag": gate_tag,
                "passed": passed,
                "predicted": outcome.get("predicted"),
                "ground_truth": outcome.get("ground_truth"),
            }
            matched_rows.append(row)
            if not passed:
                regressions.append(row)

    matched_count = len(matched_rows)
    passed_count = matched_count - len(regressions)
    return {
        "matched_rows": matched_rows,
        "regressions": regressions,
        "matched_count": matched_count,
        "passed_count": passed_count,
        "redline_pass_rate": (passed_count / matched_count) if matched_count else 1.0,
        "redline_regression_ids": sorted(
            {
                str(item.get("sample_id", "") or "").strip()
                for item in regressions
                if str(item.get("sample_id", "") or "").strip()
            }
        ),
        "redline_passed": not regressions,
    }


def slice_key(row: Dict[str, Any]) -> Tuple[str, str]:
    return (
        str(row.get("slice", "") or "").strip(),
        str(row.get("scenario", "") or "").strip(),
    )

# 模块说明：
# - 检查 bundled skill markdown 合同和 formal schema 是否漂移。
# - loop 在真正跑数据前会先过这一关。

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Sequence, Type

from src.models import ICBOFeatures, FormalDecision, FormalEvidence, FormalSkillOutput, FormalTrend, FormalUserProfile


def _model_field_names(model_type: Type[Any]) -> List[str]:
    fields = getattr(model_type, "model_fields", {}) or {}
    return list(fields.keys())


def _section_body(markdown: str, heading: str) -> str:
    pattern = rf"^##\s+{re.escape(heading)}\s*$([\s\S]*?)(?=^##\s+|\Z)"
    match = re.search(pattern, markdown, flags=re.MULTILINE)
    return match.group(1).strip() if match else ""


def _extract_backtick_bullets(text: str) -> List[str]:
    return re.findall(r"^-\s+`([^`]+)`\s*$", text, flags=re.MULTILINE)


def _extract_json_block(section_text: str) -> Dict[str, Any]:
    match = re.search(r"```json\s*(\{[\s\S]*?\})\s*```", section_text, flags=re.MULTILINE)
    if not match:
        return {}
    return json.loads(match.group(1))


def _extract_output_schema_contract(output_schema_path: Path) -> Dict[str, Any]:
    markdown = output_schema_path.read_text(encoding="utf-8")
    top_match = re.search(r"顶层字段必须包含：\s*([\s\S]*?)(?=^##\s+|\Z)", markdown, flags=re.MULTILINE)
    top_level_fields = _extract_backtick_bullets(top_match.group(1) if top_match else "")

    icbo_section = _section_body(markdown, "icbo_features")
    icbo_required = []
    icbo_match = re.search(r"必须包含：\s*([\s\S]*?)(?=^##\s+|\Z)", icbo_section, flags=re.MULTILINE)
    if icbo_match:
        icbo_required = _extract_backtick_bullets(icbo_match.group(1))

    next_step_section = _section_body(markdown, "recommended_next_step")
    next_step_values = []
    enum_match = re.search(r"枚举值只能是：\s*([\s\S]*?)\Z", next_step_section, flags=re.MULTILINE)
    if enum_match:
        next_step_values = _extract_backtick_bullets(enum_match.group(1))

    return {
        "top_level_fields": top_level_fields,
        "decision_fields": list(_extract_json_block(_section_body(markdown, "decision")).keys()),
        "user_profile_fields": list(_extract_json_block(_section_body(markdown, "user_profile")).keys()),
        "icbo_fields": icbo_required,
        "evidence_fields": list(_extract_json_block(_section_body(markdown, "evidence")).keys()),
        "trend_fields": list(_extract_json_block(_section_body(markdown, "trend")).keys()),
        "recommended_next_step_values": next_step_values,
    }


def _compare_fields(md_fields: Sequence[str], python_fields: Sequence[str]) -> Dict[str, Any]:
    md_list = list(md_fields)
    py_list = list(python_fields)
    missing_in_markdown = [field for field in py_list if field not in md_list]
    missing_in_python = [field for field in md_list if field not in py_list]
    return {
        "markdown": md_list,
        "python": py_list,
        "missing_in_markdown": missing_in_markdown,
        "missing_in_python": missing_in_python,
        "matches": not missing_in_markdown and not missing_in_python,
    }


def validate_skill_schema_contract(skill_dir: Path) -> Dict[str, Any]:
    output_schema_path = skill_dir / "references" / "output-schema.md"
    if not output_schema_path.exists():
        raise FileNotFoundError(f"Missing output schema asset: {output_schema_path}")

    markdown_contract = _extract_output_schema_contract(output_schema_path)
    python_contract = {
        "top_level_fields": _model_field_names(FormalSkillOutput),
        "decision_fields": _model_field_names(FormalDecision),
        "user_profile_fields": _model_field_names(FormalUserProfile),
        "icbo_fields": _model_field_names(ICBOFeatures),
        "evidence_fields": _model_field_names(FormalEvidence),
        "trend_fields": _model_field_names(FormalTrend),
    }

    checks = {
        "top_level_fields": _compare_fields(markdown_contract["top_level_fields"], python_contract["top_level_fields"]),
        "decision_fields": _compare_fields(markdown_contract["decision_fields"], python_contract["decision_fields"]),
        "user_profile_fields": _compare_fields(markdown_contract["user_profile_fields"], python_contract["user_profile_fields"]),
        "icbo_fields": _compare_fields(markdown_contract["icbo_fields"], python_contract["icbo_fields"]),
        "evidence_fields": _compare_fields(markdown_contract["evidence_fields"], python_contract["evidence_fields"]),
        "trend_fields": _compare_fields(markdown_contract["trend_fields"], python_contract["trend_fields"]),
    }

    warnings: List[str] = []
    next_step_field = getattr(FormalSkillOutput, "model_fields", {}).get("recommended_next_step")
    python_enforces_next_step_enum = False
    if next_step_field is not None:
        annotation_repr = str(getattr(next_step_field, "annotation", ""))
        python_enforces_next_step_enum = "Literal" in annotation_repr or "Enum" in annotation_repr or "RecommendedNextStep" in annotation_repr
    if markdown_contract["recommended_next_step_values"] and not python_enforces_next_step_enum:
        warnings.append(
            "output-schema.md constrains recommended_next_step enum, but src.models.FormalSkillOutput does not strictly enforce it"
        )

    ok = all(payload["matches"] for payload in checks.values())
    return {
        "ok": ok,
        "skill_dir": str(skill_dir),
        "output_schema_path": str(output_schema_path),
        "checks": checks,
        "markdown_contract": markdown_contract,
        "python_contract": python_contract,
        "warnings": warnings,
        "python_enforces_next_step_enum": python_enforces_next_step_enum,
    }

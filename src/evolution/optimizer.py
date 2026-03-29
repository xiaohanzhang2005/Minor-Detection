# 模块说明：
# - 根据 judge 产物编辑 candidate skill 的优化器。
# - 是当前主链在 judge 之后真正还在使用的优化模块。

"""
Skill 优化器
根据评估结果，使用 LLM 优化 skill.md prompt
"""

import difflib
import json
import math
import random
import re
import shutil
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.config import (
    ROOT_DIR,
    SKILLS_DIR,
    OPTIMIZER_MODEL,
    BENCHMARK_TRAIN_PATH,
    resolve_skill_markdown_path,
    set_active_skill_version,
)
from src.utils.llm_client import LLMClient
from src.utils.path_utils import normalize_project_paths, to_relative_posix_path
from src.evolution.evaluator import EvaluationReport

OPTIMIZER_SYSTEM_PROMPT = """You are a prompt optimizer for a teen/adult classification skill.

Revise the existing skill prompt using the evaluation report and its error cases.

Core rules:
1. Preserve the ICBO framework and the JSON output schema.
2. Make narrow, targeted edits instead of broad rewrites.
3. Preserve existing strengths. Do not sacrifice strong precision or recall unless the tradeoff clearly improves F1.
4. Do not fix FP by globally becoming more conservative.
5. Do not fix FN by globally becoming more aggressive.
6. Prefer explicit disambiguation rules for boundary cases over global tone changes.
7. If baseline FN is already zero, any change likely to introduce new FN is high risk and should be avoided.
8. If baseline FP is already zero, any change likely to introduce new FP is high risk and should be avoided.
9. When reducing FP, prefer adult-specific counter-evidence such as adult timeline, adult responsibilities, work context, financial independence, marriage-parenting context, or retrospective narration.
10. When reducing FN, prefer minor-specific evidence such as school stage, dependence on parents-teachers, adolescent time horizon, and youth-specific language patterns.

Output requirement:
- Return the full revised primary skill markdown content only.
- Do not output explanations outside the markdown content.
"""

FORMAL_REFERENCE_FILES = (
    "references/evidence-rules.md",
    "references/icbo-guidelines.md",
    "references/output-schema.md",
    "references/classifier-system.md",
    "references/classifier-user-template.md",
    "references/retrieval-query-template.md",
    "references/schema-repair-template.md",
)

FORMAL_EDITABLE_REFERENCE_FILES = (
    "references/evidence-rules.md",
    "references/classifier-system.md",
    "references/classifier-user-template.md",
    "references/retrieval-query-template.md",
    "references/schema-repair-template.md",
)

FORMAL_EDITABLE_NONREFERENCE_FILES = (
    "scripts/config.py",
)

FORMAL_EDITABLE_FILES = FORMAL_EDITABLE_REFERENCE_FILES + FORMAL_EDITABLE_NONREFERENCE_FILES

FORMAL_REVIEW_FILES = (
    "SKILL.md",
    "references/evidence-rules.md",
    "references/classifier-system.md",
    "references/classifier-user-template.md",
    "references/retrieval-query-template.md",
    "references/schema-repair-template.md",
    "scripts/config.py",
)

FORMAL_DELIVERABLE_VERSION = "minor-detection"
FORMAL_ITERATIONS_DIRNAME = "iterations"
FORMAL_DELIVERABLE_SYNC_ITEMS = (
    "SKILL.md",
    "references",
    "scripts",
    "assets",
    "agents",
)


class SkillOptimizer:
    """
    Skill 优化器
    根据评估报告，使用 LLM 优化 skill prompt
    """
    
    def __init__(
        self,
        llm_client: Optional[LLMClient] = None,
        skills_dir: Optional[str] = None,
    ):
        """
        初始化优化器
        
        Args:
            llm_client: LLM 客户端
            skills_dir: skills 目录路径
        """
        self.llm_client = llm_client or LLMClient(model=OPTIMIZER_MODEL)
        self.skills_dir = Path(skills_dir) if skills_dir else SKILLS_DIR

    def _resolve_skill_paths(self, version: str) -> Tuple[Path, Path]:
        direct_dir = self.skills_dir / version
        if direct_dir.exists():
            return direct_dir, resolve_skill_markdown_path(direct_dir)

        iteration_dir = self._formal_iterations_root() / version
        if iteration_dir.exists():
            return iteration_dir, resolve_skill_markdown_path(iteration_dir)

        skill_dir = direct_dir
        return skill_dir, resolve_skill_markdown_path(skill_dir)

    def _formal_iterations_root(self) -> Path:
        return self.skills_dir / FORMAL_ITERATIONS_DIRNAME / FORMAL_DELIVERABLE_VERSION

    def _is_formal_iteration_version(self, version: str) -> bool:
        return version.startswith(f"{FORMAL_DELIVERABLE_VERSION}-")

    def _target_version_dir(
        self,
        *,
        current_skill_dir: Path,
        current_skill_path: Path,
        new_version: str,
    ) -> Path:
        if self._is_formal_skill_bundle(current_skill_dir) and current_skill_path.name == "SKILL.md":
            root = self._formal_iterations_root()
            root.mkdir(parents=True, exist_ok=True)
            return root / new_version
        return self.skills_dir / new_version

    def _load_reference_materials(self, skill_dir: Path) -> Dict[str, str]:
        materials: Dict[str, str] = {}
        for relative_path in FORMAL_REFERENCE_FILES:
            reference_path = skill_dir / relative_path
            if reference_path.exists():
                materials[relative_path] = reference_path.read_text(encoding="utf-8")
        return materials

    def _is_formal_skill_bundle(self, skill_dir: Path) -> bool:
        return (skill_dir / "references" / "output-schema.md").exists()

    def _strip_markdown_fence(self, content: str) -> str:
        text = str(content or "").strip()
        if not text:
            return text

        fenced_block = re.search(r"```(?:markdown|md)?\s*\n(.*?)\n```", text, flags=re.DOTALL | re.IGNORECASE)
        if fenced_block:
            return fenced_block.group(1).strip()

        if text.startswith("```"):
            lines = text.split("\n")
            if lines and lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            return "\n".join(lines).strip()

        frontmatter_index = text.find("---")
        if frontmatter_index > 0:
            return text[frontmatter_index:].strip()

        heading_index = text.find("# ")
        if heading_index > 0 and frontmatter_index == -1:
            return text[heading_index:].strip()

        return text

    def _request_revised_markdown(self, prompt: str) -> str:
        messages = [
            {"role": "system", "content": OPTIMIZER_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]
        optimized_content = self.llm_client.chat(messages, temperature=0.7)
        return self._strip_markdown_fence(optimized_content)

    def _extract_frontmatter_description(self, content: str) -> str:
        text = str(content or "")
        frontmatter_match = re.match(r"\A---\n(?P<frontmatter>.*?)\n---(?:\n|\Z)", text, flags=re.DOTALL)
        if not frontmatter_match:
            return ""
        frontmatter = frontmatter_match.group("frontmatter")
        description_match = re.search(r"(?m)^description:\s*(.+?)\s*$", frontmatter)
        if not description_match:
            return ""
        return str(description_match.group(1) or "").strip()

    def _normalize_description_value(self, value: str) -> str:
        normalized = str(value or "").strip()
        normalized = re.sub(r"^description:\s*", "", normalized)
        normalized = re.sub(r"\s+", " ", normalized).strip()
        return normalized

    def _replace_frontmatter_description(self, content: str, new_description: str) -> str:
        text = str(content or "")
        normalized_description = self._normalize_description_value(new_description)
        if not normalized_description:
            raise ValueError("description cannot be empty")
        frontmatter_match = re.match(r"\A---\n(?P<frontmatter>.*?)\n---(?P<rest>(?:\n.*)?|\Z)", text, flags=re.DOTALL)
        if not frontmatter_match:
            raise ValueError("SKILL.md is missing YAML frontmatter")
        frontmatter = frontmatter_match.group("frontmatter")
        if not re.search(r"(?m)^description:\s*.+$", frontmatter):
            raise ValueError("SKILL.md frontmatter is missing description")
        updated_frontmatter = re.sub(
            r"(?m)^description:\s*.+$",
            f"description: {normalized_description}",
            frontmatter,
            count=1,
        )
        return f"---\n{updated_frontmatter}\n---{frontmatter_match.group('rest')}"

    def _description_only_skill_content(self, content: str) -> str:
        description = self._extract_frontmatter_description(content)
        if not description:
            return str(content or "")
        return self._replace_frontmatter_description(content, "__DESCRIPTION_PLACEHOLDER__")

    def _is_description_only_skill_change(self, base_content: str, candidate_content: str) -> bool:
        base_description = self._extract_frontmatter_description(base_content)
        candidate_description = self._extract_frontmatter_description(candidate_content)
        if not base_description or not candidate_description:
            return False
        if self._normalize_description_value(base_description) == self._normalize_description_value(candidate_description):
            return False
        return self._description_only_skill_content(base_content) == self._description_only_skill_content(candidate_content)

    def _apply_description_only_revision(self, current_skill: str, revised_content: str) -> str:
        revised_text = str(revised_content or "").strip()
        new_description = self._extract_frontmatter_description(revised_text)
        if not new_description:
            new_description = self._normalize_description_value(revised_text)
        return self._replace_frontmatter_description(current_skill, new_description)

    def _build_reference_prompt_sections(self, reference_materials: Dict[str, str]) -> List[str]:
        if not reference_materials:
            return []

        prompt_parts = [
            "# Bundled Reference Files",
            "- These files are read-only for this optimization round.",
            "- Edit only the primary skill markdown file.",
            "- Treat `references/output-schema.md` as contract-frozen.",
            "- Keep new or still-unstable heuristics in the primary skill markdown first; do not auto-promote them into the stable rulebook during this round.",
        ]

        for relative_path, content in reference_materials.items():
            prompt_parts.extend(
                [
                    "",
                    f"## {relative_path}",
                    "```markdown",
                    content,
                    "```",
                ]
            )

        return prompt_parts

    def _build_trigger_description_only_prompt_sections(
        self,
        *,
        editable_target: str,
        optimization_packet: Dict[str, Any],
    ) -> List[str]:
        if editable_target != "SKILL.md" or str(optimization_packet.get("task_type", "") or "") != "trigger_eval":
            return []

        return [
            "# Trigger Description Editing Rules",
            "- This is a trigger-eval optimization round.",
            "- You must edit only the YAML frontmatter `description` field in `SKILL.md`.",
            "- Do not change the skill title, headings, body sections, execution flow, scripts, output rules, references, or any other file.",
            "- The goal is only to improve whether the agent correctly triggers this skill.",
            "- Keep the description broad enough to catch true trigger intents, but do not make it globally over-trigger.",
            "- Return either the full revised `SKILL.md` or just the revised `description` value; only the description will be applied.",
        ]

    def _build_formal_skill_prompt_sections(
        self,
        *,
        editable_target: str,
    ) -> List[str]:
        if editable_target != "SKILL.md":
            return []

        return [
            "# Formal Skill Editing Rules",
            "- You are editing a bundled formal skill, not a legacy single-file prompt.",
            "- Keep the main language of the markdown as Chinese.",
            "- Keep `description` highly triggerable: it should still activate on minor detection, youth detection, student-profile analysis, school-leaning analysis, risk analysis, probability output, structured evidence output, and similar requests even when the caller does not literally say '未成年人识别'.",
            "- Preserve the skill as an executable operating manual rather than turning it into loose prose.",
            "- Keep the document concise and structured. Prefer compact sections and explicit rules over long narrative explanation.",
            "",
            "## Structural Blocks That Must Remain Present",
            "- skill positioning / scope",
            "- input contract and input limits",
            "- fixed execution flow",
            "- script invocation rules",
            "- output hard constraints",
            "- optimization boundary",
            "",
            "## Fixed Flow Requirements",
            "- Preserve the core order: mode detection -> time handling -> retrieval routing -> user modeling + ICBO -> evidence synthesis + final youth/minor judgment -> schema-constrained output.",
            "- Time handling must stay earlier than retrieval and final evidence synthesis.",
            "- RAG routing must still distinguish external_rag / internal_rag / no_rag behavior.",
            "- User modeling must remain mode-aware and must not disappear.",
            "- The skill must explicitly cover '青少年识别' as well as '未成年人识别'.",
            "",
            "## Script And Resource Rules",
            "- Keep the script references explicit: `scripts/extract_time_features.py` and `scripts/retrieve_cases.py`.",
            "- Keep script invocation conditions explicit rather than implied.",
            "- Keep `references/output-schema.md` as a hard output contract.",
            "- Keep `references/evidence-rules.md` and `references/icbo-guidelines.md` as referenced support material rather than duplicating all their details into SKILL.md.",
            "",
            "## Allowed Improvements",
            "- Make the workflow clearer and tighter.",
            "- Reduce redundancy.",
            "- Strengthen trigger wording in `description` and top-level positioning.",
            "- Clarify input requirements, script call conditions, and output restrictions.",
            "- Improve how user modeling, youth detection, and evidence synthesis are explained.",
            "",
            "## Forbidden Regressions",
            "- Do not remove any supported mode names: `single_session`, `multi_session`, `enriched`.",
            "- Do not remove explicit output fields or relax output-schema discipline.",
            "- Do not move optional script logic into external project code.",
            "- Do not rewrite this into an abstract essay; it must remain operational.",
        ]

    def _build_formal_evidence_rules_prompt_sections(
        self,
        *,
        editable_target: str,
    ) -> List[str]:
        if editable_target != "references/evidence-rules.md":
            return []

        return [
            "# Formal Evidence Rulebook Editing Rules",
            "- You are editing the stable evidence rulebook for a bundled formal skill.",
            "- Keep the main language of the markdown as Chinese.",
            "- Keep the file concise, rule-oriented, and reusable across samples.",
            "- Do not turn this file into an execution manual; execution flow belongs in `SKILL.md`.",
            "- Promote only relatively stable heuristics. Do not overfit to one or two error cases.",
            "- Keep the rulebook aligned with the current `SKILL.md`, but avoid copying the whole workflow into this file.",
            "- Treat `references/output-schema.md` as contract-frozen.",
            "- Treat `references/icbo-guidelines.md` as support material, not an editable target in this round.",
        ]

    def _build_review_diff_markdown(
        self,
        *,
        base_version: str,
        candidate_version: str,
        file_diffs: List[Dict[str, Any]],
    ) -> str:
        lines = [
            "# Formal Skill Review Diff",
            "",
            f"- base_version: `{base_version}`",
            f"- candidate_version: `{candidate_version}`",
            f"- generated_at: `{datetime.now().isoformat()}`",
            "",
            "请人工审核以下差异，再决定 approve / reject。",
        ]

        for item in file_diffs:
            lines.extend(
                [
                    "",
                    f"## {item['file']}",
                    f"- changed: `{str(item['changed']).lower()}`",
                ]
            )
            if item["changed"]:
                lines.extend(
                    [
                        "```diff",
                        item["diff_text"],
                        "```",
                    ]
                )

        return "\n".join(lines) + "\n"

    def create_formal_skill_review_artifact(
        self,
        *,
        base_version: str,
        candidate_version: str,
    ) -> Dict[str, Any]:
        base_dir, _ = self._resolve_skill_paths(base_version)
        candidate_dir, _ = self._resolve_skill_paths(candidate_version)

        review_dir = candidate_dir / "review"
        review_dir.mkdir(parents=True, exist_ok=True)

        candidate_history_path = candidate_dir / "optimization_history.json"
        candidate_history = json.loads(candidate_history_path.read_text(encoding="utf-8")) if candidate_history_path.exists() else {}
        description_only_expected = bool(candidate_history.get("description_only_mode")) or str(candidate_history.get("task_type", "") or "") == "trigger_eval"

        file_diffs: List[Dict[str, Any]] = []
        base_skill_text = ""
        candidate_skill_text = ""
        for relative_path in FORMAL_REVIEW_FILES:
            base_path = base_dir / relative_path
            candidate_path = candidate_dir / relative_path
            base_text = base_path.read_text(encoding="utf-8") if base_path.exists() else ""
            candidate_text = candidate_path.read_text(encoding="utf-8") if candidate_path.exists() else ""
            if relative_path == "SKILL.md":
                base_skill_text = base_text
                candidate_skill_text = candidate_text
            diff_lines = list(
                difflib.unified_diff(
                    base_text.splitlines(),
                    candidate_text.splitlines(),
                    fromfile=f"{base_version}/{relative_path}",
                    tofile=f"{candidate_version}/{relative_path}",
                    lineterm="",
                )
            )
            file_diffs.append(
                {
                    "file": relative_path,
                    "changed": bool(diff_lines),
                    "diff_text": "\n".join(diff_lines),
                }
            )

        non_description_changed_files = [item["file"] for item in file_diffs if item["changed"] and item["file"] != "SKILL.md"]
        description_only_passed = None
        if description_only_expected:
            description_only_passed = self._is_description_only_skill_change(base_skill_text, candidate_skill_text) and not non_description_changed_files

        review_path = review_dir / f"formal_skill_review_vs_{base_version}.md"
        review_path.write_text(
            self._build_review_diff_markdown(
                base_version=base_version,
                candidate_version=candidate_version,
                file_diffs=file_diffs,
            ),
            encoding="utf-8",
        )

        summary_path = review_dir / f"formal_skill_review_vs_{base_version}.json"
        summary_payload = {
            "base_version": base_version,
            "candidate_version": candidate_version,
            "generated_at": datetime.now().isoformat(),
            "task_type": candidate_history.get("task_type"),
            "optimization_focus": candidate_history.get("optimization_focus"),
            "description_only_check": {
                "expected": description_only_expected,
                "passed": description_only_passed,
                "non_description_changed_files": non_description_changed_files,
            },
            "files": [
                {
                    "file": item["file"],
                    "changed": item["changed"],
                }
                for item in file_diffs
            ],
        }
        summary_path.write_text(json.dumps(summary_payload, ensure_ascii=False, indent=2), encoding="utf-8")

        return normalize_project_paths(
            {
                "base_version": base_version,
                "candidate_version": candidate_version,
                "review_diff_path": str(review_path),
                "review_summary_path": str(summary_path),
                "files": summary_payload["files"],
            },
            project_root=ROOT_DIR,
            start=ROOT_DIR,
        )

    def finalize_formal_skill_review(
        self,
        *,
        base_version: str,
        candidate_version: str,
        decision: str,
    ) -> Dict[str, Any]:
        normalized_decision = decision.strip().lower()
        if normalized_decision not in {"approve", "reject"}:
            raise ValueError("decision must be either 'approve' or 'reject'")

        base_dir, _ = self._resolve_skill_paths(base_version)
        candidate_dir, _ = self._resolve_skill_paths(candidate_version)
        if not self._is_formal_skill_bundle(base_dir) or not self._is_formal_skill_bundle(candidate_dir):
            raise ValueError("formal skill review can only be finalized for bundled formal skills")

        review_summary_path = candidate_dir / "review" / f"formal_skill_review_vs_{base_version}.json"
        review_summary = json.loads(review_summary_path.read_text(encoding="utf-8")) if review_summary_path.exists() else {}
        description_only_check = review_summary.get("description_only_check") or {}
        if normalized_decision == "approve" and description_only_check.get("expected") and not description_only_check.get("passed"):
            raise ValueError("trigger-eval review approval blocked: candidate changed content beyond SKILL.md description")

        adopted_version = candidate_version if normalized_decision == "approve" else base_version
        set_active_skill_version(adopted_version)

        review_dir = candidate_dir / "review"
        review_dir.mkdir(parents=True, exist_ok=True)
        decision_path = review_dir / f"review_decision_vs_{base_version}.json"
        decision_payload = {
            "base_version": base_version,
            "candidate_version": candidate_version,
            "decision": normalized_decision,
            "adopted_version": adopted_version,
            "candidate_synced_to_base": False,
            "candidate_adopted_directly": normalized_decision == "approve",
            "decided_at": datetime.now().isoformat(),
        }
        decision_path.write_text(json.dumps(decision_payload, ensure_ascii=False, indent=2), encoding="utf-8")

        return normalize_project_paths(
            {
                "success": True,
                "decision": normalized_decision,
                "base_version": base_version,
                "candidate_version": candidate_version,
                "adopted_version": adopted_version,
                "candidate_synced_to_base": False,
                "candidate_adopted_directly": normalized_decision == "approve",
                "decision_path": str(decision_path),
            },
            project_root=ROOT_DIR,
            start=ROOT_DIR,
        )

    def _copy_skill_bundle(self, source_dir: Path, target_dir: Path) -> None:
        if target_dir.exists():
            raise FileExistsError(f"目标版本目录已存在，拒绝覆盖: {target_dir}")
        shutil.copytree(source_dir, target_dir, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))

    def _sync_formal_candidate_to_base(self, *, candidate_dir: Path, base_dir: Path) -> None:
        for item_name in FORMAL_DELIVERABLE_SYNC_ITEMS:
            candidate_path = candidate_dir / item_name
            if not candidate_path.exists():
                continue

            target_path = base_dir / item_name
            if target_path.exists():
                if target_path.is_dir():
                    shutil.rmtree(target_path)
                else:
                    target_path.unlink()

            if candidate_path.is_dir():
                shutil.copytree(candidate_path, target_path, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
            else:
                target_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(candidate_path, target_path)

    def _make_next_version_name(self, current_version: str, current_skill_path: Path) -> str:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if current_skill_path.name == "SKILL.md":
            return f"{current_version}-{timestamp}"
        return f"teen_detector_v2_{timestamp}"

    def _generate_rule_promotion_suggestions(
        self,
        current_skill: str,
        optimized_skill: str,
        reference_materials: Dict[str, str],
        max_suggestions: int = 6,
    ) -> List[Dict[str, str]]:
        if "references/evidence-rules.md" not in reference_materials:
            return []

        suggestions: List[Dict[str, str]] = []
        seen: set[str] = set()
        diff_lines = difflib.ndiff(current_skill.splitlines(), optimized_skill.splitlines())
        rule_markers = (
            "未成年",
            "成年人",
            "学生",
            "家长",
            "老师",
            "大学",
            "初中",
            "高中",
            "深夜",
            "周末",
            "假期",
            "retrieval",
            "time_evidence",
            "historical_evidence",
        )

        for line in diff_lines:
            if not line.startswith("+ "):
                continue

            candidate = line[2:].strip()
            if (
                not candidate
                or len(candidate) < 12
                or candidate.startswith("#")
                or candidate.startswith("```")
            ):
                continue

            if not any(marker in candidate for marker in rule_markers):
                continue

            normalized_candidate = candidate.lstrip("-*0123456789. ").strip()
            if normalized_candidate in seen:
                continue

            seen.add(normalized_candidate)
            suggestions.append(
                {
                    "candidate_rule": normalized_candidate,
                    "promotion_target": "references/evidence-rules.md",
                    "promotion_rule": "Promote only if the same heuristic remains useful across later optimization rounds and does not cause benchmark regression.",
                }
            )

            if len(suggestions) >= max_suggestions:
                break

        return suggestions
    
    # ── Train Set 对比检索 ─────────────────────────────
    def _retrieve_contrastive_examples(
        self,
        optimization_packet: Dict[str, Any],
        max_per_type: int = 3,
    ) -> str:
        """
        从 Train Set 中检索与错误案例类似的正、负样本，
        帮助 Optimizer 进行归纳推理（Inductive Reasoning）。
        """
        if not BENCHMARK_TRAIN_PATH.exists():
            return ""

        # 一次性加载训练集（已很小，内存安全）
        minor_samples: List[Dict] = []
        adult_samples: List[Dict] = []
        with open(BENCHMARK_TRAIN_PATH, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                s = json.loads(line)
                if s.get("is_minor"):
                    minor_samples.append(s)
                else:
                    adult_samples.append(s)

        parts: List[str] = []

        # 如果有 FP（成年人被误判为未成年），展示真正的成年 vs 未成年对比
        if optimization_packet["fp_count"] > 0:
            parts.append("## 训练集对比参考 — FP 修复灵感\n")
            parts.append("以下是训练集中 **真实成年人** 与 **真实未成年人** 的对比案例，")
            parts.append("请从中归纳区分规律并写入 Prompt：\n")
            for label, pool in [("成年人", adult_samples), ("未成年人", minor_samples)]:
                picked = random.sample(pool, min(max_per_type, len(pool)))
                for i, s in enumerate(picked, 1):
                    conv_preview = s.get("conversation", [])[:4]
                    conv_text = "\n".join(
                        f"  {t['role']}: {t['content'][:80]}" for t in conv_preview
                    )
                    icbo = s.get("icbo_features", {})
                    parts.append(f"### {label} 案例 {i}")
                    parts.append(f"对话片段:\n{conv_text}")
                    if icbo:
                        parts.append(f"ICBO: I={icbo.get('intention','?')[:60]} | "
                                     f"C={icbo.get('cognition','?')[:60]}")
                    parts.append("")

        # 如果有 FN（未成年被漏判），展示真正未成年 vs 成年的对比
        if optimization_packet["fn_count"] > 0:
            parts.append("## 训练集对比参考 — FN 修复灵感\n")
            parts.append("以下是训练集中 **真实未成年人** 与 **真实成年人** 的对比案例，")
            parts.append("请从中归纳未成年人的隐晦特征并补充进 Prompt：\n")
            for label, pool in [("未成年人", minor_samples), ("成年人", adult_samples)]:
                picked = random.sample(pool, min(max_per_type, len(pool)))
                for i, s in enumerate(picked, 1):
                    conv_preview = s.get("conversation", [])[:4]
                    conv_text = "\n".join(
                        f"  {t['role']}: {t['content'][:80]}" for t in conv_preview
                    )
                    icbo = s.get("icbo_features", {})
                    parts.append(f"### {label} 案例 {i}")
                    parts.append(f"对话片段:\n{conv_text}")
                    if icbo:
                        parts.append(f"ICBO: I={icbo.get('intention','?')[:60]} | "
                                     f"C={icbo.get('cognition','?')[:60]}")
                    parts.append("")

        return "\n".join(parts)

    def build_optimization_packet(
        self,
        report: EvaluationReport,
        max_errors: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        从评估报告中提炼出供优化器使用的优化包。
        
        Args:
            report: 评估报告
            max_errors: 最大分析错误数
            
        Returns:
            优化包
        """
        all_errors = list(getattr(report, "errors", []) or [])
        resolved_max_errors = self._resolve_max_errors(report, len(all_errors), max_errors)
        selected_errors, selection_meta = self._select_error_examples(all_errors, max_examples=resolved_max_errors)

        # 统计错误类型
        fp_count = sum(1 for e in all_errors if e["ground_truth"] == "adult")
        fn_count = sum(1 for e in all_errors if e["ground_truth"] == "minor")

        # 提取错误模式
        fp_examples = [e for e in selected_errors if e["ground_truth"] == "adult"]
        fn_examples = [e for e in selected_errors if e["ground_truth"] == "minor"]
        protected_correct_examples = self._select_protected_correct_examples(report)
        
        return {
            "total_errors": len(all_errors),
            "analyzed_errors": len(selected_errors),
            "fp_count": fp_count,
            "fn_count": fn_count,
            "fp_examples": fp_examples,
            "fn_examples": fn_examples,
            "error_selection": selection_meta,
            "max_errors": resolved_max_errors,
            "protected_correct_examples": protected_correct_examples,
            "metrics": {
                "accuracy": report.metrics.accuracy,
                "precision": report.metrics.precision,
                "recall": report.metrics.recall,
                "f1": report.metrics.f1_score,
            },
        }

    def _resolve_max_errors(
        self,
        report: EvaluationReport,
        total_errors: int,
        requested_max_errors: Optional[int],
    ) -> int:
        if total_errors <= 0:
            return 0

        if requested_max_errors is not None:
            return max(1, min(total_errors, int(requested_max_errors)))

        eval_size = int(getattr(report.metrics, "total_samples", 0) or 0)
        if eval_size <= 0:
            return min(total_errors, 12)

        # Auto budget:
        # - lower bound 12 so small runs still expose multiple error patterns
        # - about 12% of the evaluation slice so larger runs see more modes
        # - upper bound 36 to keep optimizer context from bloating
        auto_budget = max(12, math.ceil(eval_size * 0.12))
        auto_budget = min(auto_budget, 36)
        return min(total_errors, auto_budget)

    def _confidence_bucket(self, confidence: float) -> str:
        if confidence < 0.35:
            return "low"
        if confidence < 0.65:
            return "medium"
        return "high"

    def _select_error_examples(
        self,
        errors: List[Dict[str, Any]],
        max_examples: int,
    ) -> tuple[List[Dict[str, Any]], Dict[str, Any]]:
        if not errors:
            return [], {
                "strategy": "none",
                "budget": max_examples,
                "selected_count": 0,
                "total_count": 0,
                "group_counts": {},
            }

        normalized_errors: List[Dict[str, Any]] = []
        group_counts: Dict[str, int] = {}
        for error in errors:
            confidence = float(error.get("confidence", 0.5))
            error_kind = "fp" if error.get("ground_truth") == "adult" else "fn"
            confidence_bucket = self._confidence_bucket(confidence)
            group_key = f"{error_kind}_{confidence_bucket}"

            enriched = dict(error)
            enriched["_group_key"] = group_key
            enriched["_boundary_distance"] = abs(confidence - 0.5)
            normalized_errors.append(enriched)
            group_counts[group_key] = group_counts.get(group_key, 0) + 1

        if len(normalized_errors) <= max_examples:
            selected = sorted(
                normalized_errors,
                key=lambda item: (item["_group_key"], -item["_boundary_distance"], item.get("sample_id", "")),
            )
            return selected, {
                "strategy": "all_errors",
                "budget": max_examples,
                "selected_count": len(selected),
                "total_count": len(normalized_errors),
                "group_counts": dict(sorted(group_counts.items())),
            }

        grouped: Dict[str, List[Dict[str, Any]]] = {}
        for error in normalized_errors:
            grouped.setdefault(error["_group_key"], []).append(error)

        for group_key, items in grouped.items():
            items.sort(key=lambda item: (-item["_boundary_distance"], item.get("sample_id", "")))

        selected: List[Dict[str, Any]] = []
        ordered_keys = sorted(grouped.keys(), key=lambda key: (-len(grouped[key]), key))
        while len(selected) < max_examples and any(grouped.values()):
            for group_key in ordered_keys:
                if len(selected) >= max_examples:
                    break
                if grouped[group_key]:
                    selected.append(grouped[group_key].pop(0))

        return selected, {
            "strategy": "representative_by_error_type_confidence",
            "budget": max_examples,
            "selected_count": len(selected),
            "total_count": len(normalized_errors),
            "group_counts": dict(sorted(group_counts.items())),
        }

    def _select_protected_correct_examples(
        self,
        report: EvaluationReport,
        max_examples: int = 4,
        max_per_label: int = 2,
    ) -> List[Dict[str, Any]]:
        """
        选择需要保护的正确样本。

        优先保留靠近 0.5 决策边界但当前判断正确的样本，避免优化器只修错例，
        却把原本正确的边界判断改坏。
        """
        results = getattr(report, "results", []) or []
        if not results:
            return []

        def _field(item: Any, key: str, default: Any = None) -> Any:
            if isinstance(item, dict):
                return item.get(key, default)
            return getattr(item, key, default)

        grouped: Dict[str, List[Dict[str, Any]]] = {"minor": [], "adult": []}
        for result in results:
            if not _field(result, "is_correct", False):
                continue

            label = "minor" if _field(result, "ground_truth", False) else "adult"
            confidence = float(_field(result, "confidence", 0.5))
            grouped[label].append(
                {
                    "sample_id": _field(result, "sample_id", "unknown"),
                    "label": label,
                    "confidence": confidence,
                    "reasoning": _field(result, "reasoning", "")[:300],
                    "boundary_distance": abs(confidence - 0.5),
                }
            )

        for examples in grouped.values():
            examples.sort(key=lambda item: (item["boundary_distance"], item["sample_id"]))

        protected_examples: List[Dict[str, Any]] = []
        for label in ("minor", "adult"):
            protected_examples.extend(grouped[label][:max_per_label])

        if len(protected_examples) < max_examples:
            remainder: List[Dict[str, Any]] = []
            for label in ("minor", "adult"):
                remainder.extend(grouped[label][max_per_label:])
            remainder.sort(key=lambda item: (item["boundary_distance"], item["sample_id"]))
            protected_examples.extend(remainder[: max_examples - len(protected_examples)])

        protected_examples.sort(key=lambda item: (item["boundary_distance"], item["sample_id"]))
        return protected_examples[:max_examples]

    def _build_optimization_guardrails(self, optimization_packet: Dict[str, Any]) -> str:
        metrics = optimization_packet["metrics"]
        fp_count = optimization_packet["fp_count"]
        fn_count = optimization_packet["fn_count"]
        precision = metrics["precision"]
        recall = metrics["recall"]

        lines = [
            "# Optimization Guardrails",
            "- Make the smallest possible prompt change that targets the observed errors.",
            "- Preserve the current decision boundary unless the current error pattern directly proves it is wrong.",
            "- Prefer explicit disambiguation rules over global shifts such as 'be more conservative' or 'be more aggressive'.",
        ]

        if optimization_packet.get("protected_correct_examples"):
            lines.append("- Preserve the supplied correct boundary-case examples; do not fix errors by breaking them.")

        if fn_count == 0 and fp_count > 0:
            lines.extend(
                [
                    "- Baseline FN is zero. Preserve recall and avoid introducing new FN.",
                    "- Reduce FP using adult-specific counter-evidence and boundary-case rules.",
                    "- Do not solve FP by making the classifier broadly less willing to predict minor.",
                ]
            )

        if fp_count == 0 and fn_count > 0:
            lines.extend(
                [
                    "- Baseline FP is zero. Preserve precision and avoid introducing new FP.",
                    "- Reduce FN using minor-specific evidence and boundary-case rules.",
                    "- Do not solve FN by making the classifier broadly more willing to predict minor.",
                ]
            )

        if fp_count > 0 and fn_count > 0:
            lines.extend(
                [
                    "- Errors exist on both sides. Balance precision and recall; do not over-correct one side.",
                    "- Prefer better boundary disambiguation instead of moving the whole classifier threshold.",
                ]
            )

        lines.extend(
            [
                f"- Baseline precision: {precision:.4f}",
                f"- Baseline recall: {recall:.4f}",
                "- Keep the JSON output schema unchanged.",
            ]
        )

        return "\n".join(lines)
    
    def generate_optimization_prompt(
        self,
        current_skill: str,
        optimization_packet: Dict[str, Any],
        *,
        editable_target: str = "skill.md",
        reference_materials: Optional[Dict[str, str]] = None,
    ) -> str:
        formal_skill_target = editable_target == "SKILL.md"
        formal_evidence_target = editable_target == "references/evidence-rules.md"
        prompt_parts = [
            "# Optimization Topology",
            f"- Primary editable file: `{editable_target}`",
            "- Primary optimization goal: improve the editable file while keeping the external contract stable.",
            "",
        ]

        prompt_parts.extend(
            self._build_trigger_description_only_prompt_sections(
                editable_target=editable_target,
                optimization_packet=optimization_packet,
            )
        )
        prompt_parts.extend(
            self._build_formal_skill_prompt_sections(
                editable_target=editable_target,
            )
        )
        prompt_parts.extend(
            self._build_formal_evidence_rules_prompt_sections(
                editable_target=editable_target,
            )
        )

        prompt_parts.extend(
            [
                "# Current Skill Prompt",
                "```markdown",
                current_skill,
                "```",
                "",
            "# Evaluation Metrics",
            f"- Accuracy: {optimization_packet['metrics']['accuracy']:.2%}",
            f"- Precision: {optimization_packet['metrics']['precision']:.2%}",
            f"- Recall: {optimization_packet['metrics']['recall']:.2%}",
            f"- F1: {optimization_packet['metrics']['f1']:.4f}",
            f"- Total errors: {optimization_packet['total_errors']}",
            f"- Optimizer max_errors budget: {optimization_packet['max_errors']}",
            f"- Selected error examples for optimization: {optimization_packet['analyzed_errors']}",
                f"- False positives (adult -> minor): {optimization_packet['fp_count']}",
                f"- False negatives (minor -> adult): {optimization_packet['fn_count']}",
                "",
                self._build_optimization_guardrails(optimization_packet),
                "",
                "# Error Cases",
            ]
        )

        error_selection = optimization_packet.get("error_selection") or {}
        if error_selection:
            prompt_parts.extend(
                [
                    "## Error Selection",
                    f"- Strategy: {error_selection.get('strategy', 'unknown')}",
                    f"- Budget: {error_selection.get('budget', optimization_packet['max_errors'])}",
                    f"- Selected: {error_selection.get('selected_count', 0)} / {error_selection.get('total_count', 0)}",
                ]
            )
            group_counts = error_selection.get("group_counts") or {}
            if group_counts:
                prompt_parts.append("- Group counts:")
                for key, count in group_counts.items():
                    prompt_parts.append(f"  - {key}: {count}")
            prompt_parts.append("")

        if optimization_packet["fp_examples"]:
            prompt_parts.append("## False Positive Examples")
            for i, ex in enumerate(optimization_packet["fp_examples"], 1):
                prompt_parts.append(f"### Example {i}")
                if ex.get("sample_id"):
                    prompt_parts.append(f"- Sample ID: {ex['sample_id']}")
                if ex.get("_group_key"):
                    prompt_parts.append(f"- Error group: {ex['_group_key']}")
                prompt_parts.append(f"- Confidence: {ex['confidence']:.2f}")
                prompt_parts.append(f"- Reasoning: {ex['reasoning'][:300]}")

        if optimization_packet["fn_examples"]:
            prompt_parts.append("## False Negative Examples")
            for i, ex in enumerate(optimization_packet["fn_examples"], 1):
                prompt_parts.append(f"### Example {i}")
                if ex.get("sample_id"):
                    prompt_parts.append(f"- Sample ID: {ex['sample_id']}")
                if ex.get("_group_key"):
                    prompt_parts.append(f"- Error group: {ex['_group_key']}")
                prompt_parts.append(f"- Confidence: {ex['confidence']:.2f}")
                prompt_parts.append(f"- Reasoning: {ex['reasoning'][:300]}")

        if optimization_packet.get("protected_correct_examples"):
            prompt_parts.append("## Protected Correct Examples")
            prompt_parts.append("These examples are currently correct and close to the decision boundary. Preserve them while fixing the observed errors.")
            for i, ex in enumerate(optimization_packet["protected_correct_examples"], 1):
                prompt_parts.append(f"### Example {i}")
                prompt_parts.append(f"- Sample ID: {ex['sample_id']}")
                prompt_parts.append(f"- Label: {ex['label']}")
                prompt_parts.append(f"- Confidence: {ex['confidence']:.2f}")
                prompt_parts.append(f"- Reasoning: {ex['reasoning'][:300]}")

        prompt_parts.extend(self._build_reference_prompt_sections(reference_materials or {}))

        prompt_parts.extend(
            [
                "",
                "# Task",
                f"Revise only `{editable_target}` so that it addresses the observed errors while preserving current strengths and output format.",
                "Do not rewrite bundled reference files in this round.",
                "If a new heuristic seems promising but not yet fully canonical, keep it in the primary skill markdown; a later gated pass may promote it into `references/evidence-rules.md`.",
                "Return the full revised markdown for the primary editable file only.",
            ]
        )

        if formal_skill_target:
            prompt_parts.insert(
                len(prompt_parts) - 1,
                "For a formal skill target, improve clarity and execution quality without breaking the operating-manual structure.",
            )
        if formal_evidence_target:
            prompt_parts.insert(
                len(prompt_parts) - 1,
                "For the formal evidence rulebook, keep only stable reusable evidence heuristics and avoid sample-specific patch rules.",
            )

        return "\n".join(prompt_parts)

    def _load_packet_json(self, path: Path, default: Any = None) -> Any:
        if not path.exists():
            return default
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _read_packet_text(self, path: Path) -> str:
        if not path.exists():
            return ""
        return path.read_text(encoding="utf-8", errors="replace")

    def _truncate_prompt_text(self, value: Any, max_chars: int = 800) -> str:
        text = str(value or "").strip()
        if len(text) <= max_chars:
            return text
        return text[: max_chars - 3].rstrip() + "..."

    def _packet_conversation_preview(self, sample_input: Dict[str, Any], max_turns: int = 6) -> str:
        lines: List[str] = []
        for turn in (sample_input.get("conversation", []) or [])[:max_turns]:
            if not isinstance(turn, dict):
                continue
            role = str(turn.get("role", "unknown") or "unknown")
            content = self._truncate_prompt_text(turn.get("content", ""), max_chars=180)
            if content:
                lines.append(f"{role}: {content}")
        return "\n".join(lines)

    def _load_packet_examples(self, packets_dir: Path) -> List[Dict[str, Any]]:
        if not packets_dir.exists():
            return []

        examples: List[Dict[str, Any]] = []
        for packet_dir in sorted(path for path in packets_dir.iterdir() if path.is_dir()):
            sample_input = self._load_packet_json(packet_dir / "sample_input.json", default={}) or {}
            gold = self._load_packet_json(packet_dir / "gold.json", default={}) or {}
            agent_output = self._load_packet_json(packet_dir / "agent_output.json", default={}) or {}
            judge_findings = self._load_packet_json(packet_dir / "judge_findings.json", default={}) or {}
            artifact_summary = self._load_packet_json(packet_dir / "artifact_summary.json", default={}) or {}
            tool_trace = self._load_packet_json(packet_dir / "tool_trace.json", default=[]) or []
            observability = self._load_packet_json(packet_dir / "observability.json", default={}) or {}
            transcript_text = self._read_packet_text(packet_dir / "transcript.md")

            parsed_json = agent_output.get("parsed_json") if isinstance(agent_output, dict) else None
            decision = parsed_json.get("decision") if isinstance(parsed_json, dict) else {}
            predicted_bool = decision.get("is_minor") if isinstance(decision, dict) else None
            if predicted_bool is None:
                predicted_label = "unknown"
            else:
                predicted_label = "minor" if bool(predicted_bool) else "adult"
            ground_truth_bool = bool(gold.get("is_minor", False))
            ground_truth_label = "minor" if ground_truth_bool else "adult"

            failure_types = list(judge_findings.get("failure_types", []) or [])
            missing_fields = list(judge_findings.get("missing_fields", []) or [])
            confidence = artifact_summary.get("confidence")
            if confidence is None and isinstance(decision, dict):
                confidence = decision.get("minor_confidence", 0.0)
            try:
                confidence_value = float(confidence or 0.0)
            except Exception:
                confidence_value = 0.0

            sample_preview = self._packet_conversation_preview(sample_input)
            agent_output_preview = self._truncate_prompt_text(
                json.dumps(parsed_json, ensure_ascii=False, indent=2) if parsed_json is not None else agent_output.get("raw_text", ""),
                max_chars=900,
            )
            transcript_preview = self._truncate_prompt_text(transcript_text, max_chars=900)
            tool_trace_preview = self._truncate_prompt_text(json.dumps(tool_trace, ensure_ascii=False, indent=2), max_chars=600)
            observability_preview = self._truncate_prompt_text(json.dumps(observability, ensure_ascii=False, indent=2), max_chars=600)

            reasoning_lines = [
                f"packet_id={packet_dir.name}",
                f"failure_types={', '.join(failure_types) if failure_types else 'none'}",
                f"gold={ground_truth_label}",
                f"predicted={predicted_label}",
                f"confidence={confidence_value:.2f}",
            ]
            if missing_fields:
                reasoning_lines.append(f"missing_fields={', '.join(missing_fields)}")
            if judge_findings.get("schema_valid") is False:
                reasoning_lines.append("schema_valid=false")
            if judge_findings.get("time_handling_detected") is False:
                reasoning_lines.append("time_handling_detected=false")
            if judge_findings.get("script_usage_detected") is False:
                reasoning_lines.append("script_usage_detected=false")
            if judge_findings.get("step_compliant") is False:
                reasoning_lines.append("step_compliant=false")
            if judge_findings.get("observed_issues"):
                reasoning_lines.append("observed_issues=" + ", ".join(judge_findings.get("observed_issues", [])))
            if judge_findings.get("retrieval_mode"):
                reasoning_lines.append(f"retrieval_mode={judge_findings.get('retrieval_mode')}")
            if sample_preview:
                reasoning_lines.append("sample_preview:\n" + sample_preview)
            if agent_output_preview:
                reasoning_lines.append("agent_output_preview:\n" + agent_output_preview)
            if transcript_preview:
                reasoning_lines.append("transcript_preview:\n" + transcript_preview)
            if tool_trace_preview and tool_trace_preview not in {"[]", ""}:
                reasoning_lines.append("tool_trace_preview:\n" + tool_trace_preview)

            examples.append(
                {
                    "packet_id": packet_dir.name,
                    "packet_dir": str(packet_dir),
                    "sample_id": str(
                        artifact_summary.get("sample_id")
                        or gold.get("sample_id")
                        or judge_findings.get("sample_id")
                        or packet_dir.name
                    ),
                    "ground_truth": ground_truth_label,
                    "predicted": predicted_label,
                    "label": ground_truth_label,
                    "confidence": confidence_value,
                    "failure_types": failure_types,
                    "reasoning": "\n".join(reasoning_lines),
                    "missing_fields": missing_fields,
                }
            )

        return examples

    def _build_packet_optimization_packet(
        self,
        *,
        report_payload: Dict[str, Any],
        failure_examples: List[Dict[str, Any]],
        protected_examples: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        metrics = report_payload.get("metrics", {}) or {}
        failure_type_counts = report_payload.get("failure_type_counts", {}) or {}
        fp_examples = [item for item in failure_examples if "false_positive" in (item.get("failure_types") or [])]
        fn_examples = [item for item in failure_examples if "false_negative" in (item.get("failure_types") or [])]

        return {
            "total_errors": int(report_payload.get("total_errors", len(failure_examples)) or 0),
            "analyzed_errors": len(failure_examples),
            "fp_count": int(failure_type_counts.get("false_positive", len(fp_examples)) or 0),
            "fn_count": int(failure_type_counts.get("false_negative", len(fn_examples)) or 0),
            "fp_examples": fp_examples,
            "fn_examples": fn_examples,
            "error_selection": {
                "strategy": "judge_failure_packets",
                "budget": int(report_payload.get("max_errors", len(failure_examples)) or len(failure_examples)),
                "selected_count": len(failure_examples),
                "total_count": int(report_payload.get("total_errors", len(failure_examples)) or len(failure_examples)),
                "group_counts": dict(sorted(failure_type_counts.items())),
            },
            "max_errors": int(report_payload.get("max_errors", len(failure_examples)) or len(failure_examples)),
            "protected_correct_examples": [
                {
                    "sample_id": item["sample_id"],
                    "label": item["label"],
                    "confidence": item["confidence"],
                    "reasoning": item["reasoning"],
                }
                for item in protected_examples
            ],
            "metrics": {
                "accuracy": float(metrics.get("accuracy", 0.0) or 0.0),
                "precision": float(metrics.get("precision", 0.0) or 0.0),
                "recall": float(metrics.get("recall", 0.0) or 0.0),
                "f1": float(metrics.get("f1_score", 0.0) or 0.0),
            },
        }

    def _build_packet_prompt_sections(
        self,
        *,
        editable_target: str,
        report_payload: Dict[str, Any],
        failure_examples: List[Dict[str, Any]],
        protected_examples: List[Dict[str, Any]],
    ) -> str:
        failure_type_counts = report_payload.get("failure_type_counts", {}) or {}
        prompt_parts = [
            "# Judge Packet Evidence",
            "- Use the packet artifacts below as the ground truth evidence for this optimization round.",
            f"- Current editable target: `{editable_target}`",
        ]

        if editable_target == "SKILL.md":
            prompt_parts.append(
                "- Prioritize workflow clarity, step order, time handling, script invocation, and output/schema discipline."
            )
        elif editable_target == "references/evidence-rules.md":
            prompt_parts.append(
                "- Prioritize decision-boundary quality, adult/minor evidence disambiguation, and reusable rule tightening."
            )

        if failure_type_counts:
            prompt_parts.append("- Reported failure counts:")
            for failure_type, count in sorted(failure_type_counts.items()):
                prompt_parts.append(f"  - {failure_type}: {count}")

        prompt_parts.extend(["", "## Failure Packets"])
        if not failure_examples:
            prompt_parts.append("- No selected failure packets.")
        else:
            for index, example in enumerate(failure_examples, 1):
                prompt_parts.extend(
                    [
                        f"### Failure Packet {index}",
                        f"- Packet ID: {example['packet_id']}",
                        f"- Sample ID: {example['sample_id']}",
                        f"- Failure types: {', '.join(example.get('failure_types') or ['unknown'])}",
                        f"- Gold label: {example['ground_truth']}",
                        f"- Predicted label: {example['predicted']}",
                        f"- Confidence: {example['confidence']:.2f}",
                        "```text",
                        example["reasoning"],
                        "```",
                    ]
                )

        prompt_parts.extend(["", "## Protected Packets"])
        if not protected_examples:
            prompt_parts.append("- No protected packets.")
        else:
            for index, example in enumerate(protected_examples, 1):
                prompt_parts.extend(
                    [
                        f"### Protected Packet {index}",
                        f"- Packet ID: {example['packet_id']}",
                        f"- Sample ID: {example['sample_id']}",
                        f"- Label: {example['label']}",
                        f"- Confidence: {example['confidence']:.2f}",
                        "```text",
                        example["reasoning"],
                        "```",
                    ]
                )

        return "\n".join(prompt_parts)

    def resolve_packet_edit_targets(self, report_payload: Dict[str, Any]) -> List[str]:
        failure_type_counts = report_payload.get("failure_type_counts", {}) or {}
        observed_issue_counts = report_payload.get("observed_issue_counts", {}) or {}
        active_failure_types = {
            failure_type
            for failure_type, count in failure_type_counts.items()
            if int(count or 0) > 0
        }
        active_observed_issues = {
            issue
            for issue, count in observed_issue_counts.items()
            if int(count or 0) > 0
        }

        decision_failure_types = {"false_positive", "false_negative"}
        workflow_failure_types = {
            "missing_time_handling",
            "missing_script_usage",
            "step_compliance_failure",
        }
        task_type = str(report_payload.get("task_type", "") or "")
        schema_failure_types = {
            "schema_invalid",
            "fields_missing",
            "output_parse_failure",
        }
        retrieval_issue_types = {
            "retrieval_fallback",
            "retrieval_network_blocked",
        }

        targets: List[str] = []
        if task_type == "trigger_eval":
            return ["SKILL.md"]

        if active_failure_types.intersection(workflow_failure_types):
            targets.append("SKILL.md")
        if active_failure_types.intersection(decision_failure_types):
            targets.append("references/evidence-rules.md")
        if active_failure_types.intersection(schema_failure_types):
            targets.append("references/schema-repair-template.md")
        if active_observed_issues.intersection(retrieval_issue_types):
            targets.append("references/retrieval-query-template.md")

        deduped: List[str] = []
        for target in targets:
            if target not in deduped:
                deduped.append(target)
        return deduped

    def optimize_from_judge_artifacts(
        self,
        *,
        report_path: Path,
        failure_packets_dir: Path,
        protected_packets_dir: Path,
        current_version: str,
        new_version: Optional[str] = None,
        dry_run: bool = False,
    ) -> Dict[str, Any]:
        report_payload = self._load_packet_json(Path(report_path), default={}) or {}
        task_type = str(report_payload.get("task_type", "") or "")
        failure_examples = self._load_packet_examples(Path(failure_packets_dir))
        protected_examples = self._load_packet_examples(Path(protected_packets_dir))

        total_errors = int(report_payload.get("total_errors", len(failure_examples)) or 0)
        if total_errors <= 0 or not failure_examples:
            return {
                "success": True,
                "message": "no errors to optimize",
                "current_version": current_version,
                "edited_files": [],
            }

        edit_targets = self.resolve_packet_edit_targets(report_payload)
        if not edit_targets:
            return {
                "success": True,
                "message": "no editable targets resolved from judge report",
                "current_version": current_version,
                "edited_files": [],
            }

        current_skill_dir, current_skill_path = self._resolve_skill_paths(current_version)
        with open(current_skill_path, "r", encoding="utf-8") as f:
            current_skill = f.read()

        reference_materials = self._load_reference_materials(current_skill_dir)
        optimization_packet = self._build_packet_optimization_packet(
            report_payload=report_payload,
            failure_examples=failure_examples,
            protected_examples=protected_examples,
        )
        packet_prompt_sections = self._build_packet_prompt_sections(
            editable_target="SKILL.md" if "SKILL.md" in edit_targets else edit_targets[0],
            report_payload=report_payload,
            failure_examples=failure_examples,
            protected_examples=protected_examples,
        )
        contrastive_text = self._retrieve_contrastive_examples(optimization_packet)

        edited_files: Dict[str, str] = {}
        for editable_target in edit_targets:
            target_path = current_skill_dir / editable_target if editable_target != "SKILL.md" else current_skill_path
            current_content = current_skill if editable_target == "SKILL.md" else target_path.read_text(encoding="utf-8")
            target_reference_materials = (
                reference_materials
                if editable_target == "SKILL.md"
                else {path: content for path, content in reference_materials.items() if path != editable_target}
            )

            optimization_prompt = self.generate_optimization_prompt(
                current_content,
                optimization_packet,
                editable_target=editable_target,
                reference_materials=target_reference_materials,
            )
            optimization_prompt += "\n\n" + self._build_packet_prompt_sections(
                editable_target=editable_target,
                report_payload=report_payload,
                failure_examples=failure_examples,
                protected_examples=protected_examples,
            )
            if contrastive_text and editable_target == "references/evidence-rules.md":
                optimization_prompt += "\n\n" + contrastive_text
            if editable_target != "SKILL.md" and "SKILL.md" in edited_files:
                optimization_prompt += (
                    "\n\n# Updated Primary Skill Draft\n"
                    "```markdown\n"
                    f"{edited_files['SKILL.md']}\n"
                    "```"
                    "\n\n# Task Addendum\n"
                    f"Revise only `{editable_target}` so it stays aligned with the updated `SKILL.md` draft above.\n"
                    "Keep only stable reusable heuristics. Do not copy the whole execution workflow into this file.\n"
                    "Return the full revised markdown for this editable file only."
                )

            revised_content = self._request_revised_markdown(optimization_prompt)
            if task_type == "trigger_eval" and editable_target == "SKILL.md":
                revised_content = self._apply_description_only_revision(current_content, revised_content)
            edited_files[editable_target] = revised_content

        if dry_run:
            return {
                "success": True,
                "dry_run": True,
                "current_version": current_version,
                "edited_files": sorted(edited_files.keys()),
                "edit_targets": edit_targets,
                "optimization_packet": optimization_packet,
                "packet_prompt_preview": packet_prompt_sections,
            }

        if new_version is None:
            new_version = self._make_next_version_name(current_version, current_skill_path)

        new_skill_dir = self.skills_dir / new_version
        self._copy_skill_bundle(current_skill_dir, new_skill_dir)

        for relative_path, content in edited_files.items():
            write_path = new_skill_dir / relative_path if relative_path != "SKILL.md" else resolve_skill_markdown_path(new_skill_dir)
            write_path.parent.mkdir(parents=True, exist_ok=True)
            with open(write_path, "w", encoding="utf-8") as f:
                f.write(content)

        history_path = new_skill_dir / "optimization_history.json"
        history_payload = normalize_project_paths(
            {
                "parent_version": current_version,
                "optimization_time": datetime.now().isoformat(),
                "optimizer_mode": "judge_packets",
                "report_path": str(report_path),
                "failure_packets_dir": str(failure_packets_dir),
                "protected_packets_dir": str(protected_packets_dir),
                "edited_files": sorted(edited_files.keys()),
                "failure_type_counts": report_payload.get("failure_type_counts", {}),
                "metrics": report_payload.get("metrics", {}),
                "task_type": task_type,
                "optimization_focus": report_payload.get("optimization_focus"),
                "description_only_mode": task_type == "trigger_eval",
            },
            project_root=ROOT_DIR,
            start=ROOT_DIR,
        )
        with open(history_path, "w", encoding="utf-8") as f:
            json.dump(history_payload, f, ensure_ascii=False, indent=2)

        return normalize_project_paths(
            {
                "success": True,
                "current_version": current_version,
                "new_version": new_version,
                "new_skill_dir": str(new_skill_dir),
                "new_skill_path": str(resolve_skill_markdown_path(new_skill_dir)),
                "edited_files": sorted(edited_files.keys()),
                "edit_targets": edit_targets,
                "optimization_packet": optimization_packet,
            },
            project_root=ROOT_DIR,
            start=ROOT_DIR,
        )
    
    def optimize(
        self,
        report: EvaluationReport,
        current_version: str = "teen_detector_v1",
        new_version: Optional[str] = None,
        max_errors: Optional[int] = None,
        dry_run: bool = False,
    ) -> Dict[str, Any]:
        """
        执行优化
        
        Args:
            report: 评估报告
            current_version: 当前版本目录名
            new_version: 新版本目录名（自动生成如果不指定）
            dry_run: 是否只预览，不实际创建文件
            
        Returns:
            优化结果
        """
        print("[INFO] 开始 Skill 优化...")
        
        # 1. 读取当前 skill
        current_skill_dir, current_skill_path = self._resolve_skill_paths(current_version)
        
        with open(current_skill_path, "r", encoding="utf-8") as f:
            current_skill = f.read()
        reference_materials = self._load_reference_materials(current_skill_dir)
        formal_bundle = self._is_formal_skill_bundle(current_skill_dir)
        editable_asset_files = [
            relative_path
            for relative_path in FORMAL_EDITABLE_FILES
            if formal_bundle and (current_skill_dir / relative_path).exists()
        ]
        readonly_reference_files = sorted(
            path for path in reference_materials.keys() if path not in set(FORMAL_EDITABLE_REFERENCE_FILES)
        )
        
        print(f"[INFO] 加载当前 skill: {current_version}")
        
        # 2. 构建优化包
        optimization_packet = self.build_optimization_packet(report, max_errors=max_errors)
        print(f"[INFO] 优化包: FP={optimization_packet['fp_count']}, FN={optimization_packet['fn_count']}")
        
        if optimization_packet["total_errors"] == 0:
            print("[OK] 没有错误，无需优化")
            return {
                "success": True,
                "message": "no errors to optimize",
                "current_version": current_version,
            }
        
        # 3. 从 Train Set 检索对比案例（归纳推理材料）
        contrastive_text = self._retrieve_contrastive_examples(optimization_packet)
        if contrastive_text:
            print(f"[INFO] 从 Train Set 检索到对比案例 ({len(contrastive_text)} 字符)")

        # 4. 生成优化 prompt
        optimization_prompt = self.generate_optimization_prompt(
            current_skill,
            optimization_packet,
            editable_target=current_skill_path.name,
            reference_materials=reference_materials,
        )
        if contrastive_text:
            optimization_prompt += "\n\n" + contrastive_text
        
        # 5. 调用 LLM 优化
        print("[INFO] 调用 LLM 生成优化方案...")
        optimized_skill = self._request_revised_markdown(optimization_prompt)
        
        print(f"[OK] 生成优化 skill ({len(optimized_skill)} 字符)")

        rule_promotion_suggestions = self._generate_rule_promotion_suggestions(
            current_skill,
            optimized_skill,
            reference_materials,
        )
        edited_files: Dict[str, str] = {
            current_skill_path.name: optimized_skill,
        }

        for relative_path in editable_asset_files:
            target_path = current_skill_dir / relative_path
            current_reference = (
                reference_materials.get(relative_path)
                if relative_path in reference_materials
                else target_path.read_text(encoding="utf-8")
            ) or ""
            companion_references = {
                path: content
                for path, content in reference_materials.items()
                if path != relative_path
            }
            companion_prompt = self.generate_optimization_prompt(
                current_reference,
                optimization_packet,
                editable_target=relative_path,
                reference_materials=companion_references,
            )
            companion_prompt += (
                "\n\n# Updated Primary Skill Draft\n"
                "```markdown\n"
                f"{optimized_skill}\n"
                "```"
                "\n\n# Task Addendum\n"
                f"Revise only `{relative_path}` so it stays aligned with the updated `SKILL.md` draft above.\n"
                "Keep only stable evidence heuristics. Do not copy the whole execution flow into this file.\n"
                "Return the full revised markdown for this editable file only."
            )
            optimized_reference = self._request_revised_markdown(companion_prompt)
            edited_files[relative_path] = optimized_reference
            print(f"[OK] 生成优化 reference ({relative_path}, {len(optimized_reference)} 字符)")
        
        if dry_run:
            print("\n[Dry Run] 优化内容预览:")
            print("-" * 50)
            print(optimized_skill[:500] + "..." if len(optimized_skill) > 500 else optimized_skill)
            print("-" * 50)
            return {
                "success": True,
                "dry_run": True,
                "optimized_skill": optimized_skill,
                "target_file": current_skill_path.name,
                "edited_files": sorted(edited_files.keys()),
                "readonly_reference_files": readonly_reference_files,
                "rule_promotion_suggestions": rule_promotion_suggestions,
            }
        
        # 6. 创建新版本目录
        if new_version is None:
            new_version = self._make_next_version_name(current_version, current_skill_path)
        
        new_skill_dir = self._target_version_dir(
            current_skill_dir=current_skill_dir,
            current_skill_path=current_skill_path,
            new_version=new_version,
        )
        self._copy_skill_bundle(current_skill_dir, new_skill_dir)

        new_skill_path = resolve_skill_markdown_path(new_skill_dir)
        with open(new_skill_path, "w", encoding="utf-8") as f:
            f.write(optimized_skill)
        for relative_path, content in edited_files.items():
            if relative_path == current_skill_path.name:
                continue
            target_path = new_skill_dir / relative_path
            target_path.parent.mkdir(parents=True, exist_ok=True)
            with open(target_path, "w", encoding="utf-8") as f:
                f.write(content)
        
        print(f"[OK] 新版本已保存: {new_version}")
        
        # 7. 记录优化历史
        history_path = new_skill_dir / "optimization_history.json"
        history = {
            "parent_version": current_version,
            "optimization_time": datetime.now().isoformat(),
            "target_file": current_skill_path.name,
            "edited_files": sorted(edited_files.keys()),
            "readonly_reference_files": readonly_reference_files,
            "optimization_packet": {
                "total_errors": optimization_packet["total_errors"],
                "fp_count": optimization_packet["fp_count"],
                "fn_count": optimization_packet["fn_count"],
            },
            "parent_metrics": optimization_packet["metrics"],
            "rule_promotion_suggestions": rule_promotion_suggestions,
        }
        with open(history_path, "w", encoding="utf-8") as f:
            json.dump(history, f, ensure_ascii=False, indent=2)
        
        return normalize_project_paths(
            {
                "success": True,
                "current_version": current_version,
                "new_version": new_version,
                "new_skill_path": str(new_skill_path),
                "target_file": current_skill_path.name,
                "edited_files": sorted(edited_files.keys()),
                "readonly_reference_files": readonly_reference_files,
                "rule_promotion_suggestions": rule_promotion_suggestions,
                "optimization_packet": optimization_packet,
            },
            project_root=ROOT_DIR,
            start=ROOT_DIR,
        )
    
    def rollback(self, version: str) -> bool:
        """
        回滚到指定版本（更新活跃版本指针）
        
        Args:
            version: 要回滚到的版本
            
        Returns:
            是否成功
        """
        try:
            version_dir, version_path = self._resolve_skill_paths(version)
        except FileNotFoundError:
            print(f"[ERROR] 版本不存在: {version}")
            return False
        
        try:
            if self._is_formal_skill_bundle(version_dir) and version_path.name == "SKILL.md" and version_dir.parent.name == FORMAL_DELIVERABLE_VERSION:
                set_active_skill_version(FORMAL_DELIVERABLE_VERSION)
            else:
                set_active_skill_version(version)
            print(f"[OK] 已激活 skill 版本: {version}")
            return True
        except FileNotFoundError:
            print(f"[ERROR] 版本缺少技能 markdown: {version}")
            return False
    
    def list_versions(self) -> List[Dict[str, Any]]:
        """列出所有 skill 版本"""
        versions = []
        
        roots = [self.skills_dir]
        iteration_root = self._formal_iterations_root()
        if iteration_root.exists():
            roots.append(iteration_root)

        for root in roots:
            for path in root.iterdir():
                if not path.is_dir():
                    continue

                try:
                    skill_md = resolve_skill_markdown_path(path)
                except FileNotFoundError:
                    continue

                history = path / "optimization_history.json"

                version_info = {
                    "name": path.name,
                    "has_skill": skill_md.exists(),
                    "target_file": skill_md.name,
                    "parent": None,
                    "location": to_relative_posix_path(path, ROOT_DIR),
                }

                if history.exists():
                    with open(history, "r", encoding="utf-8") as f:
                        h = json.load(f)
                        version_info["parent"] = h.get("parent_version")
                        version_info["optimization_time"] = h.get("optimization_time")

                versions.append(version_info)
        
        return sorted(versions, key=lambda x: x["name"])

def run_optimization_cycle(
    current_version: str = "teen_detector_v1",
    max_samples: Optional[int] = None,
    optimization_max_errors: Optional[int] = None,
    sample_strategy: str = "sequential",
    sample_seed: int = 42,
    dry_run: bool = True,
    auto_rollback: bool = True,
    min_f1_improvement: float = 0.0,
    retriever: Any = None,
    use_memory: bool = False,
    baseline_report: Optional[EvaluationReport] = None,
    activate_accepted_version: bool = True,
) -> Dict[str, Any]:
    """
    运行一个完整的优化周期: 评估 -> 分析 -> 优化 -> 验证 -> 回滚（如果变差）
    
    Args:
        current_version: 当前版本
        max_samples: 最大评估样本数
        dry_run: 是否只预览
        auto_rollback: 是否启用自动回滚（新版本F1下降时删除新版本）
        min_f1_improvement: 最小 F1 改进阈值（低于此值视为没有改进）
        retriever: SemanticRetriever 实例（可选），传入则评估时带 RAG
        baseline_report: 已有的当前版本评估报告；若提供则直接复用，避免重复评估当前版本
        
    Returns:
        优化结果
    """
    from src.evolution.evaluator import SkillEvaluator
    from src.executor import ExecutorSkill
    from src.config import SKILLS_DIR
    import shutil
    
    print("=" * 60)
    print("[INFO] Skill 优化周期")
    print("=" * 60)
    
    # 1. 获取当前版本的评估结果
    optimizer = SkillOptimizer()
    _, skill_path = optimizer._resolve_skill_paths(current_version)

    if baseline_report is None:
        executor = ExecutorSkill(skill_path=str(skill_path))
        print("\n[INFO] Step 1: 在验证集上评估当前版本...")
        evaluator = SkillEvaluator(
            executor=executor,
            skill_version=current_version,
            retriever=retriever,
            use_memory=use_memory,
        )
        report = evaluator.evaluate(
            max_samples=max_samples,
            sample_strategy=sample_strategy,
            sample_seed=sample_seed,
            use_test_set=False,
        )
        baseline_report_reused = False
    else:
        print("\n[INFO] Step 1: 复用上游提供的当前版本评估结果...")
        report = baseline_report
        baseline_report_reused = True

    baseline_f1 = report.metrics.f1_score
    print(f"   当前版本 F1: {baseline_f1:.4f}")
    
    # 3. 优化
    print("\n[INFO] Step 2: 生成优化方案...")
    result = optimizer.optimize(
        report,
        current_version,
        max_errors=optimization_max_errors,
        dry_run=dry_run,
    )
    
    # dry_run 模式下直接返回
    if dry_run or not result.get("success") or "new_version" not in result:
        return result
    
    # 4. 在验证集上评估新版本
    new_version = result["new_version"]
    print(f"\n[INFO] Step 3: 在验证集上评估新版本 ({new_version})...")
    
    new_skill_path = Path(result.get("new_skill_path") or "")
    if not new_skill_path.exists():
        _, new_skill_path = optimizer._resolve_skill_paths(new_version)
    new_executor = ExecutorSkill(skill_path=str(new_skill_path))
    new_evaluator = SkillEvaluator(
        executor=new_executor,
        skill_version=new_version,
        retriever=retriever,
        use_memory=use_memory,
    )
    new_report = new_evaluator.evaluate(
        max_samples=max_samples,
        sample_strategy=sample_strategy,
        sample_seed=sample_seed,
        use_test_set=False,
    )
    
    new_f1 = new_report.metrics.f1_score
    f1_delta = new_f1 - baseline_f1
    
    print(f"   新版本 F1: {new_f1:.4f}")
    print(f"   F1 变化: {f1_delta:+.4f}")
    
    result["baseline_f1"] = baseline_f1
    result["new_f1"] = new_f1
    result["f1_delta"] = f1_delta
    result["baseline_report_reused"] = baseline_report_reused
    result["_new_report"] = new_report

    formal_pending_review_mode = not activate_accepted_version
    rollback_threshold = 0.0 if formal_pending_review_mode else min_f1_improvement
    
    # 5. 自动回滚检查
    if auto_rollback and f1_delta < rollback_threshold:
        threshold_label = "0.0000 (formal non-regression gate)" if formal_pending_review_mode else f"{min_f1_improvement:.4f}"
        print(f"\n[WARN] Step 4: 自动回滚 - 新版本未能改进 F1 (delta={f1_delta:.4f} < {threshold_label})")
        
        # 删除新版本目录
        new_version_dir = new_skill_path.parent
        if new_version_dir.exists():
            shutil.rmtree(new_version_dir)
            print(f"   [DEL] 已删除: {new_version}")
        
        result["rolled_back"] = True
        if formal_pending_review_mode:
            result["rollback_reason"] = f"F1 回退 (delta={f1_delta:.4f})"
        else:
            result["rollback_reason"] = f"F1 未改进 (delta={f1_delta:.4f})"
        return result
    
    result["rolled_back"] = False
    if activate_accepted_version:
        if optimizer._is_formal_iteration_version(new_version):
            set_active_skill_version(FORMAL_DELIVERABLE_VERSION)
            result["active_version"] = FORMAL_DELIVERABLE_VERSION
        else:
            set_active_skill_version(new_version)
            result["active_version"] = new_version
        print(f"\n[OK] Step 4: 新版本已接受 - F1 改进 {f1_delta:+.4f}")
        result["review_required"] = False
        result["adoption_status"] = "accepted"
    else:
        print(f"\n[PENDING] Step 4: 新版本通过评测，但等待人工审核后再采纳")
        result["active_version"] = current_version
        result["review_required"] = True
        result["adoption_status"] = "pending_review"

    return result


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Skill 优化器")
    parser.add_argument("--version", default="teen_detector_v1", help="当前版本")
    parser.add_argument("--max-samples", type=int, default=10, help="最大评估样本数")
    parser.add_argument("--run", action="store_true", help="实际执行优化（默认 dry-run）")
    parser.add_argument("--list", action="store_true", help="列出所有版本")
    parser.add_argument("--review-base-version", help="formal review 的基线版本")
    parser.add_argument("--review-candidate-version", help="formal review 的候选版本")
    parser.add_argument("--review-decision", choices=["approve", "reject"], help="人工审核结论")
    
    args = parser.parse_args()
    
    if args.list:
        optimizer = SkillOptimizer()
        versions = optimizer.list_versions()
        print("[INFO] Skill 版本列表:")
        for v in versions:
            parent = f" (from {v['parent']})" if v.get('parent') else ""
            print(f"  - {v['name']}{parent}")
    elif args.review_base_version and args.review_candidate_version and args.review_decision:
        optimizer = SkillOptimizer()
        result = optimizer.finalize_formal_skill_review(
            base_version=args.review_base_version,
            candidate_version=args.review_candidate_version,
            decision=args.review_decision,
        )
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        result = run_optimization_cycle(
            current_version=args.version,
            max_samples=args.max_samples,
            dry_run=not args.run,
        )

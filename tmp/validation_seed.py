# 模块说明：
# - 生成 Mode A 自迭代验收用的临时坏 baseline。
# - 用于验证优化闭环是否真能修复故意制造的错误。

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

from src.config import ROOT_DIR
from src.skill_loop.versioning import build_stable_version_name, build_stamped_stable_version_name
from src.utils.path_utils import to_relative_posix_path


VALIDATION_SYSTEM_OVERRIDE = """

## Temporary Validation Seed Override

以下规则仅用于 Mode A 自迭代验收的临时坏基线，用来稳定制造“漏判未成年人”的可优化错误；它们覆盖上面的通用判定偏好：

- 如果当前输入里没有“明确年龄 < 18”或“明确学段=小学/初中/高中”的硬证据，默认先按成人处理。
- `宿舍`、`寝室`、`班级`、`老师`、`家长`、`成绩`、`作业`、`晚自习` 等校园/监护线索在本临时基线中一律只算弱线索，不足以单独判定为未成年人。
- 时间证据和检索证据只能记录，不能把一个原本偏成人或证据不足的样本翻成 `is_minor=true`。
- 当未成年和成人信号接近时，优先返回 `is_minor=false`，并把置信度压低。

这是验证专用降级规则，不是正式生产规则。
""".lstrip()


VALIDATION_EVIDENCE_OVERRIDE = """

## Temporary Validation Seed Override

以下临时规则仅用于 Mode A 自迭代验收的坏基线：

- 除非出现“明确年龄 < 18”或“明确学段=小学/初中/高中”，否则不要因为校园相关词汇直接判未成年人。
- `宿舍`、`寝室`、`同桌`、`老师`、`家长`、`成绩`、`作业`、`晚自习`、`月考` 在本临时基线中全部降级为弱线索。
- 深夜活跃、节假日特征、检索到相似未成年案例，都只能写入辅助证据，不得推动最终结论翻到未成年人。
- 当证据不足时，优先保守到成人侧，制造可供优化器修复的 false negative。

这是验证专用降级规则，不是正式生产规则。
""".lstrip()


@dataclass(frozen=True)
class ValidationSeedPaths:
    version_name: str
    run_tag: str
    base_name: str
    source_dir: Path
    cleanup_skills_command: str

    def to_payload(self) -> dict:
        return {
            "version_name": self.version_name,
            "run_tag": self.run_tag,
            "base_name": self.base_name,
            "source_dir": to_relative_posix_path(self.source_dir, ROOT_DIR),
            "cleanup_skills_command": self.cleanup_skills_command,
        }


def build_validation_seed_version_name(
    *,
    base_name: str = "minor-detection-verifya",
    major: int = 0,
    minor: int = 1,
    patch: int = 0,
    run_tag: Optional[str] = None,
) -> tuple[str, str]:
    resolved_run_tag = str(run_tag or "").strip() or datetime.now().strftime("%Y%m%d_%H%M%S")
    stable_name = build_stable_version_name(base_name, major, minor, patch)
    return build_stamped_stable_version_name(stable_name, resolved_run_tag), resolved_run_tag


def create_mode_a_validation_seed(
    *,
    source_dir: Path,
    output_root: Path,
    base_name: str = "minor-detection-verifya",
    run_tag: Optional[str] = None,
) -> ValidationSeedPaths:
    version_name, resolved_run_tag = build_validation_seed_version_name(base_name=base_name, run_tag=run_tag)
    target_dir = output_root / version_name
    if target_dir.exists():
        shutil.rmtree(target_dir)
    shutil.copytree(source_dir, target_dir, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))

    _append_override(target_dir / "references" / "classifier-system.md", VALIDATION_SYSTEM_OVERRIDE)
    _append_override(target_dir / "references" / "evidence-rules.md", VALIDATION_EVIDENCE_OVERRIDE)

    metadata = {
        "kind": "mode_a_validation_seed",
        "version_name": version_name,
        "run_tag": resolved_run_tag,
        "base_name": base_name,
        "source_dir": str(source_dir),
        "target_dir": str(target_dir),
        "edited_files": [
            "references/classifier-system.md",
            "references/evidence-rules.md",
        ],
        "notes": [
            "This directory is an isolated temporary bad baseline for Mode A iteration validation.",
            "Only markdown rule assets were changed; Python runtime files were copied unchanged.",
        ],
    }
    (target_dir / ".validation_seed.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    cleanup_skills_command = (
        f"python scripts/cleanup_skill_versions.py --base-name {base_name} "
        f"--keep-latest-stable 0 --only-run-tag {resolved_run_tag}"
    )
    return ValidationSeedPaths(
        version_name=version_name,
        run_tag=resolved_run_tag,
        base_name=base_name,
        source_dir=target_dir,
        cleanup_skills_command=cleanup_skills_command,
    )


def build_mode_a_validation_command(
    *,
    seed: ValidationSeedPaths,
    dataset_path: Path,
    max_rounds: int = 1,
    max_samples: int = 3,
    sample_strategy: str = "stratified",
    sample_seed: int = 42,
    execution_mode: str = "bypass",
    codex_model: str = "gpt-5.4",
    timeout_sec: int = 600,
) -> str:
    dataset_rel = to_relative_posix_path(dataset_path, ROOT_DIR)
    source_rel = to_relative_posix_path(seed.source_dir, ROOT_DIR)
    return (
        "python scripts/run_skill_iteration_loop.py "
        f"--baseline-version {seed.version_name} "
        f"--baseline-source-dir {source_rel} "
        "--refresh-baseline-version "
        f"--dataset {dataset_rel} "
        f"--max-rounds {max_rounds} "
        f"--max-samples {max_samples} "
        f"--sample-strategy {sample_strategy} "
        f"--sample-seed {sample_seed} "
        f"--execution-mode {execution_mode} "
        f"--codex-model {codex_model} "
        f"--timeout-sec {timeout_sec}"
    )


def build_mode_a_validation_payload(
    *,
    source_dir: Path,
    output_root: Path,
    dataset_path: Path,
    base_name: str = "minor-detection-verifya",
    run_tag: Optional[str] = None,
    max_rounds: int = 1,
    max_samples: int = 3,
    sample_strategy: str = "stratified",
    sample_seed: int = 42,
    execution_mode: str = "bypass",
    codex_model: str = "gpt-5.4",
    timeout_sec: int = 600,
) -> dict:
    seed = create_mode_a_validation_seed(
        source_dir=source_dir,
        output_root=output_root,
        base_name=base_name,
        run_tag=run_tag,
    )
    source_parent_rel = to_relative_posix_path(seed.source_dir.parent, ROOT_DIR)
    return {
        "seed": seed.to_payload(),
        "run_command": build_mode_a_validation_command(
            seed=seed,
            dataset_path=dataset_path,
            max_rounds=max_rounds,
            max_samples=max_samples,
            sample_strategy=sample_strategy,
            sample_seed=sample_seed,
            execution_mode=execution_mode,
            codex_model=codex_model,
            timeout_sec=timeout_sec,
        ),
        "cleanup": {
            "seed_source_dir": to_relative_posix_path(seed.source_dir, ROOT_DIR),
            "seed_parent_dir": source_parent_rel,
            "cleanup_seed_source_command": (
                f"Remove-Item -Recurse -Force '{to_relative_posix_path(seed.source_dir, ROOT_DIR)}'"
            ),
            "cleanup_skills_command": seed.cleanup_skills_command,
        },
    }


def _append_override(path: Path, appendix: str) -> None:
    original = path.read_text(encoding="utf-8").rstrip()
    path.write_text(f"{original}\n\n{appendix.strip()}\n", encoding="utf-8")


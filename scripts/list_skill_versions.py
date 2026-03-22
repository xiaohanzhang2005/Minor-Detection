# 模块说明：
# - 列出 skill 版本库存和 cleanup preview。
# - 是删除旧 skill 目录前的安全辅助工具。

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).parent.parent.absolute()
sys.path.insert(0, str(ROOT_DIR))

from src.config import SKILLS_DIR, get_active_skill_version
from src.skill_loop.versioning import build_version_inventory


def main() -> None:
    parser = argparse.ArgumentParser(description="List managed skill snapshot versions.")
    parser.add_argument("--base-name", default="minor-detection")
    parser.add_argument("--skills-root", default=str(SKILLS_DIR))
    parser.add_argument("--keep-latest-stable", type=int, default=2)
    parser.add_argument("--keep-candidates", action="store_true", help="Preview inventory without deleting rc snapshots.")
    parser.add_argument("--only-run-tag", default="")
    parser.add_argument("--exclude-run-tag", default="")
    args = parser.parse_args()

    skills_root = Path(args.skills_root)
    payload = build_version_inventory(
        skills_root,
        base_name=args.base_name,
        active_version=get_active_skill_version(),
        keep_latest_stable=args.keep_latest_stable,
        delete_candidates=not args.keep_candidates,
        only_run_tag=args.only_run_tag,
        exclude_run_tag=args.exclude_run_tag,
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

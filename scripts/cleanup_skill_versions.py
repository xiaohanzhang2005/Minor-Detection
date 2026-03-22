# 模块说明：
# - 清理旧 skill snapshot 的命令行工具。
# - 属于当前主链的维护工具，不是历史残留。

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).parent.parent.absolute()
sys.path.insert(0, str(ROOT_DIR))

from src.config import SKILLS_DIR
from src.skill_loop.versioning import delete_skill_versions, select_cleanup_targets


def main() -> None:
    parser = argparse.ArgumentParser(description="Clean old managed skill snapshot versions.")
    parser.add_argument("--base-name", default="minor-detection")
    parser.add_argument("--skills-root", default=str(SKILLS_DIR))
    parser.add_argument("--keep-latest-stable", type=int, default=2)
    parser.add_argument("--keep-candidates", action="store_true", help="Do not delete rc snapshot directories.")
    parser.add_argument("--only-run-tag", default="", help="Only delete versions that belong to this run tag (YYYYMMDD_HHMMSS).")
    parser.add_argument("--exclude-run-tag", default="", help="Never delete versions that belong to this run tag.")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    skills_root = Path(args.skills_root)
    targets = select_cleanup_targets(
        skills_root,
        base_name=args.base_name,
        keep_latest_stable=args.keep_latest_stable,
        delete_candidates=not args.keep_candidates,
        only_run_tag=args.only_run_tag,
        exclude_run_tag=args.exclude_run_tag,
    )
    removed = [] if args.dry_run else delete_skill_versions(targets)
    payload = {
        "base_name": args.base_name,
        "skills_root": str(skills_root),
        "keep_latest_stable": args.keep_latest_stable,
        "delete_candidates": not args.keep_candidates,
        "only_run_tag": args.only_run_tag or None,
        "exclude_run_tag": args.exclude_run_tag or None,
        "dry_run": args.dry_run,
        "targets": [str(path) for path in targets],
        "removed": [str(path) for path in removed],
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

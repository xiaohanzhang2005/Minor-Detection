from __future__ import annotations

import json
from pathlib import Path


TARGET_PATH = Path("data/知识问答数据库/youth_knowledge_qa.jsonl")


def main() -> int:
    if not TARGET_PATH.exists():
        raise FileNotFoundError(f"Missing file: {TARGET_PATH}")

    updated_lines: list[str] = []
    changed = 0

    with TARGET_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                updated_lines.append(line)
                continue

            obj = json.loads(line)
            user_persona = obj.get("user_persona", {}) or {}
            age = user_persona.get("age")

            if obj.get("is_minor") is True and age == 18:
                user_persona["age"] = 17
                obj["user_persona"] = user_persona
                changed += 1

            updated_lines.append(json.dumps(obj, ensure_ascii=False) + "\n")

    with TARGET_PATH.open("w", encoding="utf-8") as f:
        f.writelines(updated_lines)

    print(f"Updated records: {changed}")
    print(f"File: {TARGET_PATH}")
    return changed


if __name__ == "__main__":
    main()

from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime
from pathlib import Path


WEEKDAY_MAP = ["周一", "周二", "周三", "周四", "周五", "周六", "周日"]
TIMESTAMP_PATTERN = re.compile(r"^\d{4}-\d{2}-\d{2} 周[一二三四五六日] \d{2}:\d{2}$")
MAX_FAIL_SAMPLES_TO_PRINT = 10


def _stable_int(text: str) -> int:
    return int(hashlib.md5(text.encode("utf-8")).hexdigest(), 16)


def _pick_year(key: str) -> int:
    return 2024 if _stable_int(key) % 2 == 0 else 2025


def _format_with_weekday(dt: datetime) -> str:
    weekday = WEEKDAY_MAP[dt.weekday()]
    return f"{dt.strftime('%Y-%m-%d')} {weekday} {dt.strftime('%H:%M')}"


def _is_precise_timestamp_format(value: str) -> bool:
    return bool(TIMESTAMP_PATTERN.match(value))


def _safe_replace_year(dt: datetime, year: int) -> datetime:
    if dt.month == 2 and dt.day == 29 and year == 2025:
        return dt.replace(year=year, day=28)
    return dt.replace(year=year)


def _parse_time_text(text: str) -> datetime | None:
    text = text.strip()

    m = re.search(r"(\d{4})-(\d{2})-(\d{2})\s+\d{2}:\d{2}", text)
    if m:
        date_part = m.group(0).split()[0]
        hm = re.search(r"(\d{2}:\d{2})", text)
        if hm:
            return datetime.strptime(f"{date_part} {hm.group(1)}", "%Y-%m-%d %H:%M")

    m_cn = re.search(r"(\d{4})年(\d{2})月(\d{2})日", text)
    hm_cn = re.search(r"(\d{2}:\d{2})", text)
    if m_cn and hm_cn:
        y, mo, d = m_cn.group(1), m_cn.group(2), m_cn.group(3)
        return datetime.strptime(f"{y}-{mo}-{d} {hm_cn.group(1)}", "%Y-%m-%d %H:%M")

    return None


def remap_opportunity_time(value: object, key: str) -> str:
    target_year = _pick_year(key)

    if isinstance(value, str) and value.strip():
        parsed = _parse_time_text(value)
        if parsed is not None:
            mapped = _safe_replace_year(parsed, target_year)
            return _format_with_weekday(mapped)

    seed = _stable_int(key)
    year = target_year
    month = seed % 12 + 1
    day = seed % 28 + 1
    hour = (seed // 31) % 24
    minute = (seed // 97) % 60
    fallback = datetime(year, month, day, hour, minute)
    return _format_with_weekday(fallback)


def main() -> None:
    current_dir = Path(__file__).resolve().parent
    project_root = current_dir.parent.parent
    source_dir = project_root / "data" / "知识问答数据库"
    output_file = source_dir / "youth_knowledge_qa.jsonl"

    files = sorted(source_dir.glob("*.json"))
    if not files:
        print(f"未找到JSON文件: {source_dir}")
        return

    total = 0
    written = 0
    skipped = 0
    format_ok = 0
    format_fail = 0
    fail_samples: list[tuple[str, str]] = []

    with output_file.open("w", encoding="utf-8") as fout:
        for path in files:
            total += 1
            try:
                with path.open("r", encoding="utf-8") as fin:
                    record = json.load(fin)

                if not isinstance(record, dict):
                    skipped += 1
                    continue

                icbo = record.get("icbo_features")
                if not isinstance(icbo, dict):
                    icbo = {}
                    record["icbo_features"] = icbo

                key = str(record.get("dataset_id") or path.stem)
                mapped_time = remap_opportunity_time(
                    icbo.get("opportunity_time"), key
                )
                icbo["opportunity_time"] = mapped_time

                if _is_precise_timestamp_format(mapped_time):
                    format_ok += 1
                else:
                    format_fail += 1
                    if len(fail_samples) < MAX_FAIL_SAMPLES_TO_PRINT:
                        fail_samples.append((path.name, mapped_time))

                fout.write(json.dumps(record, ensure_ascii=False) + "\n")
                written += 1
            except Exception:
                skipped += 1

    print(
        f"完成：扫描 {total}，写入 {written}，跳过 {skipped}。\n"
        f"时间戳格式校验(YYYY-MM-DD 周X HH:MM)：通过 {format_ok}，失败 {format_fail}。\n"
        f"输出文件：{output_file}"
    )

    if fail_samples:
        print(f"\n⚠️ 格式校验失败样本（最多显示 {MAX_FAIL_SAMPLES_TO_PRINT} 条）:")
        for idx, (file_name, mapped_time) in enumerate(fail_samples, start=1):
            print(f"  {idx}. 文件={file_name} -> opportunity_time={mapped_time}")


if __name__ == "__main__":
    main()

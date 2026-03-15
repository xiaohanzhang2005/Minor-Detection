"""
语义时间 -> 精确时间戳映射脚本（2024-2025）

优化目标：
1) 提升语义覆盖与映射正确性（开学/期中/周末/寒暑假/节假日等）
2) 贴近中国青少年真实作息（上课/课间/午休/深夜时间窗约束）
3) 增强健壮性（未见语义兜底、JSON 异常处理、ID 唯一性保护）

输出格式：YYYY-MM-DD HH:MM
"""

from __future__ import annotations

import json
import random
import re
import uuid
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, Iterable, Optional, Set, Tuple

# ================= 配置区域 =================
# 原始语义时间数据（相对路径，提升迁移性）
CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "社交问答"
INPUT_FILE = DATA_DIR / "semantic_data_v1.jsonl"
# 处理后输出文件
OUTPUT_FILE = DATA_DIR / "youth_dialogs.jsonl"

# 固定映射年份范围：仅允许 2024-2025
YEAR_CANDIDATES = (2024, 2025)

# 是否给缺失/重复 ID 的样本补齐新 ID
# 仅做时间映射时建议关闭，避免改动原始记录主键
ENSURE_UNIQUE_ID = False

# 生成提示词中的固定枚举（严格对齐）
MACRO_CYCLES = (
    "开学初期",
    "期中/月考前夕",
    "学期中段",
    "中/高考前夕",
    "法定节假日",
    "寒暑假期间",
    "普通周末",
    "其他",
)

MICRO_MOMENTS = (
    "深夜",
    "傍晚/晚饭后",
    "上课期间",
    "课间/午休",
    "其他",
)

WEEKDAY_MAP = ["周一", "周二", "周三", "周四", "周五", "周六", "周日"]
TIMESTAMP_PATTERN = re.compile(r"^\d{4}-\d{2}-\d{2} 周[一二三四五六日] \d{2}:\d{2}$")


def _daterange(start: datetime, end: datetime) -> Iterable[datetime]:
    """返回 [start, end] 闭区间内的日期序列（按天）。"""
    cursor = start
    while cursor <= end:
        yield cursor
        cursor += timedelta(days=1)


def _random_date_in_range(start: datetime, end: datetime, rng: random.Random) -> datetime:
    """在闭区间内随机取一天，保证日期真实有效。"""
    days = (end - start).days
    return start + timedelta(days=rng.randint(0, days))


def _pick_weekend_date(year: int, rng: random.Random) -> datetime:
    """在指定年份内随机挑选真实周末日期（周六/周日）。"""
    start = datetime(year, 1, 1)
    end = datetime(year, 12, 31)
    weekends = [d for d in _daterange(start, end) if d.weekday() >= 5]
    return rng.choice(weekends)


def _pick_weekday_date(year: int, rng: random.Random) -> datetime:
    """在指定年份内随机挑选工作日（周一到周五）。"""
    start = datetime(year, 1, 1)
    end = datetime(year, 12, 31)
    weekdays = [d for d in _daterange(start, end) if d.weekday() <= 4]
    return rng.choice(weekdays)


def _extract_macro_cycle(text: str) -> Tuple[str, str]:
    """从 opportunity_time 中抽取宏观周期，返回 (label, source)。"""
    for label in MACRO_CYCLES:
        if label in text:
            return label, "exact"

    # 轻量兼容：历史数据里可能存在的近义表达
    if "期中" in text or "月考" in text:
        return "期中/月考前夕", "compat"
    if "中考" in text or "高考" in text:
        return "中/高考前夕", "compat"
    if "寒假" in text or "暑假" in text:
        return "寒暑假期间", "compat"
    if "周末" in text:
        return "普通周末", "compat"
    if "开学" in text:
        return "开学初期", "compat"
    if "节" in text or "法定" in text:
        return "法定节假日", "compat"

    return "其他", "fallback"


def _extract_micro_moment(text: str) -> Tuple[str, str]:
    """从 opportunity_time 中抽取微观时刻，返回 (label, source)。"""
    for label in MICRO_MOMENTS:
        if label in text:
            return label, "exact"

    # 轻量兼容：历史数据可能出现的非标准词
    if "晚饭后" in text or "傍晚" in text or "晚上" in text:
        return "傍晚/晚饭后", "compat"
    if "课间" in text or "午休" in text:
        return "课间/午休", "compat"
    if "上课" in text:
        return "上课期间", "compat"
    if "深夜" in text:
        return "深夜", "compat"

    return "其他", "fallback"


def _pick_macro_date(macro_cycle: str, rng: random.Random) -> datetime:
    """按宏观周期标签映射日期范围。"""
    year = rng.choice(YEAR_CANDIDATES)

    if macro_cycle == "开学初期":
        if rng.random() < 0.5:
            return _random_date_in_range(datetime(year, 2, 18), datetime(year, 3, 5), rng)
        return _random_date_in_range(datetime(year, 9, 1), datetime(year, 9, 15), rng)

    if macro_cycle == "期中/月考前夕":
        if rng.random() < 0.5:
            return _random_date_in_range(datetime(year, 4, 15), datetime(year, 4, 30), rng)
        return _random_date_in_range(datetime(year, 10, 20), datetime(year, 11, 20), rng)

    if macro_cycle == "学期中段":
        if rng.random() < 0.5:
            return _random_date_in_range(datetime(year, 3, 10), datetime(year, 5, 31), rng)
        return _random_date_in_range(datetime(year, 9, 20), datetime(year, 11, 30), rng)

    if macro_cycle == "中/高考前夕":
        return _random_date_in_range(datetime(year, 5, 15), datetime(year, 6, 8), rng)

    if macro_cycle == "法定节假日":
        # 春节、五一、国庆
        spring_festival = {2024: datetime(2024, 2, 10), 2025: datetime(2025, 1, 29)}
        holiday_type = rng.choice(["spring", "may", "national"])
        if holiday_type == "spring":
            anchor = spring_festival[year]
            return _random_date_in_range(anchor - timedelta(days=3), anchor + timedelta(days=6), rng)
        if holiday_type == "may":
            return _random_date_in_range(datetime(year, 5, 1), datetime(year, 5, 5), rng)
        return _random_date_in_range(datetime(year, 10, 1), datetime(year, 10, 7), rng)

    if macro_cycle == "寒暑假期间":
        if rng.random() < 0.5:
            return _random_date_in_range(datetime(year, 1, 15), datetime(year, 2, 25), rng)
        return _random_date_in_range(datetime(year, 7, 1), datetime(year, 8, 31), rng)

    if macro_cycle == "普通周末":
        return _pick_weekend_date(year, rng)

    return _random_date_in_range(datetime(year, 1, 1), datetime(year, 12, 31), rng)


def _pick_micro_time(micro_moment: str, rng: random.Random) -> Tuple[int, int]:
    """
    微观时刻解析（核心作息约束）：
    - “上课偷偷玩/上课期间”严格落在 08:00-11:30 或 14:00-17:00
    - “深夜”严格 >= 23:00
    """
    # 深夜：严格 23:00-23:59（满足“23 点之后”要求）
    if micro_moment == "深夜":
        return 23, rng.randint(0, 59)

    # 上课场景：严格限制在上课时段
    if micro_moment == "上课期间":
        # 上午 08:00-11:30 或 下午 14:00-17:00
        if rng.random() < 0.5:
            hour = rng.randint(8, 11)
            minute = rng.randint(0, 30) if hour == 11 else rng.randint(0, 59)
            return hour, minute

        hour = rng.randint(14, 17)
        minute = 0 if hour == 17 else rng.randint(0, 59)
        return hour, minute

    # 课间：典型课间窗口
    if micro_moment == "课间/午休":
        candidates = [(9, 50, 10, 10), (15, 50, 16, 10)]
        if rng.random() < 0.6:
            return _random_hm_between(12, 0, 13, 59, rng)
        h1, m1, h2, m2 = rng.choice(candidates)
        return _random_hm_between(h1, m1, h2, m2, rng)

    # 傍晚/晚饭后：18:00-21:30
    if micro_moment == "傍晚/晚饭后":
        return _random_hm_between(18, 0, 21, 30, rng)

    # 默认兜底：全天随机
    return rng.randint(0, 23), rng.randint(0, 59)


def _random_hm_between(h1: int, m1: int, h2: int, m2: int, rng: random.Random) -> Tuple[int, int]:
    """在同一天内 [h1:m1, h2:m2] 闭区间随机采样时分。"""
    start = h1 * 60 + m1
    end = h2 * 60 + m2
    picked = rng.randint(start, end)
    return picked // 60, picked % 60


def parse_semantic_time(semantic_str: str, rng: Optional[random.Random] = None) -> str:
    """
    将中文语义时间映射为精确时间戳（YYYY-MM-DD HH:MM）。
    """
    local_rng = rng or random
    text = semantic_str or ""
    macro_cycle, _ = _extract_macro_cycle(text)
    micro_moment, _ = _extract_micro_moment(text)

    base_date = _pick_macro_date(macro_cycle, local_rng)
    hour, minute = _pick_micro_time(micro_moment, local_rng)
    final_dt = base_date.replace(hour=hour, minute=minute, second=0, microsecond=0)
    weekday = WEEKDAY_MAP[final_dt.weekday()]
    return f"{final_dt.strftime('%Y-%m-%d')} {weekday} {final_dt.strftime('%H:%M')}"


def parse_semantic_time_with_meta(
    semantic_str: str, rng: Optional[random.Random] = None
) -> Tuple[str, str, str, str, str]:
    """返回 (timestamp, macro_cycle, micro_moment, macro_source, micro_source)。"""
    local_rng = rng or random
    text = semantic_str or ""
    macro_cycle, macro_source = _extract_macro_cycle(text)
    micro_moment, micro_source = _extract_micro_moment(text)

    base_date = _pick_macro_date(macro_cycle, local_rng)
    hour, minute = _pick_micro_time(micro_moment, local_rng)
    final_dt = base_date.replace(hour=hour, minute=minute, second=0, microsecond=0)
    weekday = WEEKDAY_MAP[final_dt.weekday()]
    timestamp = f"{final_dt.strftime('%Y-%m-%d')} {weekday} {final_dt.strftime('%H:%M')}"
    return timestamp, macro_cycle, micro_moment, macro_source, micro_source


def _is_precise_timestamp_format(value: str) -> bool:
    """检查时间戳是否为 YYYY-MM-DD 周X HH:MM。"""
    return bool(TIMESTAMP_PATTERN.match(value))


def _ensure_unique_record_id(record: dict, seen_ids: Set[str]) -> None:
    """
    保证记录 ID 唯一：
    - 若缺失 id 或重复，则补一个 uuid4
    - 只改当前记录，不影响其他字段
    """
    current = record.get("id")
    if not isinstance(current, str) or not current.strip() or current in seen_ids:
        record["id"] = str(uuid.uuid4())
    seen_ids.add(record["id"])


def main() -> None:
    print(f"🚀 开始处理文件: {INPUT_FILE}")
    rng = random.Random()  # 若需要可改为 random.Random(42) 固定可复现性

    processed_count = 0
    skipped_json_lines = 0
    skipped_non_dict = 0
    missing_time_field = 0
    seen_ids: Set[str] = set()
    replaced_time_count = 0
    precise_format_ok_count = 0
    precise_format_fail_count = 0

    macro_label_counts: Dict[str, int] = {k: 0 for k in MACRO_CYCLES}
    micro_label_counts: Dict[str, int] = {k: 0 for k in MICRO_MOMENTS}
    macro_source_counts = {"exact": 0, "compat": 0, "fallback": 0}
    micro_source_counts = {"exact": 0, "compat": 0, "fallback": 0}

    with open(INPUT_FILE, "r", encoding="utf-8") as fin, open(OUTPUT_FILE, "w", encoding="utf-8") as fout:
        for i, line in enumerate(fin, start=1):
            line = line.strip()
            if not line:
                continue

            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                skipped_json_lines += 1
                print(f"⚠️ 第 {i} 行 JSON 解析失败，已跳过。")
                continue

            if not isinstance(data, dict):
                skipped_non_dict += 1
                print(f"⚠️ 第 {i} 行不是对象类型，已跳过。")
                continue

            # 保证 icbo_features 为 dict，避免 KeyError
            icbo = data.get("icbo_features")
            if not isinstance(icbo, dict):
                icbo = {}
                data["icbo_features"] = icbo

            semantic_time = icbo.get("opportunity_time", "")
            if isinstance(semantic_time, str) and semantic_time.strip():
                timestamp, macro_cycle, micro_moment, macro_source, micro_source = parse_semantic_time_with_meta(
                    semantic_time.strip(), rng
                )
                icbo["opportunity_time"] = timestamp
                replaced_time_count += 1

                macro_label_counts[macro_cycle] = macro_label_counts.get(macro_cycle, 0) + 1
                micro_label_counts[micro_moment] = micro_label_counts.get(micro_moment, 0) + 1
                macro_source_counts[macro_source] = macro_source_counts.get(macro_source, 0) + 1
                micro_source_counts[micro_source] = micro_source_counts.get(micro_source, 0) + 1

                if _is_precise_timestamp_format(timestamp):
                    precise_format_ok_count += 1
                else:
                    precise_format_fail_count += 1
            else:
                missing_time_field += 1

            if ENSURE_UNIQUE_ID:
                _ensure_unique_record_id(data, seen_ids)

            fout.write(json.dumps(data, ensure_ascii=False) + "\n")
            processed_count += 1

    print(
        "🎉 处理完成！"
        f"有效写入 {processed_count} 条；"
        f"JSON 失败 {skipped_json_lines} 条；"
        f"非对象 {skipped_non_dict} 条；"
        f"缺失 opportunity_time {missing_time_field} 条。"
    )
    print(f"📁 输出文件: {OUTPUT_FILE}")

    # === 映射审计统计 ===
    print("\n📊 映射审计统计")
    if replaced_time_count == 0:
        print("- 未替换任何 opportunity_time，无法计算命中率。")
        return

    print(f"- 替换总数: {replaced_time_count}")
    print(
        f"- 时间戳格式校验(YYYY-MM-DD 周X HH:MM): "
        f"通过 {precise_format_ok_count}，失败 {precise_format_fail_count}"
    )

    macro_fallback_rate = macro_source_counts["fallback"] / replaced_time_count
    micro_fallback_rate = micro_source_counts["fallback"] / replaced_time_count
    macro_compat_rate = macro_source_counts["compat"] / replaced_time_count
    micro_compat_rate = micro_source_counts["compat"] / replaced_time_count

    print(
        f"- 宏观来源命中: exact={macro_source_counts['exact']}, "
        f"compat={macro_source_counts['compat']}, fallback={macro_source_counts['fallback']}"
    )
    print(
        f"- 微观来源命中: exact={micro_source_counts['exact']}, "
        f"compat={micro_source_counts['compat']}, fallback={micro_source_counts['fallback']}"
    )
    print(
        f"- 兜底率: macro={macro_fallback_rate:.2%}, micro={micro_fallback_rate:.2%}"
    )
    print(
        f"- 兼容命中率: macro={macro_compat_rate:.2%}, micro={micro_compat_rate:.2%}"
    )

    print("\n- 宏观标签命中率:")
    for label in MACRO_CYCLES:
        rate = macro_label_counts.get(label, 0) / replaced_time_count
        print(f"  * {label}: {macro_label_counts.get(label, 0)} ({rate:.2%})")

    print("- 微观标签命中率:")
    for label in MICRO_MOMENTS:
        rate = micro_label_counts.get(label, 0) / replaced_time_count
        print(f"  * {label}: {micro_label_counts.get(label, 0)} ({rate:.2%})")


if __name__ == "__main__":
    main()
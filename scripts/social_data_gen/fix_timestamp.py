"""
时间戳后处理脚本

功能：
1. 解析大模型生成的 YYYY-MM-DD HH:MM 格式时间戳
2. 验证日期有效性（是否真实存在）
3. 计算星期并拼接为 YYYY-MM-DD 周X HH:MM
4. 对无效日期进行修正（智能兜底）

支持处理：
- adult_dialogs.jsonl（成年人负样本）
- youth_dialogs.jsonl（青少年正样本）
"""

from __future__ import annotations

import json
import re
import random
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional, Tuple, Dict
from collections import Counter

# ================= 配置区域 =================
CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "社交问答"

# 默认处理文件（可通过命令行参数覆盖）
DEFAULT_INPUT = DATA_DIR / "adult_dialogs.jsonl"
DEFAULT_OUTPUT = DATA_DIR / "adult_dialogs_fixed.jsonl"

# 允许的年份范围
VALID_YEARS = (2024, 2025)

# 星期映射
WEEKDAY_MAP = ["周一", "周二", "周三", "周四", "周五", "周六", "周日"]

# 时间戳正则
TIMESTAMP_PATTERN_INPUT = re.compile(r"^(\d{4})-(\d{2})-(\d{2})\s+(\d{2}):(\d{2})$")
TIMESTAMP_PATTERN_OUTPUT = re.compile(r"^\d{4}-\d{2}-\d{2} 周[一二三四五六日] \d{2}:\d{2}$")


def parse_timestamp(ts: str) -> Optional[Tuple[int, int, int, int, int]]:
    """
    解析 YYYY-MM-DD HH:MM 格式的时间戳
    返回 (year, month, day, hour, minute) 或 None
    """
    if not ts:
        return None
    
    ts = ts.strip()
    match = TIMESTAMP_PATTERN_INPUT.match(ts)
    if match:
        return (
            int(match.group(1)),
            int(match.group(2)),
            int(match.group(3)),
            int(match.group(4)),
            int(match.group(5))
        )
    
    # 兼容已有星期格式 YYYY-MM-DD 周X HH:MM
    alt_pattern = re.match(r"^(\d{4})-(\d{2})-(\d{2})\s+周[一二三四五六日]\s+(\d{2}):(\d{2})$", ts)
    if alt_pattern:
        return (
            int(alt_pattern.group(1)),
            int(alt_pattern.group(2)),
            int(alt_pattern.group(3)),
            int(alt_pattern.group(4)),
            int(alt_pattern.group(5))
        )
    
    return None


def is_valid_date(year: int, month: int, day: int) -> bool:
    """检查日期是否有效"""
    try:
        datetime(year, month, day)
        return True
    except ValueError:
        return False


def fix_invalid_date(year: int, month: int, day: int, rng: random.Random) -> Tuple[int, int, int]:
    """
    修复无效日期，返回有效的 (year, month, day)
    策略：
    1. 年份超范围 → 映射到 2024-2025
    2. 月份超范围 → 映射到 1-12
    3. 日期超范围 → 取该月最后一天
    """
    # 修复年份
    if year < VALID_YEARS[0]:
        year = VALID_YEARS[0]
    elif year > VALID_YEARS[1]:
        year = VALID_YEARS[1]
    
    # 修复月份
    if month < 1:
        month = 1
    elif month > 12:
        month = 12
    
    # 修复日期（取该月有效范围）
    # 获取该月最后一天
    if month == 12:
        last_day = 31
    else:
        next_month = datetime(year, month + 1, 1)
        last_day = (next_month - timedelta(days=1)).day
    
    if day < 1:
        day = 1
    elif day > last_day:
        day = last_day
    
    return year, month, day


def fix_invalid_time(hour: int, minute: int) -> Tuple[int, int]:
    """修复无效时间"""
    if hour < 0:
        hour = 0
    elif hour > 23:
        hour = 23
    
    if minute < 0:
        minute = 0
    elif minute > 59:
        minute = 59
    
    return hour, minute


def generate_fallback_timestamp(rng: random.Random) -> str:
    """
    当时间戳完全无法解析时，生成一个合理的兜底时间戳
    """
    year = rng.choice(VALID_YEARS)
    month = rng.randint(1, 12)
    
    # 获取该月最后一天
    if month == 12:
        last_day = 31
    else:
        next_month = datetime(year, month + 1, 1)
        last_day = (next_month - timedelta(days=1)).day
    
    day = rng.randint(1, last_day)
    
    # 时间倾向于晚间（更符合求助场景）
    time_dist = rng.random()
    if time_dist < 0.4:
        # 晚间 19:00-22:59
        hour = rng.randint(19, 22)
    elif time_dist < 0.7:
        # 深夜 23:00-23:59
        hour = 23
    else:
        # 其他时间
        hour = rng.randint(8, 22)
    
    minute = rng.randint(0, 59)
    
    dt = datetime(year, month, day, hour, minute)
    weekday = WEEKDAY_MAP[dt.weekday()]
    return f"{dt.strftime('%Y-%m-%d')} {weekday} {dt.strftime('%H:%M')}"


def process_timestamp(ts: str, rng: random.Random) -> Tuple[str, str]:
    """
    处理时间戳，返回 (fixed_timestamp, status)
    status: "ok" | "fixed" | "fallback"
    """
    # 已经是正确格式
    if ts and TIMESTAMP_PATTERN_OUTPUT.match(ts.strip()):
        return ts.strip(), "already_ok"
    
    parsed = parse_timestamp(ts)
    
    if parsed is None:
        # 无法解析，使用兜底
        return generate_fallback_timestamp(rng), "fallback"
    
    year, month, day, hour, minute = parsed
    status = "ok"
    
    # 检查年份范围
    if year not in VALID_YEARS:
        year = rng.choice(VALID_YEARS)
        status = "fixed"
    
    # 检查日期有效性
    if not is_valid_date(year, month, day):
        year, month, day = fix_invalid_date(year, month, day, rng)
        status = "fixed"
    
    # 检查时间有效性
    if hour < 0 or hour > 23 or minute < 0 or minute > 59:
        hour, minute = fix_invalid_time(hour, minute)
        status = "fixed"
    
    # 构建最终时间戳
    dt = datetime(year, month, day, hour, minute)
    weekday = WEEKDAY_MAP[dt.weekday()]
    final_ts = f"{dt.strftime('%Y-%m-%d')} {weekday} {dt.strftime('%H:%M')}"
    
    return final_ts, status


def process_file(input_path: Path, output_path: Path) -> Dict[str, int]:
    """
    处理 JSONL 文件
    返回统计信息
    """
    rng = random.Random(42)  # 固定随机种子，保证可复现
    
    stats = Counter({
        "total": 0,
        "processed": 0,
        "already_ok": 0,
        "ok": 0,
        "fixed": 0,
        "fallback": 0,
        "missing_field": 0,
        "json_error": 0,
    })
    
    # 时间分布统计
    hour_dist = Counter()
    month_dist = Counter()
    weekday_dist = Counter()
    
    with open(input_path, "r", encoding="utf-8") as fin, \
         open(output_path, "w", encoding="utf-8") as fout:
        
        for line_num, line in enumerate(fin, start=1):
            line = line.strip()
            if not line:
                continue
            
            stats["total"] += 1
            
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                stats["json_error"] += 1
                print(f"⚠️ 第 {line_num} 行 JSON 解析失败")
                continue
            
            # 获取 opportunity_time
            icbo = data.get("icbo_features", {})
            if not isinstance(icbo, dict):
                icbo = {}
                data["icbo_features"] = icbo
            
            original_ts = icbo.get("opportunity_time", "")
            
            if not original_ts or not isinstance(original_ts, str):
                # 缺失字段，生成兜底
                fixed_ts, status = generate_fallback_timestamp(rng), "fallback"
                stats["missing_field"] += 1
            else:
                fixed_ts, status = process_timestamp(original_ts, rng)
            
            icbo["opportunity_time"] = fixed_ts
            stats[status] += 1
            stats["processed"] += 1
            
            # 统计分布
            try:
                parts = fixed_ts.split()
                date_part = parts[0]
                time_part = parts[-1]
                weekday_part = parts[1] if len(parts) == 3 else ""
                
                month = int(date_part.split("-")[1])
                hour = int(time_part.split(":")[0])
                
                month_dist[month] += 1
                hour_dist[hour] += 1
                if weekday_part:
                    weekday_dist[weekday_part] += 1
            except:
                pass
            
            fout.write(json.dumps(data, ensure_ascii=False) + "\n")
    
    return {
        "stats": dict(stats),
        "hour_dist": dict(hour_dist),
        "month_dist": dict(month_dist),
        "weekday_dist": dict(weekday_dist),
    }


def print_report(result: Dict) -> None:
    """打印处理报告"""
    stats = result["stats"]
    
    print("\n" + "=" * 60)
    print("📊 时间戳处理报告")
    print("=" * 60)
    
    print(f"\n📁 总行数: {stats['total']}")
    print(f"   已处理: {stats['processed']}")
    print(f"   JSON错误: {stats['json_error']}")
    
    print(f"\n🔧 处理结果分布:")
    print(f"   已是正确格式: {stats.get('already_ok', 0)}")
    print(f"   解析成功: {stats.get('ok', 0)}")
    print(f"   修复后成功: {stats.get('fixed', 0)}")
    print(f"   兜底生成: {stats.get('fallback', 0)}")
    print(f"   缺失字段: {stats.get('missing_field', 0)}")
    
    if stats['processed'] > 0:
        fallback_rate = (stats.get('fallback', 0) + stats.get('missing_field', 0)) / stats['processed']
        print(f"\n⚠️ 兜底率: {fallback_rate:.2%}")
    
    # 时间分布
    print("\n📅 月份分布:")
    month_dist = result.get("month_dist", {})
    for month in sorted(month_dist.keys()):
        count = month_dist[month]
        pct = count / stats['processed'] * 100 if stats['processed'] > 0 else 0
        bar = "█" * int(pct / 2)
        print(f"   {month:2d}月: {count:4d} ({pct:5.1f}%) {bar}")
    
    print("\n🕐 小时分布:")
    hour_dist = result.get("hour_dist", {})
    for hour in sorted(hour_dist.keys()):
        count = hour_dist[hour]
        pct = count / stats['processed'] * 100 if stats['processed'] > 0 else 0
        bar = "█" * int(pct)
        print(f"   {hour:02d}:xx: {count:4d} ({pct:5.1f}%) {bar}")
    
    print("\n📆 星期分布:")
    weekday_dist = result.get("weekday_dist", {})
    for wd in ["周一", "周二", "周三", "周四", "周五", "周六", "周日"]:
        count = weekday_dist.get(wd, 0)
        pct = count / stats['processed'] * 100 if stats['processed'] > 0 else 0
        bar = "█" * int(pct / 2)
        print(f"   {wd}: {count:4d} ({pct:5.1f}%) {bar}")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="时间戳后处理脚本")
    parser.add_argument("-i", "--input", type=str, default=str(DEFAULT_INPUT),
                        help="输入文件路径")
    parser.add_argument("-o", "--output", type=str, default=str(DEFAULT_OUTPUT),
                        help="输出文件路径")
    parser.add_argument("--inplace", action="store_true",
                        help="原地修改（输出覆盖输入）")
    
    args = parser.parse_args()
    
    input_path = Path(args.input)
    output_path = Path(args.output) if not args.inplace else input_path.with_suffix(".tmp.jsonl")
    
    if not input_path.exists():
        print(f"❌ 输入文件不存在: {input_path}")
        return
    
    print(f"🚀 开始处理时间戳...")
    print(f"   输入: {input_path}")
    print(f"   输出: {output_path if not args.inplace else input_path}")
    
    result = process_file(input_path, output_path)
    
    # 如果是 inplace 模式，替换原文件
    if args.inplace:
        import shutil
        shutil.move(str(output_path), str(input_path))
        print(f"\n✅ 已原地更新: {input_path}")
    else:
        print(f"\n✅ 输出文件: {output_path}")
    
    print_report(result)


if __name__ == "__main__":
    main()

"""
这个脚本尚未启用
目前数据库的时间标签仍然是“开学前夕”
（此脚本可把“开学前夕”转为精确时间）
"""
import json
import random
from datetime import datetime, timedelta
import uuid

# ================= 配置区域 =================
# GPT-4 生成的、包含中文语义时间的文件
INPUT_FILE = "semantic_data.jsonl" 
# 最终处理完成、可以入库的文件
OUTPUT_FILE = "final_dataset.jsonl" 

# ================= 核心逻辑：语义时间解析器 =================
def parse_semantic_time(semantic_str):
    """
    V3 版解析器：精准解析“宏观周期”+“微观时刻”
    """
    year = random.choice([2023, 2025])
    
    # 1. 宏观周期解析
    if "开学初期" in semantic_str:
        # 假设开学在9月初
        base_date = datetime(year, 9, random.randint(1, 10))
    elif "期中" in semantic_str or "月考" in semantic_str:
        # 假设期中在4月底或11月中
        month = random.choice([4, 11])
        base_date = datetime(year, month, random.randint(20, 30))
    elif "期末" in semantic_str:
        month = random.choice([1, 6])
        base_date = datetime(year, month, random.randint(15, 25))
    elif "中考" in semantic_str:
        base_date = datetime(year, 6, random.randint(10, 15))
    elif "高考" in semantic_str:
        base_date = datetime(year, 6, random.randint(1, 6))
    elif "法定节假日" in semantic_str:
        # 随机选择一个节假日
        holidays = [(10,1), (5,1), (2,10)] # 国庆，五一，春节
        month, day = random.choice(holidays)
        base_date = datetime(year, month, day)
    elif "寒暑假" in semantic_str:
        month = random.choice([1, 2, 7, 8])
        base_date = datetime(year, month, random.randint(1, 28))
    elif "普通周末" in semantic_str:
        # 随机找一个周末
        start_date = datetime(year, 1, 1)
        random_days = random.randint(0, 364)
        random_date = start_date + timedelta(days=random_days)
        # 循环直到找到一个周六或周日
        while random_date.weekday() < 5:
            random_date += timedelta(days=1)
        base_date = random_date
    else: # “其他” 或 无法识别的情况
        start_date = datetime(year, 1, 1)
        base_date = start_date + timedelta(days=random.randint(0, 364))

    # 2. 微观时刻解析
    if "深夜" in semantic_str:
        hour = random.randint(23, 23) if random.random() < 0.7 else random.randint(0, 3)
    elif "傍晚" in semantic_str or "晚饭后" in semantic_str:
        hour = random.randint(18, 21)
    elif "上课期间" in semantic_str:
        hour = random.choice([9, 10, 14, 15])
    elif "课间" in semantic_str or "午休" in semantic_str:
        hour = random.choice([10, 12, 13, 16])
    else: # “其他”
        hour = random.randint(0, 23)
        
    minute = random.randint(0, 59)
    
    final_datetime = base_date.replace(hour=hour, minute=minute)
    
    # 3. 格式化输出
    weekday_map = ["周一", "周二", "周三", "周四", "周五", "周六", "周日"]
    weekday_str = weekday_map[final_datetime.weekday()]
    
    return final_datetime.strftime(f"%Y年%m月%d日 {weekday_str} %H:%M")

# --- 主函数 ---
def main():
    print(f"🚀 开始处理文件: {INPUT_FILE}")
    processed_count = 0
    with open(INPUT_FILE, 'r', encoding='utf-8') as fin, \
         open(OUTPUT_FILE, 'w', encoding='utf-8') as fout:
        
        for i, line in enumerate(fin):
            try:
                data = json.loads(line)
                
                # --- 核心处理逻辑 ---
                # 1. 替换 ID
                # 使用 uuid 来生成一个绝对唯一的 ID，比时间戳更稳妥
                data['dataset_id'] = f"Gen_Social_{str(uuid.uuid4())[:8]}"
                
                # 2. 解析并替换时间
                semantic_time = data['icbo_features'].get('opportunity_time', '')
                if semantic_time:
                    precise_time = parse_semantic_time(semantic_time)
                    data['icbo_features']['opportunity_time'] = precise_time
                
                # 3. 写回新文件 (一行一个 JSON)
                fout.write(json.dumps(data, ensure_ascii=False) + '\n')
                processed_count += 1
                
            except json.JSONDecodeError:
                print(f"⚠️ 警告：第 {i+1} 行无法解析为 JSON，已跳过。内容: {line.strip()}")
            except Exception as e:
                print(f"❌ 错误：处理第 {i+1} 行时发生未知错误: {e}")
                
    print(f"🎉 处理完成！共处理 {processed_count} 条数据，结果已保存至 {OUTPUT_FILE}")

if __name__ == "__main__":
    # 假设你已经有了一个 semantic_data.jsonl 文件
    # 如果没有，你可以创建一个假的来测试
    # with open("semantic_data.jsonl", "w", encoding="utf-8") as f:
    #     test_data = {"icbo_features": {"opportunity_time": "中考前夕的深夜"}}
    #     f.write(json.dumps(test_data) + '\n')
        
    main()
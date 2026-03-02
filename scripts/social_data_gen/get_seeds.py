"""
使用关键词和年龄提取双重验证的方式，从PsyQA数据集中筛选出符合条件的青少年用户问题，生成高质量的种子数据集（脚本初步筛选）
"""
import json
from datasets import load_dataset
import os
import re

# ================= 配置区域 =================
TARGET_COUNT = 10000
OUTPUT_FILE = "youth_seeds_v3.json"

# ================= 核心逻辑：年龄提取器 =================
def extract_age(text):
    """
    从文本中提取年龄，支持阿拉伯数字(15岁)和中文数字(十五岁)
    返回一个年龄列表，例如 [15, 26]
    """
    ages = []
    
    # 1. 匹配阿拉伯数字： 比如 "15岁", "26岁"
    pattern_num = r"(\d{1,2})\s*岁"
    matches_num = re.findall(pattern_num, text)
    for m in matches_num:
        try:
            age = int(m)
            ages.append(age)
        except:
            pass

    # 2. 匹配中文数字： 比如 "十五岁", "二十六岁"
    pattern_cn = r"([一二三四五六七八九十]{1,3})\s*岁"
    matches_cn = re.findall(pattern_cn, text)
    
    for m in matches_cn:
        age = cn_to_int(m)
        if age > 0:
            ages.append(age)
            
    return ages

def cn_to_int(cn_str):
    """简单的中文数字转阿拉伯数字 (支持1-99)"""
    cn_map = {'一': 1, '二': 2, '三': 3, '四': 4, '五': 5, 
              '六': 6, '七': 7, '八': 8, '九': 9, '十': 10}
    
    if len(cn_str) == 1:
        return cn_map.get(cn_str, 0)
    elif len(cn_str) == 2:
        if cn_str[0] == '十': # 十五
            return 10 + cn_map.get(cn_str[1], 0)
        elif cn_str[1] == '十': # 二十
            return cn_map.get(cn_str[0], 0) * 10
    elif len(cn_str) == 3: # 二十五
        if cn_str[1] == '十':
            return cn_map.get(cn_str[0], 0) * 10 + cn_map.get(cn_str[2], 0)
    return 0

# ================= 关键词过滤器 (升级版) =================

# 1. 身份锚点 (Identity Anchors)
# 作用：首先确定他是小于18岁的青少年（强特征），如果没有明确年龄，则寻找身份特征或场景特征来辅助判断
IDENTITY_KEYWORDS = [
    # 具体年级（最强特征）
    "初一", "初二", "初三", "初中", "初中生",
    "高一", "高二", "高三", "高中", "高中生",
    "中学生", "13岁", "14岁", "15岁", "16岁", "17岁",
    "七年级", "八年级", "九年级","未成年",
    
    # 校园强特征（中学生特有环境）
    "中考", "高考", "会考", "模考", "早自习", "晚自习", "班主任", "重点班", "普通班",
    "文理分科", "分班", "住校", "走读", "早恋", "青春期", "叛逆期","开学","考试","成绩","上学"
]

# 2. 社交与心理痛点 (Scenario Anchors)
SCENARIO_KEYWORDS = [
    # 同伴关系 (Peer Relationship)
    "孤立", "排挤", "没朋友", "被无视", "融不进", "格格不入",
    "同桌", "室友", "闺蜜", "死党", "绝交", "闹翻", "背刺", 
    "坏话", "小团体", "社恐", "不敢说话", "尴尬", "丢人", "人际关系", "同学关系",
    
    # 亲子与权威关系 (Authority Conflict)
    "代沟", "不理解", "偷看", "查手机", "没收", "隐私", 
    "偏心", "吵架", "离家出走", "断绝关系", "控制欲", "唠叨",
    "被骂", "被打", "老师针对", "请家长",
    
    # 情感与萌动 (Romance/Intimacy)
    "暗恋", "表白", "喜欢他", "喜欢她", "前任", "分手", 
    "吃醋", "暧昧", "男神", "女神", "喜欢一个人",
    
    # 自我与心理 (Self & Psychology)
    "自卑", "焦虑", "抑郁", "想死", "自残", "割腕", "活着", 
    "迷茫", "没动力", "假装开心", "敏感", "多疑", "崩溃",
    "容貌焦虑", "长得丑", "胖", "整容", "考试压力", "成绩焦虑", "升学压力"
]

# 3. 屏蔽词 (Adult Filters)
# 作用：强力剔除成年人和大学生
ADULT_KEYWORDS = [
    # 婚姻与家庭（成年视角）
    "老公", "老婆", "丈夫", "妻子", "孩子", "娃", "宝宝",
    "结婚", "孕期", "备孕", "婆婆", "岳母", "儿媳", "女婿",
    "相亲", "催婚", "房贷", "车贷", "买房", "装修",
    
    # 职场与大学（非K12阶段）
    "工作", "上班", "老板", "同事", "公司", "入职", "离职", "跳槽", "月薪", "工资", "社保",
    "本科", "硕士", "博士", "研究生", "大一", "大二", "大三", "大四",
    "考研", "保研", "论文", "答辩", "实习", "校招", 
]

def check_user_identity(text):
    """
    综合判断用户身份
    返回: (是否保留, 原因)
    """
    if not text:
        return False, "空文本"

    # --- 第一关：黑名单查杀 (Adult Filter) ---
    for word in ADULT_KEYWORDS:
        if word in text:
            return False, f"包含成年人关键词: {word}"

    # --- 第二关：显式年龄判断 (Age Gate) ---
    extracted_ages = extract_age(text)
    if extracted_ages:
        for age in extracted_ages:
            if age > 17:
                return False, f"检测到年龄 {age} > 17"
        
        # 如果年龄符合，直接通过（这是最高优先级证据）
        return True, f"检测到符合要求的年龄: {extracted_ages}"

    # --- 第三关：身份或场景单点验证 (Relaxed Check) ---
    # 逻辑：只要具备“学生身份特征” OR “社交/心理场景特征” 之一即可
    
    for word in IDENTITY_KEYWORDS:
        if word in text:
            return True, f"身份特征命中: [{word}]"
    for word in SCENARIO_KEYWORDS:
        if word in text:
            return True, f"场景特征命中: [{word}]"

    return False, "无明显青少年特征"

def main():
    print("🚀 开始运行带双重验证的筛选脚本...")
    
    try:
        # 尝试加载
        dataset = load_dataset("lsy641/PsyQA", split="train")
    except:
        # 如果失败，尝试本地读取
        try:
            print("⚠️ 尝试加载本地文件: psyqa.json")
            with open("psyqa.json", "r", encoding="utf-8") as f:
                dataset = json.load(f)
        except:
            print("❌ 没找到数据，请确认网络或文件位置。")
            return

    filtered_seeds = []
    
    for item in dataset:
        full_text = str(item.get('question', '')) + " " + str(item.get('description', ''))
        
        # 调用新的判断逻辑
        is_valid, reason = check_user_identity(full_text)
        
        if is_valid:
            seed_data = {
                "source_id": f"Seed_{len(filtered_seeds)+1}",
                "reason": reason, 
                "original_text": full_text
            }
            filtered_seeds.append(seed_data)
            
            if len(filtered_seeds) % 50 == 0:
                print(f"✅ 已收集 {len(filtered_seeds)} 条 (最新一条原因: {reason})")
        
        if len(filtered_seeds) >= TARGET_COUNT:
            break
            
    # 保存
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(filtered_seeds, f, ensure_ascii=False, indent=4)
    print(f"🎉 筛选完成！结果已保存至 {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
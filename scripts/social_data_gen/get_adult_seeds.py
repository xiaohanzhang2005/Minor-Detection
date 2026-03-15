"""
从 PsyQA 数据集中筛选成年人种子，用于生成社交领域负样本

采用【分层采样】策略：
- Tier 1 (55%): 高迷惑性 - 18-28岁 + 青少年相似场景（最难区分）
- Tier 2 (30%): 中等迷惑性 - 25-35岁 + 生活化场景  
- Tier 3 (15%): 典型成年人 - 28-45岁 + 成年人场景（保证覆盖度）

年龄采样策略（软上限）：
- 30岁以下：100% 采样
- 30-40岁：20% 概率采样
- 40岁以上：基本排除
"""
import json
import re
import os
import random
from datasets import load_dataset

# ================= 配置区域 =================
TARGET_COUNT = 2000  # 目标总数

# 分层配比
TIER_RATIOS = {
    "tier1": 0.55,  # 高迷惑性
    "tier2": 0.30,  # 中等迷惑性
    "tier3": 0.15,  # 典型成年人
}

OUTPUT_FILE = "adult_seeds_tiered.json"

# 正样本种子路径（用于去重）
POSITIVE_SEEDS_FILE = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
    "data", "社交问答", "youth_seeds_v5.json"
)

# ================= 年龄提取器 =================
def extract_age(text):
    """从文本中提取年龄"""
    ages = []
    
    # 阿拉伯数字
    pattern_num = r"(\d{1,2})\s*岁"
    matches_num = re.findall(pattern_num, text)
    for m in matches_num:
        try:
            age = int(m)
            if 10 <= age <= 60:
                ages.append(age)
        except:
            pass

    # 中文数字
    pattern_cn = r"([一二三四五六七八九十]{1,3})\s*岁"
    matches_cn = re.findall(pattern_cn, text)
    for m in matches_cn:
        age = cn_to_int(m)
        if 10 <= age <= 60:
            ages.append(age)
            
    return ages

def cn_to_int(cn_str):
    """中文数字转阿拉伯数字"""
    cn_map = {'一': 1, '二': 2, '三': 3, '四': 4, '五': 5, 
              '六': 6, '七': 7, '八': 8, '九': 9, '十': 10}
    
    if len(cn_str) == 1:
        return cn_map.get(cn_str, 0)
    elif len(cn_str) == 2:
        if cn_str[0] == '十':
            return 10 + cn_map.get(cn_str[1], 0)
        elif cn_str[1] == '十':
            return cn_map.get(cn_str[0], 0) * 10
    elif len(cn_str) == 3:
        if cn_str[1] == '十':
            return cn_map.get(cn_str[0], 0) * 10 + cn_map.get(cn_str[2], 0)
    return 0


# ================= 关键词定义 =================

# Tier 1: 高迷惑性场景词（和青少年高度相似）
TIER1_CONFUSING_KEYWORDS = [
    # 情绪/心理
    "焦虑", "抑郁", "迷茫", "压力大", "失眠", "崩溃", "绝望",
    "自卑", "社恐", "孤独", "空虚", "没动力", "想放弃",
    
    # 学业相关
    "考试", "期末", "挂科", "绩点", "考研", "考公", "论文",
    "复习", "学不进去", "不想学", "图书馆",
    
    # 人际/情感
    "室友", "舍友", "被孤立", "合群", "社交",
    "暗恋", "表白", "分手", "前任", "喜欢的人",
    
    # 家庭（依赖型）
    "和父母", "爸妈", "不理解我", "催", "唠叨", "生活费",
    
    # 生活习惯
    "打游戏", "熬夜", "赖床", "外卖", "宅", "追剧",
]

# Tier 2: 中等迷惑性场景词（成年人生活化但有情绪表达）
TIER2_LIFE_KEYWORDS = [
    # 职场初期
    "工作", "上班", "同事", "公司", "加班", "累",
    "入职", "实习", "offer", "面试", "找工作",
    
    # 生活压力
    "压力", "焦虑", "迷茫", "不知道", "怎么办",
    "未来", "人生", "意义", "方向",
    
    # 人际（成年人版）
    "朋友", "社交", "聚会", "孤独", "圈子",
    
    # 情感（成年人版）
    "恋爱", "对象", "单身", "感情", "相亲",
    
    # 家庭关系（独立视角）
    "父母", "家人", "关系", "沟通", "理解",
]

# Tier 3: 典型成年人场景词（明显成年但避免敏感话题）
TIER3_ADULT_KEYWORDS = [
    # 职业发展
    "职业", "发展", "规划", "晋升", "转行", "行业",
    
    # 生活阶段
    "毕业", "工作几年", "三十岁", "而立之年",
    
    # 成熟思考
    "人生选择", "生活", "成长", "责任", "独立",
    
    # 社会话题
    "社会", "现实", "经济", "大环境",
]

# 成年人身份锚点（用于确认是成年人）
ADULT_IDENTITY_KEYWORDS = [
    # 大学/研究生
    "大学", "大一", "大二", "大三", "大四", "本科", "专科",
    "研究生", "硕士", "博士", "研一", "研二", "研三",
    "毕业", "毕业后",
    
    # 工作相关
    "工作", "上班", "公司", "同事", "职场",
    "实习", "入职", "辞职",
    
    # 年龄标识（18+）
    "18岁", "19岁", "20岁", "21岁", "22岁", "23岁", "24岁", "25岁",
    "26岁", "27岁", "28岁", "29岁", "30岁", "三十岁",
    "31岁", "32岁", "33岁", "34岁", "35岁",
    "十八岁", "十九岁", "二十岁", "二十一", "二十二", "二十三",
    "二十四", "二十五", "二十六", "二十七", "二十八", "二十九", "三十",
]

# 硬排除词（无论哪个 Tier 都排除）
HARD_EXCLUDE_KEYWORDS = [

    # 明确未成年标识
    "初中", "高中", "中考", "高考", "班主任", "早自习", "晚自习",
    "初一", "初二", "初三", "高一", "高二", "高三",
    "13岁", "14岁", "15岁", "16岁", "17岁",
    "十三岁", "十四岁", "十五岁", "十六岁", "十七岁",
    "未成年", "中学", "小学","未成年"
]


def classify_sample(text):
    """
    对文本进行分层分类
    返回: (tier, reason, age) 或 (None, reject_reason, None)
    
    tier: "tier1" / "tier2" / "tier3" / None (拒绝)
    """
    if not text:
        return None, "空文本", None

    # --- 第一关：硬排除 ---
    for word in HARD_EXCLUDE_KEYWORDS:
        if word in text:
            return None, f"硬排除: {word}", None

    # --- 第二关：年龄检测 ---
    extracted_ages = extract_age(text)
    detected_age = None
    
    if extracted_ages:
        # 取第一个合理年龄
        for age in extracted_ages:
            if age < 18:
                return None, f"未成年年龄: {age}", None
            if age > 45:
                return None, f"年龄过大: {age}", None
            if detected_age is None:
                detected_age = age
        
        # 年龄软上限：30-40岁只有20%概率采样
        if detected_age and 30 <= detected_age <= 40:
            if random.random() > 0.20:
                return None, f"年龄{detected_age}岁被概率过滤", None
        
        # 40岁以上极低概率（5%）
        if detected_age and detected_age > 40:
            if random.random() > 0.05:
                return None, f"年龄{detected_age}岁被概率过滤", None

    # --- 第三关：检测成年人身份 ---
    has_adult_identity = False
    identity_word = ""
    for word in ADULT_IDENTITY_KEYWORDS:
        if word in text:
            has_adult_identity = True
            identity_word = word
            break

    # 如果没有明确年龄且没有成年人身份词，拒绝
    if detected_age is None and not has_adult_identity:
        return None, "无法确认成年人身份", None

    # --- 第四关：分层分类 ---
    # Tier 1: 高迷惑性（18-28岁优先）
    tier1_match = None
    for word in TIER1_CONFUSING_KEYWORDS:
        if word in text:
            tier1_match = word
            break
    
    if tier1_match:
        # 年龄在 18-28 范围内，或者有大学生身份
        if detected_age and 18 <= detected_age <= 28:
            return "tier1", f"Tier1: 年龄{detected_age} + {tier1_match}", detected_age
        if any(w in text for w in ["大学", "大一", "大二", "大三", "大四", "研究生", "硕士", "博士"]):
            return "tier1", f"Tier1: 学生身份 + {tier1_match}", detected_age
        # 其他成年人有高迷惑性词，归为 Tier 2
        if has_adult_identity:
            return "tier2", f"Tier2: 成年人 + 迷惑性词 {tier1_match}", detected_age

    # Tier 2: 中等迷惑性（25-35岁，生活化场景）
    tier2_match = None
    for word in TIER2_LIFE_KEYWORDS:
        if word in text:
            tier2_match = word
            break
    
    if tier2_match:
        return "tier2", f"Tier2: {identity_word or f'年龄{detected_age}'} + {tier2_match}", detected_age

    # Tier 3: 典型成年人（28-45岁，成年人场景）
    tier3_match = None
    for word in TIER3_ADULT_KEYWORDS:
        if word in text:
            tier3_match = word
            break
    
    if tier3_match:
        return "tier3", f"Tier3: {identity_word or f'年龄{detected_age}'} + {tier3_match}", detected_age

    # 有成年人身份但没匹配场景词，归为 Tier 2（宽松处理）
    if has_adult_identity and detected_age and detected_age >= 18:
        return "tier2", f"Tier2: 仅身份确认 {identity_word}, 年龄{detected_age}", detected_age

    return None, "无匹配场景", None


def normalize_text(text):
    """标准化文本用于去重比较"""
    # 去除空白字符，统一为小写
    return re.sub(r'\s+', '', text.strip().lower())


def load_positive_seeds():
    """加载正样本种子，构建去重集合"""
    positive_texts = set()
    
    if not os.path.exists(POSITIVE_SEEDS_FILE):
        print(f"⚠️ 正样本文件不存在: {POSITIVE_SEEDS_FILE}")
        print("   将跳过去重检查")
        return positive_texts
    
    try:
        with open(POSITIVE_SEEDS_FILE, 'r', encoding='utf-8') as f:
            positive_seeds = json.load(f)
        
        for seed in positive_seeds:
            text = seed.get('original_text', '')
            if text:
                positive_texts.add(normalize_text(text))
        
        print(f"✅ 加载正样本 {len(positive_texts)} 条用于去重")
    except Exception as e:
        print(f"⚠️ 加载正样本失败: {e}")
    
    return positive_texts


def main():
    print("🚀 开始分层筛选成年人种子...")
    print(f"   目标总数: {TARGET_COUNT}")
    print(f"   分层配比: Tier1={TIER_RATIOS['tier1']:.0%}, Tier2={TIER_RATIOS['tier2']:.0%}, Tier3={TIER_RATIOS['tier3']:.0%}")
    print(f"   输出文件: {OUTPUT_FILE}")
    print()
    
    # 加载正样本用于去重
    print("📥 加载正样本进行去重检查...")
    positive_texts = load_positive_seeds()
    print()
    
    # 计算各层目标数量
    tier_targets = {
        "tier1": int(TARGET_COUNT * TIER_RATIOS["tier1"]),
        "tier2": int(TARGET_COUNT * TIER_RATIOS["tier2"]),
        "tier3": int(TARGET_COUNT * TIER_RATIOS["tier3"]),
    }
    print(f"   各层目标: Tier1={tier_targets['tier1']}, Tier2={tier_targets['tier2']}, Tier3={tier_targets['tier3']}")
    print()
    
    # 加载数据集
    try:
        print("📥 加载 PsyQA 数据集...")
        dataset = load_dataset("lsy641/PsyQA", split="train")
        print(f"   数据集大小: {len(dataset)}")
    except Exception as e:
        print(f"❌ 加载失败: {e}")
        try:
            print("⚠️ 尝试加载本地文件: psyqa.json")
            with open("psyqa.json", "r", encoding="utf-8") as f:
                dataset = json.load(f)
        except:
            print("❌ 没找到数据，请确认网络或文件位置。")
            return

    # 分层收集
    tier_seeds = {"tier1": [], "tier2": [], "tier3": []}
    stats = {
        "total_scanned": 0,
        "hard_excluded": 0,
        "underage": 0,
        "age_filtered": 0,
        "no_identity": 0,
        "no_scene": 0,
        "duplicate": 0,  # 与正样本重复
        "tier1_accepted": 0,
        "tier2_accepted": 0,
        "tier3_accepted": 0,
    }
    
    # 打乱数据集以增加多样性
    dataset_list = list(dataset)
    random.shuffle(dataset_list)
    
    for item in dataset_list:
        stats["total_scanned"] += 1
        
        full_text = str(item.get('question', '')) + " " + str(item.get('description', ''))
        
        # 先检查是否与正样本重复
        if positive_texts and normalize_text(full_text) in positive_texts:
            stats["duplicate"] += 1
            continue
        
        tier, reason, age = classify_sample(full_text)
        
        if tier is not None:
            # 检查该层是否已满
            if len(tier_seeds[tier]) >= tier_targets[tier]:
                continue
            
            seed_data = {
                "source_id": f"AdultSeed_{tier}_{len(tier_seeds[tier])+1}",
                "tier": tier,
                "reason": reason,
                "detected_age": age,
                "original_text": full_text.strip(),
            }
            tier_seeds[tier].append(seed_data)
            stats[f"{tier}_accepted"] += 1
            
            total_collected = sum(len(v) for v in tier_seeds.values())
            if total_collected % 100 == 0:
                print(f"📊 进度: {total_collected}/{TARGET_COUNT}")
                print(f"   Tier1: {len(tier_seeds['tier1'])}/{tier_targets['tier1']}, "
                      f"Tier2: {len(tier_seeds['tier2'])}/{tier_targets['tier2']}, "
                      f"Tier3: {len(tier_seeds['tier3'])}/{tier_targets['tier3']}")
        else:
            # 统计拒绝原因
            if "硬排除" in reason:
                stats["hard_excluded"] += 1
            elif "未成年" in reason:
                stats["underage"] += 1
            elif "概率过滤" in reason or "年龄过大" in reason:
                stats["age_filtered"] += 1
            elif "无法确认" in reason:
                stats["no_identity"] += 1
            else:
                stats["no_scene"] += 1
        
        # 检查是否所有层都满了
        if all(len(tier_seeds[t]) >= tier_targets[t] for t in tier_seeds):
            break
    
    # 合并所有层
    all_seeds = []
    for tier in ["tier1", "tier2", "tier3"]:
        all_seeds.extend(tier_seeds[tier])
    
    # 重新编号
    for i, seed in enumerate(all_seeds, 1):
        seed["source_id"] = f"AdultSeed_{i}"
    
    # 保存结果
    output_path = os.path.join(os.path.dirname(__file__), OUTPUT_FILE)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(all_seeds, f, ensure_ascii=False, indent=2)
    
    print()
    print("=" * 60)
    print("📊 筛选统计:")
    print(f"   扫描总数: {stats['total_scanned']}")
    print()
    print(f"   ✅ Tier1 (高迷惑性): {stats['tier1_accepted']} / {tier_targets['tier1']}")
    print(f"   ✅ Tier2 (中等迷惑性): {stats['tier2_accepted']} / {tier_targets['tier2']}")
    print(f"   ✅ Tier3 (典型成年人): {stats['tier3_accepted']} / {tier_targets['tier3']}")
    print(f"   ✅ 总计: {len(all_seeds)}")
    print()
    print(f"   ❌ 与正样本重复: {stats['duplicate']}")
    print(f"   ❌ 硬排除 (婚育/未成年词): {stats['hard_excluded']}")
    print(f"   ❌ 未成年年龄: {stats['underage']}")
    print(f"   ❌ 年龄概率过滤: {stats['age_filtered']}")
    print(f"   ❌ 身份不明: {stats['no_identity']}")
    print(f"   ❌ 无匹配场景: {stats['no_scene']}")
    print()
    print(f"🎉 筛选完成！结果已保存至: {output_path}")
    
    # 各层样本预览
    print()
    print("=" * 60)
    print("📝 各层样本预览:")
    
    for tier in ["tier1", "tier2", "tier3"]:
        if tier_seeds[tier]:
            sample = tier_seeds[tier][0]
            print(f"\n--- {tier.upper()} 示例 ---")
            print(f"原因: {sample['reason']}")
            print(f"年龄: {sample['detected_age']}")
            print(f"文本: {sample['original_text'][:120]}...")


if __name__ == "__main__":
    main()

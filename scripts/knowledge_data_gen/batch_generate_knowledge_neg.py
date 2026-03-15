"""
批量从成人教材目录 txt 构造知识问答负样本（成年人知识 QA 对话）
基于 data/教材目录_成人/ 下 15 个成人种子文件生成

与正样本脚本 (generate_edu_dialogs.py) 的主要区别：
1. 用户年龄 >= 18，身份为大学生/考研党/职场人等
2. 知识点来源于成人考试/大学课程目录
3. 输出 is_minor: false 标签
4. 多线程并发调用 API（参考 batch_generate_adult.py）

与社交负样本 (batch_generate_adult.py) 的主要区别：
1. Seed 来自教材目录 txt（而非 JSON 种子库）
2. Prompt 聚焦"知识问答"场景（而非社交/心理）
3. 用户画像紧密绑定考试/课程身份

当前数据概况：
- 知识-正样本: 2004 条 (K12教育)
- 社交-正样本: 2603 条
- 社交-负样本: 1735 条
- 知识-负样本: 待生成 (本脚本) ——1986条

使用：
    python batch_generate_knowledge_neg.py
    python batch_generate_knowledge_neg.py --max_per_file 40 --workers 8
    python batch_generate_knowledge_neg.py --files 考研数学.txt CPA注册会计师.txt
    python batch_generate_knowledge_neg.py --dry_run   # 仅统计种子数，不调用 API
"""

import json
import os
import re
import sys
import time
import random
import argparse
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests

# ================= 配置区域 =================

API_KEY = os.environ.get("AIHUBMIX_API_KEY", "sk-your-key")
API_URL = "https://aihubmix.com/v1/chat/completions"
MODEL_NAME = "gemini-3-flash-preview"

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))

INPUT_DIR = os.path.join(project_root, "data", "教材目录_成人")
OUTPUT_FILE = os.path.join(project_root, "data", "知识问答数据库", "adult_knowledge_qa.jsonl")

MAX_WORKERS = 15
TIMEOUT_SECONDS = 120
PROXIES = None

# ================= 文件元信息回退表 =================
# 当文件头注释中未找到 Tier/身份/年龄信息时，使用此表
FALLBACK_META = {
    "大学高等数学.txt":       {"tier": "tier1", "identity": "大一大二理工科学生", "age_range": "18-20"},
    "大学线性代数.txt":       {"tier": "tier1", "identity": "大一大二理工科学生", "age_range": "18-20"},
    "大学概率论与数理统计.txt": {"tier": "tier1", "identity": "大一大二理工科学生", "age_range": "18-20"},
    "大学计算机基础.txt":     {"tier": "tier1", "identity": "大一大二计算机/信息学生", "age_range": "18-20"},
    "大学英语四六级.txt":     {"tier": "tier1", "identity": "大二大三学生", "age_range": "19-22"},
    "考研数学.txt":          {"tier": "tier2", "identity": "大三大四/往届考研党", "age_range": "21-28"},
    "考研英语.txt":          {"tier": "tier2", "identity": "大三大四/往届考研党", "age_range": "21-28"},
    "考研政治.txt":          {"tier": "tier2", "identity": "大三大四/往届考研党", "age_range": "21-28"},
    "考研408计算机.txt":     {"tier": "tier2", "identity": "大三大四/往届考研党", "age_range": "21-28"},
    "CPA注册会计师.txt":     {"tier": "tier3", "identity": "财会专业学生/在职会计师", "age_range": "22-35"},
    "公务员考试.txt":        {"tier": "tier3", "identity": "应届生/在职考公党", "age_range": "22-35"},
    "教师资格证.txt":        {"tier": "tier3", "identity": "师范生/在职转教师", "age_range": "21-35"},
    "法律职业资格考试.txt":   {"tier": "tier3", "identity": "法学生/在职法律人", "age_range": "22-35"},
    "执业医师资格.txt":      {"tier": "tier3", "identity": "医学生/规培住院医师", "age_range": "23-35"},
    "雅思托福.txt":          {"tier": "tier2", "identity": "大学生/留学申请者", "age_range": "19-26"},
}


# ================= Seed 解析 =================

def parse_file_header(lines):
    """从文件头注释中提取 Tier、身份、年龄等元信息"""
    meta = {}
    for line in lines:
        line = line.strip()
        if not line.startswith("#"):
            break
        # 匹配 "# Tier: tier1 | 身份: xxx | 年龄: 18-20"
        tier_match = re.search(r"Tier:\s*(tier\d)", line, re.IGNORECASE)
        if tier_match:
            meta["tier"] = tier_match.group(1)
        identity_match = re.search(r"身份:\s*([^|]+)", line)
        if identity_match:
            meta["identity"] = identity_match.group(1).strip()
        age_match = re.search(r"年龄:\s*(\d+-\d+)", line)
        if age_match:
            meta["age_range"] = age_match.group(1)
    return meta


def parse_age_range(age_range_str):
    """将 '18-20' 解析为 (18, 20) 元组"""
    parts = age_range_str.split("-")
    return int(parts[0]), int(parts[1])


def _split_detail_by_semicolons(detail_text):
    """
    将知识点细节按分号拆分为独立的子知识点。
    
    例如：
      "卫生法基础知识;传染病防治法;职业病防治法"
      → ["卫生法基础知识", "传染病防治法", "职业病防治法"]
      
      "函数的性质(有界/单调/周期/奇偶); 复合函数/反函数/隐函数; 数列极限定义"
      → ["函数的性质(有界/单调/周期/奇偶)", "复合函数/反函数/隐函数", "数列极限定义"]
    
    注意：顿号(、)和括号内的内容视为同一知识点的子方面，不再进一步拆分。
    """
    # 按中英文分号拆分
    chunks = re.split(r"[;；]", detail_text)
    result = []
    for chunk in chunks:
        chunk = chunk.strip()
        if chunk:
            result.append(chunk)
    return result


def iter_seeds_from_file(filepath):
    """
    解析单个成人教材 txt 文件，生成 seed 字典列表。
    
    核心逻辑：
    1. 按 " -- " 分割行，提取 subject / chapter / detail
    2. 对 detail 部分按分号 (;/；) 进一步拆分，每个分号段生成独立 seed
       → 确保"每个 seed = 1 个可独立提问的知识点"
    3. 对于无 detail 的 2 级行（如 "第一章 函数与极限 -- 映射与函数"），整行即为 1 个 seed
    
    这样可以把密集行（如执医卫生法规一行23部法律）展开为 23 个独立 seed，
    而细粒度行（如大学高数每行1个知识点）保持不变。
    """
    filename = os.path.basename(filepath)
    
    with open(filepath, "r", encoding="utf-8") as f:
        all_lines = f.readlines()

    # 1. 解析头注释获取元信息
    header_meta = parse_file_header(all_lines)
    fallback = FALLBACK_META.get(filename, {"tier": "tier2", "identity": "成年备考者", "age_range": "20-30"})

    tier = header_meta.get("tier", fallback["tier"])
    identity = header_meta.get("identity", fallback["identity"])
    age_range = header_meta.get("age_range", fallback["age_range"])

    seeds = []
    for line in all_lines:
        line = line.strip()
        # 跳过空行和注释行
        if not line or line.startswith("#"):
            continue
        
        # 按 " -- " 分割
        parts = [p.strip() for p in line.split(" -- ") if p.strip()]
        if len(parts) < 2:
            continue

        subject = parts[0]
        chapter = parts[1] if len(parts) >= 2 else ""
        raw_detail = " -- ".join(parts[2:]) if len(parts) >= 3 else ""

        # 2. 按分号拆分 detail 部分
        if raw_detail:
            sub_points = _split_detail_by_semicolons(raw_detail)
        else:
            # 2 级结构（如 "第一章 xxx -- 知识点"），chapter 本身可能含分号
            sub_points = _split_detail_by_semicolons(chapter)
            if len(sub_points) > 1:
                # chapter 中存在分号分隔的多知识点，拆分后 chapter 留空
                # 如 "微积分 -- 函数的性质; 复合函数; 数列极限定义"
                for sp in sub_points:
                    seeds.append({
                        "source_file": filename,
                        "tier": tier,
                        "identity": identity,
                        "age_range": age_range,
                        "seed_text": f"{subject} -- {sp}",
                        "subject": subject,
                        "chapter": sp,
                        "detail": "",
                    })
                continue
            else:
                # chapter 无分号，整行就是 1 个 seed
                seeds.append({
                    "source_file": filename,
                    "tier": tier,
                    "identity": identity,
                    "age_range": age_range,
                    "seed_text": line,
                    "subject": subject,
                    "chapter": chapter,
                    "detail": "",
                })
                continue

        # 3. detail 有分号 → 逐个拆分为独立 seed
        for sp in sub_points:
            seed_text = f"{subject} -- {chapter} -- {sp}"
            seeds.append({
                "source_file": filename,
                "tier": tier,
                "identity": identity,
                "age_range": age_range,
                "seed_text": seed_text,
                "subject": subject,
                "chapter": chapter,
                "detail": sp,
            })

    return seeds


def collect_all_seeds(input_dir, max_per_file=None, file_filter=None):
    """收集所有文件的 seeds，可选过滤和每文件采样"""
    all_seeds = []
    file_stats = {}
    
    for filename in sorted(os.listdir(input_dir)):
        if not filename.endswith(".txt"):
            continue
        if file_filter and filename not in file_filter:
            continue
        
        filepath = os.path.join(input_dir, filename)
        seeds = iter_seeds_from_file(filepath)
        
        if max_per_file and len(seeds) > max_per_file:
            seeds = random.sample(seeds, max_per_file)
        
        file_stats[filename] = len(seeds)
        all_seeds.extend(seeds)
    
    return all_seeds, file_stats


def tier_balanced_sample(all_seeds, total_target, tier_ratio=None):
    """
    按 Tier 配额进行平衡采样。
    
    - tier1 种子不足时：复用种子（同一 seed 可生成多次，LLM 的 temperature 保证内容不重复）
    - tier2/tier3 种子过多时：随机抽样
    
    Args:
        all_seeds: 全量种子列表
        total_target: 总目标条数
        tier_ratio: Tier 配比字典，如 {"tier1": 3, "tier2": 4, "tier3": 3}
    
    Returns:
        balanced_seeds: 平衡后的种子列表（tier1 中可能有重复 seed_text）
        tier_stats: 各 tier 的最终条数
    """
    if tier_ratio is None:
        tier_ratio = {"tier1": 3, "tier2": 4, "tier3": 3}
    
    ratio_sum = sum(tier_ratio.values())
    
    # 按 tier 分桶
    buckets = {}
    for seed in all_seeds:
        t = seed["tier"]
        buckets.setdefault(t, []).append(seed)
    
    balanced = []
    tier_stats = {}
    
    for tier, ratio in tier_ratio.items():
        target = int(total_target * ratio / ratio_sum)
        pool = buckets.get(tier, [])
        
        if not pool:
            tier_stats[tier] = 0
            continue
        
        if len(pool) >= target:
            # 种子充足 → 随机抽样
            sampled = random.sample(pool, target)
        else:
            # 种子不足 → 全部保留 + 随机复用补足差额
            sampled = list(pool)
            deficit = target - len(pool)
            # 为复用的种子添加 _repeat_idx 标记，方便 dataset_id 去重
            extras = []
            for i in range(deficit):
                dup = dict(random.choice(pool))
                dup["_repeat_idx"] = i + 1
                extras.append(dup)
            sampled.extend(extras)
        
        tier_stats[tier] = len(sampled)
        balanced.extend(sampled)
    
    random.shuffle(balanced)
    return balanced, tier_stats


# ================= Prompt 模板 =================

SYSTEM_PROMPT = """你现在是一个【成人知识问答数据生成专家】。你的任务是基于我提供的"教材知识点（Seed）"，构建一个用于 AI 安全研究的高质量中文对话数据。

目标：生成一个成年人（18岁及以上）与 AI 知识助手之间的知识问答对话。
- role = "user"：一位正在学习或备考的成年人（大学生、考研党、职场人等），说话较成熟但可能带有焦虑或困惑。
- role = "assistant"：在线 AI 学习助手，专业、简洁、友好地回答问题。
- 整体氛围更像是成年人用手机/电脑向 AI 请教专业知识，不是课堂也不是聊天。

语言风格：
- user 的语言有一定专业度，但也能体现个人特点（如考前焦虑、工作压力、吐槽考点太多等）；
- user 不会称呼 AI 为"助教""老师""同学"等角色称谓，因为这是在线 AI 工具（类似 ChatGPT/Kimi），用户通常直接提问或说"帮我看看""请问"或者其他的发问方式；
- assistant 直接回答，专业准确，不向 user 反问或引导思考。

关键约束：用户必须是 18 岁及以上的成年人。"""

USER_PROMPT_TEMPLATE = """
# Goal
生成一个符合 **ICBO模型** (Intention, Cognition, Behavior, Opportunity) 的成年人与 AI 知识助手之间的知识问答对话，用于训练"成年人 vs 未成年人"分类器（本条为成年人负样本）。

---
### Input Seed（知识点种子）
> {seed_text}

### 来源信息
> 科目/考试: {subject}
> 章节/模块: {chapter}
> Tier: {tier}
> 典型身份: {identity}
> 年龄范围: {age_range}

---
### 2024-2025 时间参照
**大学生/考研场景：**
- 期末考试：6月中-7月初 / 12月下-1月上
- 考研初试：12月第三个周末
- 四六级：6月/12月第三个周六
- 暑假备考：7-8月

**职业考试场景：**
- CPA：8月下旬（专业阶段）
- 法考：客观题9月、主观题10月
- 执业医师：实践技能6月、综合笔试8月
- 教资：笔试3月/11月、面试5月/1月
- 公务员国考：笔试11-12月
- 雅思托福：全年滚动

**时段参考：**
- 深夜：23:00-01:59（刷题/焦虑）
- 晚间：19:00-22:59（下班/自习后）
- 午休：12:00-13:30
- 自习时间：09:00-11:30, 14:00-17:30

---
### 生成指令

**【第一步：构建画像】（内部推理，不输出）**
1. 根据 Tier 和来源信息构建用户画像：
   - **Tier1 (高迷惑性)**：18-22岁大学生，说话口语化、有情绪但偶尔会有成年线索（"室友""期末""挂科"）
   - **Tier2 (中等)**：21-28岁考研/留学党，相对成熟，提到"考研""备考""真题""留学申请"等
   - **Tier3 (典型成年人)**：22-35岁职业考试备考者，语言专业化，提到"工作""执业""事务所""备考CPA"等
2. 确定精确时间戳（YYYY-MM-DD HH:MM），需与考试节奏匹配

**【第二步：填充元数据】**
- `user_persona.age`：必须 >= 18，根据 Tier 和年龄范围选取
- `user_persona.identity`：如"大三学生""考研党""审计助理""临床规培生"等
- `cognition`：该知识点的常见理解误区或难点
- `behavior_style`：如"专业但略显焦躁""简洁干练""爱吐槽考试"等
- `opportunity_time`：精确到分钟的时间戳

**【第三步：生成对话】**
- 生成 4-8 轮对话，**总字数建议 500-1000 字**（体现在线 QA 短对话特征，不要过长）
- **身份暴露**：对话中必须有**至少 1 处**明确暴露成年人身份的表达，如：
  - "我大三了""考研复习到崩溃""工作中遇到这个问题""备考CPA""实习的时候""室友说"
- **知识聚焦**：对话必须围绕 Seed 中的知识点展开
- user 先发起提问，assistant 直接回答
- **assistant 角色约束**：assistant 只负责回答和解释，不向 user 提问或反问，不使用"你觉得呢？""要不要先想一想？"等引导式问题，每轮 assistant 的发言以陈述句为主

---
### Output Format
仅输出一个标准 JSON 对象，不含 markdown 标记或额外文字：

{{
  "dataset_id": "placeholder",
  "is_minor": false,
  "icbo_features": {{
    "intention": "用户学习意图",
    "cognition": "认知难点/误区",
    "behavior_style": "语言风格特征",
    "opportunity_time": "YYYY-MM-DD HH:MM"
  }},
  "user_persona": {{
    "age": "数字",
    "gender": "男/女",
    "identity": "具体身份标签"
  }},
  "conversation": [
    {{"role": "user", "content": "..."}},
    {{"role": "assistant", "content": "..."}},
    {{"role": "user", "content": "..."}},
    {{"role": "assistant", "content": "..."}}
  ]
}}

注意：
1. age >= 18，is_minor = false
2. opportunity_time 精确到 YYYY-MM-DD HH:MM
3. 对话 4-8 轮，user 先开口，总字数 500-1000 字
4. 对话中必须有成年人身份的**明确线索**（"大三了""考研""在职""实习""室友"等），这是区分正负样本的关键信号
5. 知识内容要准确，围绕 Seed 展开
6. 时间覆盖 2024-2025 年不同月份和时段
7. assistant 不反问、不引导，直接回答知识问题
"""


# ================= API 调用 =================

file_lock = threading.Lock()
counter_lock = threading.Lock()
progress = {"success": 0, "fail": 0, "total": 0}


def _update_model(name):
    global MODEL_NAME
    MODEL_NAME = name


def call_llm(seed):
    """调用 LLM 为单个 seed 生成对话"""
    prompt = USER_PROMPT_TEMPLATE.format(
        seed_text=seed["seed_text"],
        subject=seed["subject"],
        chapter=seed["chapter"],
        tier=seed["tier"],
        identity=seed["identity"],
        age_range=seed["age_range"],
    )
    
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.5,
        "max_tokens": 3000,
    }

    max_retries = 5
    for attempt in range(max_retries):
        try:
            resp = requests.post(
                API_URL,
                headers=headers,
                json=payload,
                timeout=TIMEOUT_SECONDS,
                proxies=PROXIES,
            )
            resp.raise_for_status()
            content = resp.json()["choices"][0]["message"]["content"]

            # 清洗 markdown 包裹
            content = re.sub(r"^```json\s*", "", content.strip())
            content = re.sub(r"\s*```$", "", content).strip()

            data_obj = json.loads(content)

            # 强制标签
            data_obj["is_minor"] = False

            # 注入溯源信息
            repeat_idx = seed.get('_repeat_idx', 0)
            seed_hash = hash(seed['seed_text']) % 100000
            data_obj["dataset_id"] = f"adult_knowledge_{MODEL_NAME}_{seed['source_file']}_{seed_hash}_r{repeat_idx}"
            data_obj["_meta"] = {
                "_seed": seed["seed_text"],
                "_source_file": seed["source_file"],
                "_tier": seed["tier"],
                "_subject": seed["subject"],
                "_chapter": seed["chapter"],
            }

            # 校验年龄（LLM 可能返回字符串 "22" 而非整数）
            age = data_obj.get("user_persona", {}).get("age")
            if age is not None:
                age = int(age)
                data_obj["user_persona"]["age"] = age
                if age < 18:
                    data_obj["user_persona"]["age"] = 18

            return data_obj

        except Exception as e:
            wait = 2 ** attempt
            print(f"  ⚠️ [{seed['source_file']}] 第{attempt+1}次失败: {str(e)[:80]}... 等{wait}s")
            time.sleep(wait)

    return None


def worker(seed, output_file):
    """线程工作函数"""
    result = call_llm(seed)
    source = seed["source_file"]
    seed_short = seed["seed_text"][:50]

    with counter_lock:
        if result:
            progress["success"] += 1
            with file_lock:
                with open(output_file, "a", encoding="utf-8") as f:
                    f.write(json.dumps(result, ensure_ascii=False) + "\n")
            print(f"  ✅ [{progress['success']}/{progress['total']}] {source} | {seed_short}...")
        else:
            progress["fail"] += 1
            print(f"  ❌ [{source}] 彻底失败: {seed_short}...")
    
    return result


# ================= 主函数 =================

def main():
    parser = argparse.ArgumentParser(description="从成人教材目录批量生成知识问答负样本")
    parser.add_argument("--max_per_file", type=int, default=None,
                        help="每个 txt 文件最多采样多少条种子 (不启用 --tier_balance 时默认 30)")
    parser.add_argument("--workers", type=int, default=MAX_WORKERS,
                        help=f"并发线程数 (默认 {MAX_WORKERS})")
    parser.add_argument("--files", nargs="+", default=None,
                        help="仅处理指定文件 (如 --files 考研数学.txt CPA注册会计师.txt)")
    parser.add_argument("--dry_run", action="store_true",
                        help="仅解析种子并打印统计，不调用 API")
    parser.add_argument("--output", type=str, default=OUTPUT_FILE,
                        help=f"输出文件路径 (默认 {OUTPUT_FILE})")
    parser.add_argument("--model", type=str, default=None,
                        help=f"模型名称 (默认 {MODEL_NAME})")
    parser.add_argument("--tier_balance", action="store_true", default=True,
                        help="启用 Tier 平衡采样（默认开启）：tier1 过采样(种子复用)，tier2/3 欠采样。用 --no_tier_balance 关闭")
    parser.add_argument("--no_tier_balance", action="store_true",
                        help="关闭 Tier 平衡采样，改为按文件均匀采样")
    parser.add_argument("--total", type=int, default=2000,
                        help="--tier_balance 模式下的总目标条数 (默认 2000)")
    parser.add_argument("--tier_ratio", type=str, default="3:4:3",
                        help="tier1:tier2:tier3 配比 (默认 3:4:3)")
    args = parser.parse_args()

    # 更新全局模型名
    if args.model:
        _update_model(args.model)

    # --no_tier_balance 覆盖默认的 --tier_balance
    if args.no_tier_balance:
        args.tier_balance = False

    # 解析 tier_ratio
    tier_ratio_parts = args.tier_ratio.split(":")
    tier_ratio = {
        "tier1": int(tier_ratio_parts[0]),
        "tier2": int(tier_ratio_parts[1]),
        "tier3": int(tier_ratio_parts[2]),
    }

    print("=" * 60)
    print("🎓 成人知识问答负样本生成器")
    print("=" * 60)
    print(f"  输入目录: {INPUT_DIR}")
    print(f"  输出文件: {args.output}")
    print(f"  模型: {MODEL_NAME}")
    print(f"  线程数: {args.workers}")
    if args.tier_balance:
        print(f"  ⚖️  Tier 平衡模式: 目标 {args.total} 条, 配比 {args.tier_ratio}")
    else:
        mpf = args.max_per_file if args.max_per_file else 30
        print(f"  每文件采样: {mpf}")
    print()

    # 1. 收集全量种子（tier_balance 模式需要全量以便后续按 tier 采样）
    if args.tier_balance:
        all_seeds, file_stats = collect_all_seeds(
            INPUT_DIR,
            max_per_file=None,  # 全量采集
            file_filter=args.files,
        )
    else:
        mpf = args.max_per_file if args.max_per_file else 30
        all_seeds, file_stats = collect_all_seeds(
            INPUT_DIR,
            max_per_file=mpf,
            file_filter=args.files,
        )

    # 打印原始统计
    print("📊 原始种子统计：")
    tier_counts = {"tier1": 0, "tier2": 0, "tier3": 0}
    for filename, count in sorted(file_stats.items()):
        tier = FALLBACK_META.get(filename, {}).get("tier", "?")
        print(f"  {filename:30s}  {count:4d} 条  ({tier})")
        tier_counts[tier] = tier_counts.get(tier, 0) + count
    print(f"\n  全量: {len(all_seeds)} 条种子")
    print(f"  Tier 分布: {dict(tier_counts)}")
    print()

    # 2. Tier 平衡采样
    if args.tier_balance:
        all_seeds, tier_final = tier_balanced_sample(all_seeds, args.total, tier_ratio)
        repeated = sum(1 for s in all_seeds if "_repeat_idx" in s)
        print(f"⚖️  Tier 平衡采样完成：")
        for t in ["tier1", "tier2", "tier3"]:
            original = tier_counts.get(t, 0)
            final = tier_final.get(t, 0)
            if final > original:
                print(f"  {t}: {original} → {final} (复用 {final - original} 条种子)")
            elif final < original:
                print(f"  {t}: {original} → {final} (随机抽样)")
            else:
                print(f"  {t}: {original} → {final} (不变)")
        print(f"  总计: {len(all_seeds)} 条 (其中复用种子 {repeated} 条)")
        print()

    if args.dry_run:
        print("🔍 [Dry Run] 仅统计，不调用 API。")
        print("\n📝 示例种子 (随机 5 条)：")
        samples = random.sample(all_seeds, min(5, len(all_seeds)))
        for s in samples:
            repeat_flag = " [复用]" if "_repeat_idx" in s else ""
            print(f"  [{s['tier']}]{repeat_flag} {s['seed_text'][:80]}...")
        return

    # 3. 检查输出文件是否已存在
    if os.path.exists(args.output):
        existing_count = sum(1 for _ in open(args.output, "r", encoding="utf-8"))
        print(f"⚠️  输出文件已存在 ({existing_count} 条)，新数据将追加到末尾。")
        print(f"    如需重新生成，请先手动删除: {args.output}")
        print()

    tasks = all_seeds

    progress["total"] = len(tasks)
    print(f"📋 待处理: {len(tasks)} 条")
    print()

    # 3. 确保输出目录存在
    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    # 4. 多线程执行
    start_time = time.time()

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(worker, seed, args.output): seed for seed in tasks}
        for future in as_completed(futures):
            try:
                future.result()
            except Exception as e:
                seed = futures[future]
                print(f"  💥 线程异常 [{seed['source_file']}]: {e}")

    elapsed = time.time() - start_time

    print()
    print("=" * 60)
    print("🎉 生成完成！")
    print(f"  成功: {progress['success']} 条")
    print(f"  失败: {progress['fail']} 条")
    print(f"  耗时: {elapsed:.1f}s")
    if progress["success"] > 0:
        print(f"  速度: {progress['success']/elapsed*60:.1f} 条/分钟")
    print(f"  输出: {args.output}")
    print("=" * 60)


if __name__ == "__main__":
    main()

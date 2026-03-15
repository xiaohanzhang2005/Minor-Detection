"""
批量调用 API 生成成年人对话（负样本）
基于 adult_seeds_tiered.json 生成完整对话数据

与正样本脚本的主要区别：
1. 用户年龄必须 >= 18
2. Prompt 强调成年人身份（大学生、职场人、考研党等）
3. 输出 is_minor: false 标签
"""
import json
import os
import time
import re
import requests
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

# ================= 配置区域 =================

API_KEY = os.environ.get("AIHUBMIX_API_KEY", "sk-your-key")
API_URL = "https://aihubmix.com/v1/chat/completions"
MODEL_NAME = "gemini-3-flash-preview"

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
data_dir = os.path.join(project_root, "data", "社交问答")

# 输入输出路径
INPUT_FILE = os.path.join(data_dir, "adult_seeds_tiered.json")
OUTPUT_FILE = os.path.join(data_dir, "adult_dialogs.jsonl")

# 提速配置
MAX_WORKERS = 10
TIMEOUT_SECONDS = 120

# 网络代理
PROXIES = None

# 运行范围
START_ID = 1
END_ID = 99999

# ================= Prompt 模板（成年人版） =================

SYSTEM_PROMPT = """你现在是一个【成年人心理与行为数据生成专家】。你的任务是基于我提供的"真实求助文本（Seed）"，构建一个用于AI安全研究的高质量中文对话数据。

关键要求：生成的对话中，用户必须是【成年人（18岁及以上）】，如大学生、研究生、职场新人、考研党等。"""

USER_PROMPT_TEMPLATE = """
# Goal (目标)
生成一个符合 **ICBO模型** (Intention, Cognition, Behavior, Opportunity) 的【成年人】与AI的对话记录。你需要通过"演绎"将一段陈述性的求助文本，转化为一段**实时的、口语化的**人机对话。

**重要约束**：用户必须是成年人（18岁及以上），可以是大学生、研究生、职场新人、考研/考公党、年轻打工人等。

---
### Input Seed (输入种子)
> {seed_text}

### Tier 信息
> {tier_info}

---
### 2024-2025 关键日期参考（用于生成多样化时间戳）

**大学生场景锚点：**
- 春季学期开学：2024-02-26 / 2025-02-24
- 春季期中考试：2024-04-15~04-30 / 2025-04-14~04-28
- 春季期末考试：2024-06-17~07-05 / 2025-06-16~07-04
- 暑假：2024-07-08~08-25 / 2025-07-07~08-24
- 秋季学期开学：2024-09-02 / 2025-09-01
- 秋季期中考试：2024-10-28~11-08 / 2025-10-27~11-07
- 秋季期末考试：2024-12-23~2025-01-10 / 2025-12-22~2026-01-09
- 寒假：2025-01-13~02-23 / 2026-01-12~02-22
- 考研初试：2024-12-21~22 / 2025-12-20~21

**职场人场景锚点：**
- 春节假期：2024-02-10~17 / 2025-01-28~02-04
- 五一假期：2024-05-01~05 / 2025-05-01~05
- 国庆假期：2024-10-01~07 / 2025-10-01~07
- 年终总结：12月下旬
- 金三银四求职季：3-4月
- 金九银十求职季：9-10月

**时段参考：**
- 深夜：23:00-01:59（情绪低落、失眠场景）
- 晚间：19:00-22:59（下班/下课后）
- 午休：12:00-13:30
- 工作/上课时间：09:00-11:30, 14:00-17:30

---
### Instruction (生成指令)

**【第一步：推理与分析】- (在你的脑中完成，不要输出)**
1.  **分析种子**：通读上述种子，理解这个成年人的核心困扰是什么？
2.  **构建画像 (Persona)**：基于 Tier 信息，想象他/她是一个怎样的人？
    - **Tier1 (高迷惑性)**：18-28岁，大学生/研究生，说话可能比较口语化、有情绪，但会体现出成年人身份（如提到"室友""期末""考研""实习"等）
    - **Tier2 (中等迷惑性)**：25-35岁，职场初期，关注工作、未来、人际，语气相对成熟但也有困惑
    - **Tier3 (典型成年人)**：28-45岁，更成熟的视角，关注职业发展、人生规划
3.  **确定精确时间**：
    - 根据场景从上方「关键日期参考」中选择合适的日期范围
    - 生成具体日期（YYYY-MM-DD）和时间（HH:MM）
    - 确保时间与场景匹配（如：期末焦虑→期末考试前几天的深夜）

**【第二步：填充元数据】**
根据你的推理，填充 JSON 对象中的 `user_persona` 和 `icbo_features` 字段。
- `user_persona.age`：必须 >= 18，根据 Tier 选择合理年龄
- `user_persona.identity`：身份标签，如"大三学生""考研党""职场新人""产品经理"等
- `cognition`：心理学术语，如"灾难化思维"、"过度概括"、"完美主义焦虑"
- `behavior_style`：语言特征，如"口语化但有条理"、"偶尔自嘲"、"职场用语"
- `opportunity_time`：**精确时间戳，格式 YYYY-MM-DD HH:MM**（如 2024-11-03 23:47）

**【第三步：生成对话】**
基于画像和场景，生成 4-8 轮对话。
- **身份暴露**：对话中必须有**至少1处**明确暴露成年人身份的表达，如：
  - "我大三了""我在实习""我工作两年了""考研压力大""室友都在投简历"
- **真实性要求**：口语化、有情绪，但不要过于幼稚
- **AI 表现**：尝试共情和给出成熟的建议

---
### Output Format (输出格式)
请仅输出一个标准的 JSON 对象，不要包含任何 markdown 标记或额外解释。格式如下：

{{
  "dataset_id": "placeholder_id",
  "is_minor": false,
  "icbo_features": {{
    "intention": "简述用户意图 (I)",
    "cognition": "简述认知特征 (C)",
    "behavior_style": "简述语言风格 (B)",
    "opportunity_time": "YYYY-MM-DD HH:MM（必须是精确时间戳，如 2024-06-18 23:15）"
  }},
  "user_persona": {{
    "age": "数字，必须 >= 18",
    "gender": "男/女",
    "identity": "大三学生/考研党/职场新人/产品经理等"
  }},
  "conversation": [
    {{"role": "user", "content": "..."}},
    {{"role": "assistant", "content": "..."}},
    {{"role": "user", "content": "..."}}
  ]
}}

注意：
1. age 必须 >= 18
2. is_minor 必须为 false
3. opportunity_time 必须是精确时间戳格式 YYYY-MM-DD HH:MM
4. 对话中必须有成年人身份的明确线索
5. 时间要多样化，覆盖 2024-2025 年的不同月份、不同时段
"""

# ================= 核心逻辑 =================

file_lock = threading.Lock()

def get_seed_index(seed_id_str):
    match = re.search(r"AdultSeed_(\d+)", seed_id_str)
    if match:
        return int(match.group(1))
    match = re.search(r"(\d+)", seed_id_str)
    return int(match.group(1)) if match else -1

def call_llm_worker(seed_data):
    """单个线程的工作函数"""
    seed_id = seed_data.get("source_id", "Unknown")
    tier = seed_data.get("tier", "tier1")
    tier_reason = seed_data.get("reason", "")
    
    # 构建 tier 信息
    tier_info = f"Tier: {tier}\n原因: {tier_reason}"
    
    prompt = USER_PROMPT_TEMPLATE.format(
        seed_text=seed_data["original_text"],
        tier_info=tier_info
    )
    
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }
    data = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.7,
        "max_tokens": 2500
    }

    # 重试机制
    max_retries = 5
    for i in range(max_retries):
        try:
            response = requests.post(
                API_URL, 
                headers=headers, 
                json=data, 
                timeout=TIMEOUT_SECONDS,
                proxies=PROXIES 
            )
            response.raise_for_status()
            res_json = response.json()
            
            content = res_json["choices"][0]["message"]["content"]
            # 清洗 markdown
            content = re.sub(r"^```json\s*", "", content)
            content = re.sub(r"\s*```$", "", content).strip()
            
            # 校验 JSON
            data_obj = json.loads(content)
            
            # 强制设置标签
            data_obj["is_minor"] = False
            
            # 注入额外信息
            seed_idx = get_seed_index(seed_id)
            data_obj["dataset_id"] = f"adult_{MODEL_NAME}_{seed_idx}"
            data_obj["extra_info"] = {
                "seed_id": seed_id,
                "tier": tier,
                "detected_age": seed_data.get("detected_age"),
            }
            
            # 验证年龄
            generated_age = data_obj.get("user_persona", {}).get("age")
            if generated_age is not None and generated_age < 18:
                print(f"⚠️ [{seed_id}] 生成的年龄 {generated_age} < 18，强制修正为 18")
                data_obj["user_persona"]["age"] = 18
            
            return data_obj

        except Exception as e:
            wait_time = 2 ** i
            print(f"⚠️ [{seed_id}] 第 {i+1} 次失败: {str(e)[:100]}... 等待 {wait_time}s 重试")
            time.sleep(wait_time)
            
    print(f"❌ [{seed_id}] 彻底失败，放弃。")
    return None


def main():
    print("🚀 开始生成成年人对话（负样本）...")
    print(f"   输入: {INPUT_FILE}")
    print(f"   输出: {OUTPUT_FILE}")
    print(f"   模型: {MODEL_NAME}")
    print(f"   线程数: {MAX_WORKERS}")
    print()
    
    # 1. 读取种子
    try:
        with open(INPUT_FILE, 'r', encoding='utf-8') as f:
            seeds = json.load(f)
        print(f"✅ 加载 {len(seeds)} 条种子")
    except Exception as e:
        print(f"❌ 读取种子失败: {e}")
        return

    # 统计 tier 分布
    tier_counts = {}
    for seed in seeds:
        tier = seed.get("tier", "unknown")
        tier_counts[tier] = tier_counts.get(tier, 0) + 1
    print(f"   Tier 分布: {tier_counts}")
    print()

    # 2. 断点续传
    completed_ids = set()
    if os.path.exists(OUTPUT_FILE):
        with open(OUTPUT_FILE, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    obj = json.loads(line)
                    if "extra_info" in obj:
                        completed_ids.add(obj["extra_info"]["seed_id"])
                except: 
                    pass
    print(f"🔎 已有 {len(completed_ids)} 条数据，将自动跳过。")

    # 3. 准备任务列表
    tasks = []
    for seed in seeds:
        s_id = seed.get("source_id", "")
        s_idx = get_seed_index(s_id)
        
        if s_idx < START_ID or s_idx > END_ID:
            continue
        if s_id in completed_ids:
            continue
        
        tasks.append(seed)

    print(f"📋 准备处理 {len(tasks)} 条任务，开启 {MAX_WORKERS} 个线程...")
    print()

    # 4. 多线程执行
    success_count = 0
    start_time = time.time()
    
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_seed = {executor.submit(call_llm_worker, seed): seed for seed in tasks}
        
        for future in as_completed(future_to_seed):
            seed = future_to_seed[future]
            seed_id = seed.get("source_id")
            
            try:
                result = future.result()
                if result:
                    with file_lock:
                        with open(OUTPUT_FILE, 'a', encoding='utf-8') as f:
                            f.write(json.dumps(result, ensure_ascii=False) + '\n')
                    
                    tier = seed.get("tier", "?")
                    print(f"✅ [{seed_id}] ({tier}) 完成！")
                    success_count += 1
            except Exception as e:
                print(f"💥 线程异常 [{seed_id}]: {e}")

    elapsed = time.time() - start_time
    print()
    print("=" * 60)
    print(f"🎉 全部结束！")
    print(f"   本次新增: {success_count} 条")
    print(f"   总耗时: {elapsed:.1f}s")
    print(f"   平均速度: {success_count/elapsed*60:.1f} 条/分钟")
    print(f"   输出文件: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()

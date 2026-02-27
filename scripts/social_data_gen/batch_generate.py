"""
批量调用API生成对话
"""
import json
import os
import time
import re
import requests
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

# ================= 配置区域 (关键修改) =================

API_KEY = "sk-your-key"  # 替换成你的 Key
API_URL = "https://aihubmix.com/v1/chat/completions"  # AIHubMix 标准接口地址
MODEL_NAME = "gemini-3-flash-preview"


current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
data_dir = os.path.join(project_root, "data", "社交问答")

# 定义最终的输入输出路径
INPUT_FILE = os.path.join(data_dir, "youth_seeds_v5.json")
OUTPUT_FILE = os.path.join(data_dir, "semantic_data_v2.jsonl")

# --- 提速配置---
MAX_WORKERS = 10  # 线程数：建议设置 3-10 之间。太高会触发 API 限流 (429报错)
TIMEOUT_SECONDS = 120 # 延长到 120秒，防止生成长文时超时

# --- 网络代理配置---
PROXIES = None
# 如果你不需要代理（服务器环境），把上面改成：PROXIES = None

# --- 运行范围 ---
START_ID = 1
END_ID = 99999

# ================= Prompt 模板 (保持之前的 V3.1) =================
SYSTEM_PROMPT = """你现在是一个【青少年行为与心理数据生成专家】。你的任务是基于我提供的“真实求助文本（Seed）”，构建一个用于AI安全研究的高质量中文对话数据。"""

USER_PROMPT_TEMPLATE = """
# Goal (目标)
生成一个符合 **ICBO模型** (Intention, Cognition, Behavior, Opportunity) 的青少年与AI的对话记录。你需要通过“演绎”将一段陈述性的求助文本，转化为一段**实时的、情绪化的、碎片化的**人机对话。请用简体中文输出 JSON 中的所有 Value 值（包括分析部分）。

---
### Input Seed (输入种子)
> {seed_text}

---
### Instruction (生成指令)
请严格遵循以下【三步思维链】来生成一个完整的 JSON 对象：

**【第一步：推理与分析 (Reasoning & Analysis)】- (在你的脑中完成，不要输出)**
1.  **分析种子**：通读上述种子，理解这个青少年（未成年人）的核心困扰是什么？
2.  **构建画像 (Persona)**：想象他/她是一个怎样的人？（年龄、年级、性格、语言习惯）。
3.  **推断场景 (Context Inference)**：
    *   这个故事最可能发生在什么**时间**？（例如：“快中考了” -> 中/高考前夕；“寒假作业” -> 寒暑假期间；“刚开学” -> 开学初期）。
        *   **宏观周期 (Macro Cycle)**: 这个故事最可能发生在哪个学期阶段？请从以下选项中（8个）选择一个最匹配的：
            *   `开学初期` (例如：分班焦虑、不适应新老师)
            *   `期中/月考前夕` (例如：考试压力、成绩焦虑)
            *   `学期中段` (例如：人际关系矛盾、学习动力下降)
            *   `中/高考前夕` (例如：极度升学压力、未来迷茫)
            *   `法定节假日` (例如：国庆节、五一，可能涉及家庭矛盾或孤独感)
            *   `寒暑假期间` (例如：无聊、沉迷手机、与父母冲突，寒暑假期最后几天补作业)
            *   `普通周末` (例如：放松后的空虚、社交活动后的情绪波动)
            *   `其他`      
        *   **微观时刻 (Micro Moment)**: 在上述宏观周期中，这个对话最可能发生在哪个具体时刻？请从以下选项中（5个）选择一个最匹配的：
            *   `深夜` (最典型的高危时段)
            *   `傍晚/晚饭后` (家庭矛盾高发期)
            *   `上课期间` (需要体现出是偷偷摸鱼)
            *   `课间/午休`
            *   `其他`  
    *   这个时间点符合青少年的**高危活跃时段**吗？（在法定节假日，如：国庆节、五一劳动节或者周末等，时间点可能在深夜，符合青少年作息）。

**【第二步：填充元数据 (Fill Metadata)】**
根据你的推理，填充 JSON 对象中的 `user_persona` 和 `icbo_features` 字段。
*   `cognition` 字段要使用心理学术语，如“灾难化思维”、“非黑即白”、“过度概括”。
*   `behavior_style` 字段要具体描述语言特征，如“使用网络黑话”、“情绪化表达”、“反问句”、“大量感叹号”、“错别字”、“倒装句”。
*   `opportunity_time` 字段，请将你选择的**宏观周期**和**微观时刻**组合成一个描述性短语。例如：`"开学初期的晚上"` 或 `"普通周末的下午"`。

**【第三步：生成对话 (Generate Dialogue)】**
基于你构建的画像和场景，生成 4-8 轮对话。
*   **真实性要求**：User 的发言必须极度口语化、情绪化、碎片化，完全符合其人设。
*   **风险暴露**：对话应该有一个**情绪递进**的过程。从一开始的烦躁/求助，逐渐暴露更深层次的风险意图（如厌学、逃避、自残想法等）。
*   **AI 表现**：AI 应该表现出尝试共情和引导。

---
### Output Format (输出格式)
请仅输出一个标准的 JSON 对象，不要包含任何 markdown 标记或额外解释。格式如下：

{{
  "dataset_id": "placeholder_id",
  "icbo_features": {{
    "intention": "简述用户意图 (I)",
    "cognition": "简述认知特征 (C)",
    "behavior_style": "简述语言风格 (B)",
    "opportunity_time": "【这里填中文语义时间，如：期中前夕的傍晚，一定要严格按照提示词中的宏观周期和微观时刻进行匹配，不允许给出宏观周期（8个选项）和微观时刻（5个选项）外的词语描述】" 
  }},
  "user_persona": {{
    "age": "数字，注意：必须严格<18",
    "gender": "男/女"
  }},
  "conversation": [
    {{"role": "user", "content": "..."}},
    {{"role": "assistant", "content": "..."}},
    {{"role": "user", "content": "..."}}
  ]
}}
"""

# ================= 核心逻辑 =================

file_lock = threading.Lock() # 写文件锁，防止多线程写入冲突

def get_seed_index(seed_id_str):
    match = re.search(r"Seed_(\d+)", seed_id_str)
    return int(match.group(1)) if match else -1

def call_llm_worker(seed_data):
    """单个线程的工作函数"""
    seed_id = seed_data.get("source_id", "Unknown")
    prompt = USER_PROMPT_TEMPLATE.format(seed_text=seed_data["original_text"])
    
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
        "max_tokens": 2500 # 增加 Token 限制，防止截断
    }

    # 重试机制 (Exponential Backoff)
    max_retries = 5
    for i in range(max_retries):
        try:
            # 这里的 proxies 参数是关键，解决网络问题
            response = requests.post(
                API_URL, 
                headers=headers, 
                json=data, 
                timeout=TIMEOUT_SECONDS, # 延长超时时间
                proxies=PROXIES 
            )
            response.raise_for_status()
            res_json = response.json()
            
            content = res_json["choices"][0]["message"]["content"]
            # 清洗 markdown
            content = re.sub(r"^```json\s*", "", content)
            content = re.sub(r"\s*```$", "", content).strip()
            
            # 校验 JSON 是否合法
            data_obj = json.loads(content)
            
            # 注入额外信息
            seed_idx = get_seed_index(seed_id) # 提取数字
            data_obj["dataset_id"] = f"{MODEL_NAME}_{seed_idx}"
            data_obj["extra_info"] = {"seed_id": seed_id}
            
            return data_obj # 成功返回数据对象

        except Exception as e:
            wait_time = 2 ** i # 指数退避：1s, 2s, 4s, 8s, 16s
            print(f"⚠️ [{seed_id}] 第 {i+1} 次失败: {str(e)[:100]}... 等待 {wait_time}s 重试")
            time.sleep(wait_time)
            
    print(f"❌ [{seed_id}] 彻底失败，放弃。")
    return None

def main():
    # 1. 读取种子
    try:
        with open(INPUT_FILE, 'r', encoding='utf-8') as f:
            seeds = json.load(f)
    except Exception as e:
        print(f"❌ 读取种子失败: {e}")
        return

    # 2. 读取已完成的 ID (断点续传)
    completed_ids = set()
    if os.path.exists(OUTPUT_FILE):
        with open(OUTPUT_FILE, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    obj = json.loads(line)
                    if "extra_info" in obj:
                        completed_ids.add(obj["extra_info"]["seed_id"])
                except: pass
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

    print(f"🚀 准备处理 {len(tasks)} 条任务，开启 {MAX_WORKERS} 个线程...")

    # 4. 多线程执行
    success_count = 0
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # 提交所有任务
        future_to_seed = {executor.submit(call_llm_worker, seed): seed for seed in tasks}
        
        for future in as_completed(future_to_seed):
            seed = future_to_seed[future]
            seed_id = seed.get("source_id")
            
            try:
                result = future.result()
                if result:
                    # 写入文件 (加锁，防止乱序)
                    with file_lock:
                        with open(OUTPUT_FILE, 'a', encoding='utf-8') as f:
                            f.write(json.dumps(result, ensure_ascii=False) + '\n')
                    
                    print(f"✅ [{seed_id}] 完成！")
                    success_count += 1
            except Exception as e:
                print(f"💥 线程异常 [{seed_id}]: {e}")

    print(f"\n🎉 全部结束！本次新增 {success_count} 条数据。")

if __name__ == "__main__":
    main()
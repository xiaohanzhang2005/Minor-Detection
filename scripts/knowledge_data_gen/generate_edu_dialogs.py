#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
批量从教材目录 txt 构造 K12 知识点 Seed，并调用大模型生成对话 JSON 数据。

使用说明（示例）：
    python generate_edu_dialogs.py ^
        --input_dir data/primary_raw ^
        --output_jsonl data/edu_dialogs.jsonl ^
        --model gemini-3-flash-preview ^
        --max_per_file 50

依赖：
    pip install openai python-dateutil

环境变量：
    AIHUBMIX_API_KEY: 大模型 API Key（与 llm_service.py 保持一致，通过 AiHubMix 调用 Gemini）
"""

import argparse
import json
import os
import re
import time
import random
from datetime import datetime
from typing import Dict, Iterable, List, Optional

try:
    from openai import OpenAI  # type: ignore
except ImportError:
    OpenAI = None  # type: ignore


CHINESE_NUM_TO_INT = {
    "一": 1,
    "二": 2,
    "三": 3,
    "四": 4,
    "五": 5,
    "六": 6,
    "七": 7,
    "八": 8,
    "九": 9,
}


def detect_stage_subject_from_filename(filename: str) -> (str, str):
    """从文件名推断学段（小学/初中/高中）和学科"""
    name, _ = os.path.splitext(os.path.basename(filename))
    stage = "未知学段"
    if name.startswith("小学"):
        stage = "小学"
        subject = name[len("小学") :]
    elif name.startswith("初中"):
        stage = "初中"
        subject = name[len("初中") :]
    elif name.startswith("高中"):
        stage = "高中"
        subject = name[len("高中") :]
    else:
        subject = name
    return stage, subject


def parse_grade_from_line(line: str, stage: str) -> Optional[int]:
    """
    从诸如 "## 一年级" / "## 七年级" / "## 高一" / "必修第一册" 这样的标题行推断年级（1-12 的数字）。

    约定：
        - 小学：一年级-六年级 -> 1-6
        - 初中：七年级/八年级/九年级 -> 7-9；有的教材也会写"初一/初二/初三"，在这种情况下按 7-9 映射
        - 高中：高一/高二/高三 -> 10-12；必修第一册/第二册/第三册 -> 10/11/12
    """
    line = line.strip().lstrip("#").strip()
    
    # 高中：必修第一册 / 必修第二册 / 必修第三册 / 选择性必修1/2/3
    if stage == "高中":
        # 匹配"必修第一册"、"必修第二册"、"必修第三册"
        m = re.search(r"必修[第]?([一二三])册", line)
        if m:
            cn = m.group(1)
            num = CHINESE_NUM_TO_INT.get(cn)
            if num:
                return 9 + num  # 必修第一册->10, 必修第二册->11, 必修第三册->12
        
        # 匹配"选择性必修1"、"选择性必修2"、"选择性必修3"
        m = re.search(r"选择性必修([123])", line)
        if m:
            num = int(m.group(1))
            return 9 + num  # 选择性必修1->10, 选择性必修2->11, 选择性必修3->12
        
        # 匹配"必修1"、"必修2"、"必修3"（简化写法）
        m = re.search(r"必修([123])", line)
        if m:
            num = int(m.group(1))
            return 9 + num  # 必修1->10, 必修2->11, 必修3->12
    
    # 小学 / 初中：X年级
    m = re.match(r"([一二三四五六七八九])年级", line)
    if m:
        cn = m.group(1)
        num = CHINESE_NUM_TO_INT.get(cn)
        if not num:
            return None
        if stage == "小学":
            return num  # 1-6
        if stage == "初中":
            # 常见写法是"七年级/八年级/九年级"，此时直接映射为 7/8/9
            if cn in ("七", "八", "九"):
                return num  # 七年级->7, 八年级->8, 九年级->9
            # 少数教材可能写"初一/初二/初三"或"初一(七年级)"等，这里兜底：一/二/三 -> 7/8/9
            if cn in ("一", "二", "三"):
                return 6 + num  # 一(初一)->7, 二(初二)->8, 三(初三)->9
            return None
        # 部分高中文本也可能写"高一/一年级"，这里简单兜底
        if stage == "高中":
            return 9 + num  # 一年级->10, 二年级->11, 三年级->12
        return num

    # 高中：高一 / 高二 / 高三
    m = re.match(r"高([一二三])", line)
    if m:
        num = CHINESE_NUM_TO_INT.get(m.group(1))
        if not num:
            return None
        return 9 + num  # 高一->10, 高二->11, 高三->12

    return None


def grade_to_age(grade: int) -> int:
    """
    按要求映射：
        小学(1-6年级): 7-12岁；初中(7-9年级): 13-15岁；高中(10-12年级): 16-18岁。
    """
    if 1 <= grade <= 6:
        return 6 + grade
    if 7 <= grade <= 9:
        return 6 + grade
    if 10 <= grade <= 12:
        return 6 + grade
    # 兜底
    return max(7, min(18, 6 + grade))


def grade_to_label(grade: int) -> str:
    if 1 <= grade <= 6:
        cn = {1: "一", 2: "二", 3: "三", 4: "四", 5: "五", 6: "六"}[grade]
        return f"小{cn}"
    if 7 <= grade <= 9:
        cn = {7: "一", 8: "二", 9: "三"}[grade]
        return f"初{cn}"
    if 10 <= grade <= 12:
        cn = {10: "一", 11: "二", 12: "三"}[grade]
        return f"高{cn}"
    return f"{grade}年级"


def clean_knowledge_text(line: str) -> Optional[str]:
    """
    将目录中的一行清洗为“知识点描述”：
        - 去掉行首编号：“1. xxx”“一、xxx”
        - 去掉末尾页码信息：“（12页）”
    不适合作为知识点的行返回 None。
    """
    raw = line.strip()
    if not raw:
        return None
    if raw.startswith("#"):
        return None

    # 有的行可能是“第一单元 xxx”，也可以当作知识点
    # 去掉前缀编号（阿拉伯数字）
    s = re.sub(r"^[0-9]+\s*[\.．、]?\s*", "", raw)
    # 去掉前缀“第X单元/课/章”等，也可以保留，这里只做简单处理
    # 去掉末尾形如 "（12页）" " (p12)" 等
    s = re.sub(r"（[^）]*页）", "", s)
    s = re.sub(r"\([^)]*页\)", "", s)
    s = s.strip(" 。:：")

    if not s:
        return None
    return s


def iter_seeds_from_file(path: str) -> Iterable[Dict]:
    """
    从单个 txt 中遍历生成 seeds:
        {stage, subject, grade, grade_label, age, knowledge}
    """
    stage, subject = detect_stage_subject_from_filename(path)
    current_grade: Optional[int] = None
    # 对于高中教材，如果没有明确的年级标识，使用默认值（必修第一册->高一）
    default_grade_for_stage = {
        "小学": None,
        "初中": None,
        "高中": 10  # 默认高一，如果遇到必修第二册/第三册会更新
    }
    fallback_grade = default_grade_for_stage.get(stage)

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            # 先试图更新年级上下文
            g = parse_grade_from_line(line, stage)
            if g:
                current_grade = g
                continue

            knowledge = clean_knowledge_text(line)
            if not knowledge:
                continue
            
            # 如果没有年级信息，使用fallback（仅对高中有效）
            if not current_grade:
                if fallback_grade is not None:
                    current_grade = fallback_grade
                else:
                    # 没有年级信息时，跳过（小学和初中必须明确年级）
                    continue

            age = grade_to_age(current_grade)
            grade_label = grade_to_label(current_grade)
            seed_text = f"{grade_label}-{subject}-{knowledge}"

            yield {
                "stage": stage,
                "subject": subject,
                "grade": current_grade,
                "grade_label": grade_label,
                "age": age,
                "knowledge": knowledge,
                "seed": seed_text,
            }


def collect_seeds(input_dir: str, max_per_file: Optional[int] = None, file_filter: Optional[List[str]] = None) -> List[Dict]:
    """
    收集所有 seeds；如果设置了 max_per_file，则对每个 txt 文件内的知识点做随机抽样，
    而不是简单取前 max_per_file 条，避免只集中在目录前部。
    
    Args:
        input_dir: 输入目录
        max_per_file: 每个文件最多采样多少个知识点
        file_filter: 可选的文件名列表，只处理这些文件（例如 ["高中化学.txt", "初中生物.txt"]）
    """
    seeds: List[Dict] = []
    for name in sorted(os.listdir(input_dir)):
        if not name.endswith(".txt"):
            continue
        # 如果指定了文件过滤，只处理匹配的文件
        if file_filter is not None:
            if name not in file_filter:
                continue
        path = os.path.join(input_dir, name)
        file_seeds = list(iter_seeds_from_file(path))
        if max_per_file is not None and len(file_seeds) > max_per_file:
            file_seeds = random.sample(file_seeds, max_per_file)
        seeds.extend(file_seeds)
    return seeds


def collect_one_seed_per_file(input_dir: str, file_filter: Optional[List[str]] = None) -> List[Dict]:
    """
    从每个教材 txt 中仅采样一个知识点 Seed（通常是文件中解析到的第一个知识点）。
    适合"每个教材文件只生成一个对话样本"的场景。
    
    Args:
        input_dir: 输入目录
        file_filter: 可选的文件名列表，只处理这些文件（例如 ["高中化学.txt", "初中生物.txt"]）
    """
    seeds: List[Dict] = []
    for name in sorted(os.listdir(input_dir)):
        if not name.endswith(".txt"):
            continue
        # 如果指定了文件过滤，只处理匹配的文件
        if file_filter is not None:
            if name not in file_filter:
                continue
        path = os.path.join(input_dir, name)
        # 只取该文件解析到的第一个 seed
        first_seed: Optional[Dict] = None
        for seed in iter_seeds_from_file(path):
            first_seed = seed
            break
        if first_seed is not None:
            seeds.append(first_seed)
    return seeds


def build_messages_for_seed(seed: Dict) -> List[Dict[str, str]]:
    """
    构造调用大模型的 messages（system + user），让模型按照你给的模板生成 JSON。
    """
    system_prompt = (
        "你现在是一个【K12教育心理与知识构建专家】。\n"
        "你的任务是基于我提供的“教材知识点（Seed）”，构建一个用于 AI 教育研究的高质量中文对话数据。\n\n"
        "目标：生成一个符合青少年认知发展规律的“青少年与 AI 学习助手”的对话记录，而不是“老师与学生”的课堂对话。\n"
        "对话双方的设定如下：\n"
        "  - role = \"user\"：一位正在使用学习 App 的中国青少年学生，说话口语化、自然，可以带少量网络用语，但不要过度夸张。\n"
        "  - role = \"assistant\"：在线 AI 学习助手，不是老师，不要自称“老师”，也不要称呼对方为“同学/同学们/同学们好”等，只用“你”来称呼对方。\n"
        "  - 整体氛围更像手机上与 AI 聊天解题，而不是课堂板书讲解。\n\n"
        "语言风格要求：\n"
        "  - user 的表达贴近真实青少年，说话有情绪、有疑惑，可以有一点吐槽；\n"
        "  - assistant 的表达要有耐心、友好、逻辑清晰，可以适度使用口语（如“我们可以先这样想”“你可以试着…”），\n"
        "    但保持理性和尊重，不要批评或贬低；\n"
        "  - 【重要】assistant 只负责回答和解释，不向 user 提问或反问，不通过提问来引导思考，\n"
        "    每一轮 assistant 的发言都应以陈述句为主，不要以问号结尾（除非是在引用题目原文中的问句）。\n"
        "  - 对话整体应呈现为“user 主动提问、assistant 进行回答”的单向问答结构，而不是互相提问的双向访谈。\n\n"
        "输出要求：严格输出一个 JSON 对象，不要包含任何解释或多余文本，键名与示例完全一致。"
    )

    seed_text = seed["seed"]
    seed_grade_label = seed["grade_label"]
    seed_age = seed["age"]
    seed_stage = seed["stage"]
    seed_subject = seed["subject"]

    user_prompt = f"""
请根据以下规范生成一条高质量的对话数据（总体字数建议控制在 400-600 字之间）：

Input Seed（输入种子）：
【{seed_text}】

当前 Seed 所在学段 / 学科 / 年级 / 年龄大致信息：
- 学段：{seed_stage}
- 学科：{seed_subject}
- 年级：{seed_grade_label}
- 年龄：约 {seed_age} 岁
你生成的 user_persona.age / user_persona.grade 必须与以上信息保持一致（只允许出现轻微合理偏差，例如晚读或留级一岁以内）。

生成步骤：
1. User Profile（用户画像构建）：
    - Age & Grade：必须符合中国 K12 学段，并与上文给出的 Seed 年级和年龄信息一致。小学(1-6年级): 7-12岁；初中(7-9年级): 13-15岁；高中(10-12年级): 16-18岁。
    - Cognition (C)：设定该年龄段在该知识点上的常见知识误区。
    - Behavior Style (B)：根据该年龄和性格，设定多样化的说话风格（如“紧张但认真”“爱吐槽、爱用网络梗”“比较内向、句子简短”等），
      同一年龄段也应体现不同个体差异，避免所有样本的语气都一样。

2. Context Setting（情境设定）：
    - Opportunity (O)：设定合理的时间，并在 icbo_features.opportunity_time 字段中用“具体的日期+时间”来表示，
      格式必须为 "YYYY-MM-DD HH:MM"（例如："2026-02-12 20:30"），不能使用“周三深夜”“周末写作业时”“晚自习时间”等模糊描述。
    - Intention (I)：设定求助动机（如：题不会做、想不通原理、对实验好奇）。

3. Dialogue Generation（对话生成）：
    - 生成 4-8 轮对话，不要少于 4 轮，也不要超过 8 轮。
    - 对话必须由 user 先发起，第一轮 role 一定是 "user"，内容是针对 Seed 相关知识点的提问或求助，而不是闲聊；assistant 不主动先开口。
    - 每轮发言尽量简短、直截了当，避免特别长的段落，但整体对话可以稍微丰富一些。
    - 真实性要求：User 的话语要符合其年龄，像在手机上和 AI 学习助手聊天；避免出现“老师您好”“老师我有个问题”等明显课堂场景称呼。
    - AI 表现：assistant 作为在线 AI 学习助手，不自称“老师”，不说“同学们/同学”，用平等、友好的语气与 user 对话，
      直接给出解释、步骤、示例或建议，不向 user 提问或反问，不使用“你觉得呢？”“要不要先想一想这一步？”等引导式问题。
    - 对话中的 user 语言风格要严格体现 icbo_features.behavior_style 中描述的特点（例如“爱吐槽”“语气紧张但礼貌”等），
      可以适度使用与该风格和年龄匹配的口语或网络用语（但不要过度堆砌梗）。

Output Format（输出格式）：
请仅输出一个标准的 JSON 对象，不要包含 markdown 标记或额外解释。格式必须如下（示例中的值仅作示范，模型需根据 Seed 自行生成合理内容）：
{{
  "dataset_id": "Gen_Edu_2026-02-12_1739337600",
  "icbo_features": {{
    "intention": "搞清该知识点的关键概念和解题方法",
    "cognition": "该年龄段常把相关概念混淆，难以理解本质差异",
    "behavior_style": "语言紧张但略带幽默，夹杂少量网络用语",
    "opportunity_time": "2026-02-12 20:30"
  }},
  "user_persona": {{
    "age": 13,
    "gender": "男",
    "grade": "初一"
  }},
  "conversation": [
    {{"role": "user", "content": "第一轮用户发言示例"}},
    {{"role": "assistant", "content": "第一轮助教回应示例"}}
  ]
}}

注意：
- 必须严格遵守上述 JSON 结构和键名。
- 所有字符串必须使用双引号，不能出现未转义的换行或引号，不能在字符串中直接出现未转义的换行符。
- conversation 中轮数必须在 4-8 轮之间，总体对话内容尽量控制在 400-600 字以内。
- conversation[0].role 必须是 "user"，且内容中应明确提出与知识点相关的疑问或任务，assistant 不得在对话中向 user 提问或反问。
- icbo_features.opportunity_time 必须是具体的日期时间字符串，例如 "2026-02-12 20:30"，不能是描述性或模糊时间表达。
- 不要在 JSON 外输出任何多余文本。
"""

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]


def ensure_client():
    if OpenAI is None:
        raise RuntimeError(
            "未安装 openai SDK，请先运行：pip install openai"
        )
    # 与 llm_service.py 保持一致：通过 AiHubMix 调用 Gemini，密钥统一从环境变量获取
    api_key = os.environ.get("AIHUBMIX_API_KEY")
    if not api_key:
        raise RuntimeError(
            "未设置 AIHUBMIX_API_KEY 环境变量，请在运行前配置你的 AiHubMix API Key。"
        )
    return OpenAI(
        api_key=api_key,
        base_url="https://aihubmix.com/v1",
    )


def call_model(client, model: str, messages: List[Dict[str, str]]) -> Dict:
    """
    调用大模型，返回解析后的 JSON 对象。
    使用新的 Chat Completions 接口，并要求 response_format=json_object。
    """
    resp = client.chat.completions.create(
        model=model,
        messages=messages,
        response_format={"type": "json_object"},
        temperature=0.2,
        # 提高上限，减少长 JSON 被截断的概率
        # 适当提高上限，支持稍长的多轮对话，减少长 JSON 被截断的概率
        max_tokens=4096,
    )

    # 优先使用 SDK / 服务端已经解析好的 JSON（如果可用）
    msg = resp.choices[0].message
    parsed = getattr(msg, "parsed", None)
    if parsed is not None:
        return parsed

    content = msg.content
    if not content:
        raise ValueError("模型返回为空")
    try:
        return json.loads(content)
    except json.JSONDecodeError as e:
        # 如果模型没完全遵守 json_object，尝试简单修正，再不行就把原文写入日志方便排查
        content_stripped = content.strip()
        if content_stripped.startswith("```"):
            # 去掉可能的 ```json ``` 包裹
            content_stripped = re.sub(r"^```[a-zA-Z]*", "", content_stripped)
            content_stripped = re.sub(r"```$", "", content_stripped).strip()
        try:
            return json.loads(content_stripped)
        except json.JSONDecodeError:
            # 记录到本地文件，便于你打开查看哪里没有转义
            try:
                with open("bad_json_debug.txt", "a", encoding="utf-8") as dbg:
                    dbg.write("==== Raw model content (JSON parse failed) ====\n")
                    dbg.write(repr(content_stripped))
                    dbg.write("\n\n")
            except Exception:
                pass
            raise ValueError(
                f"JSON 解析失败，原始内容片段：{repr(content_stripped[:200])}..."
            ) from e


def main():
    parser = argparse.ArgumentParser(
        description="从教材目录 txt 批量生成 K12 教育对话 JSON 数据"
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        default="data/primary_raw",
        help="教材目录 txt 所在文件夹",
    )
    parser.add_argument(
        "--output_jsonl",
        type=str,
        default="data/edu_dialogs.jsonl",
        help="输出 JSONL 文件路径（每行一个样本）。当未指定 --output_dir 时生效。",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="如果指定，则为每条对话单独生成一个 JSON 文件到该目录中（不再写入 JSONL）。",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gemini-3-flash-preview",
        help="大模型名称，例如 gemini-3-flash-preview",
    )
    parser.add_argument(
        "--max_per_file",
        type=int,
        default=50,
        help="每个 txt 最多采样多少个知识点（防止一次生成过大）",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="总共最多生成多少条（可选）",
    )
    parser.add_argument(
        "--one_per_file",
        action="store_true",
        help="从每个教材 txt 中只选取一个知识点（通常是解析到的第一个），适合“每个教材文件生成一个对话”的场景。",
    )
    parser.add_argument(
        "--sleep",
        type=float,
        default=0.5,
        help="每次请求后 sleep 秒数，避免触发限流",
    )
    parser.add_argument(
        "--files",
        type=str,
        nargs="+",
        default=None,
        help="指定要处理的文件名列表（例如：--files 高中化学.txt 初中生物.txt），如果不指定则处理所有文件",
    )

    args = parser.parse_args()

    # 根据需求选择采样策略
    if args.one_per_file:
        seeds = collect_one_seed_per_file(args.input_dir, file_filter=args.files)
    else:
        seeds = collect_seeds(args.input_dir, max_per_file=args.max_per_file, file_filter=args.files)
    if args.limit is not None:
        seeds = seeds[: args.limit]

    if not seeds:
        print("未在目录中解析到任何 Seed，请检查输入文件格式。")
        return

    print(f"共解析到 {len(seeds)} 个知识点 Seed，将依次调用模型生成对话。")

    client = ensure_client()

    # 如果指定了 output_dir，则每条对话输出为一个独立 JSON 文件
    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
        for idx, seed in enumerate(seeds, 1):
            messages = build_messages_for_seed(seed)
            try:
                obj = call_model(client, args.model, messages)
            except Exception as e:
                print(f"[{idx}/{len(seeds)}] Seed 生成失败：{seed['seed']}，错误：{e}")
                continue

            meta = {
                "_seed": seed["seed"],
                "_stage": seed["stage"],
                "_subject": seed["subject"],
                "_grade": seed["grade"],
                "_grade_label": seed["grade_label"],
                "_age": seed["age"],
            }
            if isinstance(obj, dict):
                obj.setdefault("_meta", meta)

            # 构造一个相对简洁且可读的文件名：序号+学段+学科+年级
            stage = seed.get("stage", "未知学段")
            subject = seed.get("subject", "未知学科")
            grade_label = seed.get("grade_label", "未知年级")
            filename = f"{idx:03d}_{stage}_{subject}_{grade_label}.json"
            file_path = os.path.join(args.output_dir, filename)

            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(obj, f, ensure_ascii=False, indent=2)

            print(f"[{idx}/{len(seeds)}] 已完成：{seed['seed']} -> {file_path}")
            time.sleep(args.sleep)

        print(f"全部完成，独立 JSON 文件已保存到目录：{args.output_dir}")
        return

    # 默认行为：写入单个 JSONL 文件
    os.makedirs(os.path.dirname(args.output_jsonl), exist_ok=True)
    with open(args.output_jsonl, "w", encoding="utf-8") as out_f:
        for idx, seed in enumerate(seeds, 1):
            messages = build_messages_for_seed(seed)
            try:
                obj = call_model(client, args.model, messages)
            except Exception as e:
                print(f"[{idx}/{len(seeds)}] Seed 生成失败：{seed['seed']}，错误：{e}")
                continue

            # 给每条补充一下可溯源字段（不破坏你定义的主结构）
            meta = {
                "_seed": seed["seed"],
                "_stage": seed["stage"],
                "_subject": seed["subject"],
                "_grade": seed["grade"],
                "_grade_label": seed["grade_label"],
                "_age": seed["age"],
            }
            if isinstance(obj, dict):
                obj.setdefault("_meta", meta)
            out_f.write(json.dumps(obj, ensure_ascii=False))
            out_f.write("\n")

            print(f"[{idx}/{len(seeds)}] 已完成：{seed['seed']}")
            time.sleep(args.sleep)

    print(f"全部完成，结果已保存到：{args.output_jsonl}")


if __name__ == "__main__":
    main()



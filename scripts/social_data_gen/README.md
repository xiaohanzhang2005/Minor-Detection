# 青少年社交问答数据生成工具

本模块是为“青少年守护”项目专门设计的数据生成流水线，旨在构建一个高质量、符合特定研究框架（ICBO模型）的青少年社交问答数据库。

## 文件说明

-   **`get_seeds.py` (种子筛选)**
    -   **作用**: 从公开数据集 `lsy641/PsyQA` 中筛选出与青少年心理、社交、学业等主题相关的真实求助文本。
    -   **核心逻辑**: 通过关键词（如“初中”、“高考”、“朋友”、“父母”）和年龄（<18岁）进行双重验证，过滤掉成年人或无关内容，生成高质量的“种子”文件（如 `youth_seeds_v5.json`）。

-   **`batch_generate.py` (批量生成)**
    -   **作用**: 读取 `get_seeds.py` 生成的种子文件，利用大语言模型（LLM）批量生成模拟的青少年与AI的对话数据。
    -   **核心逻辑**: 将每个“种子”文本包装成一个精心设计的 Prompt，调用 API，引导模型输出结构化的 JSONL 数据。脚本支持多线程、断点续传和自动重试。

-   **`time.py` (时间转换)**【暂时未启用】
    -   **作用**: (此脚本目前未在主流程中启用) 用于将 `batch_generate.py` 输出结果中的模糊语义时间（如“中考前夕的深夜”）转换为精确的、带年份的日期时间格式（如 `2023年06月12日 周一 23:45`）。

## 使用流程

1.  **筛选种子**: 运行 `get_seeds.py` 脚本。
    ```bash
    python get_seeds.py
    ```
    -   这将从数据集中筛选并生成一个名为 `youth_seeds_vX.json` 的文件，后期经过人工二次筛选，得到包含用于生成对话的原始求助文本。

2.  **批量生成对话**:
    -   在 `batch_generate.py` 脚本中配置好你的 `API_KEY`。
    -   确保 `INPUT_FILE` 指向第1步生成的种子文件。
    -   运行 `batch_generate.py` 脚本。
    ```bash
    python batch_generate.py
    ```
    -   脚本会读取种子，并发调用 API 生成对话。最终结果会以 JSONL 格式追加保存在 `data/社交问答/semantic_data_v2.jsonl` 文件中。

## 数据说明

最终输出的文件为 `semantic_data_v2.jsonl`，每一行都是一个独立的 JSON 对象，其核心结构如下：

```json
{
  "dataset_id": "gemini-3-flash-preview_123",
  "icbo_features": {
    "intention": "寻求关于人际关系困扰的建议",
    "cognition": "灾难化思维，认为自己被所有人讨厌",
    "behavior_style": "情绪化表达，使用网络用语",
    "opportunity_time": "期中考试前的晚上"
  },
  "user_persona": {
    "age": "16",
    "gender": "女"
  },
  "conversation": [
    {"role": "user", "content": "烦死了，感觉大家都不喜欢我..."},
    {"role": "assistant", "content": "能具体和我说说发生了什么事吗？"},
    {"role": "user", "content": "..."}
  ],
  "extra_info": {
      "seed_id": "Seed_123"
  }
}
```

-   `icbo_features`: 描述了用户的意图(I)、认知(C)、行为(B)和发生时机(O)。
-   `user_persona`: 描述了模拟用户的基本画像。
-   `conversation`: 记录了用户与AI之间的多轮对话。

---
name: minor-detection
description: 当用户或上层系统需要判断聊天记录中的说话者是否可能是未成年人、青少年、中学生、高中生，或需要对单会话、多会话历史做年龄倾向、校园倾向、学生画像、未成年人风险与证据分析时使用此技能。即使用户没有直接说"未成年人识别"，但需求本质上是判断"像不像未成年用户"、输出未成年人概率、画像、趋势、风险等级或结构化证据，也应激活此技能。
---

# Minor Detection

把这项 skill 视为"调用固定总控脚本并原样返回 JSON"的能力，不要把它当成自由推理题。

## Hard Contract

你必须遵守以下规则：
- 必须调用 `scripts/run_minor_detection_pipeline.py`，不要自己重做未成年人识别推理。
- 不要自己直接调用 `extract_time_features.py` 或 `retrieve_cases.py`；这些由总控脚本内部决定是否调用。
- 脚本成功时，最终回复必须是脚本 stdout 返回的单个 JSON object，原样返回。
- 脚本失败时，只返回错误信息，不要自行补判。
- 必须确保时间特征提取和案例检索在证据合成前完成。

## Input Preparation

把当前输入整理成一个 JSON payload，再交给总控脚本。

优先兼容的 payload 字段：
- `mode`
- `conversation`
- `sessions`
- `context`
- `meta`
- `request_id`
- `sample_id`
- `user_id`

`context` 必须包含：
- `raw_time_hint` (原始时间线索)
- `opportunity_time` (机会时间)
- `time_features` (结构化时间特征)
- `prior_profile` (先验画像)
- `retrieved_cases` (检索案例)

如果上游已经给了完整 JSON，就不要擅自改写结构，只做最小整理。

## Execution

执行方式：
1. 确保时间特征已提取并写入 payload
2. 把整理后的 payload 写入一个临时 JSON 文件
3. 运行：`python scripts/run_minor_detection_pipeline.py --payload-file <payload-file>`
4. 读取 stdout 中的 JSON
5. 将该 JSON 原样作为最终答案返回

## Output Rules

最终回答必须满足：
- 只输出一个 JSON object
- 不输出 Markdown
- 不输出解释性前言
- 不补充 skill 外推理
- 必须包含完整的 schema 字段

## Error Handling

遇到以下情况直接返回错误：
- 脚本执行超时
- 输出 JSON 解析失败
- 缺少必填字段
- 时间特征缺失

## Boundary Cases

特别注意以下边界情况：
- 讨论学业内容但无明确学段信息时，优先检查时间特征和检索案例
- 使用"救命"等网络用语但无其他证据时，不单独作为判定依据
- 讨论历史/政治等学术内容时，需结合上下文判断是否为作业场景
---
name: minor-detection
description: 对中文聊天记录、单会话、多会话或带上下文的结构化输入做未成年人识别、青少年倾向分析、学生画像分析、校园倾向判断、概率输出和结构化证据输出时使用。无论用户是否直接说“未成年人识别”，只要需求本质上是在判断说话者是否更像未成年人、青少年或学生，并要求输出 JSON 结果，就触发此 skill。
---

# Minor Detection

把这项 skill 视为“调用固定总控脚本并原样返回 JSON”的能力，不要把它当成自由推理题。

## Hard Contract

你必须遵守以下规则：
- 必须调用 `scripts/run_minor_detection_pipeline.py`，不要自己重做未成年人识别推理。
- 不要自己直接调用 `extract_time_features.py` 或 `retrieve_cases.py`；这些由总控脚本内部决定是否调用。
- 脚本成功时，最终回复必须是脚本 stdout 返回的单个 JSON object，原样返回。
- 脚本失败时，只返回错误信息，不要自行补判。
- 对于历史政治类学术问题，需额外检查是否存在"救命"、"脑细胞烧干"等未成年典型表达，避免因检索案例偏差导致误判。

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

`context` 允许包含：
- `raw_time_hint`
- `opportunity_time`
- `time_features`
- `prior_profile`
- `retrieved_cases`

如果上游已经给了完整 JSON，就不要擅自改写结构，只做最小整理。

## Execution

执行方式：
1. 把整理后的 payload 写入一个临时 JSON 文件。
2. 运行：`python scripts/run_minor_detection_pipeline.py --payload-file <payload-file>`
3. 读取 stdout 中的 JSON。
4. 将该 JSON 原样作为最终答案返回。

## Output Rules

最终回答必须满足：
- 只输出一个 JSON object。
- 不输出 Markdown。
- 不输出解释性前言。
- 不补充 skill 外推理。
- 对于历史政治类学术问题，若存在未成年典型表达但被误判为成人，需在`uncertainty_notes`中明确标注"需复核未成年表达特征"。
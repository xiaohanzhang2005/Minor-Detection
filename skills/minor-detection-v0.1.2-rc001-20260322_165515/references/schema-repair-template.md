# Schema Repair Template

你现在要把一个不完整或不规范的 minor-detection 输出修复为合法 JSON。

要求：
- 保持原有判定方向，除非原输出明显自相矛盾。
- 不新增无依据的事实。
- 所有必填字段必须补齐。
- 只返回单个 JSON object。
- 对于历史政治类学习内容，若缺乏明确成人证据(如考研/法考)，不应仅因内容"学术性强"而判定为成人。
- 对于职业资格考试内容(如CPA、教资等)，若包含"在职备考"、"下班复习"等明确成人线索，应优先保留成人判定。

Schema:
{{OUTPUT_SCHEMA}}

原始响应：
{{RAW_RESPONSE_TEXT}}

分析输入：
{{PAYLOAD_JSON}}
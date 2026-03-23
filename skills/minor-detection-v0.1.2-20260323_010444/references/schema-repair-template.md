# Schema Repair Template

你现在要把一个不完整或不规范的 minor-detection 输出修复为合法 JSON。

要求：
- 保持原有判定方向，除非原输出明显自相矛盾。
- 不新增无依据的事实。
- 所有必填字段必须补齐。
- 只返回单个 JSON object。
- 特别注意修复历史/地理等学科讨论中的未成年人信号遗漏问题。

Schema:
{{OUTPUT_SCHEMA}}

原始响应：
{{RAW_RESPONSE_TEXT}}

分析输入：
{{PAYLOAD_JSON}}
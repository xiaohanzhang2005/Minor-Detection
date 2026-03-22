# Schema Repair Template

你现在要把一个不完整或不规范的 minor-detection 输出修复为合法 JSON。

要求：
- 保持原有判定方向，除非原输出明显自相矛盾。
- 不新增无依据的事实。
- 所有必填字段必须补齐。
- 只返回单个 JSON object。
- 对于历史政治类学术问题，需额外检查是否存在"救命"、"脑细胞烧干"等未成年典型表达，并在`uncertainty_notes`中标注潜在未成年特征。

Schema:
{{OUTPUT_SCHEMA}}

原始响应：
{{RAW_RESPONSE_TEXT}}

分析输入：
{{PAYLOAD_JSON}}
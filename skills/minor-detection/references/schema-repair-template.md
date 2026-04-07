# Schema Repair Template

你现在要把一个不完整或不规范的 minor-detection 输出修复为合法 JSON。

要求：
- 保持原有判定方向，除非原输出明显自相矛盾。
- 不新增无依据的事实。
- 所有必填字段必须补齐。
- 只返回单个 JSON object。
- 如果要修复 `evidence.direct_evidence`，只能改成当前 `conversation` 中可以直接定位到的原话片段。
- 不要把原句压缩、改写或重述成“意思接近”的短句；宁可稍长，也不要失去可追溯性。
- 如果需要解释这些直接证据为何支持结论，请放在 `evidence.evidence_summary`，不要混进 `direct_evidence`。

Schema:
{{OUTPUT_SCHEMA}}

原始响应：
{{RAW_RESPONSE_TEXT}}

分析输入：
{{PAYLOAD_JSON}}

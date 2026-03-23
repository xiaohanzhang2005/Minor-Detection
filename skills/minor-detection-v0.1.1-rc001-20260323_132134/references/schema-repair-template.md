# Schema Repair Template

你现在要把一个不完整或不规范的 minor-detection 输出修复为合法 JSON。

要求：
- 保持原有判定方向，除非原输出明显自相矛盾。
- 不新增无依据的事实。
- 所有必填字段必须补齐。
- 只返回单个 JSON object。
- 特别注意边界情况：
  - 学业内容需检查时间特征和检索案例
  - "救命"等网络用语不作为单独判定依据  
  - 历史/政治内容需结合上下文判断作业场景

Schema:
{{OUTPUT_SCHEMA}}

原始响应：
{{RAW_RESPONSE_TEXT}}

分析输入：
{{PAYLOAD_JSON}}
# Formal Skill Review Diff

- base_version: `minor-detection-v0.1.0`
- candidate_version: `minor-detection-v0.1.1-20260323_132134`
- generated_at: `2026-03-23T15:18:25.446981`

请人工审核以下差异，再决定 approve / reject。

## SKILL.md
- changed: `true`
```diff
--- minor-detection-v0.1.0/SKILL.md
+++ minor-detection-v0.1.1-20260323_132134/SKILL.md
@@ -1,11 +1,11 @@
 ---
 name: minor-detection
-description: 当用户或上层系统需要判断聊天记录中的说话者是否可能是未成年人、青少年、中学生、高中生，或需要对单会话、多会话历史做年龄倾向、校园倾向、学生画像、未成年人风险与证据分析时使用此技能。即使用户没有直接说“未成年人识别”，但需求本质上是判断“像不像未成年用户”、输出未成年人概率、画像、趋势、风险等级或结构化证据，也应激活此技能。
+description: 当用户或上层系统需要判断聊天记录中的说话者是否可能是未成年人、青少年、中学生、高中生，或需要对单会话、多会话历史做年龄倾向、校园倾向、学生画像、未成年人风险与证据分析时使用此技能。即使用户没有直接说"未成年人识别"，但需求本质上是判断"像不像未成年用户"、输出未成年人概率、画像、趋势、风险等级或结构化证据，也应激活此技能。
 ---
 
 # Minor Detection
 
-把这项 skill 视为“调用固定总控脚本并原样返回 JSON”的能力，不要把它当成自由推理题。
+把这项 skill 视为"调用固定总控脚本并原样返回 JSON"的能力，不要把它当成自由推理题。
 
 ## Hard Contract
 
@@ -14,6 +14,7 @@
 - 不要自己直接调用 `extract_time_features.py` 或 `retrieve_cases.py`；这些由总控脚本内部决定是否调用。
 - 脚本成功时，最终回复必须是脚本 stdout 返回的单个 JSON object，原样返回。
 - 脚本失败时，只返回错误信息，不要自行补判。
+- 必须确保时间特征提取和案例检索在证据合成前完成。
 
 ## Input Preparation
 
@@ -29,27 +30,44 @@
 - `sample_id`
 - `user_id`
 
-`context` 允许包含：
-- `raw_time_hint`
-- `opportunity_time`
-- `time_features`
-- `prior_profile`
-- `retrieved_cases`
+`context` 必须包含：
+- `raw_time_hint` (原始时间线索)
+- `opportunity_time` (机会时间)
+- `time_features` (结构化时间特征)
+- `prior_profile` (先验画像)
+- `retrieved_cases` (检索案例)
 
 如果上游已经给了完整 JSON，就不要擅自改写结构，只做最小整理。
 
 ## Execution
 
 执行方式：
-1. 把整理后的 payload 写入一个临时 JSON 文件。
-2. 运行：`python scripts/run_minor_detection_pipeline.py --payload-file <payload-file>`
-3. 读取 stdout 中的 JSON。
-4. 将该 JSON 原样作为最终答案返回。
+1. 确保时间特征已提取并写入 payload
+2. 把整理后的 payload 写入一个临时 JSON 文件
+3. 运行：`python scripts/run_minor_detection_pipeline.py --payload-file <payload-file>`
+4. 读取 stdout 中的 JSON
+5. 将该 JSON 原样作为最终答案返回
 
 ## Output Rules
 
 最终回答必须满足：
-- 只输出一个 JSON object。
-- 不输出 Markdown。
-- 不输出解释性前言。
-- 不补充 skill 外推理。
+- 只输出一个 JSON object
+- 不输出 Markdown
+- 不输出解释性前言
+- 不补充 skill 外推理
+- 必须包含完整的 schema 字段
+
+## Error Handling
+
+遇到以下情况直接返回错误：
+- 脚本执行超时
+- 输出 JSON 解析失败
+- 缺少必填字段
+- 时间特征缺失
+
+## Boundary Cases
+
+特别注意以下边界情况：
+- 讨论学业内容但无明确学段信息时，优先检查时间特征和检索案例
+- 使用"救命"等网络用语但无其他证据时，不单独作为判定依据
+- 讨论历史/政治等学术内容时，需结合上下文判断是否为作业场景
```

## references/evidence-rules.md
- changed: `true`
```diff
--- minor-detection-v0.1.0/references/evidence-rules.md
+++ minor-detection-v0.1.1-20260323_132134/references/evidence-rules.md
@@ -1,4 +1,4 @@
-﻿# Evidence Rules
+# Evidence Rules
 
 ## Direct Evidence Priority
 
@@ -7,6 +7,7 @@
 - 明确学段为小学、初中、高中
 - 明确校园身份，如班主任、同桌、住校、年级、月考、中考、高考、作业、晚自习
 - 明确家庭依赖与未成年限制，如家长接送、被老师或家长管理
+- 表达对家庭/校园环境的无力感时伴随具体依赖场景（如"宿舍里被排挤"、"父母争吵无法入睡"）
 
 成人强反证包括：
 - 明确年龄大于等于 18 岁
@@ -17,10 +18,10 @@
 ## Conflict Resolution
 
 当证据冲突时：
-1. 直接证据优先于辅助证据。
-2. 当前输入优先于检索相似案例。
-3. 当前稳定成人信号与当前稳定未成年信号冲突时，应下调置信度并记录到 `conflicting_signals`。
-4. 不要用单一软线索直接翻盘。
+1. 直接证据优先于辅助证据
+2. 当前输入优先于检索相似案例
+3. 当前稳定成人信号与当前稳定未成年信号冲突时，应下调置信度并记录到 `conflicting_signals`
+4. 不要用单一软线索直接翻盘
 
 ## Soft Signals
 
@@ -30,6 +31,7 @@
 - 节假日或开学季时间特征
 - 对父母、老师、同学的零散提及
 - 相似案例检索结果
+- 使用"救命"等网络流行语
 
 ## False Positive Guidance
 
@@ -46,3 +48,4 @@
 - 家长或老师主导的管束关系
 - 典型未成年人时间安排与依赖型行为
 - 多会话中稳定重复的学生身份线索
+- 表达对家庭/校园环境的无力感时伴随具体依赖场景
```

## references/classifier-system.md
- changed: `false`

## references/classifier-user-template.md
- changed: `false`

## references/retrieval-query-template.md
- changed: `false`

## references/schema-repair-template.md
- changed: `true`
```diff
--- minor-detection-v0.1.0/references/schema-repair-template.md
+++ minor-detection-v0.1.1-20260323_132134/references/schema-repair-template.md
@@ -1,4 +1,4 @@
-﻿# Schema Repair Template
+# Schema Repair Template
 
 你现在要把一个不完整或不规范的 minor-detection 输出修复为合法 JSON。
 
@@ -7,6 +7,10 @@
 - 不新增无依据的事实。
 - 所有必填字段必须补齐。
 - 只返回单个 JSON object。
+- 特别注意边界情况：
+  - 学业内容需检查时间特征和检索案例
+  - "救命"等网络用语不作为单独判定依据  
+  - 历史/政治内容需结合上下文判断作业场景
 
 Schema:
 {{OUTPUT_SCHEMA}}
```

## scripts/config.py
- changed: `false`

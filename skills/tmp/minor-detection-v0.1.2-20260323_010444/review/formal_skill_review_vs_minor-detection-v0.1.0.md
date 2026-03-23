# Formal Skill Review Diff

- base_version: `minor-detection-v0.1.0`
- candidate_version: `minor-detection-v0.1.2-20260323_010444`
- generated_at: `2026-03-23T02:13:13.676694`

请人工审核以下差异，再决定 approve / reject。

## SKILL.md
- changed: `true`
```diff
--- minor-detection-v0.1.0/SKILL.md
+++ minor-detection-v0.1.2-20260323_010444/SKILL.md
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
+- 当用户讨论中学课程内容(如历史、地理等)且表达学习压力时，应特别注意识别潜在的未成年人信号。
 
 ## Input Preparation
 
```

## references/evidence-rules.md
- changed: `true`
```diff
--- minor-detection-v0.1.0/references/evidence-rules.md
+++ minor-detection-v0.1.2-20260323_010444/references/evidence-rules.md
@@ -1,4 +1,4 @@
-﻿# Evidence Rules
+# Evidence Rules
 
 ## Direct Evidence Priority
 
@@ -7,6 +7,8 @@
 - 明确学段为小学、初中、高中
 - 明确校园身份，如班主任、同桌、住校、年级、月考、中考、高考、作业、晚自习
 - 明确家庭依赖与未成年限制，如家长接送、被老师或家长管理
+- 讨论中学课程内容(历史/地理等)并表达学习压力
+- 使用"救命"等青少年常用感叹词结合教科书内容讨论（新增）
 
 成人强反证包括：
 - 明确年龄大于等于 18 岁
@@ -21,6 +23,8 @@
 2. 当前输入优先于检索相似案例。
 3. 当前稳定成人信号与当前稳定未成年信号冲突时，应下调置信度并记录到 `conflicting_signals`。
 4. 不要用单一软线索直接翻盘。
+5. 讨论中学课程但无其他成人信号时，优先考虑未成年人可能性
+6. 使用青少年特色表达讨论教科书内容时，即使话题专业也应倾向未成年人判定（新增）
 
 ## Soft Signals
 
@@ -30,6 +34,7 @@
 - 节假日或开学季时间特征
 - 对父母、老师、同学的零散提及
 - 相似案例检索结果
+- 表达学习压力但未明确学段
 
 ## False Positive Guidance
 
@@ -46,3 +51,5 @@
 - 家长或老师主导的管束关系
 - 典型未成年人时间安排与依赖型行为
 - 多会话中稳定重复的学生身份线索
+- 讨论中学课程内容时的学习压力表达
+- 青少年特色表达结合教科书内容讨论（新增）
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
+++ minor-detection-v0.1.2-20260323_010444/references/schema-repair-template.md
@@ -1,4 +1,4 @@
-﻿# Schema Repair Template
+# Schema Repair Template
 
 你现在要把一个不完整或不规范的 minor-detection 输出修复为合法 JSON。
 
@@ -7,6 +7,7 @@
 - 不新增无依据的事实。
 - 所有必填字段必须补齐。
 - 只返回单个 JSON object。
+- 特别注意修复历史/地理等学科讨论中的未成年人信号遗漏问题。
 
 Schema:
 {{OUTPUT_SCHEMA}}
```

## scripts/config.py
- changed: `false`

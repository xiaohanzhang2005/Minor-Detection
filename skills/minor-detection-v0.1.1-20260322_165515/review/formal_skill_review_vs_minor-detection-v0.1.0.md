# Formal Skill Review Diff

- base_version: `minor-detection-v0.1.0`
- candidate_version: `minor-detection-v0.1.1-20260322_165515`
- generated_at: `2026-03-22T18:00:15.557767`

请人工审核以下差异，再决定 approve / reject。

## SKILL.md
- changed: `true`
```diff
--- minor-detection-v0.1.0/SKILL.md
+++ minor-detection-v0.1.1-20260322_165515/SKILL.md
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
 
@@ -36,6 +36,10 @@
 - `prior_profile`
 - `retrieved_cases`
 
+特别注意：
+- 对于历史政治类学习内容，需特别关注是否包含"中考/高考复习"等明确学生身份线索
+- 对于情绪表达强烈的对话，需检查是否包含"家长/老师管束"等依赖关系证据
+
 如果上游已经给了完整 JSON，就不要擅自改写结构，只做最小整理。
 
 ## Execution
@@ -53,3 +57,4 @@
 - 不输出 Markdown。
 - 不输出解释性前言。
 - 不补充 skill 外推理。
+- 对于历史政治类学习内容，若缺乏明确成人证据(如考研/法考)，不应仅因内容"学术性强"而判定为成人
```

## references/evidence-rules.md
- changed: `true`
```diff
--- minor-detection-v0.1.0/references/evidence-rules.md
+++ minor-detection-v0.1.1-20260322_165515/references/evidence-rules.md
@@ -1,4 +1,4 @@
-﻿# Evidence Rules
+# Evidence Rules
 
 ## Direct Evidence Priority
 
@@ -7,6 +7,7 @@
 - 明确学段为小学、初中、高中
 - 明确校园身份，如班主任、同桌、住校、年级、月考、中考、高考、作业、晚自习
 - 明确家庭依赖与未成年限制，如家长接送、被老师或家长管理
+- 历史政治类内容中明确提及"中考/高考复习"等学生身份线索
 
 成人强反证包括：
 - 明确年龄大于等于 18 岁
@@ -21,6 +22,7 @@
 2. 当前输入优先于检索相似案例。
 3. 当前稳定成人信号与当前稳定未成年信号冲突时，应下调置信度并记录到 `conflicting_signals`。
 4. 不要用单一软线索直接翻盘。
+5. 历史政治类学术内容若无明确成人证据(如考研/法考)，不应仅因内容"学术性强"判定为成人
 
 ## Soft Signals
 
@@ -30,6 +32,7 @@
 - 节假日或开学季时间特征
 - 对父母、老师、同学的零散提及
 - 相似案例检索结果
+- 情绪表达强烈的对话中缺乏家长/老师管束等依赖关系证据
 
 ## False Positive Guidance
 
@@ -46,3 +49,4 @@
 - 家长或老师主导的管束关系
 - 典型未成年人时间安排与依赖型行为
 - 多会话中稳定重复的学生身份线索
+- 历史政治类内容中的"中考/高考复习"等学生身份标记
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
+++ minor-detection-v0.1.1-20260322_165515/references/schema-repair-template.md
@@ -1,4 +1,4 @@
-﻿# Schema Repair Template
+# Schema Repair Template
 
 你现在要把一个不完整或不规范的 minor-detection 输出修复为合法 JSON。
 
@@ -7,6 +7,7 @@
 - 不新增无依据的事实。
 - 所有必填字段必须补齐。
 - 只返回单个 JSON object。
+- 对于历史政治类学习内容，若缺乏明确成人证据(如考研/法考)，不应仅因内容"学术性强"而判定为成人。
 
 Schema:
 {{OUTPUT_SCHEMA}}
```

## scripts/config.py
- changed: `false`

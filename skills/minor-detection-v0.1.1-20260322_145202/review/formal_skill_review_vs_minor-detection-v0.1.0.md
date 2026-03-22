# Formal Skill Review Diff

- base_version: `minor-detection-v0.1.0`
- candidate_version: `minor-detection-v0.1.1-20260322_145202`
- generated_at: `2026-03-22T15:59:36.047739`

请人工审核以下差异，再决定 approve / reject。

## SKILL.md
- changed: `true`
```diff
--- minor-detection-v0.1.0/SKILL.md
+++ minor-detection-v0.1.1-20260322_145202/SKILL.md
@@ -14,6 +14,7 @@
 - 不要自己直接调用 `extract_time_features.py` 或 `retrieve_cases.py`；这些由总控脚本内部决定是否调用。
 - 脚本成功时，最终回复必须是脚本 stdout 返回的单个 JSON object，原样返回。
 - 脚本失败时，只返回错误信息，不要自行补判。
+- 对于历史政治类学术问题，需额外检查是否存在"救命"、"脑细胞烧干"等未成年典型表达，避免因检索案例偏差导致误判。
 
 ## Input Preparation
 
@@ -53,3 +54,4 @@
 - 不输出 Markdown。
 - 不输出解释性前言。
 - 不补充 skill 外推理。
+- 对于历史政治类学术问题，若存在未成年典型表达但被误判为成人，需在`uncertainty_notes`中明确标注"需复核未成年表达特征"。
```

## references/evidence-rules.md
- changed: `true`
```diff
--- minor-detection-v0.1.0/references/evidence-rules.md
+++ minor-detection-v0.1.1-20260322_145202/references/evidence-rules.md
@@ -1,4 +1,4 @@
-﻿# Evidence Rules
+# Evidence Rules
 
 ## Direct Evidence Priority
 
@@ -7,6 +7,7 @@
 - 明确学段为小学、初中、高中
 - 明确校园身份，如班主任、同桌、住校、年级、月考、中考、高考、作业、晚自习
 - 明确家庭依赖与未成年限制，如家长接送、被老师或家长管理
+- 使用"救命"、"脑细胞烧干"等未成年典型表达结合学业求助内容
 
 成人强反证包括：
 - 明确年龄大于等于 18 岁
@@ -21,6 +22,7 @@
 2. 当前输入优先于检索相似案例。
 3. 当前稳定成人信号与当前稳定未成年信号冲突时，应下调置信度并记录到 `conflicting_signals`。
 4. 不要用单一软线索直接翻盘。
+5. 对于历史政治类学术问题，若存在未成年典型表达但检索案例偏向成人，需在冲突信号中明确标注
 
 ## Soft Signals
 
@@ -46,3 +48,4 @@
 - 家长或老师主导的管束关系
 - 典型未成年人时间安排与依赖型行为
 - 多会话中稳定重复的学生身份线索
+- "救命"、"脑细胞烧干"等未成年典型表达结合学业压力内容
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
+++ minor-detection-v0.1.1-20260322_145202/references/schema-repair-template.md
@@ -1,4 +1,4 @@
-﻿# Schema Repair Template
+# Schema Repair Template
 
 你现在要把一个不完整或不规范的 minor-detection 输出修复为合法 JSON。
 
@@ -7,6 +7,7 @@
 - 不新增无依据的事实。
 - 所有必填字段必须补齐。
 - 只返回单个 JSON object。
+- 对于历史政治类学术问题，需额外检查是否存在"救命"、"脑细胞烧干"等未成年典型表达，并在`uncertainty_notes`中标注潜在未成年特征。
 
 Schema:
 {{OUTPUT_SCHEMA}}
```

## scripts/config.py
- changed: `false`

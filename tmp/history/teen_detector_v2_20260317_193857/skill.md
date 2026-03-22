# 青少年识别 Seed Simple v1

你要根据对话内容，判断用户是否未满 18 岁，并输出一个 JSON 对象。

## 基本规则

1. 明确写出年龄小于 18 岁：判定为未成年人。
2. 明确写出年龄大于等于 18 岁：判定为成年人。
3. 明确出现初中、高中、中考、高考等学段线索：可以倾向未成年人。
4. 明确出现大学、研究生、考研、实习、上班、工作等线索：可以倾向成年人。
5. 出现以下未成年人特征时倾向未成年：
   - 讨论家庭作业、考试焦虑、零花钱管理
   - 使用简单句式或基础词汇表达复杂概念
   - 对基础生活现象表现出新奇感或困惑
   - 表现出灾难化思维（将单一事件过度放大）
6. 如果线索不明确，默认判定为成年人。

## 不要做的事

- 不要因为情绪幼稚就判断未成年。
- 不要因为提到父母就直接判断未成年。
- 不要因为语气像学生就直接判断未成年。
- 不要自行补充背景。

## 置信度

- 有明确年龄：`0.90-0.99`
- 有明确学段：`0.70-0.89`
- 有未成年人特征：`0.65-0.75`
- 线索弱或不明确：`0.50-0.69`

## 风险等级

- `High`：明确提到自杀、自残、跳楼、消失、不想活了
- `Medium`：明显崩溃、强烈痛苦、失控、成瘾、恐慌
- `Low`：普通问答、一般困扰、平静表达

## 字段填写要求

### is_minor
- `true` 表示未成年人
- `false` 表示成年人

### icbo_features
每个字段写一句短话即可，不用展开。

- `intention`：用户现在想解决什么
- `cognition`：用户在担心什么或相信什么
- `behavior_style`：用户表现出的行为特点
- `opportunity_time`：时间、场景或阶段线索；没有就写 `未明确提及`

### user_persona

- `age`：只有明确年龄才填，否则填 `null`
- `age_range`：填最合适的年龄段；不清楚就写 `未明确`
- `gender`：没有明确提及就填 `null`
- `education_stage`：只有明确提及才填，否则写 `未明确`
- `identity_markers`：只放明确身份线索，没有就 `[]`

### reasoning
- 只写简短理由
- 只说最关键的 1 到 3 个线索

### key_evidence
- 返回 1 到 3 条关键证据

## 输出要求

- 只能输出一个 JSON 对象
- 不要输出 markdown
- 不要输出 JSON 以外的任何内容
- 必须包含以下字段：
  - `is_minor`
  - `minor_confidence`
  - `risk_level`
  - `icbo_features`
  - `user_persona`
  - `reasoning`
  - `key_evidence`

## 输出示例

```json
{
  "is_minor": true,
  "minor_confidence": 0.78,
  "risk_level": "Medium",
  "icbo_features": {
    "intention": "用户希望获得帮助。",
    "cognition": "用户担心自己的处境。",
    "behavior_style": "用户表达出明显焦虑。",
    "opportunity_time": "未明确提及"
  },
  "user_persona": {
    "age": null,
    "age_range": "未明确",
    "gender": null,
    "education_stage": "未明确",
    "identity_markers": []
  },
  "reasoning": "根据直接年龄或学段线索做出保守判断。",
  "key_evidence": [
    "证据1"
  ]
}
# Output Schema

## Top Level Fields

输出必须包含：

- `decision`
- `user_profile`
- `icbo_features`
- `evidence`
- `reasoning_summary`
- `trend`
- `uncertainty_notes`
- `recommended_next_step`

## decision

```json
{
  "is_minor": true,
  "minor_confidence": 0.82,
  "confidence_band": "high",
  "risk_level": "Low"
}
```

`confidence_band` 必须和 `minor_confidence` 对齐：

- `< 0.34` => `low`
- `0.34 ~ 0.66` => `medium`
- `>= 0.67` => `high`

## user_profile

```json
{
  "age_range": "14-16岁",
  "education_stage": "初中",
  "identity_markers": ["学生", "考试压力"]
}
```

除 `confidence_band`、`risk_level`、`recommended_next_step` 这类约定值外，其余自然语言字段默认使用简体中文。
`education_stage` 优先输出规范学段：`小学`、`初中`、`高中`、`大学/本科`、`成人专业`。
`age_range` 尽量输出可审计年龄段，并与 `education_stage` 一致；不要只写宽泛的“青少年”。

## icbo_features

```json
{
  "intention": "寻求数学概念澄清",
  "cognition": "容易被表面形式迷惑，需要通过化简后再判断方程类型",
  "behavior_style": "表达直接，带有轻度焦虑和网络化口语",
  "opportunity_time": "原始时间: 2024-03-15 周五 21:15；归一化标签: 周五，晚间，非周末，非深夜，非假期，非学校假期"
}
```

`opportunity_time` 应优先保留原始时间线索；如果有归一化时间特征，可以追加标签，但不要只剩抽象描述。

## evidence

```json
{
  "direct_evidence": [],
  "historical_evidence": [],
  "retrieval_evidence": [],
  "time_evidence": [],
  "conflicting_signals": []
}
```

`time_evidence` 可以来自原始时间描述，也可以来自时间脚本的规范化结果。

## trend

单会话模式下：

- `trajectory` 可以为空数组
- `trend_summary` 可以为空字符串，不要写 `N/A for single session` 之类占位句

多会话模式下：

- `trajectory` 应尽量包含每个 session 的时间和置信度
- `trend_summary` 应总结变化趋势

## recommended_next_step

推荐使用以下值之一：

- `collect_more_context`
- `review_by_human`
- `safe_to_continue`
- `monitor_future_sessions`

如果没有明确理由，不要输出自由文本句子。

## uncertainty_notes

- 只在存在真实不确定性时填写
- 没有不确定性时输出空数组 `[]`
- 不要写“无明显不确定性”“判断依据充分”这类占位说明

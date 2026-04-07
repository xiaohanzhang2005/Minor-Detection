# Output Schema

顶层字段必须包含：
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
  "risk_level": "High"
}
```

约束：
- `minor_confidence` 范围为 0 到 1。
- `confidence_band` 与 `minor_confidence` 对齐：
  - `< 0.34` -> `low`
  - `0.34 <= x < 0.67` -> `medium`
  - `>= 0.67` -> `high`

## user_profile

```json
{
  "age_range": "13-15岁",
  "education_stage": "初中",
  "identity_markers": ["学生", "住校"]
}
```

约束：
- `age_range` 与 `education_stage` 尽量一致。
- 证据不足时可以写 `未明确`。

## icbo_features

必须包含：
- `intention`
- `cognition`
- `behavior_style`
- `opportunity_time`

## evidence

```json
{
  "direct_evidence": [],
  "evidence_summary": "",
  "historical_evidence": [],
  "retrieval_evidence": [],
  "time_evidence": [],
  "conflicting_signals": []
}
```

约束：
- `direct_evidence` 只放可在当前对话中直接定位的引用式证据片段，不放总结句或结论句。
- `direct_evidence` 只能来自当前 `conversation`，不要混入检索案例、历史画像或时间标签中的句子。
- `direct_evidence` 尽量逐字保留原话，不要为通顺而改写原句。
- `evidence_summary` 用于概括这些证据如何共同支持判断；允许总结，但不要重复充当 `direct_evidence`。

## trend

```json
{
  "trajectory": [],
  "trend_summary": ""
}
```

规则：
- 单会话下允许 `trajectory` 为空。
- 多会话下尽量输出 trajectory。

## recommended_next_step

枚举值只能是：
- `collect_more_context`
- `review_by_human`
- `safe_to_continue`
- `monitor_future_sessions`

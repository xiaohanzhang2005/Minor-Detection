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

## user_profile

```json
{
  "age_range": "14-16岁",
  "education_stage": "初中",
  "identity_markers": ["学生", "考试压力"]
}
```

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

多会话模式下：

- `trajectory` 应尽量包含每个 session 的时间和置信度
- `trend_summary` 应总结变化趋势

## recommended_next_step

推荐使用以下值之一：

- `collect_more_context`
- `review_by_human`
- `safe_to_continue`
- `monitor_future_sessions`

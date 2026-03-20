# 正式 Skill 输入输出契约

## 1. 文档目标

本文档定义未成年人识别正式 Skill 的输入输出契约，用于统一：

- Skill 本体设计
- 外部运行时系统接入
- 网站展示层渲染
- 评测集构造
- 自迭代优化边界

该契约的核心原则是：

- Skill 默认面向 Agent/系统消费，而不是直接面向终端用户自由回答
- Skill 必须支持单会话、多会话、外部摘要增强三种输入模式
- 输出必须稳定、可解释、可被程序消费

---

## 2. 总体设计原则

### 2.1 统一 payload

无论上层输入来自聊天窗口、文件上传、API、网站，最终都应归一化为同一种 `analysis_payload`。

### 2.2 统一 schema

Skill 输出必须是固定 JSON 结构，避免上层系统每次适配不同字段。

### 2.3 直接证据优先

当前输入中的直接年龄/学段/身份线索优先于历史摘要、RAG 和时间特征。

### 2.4 弱证据只做辅助

时间特征、RAG、历史摘要均不得单独主导结论。

### 2.5 时间证据原则

- 当前阶段优先使用输入中显式出现的时间描述作为时间证据
- 只有当输入时间信息不完整、格式不统一或缺少稳定派生字段时，才启用时间脚本
- 时间脚本的作用是规范化与补全，不是替代已有时间信息

### 2.6 趋势是数据，不是图

Skill 输出趋势轨迹数据；曲线图由网站或其他上层系统绘制。

---

## 3. 输入契约

### 3.1 顶层结构

```json
{
  "mode": "single_session | multi_session | enriched",
  "conversation": [],
  "sessions": [],
  "context": {
    "time_features": {},
    "retrieved_cases": [],
    "prior_profile": {},
    "channel": "",
    "locale": ""
  },
  "meta": {
    "user_id": "",
    "request_id": "",
    "source": ""
  }
}
```

### 3.2 顶层字段定义

#### `mode`

必填。

可取值：

- `single_session`
- `multi_session`
- `enriched`

含义：

- `single_session`：输入一个会话
- `multi_session`：输入同一用户多个会话
- `enriched`：输入当前会话，并携带外部运行时整理好的增强摘要

#### `conversation`

用于 `single_session` 和 `enriched`。

类型：

```json
[
  {
    "role": "user | assistant | system",
    "content": "..."
  }
]
```

要求：

- 至少包含一条消息
- 顺序必须为真实对话顺序
- `content` 为原始文本，不要预先写入推断结论

#### `sessions`

用于 `multi_session`。

类型：

```json
[
  {
    "session_id": "s1",
    "session_time": "2026-03-18T23:40:00+08:00",
    "conversation": []
  }
]
```

要求：

- 必须按时间升序排列
- 每个 session 必须带唯一 `session_id`
- `session_time` 推荐 ISO 8601

#### `context`

可选。

用于承载外部增强信息。

#### `meta`

可选。

用于追踪调用来源，不参与核心判定，只作为上层系统识别请求的辅助字段。

### 3.3 `context` 字段定义

#### `context.time_features`

来源：

- 宿主系统可选预填的结构化时间特征
- Skill 包内置时间脚本
- 或由 Skill 内部工作流在必要时调用同一脚本后生成

推荐结构：

```json
{
  "local_time": "2026-03-18 23:40:00",
  "timezone": "Asia/Shanghai",
  "weekday": "Wednesday",
  "is_weekend": false,
  "is_late_night": true,
  "time_bucket": "late_night",
  "holiday_label": null,
  "school_holiday_hint": false
}
```

使用规则：

- 如果输入已经包含明确时间表达，例如“2026-03-18 周三 8:59”，优先直接使用该时间信息
- 只有当输入缺少星期、时区、假期标签或稳定时间桶等字段时，才建议由 Skill 内部额外运行时间脚本
- 仅为辅助证据
- 不能单独决定 `is_minor`

默认路径：

- 宿主系统不需要预先填充 `time_features`
- 默认只需把聊天记录或 session 输入 Skill，由 Skill 自行判断是否需要调用时间脚本

#### `context.retrieved_cases`

来源：

- 内置 RAG
- 外置 RAG 服务

统一结构：

```json
[
  {
    "source": "builtin_rag | external_rag",
    "sample_id": "case_001",
    "score": 0.84,
    "label": "minor | adult | unknown",
    "summary": "与当前输入最相关的案例摘要",
    "key_signals": ["考试焦虑", "班主任", "初三"]
  }
]
```

使用规则：

- 只能作为辅助证据
- 当前对话直接证据优先
- 与当前证据冲突时优先降低置信度

#### `context.prior_profile`

来源：

- 外部运行时系统对长期画像或历史分析结果的压缩摘要

推荐结构：

```json
{
  "total_sessions": 6,
  "estimated_age_range": "14-16岁",
  "education_stage": "初中",
  "identity_markers": ["学生", "考试压力"],
  "minor_confidence_mean": 0.78,
  "recent_confidence_trend": "up",
  "summary": "过去多次会话持续出现校园与学业压力线索"
}
```

使用规则：

- 属于历史辅助信息
- 不得覆盖当前强直接证据

#### `context.channel`

可选，用于记录渠道来源。

示例：

- `web_demo`
- `api`
- `review_console`
- `chat_agent`

#### `context.locale`

可选，用于语言和地域环境提示。

示例：

- `zh-CN`
- `en-US`

### 3.4 三种输入模式的最小要求

#### 模式 A：`single_session`

最小要求：

- `mode`
- `conversation`

不强依赖：

- RAG
- `prior_profile`

#### 模式 B：`multi_session`

最小要求：

- `mode`
- `sessions`

可选增强：

- `context.time_features`
- `context.retrieved_cases`

#### 模式 C：`enriched`

最小要求：

- `mode`
- `conversation`
- 至少一个 `context` 子字段

适用场景：

- 外部系统已存在用户历史摘要
- 外部系统已做 RAG
- 外部系统可选预填 `time_features`

### 3.5 输入示例

#### 单会话输入示例

```json
{
  "mode": "single_session",
  "conversation": [
    {
      "role": "user",
      "content": "明天月考，我真的不想去学校了"
    }
  ],
  "context": {},
  "meta": {
    "request_id": "req_001",
    "source": "web_demo"
  }
}
```

#### 多会话输入示例

```json
{
  "mode": "multi_session",
  "sessions": [
    {
      "session_id": "s1",
      "session_time": "2026-03-01T20:10:00+08:00",
      "conversation": [
        {"role": "user", "content": "最近作业太多了"}
      ]
    },
    {
      "session_id": "s2",
      "session_time": "2026-03-08T22:40:00+08:00",
      "conversation": [
        {"role": "user", "content": "班主任今天又找我谈成绩"}
      ]
    }
  ],
  "context": {
    "time_features": {
      "timezone": "Asia/Shanghai"
    }
  },
  "meta": {
    "user_id": "user_001",
    "request_id": "req_002",
    "source": "api"
  }
}
```

#### 外部增强输入示例

```json
{
  "mode": "enriched",
  "conversation": [
    {
      "role": "user",
      "content": "我真的不想面对明天的考试"
    }
  ],
  "context": {
    "time_features": {
      "local_time": "2026-03-18 23:40:00",
      "weekday": "Wednesday",
      "is_late_night": true,
      "time_bucket": "late_night"
    },
    "retrieved_cases": [
      {
        "source": "external_rag",
        "sample_id": "case_087",
        "score": 0.87,
        "label": "minor",
        "summary": "初中生考试焦虑案例",
        "key_signals": ["考试", "学校", "焦虑"]
      }
    ],
    "prior_profile": {
      "total_sessions": 4,
      "estimated_age_range": "14-16岁",
      "education_stage": "初中",
      "minor_confidence_mean": 0.76,
      "recent_confidence_trend": "up",
      "summary": "近四次会话中持续出现校园与学业线索"
    }
  },
  "meta": {
    "user_id": "user_001",
    "request_id": "req_003",
    "source": "runtime_system"
  }
}
```

---

## 4. 输出契约

### 4.1 顶层结构

```json
{
  "decision": {},
  "user_profile": {},
  "icbo_features": {},
  "evidence": {},
  "reasoning_summary": "",
  "trend": {},
  "uncertainty_notes": [],
  "recommended_next_step": ""
}
```

### 4.2 顶层字段定义

#### `decision`

必填。

结构：

```json
{
  "is_minor": true,
  "minor_confidence": 0.82,
  "confidence_band": "high",
  "risk_level": "Low"
}
```

约束：

- `is_minor`：布尔值
- `minor_confidence`：0 到 1
- `confidence_band`：`low | medium | high`
- `risk_level`：`Low | Medium | High`

#### `user_profile`

必填。

结构：

```json
{
  "age_range": "14-16岁",
  "education_stage": "初中",
  "identity_markers": ["学生", "考试压力"]
}
```

约束：

- `age_range` 必须是字符串；未知时写 `未明确`
- `education_stage` 必须是字符串；未知时写 `未明确`
- `identity_markers` 必须是数组

#### `icbo_features`

必填。

结构：

```json
{
  "intention": "",
  "cognition": "",
  "behavior_style": "",
  "opportunity_time": ""
}
```

要求：

- 每个字段都必须是简短可读字符串
- 不允许空缺字段

#### `evidence`

必填。

结构：

```json
{
  "direct_evidence": [],
  "historical_evidence": [],
  "retrieval_evidence": [],
  "time_evidence": [],
  "conflicting_signals": []
}
```

定义：

- `direct_evidence`：来自当前会话或当前输入中的直接线索
- `historical_evidence`：来自同一用户历史会话的稳定线索
- `retrieval_evidence`：来自 RAG 的辅助线索
- `time_evidence`：来自原始时间描述或时间脚本规范化结果的辅助线索
- `conflicting_signals`：支持成年人或削弱未成年人判断的信号

#### `reasoning_summary`

必填。

要求：

- 一段简洁总结
- 必须体现结论如何形成
- 必须体现直接证据与辅助证据的区别

#### `trend`

必填，但可为空结构。

建议结构：

```json
{
  "trajectory": [
    {
      "session_id": "s1",
      "session_time": "2026-03-01T20:10:00+08:00",
      "minor_confidence": 0.68
    }
  ],
  "trend_summary": ""
}
```

规则：

- 单会话模式下 `trajectory` 可以为空数组
- 多会话模式下建议输出完整轨迹

#### `uncertainty_notes`

必填。

用于记录：

- 证据不足
- 证据冲突
- 历史与当前输入不一致
- 时间和 RAG 仅为弱证据等说明

#### `recommended_next_step`

必填。

用于给上层系统建议下一步动作，例如：

- `collect_more_context`
- `review_by_human`
- `safe_to_continue`
- `monitor_future_sessions`

### 4.3 输出示例

```json
{
  "decision": {
    "is_minor": true,
    "minor_confidence": 0.82,
    "confidence_band": "high",
    "risk_level": "Low"
  },
  "user_profile": {
    "age_range": "14-16岁",
    "education_stage": "初中",
    "identity_markers": ["学生", "考试压力"]
  },
  "icbo_features": {
    "intention": "表达考试压力并寻求情绪缓解",
    "cognition": "对考试结果有明显焦虑，关注短期学业后果",
    "behavior_style": "表达直接，带有明显情绪化与校园语境",
    "opportunity_time": "深夜、考试前情境"
  },
  "evidence": {
    "direct_evidence": [
      "明确提到明天考试",
      "明确提到学校场景"
    ],
    "historical_evidence": [
      "历史多次出现班主任、成绩、作业线索"
    ],
    "retrieval_evidence": [
      "相似案例多为初中生考试焦虑场景"
    ],
    "time_evidence": [
      "深夜时间段可能增强情绪表达强度"
    ],
    "conflicting_signals": []
  },
  "reasoning_summary": "当前会话中出现明确校园和考试线索，且历史会话持续指向在校学生画像；RAG 与时间特征仅作为补强，因此综合判断为较高概率未成年人。",
  "trend": {
    "trajectory": [
      {
        "session_id": "s1",
        "session_time": "2026-03-01T20:10:00+08:00",
        "minor_confidence": 0.68
      },
      {
        "session_id": "s2",
        "session_time": "2026-03-08T22:40:00+08:00",
        "minor_confidence": 0.76
      },
      {
        "session_id": "s3",
        "session_time": "2026-03-18T23:40:00+08:00",
        "minor_confidence": 0.82
      }
    ],
    "trend_summary": "随着多次会话中校园、考试和师生关系线索持续出现，未成年人概率逐步上升。"
  },
  "uncertainty_notes": [
    "时间特征和 RAG 仅为辅助证据"
  ],
  "recommended_next_step": "monitor_future_sessions"
}
```

### 4.4 输出使用规范

#### 对 Skill 本体的约束

- 只能输出一个 JSON 对象
- 不输出 Markdown 包裹
- 不输出多版本候选答案

#### 对外部运行时系统的约束

- 不要依赖自由文本解析核心字段
- 所有关键判断都从 JSON 固定字段读取
- 展示层只负责渲染，不改写原始判断字段

#### 对网站展示层的约束

- 网站优先显示 `decision`、`user_profile`、`evidence`
- 趋势图来自 `trend.trajectory`
- 风险说明来自 `decision.risk_level`
- 解释文案来自 `reasoning_summary`

---

## 5. 兼容性与回退策略

### 5.1 无增强信息时

如果没有 `time_features`、`retrieved_cases`、`prior_profile`：

- Skill 必须仍能正常工作
- 对应证据数组返回空数组
- 只要原始输入中仍有可读时间描述，Skill 仍可把它作为时间证据使用

### 5.2 单会话无历史时

- `historical_evidence` 为空
- `trend.trajectory` 可为空
- `trend.trend_summary` 可为空字符串

### 5.3 外部系统仅传摘要时

- 允许不传全量历史
- Skill 基于 `prior_profile` 与当前会话联合分析

---

## 6. 后续工作建议

本文档冻结后，后续应继续产出：

- `Skill 内外边界清单.md`
- `正式 Skill 评测设计.md`
- `description 触发评测集设计.md`

这三份文档将分别解决：

- 哪些能力属于 Skill
- 如何评测输出质量
- 如何优化 Skill 触发能力

---
name: minor-detection
description: 当用户或上层系统需要判断聊天记录中的说话者是否可能是未成年人、青少年、中学生、高中生，或需要对单会话、多会话历史做年龄倾向、校园倾向、学生画像、未成年人风险与证据分析时使用此技能。即使用户没有直接说"未成年人识别"，但需求本质上是判断"像不像未成年用户"、输出未成年人概率、画像、趋势、风险等级或结构化证据，也应激活此技能。特别关注以下未成年人特征：极端情绪表达、学业压力相关崩溃、社交挫败的灾难化思维、对同伴压力的过度敏感、对新型风险的恐慌性反应。
---

# Minor Detection

将当前任务视为"青少年识别 / 未成年人识别分析任务"。

输出要求只有一条：输出且只输出一个 JSON 对象。

## Skill Positioning

本技能用于根据中文聊天记录，判断说话者是否更像未成年人、青少年或成年人，并输出结构化证据。

本技能负责：

1. 判断是否更像未成年人。
2. 输出 `minor_confidence`、`risk_level`、`confidence_band`。
3. 输出用户画像与 ICBO 特征。
4. 在多会话输入下输出趋势与变化轨迹。
5. 输出直接证据、历史证据、检索证据、时间证据和冲突证据。

新增未成年人强特征识别：
- 极端情绪化表达（如"气到手抖"、"恨得要死"）
- 学业压力导致的崩溃性表达
- 对社交挫败的灾难化思维
- 对同伴压力的过度敏感
- 对新型风险的恐慌性反应

本技能不负责：

1. 持久化数据库。
2. 长期记忆存储。
3. 技能包外部代码调度。

本技能只可使用本技能目录内资源：

1. `scripts/extract_time_features.py`
2. `scripts/retrieve_cases.py`
3. `references/evidence-rules.md`
4. `references/icbo-guidelines.md`
5. `references/output-schema.md`
6. `assets/retrieval_assets/*`

## Accepted Input Modes

接受三种输入模式：

1. `single_session`
   当前输入只包含一个会话或一段聊天记录，核心字段通常为 `conversation`。
2. `multi_session`
   当前输入包含同一用户多个按时间顺序组织的历史会话，核心字段通常为 `sessions`。
3. `enriched`
   当前输入除当前会话外，还显式包含 `prior_profile`、`retrieved_cases`、`time_features` 等增强字段。

模式识别规则：

1. 如果存在 `sessions`，优先判为 `multi_session`。
2. 如果不存在 `sessions`，但存在明确增强字段，判为 `enriched`。
3. 如果只有单段 `conversation`，判为 `single_session`。
4. 如果输入存在歧义，选择最简单且合法的模式，并在 `uncertainty_notes` 中说明。

时间信息的可用来源包括：

1. `time_features`
2. `raw_time_hint`
3. `opportunity_time`
4. 对话文本内可解析的精确时间戳

## Fixed Execution Flow

每次调用必须按以下顺序执行，不要跳步，不要打乱：

1. 识别输入模式并整理输入。
2. 归一化时间证据。
3. 选择检索路由。
4. 执行第一阶段用户建模。
5. 提取 `icbo_features`。
6. 综合证据并执行第二阶段用户建模。
7. 做出最终青少年识别 / 未成年人识别判断。
8. 按输出契约校验并生成结果。

## Step 1: Mode Detection And Input Normalization

先确定当前模式是 `single_session`、`multi_session` 还是 `enriched`。

然后整理当前可用输入：

1. 当前对话内容。
2. 历史会话内容。
3. `raw_time_hint` 或等价时间来源。
4. `time_features`。
5. `retrieved_cases`。
6. `prior_profile`。

如果上层系统传入的是 `opportunity_time`，可将其视为 `raw_time_hint` 的原始时间来源。

## Step 2: Time Evidence Normalization

时间处理必须早于检索与最终判断。

处理规则：

1. 如果输入已经提供可用的 `time_features`，直接使用。
2. 如果没有 `time_features`，但存在可解析的精确时间信息，则按需调用 `scripts/extract_time_features.py`。
3. 如果没有可用时间信息，则在无时间增强条件下继续分析。

`scripts/extract_time_features.py` 仅在以下情况下调用：

1. 输入没有现成 `time_features`。
2. 但存在 `raw_time_hint`、`opportunity_time` 或文本中的精确时间戳。
3. 且这些信息足以补出更稳定的时间标签。

时间脚本的用途是补充如下辅助字段：

1. `weekday`
2. `is_weekend`
3. `is_late_night`
4. `time_bucket`
5. `holiday_label`
6. `school_holiday_hint`

时间证据只能作为辅助证据，不能单独决定 `is_minor`。

## Step 3: Retrieval Routing

RAG 只能在以下三种路径中三选一：

1. `external_rag`
   如果输入已带非空 `retrieved_cases`，直接消费这些结果。
2. `internal_rag`
   如果输入未带 `retrieved_cases`，但技能包内同时存在：
   `scripts/retrieve_cases.py` 与 `assets/retrieval_assets/*`
   则按需调用内置检索。
3. `no_rag`
   如果外置检索和内置检索都不可用，则在无检索证据条件下继续。

`scripts/retrieve_cases.py` 仅在以下条件下调用：

1. 输入未提供 `retrieved_cases`。
2. 内置检索脚本存在且可运行。
3. 内置检索资产存在且可用。

调用内置检索时，优先使用：

1. 当前对话内容中的 user-side 信息。
2. `raw_time_hint`
3. `time_features`

检索证据只能作为辅助证据，不得压过当前输入中的强直接证据。

## Step 4: Stage-1 User Modeling

先做第一阶段用户建模，再进入 ICBO 与综合判断。

第一阶段用户建模至少输出：

1. `age_range`
2. `education_stage`
3. `identity_markers`

特别注意以下未成年人特征：
- 极端情绪表达（如"气到手抖"、"恨得要死"）
- 学业压力导致的崩溃性表达
- 对社交挫败的灾难化思维
- 对同伴压力的过度敏感
- 对新型风险的恐慌性反应

模式要求：

1. `single_session`
   只基于当前会话抽取画像，不编造长期趋势。
2. `multi_session`
   聚合同一用户多次会话中的稳定线索，建立跨会话初步画像。
3. `enriched`
   先看当前输入，再与 `prior_profile` 对照，不盲目信任旧画像。

这里的用户建模是技能内部分析能力，不等于持久化长期记忆。

## Step 5: ICBO Extraction

必须输出 `icbo_features`：

1. `intention`
2. `cognition`
3. `behavior_style`
4. `opportunity_time`

特别注意未成年人特有的认知模式：
- 灾难化思维
- 非黑即白思维
- 过度概括
- 对社交评价的极端敏感

如果 ICBO 归类边界不清，读取：

1. `references/icbo-guidelines.md`

不要为了凑字段而编造过度具体的心理描述。

## Step 6: Evidence Synthesis And Stage-2 User Modeling

在这一阶段，综合以下信息：

1. 当前对话中的直接证据。
2. 历史会话中的稳定证据。
3. `retrieved_cases` 中的检索辅助证据。
4. `time_features` 中的时间辅助证据。
5. 第一阶段用户画像。
6. `icbo_features`。
7. `prior_profile`。

在这一阶段完成两件事：

1. 修正第一阶段用户画像。
2. 完成青少年识别 / 未成年人识别的证据综合。

证据优先级固定为：

1. 直接证据
2. 历史稳定证据
3. 检索辅助证据
4. 时间辅助证据

如果证据冲突：

1. 优先保留更强直接证据。
2. 下调 `minor_confidence`。
3. 在 `evidence.conflicting_signals` 中记录冲突来源。
4. 在 `uncertainty_notes` 中说明为何保守判断。

在做最终结论前，必须读取：

1. `references/evidence-rules.md`

## Step 7: Final Decision

在综合证据后，完成最终判断：

1. 是否更像未成年人。
2. 是否更像青少年。
3. 置信度处于什么区间。
4. 风险等级是什么。

特别注意以下未成年人强特征：
- 极端情绪表达
- 学业压力导致的崩溃性表达
- 对社交挫败的灾难化思维
- 对同伴压力的过度敏感
- 对新型风险的恐慌性反应

避免以下错误：

1. 仅因语气幼稚就直接判为未成年人。
2. 仅因提到父母就直接判为未成年人。
3. 仅因深夜聊天就直接判为未成年人。
4. 仅因检索结果偏向未成年人就直接改判。
5. 仅因存在 `prior_profile` 就忽视当前强直接证据。

## Step 8: Output Contract Validation

输出前必须读取：

1. `references/output-schema.md`

最终输出必须严格遵守输出契约，顶层必须包含：

1. `decision`
2. `user_profile`
3. `icbo_features`
4. `evidence`
5. `reasoning_summary`
6. `trend`
7. `uncertainty_notes`
8. `recommended_next_step`

## Output Hard Constraints

遵守以下硬约束：

1. 输出且只输出一个 JSON 对象。
2. 不输出解释性前言，不输出 Markdown，不输出代码块。
3. `decision.minor_confidence` 必须是 0 到 1 之间的数值。
4. `user_profile.age_range` 和 `user_profile.education_stage` 必须始终有值，不明确时写"未明确"。
5. `evidence.direct_evidence`、`historical_evidence`、`retrieval_evidence`、`time_evidence`、`conflicting_signals` 必须按语义正确归类。
6. `single_session` 下不要伪造长期趋势。
7. `multi_session` 下尽量输出 `trend.trajectory` 与 `trend.trend_summary`。
8. `enriched` 下可以使用 `prior_profile`，但不能让它覆盖当前输入中的强直接证据。

## Reference Loading Order

参考文件读取顺序如下：

1. 在综合证据前读取 `references/evidence-rules.md`
2. 在 ICBO 边界不清时读取 `references/icbo-guidelines.md`
3. 在最终输出前读取 `references/output-schema.md`

如果参考文件之间出现张力，优先级为：

1. `references/output-schema.md`
2. `references/evidence-rules.md`
3. `references/icbo-guidelines.md`

## Optimization Boundary

本技能允许后续自迭代优化，但不要破坏以下固定骨架：

1. 三种输入模式：`single_session`、`multi_session`、`enriched`
2. 固定执行顺序：模式识别 -> 时间处理 -> 检索路由 -> 第一阶段用户建模 -> ICBO -> 证据综合与第二阶段建模 -> 最终判断 -> 输出校验
3. 时间证据和检索证据都属于辅助证据
4. 输出契约必须与 `references/output-schema.md` 保持一致

允许优化的方向：

1. 触发描述更准确
2. 流程表达更清晰
3. 用户建模口径更稳
4. 证据冲突处理更细
5. 边界样本规则更强

不要把所有细节规则都塞回主文件；稳定启发式优先沉淀到 `references/evidence-rules.md`。
# `minor-detection` 脚本主导化与自迭代重构最终方案

## Summary
把当前 `minor-detection` skill 重构成“**Agent 只发起，skill 内总控脚本完成全流程**”的架构。`SKILL.md` 不再承载细粒度 SOP 推理，而只负责触发、输入约束、调用总控脚本、原样返回 JSON。  
真实 Agent 主链路默认使用 **`execution_mode=bypass`**，因为已实测只有 bypass 模式下，skill 内脚本才能稳定外调第三方 API；sandbox 路径保留为显式调试选项，但不作为默认主路径。  
旧本地自迭代链路继续保留，作为速度、成本、效果基线，不与 skill 内执行链混写。

---

## 1. 总体架构与边界

### 1.1 Skill 内部负责什么
skill 内部只负责“**一次识别如何完成**”，目录保持为三类：

- `assets/`
  - 检索索引、语料、manifest 等静态资产
- `references/`
  - 所有可被优化器编辑的 Markdown 规则资产
- `scripts/`
  - 所有稳定执行代码、总控脚本、配置、schema repair、API 调用实现

### 1.2 Skill 外部自迭代链条负责什么
skill 外部继续负责：

- 数据集采样
- runner
- 真实 Agent 调度
- judge
- failure/protected packet 生成
- optimizer
- compare / rollback / promote
- review artifact
- manual final test
- 版本推进与报告

### 1.3 Agent 的职责
Agent 在新架构下只负责：

- 读取 `SKILL.md`
- 将当前输入整理成统一 payload
- 调用 `scripts/run_minor_detection_pipeline.py`
- 如果脚本成功返回合法 JSON，则**原样输出**
- 不允许 Agent 自己重新进行青少年识别推理

---

## 2. Skill 目录重构方案

### 2.1 `SKILL.md` 的最终定位
`SKILL.md` 收缩为“调用契约”，只保留：

- skill 触发描述
- 输入整理要求
- 必须调用总控脚本的硬约束
- 脚本成功返回 JSON 后必须原样输出的硬约束
- 脚本失败时返回脚本错误，不允许 Agent 自行补判

不再在 `SKILL.md` 中展开“模式识别 -> 时间处理 -> RAG -> 建模 -> 判别 -> 输出契约”的执行细节，这些全部下沉到 skill 内代码和 `references/*.md`。

### 2.2 `references/` 中每个文件的职责
`references/` 统一承载“规则资产”，供总控脚本读取，也供优化器编辑。

- `references/evidence-rules.md`
  - 成年/未成年边界规则
  - 成人/青少年冲突证据的优先级
  - false positive / false negative 调优主目标
- `references/icbo-guidelines.md`
  - `icbo_features` 的抽取规范与字段解释
  - 不作为主要优化目标，默认只读
- `references/output-schema.md`
  - 最终 JSON 契约说明
  - 字段、枚举、置信度区间、输出硬约束
  - 默认只读
- `references/classifier-system.md`
  - classifier API 的 system prompt
  - 规定模型如何综合 conversation、time、retrieval、profile、evidence-rules 做判别
- `references/classifier-user-template.md`
  - classifier API 的 user prompt 模板
  - 规定脚本如何把结构化证据包拼成请求
- `references/retrieval-query-template.md`
  - retrieval query 的构造模板
  - 决定如何把 conversation、时间特征、身份线索拼成 retrieval 查询
- `references/schema-repair-template.md`
  - 当 classifier 返回结果缺字段或不合法时，用于 schema repair 的模板
  - 主要服务 `schema_invalid / fields_missing / output_parse_failure`

### 2.3 `scripts/` 中需要实现的脚本与功能
`skills/minor-detection-v0.1.0/scripts/` 需要包含：

- `run_minor_detection_pipeline.py`
  - skill 内唯一主入口
  - 执行完整识别流程
  - 输出最终 JSON
- `extract_time_features.py`
  - 输入原始时间线索
  - 输出结构化时间特征
- `retrieve_cases.py`
  - 负责 retrieval
  - 主路径远程 dense embedding
  - 失败时 lexical fallback
- `config.py`
  - 统一管理第三方 API 和流程配置
- 内部 helper 模块
  - `_classifier_client.py` 或等价实现
  - `_schema_repair.py` 或等价实现
  - `_payload_normalizer.py` 或等价实现
  - `_profile_merge.py` 或等价实现
  - 这些 helper 不作为优化器编辑目标

---

## 3. 总控脚本的输入、输出与完整流程

### 3.1 输入接口
总控脚本输入为单个 JSON payload。

顶层字段：

- 必填
  - `mode`
  - `conversation`
  - `request_id`
  - `sample_id`
  - `context`
- 可选
  - `user_id`

`context` 中允许的可选字段：

- `raw_time_hint`
- `opportunity_time`
- `time_features`
- `prior_profile`
- `retrieved_cases`

时间不是强制顶层字段，而是 **context 内可选字段**。  
时间处理规则是“有则用、无则补提取、再无则无时间增强继续”。

### 3.2 输出接口
总控脚本直接输出最终正式 JSON，不再依赖 Agent 组装。输出必须满足现有 schema，包括：

- `decision`
- `user_profile`
- `icbo_features`
- `evidence`
- `reasoning_summary`
- `trend`
- `uncertainty_notes`
- `recommended_next_step`

### 3.3 固定主流程
总控脚本按以下固定顺序执行，不允许由 Agent 改流程：

1. 输入 normalize
2. 模式识别
3. 时间线索抽取与时间增强
4. retrieval query 组装
5. dense retrieval
6. retrieval fallback
7. 证据包组装
8. classifier API 一次调用
9. 确定性 profile/trend merge
10. schema repair
11. 最终 JSON 输出

### 3.4 各阶段具体职责
#### 1. 输入 normalize
- 把 Agent 传入的 payload 统一成内部单一结构
- 修正字段别名、空值、模式不标准输入
- 不做判别

#### 2. 模式识别
- 识别 `single_session / multi_session / enriched`
- 输出统一内部表示
- v1 不在 Agent 侧做复杂模式推理

#### 3. 时间线索抽取与时间增强
- 如果已有合法 `context.time_features`，直接使用
- 否则优先看 `context.raw_time_hint` / `context.opportunity_time`
- 若仍缺失，则从 `conversation` 中自动抽取时间线索
- 只有抽到可解析时间时才调 `extract_time_features.py`
- 若没有可用时间线索，则继续走无时间增强路径

#### 4. retrieval query 组装
- 读取 `references/retrieval-query-template.md`
- 组合 conversation、time features、显性身份线索
- 生成 retrieval query

#### 5. dense retrieval
- 调 `retrieve_cases.py`
- `retrieve_cases.py` 主路径调用 embedding API
- 结果返回 `mode=embedding`

#### 6. retrieval fallback
- dense embedding 失败时自动 lexical fallback
- 结果返回 `mode=fallback:*`
- fallback 不中断总流程

#### 7. 证据包组装
脚本把以下信息整理给 classifier API：

- conversation
- raw_time_hint / opportunity_time / time_features
- retrieved_cases
- prior_profile
- 明显身份线索
- evidence-rules
- request metadata

这里不是“先做完整用户建模”，这里只做**证据预处理与证据包拼装**。

#### 8. classifier API 一次调用
- 读取：
  - `references/classifier-system.md`
  - `references/classifier-user-template.md`
  - `references/evidence-rules.md`
- 第三方 API 一次性输出：
  - `decision`
  - `user_profile`
  - `icbo_features`
  - `evidence`
  - `reasoning_summary`
  - `uncertainty_notes`
  - `recommended_next_step`
  - 可选 `trend` 初稿
- classifier API 是真正完成“建模 + 判别 + 证据综合”的阶段

#### 9. 确定性 profile/trend merge
- 不再调第二次模型
- 只做脚本内的确定性 merge：
  - 补齐 `trend.trajectory`
  - 合并 `prior_profile` 和当前 `user_profile`
  - 规范化 `identity_markers`
  - 保持字段稳定

#### 10. schema repair
- 若 classifier 返回字段缺失或值不合法
- 读取 `references/output-schema.md` 与 `references/schema-repair-template.md`
- 执行 repair
- repair 后再作为最终输出

#### 11. 最终 JSON 输出
- 输出单个 JSON object
- 不输出 Markdown
- 不输出解释性前言
- 直接供 Agent 原样返回

### 3.5 外部 API 调用点
skill 内只有两类第三方 API 外调：

- embedding API
  - 由 `retrieve_cases.py` 调用
  - 用于 dense retrieval
- classifier API
  - 由 `run_minor_detection_pipeline.py` 调用
  - 用于建模 + 判别 + 证据综合

`extract_time_features.py` 不外调第三方 API。

---

## 4. Fallback、异常与配置

### 4.1 受控 fallback
v1 只保留这些 fallback：

- 时间缺失 -> 自动抽取 -> 抽不到则无时间增强继续
- dense retrieval 失败 -> lexical fallback
- schema 不合法 -> schema repair

### 4.2 不做的 fallback
v1 明确不做：

- classifier API 失败时回退给 Agent
- classifier API 失败时切备用模型
- classifier API 失败时改走规则模板判别
- 用户画像持久化写入

### 4.3 classifier API 失败策略
- classifier API 超时或失败时，**直接报错**
- 由外部 runner / judge 记录该次失败
- 不在 skill 内做静默兜底
- 原因：你已明确接受“第三方 API 不出问题”的前提，且希望边界干净

### 4.4 `scripts/config.py`
`config.py` 统一管理：

- `CLASSIFIER_BASE_URL`
- `CLASSIFIER_API_KEY`
- `CLASSIFIER_MODEL`
- `CLASSIFIER_TIMEOUT_SEC`
- `EMBEDDING_BASE_URL`
- `EMBEDDING_API_KEY`
- `EMBEDDING_MODEL`
- `RETRIEVAL_TOP_K`
- 决策阈值
- fallback 开关

优先级：

- 环境变量
- skill 内默认值

---

## 5. 真实 Agent 执行模式

### 5.1 默认执行模式
真实 Agent 主链路默认：

- `execution_mode = bypass`

原因：

- 普通 Python 直连 embedding API 成功
- sandbox 模式下 skill 内脚本外调 API 失败
- bypass 模式下已验证 retrieval 可恢复为 `mode=embedding`

### 5.2 runner 规则
runner 保留两种模式：

- `execution_mode=sandbox`
- `execution_mode=bypass`

规则：

- `sandbox_mode` 仅在 `execution_mode=sandbox` 时生效
- bypass 时使用 `--dangerously-bypass-approvals-and-sandbox`
- sandbox 仅保留为调试路径，不作为默认主路径

### 5.3 manifest 与 metadata
run manifest / metadata / observability 必须持续记录：

- `execution_mode`
- `sandbox_mode`
- `retrieval_mode`
- `used_fallback`
- embedding runtime snapshot
- classifier runtime snapshot
- shell quoting failure
- command failure

### 5.4 当前已知问题
当前 bypass 路径已打通 API，但仍有 PowerShell quoting 问题。  
正式实施时必须把 `retrieve_cases.py --time-features-json ...` 的传参方式修稳，不能依赖“先失败一次再重试成功”的脏行为。

---

## 6. Stratified Sampling、Judge、max_errors、Compare

### 6.1 分层取样
真实 Agent 链路继续沿用现有实现：

- 当 `max_samples < total_samples` 时，默认 `sample_strategy="stratified"`
- strata key 为：
  - `source::age_bucket`

`age_bucket` 规则保持不变：

- minor:
  - `minor_07_12`
  - `minor_13_15`
  - `minor_16_17`
- adult:
  - `adult_18_22`
  - `adult_23_26`
  - `adult_27_plus`
- 无年龄：
  - `minor_unknown`
  - `adult_unknown`

分配策略保持现有实现：

- 若预算足够，先保证每个 strata 至少 1 个
- 剩余样本按容量比例分配
- remainder 按最大余数继续补齐
- sampling 信息必须写入 manifest

### 6.2 Judge 脚本职责
judge 继续负责：

- 解析 agent output
- 校验 schema
- 检查脚本使用情况
- 统计 retrieval mode 与 observed issues
- 生成 failure packets
- 生成 protected packets
- 输出 judge report

现有 failure types 保持：

- `output_parse_failure`
- `schema_invalid`
- `fields_missing`
- `missing_time_handling`
- `missing_script_usage`
- `false_positive`
- `false_negative`
- `step_compliance_failure`

### 6.3 `max_errors` 默认公式
保持当前精确公式：

`min(total_errors, min(36, max(12, ceil(eval_size * 0.12))))`

规则：

- 若 `total_errors <= 0`，则 `max_errors = 0`
- 若用户显式传入 `max_errors`，则仍裁剪为：
  - `max(1, min(total_errors, user_max_errors))`

### 6.4 failure packet 选择
failure packet 的选择策略保持不变：

- 按 failure type 分组
- 组内按 `abs(confidence - 0.5)` 和 `sample_id` 排序
- 分组轮转抽样直到达到 `max_errors`

### 6.5 protected packet
protected packet 继续从正确样本中选出固定数量，作为 compare 的非回归门禁输入。

### 6.6 Compare / Promote / Rollback
compare 逻辑保持现有精确门禁：

candidate 只有在以下 gate **全部通过** 时才 `promote`：

- `f1_improved`
- `schema_non_regression`
- `invocation_non_regression`
- `step_compliance_non_regression`
- `protected_non_regression`

否则一律 `rollback`。

`protected_non_regression` 的定义保持不变：

- baseline 的 protected sample id 与 candidate 的 error sample id 不能有交集

### 6.7 Loop 行为
loop 行为保持现有逻辑：

- baseline 先评估
- candidate 生成后评估
- compare 决定 promote / rollback
- 若 judge 无错误可优化，则直接返回：
  - `no errors to optimize on current eval slice`

---

## 7. 优化器重构

### 7.1 优化器的新目标
优化器从“改 `SKILL.md` 为主”切换为“改规则资产为主”。

v1 可编辑目标固定为：

- `SKILL.md`
- `references/evidence-rules.md`
- `references/classifier-system.md`
- `references/classifier-user-template.md`
- `references/retrieval-query-template.md`
- `references/schema-repair-template.md`
- `scripts/config.py` 中可安全参数化的配置段

### 7.2 v1 冻结内容
v1 不允许优化器编辑：

- `scripts/run_minor_detection_pipeline.py`
- `scripts/extract_time_features.py`
- `scripts/retrieve_cases.py`
- 其他 Python 控制流实现

理由：

- 保证流程稳定
- 保持代码简洁干净
- 防止优化器把执行链越改越脏

### 7.3 judge -> optimizer 路由
编辑目标路由写死为：

- `missing_script_usage` -> `SKILL.md`
- `false_positive / false_negative` -> `references/evidence-rules.md` + classifier templates + 阈值配置
- retrieval 质量问题 -> `references/retrieval-query-template.md` + retrieval 配置
- `schema_invalid / fields_missing / output_parse_failure` -> `references/schema-repair-template.md` + schema/output 配置
- 时间相关缺失 -> `SKILL.md` 或时间相关配置
- `icbo` 抽取问题 -> `references/icbo-guidelines.md` 默认只读，v1 不纳入自动编辑，先靠 classifier template 与 evidence-rules 解决

### 7.4 优化器输入
优化器继续接收：

- judge report
- failure packets
- protected packets
- sampling 信息
- 当前 skill 版本快照

不改变外部优化拓扑，只改变 edit targets 与 prompt 素材来源。

---

## 8. 人工审核与最终交付

### 8.1 人工审核阶段
人工审核保留，并明确为正式流程的一部分。

规则：

- 只要最终版本不同于 baseline，就生成 review artifact
- `manual_review_required = true` 继续保持

### 8.2 review artifact
人工审核产物继续由优化器生成：

- diff markdown
- review summary

内容至少包含：

- base version
- candidate version
- 各可编辑文件的 diff
- review diff path
- review summary path

### 8.3 人工最终测试
保留手动最终测试脚本：

- `scripts/run_final_test.py`

它负责：

- 在选定版本上跑 test split
- 走真实 Agent 执行链
- 生成 judge report
- 供人工最终验收

### 8.4 上线前要求
正式上线前至少满足：

- bypass 模式下 retrieval 主路径可稳定达到 `embedding`
- quoting 问题修复完成
- step compliance 保持 1.0
- script usage 保持 1.0
- 最终 JSON 可稳定通过 schema 校验
- 旧本地链路仍可作为对照链跑通

---

## 9. Tests and Acceptance Criteria

### 9.1 skill 内单测
- normalize 输入
- 时间提取按需调用
- retrieval embedding 成功
- retrieval lexical fallback 成功
- classifier API 成功
- classifier API 失败时报错
- schema repair 成功
- profile/trend merge 稳定

### 9.2 skill 集成测试
- Agent 按 `SKILL.md` 调总控脚本
- Agent 对成功 JSON 原样返回
- bypass 模式下 retrieval mode 为 `embedding`
- bypass 模式下 quoting 不再出现第一次失败
- script usage 检测保持通过

### 9.3 自迭代测试
- stratified sampling 保持现有行为
- `max_errors` 公式保持不变
- failure packet 选择策略保持不变
- optimizer 只改规则资产，不改 Python
- compare gate 行为不变

### 9.4 回归测试
- 旧本地无-agent链路仍可跑
- 旧 judge / compare / rollback 逻辑不破坏
- runner 的 sandbox/bypass 双模式都可运行

---

## Assumptions
- skill 目录结构固定为 `assets/ + references/ + scripts/`，不新增 `prompts/`
- 真实 Agent 主链路默认使用 `execution_mode=bypass`
- classifier API 失败时 v1 直接报错
- 用户二次建模更新只做脚本内确定性 merge，不做持久化
- 优化器 v1 只改 Markdown 与配置资产，不改 Python 流程代码
- 旧本地自迭代链路继续保留，作为独立基线与对照链

# 正式 Skill 交付前验收规范

## 1. 目的

本规范用于统一 `minor-detection` formal skill 的交付标准，明确以下问题：

- 什么叫“Skill 本体已完成”
- 什么叫“可接入真实 Agent”
- 什么叫“完成真实 Agent 接入验收”
- 当前自迭代链路的价值和边界是什么

本规范的目标不是替代研发流程，而是给研发、评测、验收、对外汇报提供同一套口径。

---

## 2. 核心结论

正式 Skill 的交付验收分为三层，不能混为一谈：

1. Skill 本体验收
2. 真实 Agent 触发验收
3. 真实 Agent 执行验收

当前项目已经较完整覆盖了第 1 层，并为第 2、3 层做了接口和资料准备，但还没有完成真实 Agent 侧的端到端验收。

因此，当前阶段可以说：

- 我们正在做“正式 Skill 的自迭代”
- 我们可以交付“经过本地 runtime 验证的 formal skill 包”
- 我们暂时还不能宣称“真实 Agent 端到端接入已验收完成”

---

## 3. 交付对象定义

正式交付物不是单个 prompt，而是一个完整的 Skill bundle，至少包括：

- `SKILL.md`
- `references/*`
- `scripts/*`
- `assets/*`
- 输入输出契约文档
- 配套评测与验收记录

对本项目而言，交付对象默认指向：

- `skills/minor-detection/`

迭代候选版本默认位于：

- `skills/iterations/minor-detection/<candidate-version>/`

---

## 4. 三层验收模型

### 4.1 第一层：Skill 本体验收

回答的问题是：

- 这个 formal skill 包本身是否完整、自洽、可运行

重点验证：

- `SKILL.md` 的执行流程是否清晰
- 输入 payload 是否稳定
- 输出 schema 是否稳定
- 引用的 `references`、`scripts`、`assets` 是否可用
- 本地 runtime 是否能正确补齐时间特征、检索结果、fallback
- 在固定 benchmark 下效果是否达标

这一层允许使用本项目现有的本地模拟链路完成，例如：

- `src/models.py`
- `src/executor/executor.py`
- `src/runtime/skill_runtime_adapter.py`
- `src/evolution/evaluator.py`
- `scripts/run_acceptance_suite.py`
- `scripts/run_pipeline.py`

这一层的本质是：

- 验证“如果 Agent 已经正确选中了这个 skill，并把标准化输入交给它，它会表现成什么样”

### 4.2 第二层：真实 Agent 触发验收

回答的问题是：

- 真实 Agent 在实际用户请求下，会不会真的选中这个 skill

重点验证：

- should-trigger 样本是否能触发 skill
- should-not-trigger 样本是否不会误触发
- 与其他 skill 共存时，是否会被抢触发
- skill 的 `name` 和 `description` 是否足够可触发

这一层不能只靠本地 adapter 验证，因为 adapter 默认是“已经进入 skill 之后”的链路。

这一层应尽量在真实目标环境中执行，例如：

- 安装打包后的 `.skill`
- 使用真实 Agent / 宿主环境跑触发样本
- 记录 trigger rate、误触发率、漏触发率

### 4.3 第三层：真实 Agent 执行验收

回答的问题是：

- 真实 Agent 在触发 skill 后，是否真的按 skill 指南执行，并产出预期结果

重点验证：

- Agent 是否真的读取 `SKILL.md`
- 是否按需读取 `references`
- 是否真的调用 Skill 内脚本
- 是否遵守输出 schema
- 与 baseline 相比是否更优
- token、时延、失败率是否可接受

这一层是完整意义上的“真实 Agent 接入验收”。

---

## 5. 当前项目对应关系

### 5.1 当前已覆盖能力

当前项目已经较强覆盖第一层：

- formal payload 建模
- formal output 建模
- runtime adapter 自动补时间特征
- runtime adapter 自动补 retrieval cases
- 评测器直接走 formal runtime
- acceptance suite 可做多 seed、多 gate 验证
- optimizer 可持续迭代 formal skill 版本

因此当前自迭代并不是“无意义模拟”，而是在做：

- formal skill 本体的持续优化
- formal skill runtime contract 的持续优化

### 5.2 当前尚未完成能力

当前项目尚未真正完成第二、三层中的真实 Agent 部分：

- 没有直接验证真实 Agent 是否会触发该 skill
- 没有直接验证真实 Agent 是否会严格按 `SKILL.md` 执行
- 没有直接验证真实 Agent 的工具/上下文编排与本地 runtime 是否一致

---

## 6. 交付前强制验收项

### 6.1 P0：Skill 本体必须通过

以下项目必须全部通过，才允许进入“可交付候选”：

1. Skill bundle 结构完整
2. `SKILL.md` 可读且包含明确执行流程
3. 输出契约与 `src/models.py` 一致
4. 本地 runtime 可成功跑通 single-session、multi-session、enriched 至少一种主路径
5. acceptance suite 无 fail gate
6. 主 benchmark 指标达到当前阶段设定阈值
7. 关键 fallback 行为可解释、可追踪

建议验收证据：

- `test/test_formal_skill_runtime.py`
- `reports/acceptance_suite_*.json`
- `reports/pipeline_runs/pipeline_run_*.json`

### 6.2 P1：真实 Agent 触发必须通过

以下项目全部通过，才允许宣称“可被真实 Agent 正常接入”：

1. 准备一组 should-trigger 样本
2. 准备一组 should-not-trigger 样本
3. 在真实 Agent 环境安装 skill 包
4. 对每个样本记录是否触发 skill
5. 统计触发 precision、recall
6. 保存失败样本与原因分析

最低建议标准：

- should-trigger recall 不低于 80%
- should-not-trigger precision 不低于 90%

阈值可按阶段调整，但必须事先写明。

### 6.3 P2：真实 Agent 执行必须通过

以下项目全部通过，才允许宣称“真实 Agent 端到端验收通过”：

1. 真实 Agent 对 should-trigger 样本成功进入 skill
2. 输出结构满足 schema
3. 关键脚本/引用资源确实被用到
4. 与 baseline 相比，结果质量有明确提升或至少不退化
5. 执行时延、token、失败率可接受
6. 关键失败案例有归因

建议采用双跑法：

1. 同一组样本跑本地 runtime
2. 同一组样本跑真实 Agent
3. 比较两边差异

差异必须归类到以下四种之一：

- 触发差异
- 前处理差异
- 执行差异
- 输出差异

---

## 7. 推荐验收流程

### 阶段 A：本地候选版本验收

1. 运行 formal runtime 相关测试
2. 跑 acceptance suite
3. 跑 pipeline baseline + evolution + final review
4. 生成候选版本
5. 完成人工 review

通过标准：

- 第一层全部通过

产出物：

- 候选 skill 目录
- review diff
- acceptance 报告
- pipeline 报告

### 阶段 B：打包与真实 Agent 触发验收

1. 将候选版本打包为 `.skill`
2. 安装到真实 Agent 环境
3. 跑 trigger eval 数据集
4. 输出触发评测报告

通过标准：

- 第二层全部通过

产出物：

- `.skill` 包
- trigger eval 报告
- 失败样本归因表

### 阶段 C：真实 Agent 执行验收

1. 选取正式 acceptance prompts
2. 用真实 Agent 执行 with-skill
3. 同一批样本跑 baseline
4. 做 grading
5. 做 benchmark 聚合
6. 做人工 review
7. 形成最终接入验收结论

通过标准：

- 第三层全部通过

产出物：

- execution benchmark
- 人工 review 记录
- 最终接入验收结论

---

## 8. 与 Claude skill-creator 的关系

`claude-skill-creator` 的强项在于覆盖第二层和第三层，尤其是：

- 真实 skill 使用场景下的 with-skill / without-skill 对比
- grading
- benchmark 聚合
- 人工 review viewer
- description trigger optimization
- `.skill` 打包

它的典型流程是：

1. 写 skill
2. 设计 eval prompts
3. 跑 with-skill 和 baseline
4. grading
5. benchmark 聚合
6. 人工 review
7. 改 skill
8. 重复
9. 打包 skill

因此可以这样理解两者关系：

- 本项目当前链路更像 formal skill 的本地 runtime harness
- `claude-skill-creator` 更像面向真实 Claude skill 生态的创建、评测、打包、触发优化流水线

两者不是冲突关系，而是上下游关系。

推荐组合方式：

- 用本项目链路做快速内循环
- 用真实 Agent + skill-creator 风格流程做外循环验收

---

## 9. 对“自迭代”的统一解释

### 9.1 当前阶段建议口径

如果老板说“做自迭代”，当前阶段最合理的解释是：

- 让正式 Skill 本体可以被持续评测、持续优化、持续出候选版本

这主要指向：

- 正文规则优化
- 输出结构稳定性优化
- retrieval / time / evidence 相关规则优化
- trigger description 优化

不必默认理解成：

- 已经必须把真实 Agent 编排系统全部搭建完

### 9.2 为什么这么判断

从现有规划文档看，项目目标本来就被拆成三块：

1. 正式 Skill 包
2. 调用 Skill 的外部运行时系统
3. 保留并升级当前自迭代研发链路

这说明“自迭代”在项目语义里首先是一个独立研发链路，用来服务正式 Skill 的持续优化，而不是与真实 Agent 编排完全等价。

### 9.3 但长期定义应该升级

进入正式交付阶段后，“自迭代”不应只停留在本地 runtime 层。

长期建议定义为两段式：

1. 内循环自迭代
   目标：快速优化 formal skill 本体
   环境：本地 runtime / evaluator / optimizer

2. 外循环自迭代
   目标：验证真实 Agent 触发与执行效果
   环境：真实 Agent + trigger eval + execution benchmark

如果只有第一段，没有第二段，则只能说：

- 我们具备 Skill 本体自迭代能力

不能完全说：

- 我们已经具备真实 Agent 层面的完整自迭代能力

---

## 10. 对内统一说法

建议团队统一使用以下表述：

### 说法 A：当前研发状态

我们当前已经实现正式 Skill 的本地自迭代链路，能够对 Skill 本体、规则和输出契约进行持续评测和优化。

### 说法 B：当前可交付状态

我们当前可以交付经过本地 runtime 验证的 formal skill 包。

### 说法 C：后续补齐事项

在此基础上，还需要补真实 Agent 触发验收和真实 Agent 执行验收，完成端到端接入闭环。

---

## 11. 最终判定标准

### 可交付 formal skill

满足以下条件即可：

- 第一层通过
- 候选版本完成 review
- Skill bundle 可打包

### 可交付给真实 Agent 接入方

满足以下条件即可：

- 第一层通过
- 第二层通过
- 有接入说明和已知边界

### 可宣称真实 Agent 端到端验收通过

必须同时满足：

- 第一层通过
- 第二层通过
- 第三层通过

---

## 12. 当前建议

当前项目建议继续沿用以下策略：

1. 保持现有 formal skill 自迭代主线不变
2. 将 acceptance suite 和 pipeline 作为第一层强制门禁
3. 新增 trigger eval 数据集与真实 Agent 安装测试
4. 新增真实 Agent execution benchmark
5. 将最终对外口径从“最佳 prompt”统一升级为“正式 Skill 包 + 运行时契约 + 验收报告”

这样既不会否定当前工作的价值，也能自然过渡到真实 Agent 接入阶段。

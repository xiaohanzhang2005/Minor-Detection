# Skill 版本推进 SOP

## 1. 目标

本 SOP 用于说明 formal skill 从“当前研发态”走到“可交付、可被 Agent 使用”的推荐路径。

核心原则只有三条：

1. 迭代对象始终是 Skill 源码目录，不是 `.skill` 文件
2. `.skill` 是候选安装包，是验证和发布产物
3. 自迭代分内循环和外循环，不能只做一边

---

## 2. 一句话流程

源码目录 -> 本地自迭代 -> 候选 `.skill` -> Agent 外循环验证 -> 回写源码 -> 重新打包 -> description 优化 -> 发布

---

## 3. 版本阶段

### M1：Skill 本体打底

目标：

- 把 formal skill 从“研究 prompt”整理成正式 Skill bundle

本阶段主要工作：

- 固定 `SKILL.md`
- 固定 `references/*`
- 固定 `scripts/*`
- 固定 `assets/*`
- 固定输入输出 schema
- 跑通本地 runtime 主路径

通过标准：

- Skill 目录结构完整
- 本地 payload 可稳定进入 executor
- 输出结构基本稳定
- 关键脚本和资源都能工作

本阶段产物：

- `skills/minor-detection/` 主版本
- formal runtime 测试

注意：

- 这一阶段先不要急着优化 trigger description
- 先让 skill“会做事”，再让它“容易被触发”

### M2：内循环自迭代

目标：

- 用本地评测链路持续优化 Skill 本体

本阶段主要工作：

- 跑 evaluator
- 跑 acceptance suite
- 跑 pipeline
- 生成候选迭代版本
- 做 review 和人工筛选
- 继续改 `SKILL.md` 和规则

优化对象包括：

- 推理规则
- evidence 规则
- retrieval/time 使用规则
- 输出稳定性
- fallback 质量

通过标准：

- acceptance 无 fail gate
- 主 benchmark 不退化
- 候选版本有 review 记录

本阶段产物：

- `skills/iterations/minor-detection/<candidate-version>/`
- benchmark / acceptance / review 报告

注意：

- 这个阶段的核心是“把 skill 本体做稳定”
- 还不是最终的真实 Agent 接入验收

### M3：候选 `.skill` 打包

目标：

- 形成可安装到真实 Agent 环境中的候选包

本阶段主要工作：

- 从当前候选源码目录打包 `.skill`
- 记录版本号、来源版本、打包时间

通过标准：

- `.skill` 可成功打包
- 包内目录结构正确
- 包内容与源码目录一致

本阶段产物：

- `candidate.skill`

注意：

- 打包不意味着冻结开发
- 如果后续 Agent 验证发现问题，应该回源码目录修改，再重新打包

### M4：真实 Agent 外循环

目标：

- 验证真实 Agent 是否会触发 skill，以及触发后是否按预期执行

本阶段拆成两部分：

#### M4-A：触发验证

要回答：

- 真实 Agent 会不会选中这个 skill

主要工作：

- 准备 should-trigger 样本
- 准备 should-not-trigger 样本
- 安装候选 `.skill`
- 在真实 Agent 环境中跑 trigger eval
- 统计触发率、误触发率、漏触发率

发现的问题通常回写到：

- frontmatter 的 `description`
- `name`
- skill 顶层定位语句

#### M4-B：执行验证

要回答：

- 真实 Agent 触发 skill 后，执行效果是否达标

主要工作：

- 跑 with-skill
- 跑 baseline
- grading
- benchmark 聚合
- 人工 review

发现的问题通常回写到：

- `SKILL.md`
- `references/*`
- `scripts/*`

注意：

- 这一阶段不要直接改安装中的 `.skill`
- 一定是回源码目录改，再重新打包、重新安装、重新测

### M5：description 专项优化

目标：

- 在 skill 本体已经稳定后，专门优化触发质量

主要工作：

- 跑 trigger eval
- 迭代 `description`
- 比较 should-trigger / should-not-trigger 指标

适合放在这个阶段的原因：

- 这时 skill 本体已经基本稳定
- 不容易出现“很会触发，但触发后做不好”的假繁荣

注意：

- description 优化不应替代 skill 本体优化
- 触发变好但执行变差，不算成功

### M6：发布版本

目标：

- 形成可对外或可正式接入的稳定版本

发布前至少应满足：

- 本地内循环通过
- 候选 `.skill` 已验证
- 真实 Agent 触发通过
- 真实 Agent 执行通过
- 有已知边界说明

发布产物：

- 正式 `.skill`
- 版本说明
- 验收报告
- 接入说明

---

## 4. 每轮迭代的标准动作

每一轮都建议按这个顺序：

1. 修改源码目录
2. 本地跑关键样本
3. 跑正式评测
4. 生成候选版本
5. 如需 Agent 验证，则打包 `.skill`
6. 安装到真实 Agent 环境
7. 记录触发和执行结果
8. 把问题回写到源码目录
9. 进入下一轮

---

## 5. 哪些问题在哪一层解决

### 更像内循环问题

- schema 不稳定
- reasoning 不稳定
- evidence 组织混乱
- time / retrieval 规则不清
- fallback 太差
- benchmark 指标不稳定

这些问题优先在本地自迭代解决。

### 更像外循环问题

- Agent 不触发
- Agent 误触发
- Agent 没按 skill 指引做
- Agent 没用 skill 脚本
- 真实环境效果和本地差很多

这些问题必须进入真实 Agent 外循环解决。

---

## 6. 当前项目推荐做法

对当前 `minor-detection` 项目，推荐节奏是：

1. 继续沿当前 formal runtime 思路推进
2. 把 `minor-detection` 做到“可打包候选版”
3. 尽快打出第一版候选 `.skill`
4. 用真实 Agent 做第一轮触发验证
5. 再做真实 Agent 执行验证
6. 后期再专门做 description 优化

换句话说：

- 不是一直闷头只做本地内循环
- 也不是一开始就把重点放在 `.skill` 安装包上
- 而是先把 skill 本体做成候选版，再尽快进入 Agent 外循环

---

## 7. 推荐口径

### 对内

我们当前迭代的是 Skill 源码目录，`.skill` 只是每轮候选安装包。先通过本地自迭代把 Skill 本体做稳定，再通过真实 Agent 外循环验证触发和执行效果。

### 对外

我们交付的是 formal skill 包及其验收结果，而不是单次 prompt 调优结果。

---

## 8. 最简决策规则

如果现在遇到“下一步该干什么”的分歧，按下面判断：

### 情况 A

如果 Skill 本体还不稳定：

- 继续做当前内循环

### 情况 B

如果 Skill 本体已经基本稳定，但还没在真实 Agent 里测过：

- 立刻打一个候选 `.skill`，进入 Agent 外循环

### 情况 C

如果真实 Agent 已经能稳定触发并执行，但触发率不够好：

- 再做 description 专项优化

---

## 9. 禁止误区

不要这样做：

- 把 `.skill` 当成源码直接维护
- 没经过真实 Agent 验证就宣称接入完成
- 只优化 description，不优化 skill 本体
- 只看本地 benchmark，不看真实 Agent 触发
- 只看真实触发，不看执行质量

---

## 10. 最后结论

推荐路径不是：

- 一直做到最终 `.skill` 再开始验证

也不是：

- 一装到 Agent 里就把安装包当主版本持续乱改

而是：

- 先把源码目录做成稳定候选版
- 尽快打包候选 `.skill`
- 用真实 Agent 做外循环
- 始终回源码目录迭代
- 在后期单独优化 description

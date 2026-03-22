# minor-protection

这个项目现在围绕两条主线运行：

- 模式 A：Agent full loop 分层采样 -> Agent 调 skill -> skill 内总控脚本 -> Agent 返回结果 -> judge -> optimizer -> compare 或 rollback -> 人审
- 模式 B：Direct full loop 分层采样 -> skill 内总控脚本 -> judge -> optimizer -> compare 或 rollback -> 人审

当前真正的主链是 `src/skill_loop/*`、`scripts/run_skill_iteration_loop.py`、`scripts/run_direct_iteration_loop.py`、`scripts/run_final_test.py`，以及 `skills/minor-detection/*` 这套 bundled skill。

## 两种模式

### Mode A

入口：`scripts/run_skill_iteration_loop.py` （目前使用的是 Codex-Agent 后期可扩展 ）

主流程：

1. `src/skill_loop/loop.py` 创建工作区、冻结 baseline、调用 runner。
2. `src/skill_loop/runner.py` 对验证集做分层采样，把 skill 快照安装到隔离环境，逐样本让 Agent 只执行一次预制 launcher。
3. launcher 调 `skills/minor-detection/scripts/run_minor_detection_pipeline.py`。
4. pipeline 内部按顺序执行：payload 归一化 -> 时间脚本 -> 检索脚本（RAG） -> 分类器 -> schema repair -> 输出 observability。
5. `src/skill_loop/judge.py` 把 run artifacts 转成 judge report、failure packets、protected packets。
6. `src/evolution/optimizer.py` 根据 judge 产物生成 candidate skill 版本。
7. `src/skill_loop/compare.py` 决定 promote 还是 rollback。
8. 最终结果交给人审，并可再用 `scripts/run_final_test.py` 跑 test 集。

### Mode B

入口：`scripts/run_direct_iteration_loop.py`

和 Mode A 的区别只有执行层：

- Mode A 用 `src/skill_loop/runner.py`，由 Agent 去调一次预制 launcher。
- Mode B 用 `src/skill_loop/direct_runner.py`，直接跑 `run_minor_detection_pipeline.py`。
- judge、optimizer、compare、versioning、报告结构都是同一套。

## 目录结构

```text
.
├── README.md
├── app_formal.py                      # 正式 runtime 演示页，便于人工查看 payload/context/output
├── app_v2.py                          # 旧版 executor/RAG/memory 演示页，已不在当前主链
├── scripts/
│   ├── run_skill_iteration_loop.py    # Mode A 入口
│   ├── run_direct_iteration_loop.py   # Mode B 入口
│   ├── run_final_test.py              # 手动 final test 入口
│   ├── list_skill_versions.py         # 版本盘点
│   ├── cleanup_skill_versions.py      # 版本清理
│   ├── prepare_mode_a_validation_seed.py
│   ├── inspect_formal_sample.py
│   ├── export_retrieval_assets.py
│   ├── prepare_data.py                # 数据集准备
│   ├── run_acceptance_suite.py        # 旧验收链路
│   └── run_pipeline.py                # 旧离线总控链路
├── src/
│   ├── config.py
│   ├── models.py
│   ├── utils/
│   ├── evolution/
│   ├── executor/                      # 旧 executor 兼容层
│   ├── retriever/                     # 旧 external RAG 兼容层
│   ├── memory/                        # 旧 memory 兼容层
│   ├── runtime/                       # formal runtime 适配层，给 demo/inspection 用
│   └── skill_loop/                    # 当前主线核心
├── skills/
│   ├── active_version.txt
│   ├── minor-detection/               # 当前 source-of-truth skill 目录
│   ├── minor-detection-v0.1.0/        # baseline 快照
│   └── 其他带版本号目录                 # 历史候选 / 验证 seed / 已发布快照
├── test/
│   ├── test_skill_loop.py             # 当前主链最重要测试
│   ├── test_validation_seed.py
│   ├── test_versioning.py
│   ├── test_formal_skill_runtime.py
│   └── 若干旧 demo/旧技术文档测试
├── reports/                           # 各次 loop / final test 输出
├── tmp/                               # 临时实验、历史草稿、validation seed 残留
└── claude-skill-creator/              # 打包/校验 bundled skill 需要的外部工具
```

## 当前应该重点看的文件

### 主线入口

- `scripts/run_skill_iteration_loop.py`：Mode A CLI 入口。
- `scripts/run_direct_iteration_loop.py`：Mode B CLI 入口。
- `scripts/run_final_test.py`：在 test 集上做最终人工前检查。

### Loop 核心

- `src/skill_loop/loop.py`：把 runner、judge、optimizer、compare 串成完整闭环。
- `src/skill_loop/runner.py`：Mode A 的 Agent 执行器。
- `src/skill_loop/direct_runner.py`：Mode B 的 Direct 执行器。
- `src/skill_loop/judge.py`：从样本级 artifacts 聚合成 judge report 和 packet。
- `src/skill_loop/compare.py`：是否 promote 的门禁逻辑。
- `src/skill_loop/versioning.py`：skill 版本命名、快照、库存、清理预览。
- `src/skill_loop/schema_consistency.py`：保证 prompt 合同和 formal schema 没漂。
- `src/skill_loop/validation_seed.py`：Mode A 自迭代验收专用坏 baseline 生成器。
- `src/skill_loop/packaging.py`：调用 `claude-skill-creator` 做校验和打包。

### 优化器与共享契约

- `src/evolution/optimizer.py`：根据 judge failure / protected packets 改 skill。
- `src/models.py`：formal 输出、legacy 输出、payload 的统一数据模型。
- `src/config.py`：全局路径和 active skill 指针。
- `src/utils/path_utils.py`：把报告里的绝对路径转成稳定相对路径。

### Skill 内部总控

- `skills/minor-detection/scripts/run_minor_detection_pipeline.py`：真正被 Mode A / B 调起的 skill 总控脚本。
- `skills/minor-detection/scripts/extract_time_features.py`：时间特征脚本。
- `skills/minor-detection/scripts/retrieve_cases.py`：内置检索脚本。
- `skills/minor-detection/scripts/_payload_normalizer.py`：payload 标准化。
- `skills/minor-detection/scripts/_classifier_client.py`：分类模型请求封装。
- `skills/minor-detection/scripts/_schema_repair.py`：输出修复。
- `skills/minor-detection/scripts/_profile_merge.py`：补齐 profile / evidence / risk 字段。
- `skills/minor-detection/scripts/config.py`：skill 内部运行时配置。

### skill版本

- `skills/minor-detection/`：当前 source-of-truth。
- `skills/minor-detection-v0.1.0/`：当前 loop 默认 baseline。
- `skills/active_version.txt`：旧 runtime 仍会读它。
- `claude-skill-creator/`：`src/skill_loop/packaging.py` 依赖它做校验和打包。

## 更详细的结构说明

详细版请看：`docx/项目结构代码介绍.md`
# minor-protection

`minor-protection` 是一套面向 AI 对话产品的未成年人识别与自进化优化工具链。

项目要解决的核心问题是：

> 当 AI 产品正在和用户持续互动时，系统能否识别说话者是否疑似未成年人，并自动触发适龄保护、风险分级和证据留存流程？

## 项目简介

本仓库主要开源以下内容：

- 未成年人识别自进化工具链
- `minor-detection` bundled skill
- 正式运行时桥接层
- Streamlit 前端演示界面
- 配套测试与核心脚本

本仓库定位为**代码仓库**。  
公开数据集已单独发布在 Hugging Face，不随本仓库一起提供。

## 已发布资源

- Hugging Face 知识子集：<https://huggingface.co/datasets/xiao2005/minor-detection-knowledge-subset>
- Hugging Face 社交子集：<https://huggingface.co/datasets/xiao2005/minor-detection-social-subset>
- ClawHub Skill：<https://clawhub.ai/xiaohanzhang2005/minor-detection>

## 核心能力

- 判断当前聊天窗口或任务请求是否需要触发未成年人深度识别
- 支持单会话、多会话未成年人识别
- 综合多维证据进行判断，包括：
  - 当前对话
  - 历史画像
  - 时间特征
  - 相似案例检索
- 输出结构化结果，包括：
  - 是否疑似未成年人
  - 用户画像
  - 证据链
  - 风险等级
  - 下一步建议
- 支持离线自进化闭环，包括：
  - 自动评测
  - judge 报告
  - candidate 优化
  - promote / rollback 门禁
  - 人审接口

## 仓库结构

```text
.
├── src/                         # 核心运行时、loop、optimizer、models
├── scripts/                     # CLI 入口和维护脚本
├── skills/minor-detection/      # 当前 source-of-truth skill
├── test/                        # 运行时和 loop 测试
├── demo_inputs/                 # 最小 demo 输入
├── app_minor_detection.py       # Streamlit 前端演示页
├── app_formal.py                # formal runtime 调试页
└── requirements.txt             # 依赖列表
```

## 主要执行链路

### 1. Direct Loop

Direct Loop 直接调用 bundled skill 的总控脚本：

```text
scripts/run_direct_iteration_loop.py
  -> src/skill_loop/loop.py
  -> direct runner
  -> skills/minor-detection/scripts/run_minor_detection_pipeline.py
  -> judge
  -> optimizer
  -> compare / rollback / promote
```

### 2. Agent Loop

Agent Loop 通过预制 launcher 调用同一套 skill：

```text
scripts/run_skill_iteration_loop.py
  -> src/skill_loop/loop.py
  -> agent runner
  -> launcher
  -> skills/minor-detection/scripts/run_minor_detection_pipeline.py
  -> judge
  -> optimizer
  -> compare / rollback / promote
```

## 前端与运行时

当前对外演示前端入口：

- `app_minor_detection.py`

该页面用于展示：

- 单会话检测
- 多会话检测
- 外部上下文注入
- 风险等级与证据链展示

补充调试页：

- `app_formal.py`

## Bundled Skill

当前核心 skill 位于：

- `skills/minor-detection/`

其主流程包括：

- payload 归一化
- 时间特征提取
- 相似案例检索
- 分类器调用
- schema 修复
- formal 输出合并

主要入口脚本：

- `skills/minor-detection/scripts/run_minor_detection_pipeline.py`

## 快速开始

安装依赖：

```bash
pip install -r requirements.txt
```

运行前端演示：

```bash
streamlit run app_minor_detection.py
```

运行 formal 调试页：

```bash
streamlit run app_formal.py
```

运行测试：

```bash
python -m unittest discover -s test
```

## 环境变量

bundled skill 主要读取以下环境变量：

- `MINOR_DETECTION_CLASSIFIER_BASE_URL`
- `MINOR_DETECTION_CLASSIFIER_API_KEY`
- `MINOR_DETECTION_CLASSIFIER_MODEL`
- `MINOR_DETECTION_EMBEDDING_BASE_URL`
- `MINOR_DETECTION_EMBEDDING_API_KEY`
- `MINOR_DETECTION_EMBEDDING_MODEL`

如果没有配置分类器凭证，运行时不会静默调用未知远程接口，而是直接报错。

## 工程说明

本公开代码仓库不包含以下内容：

- Hugging Face 数据集发布中间文件
- 评测报告与运行产物
- 临时实验目录
- 本地 IDE / Codex / pytest 缓存
- 外部打包辅助目录 `claude-skill-creator/`

当前仓库保留：

- 核心代码
- 前端演示页
- `data/` 目录中的项目数据文件

## 当前状态

- 工程原型：已完成
- bundled skill：已完成
- 前端演示页：已完成
- 内部评测链路：已完成
- Hugging Face 数据子集发布：已完成
- GitHub 代码仓库发布：准备中

<div align="center">

# Minor Detection

### 面向 AI 拟人化互动的自进化未成年人识别智能体

<p>
  <a href="https://huggingface.co/datasets/xiao2005/minor-detection-social-subset">
    <img src="https://img.shields.io/badge/Dataset-Social_Subset-0A66C2?style=for-the-badge" alt="social dataset" />
  </a>
  <a href="https://huggingface.co/datasets/xiao2005/minor-detection-knowledge-subset">
    <img src="https://img.shields.io/badge/Dataset-Knowledge_Subset-146356?style=for-the-badge" alt="knowledge dataset" />
  </a>
  <a href="https://clawhub.ai/xiaohanzhang2005/minor-detection">
    <img src="https://img.shields.io/badge/ClawHub-Minor_Detection_Skill-DB4437?style=for-the-badge" alt="clawhub skill" />
  </a>
  <a href="https://github.com/xiaohanzhang2005/Minor-Detection">
    <img src="https://img.shields.io/badge/GitHub-Minor_Detection-111111?style=for-the-badge" alt="github repo" />
  </a>
</p>

<p><strong>离线自进化工具链 × 轻量化未成年人检测 Skill × 多维证据链系统</strong></p>

</div>

> 当 AI 产品开始长期陪伴、持续对话、拟人化互动，未成年人保护就不再只是“加分项”，而是产品能力、工程能力与合规能力的底线。  
> **Minor Detection** 想解决的，不是单点年龄猜测，而是从“疑似未成年人识别”到“风险干预、模式切换、人工复核”的整条治理链路。

<div align="center">

**为 AI 拟人化互动场景提供未成年人识别、证据链输出、风险分级与自进化优化能力**

</div>

## 项目概览

**Minor Detection** 是一套面向 AI 陪伴、教育、客服、审核等对话产品的未成年人识别解决方案，提供：

- 面向真实业务的未成年人识别与风险分级
- 支持单会话、多会话、时间线索、RAG 检索与长期画像建模
- 可落地的结构化证据链输出
- 面向上线治理的模式切换与人工复核前置能力
- 一条可持续优化的离线自进化工具链

它不是一次性的“年龄猜测”分类器，而是面向 AI 拟人化互动场景的**可嵌入式风险治理能力层**。

## 为什么值得关注

| 常见方案 | Minor Detection |
| --- | --- |
| 只做一次性年龄猜测 | 做从触发判断到人工复核的完整治理链路 |
| 只看单轮文本 | 支持单会话、多会话、时间线索、长期画像与 RAG 证据 |
| 规则写死、上线即冻结 | 围绕数据集、评测、优化与人审持续进化 |
| 研究原型难接业务 | 支持 Skill、工作台、运行时桥接与版本化优化链路 |
| 依赖实名、人脸或平台账户 | 可直接嵌入现有文本对话产品 |

## 核心亮点

| 模块 | 亮点 |
| --- | --- |
| 数据基座 | 覆盖社交/心理与知识/教育两类成长路径，支撑显式与隐式年龄信号识别 |
| 触发边界 | 单独用 Trigger-Eval 优化“什么时候该启动未成年人深判” |
| 证据链 | 综合当前对话、历史画像、时间特征、相似案例检索与反向信号 |
| 结构化输出 | 输出未成年人概率、画像、证据链、风险等级、下一步建议 |
| 自进化机制 | 支持报告驱动优化、版本对比、promote / rollback、人审门禁 |
| 工程形态 | 支持 Skill、脚本、运行时桥接、前端工作台等多种接入方式 |

## 系统演示

### 前端工作台 GIF

| 工作流 | 演示 |
| --- | --- |
| 上传文件与工作台入口 | ![](GIF/1-工作台上传文件.gif) |
| 输入预览 | ![](GIF/2-输入预览.gif) |
| 多会话识别中 | ![](GIF/3-多会话识别中.gif) |
| 识别结果 | ![](GIF/4-识别结果.gif) |
| 未成年人概率曲线 | ![](GIF/5-多会话输出未成年人概率曲线.gif) |
| 多维证据链与外部上下文注入 | ![](GIF/6-多维证据链与外部上下文注入汇总.gif) |

### 演示视频

- 项目演示视频：[Minor-Detection-字幕配音版.mp4](video/Minor-Detection-字幕配音版.mp4)

## 一句话看懂方案

```text
聊天窗口 / 多会话历史 / 上游任务请求
                |
         Trigger 边界判断
                |
      Minor Detection Pipeline
 (时间特征 + RAG相似案例 + 分类器 + Schema修复)
                |
  结构化输出：未成年人概率 / 用户画像 / 证据链 / 风险等级 / 下一步建议
                |
 未成年人模式切换 / 人工复核 / 家长侧提醒 / 审核中台 / 风险运营
                |
      离线自进化闭环持续优化
```

> **离线自进化工具链 + 轻量化判定 Skill + 多维证据链系统 = 数据驱动、自适应演进的未成年人风险监测智能体**

## 为什么这个问题正在变成刚需

2025 年 12 月 27 日，[国家互联网信息办公室发布《人工智能拟人化互动服务管理暂行办法（征求意见稿）》](https://www.cac.gov.cn/2025-12/27/c_1768571207311996.htm)。从监管口径看，拟人化互动服务正在被要求具备更强的状态识别、特殊群体保护与风险干预能力。

其中与本项目最直接相关的重点包括：

- **第十一条**：要求提供者具备用户状态识别能力，在保护用户个人隐私前提下评估情绪、依赖程度与极端风险，并在必要时介入干预。
- **第十二条**：要求提供者建立未成年人模式，向用户提供未成年人模式切换、定期现实提醒、使用时长限制等个性化安全设置选项。
- **第十二条同时提出**：提供者应具备识别未成年人身份的能力，在保护用户个人隐私前提下识别为疑似未成年人的，应切换至未成年人模式，并提供申诉渠道。

这意味着，未成年人保护正从“平台倡议”加速走向“监管刚需”。  
对于 AI 陪伴与拟人化互动产品来说，真正的问题已经不再是“要不要做”，而是：

> 如何在不依赖实名、人脸或平台级账户体系的前提下，仅基于对话内容与行为线索，做出可解释、可落地、可持续优化的疑似未成年人识别？

## 行业痛点

现有对话产品在未成年人保护上通常同时面临三类难题：

| 痛点 | 典型表现 | 为什么难 |
| --- | --- | --- |
| 识别难 | 用户不会直接说年龄，更多通过“晚自习、补课、家长管控、班主任、宿舍”等隐性信号暴露身份 | 传统关键词和单轮分类很难处理隐性校园语境与组合证据 |
| 落地难 | 企业缺少“疑似未成年人识别 -> 风险分级 -> 证据留存 -> 人工复核”的完整闭环 | 模型即使能猜年龄，也常常无法直接接到业务链路 |
| 合规难 | 拟人化互动服务的监管要求正在提高 | 企业需要低改造成本、可快速接入、可规模推广的工程化方案 |

## 项目特色

### 1. 面向拟人化互动，而不是普通文本分类

我们关注的是 AI 和用户持续互动中的风险识别，而不是静态文本标签预测。  
因此系统天然支持：

- 单会话识别
- 多会话历史识别
- 长期用户画像建模
- 时段与行为机会窗口分析
- 风险等级与后续建议输出

### 2. 不依赖实名、人脸或账号体系

相比账号级治理或自拍估龄路线，Minor Detection 更适合直接嵌入企业已有文本对话流：

- 不要求实名注册
- 不要求上传自拍或证件
- 不依赖平台级账户控制权
- 更适合隐私敏感、低摩擦、连续互动场景

### 3. 关注“什么时候该触发”，而不只是“最后怎么判”

在真实产品里，系统首先要解决的是：

- 当前聊天窗口是否已经值得启动深度识别
- 当前请求是否真的在指向未成年人判断

因此我们单独设计了 **Trigger-Eval** 数据集，用于优化 `minor-detection` skill 的触发边界，而不是把所有问题混在一个分类器里处理。

### 4. 能持续优化，而不是一次部署后长期冻结

系统支持围绕固定数据集进行：

**评测 → 诊断 → 优化 → 晋级 / 回滚 → 人工审核**

这样做的核心价值是：让能力提升从“人工经验调规则”，转为“围绕真实边界样本持续演化”。

## 数据规模与公开资源

### 数据规模

项目当前公开与内部说明中的核心规模如下：

| 数据模块 | 说明 | 规模 |
| --- | --- | --- |
| Benchmark 社交/心理领域 | 心理咨询平台真实数据整理后的公开边界版本 | `2603` 正 + `1735` 负 |
| Benchmark 知识领域 | K12 课本、课程知识点与成人考试考纲构造 | `2004` 正 + `1986` 负 |
| Trigger-Eval | skill 触发边界 benchmark | `160` 条 |
| RAG 检索案例库 | 运行时相似案例辅助判断与离线优化参考 | `5829` 条 |

合计可强调为：

- **8,328 条 Benchmark 数据基座**
- **5,829 条 RAG 检索案例**
- **160 条 Trigger-Eval 触发边界数据集**

### 公开入口

- 社交对话数据集：<https://huggingface.co/datasets/xiao2005/minor-detection-social-subset>
- 知识对话数据集：<https://huggingface.co/datasets/xiao2005/minor-detection-knowledge-subset>
- ClawHub Skill：<https://clawhub.ai/xiaohanzhang2005/minor-detection>

### 为什么同时开源数据集、Skill 与工具链

| 开源对象 | 作用 |
| --- | --- |
| Hugging Face 数据集 | 提供可复核、可评测、可复现实验的数据基座 |
| ClawHub Skill | 提供可直接调用的轻量化未成年人检测能力 |
| GitHub 工具链 | 提供自进化、版本对比、人审门禁、运行时桥接与前端展示 |

## 与常见方案相比，我们的不同

| 路线 | 代表方案 | 更适合什么场景 | 我们的差异 |
| --- | --- | --- | --- |
| 平台级年龄预测 / 账户治理 | OpenAI、Meta | 自营平台、账号体系完备的消费级产品 | 我们更适合企业已有对话流的 B 端嵌入，不依赖自拍、实名或平台级账户控制 |
| 生物特征年龄估计 | Yoti 等 | 注册、支付、成人内容等高强校验场景 | 我们不采集生物特征，交互摩擦更低，更适合连续对话与陪伴式互动 |
| 传统规则 / 关键词方案 | 常规风控规则库 | 快速初筛 | 很难处理隐性校园信号、多会话趋势、时段异常和复杂证据链 |

从工程视角看，Minor Detection 的优势在于：

- 不依赖实名或人脸，接入门槛更低
- 不只输出标签，还输出证据链、风险等级与下一步建议
- 支持单会话、多会话与触发边界判断
- 具备持续进化能力，而不是一次部署后长期固定

## 理论基础

<details>
<summary><strong>1. ICBO：从“像不像未成年人”到“为什么这么判断”</strong></summary>

ICBO 是本项目组织用户画像与证据解释的核心视角：

- **I - Intention**：用户当前的直接意图，例如作业求助、校园压力倾诉、考试安排讨论
- **C - Cognition**：以克制、可审计的方式描述其认知特点，而非过度心理诊断
- **B - Behavior Style**：关注语言与行为风格，例如表达方式、校园化措辞、情绪起伏
- **O - Opportunity Time**：保留原始时间线索，并追加结构化时间标签，用于时段机会窗口分析

![ICBO](picture/ICBO.png)

</details>

<details>
<summary><strong>2. 自迭代链条：评测、诊断、优化、晋级/回滚、人审</strong></summary>

我们的核心不是一次性写出一份规则，而是构建一条可重复运行的离线演化链路：

1. 基于固定数据集评测当前版本 Skill
2. judge 生成失败样本、护栏样本与结构化报告
3. optimizer 针对性改写触发边界或描述
4. 新旧版本对比，决定 promote 或 rollback
5. 人工审核作为最终门禁，防止为了指标而偷换边界

这种机制让 Skill 的优化从“人工拍脑袋改规则”，转成“围绕数据边界持续收敛”的工程过程。

</details>

<details>
<summary><strong>3. Trigger-Eval：先解决触发时机，再做深度判别</strong></summary>

Trigger-Eval 回答的问题不是“这个人最终是不是未成年人”，而是：

- 当前窗口是否已经值得调用 `minor-detection`
- 上游请求是否已经明确在要求未成年人识别
- 哪些边界场景必须触发，哪些场景不能误触发

这种“先做触发边界，再做深度判别”的两阶段设计，可以显著降低误触发与乱触发。

</details>

<details>
<summary><strong>4. 长期用户建模：单轮不够，多轮趋势更重要</strong></summary>

真实产品中的未成年人信号往往不会一次性直接出现。  
因此系统支持：

- 单会话窗口扫描
- 多会话历史归纳
- 时间特征提取
- 历史画像合并
- 概率曲线与趋势判断

这使它更适合长期互动、拟人陪伴与连续服务场景，而不是一次性短文本分类。

</details>

<details>
<summary><strong>5. 人工审核为什么必须保留</strong></summary>

未成年人识别具有明显的伦理与合规敏感性，因此本项目明确保留人工审核作为最终门禁。

人工审核主要防止：

- 优化器为了指标而偷换边界
- 版本升级引入不可解释的误伤
- 在高风险场景中把概率判断误当成确定身份

我们的立场是：**模型负责发现风险与提供证据，人类负责最终治理决策。**

</details>

## 下游落地场景

Minor Detection 不只是一个判别器，更适合作为风险治理基础设施的一部分，用于：

- 未成年人模式自动切换
- 人工复核分流
- 家长侧提醒与监护人控制联动
- 审核中台接入
- 风险运营与高风险用户预警
- AI 陪伴、教育大模型、智能客服、社区审核等产品线

## 仓库内容

本仓库主要开源以下内容：

- 未成年人识别自进化工具链
- `minor-detection` bundled skill
- 正式运行时桥接层
- Streamlit 前端演示界面
- 配套测试与核心脚本

```text
.
├── src/                         # 核心运行时、loop、optimizer、models
├── scripts/                     # CLI 入口和维护脚本
├── skills/minor-detection/      # 当前 source-of-truth skill
├── test/                        # 运行时和 loop 测试
├── demo_inputs/                 # 最小 demo 输入
├── GIF/                         # 前端动图演示
├── picture/                     # README 配图
├── video/                       # 演示视频
├── app_minor_detection.py       # Streamlit 前端演示页
└── requirements.txt             # 依赖列表
```

## 快速开始

安装依赖：

```bash
pip install -r requirements.txt
```

运行前端演示：

```bash
streamlit run app_minor_detection.py
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

## 伦理与使用声明

<details>
<summary><strong>请在引用、部署或二次开发前阅读</strong></summary>

- 本项目面向未成年人保护、风险识别与产品安全治理，不用于法律意义上的年龄认证。
- 基于对话内容的未成年人判断本质上是概率推断，而非身份事实确认。
- 项目不鼓励将模型输出直接用于惩罚性、歧视性或不可申诉的自动化决策。
- 涉及高风险处置、模式切换、账号限制或监护人联动时，应保留人工复核与申诉机制。
- 数据集公开部分遵循保守边界，不发布可追溯真实个体身份的信息。

</details>

## 引用

如果你在论文、报告、产品评测或开源项目中使用了本项目，欢迎引用：

```bibtex
@misc{minor_detection_github_2026,
  title        = {Minor Detection: Self-Evolving Minor-User Identification Agent for Anthropomorphic AI Interaction},
  author       = {Xiaohan Zhang and Yukun Wei and Kaibo Huang and Zhongliang Yang and Linna Zhou},
  year         = {2026},
  howpublished = {\url{https://github.com/xiaohanzhang2005/Minor-Detection}},
  note         = {GitHub repository}
}
```

## 致谢

本项目聚焦于 AI 拟人化互动场景中的未成年人保护，希望为行业提供一套兼顾**合规要求、工程可落地性与持续优化能力**的开源实践。

## GitHub Repo Setup

如果你准备继续优化 GitHub 仓库首页展示，建议同步填写：

- `About`：
  `Self-evolving minor-user identification agent for anthropomorphic AI interaction, with trigger evaluation, evidence chains, and deployable protection workflows.`
- `Website`：
  `https://clawhub.ai/xiaohanzhang2005/minor-detection`
- `Topics`：
  `minor-protection`, `ai-safety`, `dialogue-systems`, `risk-detection`, `rag`, `agent`, `streamlit`, `huggingface-datasets`

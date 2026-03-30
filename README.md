<div align="center">

# Minor Detection

### 面向 AI 拟人化互动的自进化未成年人识别智能体

<p>
  <a href="https://huggingface.co/datasets/xiao2005/minor-detection-social-subset">
    <img src="https://img.shields.io/badge/HuggingFace-Social_Subset-0A66C2?style=for-the-badge" alt="Social Subset" />
  </a>
  <a href="https://huggingface.co/datasets/xiao2005/minor-detection-knowledge-subset">
    <img src="https://img.shields.io/badge/HuggingFace-Knowledge_Subset-146356?style=for-the-badge" alt="Knowledge Subset" />
  </a>
  <a href="https://clawhub.ai/xiaohanzhang2005/minor-detection">
    <img src="https://img.shields.io/badge/ClawHub-Minor_Detection_Skill-DB4437?style=for-the-badge" alt="ClawHub Skill" />
  </a>
  <a href="./video/minor-detection-demo.mp4">
    <img src="https://img.shields.io/badge/Demo-Video-111111?style=for-the-badge" alt="Demo Video" />
  </a>
</p>

<p><strong>离线自进化工具链 × 轻量化未成年人检测 Skill × 多维证据链系统</strong></p>

<p>为 AI 陪伴、教育、客服、社区审核等对话产品提供未成年人识别、证据链输出、风险分级与可持续优化能力。</p>

</div>

---

## Overview

**Minor Detection** 不是一次性的年龄猜测模型，而是一套面向 AI 拟人化互动场景的可嵌入式风险治理能力层。

它解决的是一条完整链路：

- 什么时候该触发未成年人识别
- 如何结合多维证据做出判断
- 如何输出可审计的证据链与风险等级
- 如何衔接未成年人模式、人工复核和风险运营
- 如何围绕真实边界样本持续优化能力

---

## Demo

<div align="center">
  <table>
    <tr>
      <td align="center" width="50%">
        <strong>工作台上传文件</strong><br/><br/>
        <img src="GIF/workbench-upload.gif" alt="workbench upload" width="100%"/>
      </td>
      <td align="center" width="50%">
        <strong>输入预览</strong><br/><br/>
        <img src="GIF/input-preview.gif" alt="input preview" width="100%"/>
      </td>
    </tr>
    <tr>
      <td align="center" width="50%">
        <strong>多会话识别中</strong><br/><br/>
        <img src="GIF/multi-session-processing.gif" alt="multi session processing" width="100%"/>
      </td>
      <td align="center" width="50%">
        <strong>识别结果</strong><br/><br/>
        <img src="GIF/result-overview.gif" alt="result overview" width="100%"/>
      </td>
    </tr>
    <tr>
      <td align="center" width="50%">
        <strong>未成年人概率曲线</strong><br/><br/>
        <img src="GIF/minor-probability-curve.gif" alt="minor probability curve" width="100%"/>
      </td>
      <td align="center" width="50%">
        <strong>多维证据链与外部上下文注入</strong><br/><br/>
        <img src="GIF/evidence-and-context.gif" alt="evidence and context" width="100%"/>
      </td>
    </tr>
  </table>
</div>

<div align="center">

<a href="./video/minor-detection-demo.mp4"><strong>Watch Full Demo Video</strong></a>

</div>

---

## Why Now

2025 年 12 月 27 日，[国家互联网信息办公室发布《人工智能拟人化互动服务管理暂行办法（征求意见稿）》](https://www.cac.gov.cn/2025-12/27/c_1768571207311996.htm)，未成年人保护在 AI 拟人化互动场景中正从产品倡议加速走向监管刚需。

与本项目直接相关的政策重点包括：

- 第十一条强调用户状态识别能力，以及对极端情绪、沉迷与高风险依赖的必要干预
- 第十二条要求建立未成年人模式，提供模式切换、现实提醒、时长限制等安全设置
- 第十二条同时提出，提供者应具备识别未成年人身份的能力，识别为疑似未成年人时应切换至未成年人模式并提供申诉渠道

这意味着，对于 AI 陪伴与拟人化互动产品，真正的问题已经不再是“要不要做”，而是：

> 如何在不依赖实名、人脸或平台级账户体系的前提下，仅基于对话内容与行为线索，做出可解释、可落地、可持续优化的疑似未成年人识别？

---

## What Makes It Different

**Minor Detection** 试图避免传统方案常见的四个问题：

- 不是只做一次性年龄猜测，而是覆盖触发判断、深度识别、证据链输出和人工复核的完整治理链路
- 不是只看单轮文本，而是支持单会话、多会话、时间线索、长期画像与 RAG 证据融合
- 不是规则写死上线即冻结，而是围绕数据集、评测、优化和人审持续进化
- 不是难以接入业务的研究原型，而是支持 Skill、工作台、运行时桥接和版本化迭代的工程方案

---

## Pipeline

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

<div align="center">
<strong>离线自进化工具链 + 轻量化判定 Skill + 多维证据链系统 = 数据驱动、自适应演进的未成年人风险监测智能体</strong>
</div>

---

## Key Capabilities

- **Trigger 边界判断**
  决定当前窗口或任务请求是否已经值得启动未成年人深度识别

- **多维证据融合**
  综合当前对话、历史画像、时间特征、相似案例检索与反向信号进行判断

- **结构化输出**
  输出未成年人概率、用户画像、证据链、风险等级与下一步建议

- **长期用户建模**
  支持从单轮判断扩展到多会话趋势与持续性风险识别

- **离线自进化**
  围绕固定数据集执行 `评测 → 诊断 → 优化 → 晋级 / 回滚 → 人工审核`

---

## Data And Open Resources

### Public Resources

- 社交对话数据集  
  <https://huggingface.co/datasets/xiao2005/minor-detection-social-subset>

- 知识对话数据集  
  <https://huggingface.co/datasets/xiao2005/minor-detection-knowledge-subset>

- ClawHub Skill  
  <https://clawhub.ai/xiaohanzhang2005/minor-detection>

- 项目演示视频  
  [video/minor-detection-demo.mp4](video/minor-detection-demo.mp4)

### Data Scale

- `8,328` 条 Benchmark 数据基座
- `5,829` 条 RAG 检索案例
- `160` 条 Trigger-Eval 触发边界数据集

进一步拆开看：

- 社交/心理领域：`2603` 正样本 + `1735` 负样本
- 知识领域：`2004` 正样本 + `1986` 负样本
- Trigger-Eval：`window_scan = 120`，`direct_request = 40`

### Why Open All Three

- **Hugging Face 数据集**：提供可复核、可评测、可复现实验的数据基座
- **ClawHub Skill**：提供可直接调用的轻量化未成年人检测能力
- **GitHub 工具链**：提供自进化、版本对比、人审门禁、运行时桥接与前端展示

---

## Comparison

与常见路线相比，Minor Detection 更适合文本对话场景中的 B 端嵌入：

- 相比平台级年龄预测或账户治理方案
  我们不依赖平台账户体系，更适合已有对话产品的外嵌接入

- 相比自拍 / 人脸年龄估计方案
  我们不采集生物特征，交互摩擦更低，更适合连续互动和隐私敏感场景

- 相比传统规则 / 关键词方案
  我们更能处理隐性校园信号、多会话趋势、时段异常和复杂证据链

---

## Theory

<details>
<summary><strong>1. ICBO：从“像不像未成年人”到“为什么这么判断”</strong></summary>

ICBO 是本项目组织用户画像与证据解释的核心视角：

- **I - Intention**：用户当前的直接意图，例如作业求助、校园压力倾诉、考试安排讨论
- **C - Cognition**：以克制、可审计的方式描述其认知特点，而非过度心理诊断
- **B - Behavior Style**：关注语言与行为风格，例如表达方式、校园化措辞、情绪起伏
- **O - Opportunity Time**：保留原始时间线索，并追加结构化时间标签，用于时段机会窗口分析

<br/>

<img src="picture/ICBO.png" alt="ICBO" width="100%"/>

</details>

<details>
<summary><strong>2. Trigger-Eval：先解决触发时机，再做深度判别</strong></summary>

Trigger-Eval 回答的问题不是“这个人最终是不是未成年人”，而是：

- 当前窗口是否已经值得调用 `minor-detection`
- 上游请求是否已经明确在要求未成年人识别
- 哪些边界场景必须触发，哪些场景不能误触发

这种“先做触发边界，再做深度判别”的两阶段设计，可以显著降低误触发与乱触发。

</details>

<details>
<summary><strong>3. 自迭代链条：评测、诊断、优化、晋级/回滚、人审</strong></summary>

我们的核心不是一次性写出一份规则，而是构建一条可重复运行的离线演化链路：

1. 基于固定数据集评测当前版本 Skill
2. judge 生成失败样本、护栏样本与结构化报告
3. optimizer 针对性改写触发边界或描述
4. 新旧版本对比，决定 promote 或 rollback
5. 人工审核作为最终门禁，防止为了指标而偷换边界

</details>

<details>
<summary><strong>4. 为什么必须保留人工审核</strong></summary>

未成年人识别具有明显的伦理与合规敏感性，因此本项目明确保留人工审核作为最终门禁。

人工审核主要防止：

- 优化器为了指标而偷换边界
- 版本升级引入不可解释的误伤
- 在高风险场景中把概率判断误当成确定身份

我们的立场是：**模型负责发现风险与提供证据，人类负责最终治理决策。**

</details>

---

## Deployment Scenarios

Minor Detection 适合作为风险治理基础设施的一部分，用于：

- 未成年人模式自动切换
- 人工复核分流
- 家长侧提醒与监护人控制联动
- 审核中台接入
- 风险运营与高风险用户预警
- AI 陪伴、教育大模型、智能客服、社区审核等产品线

---

## Repository Structure

```text
.
├── src/                         # 核心运行时、loop、optimizer、models
├── scripts/                     # CLI 入口和维护脚本
├── skills/minor-detection/      # 当前 source-of-truth skill
├── test/                        # 运行时和 loop 测试
├── demo_inputs/                 # 最小 demo 输入
├── GIF/                         # README 动图演示
├── picture/                     # README 配图
├── video/                       # 项目演示视频
├── app_minor_detection.py       # Streamlit 前端演示页
└── requirements.txt             # 依赖列表
```

---

## Quick Start

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

---

## Environment Variables

bundled skill 主要读取以下环境变量：

- `MINOR_DETECTION_CLASSIFIER_BASE_URL`
- `MINOR_DETECTION_CLASSIFIER_API_KEY`
- `MINOR_DETECTION_CLASSIFIER_MODEL`
- `MINOR_DETECTION_EMBEDDING_BASE_URL`
- `MINOR_DETECTION_EMBEDDING_API_KEY`
- `MINOR_DETECTION_EMBEDDING_MODEL`

如果没有配置分类器凭证，运行时不会静默调用未知远程接口，而是直接报错。

---

## Ethics

<details>
<summary><strong>请在引用、部署或二次开发前阅读</strong></summary>

- 本项目面向未成年人保护、风险识别与产品安全治理，不用于法律意义上的年龄认证
- 基于对话内容的未成年人判断本质上是概率推断，而非身份事实确认
- 项目不鼓励将模型输出直接用于惩罚性、歧视性或不可申诉的自动化决策
- 涉及高风险处置、模式切换、账号限制或监护人联动时，应保留人工复核与申诉机制
- 数据集公开部分遵循保守边界，不发布可追溯真实个体身份的信息

</details>

---

## Citation

```bibtex
@misc{minor_detection_github_2026,
  title        = {Minor Detection: Self-Evolving Minor-User Identification Agent for Anthropomorphic AI Interaction},
  author       = {Xiaohan Zhang and Yukun Wei and Kaibo Huang and Zhongliang Yang and Linna Zhou},
  year         = {2026},
  howpublished = {\url{https://github.com/xiaohanzhang2005/Minor-Detection}},
  note         = {GitHub repository}
}
```

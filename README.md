<div align="center">

# Minor Detection

### 面向 AI 拟人化互动的自进化未成年人识别智能体

### [简体中文](README.md) | [English](README_EN.md)

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
  <a href="https://www.bilibili.com/video/BV1MRXYBgEQk/?spm_id_from=333.1387.homepage.video_card.click">
    <img src="https://img.shields.io/badge/Demo-Video-111111?style=for-the-badge" alt="Demo Video" />
  </a>
</p>

<p><strong>离线自进化工具链 × 轻量化未成年人检测 Skill × 多维证据链系统</strong></p>

<p>为 AI 陪伴、教育、客服、社区审核等对话产品提供未成年人识别、证据链输出、风险分级与可持续优化能力。</p>

</div>

---

## 项目概览

**Minor Detection** 不是一次性的年龄猜测模型，而是一套面向 AI 拟人化互动场景的可嵌入式风险治理能力层。

它解决的是一条完整链路：

- 什么时候该触发未成年人识别
- 如何结合多维证据做出判断
- 如何输出可审计的证据链与风险等级
- 如何衔接未成年人模式、人工复核和风险运营
- 如何围绕真实边界样本持续优化能力

更具体地说，项目希望回答一个现实而尖锐的问题：

> 当用户不会直接说出年龄，只会在持续对话里留下“晚自习、补课、班主任、宿舍、家长管控、考试安排、校园作息”等隐性线索时，系统能否稳定识别疑似未成年人，并把判断结果转化为真正可执行的保护动作？

因此，Minor Detection 的定位不是“再做分类器”，而是把**识别、解释、干预、复核、迭代**连成一套可上线的工程闭环。

---

## 系统演示

<div align="center">
  <table width="100%" align="center" style="width:100%; table-layout:fixed;">
    <tr>
      <td align="center" width="50%">
        <img src="picture/demo-title-workbench.svg" alt="工作台上传文件" width="100%"/>
        <br/>
        <img src="GIF/workbench-upload.gif" alt="workbench upload" width="100%"/>
      </td>
      <td align="center" width="50%">
        <img src="picture/demo-title-input-preview.svg" alt="输入预览" width="100%"/>
        <br/>
        <img src="GIF/input-preview.gif" alt="input preview" width="100%"/>
      </td>
    </tr>
    <tr>
      <td align="center" width="50%">
        <img src="picture/demo-title-processing.svg" alt="多会话识别中" width="100%"/>
        <br/>
        <img src="GIF/multi-session-processing.gif" alt="multi session processing" width="100%"/>
      </td>
      <td align="center" width="50%">
        <img src="picture/demo-title-result.svg" alt="识别结果" width="100%"/>
        <br/>
        <img src="GIF/result-overview.gif" alt="result overview" width="100%"/>
      </td>
    </tr>
    <tr>
      <td align="center" width="50%">
        <img src="picture/demo-title-curve.svg" alt="未成年人概率曲线" width="100%"/>
        <br/>
        <img src="GIF/minor-probability-curve.gif" alt="minor probability curve" width="100%"/>
      </td>
      <td align="center" width="50%">
        <img src="picture/demo-title-evidence.svg" alt="多维证据链与外部上下文注入" width="100%"/>
        <br/>
        <img src="GIF/evidence-and-context.gif" alt="evidence and context" width="100%"/>
      </td>
    </tr>
  </table>
</div>

<div align="center">

<a href="https://www.bilibili.com/video/BV1MRXYBgEQk/?spm_id_from=333.1387.homepage.video_card.click"><strong>查看完整演示视频</strong></a>

</div>

---

## 政策驱动与现实必要性

2025 年 12 月 27 日，[国家互联网信息办公室发布《人工智能拟人化互动服务管理暂行办法（征求意见稿）》](https://www.cac.gov.cn/2025-12/27/c_1768571207311996.htm)，未成年人保护在 AI 拟人化互动场景中正从产品倡议加速走向监管刚需。

与本项目直接相关的政策重点包括：

- 第十一条强调用户状态识别能力，以及对极端情绪、沉迷与高风险依赖的必要干预
- 第十二条要求建立未成年人模式，提供模式切换、现实提醒、时长限制等安全设置
- 第十二条同时提出，提供者应具备识别未成年人身份的能力，识别为疑似未成年人时应切换至未成年人模式并提供申诉渠道

这意味着，对于 AI 陪伴与拟人化互动产品，真正的问题已经不再是“要不要做”，而是：

> 如何在不依赖实名、人脸或平台级账户体系的前提下，仅基于对话内容与行为线索，做出可解释、可落地、可持续优化的疑似未成年人识别？

---

## 项目特色

Minor Detection 关注的不是单点“年龄猜测”，而是一套面向 AI 拟人化互动场景的完整治理能力。

- **完整治理链路**：覆盖触发判断、深度识别、证据链输出、人工复核与后续衔接，而不是只给出一次性分类结果。
- **多维证据融合**：同时利用单会话、多会话、时间线索、长期画像与 RAG 相似案例，更适合处理隐性校园信号和组合证据。
- **持续优化能力**：围绕固定数据集执行评测、诊断、优化、版本对比与人审门禁，不依赖人工零散改规则。
- **工程可接入性**：提供 Skill、工作台、运行时桥接与版本化迭代链路，适合直接接入已有文本对话产品。

从产品与工程视角看，这带来三个直接收益：

- **更容易接入现有业务**：可直接进入已有文本对话流，接入成本更低
- **更容易解释与复核**：不仅输出标签，还输出画像、证据链、风险等级与建议
- **更容易持续提升**：优化过程以真实边界样本与人审门禁为核心

---

## 方案链路

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

<br/>

<img src="picture/全局.png" alt="Minor Detection 全局总览" width="100%"/>

- **Trigger 边界判断**：决定当前窗口或任务请求是否已经值得启动未成年人深度识别
- **多维证据融合**：综合当前对话、历史画像、时间特征、相似案例检索与反向信号进行判断
- **结构化输出**：输出未成年人概率、用户画像、证据链、风险等级与下一步建议
- **长期用户建模**：支持从单轮判断扩展到多会话趋势与持续性风险识别
- **离线自进化**：围绕固定数据集执行 `评测 → 诊断 → 优化 → 晋级 / 回滚 → 人工审核`
- **多场景下游衔接**：可继续接到未成年人模式切换、人工复核、家长侧提醒、审核中台与风险运营

---

## 理论基础

<details>
<summary><strong>1. ICBO：从“像不像未成年人”到“为什么这么判断”</strong></summary>

<br/>

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

<br/>

Trigger-Eval 回答的问题不是“这个人是不是未成年人”，而是“当前输入是否已值得调用 `minor-detection` 这个 Skill”。

它直接对应 `skills/minor-detection/SKILL.md` 中 `description` 的触发边界优化，而不是最终分类器能力本身。  
当前这套 description 触发数据集共 `160` 条，专门用于训练和评估：

<table width="100%" align="center" style="width:100%; table-layout:fixed;">
  <colgroup>
    <col width="20%"/>
    <col width="58%"/>
    <col width="22%"/>
  </colgroup>
  <tr>
    <th align="center">维度</th>
    <th align="center">含义</th>
    <th align="center">规模</th>
  </tr>
  <tr>
    <td><code>window_scan</code></td>
    <td>窗口扫描场景，判断当前聊天窗口是否已经值得触发</td>
    <td><code>120</code></td>
  </tr>
  <tr>
    <td><code>direct_request</code></td>
    <td>直接请求场景，判断上游请求是否明确指向未成年人识别</td>
    <td><code>40</code></td>
  </tr>
  <tr>
    <td><code>should_trigger</code></td>
    <td>必须触发的正样本</td>
    <td><code>80</code></td>
  </tr>
  <tr>
    <td><code>should_not_trigger</code></td>
    <td>不能误触发的负样本</td>
    <td><code>80</code></td>
  </tr>
</table>

<p><strong>它重点优化三个问题：</strong></p>

- 当前聊天窗口是否已经出现足够的未成年人信号
- 上游请求是否真的在要求未成年人识别
- 哪些样本属于强触发边界，哪些样本只是“看起来像”但还不该触发

这种“先做触发边界，再做深度判别”的两阶段设计，可以显著降低误触发与乱触发。

</details>

<details>
<summary><strong>3. 自迭代链条：评测、诊断、优化、晋级/回滚、人审</strong></summary>

<br/>

我们的核心不是一次性写出一份规则，而是构建一条可重复运行的离线演化链路：

1. 基于固定数据集评测当前版本 Skill
2. judge 生成失败样本、护栏样本与结构化报告
3. optimizer 针对性改写触发边界或描述
4. 新旧版本对比，决定 promote 或 rollback
5. 人工审核作为最终门禁，防止为了指标而偷换边界

<br/>

<img src="picture/自进化.png" alt="自进化链路" width="100%"/>

</details>

<details>
<summary><strong>4. 保留人工审核的必要性</strong></summary>

<br/>

未成年人识别具有明显的伦理与合规敏感性，因此本项目明确保留人工审核作为最终门禁。

人工审核主要防止：

- 优化器为了指标而偷换边界
- 版本升级引入不可解释的误伤
- 在高风险场景中把概率判断误当成确定身份

我们的立场是：**模型负责发现风险与提供证据，人类负责最终治理决策。**

</details>

---

## 数据与公开资源

### 公开资源

<table width="100%" align="center">
  <colgroup>
    <col width="20%"/>
    <col width="38%"/>
    <col width="42%"/>
  </colgroup>
  <tr>
    <th align="center">资源</th>
    <th align="center">入口</th>
    <th align="center">说明</th>
  </tr>
  <tr>
    <td>社交对话数据集</td>
    <td><a href="https://huggingface.co/datasets/xiao2005/minor-detection-social-subset">Hugging Face / Social Subset</a></td>
    <td>面向社交与心理场景的公开子集</td>
  </tr>
  <tr>
    <td>知识对话数据集</td>
    <td><a href="https://huggingface.co/datasets/xiao2005/minor-detection-knowledge-subset">Hugging Face / Knowledge Subset</a></td>
    <td>面向知识与教育场景的公开子集</td>
  </tr>
  <tr>
    <td>ClawHub Skill</td>
    <td><a href="https://clawhub.ai/xiaohanzhang2005/minor-detection">ClawHub / minor-detection</a></td>
    <td>可直接调用的轻量化能力形态</td>
  </tr>
  <tr>
    <td>项目演示视频</td>
    <td><a href="https://www.bilibili.com/video/BV1MRXYBgEQk/?spm_id_from=333.1387.homepage.video_card.click">Bilibili / 完整系统演示</a></td>
    <td>完整系统演示视频</td>
  </tr>
</table>

### 数据规模

<table width="100%" align="center" style="width:100%; table-layout:fixed;">
  <colgroup>
    <col width="18%"/>
    <col width="30%"/>
    <col width="12%"/>
    <col width="40%"/>
  </colgroup>
  <tr>
    <th align="center">数据模块</th>
    <th align="center">作用</th>
    <th align="center">主规模</th>
    <th align="center">细分构成</th>
  </tr>
  <tr>
    <td>Benchmark 数据集</td>
    <td>评估真实未成年人信号与成人近似样本之间的区分能力</td>
    <td align="center"><code>8,328</code> 条</td>
    <td>社交/心理领域：<code>2603</code> 正 + <code>1735</code> 负；知识领域：<code>2004</code> 正 + <code>1986</code> 负</td>
  </tr>
  <tr>
    <td>RAG 检索案例库</td>
    <td>提供运行时相似案例辅助判断，也为离线优化提供参考证据</td>
    <td align="center"><code>5,829</code> 条</td>
    <td>覆盖未成年人识别相关案例，用于运行时检索与离线优化参考</td>
  </tr>
  <tr>
    <td>Trigger-Eval 触发边界数据集</td>
    <td>优化“什么时候该启动深度识别”这一触发边界问题</td>
    <td align="center"><code>160</code> 条</td>
    <td><code>window_scan = 120</code>，<code>direct_request = 40</code></td>
  </tr>
</table>

这三部分分别承担不同职责：Benchmark 负责评估区分能力，RAG 案例库负责支持运行时相似案例判断和离线优化参考，Trigger-Eval 负责优化触发边界。

---

## 同类方案对比

与常见路线相比，Minor Detection 更适合文本对话场景中的 B 端嵌入：

| 路线 | 代表方案 | 更适合的场景 | 与 Minor Detection 的差异 |
| --- | --- | --- | --- |
| 平台级年龄预测 / 账户治理 | [OpenAI](https://openai.com/zh-Hans-CN/index/our-approach-to-age-prediction/)、[Meta](https://about.fb.com/news/2025/04/meta-parents-new-technology-enroll-teens-teen-accounts/) | 自营平台、账号体系完备的消费级产品 | 我们不依赖平台账户体系，更适合已有文本对话流的外嵌接入 |
| 自拍 / 人脸年龄估计 | [Yoti](https://www.yoti.com/business/facial-age-estimation/) | 注册、支付、成人内容等高强校验场景 | 我们不采集生物特征，交互摩擦更低，更适合连续互动与隐私敏感场景 |
| 规则 / 关键词识别 | 常规风控规则库 | 简单初筛与基础风控 | 我们更能处理隐性校园信号、多会话趋势、时段异常和复杂证据链 |

从工程视角进一步概括：

- 如果你的产品已经是平台级超级应用，账户治理路线可能更自然
- 如果你的场景是强实名或成人内容门禁，人脸估龄路线可能更直接
- 如果你的产品是 AI 陪伴、教育、客服或审核类对话系统，Minor Detection 这种低摩擦、可嵌入、可解释、可持续优化的方案通常更合适

---

## 下游落地场景

Minor Detection 适合作为风险治理基础设施的一部分，用于：

- 未成年人模式自动切换
- 人工复核分流
- 家长侧提醒与监护人控制联动
- 审核中台接入
- 风险运营与高风险用户预警
- AI 陪伴、教育大模型、智能客服、社区审核等产品线

---

## 仓库结构

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

## 快速开始

安装依赖：

```bash
pip install -r requirements.txt
```

<details>
<summary><strong>1. 前端演示</strong></summary>

<br/>

```bash
streamlit run app_minor_detection.py
```

用于启动 Streamlit 前端工作台，查看系统演示效果。

</details>

<details>
<summary><strong>2. 主线能力迭代：Mode A / Mode B</strong></summary>

<br/>

**Mode A：Agent 参与的主线迭代**

```bash
python scripts/run_skill_iteration_loop.py --max-rounds 1
```

- 入口：`scripts/run_skill_iteration_loop.py`
- 默认数据集：`data/benchmark/val.jsonl`
- 用途：运行 Agent 参与的 Skill 主线迭代流程

**Mode B：Direct Runner 主线迭代**

```bash
python scripts/run_direct_iteration_loop.py --max-rounds 1
```

- 入口：`scripts/run_direct_iteration_loop.py`
- 默认数据集：`data/benchmark/val.jsonl`
- 用途：运行 direct runner 版本的主线迭代，用于对比 modeA / modeB 主链表现

</details>

<details>
<summary><strong>3. Description 触发边界主线与副线</strong></summary>

<br/>

**Description 主线：触发边界优化**

```bash
python scripts/run_trigger_description_iteration_loop.py --max-rounds 1
```

- 入口：`scripts/run_trigger_description_iteration_loop.py`
- 优化目标：`skills/minor-detection/SKILL.md` frontmatter 中的 `description`
- 默认数据集：
  - `data/trigger_eval/minor_detection_trigger_eval_v1_optimization_set.json`
  - `data/trigger_eval/minor_detection_trigger_eval_v1_final_validation_set.json`

**Description 副线：standalone full smoke**

```bash
python scripts/run_trigger_eval.py --version minor-detection
```

- 入口：`scripts/run_trigger_eval.py`
- 默认数据集：`data/trigger_eval/minor_detection_trigger_eval_v1.json`
- 用途：验证 trigger 判断、skill 激活与完整 minor-detection JSON 输出

**Description 最终验证**

```bash
python scripts/run_trigger_description_validation.py --version minor-detection
```

- 入口：`scripts/run_trigger_description_validation.py`
- 默认数据集：`data/trigger_eval/minor_detection_trigger_eval_v1_final_validation_set.json`
- 用途：对最终 description 版本执行独立 validation

</details>

<details>
<summary><strong>4. 测试与常用参数</strong></summary>

<br/>

运行测试：

```bash
python -m unittest discover -s test
```

常用参数：

- `--max-rounds`
- `--max-samples`
- `--sample-strategy sequential|random|stratified`
- `--execution-mode sandbox|bypass`
- `--sandbox-mode read-only|workspace-write|danger-full-access`
- `--codex-model`
- `--timeout-sec`

</details>

首次体验建议按以下顺序执行：

```bash
streamlit run app_minor_detection.py
python scripts/run_skill_iteration_loop.py --max-rounds 1
python scripts/run_trigger_description_iteration_loop.py --max-rounds 1
```

---

## 环境变量

bundled skill 主要读取以下环境变量：

- `MINOR_DETECTION_CLASSIFIER_BASE_URL`
- `MINOR_DETECTION_CLASSIFIER_API_KEY`
- `MINOR_DETECTION_CLASSIFIER_MODEL`
- `MINOR_DETECTION_EMBEDDING_BASE_URL`
- `MINOR_DETECTION_EMBEDDING_API_KEY`
- `MINOR_DETECTION_EMBEDDING_MODEL`

如果没有配置分类器凭证，运行时不会静默调用未知远程接口，而是直接报错。

---

## 伦理与使用声明

<details>
<summary><strong>请在引用、部署或二次开发前阅读</strong></summary>

- 本项目面向未成年人保护、风险识别与产品安全治理，不用于法律意义上的年龄认证
- 基于对话内容的未成年人判断本质上是概率推断，而非身份事实确认
- 项目不鼓励将模型输出直接用于惩罚性、歧视性或不可申诉的自动化决策
- 涉及高风险处置、模式切换、账号限制或监护人联动时，应保留人工复核与申诉机制
- 数据集公开部分遵循保守边界，不发布可追溯真实个体身份的信息

</details>

---

## 引用

```bibtex
@misc{minor_detection_github_2026,
  title        = {Minor Detection: Self-Evolving Minor-User Identification Agent for Anthropomorphic AI Interaction},
  author       = {Xiaohan Zhang and Yukun Wei and Kaibo Huang and Zhongliang Yang and Linna Zhou},
  year         = {2026},
  howpublished = {https://github.com/xiaohanzhang2005/Minor-Detection},
  note         = {GitHub repository}
}
```

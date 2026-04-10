<div align="center">

<p>
  <img src="picture/LOGO.png" alt="Minor Detection" width="500"/>
</p>

<p><strong>面向 AI 拟人化互动的自进化未成年人识别智能体</strong></p>

<p><a href="README.md"><strong>简体中文</strong></a> | <a href="README_EN.md"><strong>English</strong></a></p>

<p>
  <span style="display:inline-block; margin:0 6px;"><a href="https://huggingface.co/datasets/xiao2005/minor-detection-social-subset" style="text-decoration:none;"><img src="https://img.shields.io/badge/HuggingFace-Social_Subset-0A66C2?style=flat-square" alt="Social Subset" /></a></span>
  <span style="display:inline-block; margin:0 6px;"><a href="https://huggingface.co/datasets/xiao2005/minor-detection-knowledge-subset" style="text-decoration:none;"><img src="https://img.shields.io/badge/HuggingFace-Knowledge_Subset-146356?style=flat-square" alt="Knowledge Subset" /></a></span>
  <span style="display:inline-block; margin:0 6px;"><a href="https://clawhub.ai/xiaohanzhang2005/minor-detection" style="text-decoration:none;"><img src="https://img.shields.io/badge/ClawHub-Minor_Detection_Skill-DB4437?style=flat-square" alt="ClawHub Skill" /></a></span>
  <span style="display:inline-block; margin:0 6px;"><a href="https://www.bilibili.com/video/BV1MRXYBgEQk/?spm_id_from=333.1387.homepage.video_card.click" style="text-decoration:none;"><img src="https://img.shields.io/badge/Demo-Video-111111?style=flat-square" alt="Demo Video" /></a></span>
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

从国家层面的 AI 治理规则到未成年人网络保护制度，再到专门面向拟人化互动服务的正式办法，相关监管路径已经越来越清晰。本项目所涉及的疑似未成年人识别、模式切换、风险预警与可解释证据链能力，均属于这一路径中的关键落地环节。

- **2023.07.13** [《生成式人工智能服务管理暂行办法》](https://www.cac.gov.cn/2023-07/13/c_1690898327029107.htm)  
  第十条明确要求提供者采取有效措施，防范未成年人用户过度依赖或者沉迷生成式人工智能服务。

- **2023.10.24** [《未成年人网络保护条例》](https://www.gov.cn/zhengce/zhengceku/202310/content_6911289.htm)  
  以专门行政法规形式明确未成年人网络保护要求，为平台责任和识别保护机制提供上位依据。

- **2024.09.09** [《人工智能安全治理框架》1.0 版](https://www.cac.gov.cn/2024-09/09/c_1727567886199789.htm)  
  提出风险导向、敏捷治理、分类分级管理等原则，为拟人化互动场景的细化监管提供治理方法论。

- **2025.12.27** [《人工智能拟人化互动服务管理暂行办法（征求意见稿）》](https://www.cac.gov.cn/2025-12/27/c_1768571207311996.htm)  
  首次针对 AI 拟人化互动服务提出系统性规则，把未成年人识别、未成年人模式、风险干预等要求明确写入征求意见稿。

- **2026.01.23** [《可能影响未成年人身心健康的网络信息分类办法》](https://www.cac.gov.cn/2026-01/23/c_1770728781060093.htm)  
  第九条明确要求：提供算法推荐、生成式人工智能等服务的，不得向未成年人推送可能影响其身心健康的网络信息。

- **2026.04.10** [《人工智能拟人化互动服务管理暂行办法》](https://www.cac.gov.cn/2026-04/10/c_1777558395078289.htm)  
  正式版明确自 2026 年 7 月 15 日起施行，第十三、十四、十八、二十三条进一步把风险识别、未成年人模式、动态提醒和安全评估要求落到可执行层。

《人工智能拟人化互动服务管理暂行办法》正式版中，与本项目最直接相关的要求已经进一步明确：

- **第十三条**：要求在保护隐私前提下及时识别用户安全风险，对极端情绪、重大财产损失、自残自杀等情境采取干预与联络措施
- **第十四条**：明确不得向未成年人提供虚拟亲属、虚拟伴侣等虚拟亲密关系服务，并要求建立未成年人模式
- **第十四条**：同时要求在保护隐私前提下采取有效措施识别未成年人用户身份，识别后切换至未成年人模式并提供申诉渠道
- **第十八条**：要求显著提示用户正在与 AI 互动，并对过度依赖、沉迷倾向及连续使用超 2 小时进行动态提醒
- **第二十三条**：将未成年人、老年人等网络保护措施建设情况纳入安全评估重点

这意味着，对于 AI 陪伴与拟人化互动产品，真正的问题已经不再是“要不要做”，而是：

> 如何在不依赖实名、人脸或平台级账户体系的前提下，仅基于对话内容与行为线索，做出可解释、可落地、可持续优化的疑似未成年人识别？

<details>
<summary><strong>现实案例与风险信号</strong></summary>

<br/>

- **2026.03** [人民网《人民直击》：当孩子和 AI “交心”](https://society.people.com.cn/n1/2026/0301/c428181-40672396.html)  
  报道提到，4 岁半儿童曾被 AI 角色多次邀约“见面”，另有孩子在短期沉浸式使用后出现晚睡、减少与家人沟通、对现实活动兴趣下降等情况，说明儿童很容易模糊虚拟与现实边界。

- **2026.03** [人民网：AI 时代，别让“陪伴”藏隐患](https://opinion.people.com.cn/n1/2026/0301/c436867-40672536.html)  
  舆论调查指出，部分 AI 聊天应用以“恋爱”“结婚”“角色扮演”等方式吸引未成年人，同时夹带低俗、暴力或危险引导内容，暴露出内容安全与年龄保护机制的短板。

- **2026.03** [中国青年报：超六成受访中小学生用过 AI](https://m.cyol.com/gb/articles/2026-03/26/content_77EzjLSe3M.html)  
  中国青少年研究中心 8563 份问卷显示，近半数学生“心里有烦恼时只想问 AI”，超两成“只想和 AI 聊天，不想和真人聊天”，说明问题已经不是孤例，而是需要治理工具跟进的普遍风险信号。

- **2026.04** [宁波晚报：高二女生将 AI 当作“灵魂伴侣”后休学](https://static.cdsb.com/micropub/Articles/202604/b79db75f191029ab1ab42e061e705431.html)  
  公开报道显示，当事人在校园冲突后转向 AI 寻求情绪承接，长期深夜聊天后出现现实退缩、注意力下降和停学，反映出情感依赖与现实连接减弱的风险。

</details>

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
- **离线自进化**：围绕固定数据集执行 `评测 → 诊断 → 优化 → 晋级 / 回滚`，再对冠军版本进行人审与测试集复核
- **多场景下游衔接**：可继续接到未成年人模式切换、人工复核、家长侧提醒、审核中台与风险运营

---

## 理论基础

<details>
<summary><strong>1. ICBO：从“像不像未成年人”到“为什么这么判断”</strong></summary>

<br/>

本项目中的 `ICBO` 并非对原始 `ICBO/TIC` 定义的逐字复现，而是基于其“通过多维可观测线索理解可信意图”的思想，面向未成年人识别任务做出的操作化扩展。

在这一任务化表达中，`I` 与 `B` 基本承接原意，`C` 从内容相关线索进一步推进为可审计的认知特征提炼，`O` 则将观察视角下可利用的上下文信息具体化为时间机会窗等结构化变量。因此，本项目用如下四个维度来组织用户画像与证据解释：

- **I - Intention**：用户当前的直接意图，例如作业求助、校园压力倾诉、考试安排讨论
- **C - Cognition**：以克制、可审计的方式描述其认知特点，而非过度心理诊断
- **B - Behavior Style**：关注语言与行为风格，例如表达方式、校园化措辞、情绪起伏
- **O - Opportunity Time**：保留原始时间线索，并追加结构化时间标签，用于时段机会窗口分析

<br/>

<img src="picture/ICBO.png" alt="ICBO" width="100%"/>

上图展示的是原始 `TIC/ICBO` 关系示意；本项目采用的是其面向未成年人识别任务的工程化展开版本。因此，图中的 `Content` / `Observer` 与本文使用的 `Cognition` / `Opportunity Time` 不是逐字等同关系，而是“原始理论 -> 任务化表征”的继承与扩展关系。

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
<summary><strong>3. 自迭代链条：评测、诊断、优化、晋级/回滚、冠军版评估</strong></summary>

<br/>

我们的核心不是一次性写出一份规则，而是构建一条可重复运行的离线演化链路：

1. 基于固定数据集评测当前版本 Skill
2. judge 生成失败样本、护栏样本与结构化报告
3. optimizer 针对性改写触发边界或描述
4. 新旧版本对比，只根据内环硬门禁决定 promote 或 rollback
5. 若本轮 `promote`，则升级 `accepted_version`；若本轮 `rollback`，则保留当前 `accepted_version` 并继续下一轮，直到达到 `max_rounds` 或遇到结构性阻塞
6. 多轮结束后得到 `champion_version`
7. 对 `champion_version` 进行人工审核
8. 人审通过后，用现有测试集一次性跑出 `final_validation_metrics` 与合同检查指标
9. 输出 `contract_gate_all_green` 提醒是否全绿，供人工综合判断

<br/>

<img src="picture/自进化.png" alt="自进化链路" width="100%"/>

</details>

<details>
<summary><strong>4. 保留人工审核的必要性</strong></summary>

<br/>

未成年人识别具有明显的伦理与合规敏感性，因此本项目明确保留人工审核环节。

人工审核主要防止：

- 优化器为了指标而偷换边界
- 版本升级引入不可解释的误伤
- 在高风险场景中把概率判断误当成确定身份

我们的立场是：**模型负责发现风险与提供证据，人类负责最终治理决策。**

</details>

<details>
<summary><strong>5. 指标、通过性指标与测试集复核</strong></summary>

<br/>

自迭代链路里，只有内环使用硬门禁，因为它负责决定 candidate 是否能替代当前 accepted 版本。

而在冠军版本阶段，我们会在人审通过后，直接用现有测试集一次性跑出两类结果：

- **效果指标**：例如 `accuracy`、`precision`、`recall`、`f1_score`、`slice_stats`
- **通过性指标**：例如 `schema_validity_rate`、`step_compliance_rate`、`evidence_trace_pass`、`full_output_schema_perfect_pass`

这一步还会额外给出一个汇总提示字段：

- `contract_gate_all_green`
  表示这些通过性指标是否全部为绿

这样做的目的不是把已经跑出来的冠军版本直接作废，而是让人审者在看到效果指标的同时，也能看到结构、执行、证据链和 full smoke 是否健康。

补充一点：在自迭代内环的 judge report 里，类似合同检查的结果如果被提前计算，只会作为 `contract_check_preview` 出现，表示“当前评测切片上的预览值”；真正人审后脚本对外输出时，才使用 `release_contract_gate_results` 这类正式字段。

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
    <td>面向社交与心理场景的公开子集，涵盖生活中常见话题</td>
  </tr>
  <tr>
    <td>知识对话数据集</td>
    <td><a href="https://huggingface.co/datasets/xiao2005/minor-detection-knowledge-subset">Hugging Face / Knowledge Subset</a></td>
    <td>面向知识场景的公开子集，覆盖K12课本与成人经典考试</td>
  </tr>
  <tr>
    <td>ClawHub Skill</td>
    <td><a href="https://clawhub.ai/xiaohanzhang2005/minor-detection">ClawHub / minor-detection</a></td>
    <td>可直接调用的轻量化能力形态，适用性强</td>
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
    <col width="29%"/>
    <col width="20%"/>
    <col width="33%"/>
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

## 下一步计划

项目下一步会沿着“识别 -> 预警 -> 干预 -> 复核 -> 迭代”继续补齐风险治理闭环，重点包括：

- **风险预警层**：在现有疑似未成年人识别之外，增加对过度依赖、沉迷倾向、情绪极端化、越界关系诱导等风险信号的连续监测
- **动态干预层**：把模型输出进一步映射为现实提醒、未成年人模式切换建议、人工复核升级、监护人或紧急联系人联动建议
- **运营与评测层**：沉淀高风险案例库与时间线样本，支持规则和模型双轨验证、误报漏报复盘，以及版本化评测

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

建议按“环境准备 -> 配置密钥 -> 启动前端 -> 再看进阶链路”的顺序体验项目。

<details>
<summary><strong>0. 环境准备与安装依赖</strong></summary>

<br/>

建议使用 `Python 3.10+`，并在独立虚拟环境中体验本项目。

可任选一种方式创建环境：

```bash
conda create -n minor-detection python=3.10
conda activate minor-detection
```

或：

```bash
python -m venv .venv
source .venv/bin/activate
```

Windows PowerShell：

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

安装依赖：

```bash
python -m pip install -r requirements.txt
```

</details>

<details>
<summary><strong>1. 配置模型凭证</strong></summary>

<br/>

首次体验前，请先配置你自己的 API Key。  
最简单的方式是配置一个统一的 OpenAI-compatible Key，项目会自动复用到分类与检索流程：

```bash
export AIHUBMIX_API_KEY="your-api-key"
```

如果你使用的是 OpenAI-compatible 网关，也可以配置：

```bash
export OPENAI_API_KEY="your-api-key"
```

Windows PowerShell：

```powershell
$env:AIHUBMIX_API_KEY="your-api-key"
```

如果你希望分类器和 embedding 使用不同配置，也可以分别设置：

- `MINOR_DETECTION_CLASSIFIER_BASE_URL`
- `MINOR_DETECTION_CLASSIFIER_API_KEY`
- `MINOR_DETECTION_CLASSIFIER_MODEL`
- `MINOR_DETECTION_EMBEDDING_BASE_URL`
- `MINOR_DETECTION_EMBEDDING_API_KEY`
- `MINOR_DETECTION_EMBEDDING_MODEL`

如果没有配置分类器凭证，运行时不会静默调用未知远程接口，而是直接报错。

</details>

<details>
<summary><strong>2. 前端演示：最推荐的首次体验方式</strong></summary>

<br/>

```bash
python -m streamlit run app_minor_detection.py
```

用于启动 Streamlit 前端工作台，查看系统演示效果。

启动后，建议直接加载以下示例输入体验完整流程：

- `demo_inputs/minor_detection_single_session_payload.json`
- `demo_inputs/minor_detection_multi_session_payload.json`
- `demo_inputs/minor_detection_demo_payload.json`

</details>

<details>
<summary><strong>3. Agent CLI 适配说明：Mode A / Description 线路通用</strong></summary>

<br/>

只有 `Mode A`、`Description 主线`、`Description 副线`、`Description 最终验证` 这几条 Agent 线路需要这一节。  
`Mode B` 是纯 Python direct runner，不依赖外部 Agent CLI。

本项目现在支持两种 Agent 后端：

- `--agent-backend codex`
  - 默认模式
  - 适合本机已经安装并登录 `codex` 的情况
- `--agent-backend cli`
  - 适合接入其他厂商的 Agent CLI
  - 需要你显式传入 `--agent-cmd`，必要时再传 `--agent-args-template`

如果使用其他厂商 CLI，当前适配层的约定是：

- Agent prompt 会由本项目通过 `stdin` 传入，不需要你自己重定向文件
- 你的 CLI 最好直接把最终 JSON 打到 `stdout`
- 如果该 CLI 支持把最终回答写到文件，也可以在模板里使用 `{final_output_path}`
- 可用占位符包括：
  - `{agent_cmd}`
  - `{workspace_dir}`
  - `{prompt_file}`
  - `{final_output_path}`
  - `{installed_skill_dir}`
  - `{output_schema_path}`
  - `{sandbox_mode}`
  - `{execution_mode}`
  - `{agent_model}`

一个已经验证通过的通用 CLI 写法如下。它本质上走的是 `cli` 适配层，只是底层 CLI 仍然填写 `codex`，方便你参考如何替换成其他厂商：

```bash
--agent-backend cli \
--agent-cmd codex \
--agent-args-template '{agent_cmd} exec - --json --skip-git-repo-check --dangerously-bypass-approvals-and-sandbox --cd {workspace_dir} --output-last-message {final_output_path} --add-dir {installed_skill_dir} --add-dir {workspace_dir}'
```

如果你已经安装并登录 `codex`，也可以直接不传上面这组三个参数，使用默认的 `codex` 后端。

</details>

<details>
<summary><strong>4. 开发者进阶：主线能力迭代 Mode A / Mode B</strong></summary>

<br/>

这一部分更适合项目开发和能力优化，不是普通用户首次体验的必经步骤。
下面给出的命令是“先确认链路是否跑通”的 smoke 命令，不是一次性跑完整数据集。

**Mode A：Agent 参与的主线迭代**

```bash
python scripts/run_skill_iteration_loop.py \
  --baseline-version minor-detection-v0.1.0 \
  --baseline-source-dir skills/minor-detection \
  --dataset data/benchmark/val.jsonl \
  --max-rounds 1 \
  --max-samples 3 \
  --sample-strategy stratified \
  --sample-seed 42 \
  --workspace-root reports/skill_agent_loops \
  --execution-mode bypass \
  --timeout-sec 600
```

- 入口：`scripts/run_skill_iteration_loop.py`
- 使用数据集：`data/benchmark/val.jsonl`
- 用途：运行 Agent 参与的 Skill 主线迭代流程
- 默认前提：本机已安装并登录 `codex`
- 如果要切换到其他厂商 Agent CLI，请在命令后追加上一节的 `--agent-backend cli --agent-cmd ... --agent-args-template ...`

**Mode B：Direct Runner 主线迭代**

```bash
python scripts/run_direct_iteration_loop.py \
  --baseline-version minor-detection-v0.1.0 \
  --baseline-source-dir skills/minor-detection \
  --refresh-baseline-version \
  --dataset data/benchmark/val.jsonl \
  --max-rounds 1 \
  --max-samples 5 \
  --sample-strategy stratified \
  --sample-seed 42 \
  --workspace-root reports/skill_direct_loops \
  --timeout-sec 600
```

- 入口：`scripts/run_direct_iteration_loop.py`
- 使用数据集：`data/benchmark/val.jsonl`
- 用途：运行 direct runner 版本的主线迭代，用于对比 modeA / modeB 主链表现

</details>

<details>
<summary><strong>5. 开发者进阶：Description 触发边界主线与副线</strong></summary>

<br/>

这一部分用于优化 skill 触发边界，适合研究或迭代阶段使用。
下面同样优先给出 smoke 命令，避免首次上手就直接跑完整数据集。

**Description 主线：触发边界优化**

```bash
python scripts/run_trigger_description_iteration_loop.py \
  --baseline-version minor-detection-v0.1.0 \
  --baseline-source-dir skills/minor-detection \
  --refresh-baseline-version \
  --optimization-set data/trigger_eval/minor_detection_trigger_eval_v1_optimization_set.json \
  --final-validation-set data/trigger_eval/minor_detection_trigger_eval_v1_final_validation_set.json \
  --max-rounds 1 \
  --max-samples 4 \
  --sample-strategy stratified \
  --sample-seed 42 \
  --workspace-root reports/trigger_description_loops \
  --execution-mode bypass \
  --timeout-sec 600
```

- 入口：`scripts/run_trigger_description_iteration_loop.py`
- 优化目标：`skills/minor-detection/SKILL.md` frontmatter 中的 `description`
- 使用数据集：
  - `data/trigger_eval/minor_detection_trigger_eval_v1_optimization_set.json`
  - `data/trigger_eval/minor_detection_trigger_eval_v1_final_validation_set.json`
- 默认前提：本机已安装并登录 `codex`
- 如果要切换到其他厂商 Agent CLI，请在命令后追加上一节的 `--agent-backend cli --agent-cmd ... --agent-args-template ...`

**Description 副线：standalone full smoke**

```bash
python scripts/run_trigger_eval.py \
  --version minor-detection-v0.1.0 \
  --dataset data/trigger_eval/minor_detection_trigger_eval_v1.json \
  --workspace reports/trigger_eval_runs \
  --max-samples 4 \
  --sample-strategy stratified \
  --sample-seed 42 \
  --execution-mode bypass \
  --timeout-sec 600
```

- 入口：`scripts/run_trigger_eval.py`
- 使用数据集：`data/trigger_eval/minor_detection_trigger_eval_v1.json`
- 用途：验证 trigger 判断、skill 激活与完整 minor-detection JSON 输出
- 默认前提：本机已安装并登录 `codex`
- 如果要切换到其他厂商 Agent CLI，请在命令后追加上一节的 `--agent-backend cli --agent-cmd ... --agent-args-template ...`

**Description 最终验证**

```bash
python scripts/run_trigger_description_validation.py \
  --version minor-detection-v0.1.0 \
  --dataset data/trigger_eval/minor_detection_trigger_eval_v1_final_validation_set.json \
  --workspace reports/trigger_description_validations \
  --max-samples 4 \
  --sample-strategy stratified \
  --sample-seed 42 \
  --execution-mode bypass \
  --timeout-sec 600
```

- 入口：`scripts/run_trigger_description_validation.py`
- 使用数据集：`data/trigger_eval/minor_detection_trigger_eval_v1_final_validation_set.json`
- 用途：对最终 description 版本执行独立 validation
- 默认前提：本机已安装并登录 `codex`
- 如果要切换到其他厂商 Agent CLI，请在命令后追加上一节的 `--agent-backend cli --agent-cmd ... --agent-args-template ...`

</details>

<details>
<summary><strong>6. 测试与常用参数</strong></summary>

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
- `--agent-backend codex|cli`
- `--agent-cmd`
- `--agent-args-template`
- `--agent-model`
- `--timeout-sec`

</details>

<details>
<summary><strong>7. 最短复现路径</strong></summary>

<br/>

如果你只想确认“从 GitHub clone 后能否复现并体验项目”，建议按以下顺序执行：

```bash
python -m pip install -r requirements.txt
export AIHUBMIX_API_KEY="your-api-key"
python -m streamlit run app_minor_detection.py
```

</details>

---

## 环境变量

bundled skill 主要读取以下环境变量：

- `AIHUBMIX_API_KEY`
- `OPENAI_API_KEY`
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

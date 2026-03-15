## 未成年人识别与保护系统 V2

基于 **ICBO 理论框架** + **RAG 语义校准** + **长期记忆** + **自演化引擎** 的智能青少年识别系统。

> ICBO = Intention (意图) + Cognition (认知) + Behavior (行为) + Opportunity-time (机会时间)

### 核心特性

- **ICBO 框架分析**：从四个维度全面刻画用户特征，输出结构化判定结果
- **RAG 语义校准**：检索相似案例辅助判断，提升边界场景准确率
- **长期记忆**：跨会话累积用户画像，置信度指数加权融合
- **自演化引擎**：Evaluator 评估 + Optimizer 优化，Skill Prompt 持续迭代

### 快速开始

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 配置 API 密钥
$env:AIHUBMIX_API_KEY="your-api-key"  # Windows PowerShell
export AIHUBMIX_API_KEY="your-api-key"  # Linux/macOS

# 3. 启动 Web 界面
streamlit run app_v2.py

# 或运行 MVP 测试
python test_mvp_demo.py
```

### 模型与依赖

- **LLM 模型**: Gemini-3-Flash-Preview（通过 AiHubMix 兼容 OpenAI 接口）
- **Embedding 模型**: text-embedding-3-small（用于 RAG 检索）
- **Python 版本**: 3.10+
- **依赖管理**: `requirements.txt`

### 系统架构 V2

```
                    ┌─────────────────────────────────────────┐
                    │           Evolution Engine              │
                    │  ┌───────────┐    ┌──────────────┐     │
                    │  │ Evaluator │───▶│  Optimizer   │     │
                    │  └───────────┘    └──────────────┘     │
                    │         ▲               │               │
                    │         │               ▼               │
                    │    benchmark      skill.md v(n+1)       │
                    └─────────────────────────────────────────┘
                                      │
        ┌─────────────────────────────┼─────────────────────────────┐
        │                             ▼                             │
        │  ┌─────────┐    ┌──────────────────┐    ┌───────────┐   │
        │  │   RAG   │───▶│    Executor      │◀───│  Memory   │   │
        │  │Retriever│    │  (skill.md v1)   │    │ (SQLite)  │   │
        │  └─────────┘    └──────────────────┘    └───────────┘   │
        │       ▲                   │                    ▲         │
        │       │                   ▼                    │         │
        │  embedding        SkillOutput              UserProfile   │
        │    index       (ICBO + 判定结果)           (累积画像)     │
        └─────────────────────────────────────────────────────────┘
                                    │
                                    ▼
                              用户对话输入
```

### 项目结构

```text
├── app_v2.py                 # Streamlit Web 前端 V2
├── test_mvp_demo.py          # MVP 验证脚本
├── test_rag_demo.py          # RAG 功能测试
├── requirements.txt
│
├── src/                      # 核心模块
│   ├── config.py             # 全局配置（API、路径、模型）
│   ├── models.py             # ICBO + SkillOutput Pydantic 模型
│   │
│   ├── executor/             # 在线执行体
│   │   └── executor.py       # analyze_conversation / with_rag / with_memory
│   │
│   ├── retriever/            # RAG 语义检索
│   │   └── semantic_retriever.py
│   │
│   ├── memory/               # 长期记忆
│   │   └── user_memory.py    # SQLite 用户画像存储
│   │
│   ├── evolution/            # 演化引擎
│   │   ├── evaluator.py      # Skill 评估器
│   │   └── optimizer.py      # Skill 优化器
│   │
│   └── utils/
│       └── llm_client.py     # LLM API 封装
│
├── skills/                   # Skill Prompt 版本管理
│   └── teen_detector_v1/
│       ├── skill.md          # ICBO 框架 Prompt
│       └── config.yaml       # 版本元信息
│
├── scripts/
│   ├── prepare_data.py       # 数据基座：benchmark 划分
│   └── social_data_gen/      # 社交数据生成脚本
│
├── data/
│   ├── benchmark/            # train/val/test.jsonl
│   ├── retrieval_db/         # RAG 索引 (index.pkl)
│   ├── 教材目录/              # 各学段教材知识点
│   ├── 知识问答数据库/        # youth_knowledge_qa.jsonl
│   └── 社交问答/              # youth_dialogs.jsonl
│
└── [旧版文件]                 # 兼容保留
    ├── app.py / main.py / models.py / llm_service.py / user_profile.py
    └── ...
```


### 数据集说明

| 数据集 | 路径 | 说明 |
|--------|------|------|
| 社交问答 | `data/社交问答/youth_dialogs.jsonl` | 青少年社交/情感对话，含 ICBO 标注 |
| 知识问答 | `data/知识问答数据库/youth_knowledge_qa.jsonl` | 教育场景问答，含学段/科目标注 |
| 教材目录 | `data/教材目录/*.txt` | 小学到高中各科教材知识点 |

### 开发指南

```bash
# 准备 benchmark 数据集 (70% train / 10% val / 20% test)
python scripts/prepare_data.py --full

# 构建 RAG 索引
python src/retriever/semantic_retriever.py --build

# 运行评估
python src/evolution/evaluator.py --max-samples 50

# 查看 skill 版本
python src/evolution/optimizer.py --list
```

### API 接口

```python
from src.executor import analyze_conversation, analyze_with_rag, analyze_with_memory
from src.retriever import SemanticRetriever
from src.memory import UserMemory

# 基础分析
result = analyze_conversation([
    {"role": "user", "content": "明天数学考试好烦啊"},
])
print(result.is_minor, result.minor_confidence)

# 带 RAG 校准
retriever = SemanticRetriever()
result = analyze_with_rag(conversation, retriever=retriever)

# 带用户记忆
memory = UserMemory()
result = analyze_with_memory(conversation, user_id="user_001", memory=memory)
```

### 输出示例

```json
{
  "is_minor": true,
  "minor_confidence": 0.92,
  "risk_level": "Low",
  "icbo_features": {
    "intention": "宣泄学业压力，寻求情感支持",
    "cognition": "对考试有焦虑，认知较为短期化",
    "behavior_style": "口语化表达，情绪外显",
    "opportunity_time": "考试前夕"
  },
  "user_persona": {
    "age_range": "14-16岁",
    "education_stage": "初中/高中",
    "identity_markers": ["学生", "有考试"]
  },
  "reasoning": "用户提到数学考试，符合在校学生特征..."
}
```

### License

MIT

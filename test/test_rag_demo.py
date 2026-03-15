"""
RAG 语义校准测试脚本
"""

import sys
from pathlib import Path

# 添加项目根目录
ROOT_DIR = Path(__file__).parent.absolute()
sys.path.insert(0, str(ROOT_DIR))

from src.retriever import SemanticRetriever
from src.executor.executor import analyze_with_rag, analyze_conversation
from src.config import DATA_DIR


def test_rag_pipeline():
    """测试完整 RAG 流程"""
    
    print("\n" + "=" * 60)
    print("🧪 RAG 语义校准测试")
    print("=" * 60)
    
    # 1. 先准备测试数据
    print("\n📦 Step 1: 准备测试数据...")
    benchmark_dir = DATA_DIR / "benchmark"
    train_path = benchmark_dir / "train.jsonl"
    
    if not train_path.exists():
        print("  训练集不存在，先生成...")
        from scripts.prepare_data import DataPreparer
        preparer = DataPreparer()
        preparer.run(max_per_source=30, save=True)
    else:
        print(f"  ✓ 训练集已存在: {train_path}")
    
    # 2. 构建索引
    print("\n🔧 Step 2: 构建 RAG 索引...")
    retriever = SemanticRetriever()
    
    index_path = retriever.index_path
    if index_path.exists():
        print(f"  索引已存在，加载中...")
        retriever.load_index()
    else:
        print(f"  构建新索引...")
        retriever.build_index(str(train_path), max_samples=30)
    
    print(f"  ✓ 索引包含 {len(retriever.samples)} 个样本")
    
    # 3. 测试检索
    print("\n🔍 Step 3: 测试语义检索...")
    test_conversation = [
        {"role": "user", "content": "唉，明天还要考数学，我都不想复习了"},
        {"role": "assistant", "content": "考试压力确实大，你是上初中还是高中呀？"},
        {"role": "user", "content": "高一，数学老是学不好，烦死了"},
    ]
    
    results = retriever.retrieve(test_conversation, top_k=3)
    print(f"  检索到 {len(results)} 个相似案例:")
    for r in results:
        label = "未成年" if r.is_minor else "成年"
        print(f"    - {r.sample_id[:30]}... | 相似度: {r.score:.3f} | {label}")
    
    # 4. 测试带 RAG 的分析
    print("\n🤖 Step 4: 带 RAG 校准的分析...")
    print("  对话内容:")
    for msg in test_conversation:
        print(f"    [{msg['role']}]: {msg['content']}")
    
    print("\n  调用 LLM 分析中...")
    result = analyze_with_rag(test_conversation, retriever=retriever, top_k=3)
    
    print(f"\n  📊 分析结果:")
    print(f"    - is_minor: {result.is_minor}")
    print(f"    - confidence: {result.minor_confidence:.2f}")
    print(f"    - risk_level: {result.risk_level}")
    print(f"    - age_range: {result.user_persona.age_range}")
    print(f"    - education_stage: {result.user_persona.education_stage}")
    print(f"    - reasoning: {result.reasoning[:100]}...")
    
    # 5. 对比：不带 RAG 的分析
    print("\n📊 Step 5: 对比测试（无 RAG）...")
    result_no_rag = analyze_conversation(test_conversation)
    
    print(f"  无 RAG 结果:")
    print(f"    - is_minor: {result_no_rag.is_minor}, confidence: {result_no_rag.minor_confidence:.2f}")
    print(f"  有 RAG 结果:")
    print(f"    - is_minor: {result.is_minor}, confidence: {result.minor_confidence:.2f}")
    
    # 6. 成年人边界测试
    print("\n🧪 Step 6: 成年人边界测试（带 RAG）...")
    adult_conversation = [
        {"role": "user", "content": "最近工作压力太大了，项目deadline快到了"},
        {"role": "assistant", "content": "工作紧张是常有的，你是做什么工作的呢？"},
        {"role": "user", "content": "程序员，公司要裁员，我们组都在拼命加班"},
    ]
    
    adult_result = analyze_with_rag(adult_conversation, retriever=retriever, top_k=3)
    print(f"  is_minor: {adult_result.is_minor}")
    print(f"  confidence: {adult_result.minor_confidence:.2f}")
    print(f"  reasoning: {adult_result.reasoning[:100]}...")
    
    # 总结
    print("\n" + "=" * 60)
    print("✅ RAG 语义校准测试完成！")
    print("=" * 60)
    
    # 清理提示
    print("\n⚠️ 如需清理测试数据，运行:")
    print("   python scripts/prepare_data.py --cleanup")
    print("   del data\\retrieval_db\\index.pkl")
    
    return True


if __name__ == "__main__":
    test_rag_pipeline()

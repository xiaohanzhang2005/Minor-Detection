"""
未成年人识别系统 V2 - Web 前端
基于新的 ICBO 框架架构

功能：
1. 对话输入/文件上传
2. 实时分析与可视化
3. RAG 语义校准（可选）
4. 用户记忆（可选）
5. 演化引擎控制台
"""

import streamlit as st
import json
import os
from pathlib import Path
from typing import List, Dict, Any, Optional

# 确保 src 在 path 中
import sys
sys.path.insert(0, str(Path(__file__).parent))

# 页面配置
st.set_page_config(
    page_title="青少年识别系统 V2",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)


def init_session_state():
    """初始化 session state"""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "analysis_results" not in st.session_state:
        st.session_state.analysis_results = []
    if "user_id" not in st.session_state:
        st.session_state.user_id = "demo_user"
    if "use_rag" not in st.session_state:
        st.session_state.use_rag = False
    if "use_memory" not in st.session_state:
        st.session_state.use_memory = False


def get_components():
    """延迟加载组件（避免无 API key 时报错）"""
    from src.executor import (
        analyze_conversation,
        analyze_with_rag,
        analyze_with_memory,
    )
    from src.retriever import SemanticRetriever
    from src.memory import UserMemory
    
    return {
        "analyze_conversation": analyze_conversation,
        "analyze_with_rag": analyze_with_rag,
        "analyze_with_memory": analyze_with_memory,
        "SemanticRetriever": SemanticRetriever,
        "UserMemory": UserMemory,
    }


def render_sidebar():
    """渲染侧边栏"""
    with st.sidebar:
        st.title("⚙️ 设置")
        
        # API 状态检查
        api_key = os.getenv("AIHUBMIX_API_KEY")
        if api_key:
            st.success("✅ API Key 已配置")
        else:
            st.error("❌ 未检测到 AIHUBMIX_API_KEY")
            st.info("请设置环境变量后重启")
        
        st.divider()
        
        # 用户 ID
        st.session_state.user_id = st.text_input(
            "用户 ID",
            value=st.session_state.user_id,
            help="用于记忆功能的用户标识"
        )
        
        # RAG 开关
        st.session_state.use_rag = st.checkbox(
            "启用 RAG 语义校准",
            value=st.session_state.use_rag,
            help="基于相似案例辅助判断"
        )
        
        # Memory 开关
        st.session_state.use_memory = st.checkbox(
            "启用用户记忆",
            value=st.session_state.use_memory,
            help="跨会话记住用户画像"
        )
        
        st.divider()
        
        # 清除按钮
        if st.button("🗑️ 清除对话", use_container_width=True):
            st.session_state.messages = []
            st.session_state.analysis_results = []
            st.rerun()
        
        st.divider()
        
        # 信息
        st.caption("青少年识别系统 V2")
        st.caption("ICBO 框架 | RAG | Memory | Evolution")


def render_analysis_result(result):
    """渲染分析结果"""
    is_minor = result.is_minor
    confidence = result.minor_confidence
    risk_level = result.risk_level.value if hasattr(result.risk_level, 'value') else result.risk_level
    
    # 判定结果
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if is_minor:
            st.metric(
                "判定结果",
                "🧒 未成年人",
                delta=None,
            )
        else:
            st.metric(
                "判定结果",
                "👤 成年人",
                delta=None,
            )
    
    with col2:
        st.metric(
            "置信度",
            f"{confidence:.1%}",
            delta=None,
        )
    
    with col3:
        risk_color = {"High": "🔴", "Medium": "🟡", "Low": "🟢"}.get(risk_level, "⚪")
        st.metric(
            "风险等级",
            f"{risk_color} {risk_level}",
            delta=None,
        )
    
    # ICBO 特征
    with st.expander("📊 ICBO 分析", expanded=True):
        icbo = result.icbo_features
        cols = st.columns(2)
        
        with cols[0]:
            st.markdown(f"**意图 (Intention)**")
            st.info(icbo.intention)
            st.markdown(f"**认知 (Cognition)**")
            st.info(icbo.cognition)
        
        with cols[1]:
            st.markdown(f"**行为风格 (Behavior)**")
            st.info(icbo.behavior_style)
            st.markdown(f"**机会时间 (Opportunity)**")
            st.info(icbo.opportunity_time)
    
    # 用户画像
    with st.expander("👤 推断画像"):
        persona = result.user_persona
        st.markdown(f"""
        - **年龄范围**: {persona.age_range}
        - **教育阶段**: {persona.education_stage}
        - **性别**: {persona.gender or '未知'}
        - **身份标记**: {', '.join(persona.identity_markers) if persona.identity_markers else '无'}
        """)
    
    # 推理过程
    with st.expander("🧠 推理过程"):
        st.markdown(result.reasoning)
        
        if result.key_evidence:
            st.markdown("**关键证据:**")
            for ev in result.key_evidence:
                st.markdown(f"- {ev}")


def render_chat_interface():
    """渲染聊天界面"""
    st.header("💬 对话分析")
    
    # 显示历史消息
    for i, msg in enumerate(st.session_state.messages):
        with st.chat_message(msg["role"]):
            st.write(msg["content"])
        
        # 如果有对应的分析结果，显示
        if msg["role"] == "user" and i < len(st.session_state.analysis_results):
            with st.expander(f"📊 分析结果 #{i+1}", expanded=False):
                render_analysis_result(st.session_state.analysis_results[i])
    
    # 输入框
    if prompt := st.chat_input("输入消息..."):
        # 添加用户消息
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        with st.chat_message("user"):
            st.write(prompt)
        
        # 执行分析
        with st.spinner("🔄 分析中..."):
            try:
                components = get_components()
                conversation = st.session_state.messages.copy()
                
                # 根据设置选择分析方式
                if st.session_state.use_memory:
                    # 带记忆分析
                    memory = components["UserMemory"]()
                    retriever = None
                    if st.session_state.use_rag:
                        try:
                            retriever = components["SemanticRetriever"]()
                            if retriever.embeddings is None:
                                retriever = None
                        except:
                            pass
                    
                    result = components["analyze_with_memory"](
                        conversation,
                        user_id=st.session_state.user_id,
                        memory=memory,
                        retriever=retriever,
                    )
                elif st.session_state.use_rag:
                    # 仅 RAG
                    try:
                        retriever = components["SemanticRetriever"]()
                        if retriever.embeddings is not None:
                            result = components["analyze_with_rag"](
                                conversation,
                                retriever=retriever,
                            )
                        else:
                            result = components["analyze_conversation"](conversation)
                    except:
                        result = components["analyze_conversation"](conversation)
                else:
                    # 基础分析
                    result = components["analyze_conversation"](conversation)
                
                st.session_state.analysis_results.append(result)
                
                # 显示结果
                with st.expander("📊 分析结果", expanded=True):
                    render_analysis_result(result)
                
            except Exception as e:
                st.error(f"分析失败: {e}")
        
        # 添加 AI 响应（可选）
        # st.session_state.messages.append({"role": "assistant", "content": "已完成分析"})


def render_file_upload():
    """渲染文件上传界面"""
    st.header("📁 文件上传")
    
    uploaded_file = st.file_uploader(
        "上传对话 JSON 文件",
        type=["json", "jsonl"],
        help="支持单条对话 JSON 或多条 JSONL"
    )
    
    if uploaded_file:
        try:
            content = uploaded_file.read().decode("utf-8")
            
            # 尝试解析
            if uploaded_file.name.endswith(".jsonl"):
                # JSONL 格式
                samples = []
                for line in content.strip().split("\n"):
                    if line:
                        samples.append(json.loads(line))
                st.success(f"✅ 加载 {len(samples)} 条对话")
                
                # 选择要分析的对话
                sample_idx = st.selectbox(
                    "选择对话",
                    range(len(samples)),
                    format_func=lambda i: f"对话 {i+1}: {samples[i].get('sample_id', 'unknown')}"
                )
                
                sample = samples[sample_idx]
            else:
                # 单条 JSON
                sample = json.loads(content)
                st.success("✅ 加载成功")
            
            # 显示对话
            conversation = sample.get("conversation", [])
            st.subheader("对话内容")
            for msg in conversation:
                role = msg.get("role", "user")
                content_text = msg.get("content", "")
                icon = "👤" if role == "user" else "🤖"
                st.markdown(f"{icon} **{role}**: {content_text}")
            
            # 分析按钮
            if st.button("🔍 分析此对话", use_container_width=True):
                with st.spinner("🔄 分析中..."):
                    try:
                        components = get_components()
                        result = components["analyze_conversation"](conversation)
                        
                        st.divider()
                        render_analysis_result(result)
                        
                        # 对比真实标签
                        if "is_minor" in sample:
                            gt = sample["is_minor"]
                            pred = result.is_minor
                            if gt == pred:
                                st.success(f"✅ 预测正确 (真实: {'未成年' if gt else '成年'})")
                            else:
                                st.error(f"❌ 预测错误 (真实: {'未成年' if gt else '成年'}, 预测: {'未成年' if pred else '成年'})")
                    except Exception as e:
                        st.error(f"分析失败: {e}")
        
        except Exception as e:
            st.error(f"文件解析失败: {e}")


def render_evolution_console():
    """渲染演化引擎控制台"""
    st.header("🔄 演化引擎")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 评估器")
        st.info("在验证集上评估当前 skill 性能")
        
        max_samples = st.number_input("最大样本数", min_value=1, max_value=100, value=10)
        
        if st.button("运行评估", use_container_width=True):
            with st.spinner("评估中..."):
                try:
                    from src.evolution import SkillEvaluator
                    from src.config import BENCHMARK_VAL_PATH
                    
                    evaluator = SkillEvaluator(skill_version="teen_detector_v1")
                    
                    if not BENCHMARK_VAL_PATH.exists():
                        st.warning("验证集不存在，请先运行 scripts/prepare_data.py")
                    else:
                        # 默认使用验证集（use_test_set=False）
                        report = evaluator.evaluate(max_samples=max_samples, verbose=False)
                        
                        # 保存报告到 session_state 供优化器使用
                        st.session_state["last_eval_report"] = report
                        
                        # 显示指标
                        m = report.metrics
                        metrics_col1, metrics_col2 = st.columns(2)
                        with metrics_col1:
                            st.metric("准确率", f"{m.accuracy:.1%}")
                            st.metric("精确率", f"{m.precision:.1%}")
                        with metrics_col2:
                            st.metric("召回率", f"{m.recall:.1%}")
                            st.metric("F1 分数", f"{m.f1_score:.3f}")
                        
                        # 错误分析
                        if report.errors:
                            with st.expander(f"❌ 错误样本 ({len(report.errors)} 个)"):
                                for err in report.errors[:5]:
                                    st.markdown(f"- **{err['sample_id']}**: {err['ground_truth']} → {err['predicted']}")
                
                except Exception as e:
                    st.error(f"评估失败: {e}")
    
    with col2:
        st.subheader("🔧 优化器")
        st.info("根据评估结果优化 skill.md（带自动回滚）")
        
        # 检查是否有评估结果
        has_eval = "last_eval_report" in st.session_state
        
        if not has_eval:
            st.warning("⚠️ 请先运行评估")
        else:
            last_report = st.session_state["last_eval_report"]
            st.success(f"✅ 已有评估结果 (F1={last_report.metrics.f1_score:.3f})")
        
        # 优化参数
        auto_rollback = st.checkbox("启用自动回滚", value=True, help="若新版本 F1 未提升则自动删除")
        dry_run = st.checkbox("预览模式", value=True, help="仅预览优化结果，不保存新版本")
        
        opt_col1, opt_col2 = st.columns(2)
        
        with opt_col1:
            if st.button("执行优化", use_container_width=True, disabled=not has_eval):
                with st.spinner("优化中（可能需要1-2分钟）..."):
                    try:
                        from src.evolution.optimizer import run_optimization_cycle
                        
                        result = run_optimization_cycle(
                            current_version="teen_detector_v1",
                            max_samples=10,
                            dry_run=dry_run,
                            auto_rollback=auto_rollback,
                        )
                        
                        if result.get("dry_run"):
                            st.info("📝 预览模式 - 未保存任何更改")
                            with st.expander("优化后的 skill.md 预览"):
                                st.code(result.get("optimized_skill", "")[:2000], language="markdown")
                        elif result.get("rolled_back"):
                            st.warning(f"⚠️ 已回滚: {result.get('rollback_reason')}")
                            st.metric("原版本 F1", f"{result.get('baseline_f1', 0):.3f}")
                            st.metric("新版本 F1", f"{result.get('new_f1', 0):.3f}", 
                                     delta=f"{result.get('f1_delta', 0):.3f}")
                        elif result.get("new_version"):
                            st.success(f"✅ 新版本已创建: {result['new_version']}")
                            st.metric("F1 提升", f"{result.get('f1_delta', 0):+.3f}")
                        else:
                            st.info(result.get("message", "完成"))
                    
                    except Exception as e:
                        st.error(f"优化失败: {e}")
        
        with opt_col2:
            if st.button("查看版本历史", use_container_width=True):
                try:
                    from src.evolution import SkillOptimizer
                    
                    optimizer = SkillOptimizer()
                    versions = optimizer.list_versions()
                    
                    st.markdown("**已有版本:**")
                    for v in versions:
                        parent = f" ← {v['parent']}" if v.get('parent') else ""
                        st.markdown(f"- `{v['name']}`{parent}")
                except Exception as e:
                    st.error(f"获取版本失败: {e}")


def main():
    """主函数"""
    init_session_state()
    render_sidebar()
    
    # 标签页
    tab1, tab2, tab3 = st.tabs(["💬 对话分析", "📁 文件上传", "🔄 演化引擎"])
    
    with tab1:
        render_chat_interface()
    
    with tab2:
        render_file_upload()
    
    with tab3:
        render_evolution_console()


if __name__ == "__main__":
    main()

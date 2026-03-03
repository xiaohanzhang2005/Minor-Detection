"""
未成年人识别与保护系统 - Web前端界面
基于Streamlit构建，支持对话记录上传和实时分析可视化
"""

import streamlit as st
import json
import os
from typing import List, Dict, Any, Optional
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import io

# 导入系统模块
from models import IntentType, KnowledgeAnalysis, SocialAnalysis
from user_profile import UserProfile
from llm_service import get_llm_service


def load_conversation_from_json(json_data: Dict[str, Any]) -> List[Dict[str, str]]:
    """
    从JSON数据中加载对话记录

    Args:
        json_data: JSON格式的对话数据

    Returns:
        对话记录列表
    """
    conversation = json_data.get('conversation', [])

    # 标准化对话格式
    standardized_conversation = []
    for msg in conversation:
        if isinstance(msg, dict) and 'role' in msg and 'content' in msg:
            standardized_conversation.append({
                'role': msg['role'],
                'content': msg['content']
            })

    return standardized_conversation


def analyze_single_message(message: Dict[str, str], llm_service, user_profile: UserProfile) -> Dict[str, Any]:
    """
    分析单条消息

    Args:
        message: 消息数据
        llm_service: LLM服务实例
        user_profile: 用户画像

    Returns:
        分析结果
    """
    content = message['content']

    try:
        # 意图分类
        intent = llm_service.classify_intent(content)

        # 根据意图进行相应分析
        if intent == IntentType.KNOWLEDGE_QA:
            analysis = llm_service.analyze_knowledge(content)
        else:
            analysis = llm_service.analyze_social(content)

        # 更新用户画像
        user_profile.update_score(analysis, intent)

        # 获取更新后的概率
        minor_prob = user_profile.minor_probability

        return {
            'success': True,
            'intent': intent.value,
            'analysis': analysis,
            'minor_probability': minor_prob,
            'history_summary': user_profile.history_summary[-3:],  # 最近3个特征
            'knowledge_stats': user_profile.knowledge_stats.copy()
        }

    except Exception as e:
        return {
            'success': False,
            'error': str(e)
        }


def create_probability_chart(probability_history: List[float]) -> go.Figure:
    """
    创建未成年人概率变化图表

    Args:
        probability_history: 概率历史记录

    Returns:
        Plotly图表对象
    """
    fig = go.Figure()

    # 添加概率变化线
    fig.add_trace(go.Scatter(
        x=list(range(1, len(probability_history) + 1)),
        y=probability_history,
        mode='lines+markers',
        name='未成年人概率',
        line=dict(color='red', width=3),
        marker=dict(size=8)
    ))

    # 添加阈值线
    fig.add_hline(y=0.65, line_dash="dash", line_color="orange",
                  annotation_text="预测阈值 (0.65)", annotation_position="bottom right")

    fig.update_layout(
        title="未成年人概率动态变化",
        xaxis_title="对话轮次",
        yaxis_title="概率值",
        yaxis_range=[0, 1],
        height=400
    )

    return fig


def create_analysis_summary_chart(analysis_results: List[Dict[str, Any]]) -> go.Figure:
    """
    创建分析结果汇总图表

    Args:
        analysis_results: 分析结果列表

    Returns:
        Plotly图表对象
    """
    # 统计意图分布
    intent_counts = {}
    for result in analysis_results:
        if result['success']:
            intent = result['intent']
            intent_counts[intent] = intent_counts.get(intent, 0) + 1

    # 意图分布饼图
    if intent_counts:
        fig = px.pie(
            values=list(intent_counts.values()),
            names=list(intent_counts.keys()),
            title="意图分类分布"
        )
        fig.update_traces(textposition='inside', textinfo='percent+label')
        return fig
    else:
        return go.Figure()


def main():
    """主函数"""
    st.set_page_config(
        page_title="未成年人识别与保护系统",
        page_icon="🛡️",
        layout="wide"
    )

    st.title("🛡️ 未成年人识别与保护系统")
    st.markdown("---")

    # 初始化LLM服务
    try:
        llm_service = get_llm_service()
        st.success("✅ LLM服务连接成功")
    except Exception as e:
        st.error(f"❌ LLM服务连接失败: {e}")
        return

    # 文件上传区域
    st.header("📁 上传对话记录")

    uploaded_file = st.file_uploader(
        "选择JSON格式的对话记录文件",
        type=['json'],
        help="文件应包含conversation字段，格式参考系统示例"
    )

    if uploaded_file is not None:
        try:
            # 优化文件读取速度
            uploaded_file.seek(0)  # 确保文件指针在文件开头
            json_data = json.load(io.TextIOWrapper(uploaded_file, encoding='utf-8'))

            # 提取对话记录
            conversation = load_conversation_from_json(json_data)

            if not conversation:
                st.error("❌ 文件中未找到有效的对话记录")
                return

            st.success(f"✅ 成功加载 {len(conversation)} 轮对话")

            # 显示对话基本信息
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("对话轮数", len(conversation))
            with col2:
                user_msgs = len([m for m in conversation if m['role'] == 'user'])
                st.metric("用户消息", user_msgs)
            with col3:
                ai_msgs = len([m for m in conversation if m['role'] == 'assistant'])
                st.metric("AI回复", ai_msgs)

            # 分析按钮
            if st.button("🔍 开始分析", type="primary", use_container_width=True):
                analyze_conversation(conversation, llm_service)

        except json.JSONDecodeError:
            st.error("❌ 文件格式错误，请上传有效的JSON文件")
        except Exception as e:
            st.error(f"❌ 文件处理错误: {e}")


def analyze_conversation(conversation: List[Dict[str, str]], llm_service):
    """
    分析整个对话过程

    Args:
        conversation: 对话记录
        llm_service: LLM服务实例
    """
    # 创建用户画像（每次分析为独立会话）
    user_id = f"web_user_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    user_profile = UserProfile(user_id)

    # 将原始消息按照“用户问 -> AI答”的方式分组为轮次
    def group_conversation_rounds(conversation: List[Dict[str, str]]):
        rounds = []
        open_round: Optional[Dict[str, Optional[str]]] = None
        for msg in conversation:
            role = msg.get('role', '').lower()
            content = msg.get('content', '')
            if role == 'user':
                # 如果已有未闭合的轮次（用户连续发言），先保存旧轮次再开启新轮次
                if open_round is not None and open_round.get('user') and not open_round.get('assistant'):
                    rounds.append(open_round)
                open_round = {'user': content, 'assistant': None}
            elif role == 'assistant':
                if open_round is None:
                    # AI先发言，创建一个空用户占位并填充AI回复
                    open_round = {'user': None, 'assistant': content}
                    rounds.append(open_round)
                    open_round = None
                else:
                    # 填充当前轮次的AI回复并关闭轮次
                    open_round['assistant'] = content
                    rounds.append(open_round)
                    open_round = None
            else:
                # 未知角色，忽略
                continue
        # 如果最后还有未闭合的用户提问，保存为单轮（无AI回复）
        if open_round is not None:
            rounds.append(open_round)
        return rounds

    rounds = group_conversation_rounds(conversation)

    # 存储分析结果（按轮）
    analysis_results = []
    probability_history = [0.5]  # 初始概率

    # 创建进度条和状态文本
    progress_bar = st.progress(0)
    status_text = st.empty()

    for i, rnd in enumerate(rounds):
        status_text.text(f"正在分析第 {i+1}/{len(rounds)} 轮对话...")

        # 组合成包含用户与AI回复的文本，供LLM参考完整上下文
        combined_parts = []
        if rnd.get('user'):
            combined_parts.append(f"用户: {rnd['user']}")
        if rnd.get('assistant'):
            combined_parts.append(f"AI: {rnd['assistant']}")
        combined_content = "\n".join(combined_parts)

        # 调用LLM并更新画像（使用组合内容以便判断连贯性）
        try:
            intent = llm_service.classify_intent(combined_content)
            if intent == IntentType.KNOWLEDGE_QA:
                analysis = llm_service.analyze_knowledge(combined_content)
            else:
                analysis = llm_service.analyze_social(combined_content)

            # 更新用户画像
            user_profile.update_score(analysis, intent)

            result = {
                'success': True,
                'intent': intent.value,
                'analysis': analysis,
                'minor_probability': user_profile.minor_probability,
                'history_summary': user_profile.history_summary[-3:],
                'knowledge_stats': user_profile.knowledge_stats.copy(),
                'user': rnd.get('user'),
                'assistant': rnd.get('assistant')
            }
        except Exception as e:
            result = {
                'success': False,
                'error': str(e),
                'user': rnd.get('user'),
                'assistant': rnd.get('assistant')
            }

        analysis_results.append(result)

        # 更新概率历史
        if result['success']:
            probability_history.append(result['minor_probability'])
        else:
            probability_history.append(probability_history[-1])

        progress_bar.progress((i + 1) / len(rounds))

    progress_bar.empty()
    status_text.empty()

    # 显示分析结果（按照轮次）
    display_analysis_results(rounds, analysis_results, probability_history)


def display_analysis_results(conversation: List[Dict[str, str]],
                           analysis_results: List[Dict[str, Any]],
                           probability_history: List[float]):
    """
    显示分析结果

    Args:
        conversation: 原始对话
        analysis_results: 分析结果
        probability_history: 概率历史
    """
    st.header("📊 分析结果")

    # 创建标签页
    tab1, tab2, tab3, tab4 = st.tabs(["📝 详细分析", "📈 概率变化", "📊 统计汇总", "🔍 用户画像"])

    with tab1:
        display_detailed_analysis(conversation, analysis_results)

    with tab2:
        display_probability_chart(probability_history)

    with tab3:
        display_statistics_summary(analysis_results)

    with tab4:
        display_user_profile_analysis(analysis_results)


def display_detailed_analysis(rounds: List[Dict[str, Optional[str]]],
                            analysis_results: List[Dict[str, Any]]):
    """
    显示详细分析结果（按轮次：用户问 -> AI答）
    """
    st.subheader("对话详细分析")

    for i, (rnd, result) in enumerate(zip(rounds, analysis_results)):
        with st.expander(f"第 {i+1} 轮对话", expanded=(i < 3)):  # 默认展开前3轮
            # 显示用户消息和AI回复（如果有）
            user_content = rnd.get('user') or ""
            assistant_content = rnd.get('assistant') or ""
            st.markdown(f"**👤 用户:** {user_content}")
            if assistant_content:
                st.markdown(f"**🤖 AI:** {assistant_content}")

            if result.get('success'):
                # 显示分析结果
                col1, col2 = st.columns(2)

                with col1:
                    st.markdown(f"**🎭 意图分类:** {result['intent']}")

                    if result['intent'] == 'knowledge_qa':
                        analysis = result['analysis']
                        st.markdown(f"**📚 教育水平:** {analysis.predicted_education_level}")
                        st.markdown(f"**📚 是否作业:** {analysis.is_homework}")
                    else:
                        analysis = result['analysis']
                        st.markdown(f"**👥 情绪波动:** {analysis.emotional_score:.3f}")
                        st.markdown(f"**👥 逻辑性:** {analysis.logic_score:.3f}")
                        st.markdown(f"**👥 未成年人倾向:** {analysis.minor_tendency_score:.3f}")

                with col2:
                    st.markdown(f"**🎯 未成年人概率:** {result['minor_probability']:.3f}")
                    st.markdown(f"**🎯 预测结果:** {'⚠️ 未成年人' if result['minor_probability'] > 0.65 else '✅ 正常用户'}")

                    # 显示推理过程
                    with st.expander("💭 详细推理过程"):
                        if hasattr(result['analysis'], 'reasoning'):
                            st.write(result['analysis'].reasoning)
            else:
                st.error(f"❌ 分析失败: {result.get('error')}")


def display_probability_chart(probability_history: List[float]):
    """
    显示概率变化图表
    """
    st.subheader("未成年人概率变化趋势")

    fig = create_probability_chart(probability_history)
    st.plotly_chart(fig, use_container_width=True)

    # 显示当前概率
    current_prob = probability_history[-1]
    st.metric(
        "当前未成年人概率",
        f"{current_prob * 100:.1f}%",
        delta=f"{'高于' if current_prob > 0.65 else '低于'}阈值",
        delta_color="inverse" if current_prob > 0.65 else "normal"
    )


def display_statistics_summary(analysis_results: List[Dict[str, Any]]):
    """
    显示统计汇总
    """
    st.subheader("分析统计汇总")

    successful_analyses = [r for r in analysis_results if r['success']]

    if successful_analyses:
        col1, col2, col3 = st.columns(3)

        with col1:
            total_rounds = len(analysis_results)
            success_rate = len(successful_analyses) / total_rounds * 100
            st.metric("分析成功率", f"{success_rate:.1f}%")

        with col2:
            intent_distribution = {}
            for result in successful_analyses:
                intent = result['intent']
                intent_distribution[intent] = intent_distribution.get(intent, 0) + 1

            most_common_intent = max(intent_distribution, key=intent_distribution.get)
            st.metric("主要意图类型", most_common_intent)

        with col3:
            avg_probability = sum(r['minor_probability'] for r in successful_analyses) / len(successful_analyses)
            st.metric("平均未成年人概率", f"{avg_probability * 100:.1f}%")

        # 意图分布图表
        st.subheader("意图分布")
        fig = create_analysis_summary_chart(successful_analyses)
        st.plotly_chart(fig, use_container_width=True)


def display_user_profile_analysis(analysis_results: List[Dict[str, Any]]):
    """
    显示用户画像分析
    """
    st.subheader("用户画像分析")

    successful_analyses = [r for r in analysis_results if r['success']]

    if successful_analyses:
        latest_result = successful_analyses[-1]

        # 显示最终概率
        final_prob = latest_result['minor_probability']
        st.metric("最终未成年人概率", f"{final_prob * 100:.1f}%")

        # 显示知识统计
        knowledge_stats = latest_result.get('knowledge_stats', {})
        if knowledge_stats:
            st.subheader("知识水平统计")
            stats_df = pd.DataFrame(
                list(knowledge_stats.items()),
                columns=['教育水平', '出现次数']
            )
            st.bar_chart(stats_df.set_index('教育水平'))

        # 显示关键特征
        history_summary = latest_result.get('history_summary', [])
        if history_summary:
            st.subheader("关键特征")
            # 去重并保留顺序
            unique_features = list(dict.fromkeys(history_summary[-5:]))
            for feature in unique_features:
                st.info(feature)


if __name__ == "__main__":
    main()

import html
import json
import re
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import plotly.graph_objects as go
import streamlit as st

from src.models import AnalysisPayload, FormalSkillOutput
from src.runtime import (
    build_formal_enriched_payload,
    build_formal_multi_session_payload,
    build_formal_single_session_payload,
    enrich_multi_session_context,
    enrich_single_session_context,
    get_formal_executor,
)


st.set_page_config(
    page_title="Minor Detection",
    page_icon="M",
    layout="wide",
    initial_sidebar_state="collapsed",
)


ROOT_DIR = Path(__file__).resolve().parent
DEMO_PAYLOAD_PATH = ROOT_DIR / "demo_inputs" / "minor_detection_demo_payload.json"
DEMO_SINGLE_SESSION_PATH = ROOT_DIR / "demo_inputs" / "minor_detection_single_session_payload.json"
DEMO_MULTI_SESSION_PATH = ROOT_DIR / "demo_inputs" / "minor_detection_multi_session_payload.json"
DEMO_EXTERNAL_CONTEXT_PATH = ROOT_DIR / "demo_inputs" / "minor_detection_external_context_payload.json"
PROFILE_STORE_DB_PATH = ROOT_DIR / "data" / "minor_detection_profile_store.sqlite"
PROFILE_STORE_LEGACY_PATH = ROOT_DIR / "data" / "minor_detection_profile_store.json"
DEMO_PAYLOADS = {
    "单会话模式示例模板": DEMO_SINGLE_SESSION_PATH,
    "多会话模式示例模板": DEMO_MULTI_SESSION_PATH,
    "增强输入模式示例模板": DEMO_PAYLOAD_PATH,
}

RISK_LABELS = {
    "High": "高风险",
    "Medium": "中风险",
    "Low": "低风险",
}

NEXT_STEP_LABELS = {
    "collect_more_context": "继续收集上下文",
    "review_by_human": "转人工复核",
    "safe_to_continue": "可继续观察",
    "monitor_future_sessions": "持续监测后续会话",
}

RAG_MODE_DISPLAY = {
    "internal_rag": "内置检索增强",
    "external_rag": "外部检索已注入",
    "no_rag": "未使用检索增强",
}

TIME_MODE_DISPLAY = {
    "script": "脚本解析时间特征",
    "provided": "使用外部时间标签",
    "none": "未检测到时间线索",
}

TIME_FEATURE_LABELS = {
    "local_time": "本地时间",
    "timezone": "时区",
    "weekday": "星期",
    "is_weekend": "是否周末",
    "is_late_night": "是否深夜",
    "time_bucket": "时段标签",
    "holiday_label": "假期标签",
    "school_holiday_hint": "学段假期提示",
}

WEEKDAY_DISPLAY = {
    "monday": "周一",
    "tuesday": "周二",
    "wednesday": "周三",
    "thursday": "周四",
    "friday": "周五",
    "saturday": "周六",
    "sunday": "周日",
}

TIME_BUCKET_DISPLAY = {
    "early_morning": "清晨",
    "morning": "上午",
    "noon": "中午",
    "afternoon": "下午",
    "evening": "晚间",
    "late_night": "深夜",
}

HOLIDAY_DISPLAY = {
    "none": "非假期",
    "winter_vacation": "寒假",
    "summer_vacation": "暑假",
}

ProgressCallback = Callable[[int, str], None]


def inject_styles() -> None:
    st.markdown(
        """
        <style>
        .stApp {
            background:
                radial-gradient(circle at top left, rgba(44, 123, 229, 0.18), transparent 28%),
                radial-gradient(circle at top right, rgba(0, 210, 255, 0.14), transparent 20%),
                linear-gradient(180deg, #07111f 0%, #0a1728 52%, #0d1d32 100%);
            color: #edf4ff;
        }
        .stApp, .stApp p, .stApp label, .stApp li, .stApp span, .stApp div {
            color: #edf4ff;
        }
        .stMarkdown, .stText, .stCaption, .stAlert {
            color: #edf4ff;
        }
        .block-container {
            padding-top: 2.1rem;
            padding-bottom: 2.5rem;
            max-width: 1380px;
        }
        div[data-baseweb="select"] > div {
            background: rgba(12, 29, 52, 0.96) !important;
            border: 1px solid rgba(121, 171, 255, 0.24) !important;
            color: #eef5ff !important;
            border-radius: 16px !important;
        }
        div[data-baseweb="select"] * {
            color: #eef5ff !important;
        }
        [data-testid="stFileUploader"] {
            color: #eef5ff !important;
        }
        [data-testid="stFileUploader"] section,
        [data-testid="stFileUploaderDropzone"] {
            background: linear-gradient(180deg, rgba(14, 34, 61, 0.98), rgba(8, 22, 42, 0.98)) !important;
            border: 1px dashed rgba(123, 173, 255, 0.58) !important;
            border-radius: 22px !important;
            box-shadow: inset 0 1px 0 rgba(255,255,255,0.02) !important;
            min-height: 200px !important;
        }
        [data-testid="stFileUploader"] section > div,
        [data-testid="stFileUploaderDropzone"] > div {
            background: transparent !important;
        }
        [data-testid="stFileUploader"] section *,
        [data-testid="stFileUploaderDropzone"] * {
            color: #eef5ff !important;
            opacity: 1 !important;
            text-shadow: none !important;
        }
        [data-testid="stFileUploaderDropzoneInstructions"],
        [data-testid="stFileUploaderDropzoneInstructions"] span,
        [data-testid="stFileUploaderDropzoneInstructions"] small,
        [data-testid="stFileUploaderDropzoneInstructions"] p,
        [data-testid="stFileUploaderDropzoneInstructions"] div,
        [data-testid="stFileUploaderDropzone"] small,
        [data-testid="stFileUploaderDropzone"] span,
        [data-testid="stFileUploaderDropzone"] p {
            color: #f3f8ff !important;
            opacity: 1 !important;
            font-weight: 600 !important;
        }
        [data-testid="stFileUploaderDropzone"] svg,
        [data-testid="stFileUploaderDropzone"] path,
        [data-testid="stFileUploader"] section svg,
        [data-testid="stFileUploader"] section path {
            fill: #8bc9ff !important;
            stroke: #8bc9ff !important;
        }
        [data-testid="stFileUploader"] small,
        [data-testid="stFileUploader"] span,
        [data-testid="stFileUploader"] label {
            color: #d9e9ff !important;
        }
        [data-testid="stFileUploaderFile"] {
            background: rgba(10, 25, 45, 0.88) !important;
            border: 1px solid rgba(121, 171, 255, 0.18) !important;
            border-radius: 14px !important;
        }
        [data-testid="stFileUploaderFile"] * {
            color: #eef5ff !important;
        }
        [data-testid="stFileUploaderDropzone"] button,
        [data-testid="stFileUploader"] button {
            background: linear-gradient(180deg, #eff6ff 0%, #dce9ff 100%) !important;
            color: #18395f !important;
            border: 1px solid rgba(23, 56, 95, 0.18) !important;
            font-weight: 700 !important;
            opacity: 1 !important;
        }
        [data-testid="stFileUploaderDropzone"] button:hover,
        [data-testid="stFileUploader"] button:hover {
            background: #ffffff !important;
            color: #15324f !important;
        }
        .upload-helper-card {
            margin: 0.35rem 0 0.85rem 0;
            padding: 16px 18px 15px;
            border-radius: 20px;
            background: linear-gradient(180deg, rgba(15, 35, 62, 0.94), rgba(10, 25, 45, 0.94));
            border: 1px solid rgba(120, 174, 255, 0.18);
        }
        .workspace-intro {
            margin: 0.15rem 0 1rem;
            padding: 19px 22px 17px;
            border-radius: 22px;
            background: linear-gradient(180deg, rgba(14, 33, 59, 0.95), rgba(10, 25, 45, 0.92));
            border: 1px solid rgba(118, 175, 255, 0.18);
            box-shadow: 0 14px 34px rgba(6, 14, 24, 0.18);
        }
        .workspace-intro-title {
            color: #f4f8ff;
            font-size: 1.08rem;
            font-weight: 800;
        }
        .workspace-intro-copy {
            margin-top: 0.42rem;
            color: rgba(231, 241, 255, 0.84);
            font-size: 0.9rem;
            line-height: 1.68;
        }
        .upload-helper-title {
            color: #f4f8ff;
            font-size: 0.95rem;
            font-weight: 800;
            letter-spacing: 0.01em;
        }
        .upload-helper-copy {
            margin-top: 0.3rem;
            color: rgba(232, 241, 255, 0.88);
            font-size: 0.86rem;
            line-height: 1.65;
        }
        .stButton > button,
        .stDownloadButton > button {
            background: linear-gradient(180deg, #204a80 0%, #17385f 100%) !important;
            color: #f5f9ff !important;
            border: 1px solid rgba(116, 177, 255, 0.28) !important;
            border-radius: 14px !important;
            font-weight: 700 !important;
            box-shadow: 0 10px 24px rgba(5, 13, 24, 0.22);
        }
        .stButton > button:hover,
        .stDownloadButton > button:hover {
            border-color: rgba(135, 197, 255, 0.5) !important;
            background: linear-gradient(180deg, #285895 0%, #1d4372 100%) !important;
            color: #ffffff !important;
        }
        .stButton > button[kind="primary"] {
            background: linear-gradient(180deg, #2f7fe8 0%, #2261bf 100%) !important;
            border-color: rgba(110, 177, 255, 0.38) !important;
            color: #ffffff !important;
            box-shadow: 0 14px 30px rgba(20, 73, 146, 0.32) !important;
        }
        .st-key-input_ops_shell .stButton > button,
        .st-key-input_ops_shell .stDownloadButton > button,
        .st-key-current_entity_shell .stButton > button,
        .st-key-current_entity_shell .stDownloadButton > button {
            min-height: 50px !important;
            padding: 0.32rem 1.05rem !important;
            font-size: 0.92rem !important;
            box-shadow: 0 7px 16px rgba(6, 15, 28, 0.15) !important;
        }
        .st-key-input_template_controls .stButton > button,
        .st-key-input_template_controls .stDownloadButton > button,
        .st-key-input_profile_controls .stButton > button,
        .st-key-input_profile_controls .stDownloadButton > button {
            width: 100% !important;
            min-width: 0 !important;
            margin-left: 0 !important;
            margin-right: 0 !important;
            display: block !important;
        }
        .st-key-input_ops_shell .stButton > button[kind="secondary"],
        .st-key-input_ops_shell .stDownloadButton > button[kind="secondary"],
        .st-key-current_entity_shell .stButton > button[kind="secondary"] {
            background: linear-gradient(180deg, rgba(25, 58, 100, 0.9), rgba(18, 43, 74, 0.92)) !important;
            border-color: rgba(104, 162, 247, 0.2) !important;
            color: #dbeaff !important;
        }
        .stButton > button:disabled,
        .stDownloadButton > button:disabled {
            background: rgba(113, 136, 170, 0.18) !important;
            color: rgba(226, 236, 255, 0.48) !important;
            border-color: rgba(255, 255, 255, 0.08) !important;
        }
        .stTabs [data-baseweb="tab-list"] button {
            color: rgba(214, 229, 255, 0.72) !important;
            font-weight: 700 !important;
        }
        .stTabs [data-baseweb="tab-list"] {
            margin-top: 1.35rem !important;
            margin-bottom: 0.95rem !important;
        }
        .stTabs [aria-selected="true"] {
            color: #ffffff !important;
        }
        .stTabs [data-baseweb="tab-highlight"] {
            background: #ff5968 !important;
        }
        div[data-testid="stTextArea"] textarea {
            background: linear-gradient(180deg, rgba(7, 18, 34, 0.98), rgba(7, 19, 36, 0.96)) !important;
            color: #eff6ff !important;
            border-radius: 20px !important;
            border: 1px solid rgba(130, 184, 255, 0.28) !important;
            font-size: 0.94rem !important;
            line-height: 1.72 !important;
            font-family: "SFMono-Regular", Consolas, "Liberation Mono", Menlo, monospace !important;
            resize: none !important;
            min-height: 560px !important;
            padding: 18px 18px 20px !important;
            box-shadow: inset 0 1px 0 rgba(255,255,255,0.03), 0 14px 28px rgba(3, 9, 18, 0.18) !important;
        }
        div[data-testid="stTextArea"] textarea:disabled {
            color: #eef5ff !important;
            -webkit-text-fill-color: #eef5ff !important;
            opacity: 1 !important;
            background: linear-gradient(180deg, rgba(11, 27, 49, 0.98), rgba(11, 26, 45, 0.97)) !important;
            border-color: rgba(141, 189, 255, 0.28) !important;
            box-shadow: inset 0 1px 0 rgba(255,255,255,0.02) !important;
        }
        div[data-testid="stTextArea"] textarea::-webkit-scrollbar,
        .transcript-shell::-webkit-scrollbar {
            width: 10px;
        }
        div[data-testid="stTextArea"] textarea::-webkit-scrollbar-track,
        .transcript-shell::-webkit-scrollbar-track {
            background: rgba(255, 255, 255, 0.05);
            border-radius: 999px;
        }
        div[data-testid="stTextArea"] textarea::-webkit-scrollbar-thumb,
        .transcript-shell::-webkit-scrollbar-thumb {
            background: rgba(111, 179, 255, 0.46);
            border-radius: 999px;
            border: 2px solid transparent;
            background-clip: padding-box;
        }
        div[data-testid="stCodeBlock"] {
            border-radius: 18px;
            overflow: hidden;
            border: 1px solid rgba(116, 171, 255, 0.16);
        }
        div[data-testid="stCodeBlock"] pre,
        div[data-testid="stCodeBlock"] code {
            background: linear-gradient(180deg, rgba(7, 18, 34, 0.98), rgba(9, 23, 42, 0.98)) !important;
            color: #dbeafe !important;
        }
        div[data-testid="stCodeBlock"] span {
            color: inherit !important;
        }
        .hero-shell {
            position: relative;
            overflow: hidden;
            padding: 28px 30px 24px 30px;
            border-radius: 28px;
            background:
                linear-gradient(135deg, rgba(12, 32, 59, 0.96), rgba(12, 41, 83, 0.78)),
                linear-gradient(90deg, rgba(83, 163, 255, 0.18), transparent);
            border: 1px solid rgba(120, 174, 255, 0.24);
            box-shadow: 0 24px 60px rgba(4, 11, 22, 0.42);
            margin-bottom: 1.3rem;
        }
        .hero-shell::after {
            content: "";
            position: absolute;
            inset: auto -60px -80px auto;
            width: 220px;
            height: 220px;
            border-radius: 50%;
            background: radial-gradient(circle, rgba(46, 180, 255, 0.32), transparent 72%);
        }
        .eyebrow {
            display: inline-flex;
            align-items: center;
            gap: 0.45rem;
            padding: 0.36rem 0.7rem;
            border-radius: 999px;
            background: rgba(116, 185, 255, 0.12);
            border: 1px solid rgba(116, 185, 255, 0.22);
            color: #9dd5ff;
            font-size: 0.82rem;
            font-weight: 700;
            letter-spacing: 0.02em;
            text-transform: uppercase;
        }
        .hero-title {
            margin: 0.75rem 0 0.4rem 0;
            font-size: 2.45rem;
            line-height: 1.08;
            font-weight: 800;
            letter-spacing: -0.03em;
            color: #f4f8ff;
        }
        .hero-subtitle {
            max-width: 870px;
            margin: 0;
            color: rgba(230, 241, 255, 0.8);
            font-size: 1.02rem;
            line-height: 1.75;
        }
        .panel {
            background: rgba(10, 24, 43, 0.9);
            border: 1px solid rgba(126, 171, 255, 0.18);
            border-radius: 26px;
            padding: 22px 22px 20px 22px;
            box-shadow: 0 18px 44px rgba(0, 0, 0, 0.18);
        }
        .panel-title {
            margin: 0 0 0.65rem 0;
            font-size: 1.12rem;
            font-weight: 700;
            color: #f3f7ff;
        }
        .panel-subtitle {
            margin: 0 0 1rem 0;
            font-size: 0.92rem;
            line-height: 1.65;
            color: rgba(234, 243, 255, 0.88);
        }
        .panel-head {
            min-height: 54px;
            display: flex;
            flex-direction: column;
            justify-content: flex-start;
            margin-bottom: 0.8rem;
        }
        .panel-kicker {
            color: #97cbff;
            font-size: 0.76rem;
            font-weight: 800;
            letter-spacing: 0.08em;
            text-transform: uppercase;
        }
        .panel-head .panel-title {
            margin: 0.42rem 0 0 0;
            font-size: 1.02rem;
            font-weight: 780;
        }
        .panel-head-copy {
            margin: 0;
            color: rgba(233, 242, 255, 0.84);
            font-size: 0.86rem;
            line-height: 1.62;
        }
        .field-label {
            margin: 0 0 0.55rem 0;
            color: rgba(208, 224, 248, 0.76);
            font-size: 0.76rem;
            font-weight: 700;
            letter-spacing: 0.04em;
            text-transform: uppercase;
        }
        .workbench-section-title {
            margin: 0.15rem 0 0.8rem 0;
            color: #f2f7ff;
            font-size: 1.02rem;
            font-weight: 800;
        }
        .st-key-input_ops_shell[data-testid="stVerticalBlockBorderWrapper"],
        .st-key-current_entity_shell[data-testid="stVerticalBlockBorderWrapper"] {
            position: relative;
            overflow: hidden;
            padding: 22px 24px 21px !important;
            border-radius: 22px !important;
            border: 1px solid rgba(118, 175, 255, 0.18) !important;
            background: linear-gradient(180deg, rgba(14, 33, 59, 0.95), rgba(10, 25, 45, 0.92)) !important;
            box-shadow:
                inset 0 1px 0 rgba(255,255,255,0.04),
                0 14px 34px rgba(6, 14, 24, 0.18) !important;
        }
        .st-key-json_editor_shell[data-testid="stVerticalBlockBorderWrapper"] {
            position: relative;
            overflow: hidden;
            padding: 26px 26px 24px !important;
            border-radius: 30px !important;
            border: 1px solid rgba(150, 205, 255, 0.44) !important;
            background:
                linear-gradient(180deg, rgba(20, 46, 80, 0.995), rgba(13, 31, 54, 0.985)) !important;
            box-shadow:
                inset 0 1px 0 rgba(255,255,255,0.08),
                inset 0 0 0 1px rgba(178, 218, 255, 0.05),
                0 32px 72px rgba(2, 9, 19, 0.38),
                0 12px 24px rgba(7, 20, 37, 0.24),
                0 0 0 1px rgba(126, 189, 255, 0.08) !important;
        }
        .st-key-input_ops_shell[data-testid="stVerticalBlockBorderWrapper"]::before,
        .st-key-current_entity_shell[data-testid="stVerticalBlockBorderWrapper"]::before {
            content: "";
            position: absolute;
            inset: 0 auto auto 0;
            width: 100%;
            height: 1px;
            background: rgba(255, 255, 255, 0.05);
            pointer-events: none;
        }
        .st-key-json_editor_shell[data-testid="stVerticalBlockBorderWrapper"]::before {
            content: "";
            position: absolute;
            inset: 0 auto auto 0;
            width: 100%;
            height: 5px;
            background: linear-gradient(90deg, rgba(101, 193, 255, 1), rgba(139, 240, 255, 0.92));
            pointer-events: none;
        }
        .st-key-input_ops_shell[data-testid="stVerticalBlockBorderWrapper"]::after,
        .st-key-current_entity_shell[data-testid="stVerticalBlockBorderWrapper"]::after {
            content: "";
            position: absolute;
            inset: auto -44px -54px auto;
            width: 200px;
            height: 200px;
            border-radius: 50%;
            background: radial-gradient(circle, rgba(62, 143, 232, 0.12), transparent 72%);
            pointer-events: none;
        }
        .st-key-json_editor_shell[data-testid="stVerticalBlockBorderWrapper"]::after {
            content: "";
            position: absolute;
            inset: auto -52px -56px auto;
            width: 220px;
            height: 220px;
            border-radius: 50%;
            background: radial-gradient(circle, rgba(79, 170, 255, 0.16), transparent 72%);
            pointer-events: none;
        }
        .st-key-input_ops_shell[data-testid="stVerticalBlock"],
        .st-key-current_entity_shell[data-testid="stVerticalBlock"],
        .st-key-json_editor_shell[data-testid="stVerticalBlock"] {
            gap: 0.95rem;
        }
        .st-key-input_ops_shell[data-testid="stVerticalBlockBorderWrapper"],
        .st-key-current_entity_shell[data-testid="stVerticalBlockBorderWrapper"] {
            min-height: 712px !important;
        }
        .module-card-head {
            position: relative;
            z-index: 1;
            margin: 0 0 1.15rem 0;
            padding: 0 0 1rem 0;
            border-radius: 0;
            background: transparent;
            border: none;
            border-bottom: 1px solid rgba(143, 197, 255, 0.14);
            box-shadow: none;
        }
        .module-card-title {
            margin: 0;
            color: #f6f9ff;
            font-size: 1.2rem;
            font-weight: 820;
            letter-spacing: -0.01em;
        }
        .module-card-copy {
            margin: 0.45rem 0 0 0;
            max-width: 640px;
            color: rgba(224, 236, 255, 0.76);
            font-size: 0.86rem;
            line-height: 1.66;
        }
        .entity-info-block,
        .st-key-input_template_card,
        .st-key-input_profile_card,
        .st-key-input_template_card[data-testid="stVerticalBlockBorderWrapper"],
        .st-key-input_profile_card[data-testid="stVerticalBlockBorderWrapper"],
        .st-key-input_template_card > [data-testid="stVerticalBlockBorderWrapper"],
        .st-key-input_profile_card > [data-testid="stVerticalBlockBorderWrapper"],
        .st-key-input_template_card [data-testid="stVerticalBlockBorderWrapper"],
        .st-key-input_profile_card [data-testid="stVerticalBlockBorderWrapper"] {
            background: linear-gradient(180deg, rgba(10, 25, 43, 0.985), rgba(8, 20, 36, 0.97)) !important;
            border: 1px solid rgba(109, 168, 242, 0.18) !important;
            border-radius: 22px !important;
            box-shadow:
                inset 0 1px 0 rgba(255,255,255,0.03),
                0 8px 18px rgba(3, 10, 18, 0.14) !important;
        }
        .st-key-input_template_card,
        .st-key-input_profile_card,
        .st-key-input_template_card[data-testid="stVerticalBlockBorderWrapper"],
        .st-key-input_profile_card[data-testid="stVerticalBlockBorderWrapper"],
        .st-key-input_template_card > [data-testid="stVerticalBlockBorderWrapper"],
        .st-key-input_profile_card > [data-testid="stVerticalBlockBorderWrapper"],
        .st-key-input_template_card [data-testid="stVerticalBlockBorderWrapper"],
        .st-key-input_profile_card [data-testid="stVerticalBlockBorderWrapper"] {
            padding: 18px 18px 16px !important;
        }
        .st-key-input_template_card[data-testid="stVerticalBlock"],
        .st-key-input_profile_card[data-testid="stVerticalBlock"],
        .st-key-input_template_card > [data-testid="stVerticalBlock"],
        .st-key-input_profile_card > [data-testid="stVerticalBlock"],
        .st-key-input_template_card [data-testid="stVerticalBlock"],
        .st-key-input_profile_card [data-testid="stVerticalBlock"] {
            gap: 0.38rem;
        }
        .st-key-input_template_card {
            margin-bottom: 0.6rem;
        }
        .st-key-input_template_card,
        .st-key-input_template_card[data-testid="stVerticalBlockBorderWrapper"] {
            min-height: 194px !important;
        }
        .st-key-input_template_card > [data-testid="stVerticalBlockBorderWrapper"],
        .st-key-input_template_card [data-testid="stVerticalBlockBorderWrapper"] {
            min-height: 194px !important;
        }
        .st-key-input_profile_card,
        .st-key-input_profile_card[data-testid="stVerticalBlockBorderWrapper"] {
            min-height: 138px !important;
        }
        .st-key-input_profile_card > [data-testid="stVerticalBlockBorderWrapper"],
        .st-key-input_profile_card [data-testid="stVerticalBlockBorderWrapper"] {
            min-height: 138px !important;
        }
        .control-card-head {
            margin-bottom: 0.95rem;
        }
        .control-card-head .entity-info-value {
            margin-top: 0;
            font-size: 1.12rem;
            line-height: 1.34;
        }
        .control-card-head .entity-info-meta {
            margin-top: 0.56rem;
        }
        .st-key-input_template_controls[data-testid="stVerticalBlockBorderWrapper"],
        .st-key-input_profile_controls[data-testid="stVerticalBlockBorderWrapper"] {
            background: linear-gradient(180deg, rgba(8, 20, 36, 0.985), rgba(6, 16, 29, 0.975)) !important;
            border: 1px solid rgba(96, 151, 223, 0.16) !important;
            border-radius: 18px !important;
            box-shadow:
                inset 0 1px 0 rgba(255,255,255,0.02),
                0 6px 14px rgba(3, 9, 18, 0.12) !important;
            padding: 12px 14px 6px !important;
        }
        .st-key-input_template_controls[data-testid="stVerticalBlock"],
        .st-key-input_profile_controls[data-testid="stVerticalBlock"] {
            gap: 0.62rem;
        }
        .st-key-input_template_controls div[data-baseweb="select"] {
            margin-bottom: 0.8rem;
        }
        .st-key-input_template_controls [data-testid="stHorizontalBlock"],
        .st-key-input_profile_controls [data-testid="stHorizontalBlock"] {
            margin-top: 0.05rem;
        }
        .entity-card-grid {
            display: grid;
            grid-template-columns: repeat(2, minmax(0, 1fr));
            gap: 0.8rem;
            margin-top: 0.2rem;
        }
        .entity-info-block {
            min-height: 126px;
            padding: 16px 16px 14px;
        }
        .entity-info-block.user-id,
        .entity-info-block.matching {
            grid-column: 1 / -1;
            min-height: 118px;
        }
        .entity-info-label {
            color: rgba(203, 221, 247, 0.66);
            font-size: 0.72rem;
            font-weight: 700;
            letter-spacing: 0.04em;
            text-transform: uppercase;
        }
        .entity-info-value {
            margin-top: 0.56rem;
            color: #f5f9ff;
            font-size: 0.96rem;
            font-weight: 760;
            line-height: 1.42;
            word-break: break-word;
        }
        .entity-info-meta {
            margin-top: 0.42rem;
            color: rgba(220, 233, 255, 0.64);
            font-size: 0.8rem;
            line-height: 1.54;
        }
        .st-key-current_entity_shell .entity-info-label {
            font-size: 0.74rem;
            letter-spacing: 0.03em;
        }
        .st-key-current_entity_shell .entity-info-value {
            margin-top: 0.5rem;
            font-size: 0.9rem;
            font-weight: 740;
            line-height: 1.38;
        }
        .st-key-current_entity_shell .entity-info-meta {
            margin-top: 0.34rem;
            font-size: 0.76rem;
            line-height: 1.48;
        }
        .st-key-current_entity_shell .entity-info-block {
            min-height: 120px;
            padding: 15px 15px 13px;
        }
        .st-key-current_entity_shell .entity-info-block.user-id,
        .st-key-current_entity_shell .entity-info-block.matching {
            min-height: 110px;
        }
        .json-card-copy {
            margin: 0.05rem 0 0.9rem 0;
            color: rgba(221, 233, 255, 0.7);
            font-size: 0.84rem;
            line-height: 1.66;
            max-width: 720px;
        }
        .trend-lead {
            max-width: 900px;
            margin: 1rem 0 1.35rem 0.45rem;
            color: rgba(229, 239, 255, 0.78);
            font-size: 0.9rem;
            line-height: 1.72;
        }
        .preview-section-banner {
            margin: 0.35rem 0 0.75rem 0;
            padding: 11px 16px;
            border-radius: 16px;
            background: rgba(8, 21, 40, 0.72);
            border: 1px solid rgba(117, 170, 255, 0.1);
            color: #f3f7ff;
            font-size: 0.96rem;
            font-weight: 800;
        }
        .badge-row {
            display: flex;
            flex-wrap: wrap;
            gap: 0.55rem;
            margin: 0.8rem 0 0.4rem;
        }
        .badge-pill {
            display: inline-flex;
            align-items: center;
            gap: 0.4rem;
            padding: 0.4rem 0.78rem;
            border-radius: 999px;
            font-size: 0.83rem;
            font-weight: 700;
            letter-spacing: 0.01em;
            border: 1px solid rgba(136, 181, 255, 0.16);
            background: rgba(94, 142, 255, 0.12);
            color: #dbe8ff;
        }
        .badge-pill.success {
            background: rgba(17, 185, 129, 0.16);
            color: #b4ffe4;
            border-color: rgba(71, 218, 166, 0.24);
        }
        .badge-pill.info {
            background: rgba(63, 141, 255, 0.18);
            color: #d7e9ff;
            border-color: rgba(122, 184, 255, 0.28);
        }
        .badge-pill.warning {
            background: rgba(255, 180, 0, 0.15);
            color: #ffe9a8;
            border-color: rgba(255, 196, 62, 0.24);
        }
        .badge-pill.danger {
            background: rgba(255, 92, 122, 0.16);
            color: #ffd1db;
            border-color: rgba(255, 119, 145, 0.24);
        }
        .badge-pill.ghost {
            background: rgba(255, 255, 255, 0.06);
            color: #c9d7f2;
            border-color: rgba(255, 255, 255, 0.08);
        }
        .metric-card {
            position: relative;
            min-height: 188px;
            padding: 20px 20px 18px;
            border-radius: 22px;
            background: linear-gradient(180deg, rgba(16, 37, 67, 0.94), rgba(10, 27, 49, 0.92));
            border: 1px solid rgba(126, 171, 255, 0.16);
            box-shadow: inset 0 1px 0 rgba(255,255,255,0.03);
            display: flex;
            flex-direction: column;
            justify-content: flex-start;
            gap: 0.55rem;
            overflow: hidden;
        }
        .metric-card::after {
            content: "";
            position: absolute;
            left: 0;
            right: 0;
            top: 0;
            height: 5px;
            border-radius: 22px 22px 0 0;
            background: linear-gradient(90deg, rgba(68, 164, 255, 0.95), rgba(117, 232, 255, 0.85));
        }
        .metric-card.risk-high::after {
            background: linear-gradient(90deg, #ff5375, #ff8a63);
        }
        .metric-card.risk-high {
            border-color: rgba(255, 115, 139, 0.34);
            box-shadow: inset 0 1px 0 rgba(255,255,255,0.03), 0 18px 34px rgba(85, 17, 31, 0.18);
        }
        .metric-card.risk-medium::after {
            background: linear-gradient(90deg, #ffb020, #ffd166);
        }
        .metric-card.risk-medium {
            border-color: rgba(255, 193, 87, 0.28);
        }
        .metric-card.risk-low::after {
            background: linear-gradient(90deg, #1bd6a6, #5be3ff);
        }
        .metric-card.risk-low {
            border-color: rgba(88, 226, 196, 0.26);
        }
        .metric-label {
            color: rgba(206, 224, 255, 0.72);
            font-size: 0.83rem;
            font-weight: 600;
            letter-spacing: 0.02em;
            text-transform: uppercase;
            min-height: 1.35rem;
        }
        .metric-body {
            display: flex;
            flex-direction: column;
            justify-content: flex-start;
            gap: 0.3rem;
            min-height: 118px;
        }
        .metric-value {
            margin: 0.52rem 0 0.15rem 0;
            font-size: clamp(1.72rem, 2.1vw, 2.4rem);
            font-weight: 800;
            line-height: 1.08;
            color: #f6f9ff;
            letter-spacing: -0.03em;
        }
        .metric-help {
            margin: 0;
            color: rgba(234, 242, 255, 0.84);
            font-size: 0.9rem;
            line-height: 1.62;
        }
        .confidence-track {
            width: 100%;
            height: 10px;
            margin-top: 0.3rem;
            border-radius: 999px;
            background: rgba(255, 255, 255, 0.09);
            overflow: hidden;
        }
        .confidence-fill {
            height: 100%;
            border-radius: 999px;
            background: linear-gradient(90deg, #43b3ff 0%, #7af0ff 100%);
        }
        .section-shell {
            margin-top: 1rem;
            padding: 20px;
            border-radius: 22px;
            background: rgba(8, 21, 40, 0.84);
            border: 1px solid rgba(122, 170, 255, 0.14);
        }
        .section-banner {
            margin-top: 1rem;
            margin-bottom: 0.9rem;
            padding: 15px 18px;
            border-radius: 18px;
            background: rgba(8, 21, 40, 0.88);
            border: 1px solid rgba(122, 170, 255, 0.14);
        }
        .section-heading {
            margin: 0 0 0.9rem 0;
            font-size: 1rem;
            font-weight: 700;
            color: #f3f7ff;
        }
        .section-banner .section-heading {
            margin: 0;
        }
        .mini-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
            gap: 0.8rem;
        }
        .info-card {
            padding: 16px 16px 14px;
            border-radius: 18px;
            background: rgba(15, 34, 61, 0.88);
            border: 1px solid rgba(114, 164, 255, 0.14);
            min-height: 118px;
        }
        .info-label {
            margin: 0 0 0.42rem 0;
            color: rgba(205, 222, 255, 0.68);
            font-size: 0.8rem;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.03em;
        }
        .info-value {
            color: #f7fbff;
            font-size: 1rem;
            line-height: 1.55;
            font-weight: 700;
        }
        .chip-collection {
            display: flex;
            flex-wrap: wrap;
            gap: 0.5rem;
            margin-top: 0.75rem;
        }
        .value-chip {
            padding: 0.46rem 0.78rem;
            border-radius: 999px;
            background: linear-gradient(180deg, rgba(37, 63, 97, 0.86), rgba(25, 46, 74, 0.92));
            border: 1px solid rgba(116, 175, 255, 0.22);
            color: #edf5ff;
            font-size: 0.84rem;
            font-weight: 700;
            box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.02);
        }
        .evidence-stack {
            display: grid;
            gap: 0.85rem;
        }
        .evidence-card {
            position: relative;
            padding: 18px 18px 16px 22px;
            border-radius: 20px;
            background: rgba(16, 34, 59, 0.94);
            border: 1px solid rgba(124, 170, 255, 0.14);
            overflow: hidden;
        }
        .evidence-card::before {
            content: "";
            position: absolute;
            inset: 0 auto 0 0;
            width: 4px;
            background: linear-gradient(180deg, #52b3ff, #6cecff);
        }
        .evidence-card.historical::before {
            background: linear-gradient(180deg, #7f7bff, #a78bff);
        }
        .evidence-card.retrieval::before {
            background: linear-gradient(180deg, #20d5b8, #6df3dd);
        }
        .evidence-card.time::before {
            background: linear-gradient(180deg, #ffb648, #ffe37a);
        }
        .evidence-card.conflict::before {
            background: linear-gradient(180deg, #ff667f, #ff9e9e);
        }
        .evidence-header {
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 0.8rem;
            margin-bottom: 0.6rem;
        }
        .evidence-title {
            font-size: 1rem;
            font-weight: 700;
            color: #f4f8ff;
        }
        .evidence-count {
            min-width: 28px;
            padding: 0.18rem 0.52rem;
            border-radius: 999px;
            background: rgba(255, 255, 255, 0.08);
            color: #d8e6ff;
            font-size: 0.8rem;
            font-weight: 700;
            text-align: center;
        }
        .evidence-list {
            margin: 0;
            padding-left: 1rem;
            color: rgba(231, 240, 255, 0.86);
            line-height: 1.92;
            font-size: 0.95rem;
        }
        .evidence-empty {
            margin: 0.12rem 0 0;
            color: rgba(215, 229, 249, 0.78);
            font-size: 0.92rem;
        }
        .evidence-subsection {
            margin-top: 0.8rem;
        }
        .evidence-subsection-title {
            margin: 0 0 0.5rem;
            color: rgba(228, 239, 255, 0.86);
            font-size: 0.86rem;
            font-weight: 700;
            letter-spacing: 0.02em;
        }
        .reasoning-box {
            margin-top: 0.5rem;
            padding: 18px 18px 16px;
            border-radius: 18px;
            background: rgba(255, 255, 255, 0.04);
            border: 1px solid rgba(255, 255, 255, 0.06);
            color: rgba(236, 243, 255, 0.86);
            line-height: 1.82;
            font-size: 0.96rem;
        }
        .section-lead {
            margin: 0.1rem 0 0.8rem;
            color: rgba(230, 240, 255, 0.84);
            font-size: 0.94rem;
            line-height: 1.75;
        }
        .inline-actions {
            display: flex;
            flex-wrap: wrap;
            gap: 0.65rem;
            margin-top: 0.8rem;
        }
        .retrieval-grid {
            display: grid;
            gap: 0.8rem;
        }
        .retrieval-card {
            padding: 16px 16px 14px;
            border-radius: 18px;
            background: rgba(12, 29, 52, 0.9);
            border: 1px solid rgba(109, 163, 255, 0.14);
        }
        .retrieval-topline {
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 0.8rem;
            margin-bottom: 0.55rem;
        }
        .retrieval-id {
            color: #f3f8ff;
            font-size: 0.92rem;
            font-weight: 700;
        }
        .retrieval-meta {
            color: rgba(216, 231, 255, 0.66);
            font-size: 0.8rem;
        }
        .retrieval-summary {
            margin: 0.15rem 0 0;
            color: rgba(231, 239, 255, 0.84);
            line-height: 1.72;
            font-size: 0.93rem;
        }
        .placeholder-shell {
            position: relative;
            overflow: hidden;
            display: grid;
            grid-template-columns: minmax(260px, 0.9fr) minmax(360px, 1.1fr);
            gap: 1.25rem;
            align-items: center;
            min-height: 500px;
            border-radius: 26px;
            background:
                linear-gradient(180deg, rgba(10, 24, 43, 0.92), rgba(7, 20, 38, 0.96)),
                radial-gradient(circle at center, rgba(84, 169, 255, 0.12), transparent 38%);
            border: 1px dashed rgba(129, 178, 255, 0.22);
            padding: 1.8rem;
        }
        .placeholder-shell::after {
            content: "";
            position: absolute;
            inset: auto -40px -50px auto;
            width: 220px;
            height: 220px;
            border-radius: 50%;
            background: radial-gradient(circle, rgba(58, 171, 255, 0.18), transparent 70%);
            pointer-events: none;
        }
        .placeholder-title {
            margin: 0.55rem 0 0.4rem;
            color: #f3f7ff;
            font-size: 1.42rem;
            font-weight: 800;
        }
        .placeholder-copy {
            max-width: 540px;
            margin: 0;
            color: rgba(231, 240, 255, 0.86);
            line-height: 1.8;
            font-size: 0.97rem;
        }
        .placeholder-panel {
            position: relative;
            padding: 1.1rem;
            border-radius: 24px;
            background: linear-gradient(180deg, rgba(18, 40, 69, 0.9), rgba(10, 25, 45, 0.88));
            border: 1px solid rgba(120, 177, 255, 0.16);
            box-shadow: inset 0 1px 0 rgba(255,255,255,0.03);
        }
        .placeholder-visual {
            position: relative;
            min-height: 280px;
        }
        .placeholder-glow {
            position: absolute;
            inset: 18px 24px auto auto;
            width: 120px;
            height: 120px;
            border-radius: 50%;
            background: radial-gradient(circle, rgba(111, 198, 255, 0.22), transparent 72%);
        }
        .placeholder-node {
            position: absolute;
            border-radius: 20px;
            background: linear-gradient(180deg, rgba(23, 53, 91, 0.96), rgba(13, 33, 58, 0.94));
            border: 1px solid rgba(111, 177, 255, 0.18);
            box-shadow: 0 12px 26px rgba(5, 12, 24, 0.24);
        }
        .placeholder-node.primary {
            inset: 20px 28px auto 20px;
            min-height: 124px;
            padding: 16px;
        }
        .placeholder-node.secondary {
            inset: auto 18px 18px 92px;
            min-height: 104px;
            padding: 14px 16px;
        }
        .placeholder-node.tertiary {
            inset: auto auto 28px 22px;
            width: 96px;
            min-height: 86px;
            padding: 12px;
        }
        .placeholder-kicker {
            color: #8dc9ff;
            font-size: 0.78rem;
            font-weight: 800;
            letter-spacing: 0.05em;
            text-transform: uppercase;
        }
        .placeholder-line {
            display: block;
            height: 10px;
            margin-top: 10px;
            border-radius: 999px;
            background: linear-gradient(90deg, rgba(143, 205, 255, 0.92), rgba(87, 164, 255, 0.4));
        }
        .placeholder-line.short {
            width: 42%;
        }
        .placeholder-line.medium {
            width: 68%;
        }
        .placeholder-line.long {
            width: 88%;
        }
        .placeholder-mini-grid {
            display: grid;
            grid-template-columns: repeat(2, minmax(0, 1fr));
            gap: 0.8rem;
            margin-top: 1rem;
            margin-bottom: 1.55rem;
        }
        .placeholder-mini-card {
            padding: 14px 14px 12px;
            border-radius: 18px;
            background: rgba(255, 255, 255, 0.04);
            border: 1px solid rgba(255, 255, 255, 0.06);
            box-shadow: none;
        }
        .placeholder-footnote {
            margin: 0;
            padding: 0;
            border-top: none;
            color: rgba(219, 232, 252, 0.78);
            font-size: 0.88rem;
            line-height: 1.78;
        }
        .json-shell {
            margin-top: 0.8rem;
            padding: 14px;
            border-radius: 20px;
            background: rgba(8, 20, 38, 0.68);
            border: 1px solid rgba(116, 171, 255, 0.14);
        }
        .workflow-step {
            margin-top: 1rem;
            padding: 18px 20px 16px;
            border-radius: 22px;
            background: linear-gradient(180deg, rgba(11, 27, 49, 0.92), rgba(8, 21, 39, 0.9));
            border: 1px solid rgba(119, 173, 255, 0.14);
        }
        .workflow-topline {
            display: flex;
            align-items: center;
            gap: 0.8rem;
            margin-bottom: 0.55rem;
        }
        .workflow-index {
            display: inline-flex;
            align-items: center;
            justify-content: center;
            width: 32px;
            height: 32px;
            border-radius: 50%;
            background: linear-gradient(180deg, rgba(53, 123, 221, 0.94), rgba(36, 88, 161, 0.96));
            color: #ffffff;
            font-size: 0.9rem;
            font-weight: 800;
            box-shadow: 0 10px 20px rgba(17, 53, 103, 0.28);
        }
        .workflow-title {
            color: #f4f8ff;
            font-size: 1rem;
            font-weight: 800;
        }
        .workflow-copy {
            margin: 0;
            color: rgba(228, 239, 255, 0.82);
            font-size: 0.92rem;
            line-height: 1.72;
        }
        .workflow-nav {
            display: grid;
            grid-template-columns: repeat(4, minmax(0, 1fr));
            gap: 0.9rem;
            margin: 0.35rem 0 1.2rem;
        }
        .workflow-nav-item {
            position: relative;
            padding: 16px 16px 14px;
            border-radius: 20px;
            background: linear-gradient(180deg, rgba(12, 28, 50, 0.9), rgba(8, 22, 40, 0.88));
            border: 1px solid rgba(120, 170, 255, 0.12);
            overflow: hidden;
        }
        .workflow-nav-item::before {
            content: "";
            position: absolute;
            inset: 0 auto auto 0;
            width: 100%;
            height: 3px;
            background: rgba(124, 170, 255, 0.12);
        }
        .workflow-nav-item.active {
            border-color: rgba(110, 182, 255, 0.34);
            box-shadow: 0 16px 34px rgba(10, 37, 70, 0.22);
        }
        .workflow-nav-item.active::before {
            background: linear-gradient(90deg, #52a9ff, #79ecff);
        }
        .workflow-nav-item.completed {
            border-color: rgba(72, 214, 173, 0.28);
        }
        .workflow-nav-item.completed::before {
            background: linear-gradient(90deg, #1fd1aa, #63f0df);
        }
        .workflow-nav-topline {
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 0.8rem;
        }
        .workflow-nav-index {
            display: inline-flex;
            align-items: center;
            justify-content: center;
            width: 30px;
            height: 30px;
            border-radius: 50%;
            background: rgba(92, 150, 232, 0.18);
            border: 1px solid rgba(122, 177, 255, 0.22);
            color: #f3f8ff;
            font-size: 0.84rem;
            font-weight: 800;
        }
        .workflow-nav-item.completed .workflow-nav-index {
            background: rgba(23, 182, 130, 0.16);
            border-color: rgba(93, 231, 188, 0.24);
            color: #cffff1;
        }
        .workflow-nav-item.active .workflow-nav-index {
            background: rgba(62, 135, 233, 0.26);
            border-color: rgba(122, 190, 255, 0.34);
        }
        .workflow-nav-status {
            color: rgba(222, 236, 255, 0.62);
            font-size: 0.72rem;
            font-weight: 800;
            letter-spacing: 0.04em;
            text-transform: uppercase;
        }
        .workflow-nav-item.active .workflow-nav-status {
            color: #8bcfff;
        }
        .workflow-nav-item.completed .workflow-nav-status {
            color: #82f2d5;
        }
        .workflow-nav-link {
            display: block;
            color: inherit !important;
        }
        .workflow-nav-link,
        .workflow-nav-link:visited,
        .workflow-nav-link:hover,
        .workflow-nav-link:active {
            text-decoration: none !important;
            color: inherit !important;
        }
        .workflow-nav-link:hover .workflow-nav-item {
            border-color: rgba(140, 194, 255, 0.3);
            transform: translateY(-1px);
            transition: border-color 0.18s ease, transform 0.18s ease;
        }
        .workflow-nav-link.disabled {
            pointer-events: none;
            cursor: default;
        }
        .workflow-nav-link.disabled .workflow-nav-item {
            opacity: 1;
        }
        .workflow-nav-title {
            margin-top: 0.8rem;
            color: #f4f8ff;
            font-size: 0.98rem;
            font-weight: 800;
        }
        .workflow-nav-copy {
            margin-top: 0.35rem;
            color: rgba(227, 238, 255, 0.74);
            font-size: 0.84rem;
            line-height: 1.65;
        }
        .action-bar {
            margin: 1rem 0 0.4rem;
            padding: 18px 20px;
            border-radius: 22px;
            background: linear-gradient(180deg, rgba(13, 32, 58, 0.94), rgba(9, 24, 44, 0.92));
            border: 1px solid rgba(116, 177, 255, 0.18);
            box-shadow: 0 18px 40px rgba(6, 14, 25, 0.18);
        }
        .page-spacer-sm {
            height: 0.45rem;
        }
        .page-spacer-md {
            height: 0.85rem;
        }
        .page-spacer-lg {
            height: 1.2rem;
        }
        .result-actions-shell {
            height: 0;
        }
        .st-key-result_action_bar[data-testid="stVerticalBlock"],
        .st-key-result_action_bar > [data-testid="stVerticalBlock"] {
            gap: 0 !important;
        }
        .st-key-result_action_bar [data-testid="stHorizontalBlock"] {
            margin-bottom: 0 !important;
        }
        .st-key-result_action_bar .stButton {
            margin-bottom: 0 !important;
        }
        .st-key-result_feedback_shell[data-testid="stVerticalBlock"],
        .st-key-result_feedback_shell > [data-testid="stVerticalBlock"] {
            gap: 0 !important;
        }
        .result-feedback {
            margin: -0.3rem 0 1.1rem;
            padding: 0.68rem 1rem;
            border-radius: 16px;
            border: 1px solid rgba(118, 177, 255, 0.18);
            background: linear-gradient(180deg, rgba(16, 40, 72, 0.92), rgba(11, 30, 54, 0.94));
            box-shadow:
                inset 0 1px 0 rgba(255,255,255,0.03),
                0 8px 18px rgba(4, 11, 20, 0.12);
            color: #edf5ff;
            font-size: 0.92rem;
            font-weight: 700;
            line-height: 1.5;
        }
        .result-feedback.success {
            border-color: rgba(112, 182, 255, 0.24);
            background: linear-gradient(180deg, rgba(14, 54, 96, 0.92), rgba(10, 38, 68, 0.95));
            color: #edf5ff;
        }
        .result-feedback.warning {
            border-color: rgba(255, 196, 62, 0.24);
            background: linear-gradient(180deg, rgba(79, 53, 18, 0.9), rgba(61, 40, 14, 0.93));
            color: #ffefc2;
        }
        .result-panel-shell {
            height: 0.7rem;
        }
        .run-meta-grid {
            display: grid;
            grid-template-columns: repeat(4, minmax(0, 1fr));
            gap: 0.8rem;
            margin: 0.95rem 0 1.05rem;
        }
        .run-meta-card {
            min-height: 150px;
            padding: 16px 17px 15px;
            border-radius: 20px;
            background: linear-gradient(180deg, rgba(16, 37, 66, 0.92), rgba(10, 27, 48, 0.94));
            border: 1px solid rgba(124, 170, 255, 0.14);
        }
        .identity-status-chip {
            display: inline-flex;
            align-items: center;
            margin-top: 0.72rem;
            padding: 0.34rem 0.66rem;
            border-radius: 999px;
            background: rgba(26, 177, 129, 0.12);
            border: 1px solid rgba(81, 226, 176, 0.14);
            color: #bafce7;
            font-size: 0.74rem;
            font-weight: 700;
        }
        .identity-status-chip.muted {
            background: rgba(255, 255, 255, 0.04);
            border-color: rgba(255, 255, 255, 0.06);
            color: #cfdcf3;
        }
        .context-summary-grid {
            display: grid;
            grid-template-columns: repeat(4, minmax(0, 1fr));
            gap: 0.85rem;
            margin-top: 1rem;
        }
        .context-summary-card {
            position: relative;
            min-height: 156px;
            padding: 18px 18px 16px;
            border-radius: 20px;
            background: linear-gradient(180deg, rgba(15, 35, 62, 0.96), rgba(10, 27, 48, 0.92));
            border: 1px solid rgba(117, 171, 255, 0.15);
            box-shadow: inset 0 1px 0 rgba(255,255,255,0.03);
            overflow: hidden;
        }
        .context-summary-card::before {
            content: "";
            position: absolute;
            inset: 0 auto auto 0;
            width: 100%;
            height: 3px;
            background: linear-gradient(90deg, rgba(83, 180, 255, 0.96), rgba(120, 232, 255, 0.84));
        }
        .context-summary-label {
            color: rgba(206, 223, 248, 0.7);
            font-size: 0.78rem;
            font-weight: 700;
            letter-spacing: 0.04em;
            text-transform: uppercase;
        }
        .context-summary-value {
            margin-top: 0.72rem;
            color: #f5f9ff;
            font-size: 1.02rem;
            font-weight: 800;
            line-height: 1.48;
            word-break: break-word;
        }
        .context-summary-meta {
            margin-top: 0.62rem;
            color: rgba(225, 237, 255, 0.7);
            font-size: 0.8rem;
            line-height: 1.6;
        }
        .run-meta-label {
            color: rgba(206, 223, 248, 0.7);
            font-size: 0.76rem;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.04em;
        }
        .run-meta-value {
            margin-top: 0.78rem;
            color: #f4f8ff;
            font-size: 0.98rem;
            font-weight: 800;
            line-height: 1.5;
            word-break: break-word;
        }
        .run-meta-sub {
            margin-top: 0.58rem;
            color: rgba(225, 237, 255, 0.72);
            font-size: 0.82rem;
            line-height: 1.58;
        }
        .result-await-card {
            margin-top: 0.25rem;
            padding: 18px 20px 16px;
            border-radius: 22px;
            background: linear-gradient(180deg, rgba(10, 24, 44, 0.9), rgba(8, 21, 38, 0.92));
            border: 1px dashed rgba(122, 172, 255, 0.18);
        }
        .result-await-title {
            color: #f3f8ff;
            font-size: 1rem;
            font-weight: 800;
        }
        .result-await-copy {
            margin-top: 0.45rem;
            color: rgba(228, 239, 255, 0.82);
            font-size: 0.92rem;
            line-height: 1.74;
        }
        .progress-stage-shell {
            margin-top: 1.1rem;
        }
        .progress-card {
            margin-top: 1rem;
            padding: 20px 22px 18px;
            border-radius: 22px;
            background: linear-gradient(180deg, rgba(11, 27, 49, 0.92), rgba(8, 21, 38, 0.94));
            border: 1px solid rgba(116, 171, 255, 0.14);
        }
        .progress-card .section-heading {
            margin-bottom: 0.5rem;
        }
        .action-note {
            margin: 0 0 0.85rem 0;
            color: rgba(227, 238, 255, 0.82);
            font-size: 0.92rem;
            line-height: 1.7;
        }
        .tiny-note {
            margin-top: 0.8rem;
            color: rgba(217, 231, 252, 0.78);
            font-size: 0.82rem;
            line-height: 1.7;
        }
        @media (max-width: 980px) {
            .placeholder-shell {
                grid-template-columns: 1fr;
            }
            .workflow-nav,
            .run-meta-grid,
            .context-summary-grid {
                grid-template-columns: 1fr 1fr;
            }
        }
        .preview-list {
            display: grid;
            gap: 0.75rem;
        }
        .preview-toolbar {
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 0.8rem;
            margin-bottom: 0.9rem;
        }
        .preview-toolbar-note {
            color: rgba(222, 236, 255, 0.8);
            font-size: 0.88rem;
            line-height: 1.6;
        }
        .preview-card {
            position: relative;
            padding: 16px 17px 15px;
            border-radius: 20px;
            background: rgba(12, 28, 51, 0.88);
            border: 1px solid rgba(109, 163, 255, 0.12);
            overflow: hidden;
        }
        .preview-card::before {
            content: "";
            position: absolute;
            inset: 0 auto 0 0;
            width: 3px;
            background: linear-gradient(180deg, #57b6ff, #73f0ff);
        }
        .preview-card.user {
            background: linear-gradient(180deg, rgba(22, 52, 90, 0.96), rgba(13, 35, 63, 0.94));
            border-color: rgba(96, 171, 255, 0.18);
        }
        .preview-card.user::before {
            background: linear-gradient(180deg, #54a9ff, #8cd6ff);
        }
        .preview-card.assistant {
            background: linear-gradient(180deg, rgba(16, 42, 63, 0.96), rgba(10, 29, 45, 0.94));
            border-color: rgba(86, 218, 193, 0.16);
        }
        .preview-card.assistant::before {
            background: linear-gradient(180deg, #31d7b5, #7af3e0);
        }
        .turn-header,
        .preview-header {
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 0.7rem;
        }
        .turn-role-badge,
        .preview-role-badge {
            display: inline-flex;
            align-items: center;
            justify-content: center;
            width: 28px;
            height: 28px;
            border-radius: 50%;
            background: rgba(103, 173, 255, 0.16);
            border: 1px solid rgba(124, 182, 255, 0.18);
            color: #eef5ff;
            font-size: 0.76rem;
            font-weight: 800;
        }
        .preview-card.user .preview-role-badge,
        .turn-bubble.user .turn-role-badge {
            background: rgba(80, 157, 255, 0.2);
            border-color: rgba(121, 188, 255, 0.28);
            color: #eff7ff;
        }
        .preview-card.assistant .preview-role-badge,
        .turn-bubble.assistant .turn-role-badge {
            background: rgba(42, 196, 170, 0.18);
            border-color: rgba(94, 235, 210, 0.26);
            color: #e8fff9;
        }
        .preview-role {
            color: #9fd0ff;
            font-size: 0.78rem;
            font-weight: 700;
            letter-spacing: 0.03em;
            text-transform: uppercase;
        }
        .preview-card.assistant .preview-role,
        .turn-bubble.assistant .turn-role {
            color: #90f1da;
        }
        .preview-text {
            margin: 0.35rem 0 0;
            color: rgba(237, 244, 255, 0.88);
            font-size: 0.93rem;
            line-height: 1.72;
        }
        .preview-time {
            margin-top: 0.35rem;
            color: rgba(214, 228, 248, 0.78);
            font-size: 0.78rem;
        }
        .session-meta-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
            gap: 0.75rem;
            margin-bottom: 0.95rem;
        }
        .session-meta-card {
            padding: 14px 15px 12px;
            border-radius: 16px;
            background: rgba(13, 30, 55, 0.92);
            border: 1px solid rgba(117, 170, 255, 0.14);
        }
        .session-meta-label {
            color: rgba(206, 223, 248, 0.7);
            font-size: 0.76rem;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.04em;
        }
        .session-meta-value {
            margin-top: 0.38rem;
            color: #f5f9ff;
            font-size: 0.98rem;
            font-weight: 700;
            line-height: 1.45;
        }
        .transcript-shell {
            max-height: 360px;
            overflow-y: auto;
            padding-right: 0.2rem;
            display: grid;
            gap: 0.72rem;
        }
        .turn-bubble {
            position: relative;
            padding: 15px 16px 13px;
            border-radius: 20px;
            border: 1px solid rgba(120, 170, 255, 0.12);
            background: rgba(12, 28, 51, 0.88);
            overflow: hidden;
        }
        .turn-bubble::before {
            content: "";
            position: absolute;
            inset: 0 auto 0 0;
            width: 3px;
            background: linear-gradient(180deg, #57b6ff, #73f0ff);
        }
        .turn-bubble.user {
            background: linear-gradient(180deg, rgba(20, 49, 86, 0.98), rgba(14, 37, 68, 0.95));
            border-color: rgba(108, 175, 255, 0.18);
        }
        .turn-bubble.user::before {
            background: linear-gradient(180deg, #54a9ff, #8cd6ff);
        }
        .turn-bubble.assistant {
            background: linear-gradient(180deg, rgba(16, 42, 63, 0.96), rgba(10, 29, 45, 0.94));
            border-color: rgba(86, 218, 193, 0.16);
        }
        .turn-bubble.assistant::before {
            background: linear-gradient(180deg, #31d7b5, #7af3e0);
        }
        .turn-role {
            color: #9fd0ff;
            font-size: 0.76rem;
            font-weight: 800;
            letter-spacing: 0.04em;
            text-transform: uppercase;
        }
        .turn-content {
            margin-top: 0.38rem;
            color: rgba(239, 245, 255, 0.92);
            font-size: 0.96rem;
            line-height: 1.75;
            word-break: break-word;
        }
        .turn-sub {
            margin-top: 0.42rem;
            color: rgba(203, 220, 244, 0.66);
            font-size: 0.78rem;
        }
        .equal-panel {
            min-height: 328px;
            height: 100%;
        }
        .trajectory-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
            gap: 0.8rem;
            margin-top: 1.1rem;
        }
        .trajectory-card {
            position: relative;
            padding: 16px 16px 14px;
            border-radius: 18px;
            background: linear-gradient(180deg, rgba(15, 34, 60, 0.94), rgba(10, 27, 48, 0.92));
            border: 1px solid rgba(118, 170, 255, 0.14);
            overflow: hidden;
        }
        .trajectory-card::before {
            content: "";
            position: absolute;
            inset: 0 auto auto 0;
            width: 100%;
            height: 3px;
            background: linear-gradient(90deg, #5fc1ff, #87f0ff);
        }
        .trajectory-session {
            color: #f5f9ff;
            font-size: 0.96rem;
            font-weight: 800;
        }
        .trajectory-time {
            margin-top: 0.25rem;
            color: rgba(214, 228, 248, 0.72);
            font-size: 0.78rem;
            line-height: 1.55;
        }
        .trajectory-probability {
            margin-top: 0.8rem;
            color: #ffffff;
            font-size: 1.45rem;
            font-weight: 800;
            letter-spacing: -0.02em;
        }
        .trajectory-meta {
            margin-top: 0.45rem;
            color: rgba(231, 239, 255, 0.82);
            font-size: 0.85rem;
            line-height: 1.7;
        }
        .trajectory-track {
            width: 100%;
            height: 7px;
            margin-top: 0.85rem;
            border-radius: 999px;
            background: rgba(255, 255, 255, 0.08);
            overflow: hidden;
        }
        .trajectory-fill {
            height: 100%;
            border-radius: 999px;
            background: linear-gradient(90deg, #56beff 0%, #6fe9ff 100%);
        }
        @media (max-width: 680px) {
            .workflow-nav,
            .run-meta-grid,
            .context-summary-grid {
                grid-template-columns: 1fr;
            }
            .entity-card-grid {
                grid-template-columns: 1fr;
            }
            .entity-info-block.user-id,
            .entity-info-block.matching {
                grid-column: auto;
            }
            .st-key-input_ops_shell [data-testid="stVerticalBlockBorderWrapper"],
            .st-key-current_entity_shell [data-testid="stVerticalBlockBorderWrapper"] {
                min-height: auto;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def load_demo_payload_text() -> str:
    return DEMO_PAYLOAD_PATH.read_text(encoding="utf-8")


def ensure_state() -> None:
    if "minor_detection_input_text" not in st.session_state:
        st.session_state.minor_detection_input_text = load_demo_payload_text()
    if "minor_detection_input_editor" not in st.session_state:
        st.session_state.minor_detection_input_editor = st.session_state.minor_detection_input_text
    if "minor_detection_last_result" not in st.session_state:
        st.session_state.minor_detection_last_result = None
    if "minor_detection_uploaded_snapshot" not in st.session_state:
        st.session_state.minor_detection_uploaded_snapshot = ""
    if "minor_detection_uploaded_name" not in st.session_state:
        st.session_state.minor_detection_uploaded_name = ""
    if "minor_detection_selected_demo" not in st.session_state:
        st.session_state.minor_detection_selected_demo = "增强输入模式"
    if "minor_detection_input_origin" not in st.session_state:
        st.session_state.minor_detection_input_origin = "template:增强输入模式"
    if "minor_detection_last_input_snapshot" not in st.session_state:
        st.session_state.minor_detection_last_input_snapshot = st.session_state.minor_detection_input_text
    if "minor_detection_current_step" not in st.session_state:
        st.session_state.minor_detection_current_step = 1
    if "minor_detection_pending_run" not in st.session_state:
        st.session_state.minor_detection_pending_run = False


def clear_last_result() -> None:
    st.session_state.minor_detection_last_result = None


def set_query_step(step: int) -> None:
    st.query_params["step"] = str(step)


def jump_to_step(step: int) -> None:
    step = max(1, min(4, int(step)))
    st.session_state.minor_detection_current_step = step
    set_query_step(step)


def reset_workflow_to_input() -> None:
    clear_last_result()
    st.session_state.minor_detection_pending_run = False
    for key in list(st.session_state.keys()):
        if key.startswith("minor_detection_preview_window_"):
            del st.session_state[key]
    jump_to_step(1)


def get_input_text() -> str:
    return st.session_state.get("minor_detection_input_text", "")


def sanitize_json_text(text: str) -> str:
    text = str(text or "")
    # Tolerate BOM / zero-width markers accidentally introduced by editors.
    return text.lstrip("\ufeff\u200b\u200c\u200d\u2060")


def set_input_text(text: str) -> None:
    st.session_state.minor_detection_input_text = text
    st.session_state.minor_detection_input_editor = text


def set_input_origin(origin: str) -> None:
    st.session_state.minor_detection_input_origin = str(origin or "").strip() or "unknown"


def get_input_origin() -> str:
    return str(st.session_state.get("minor_detection_input_origin", "unknown") or "unknown")


def commit_input_editor() -> None:
    current_text = st.session_state.get("minor_detection_input_editor", "")
    previous_text = st.session_state.get("minor_detection_input_text", "")
    if current_text != previous_text:
        reset_workflow_to_input()
        st.session_state.minor_detection_input_text = current_text
        st.session_state.minor_detection_last_input_snapshot = current_text
        st.session_state.minor_detection_uploaded_name = ""
        set_input_origin("editor")


def sync_input_snapshot() -> None:
    current_text = get_input_text()
    previous_text = st.session_state.get("minor_detection_last_input_snapshot", "")
    if current_text != previous_text:
        st.session_state.minor_detection_last_input_snapshot = current_text


def escape(value: Any) -> str:
    return html.escape(str(value))


def pretty_json(data: Any) -> str:
    return json.dumps(data, ensure_ascii=False, indent=2)


def _normalize_profile_entry(raw_entry: Any) -> Optional[Dict[str, Any]]:
    if not isinstance(raw_entry, dict):
        return None
    raw_identity_markers = raw_entry.get("identity_markers") or []
    if not isinstance(raw_identity_markers, list):
        raw_identity_markers = []
    return {
        "total_sessions": int(raw_entry.get("total_sessions", 0) or 0),
        "estimated_age_range": str(raw_entry.get("estimated_age_range", "") or ""),
        "education_stage": str(raw_entry.get("education_stage", "") or ""),
        "recent_confidence_trend": str(raw_entry.get("recent_confidence_trend", "") or ""),
        "summary": str(raw_entry.get("summary", "") or ""),
        "identity_markers": [str(item) for item in raw_identity_markers],
        "last_minor_probability": (
            round(float(raw_entry["last_minor_probability"]), 4)
            if raw_entry.get("last_minor_probability") not in (None, "")
            else None
        ),
        "updated_at": str(raw_entry.get("updated_at", "") or ""),
    }


def _load_legacy_profile_store() -> Dict[str, Any]:
    if not PROFILE_STORE_LEGACY_PATH.exists():
        return {}
    try:
        raw_store = json.loads(PROFILE_STORE_LEGACY_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {}
    if not isinstance(raw_store, dict):
        return {}

    normalized_store: Dict[str, Any] = {}
    for user_id, entry in raw_store.items():
        normalized_entry = _normalize_profile_entry(entry)
        if normalized_entry is not None:
            normalized_store[str(user_id)] = normalized_entry
    return normalized_store


def _profile_store_connection() -> sqlite3.Connection:
    PROFILE_STORE_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(PROFILE_STORE_DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS profile_store (
            user_id TEXT PRIMARY KEY,
            total_sessions INTEGER NOT NULL DEFAULT 0,
            estimated_age_range TEXT NOT NULL DEFAULT '',
            education_stage TEXT NOT NULL DEFAULT '',
            recent_confidence_trend TEXT NOT NULL DEFAULT '',
            summary TEXT NOT NULL DEFAULT '',
            identity_markers_json TEXT NOT NULL DEFAULT '[]',
            last_minor_probability REAL,
            updated_at TEXT NOT NULL DEFAULT ''
        )
        """
    )

    row_count = conn.execute("SELECT COUNT(*) FROM profile_store").fetchone()[0]
    if row_count == 0:
        legacy_store = _load_legacy_profile_store()
        if legacy_store:
            conn.executemany(
                """
                INSERT OR REPLACE INTO profile_store (
                    user_id,
                    total_sessions,
                    estimated_age_range,
                    education_stage,
                    recent_confidence_trend,
                    summary,
                    identity_markers_json,
                    last_minor_probability,
                    updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        user_id,
                        entry["total_sessions"],
                        entry["estimated_age_range"],
                        entry["education_stage"],
                        entry["recent_confidence_trend"],
                        entry["summary"],
                        json.dumps(entry["identity_markers"], ensure_ascii=False),
                        entry["last_minor_probability"],
                        entry["updated_at"],
                    )
                    for user_id, entry in legacy_store.items()
                ],
            )
            conn.commit()
    return conn


def load_profile_store() -> Dict[str, Any]:
    store: Dict[str, Any] = {}
    conn: Optional[sqlite3.Connection] = None
    try:
        conn = _profile_store_connection()
        rows = conn.execute(
            """
            SELECT
                user_id,
                total_sessions,
                estimated_age_range,
                education_stage,
                recent_confidence_trend,
                summary,
                identity_markers_json,
                last_minor_probability,
                updated_at
            FROM profile_store
            ORDER BY user_id
            """
        ).fetchall()
    except Exception:
        return {}
    finally:
        if conn is not None:
            conn.close()

    for row in rows:
        try:
            identity_markers = json.loads(row["identity_markers_json"] or "[]")
        except Exception:
            identity_markers = []
        if not isinstance(identity_markers, list):
            identity_markers = []

        store[str(row["user_id"])] = {
            "total_sessions": int(row["total_sessions"] or 0),
            "estimated_age_range": row["estimated_age_range"] or "",
            "education_stage": row["education_stage"] or "",
            "recent_confidence_trend": row["recent_confidence_trend"] or "",
            "summary": row["summary"] or "",
            "identity_markers": [str(item) for item in identity_markers],
            "last_minor_probability": row["last_minor_probability"],
            "updated_at": row["updated_at"] or "",
        }
    return store


def save_profile_store(store: Dict[str, Any]) -> None:
    normalized_rows = []
    for user_id, raw_entry in store.items():
        normalized_entry = _normalize_profile_entry(raw_entry)
        if normalized_entry is None:
            continue
        normalized_rows.append(
            (
                str(user_id),
                normalized_entry["total_sessions"],
                normalized_entry["estimated_age_range"],
                normalized_entry["education_stage"],
                normalized_entry["recent_confidence_trend"],
                normalized_entry["summary"],
                json.dumps(normalized_entry["identity_markers"], ensure_ascii=False),
                normalized_entry["last_minor_probability"],
                normalized_entry["updated_at"],
            )
        )

    conn = _profile_store_connection()
    try:
        conn.execute("DELETE FROM profile_store")
        if normalized_rows:
            conn.executemany(
                """
                INSERT INTO profile_store (
                    user_id,
                    total_sessions,
                    estimated_age_range,
                    education_stage,
                    recent_confidence_trend,
                    summary,
                    identity_markers_json,
                    last_minor_probability,
                    updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                normalized_rows,
            )
        conn.commit()
    finally:
        conn.close()


def normalize_uploaded_payload(raw_data: Any) -> Dict[str, Any]:
    if isinstance(raw_data, list):
        return {
            "mode": "single_session",
            "conversation": raw_data,
            "context": {},
            "meta": {},
        }
    if isinstance(raw_data, dict):
        if "mode" in raw_data:
            return raw_data
        if "sessions" in raw_data:
            return {
                "mode": "multi_session",
                "sessions": raw_data.get("sessions", []),
                "context": raw_data.get("context", {}),
                "meta": raw_data.get("meta", {}),
            }
        if "conversation" in raw_data:
            return {
                "mode": "single_session",
                "conversation": raw_data.get("conversation", []),
                "context": raw_data.get("context", {}),
                "meta": raw_data.get("meta", {}),
            }
    raise ValueError("上传内容必须是 analysis_payload，或 conversation/sessions JSON。")


def get_current_input_payload() -> Dict[str, Any]:
    raw_payload = json.loads(sanitize_json_text(get_input_text()) or "{}")
    return normalize_uploaded_payload(raw_payload)


def current_input_user_id() -> str:
    try:
        payload = get_current_input_payload()
    except Exception:
        return ""
    meta = payload.get("meta", {}) or {}
    return str(meta.get("user_id", "") or "").strip()


def inject_profile_from_store_into_input() -> bool:
    user_id = current_input_user_id()
    if not user_id:
        return False
    store = load_profile_store()
    entry = store.get(user_id)
    if not isinstance(entry, dict):
        return False

    payload = get_current_input_payload()
    payload.setdefault("context", {})
    payload["context"]["prior_profile"] = entry
    formatted = pretty_json(payload)
    set_input_text(formatted)
    set_input_origin("profile_injected")
    st.session_state.minor_detection_last_input_snapshot = formatted
    st.session_state.minor_detection_uploaded_snapshot = ""
    st.session_state.minor_detection_uploaded_name = ""
    reset_workflow_to_input()
    st.session_state.minor_detection_last_input_snapshot = formatted
    return True


def save_last_result_to_store() -> bool:
    run_result = st.session_state.get("minor_detection_last_result")
    if not isinstance(run_result, dict) or "formal_output" not in run_result:
        return False

    final_payload = run_result["final_payload"]
    payload_dict = final_payload.model_dump() if hasattr(final_payload, "model_dump") else final_payload.dict()
    meta = payload_dict.get("meta", {}) or {}
    user_id = str(meta.get("user_id", "") or "").strip()
    if not user_id:
        return False

    output = run_result["formal_output"]
    normalized_payload = run_result["normalized_payload"]
    session_count = len(normalized_payload.sessions) if normalized_payload.mode == "multi_session" else 1

    store = load_profile_store()
    previous = store.get(user_id, {}) if isinstance(store.get(user_id), dict) else {}
    previous_confidence = previous.get("last_minor_probability")
    current_confidence = float(output.decision.minor_confidence)
    trend_text = "首次写入"
    if isinstance(previous_confidence, (int, float)):
        if current_confidence > float(previous_confidence) + 0.03:
            trend_text = "较历史上升"
        elif current_confidence < float(previous_confidence) - 0.03:
            trend_text = "较历史下降"
        else:
            trend_text = "与历史接近"

    entry = {
        "total_sessions": int(previous.get("total_sessions", 0) or 0) + session_count,
        "estimated_age_range": output.user_profile.age_range,
        "education_stage": output.user_profile.education_stage,
        "recent_confidence_trend": trend_text,
        "summary": output.reasoning_summary,
        "identity_markers": list(output.user_profile.identity_markers or []),
        "last_minor_probability": round(current_confidence, 4),
        "updated_at": datetime.now().isoformat(timespec="seconds"),
    }
    store[user_id] = entry
    save_profile_store(store)
    return True


def payload_stats(payload: AnalysisPayload) -> Dict[str, Any]:
    if payload.mode == "multi_session":
        sessions = payload.sessions
        turn_count = sum(len(session.conversation) for session in sessions)
        return {
            "mode": payload.mode,
            "sessions": len(sessions),
            "turns": turn_count,
            "has_prior_profile": bool(payload.context.prior_profile),
            "has_external_rag": bool(payload.context.retrieved_cases),
        }
    return {
        "mode": payload.mode,
        "sessions": 0,
        "turns": len(payload.conversation),
        "has_prior_profile": bool(payload.context.prior_profile),
        "has_external_rag": bool(payload.context.retrieved_cases),
    }


def prepare_and_run(
    payload_dict: Dict[str, Any],
    *,
    progress_callback: Optional[ProgressCallback] = None,
) -> Dict[str, Any]:
    def report(progress: int, message: str) -> None:
        if progress_callback is not None:
            progress_callback(max(0, min(100, int(progress))), message)

    report(4, "解析输入并初始化检测任务")
    normalized_payload = AnalysisPayload(**normalize_uploaded_payload(payload_dict))
    report(8, "加载正式识别执行器")
    executor = get_formal_executor()
    base_context = (
        normalized_payload.context.model_dump()
        if hasattr(normalized_payload.context, "model_dump")
        else normalized_payload.context.dict()
    )
    meta = (
        normalized_payload.meta.model_dump()
        if hasattr(normalized_payload.meta, "model_dump")
        else normalized_payload.meta.dict()
    )
    report(12, "读取上下文与请求元数据")

    if normalized_payload.mode == "multi_session":
        report(20, "注入多会话时间特征与内置 RAG")
        sessions = [
            session.model_dump() if hasattr(session, "model_dump") else session.dict()
            for session in normalized_payload.sessions
        ]
        enriched_context = enrich_multi_session_context(sessions, context=base_context)
        report(30, "构建多会话 formal payload")
        final_payload = build_formal_multi_session_payload(
            sessions,
            user_id=meta.get("user_id", ""),
            request_id=meta.get("request_id", ""),
            source=meta.get("source", "minor_detection_frontend"),
            context=enriched_context,
        )
    elif normalized_payload.mode == "enriched":
        conversation = list(normalized_payload.conversation)
        report(20, "增强当前会话上下文")
        enriched_context = enrich_single_session_context(conversation, context=base_context)
        report(30, "构建增强输入 formal payload")
        final_payload = build_formal_enriched_payload(
            conversation,
            user_id=meta.get("user_id", ""),
            request_id=meta.get("request_id", ""),
            source=meta.get("source", "minor_detection_frontend"),
            context=enriched_context,
        )
    else:
        conversation = list(normalized_payload.conversation)
        report(20, "提取单会话时间特征与内置 RAG")
        enriched_context = enrich_single_session_context(conversation, context=base_context)
        report(30, "构建单会话 formal payload")
        final_payload = build_formal_single_session_payload(
            conversation,
            user_id=meta.get("user_id", ""),
            request_id=meta.get("request_id", ""),
            source=meta.get("source", "minor_detection_frontend"),
            context=enriched_context,
        )

    report(45, "执行正式未成年人识别链路")
    result = executor.run_formal_payload(final_payload)
    report(68, "整理结构化识别结果")
    run_result = {
        "normalized_payload": normalized_payload,
        "final_payload": final_payload,
        "enriched_context": enriched_context,
        "formal_output": result,
    }
    if normalized_payload.mode == "multi_session":
        report(76, "计算递进更新概率曲线")
        run_result["session_curve"] = build_multi_session_curve(
            executor=executor,
            sessions=sessions,
            meta=meta,
            base_context=base_context,
            final_result=result,
            progress_callback=progress_callback,
            start_progress=76,
            end_progress=94,
        )
    report(100, "检测完成")
    return run_result


def build_multi_session_curve(
    *,
    executor: Any,
    sessions: List[Dict[str, Any]],
    meta: Dict[str, Any],
    base_context: Dict[str, Any],
    final_result: FormalSkillOutput,
    progress_callback: Optional[ProgressCallback] = None,
    start_progress: int = 76,
    end_progress: int = 98,
) -> List[Dict[str, Any]]:
    points: List[Dict[str, Any]] = []
    total_sessions = max(len(sessions), 1)
    for index, session in enumerate(sessions):
        if progress_callback is not None:
            progress_value = start_progress + int((end_progress - start_progress) * index / total_sessions)
            progress_callback(progress_value, f"递进更新至第 {index + 1}/{total_sessions} 个会话")

        session_id = str(session.get("session_id", "") or f"s{index + 1}")
        session_time = str(session.get("session_time", "") or "").strip()
        prefix_sessions = sessions[: index + 1]
        try:
            if index == total_sessions - 1:
                single_result = final_result
            else:
                prefix_context = enrich_multi_session_context(prefix_sessions, context=base_context)
                prefix_payload = build_formal_multi_session_payload(
                    prefix_sessions,
                    user_id=meta.get("user_id", ""),
                    request_id=f"{meta.get('request_id', 'req')}_progressive_{index + 1}",
                    source=meta.get("source", "minor_detection_frontend"),
                    context=prefix_context,
                )
                single_result = executor.run_formal_payload(prefix_payload)
            points.append(
                {
                    "session_id": session_id,
                    "session_time": session_time,
                    "minor_confidence": float(single_result.decision.minor_confidence),
                    "risk_level": str(single_result.decision.risk_level.value if hasattr(single_result.decision.risk_level, "value") else single_result.decision.risk_level),
                    "education_stage": single_result.user_profile.education_stage,
                    "is_minor": bool(single_result.decision.is_minor),
                }
            )
        except Exception as exc:
            points.append(
                {
                    "session_id": str(session.get("session_id", "") or f"s{index + 1}"),
                    "session_time": session_time,
                    "minor_confidence": None,
                    "risk_level": "Unknown",
                    "education_stage": "",
                    "is_minor": None,
                    "error": str(exc),
                }
            )
    if progress_callback is not None:
        progress_callback(end_progress, "递进更新概率曲线计算完成")
    return points


def render_badges(items: List[Tuple[str, str]]) -> None:
    badge_html = "".join(
        f"<span class='badge-pill {escape(tone)}'>{escape(label)}</span>"
        for label, tone in items
    )
    st.markdown(f"<div class='badge-row'>{badge_html}</div>", unsafe_allow_html=True)


def render_section_banner(title: str) -> None:
    st.markdown(
        f"<div class='section-banner'><div class='section-heading'>{escape(title)}</div></div>",
        unsafe_allow_html=True,
    )


def render_section_shell(title: str, body_html: str, *, lead_html: str = "") -> None:
    st.markdown(
        "<div class='section-shell'>"
        f"<div class='section-heading'>{escape(title)}</div>"
        f"{lead_html}"
        f"{body_html}"
        "</div>",
        unsafe_allow_html=True,
    )


def render_workflow_step(index: int, title: str, copy: str) -> None:
    st.markdown(
        "<div class='workflow-step'>"
        "<div class='workflow-topline'>"
        f"<div class='workflow-index'>{index}</div>"
        f"<div class='workflow-title'>{escape(title)}</div>"
        "</div>"
        f"<p class='workflow-copy'>{escape(copy)}</p>"
        "</div>",
        unsafe_allow_html=True,
    )


def get_available_steps(preview_payload: Optional[AnalysisPayload]) -> List[int]:
    available = [1]
    if preview_payload is not None:
        available.append(2)
    if st.session_state.get("minor_detection_pending_run"):
        available.append(3)
    last_result = st.session_state.get("minor_detection_last_result")
    if isinstance(last_result, dict) and last_result:
        available.append(4)
    return sorted(set(available))


def resolve_current_step(available_steps: List[int]) -> int:
    requested_step = st.query_params.get("step")
    fallback_step = st.session_state.get("minor_detection_current_step", 1)
    try:
        step = int(requested_step or fallback_step)
    except (TypeError, ValueError):
        step = int(fallback_step)

    if step not in available_steps:
        lower_steps = [candidate for candidate in available_steps if candidate <= step]
        step = lower_steps[-1] if lower_steps else available_steps[0]

    st.session_state.minor_detection_current_step = step
    set_query_step(step)
    return step


def render_workflow_navigator(current_step: int, available_steps: List[int]) -> None:
    items = [
        (1, "配置输入", "上传 JSON、加载样例或注入历史画像"),
        (2, "确认概览", "检查模式、窗口与对话内容是否正确"),
        (3, "运行识别", "启动正式未成年人识别流程"),
        (4, "查看结果", "查看概率、画像与证据链"),
    ]
    cards: List[str] = []
    for index, title, copy in items:
        if index < current_step:
            klass = "completed"
            status = "DONE"
        elif index == current_step:
            klass = "active"
            status = "CURRENT"
        else:
            klass = ""
            status = "PENDING"
        clickable = index in available_steps and (index != 3 or st.session_state.get("minor_detection_pending_run"))
        body = (
            f"<div class='workflow-nav-item {klass}'>"
            "<div class='workflow-nav-topline'>"
            f"<div class='workflow-nav-index'>{index}</div>"
            f"<div class='workflow-nav-status'>{status}</div>"
            "</div>"
            f"<div class='workflow-nav-title'>{escape(title)}</div>"
            f"<div class='workflow-nav-copy'>{escape(copy)}</div>"
            "</div>"
        )
        if clickable:
            cards.append(
                f"<a class='workflow-nav-link' href='?step={index}'>{body}</a>"
            )
        else:
            cards.append(
                f"<div class='workflow-nav-link disabled'>{body}</div>"
            )
    st.markdown(f"<div class='workflow-nav'>{''.join(cards)}</div>", unsafe_allow_html=True)


def render_hero() -> None:
    st.markdown(
        """
        <div class="hero-shell">
          <span class="eyebrow">Minor Detection</span>
          <h1 class="hero-title">Minor Detection</h1>
          <p class="hero-subtitle">
            面向真实接入场景的未成年人识别系统。当前版本采用分步式工作流，
            支持配置输入、确认多会话概览、执行正式识别，并以结构化方式展示
            未成年人概率、递进更新轨迹、证据链、用户画像与外部注入上下文。
          </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def format_mode_display(mode: str) -> str:
    return {
        "single_session": "单会话",
        "multi_session": "多会话",
        "enriched": "增强输入",
    }.get(str(mode or "").strip(), str(mode or "未明确"))


def format_source_display(source: str) -> str:
    text = str(source or "").strip()
    if not text:
        return "未标注来源"
    if text.startswith("minor_detection_demo"):
        return "系统样例"
    if text == "minor_detection_frontend":
        return "前端工作台"
    return text.replace("_", " ")


def format_input_origin_display(origin: str) -> str:
    text = str(origin or "").strip()
    if text.startswith("template:"):
        return f"系统模板 · {text.split(':', 1)[1].strip() or '未命名模板'}"
    return {
        "upload": "上传文件",
        "editor": "工作台编辑",
        "profile_injected": "画像注入后输入",
        "unknown": "未标注来源",
    }.get(text, text or "未标注来源")


def render_input_panel() -> None:
    st.markdown(
        """
        <div class='workspace-intro'>
          <div class='workspace-intro-title'>输入工作台</div>
          <div class='workspace-intro-copy'>
            围绕当前输入实体完成载入、编辑、画像注入与结果前确认。
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        """
        <div class='upload-helper-card'>
          <div class='upload-helper-title'>上传输入文件</div>
          <div class='upload-helper-copy'>
            可直接上传待分析 JSON；载入后会自动进入当前工作台，
            支持标准分析输入、对话数组与多会话结构。
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    uploaded_file = st.file_uploader(
        "上传待分析 JSON 文件",
        type=["json"],
        label_visibility="collapsed",
    )
    if uploaded_file is not None:
        uploaded_text = sanitize_json_text(uploaded_file.getvalue().decode("utf-8-sig"))
        if uploaded_text != st.session_state.minor_detection_uploaded_snapshot:
            set_input_text(uploaded_text)
            set_input_origin("upload")
            st.session_state.minor_detection_uploaded_snapshot = uploaded_text
            st.session_state.minor_detection_uploaded_name = str(uploaded_file.name or "").strip()
            st.session_state.minor_detection_last_input_snapshot = uploaded_text
            reset_workflow_to_input()

    store = load_profile_store()
    current_user = current_input_user_id()
    injected_available = bool(current_user and isinstance(store.get(current_user), dict))
    current_mode = "未识别"
    profile_state = "暂无可注入画像"
    profile_meta = "当前没有匹配到同 user_id 历史画像"
    try:
        current_mode = format_mode_display(get_current_input_payload().get("mode", ""))
    except Exception:
        pass
    if injected_available:
        profile_state = "可注入历史画像"
        profile_meta = "已匹配到同 user_id 历史画像"

    top_left, top_right = st.columns([1.1, 0.9])
    with top_left:
        with st.container(border=True, key="input_ops_shell"):
            st.markdown(
                """
                <div class='module-card-head'>
                  <div class='module-card-title'>输入操作</div>
                  <div class='module-card-copy'>上传当前输入、切换系统模板，或围绕当前实体执行画像注入与写回。</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            with st.container(border=True, key="input_template_card"):
                st.markdown(
                    """
                    <div class='control-card-head'>
                      <div class='entity-info-value'>系统模板</div>
                      <div class='entity-info-meta'>三种标准模式都可以直接载入为当前输入，也可下载为正式 JSON。</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                with st.container(border=True, key="input_template_controls"):
                    selected_demo = st.selectbox(
                        "系统模板",
                        list(DEMO_PAYLOADS.keys()),
                        key="minor_detection_selected_demo",
                        label_visibility="collapsed",
                    )
                    action_left, action_right = st.columns([1, 1])
                    with action_left:
                        if st.button("载入当前模板", use_container_width=True, type="primary"):
                            demo_text = DEMO_PAYLOADS[selected_demo].read_text(encoding="utf-8")
                            set_input_text(demo_text)
                            set_input_origin(f"template:{selected_demo}")
                            st.session_state.minor_detection_uploaded_snapshot = ""
                            st.session_state.minor_detection_uploaded_name = ""
                            st.session_state.minor_detection_last_input_snapshot = demo_text
                            reset_workflow_to_input()
                            st.rerun()
                    with action_right:
                        st.download_button(
                            "下载当前模板",
                            data=DEMO_PAYLOADS[st.session_state.minor_detection_selected_demo].read_text(encoding="utf-8"),
                            file_name=DEMO_PAYLOADS[st.session_state.minor_detection_selected_demo].name,
                            mime="application/json",
                            use_container_width=True,
                            type="secondary",
                        )

            with st.container(border=True, key="input_profile_card"):
                st.markdown(
                    """
                    <div class='control-card-head'>
                      <div class='entity-info-value'>画像操作</div>
                      <div class='entity-info-meta'>围绕当前 user_id 读取历史画像，并将其注入到当前输入。</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                with st.container(border=True, key="input_profile_controls"):
                    if st.button("从画像库注入", use_container_width=True, type="primary"):
                        if inject_profile_from_store_into_input():
                            st.success("已将本地画像库中的历史画像注入当前输入。")
                            st.rerun()
                        else:
                            st.warning("当前 user_id 没有可注入的本地画像记录。")

    with top_right:
        chip_class = "identity-status-chip" if injected_available else "identity-status-chip muted"
        chip_label = "可注入历史画像" if injected_available else "暂无可注入画像"
        with st.container(border=True, key="current_entity_shell"):
            st.markdown(
                """
                <div class='module-card-head'>
                  <div class='module-card-title'>当前实体</div>
                  <div class='module-card-copy'>当前工作台已经生效的输入状态会在这里集中显示，并决定后续概览与识别路径。</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.markdown(
                "<div class='entity-card-grid'>"
                "<div class='entity-info-block'>"
                "<div class='entity-info-label'>输入来源</div>"
                f"<div class='entity-info-value'>{escape(format_input_origin_display(get_input_origin()))}</div>"
                "<div class='entity-info-meta'>当前工作台中已生效的输入来源</div>"
                "</div>"
                "<div class='entity-info-block'>"
                "<div class='entity-info-label'>当前模式</div>"
                f"<div class='entity-info-value'>{escape(current_mode)}</div>"
                "<div class='entity-info-meta'>系统会按该模式生成后续概览与识别链路</div>"
                "</div>"
                "<div class='entity-info-block user-id'>"
                "<div class='entity-info-label'>当前 user_id</div>"
                f"<div class='entity-info-value'>{escape(current_user or '未设置 user_id')}</div>"
                f"<div class='entity-info-meta'>画像库记录数：{len(store)}</div>"
                "</div>"
                "<div class='entity-info-block matching'>"
                "<div class='entity-info-label'>画像匹配状态</div>"
                f"<div class='entity-info-value'>{escape(profile_state)}</div>"
                f"<div class='entity-info-meta'>{escape(profile_meta)}</div>"
                f"<div class='{chip_class}'>{escape(chip_label)}</div>"
                "</div>"
                "</div>",
                unsafe_allow_html=True,
            )

    with st.container(border=True, key="json_editor_shell"):
        st.markdown(
            """
            <div class='module-card-head'>
              <div class='module-card-title'>JSON 编辑区</div>
              <div class='module-card-copy'>直接修改当前生效输入；保存后，下一步会基于这里的内容生成输入概览。</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown(
            "<div class='json-card-copy'>当前编辑内容始终代表工作台里的最新输入。</div>",
            unsafe_allow_html=True,
        )
        st.text_area(
            "JSON 输入",
            key="minor_detection_input_editor",
            height=560,
            label_visibility="collapsed",
            on_change=commit_input_editor,
        )


def render_input_snapshot(payload: AnalysisPayload) -> None:
    stats = payload_stats(payload)
    badges = [
        (f"模式 {stats['mode']}", "success"),
        (f"轮次 {stats['turns']}", "info"),
    ]
    if stats["sessions"]:
        badges.append((f"会话 {stats['sessions']}", "info"))
    badges.append(("已带历史画像" if stats["has_prior_profile"] else "无历史画像", "warning" if stats["has_prior_profile"] else "ghost"))
    badges.append(("外部 RAG 已注入" if stats["has_external_rag"] else "等待内置 RAG", "success" if stats["has_external_rag"] else "warning"))
    render_badges(badges)


def render_run_ready_summary(payload: AnalysisPayload) -> None:
    stats = payload_stats(payload)
    meta = payload.meta.model_dump() if hasattr(payload.meta, "model_dump") else payload.meta.dict()
    mode_label = format_mode_display(payload.mode)
    scope_text = (
        f"{stats['sessions']} 个会话窗口 / {stats['turns']} 条消息"
        if payload.mode == "multi_session"
        else f"1 个当前会话 / {stats['turns']} 条消息"
    )
    context_text = " / ".join(
        [
            "已注入历史画像" if stats["has_prior_profile"] else "无历史画像",
            "已注入外部 RAG" if stats["has_external_rag"] else "等待内置增强",
        ]
    )
    cards = [
        ("输入模式", mode_label, "本次概览所对应的输入形态"),
        ("用户 ID", str(meta.get("user_id") or "未设置"), "当前概览绑定的目标实体"),
        ("识别范围", scope_text, "当前输入将进入的检测窗口规模"),
        ("上下文状态", context_text, "历史画像与检索增强的注入状态"),
    ]
    cards_html = "".join(
        "<div class='run-meta-card'>"
        f"<div class='run-meta-label'>{escape(label)}</div>"
        f"<div class='run-meta-value'>{escape(value)}</div>"
        f"<div class='run-meta-sub'>{escape(meta_text)}</div>"
        "</div>"
        for label, value, meta_text in cards
    )
    st.markdown(f"<div class='run-meta-grid'>{cards_html}</div>", unsafe_allow_html=True)


def render_input_preview(payload: AnalysisPayload) -> None:
    st.markdown("<div class='preview-section-banner'>输入概览</div>", unsafe_allow_html=True)
    if payload.mode == "multi_session":
        sessions = list(payload.sessions)
        options = [f"S{i + 1}" for i in range(len(sessions))]
        slider_key = f"minor_detection_preview_window_{len(sessions)}"
        if options and st.session_state.get(slider_key) not in options:
            st.session_state[slider_key] = options[0]
        st.markdown(
            "<div class='preview-toolbar'>"
            "<div class='preview-toolbar-note'>多会话模式下支持按窗口查看单个会话，当前预览会展示所选窗口中的完整对话流。</div>"
            "</div>",
            unsafe_allow_html=True,
        )
        selected_label = st.select_slider(
            "浏览会话窗口",
            options=options,
            key=slider_key,
        )
        selected_index = options.index(selected_label) if selected_label in options else 0
        selected_session = sessions[selected_index]
        conversation = selected_session.conversation or []
        user_turns = sum(1 for turn in conversation if str(turn.get("role", "user")) == "user")
        assistant_turns = max(0, len(conversation) - user_turns)

        st.markdown(
            "<div class='session-meta-grid'>"
            "<div class='session-meta-card'>"
            "<div class='session-meta-label'>当前窗口</div>"
            f"<div class='session-meta-value'>{escape(selected_session.session_id or selected_label)}</div>"
            "</div>"
            "<div class='session-meta-card'>"
            "<div class='session-meta-label'>会话时间</div>"
            f"<div class='session-meta-value'>{escape(selected_session.session_time or '未标注时间')}</div>"
            "</div>"
            "<div class='session-meta-card'>"
            "<div class='session-meta-label'>对话轮次</div>"
            f"<div class='session-meta-value'>{len(conversation)} 条消息</div>"
            "</div>"
            "<div class='session-meta-card'>"
            "<div class='session-meta-label'>角色分布</div>"
            f"<div class='session-meta-value'>用户 {user_turns} / 助手 {assistant_turns}</div>"
            "</div>"
            "</div>",
            unsafe_allow_html=True,
        )

        bubbles: List[str] = []
        for turn in conversation:
            role = str(turn.get("role", "user"))
            content = str(turn.get("content", "")).strip() or "无内容"
            timestamp = str(turn.get("timestamp", "")).strip()
            role_label = "USER" if role == "user" else "ASSISTANT" if role == "assistant" else role.upper()
            role_avatar = "U" if role == "user" else "A" if role == "assistant" else role_label[:1]
            time_html = f"<div class='turn-sub'>{escape(timestamp)}</div>" if timestamp else ""
            bubbles.append(
                f"<div class='turn-bubble {escape(role)}'>"
                "<div class='turn-header'>"
                f"<div class='turn-role-badge'>{escape(role_avatar)}</div>"
                f"<div class='turn-role'>{escape(role_label)}</div>"
                "</div>"
                f"<div class='turn-content'>{escape(content)}</div>"
                f"{time_html}"
                "</div>"
            )
        st.markdown(f"<div class='transcript-shell'>{''.join(bubbles)}</div>", unsafe_allow_html=True)
    else:
        cards: List[str] = []
        turns = payload.conversation[:4]
        for turn in turns:
            role = str(turn.get("role", "user"))
            text = str(turn.get("content", "")).strip()
            timestamp = str(turn.get("timestamp", "")).strip()
            role_avatar = "U" if role == "user" else "A" if role == "assistant" else role[:1].upper()
            role_label = "USER" if role == "user" else "ASSISTANT" if role == "assistant" else role.upper()
            time_html = f"<div class='preview-time'>{escape(timestamp)}</div>" if timestamp else ""
            cards.append(
                f"<div class='preview-card {escape(role)}'>"
                "<div class='preview-header'>"
                f"<div class='preview-role-badge'>{escape(role_avatar)}</div>"
                f"<div class='preview-role'>{escape(role_label)}</div>"
                "</div>"
                f"<p class='preview-text'>{escape(text[:190] or '无内容')}</p>"
                f"{time_html}"
                "</div>"
            )
        st.markdown(f"<div class='preview-list'>{''.join(cards)}</div>", unsafe_allow_html=True)


def metric_card(title: str, value: str, help_text: str, klass: str = "", extra_html: str = "") -> str:
    return (
        f"<div class='metric-card {escape(klass)}'>"
        f"<div class='metric-label'>{escape(title)}</div>"
        "<div class='metric-body'>"
        f"<div class='metric-value'>{escape(value)}</div>"
        f"<p class='metric-help'>{escape(help_text)}</p>"
        f"{extra_html}"
        "</div>"
        "</div>"
    )


def render_overview(output: FormalSkillOutput, mode: str) -> None:
    risk_value = output.decision.risk_level.value if hasattr(output.decision.risk_level, "value") else str(output.decision.risk_level)
    risk_key = f"risk-{risk_value.lower()}"
    decision_value = "疑似未成年人" if output.decision.is_minor else "疑似成年人"
    confidence_pct = output.decision.minor_confidence * 100
    next_step = NEXT_STEP_LABELS.get(output.recommended_next_step.value, output.recommended_next_step.value)
    probability_title = "综合未成年人概率" if mode == "multi_session" else "当前会话未成年人概率"
    probability_help = "综合多个会话窗口后的总判定概率。" if mode == "multi_session" else f"置信带 {output.decision.confidence_band.value.upper()}"
    decision_help = "基于多会话轨迹与增强上下文的综合判定。" if mode == "multi_session" else "基于当前会话与增强上下文的综合判定。"

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(
            metric_card("识别结论", decision_value, decision_help),
            unsafe_allow_html=True,
        )
    with col2:
        st.markdown(
            metric_card(
                probability_title,
                f"{confidence_pct:.1f}%",
                probability_help,
                extra_html=f"<div class='confidence-track'><div class='confidence-fill' style='width:{confidence_pct:.1f}%'></div></div>",
            ),
            unsafe_allow_html=True,
        )
    with col3:
        st.markdown(
            metric_card("风险等级", RISK_LABELS.get(risk_value, risk_value), "越高意味着越需要尽快人工或策略介入。", risk_key),
            unsafe_allow_html=True,
        )
    with col4:
        st.markdown(
            metric_card("推荐动作", next_step, "系统给出的下一步处置建议。"),
            unsafe_allow_html=True,
        )


def render_probability_trend(run_result: Dict[str, Any]) -> None:
    points = [point for point in run_result.get("session_curve", []) if point.get("minor_confidence") is not None]
    if not points:
        return

    render_section_banner("递进更新未成年人概率")
    st.markdown(
        "<div class='trend-lead'>"
        "折线图中的第 N 个点表示系统读完前 N 个会话、完成画像与证据累积更新后的当前判断概率；"
        "最后一个点与综合多会话总概率保持一致。"
        "</div>",
        unsafe_allow_html=True,
    )
    x_values = [point.get("session_id", f"S{index + 1}") for index, point in enumerate(points)]
    y_values = [float(point["minor_confidence"]) for point in points]
    hover_text = [
        "<br>".join(
            [
                f"会话: {point.get('session_id', '-')}",
                f"时间: {point.get('session_time', '未标注')}",
                f"概率: {float(point['minor_confidence']):.1%}",
                f"风险: {point.get('risk_level', '-')}",
                f"阶段: {point.get('education_stage', '未明确')}",
            ]
        )
        for point in points
    ]

    figure = go.Figure()
    figure.add_trace(
        go.Scatter(
            x=x_values,
            y=y_values,
            mode="lines+markers",
            line={"color": "#7fd7ff", "width": 4, "shape": "spline"},
            marker={
                "size": 11,
                "color": "#ff5968",
                "line": {"width": 2, "color": "#ffffff"},
            },
            fill="tozeroy",
            fillcolor="rgba(72, 187, 255, 0.15)",
            hovertemplate="%{text}<extra></extra>",
            text=hover_text,
        )
    )
    figure.update_layout(
        height=320,
        margin=dict(l=18, r=18, t=14, b=18),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(12,29,52,0.65)",
        xaxis=dict(
            title="会话窗口",
            showgrid=False,
            tickfont=dict(color="#e9f2ff"),
            title_font=dict(color="#dce9ff"),
        ),
        yaxis=dict(
            title="未成年人概率",
            range=[0, 1],
            tickformat=".0%",
            gridcolor="rgba(145,177,227,0.18)",
            zeroline=False,
            tickfont=dict(color="#e9f2ff"),
            title_font=dict(color="#dce9ff"),
        ),
        font=dict(color="#eef5ff"),
    )
    st.plotly_chart(figure, use_container_width=True, config={"displayModeBar": False})

    overall_output: FormalSkillOutput = run_result["formal_output"]
    overall_confidence = overall_output.decision.minor_confidence
    latest_confidence = y_values[-1]
    delta = latest_confidence - y_values[0] if len(y_values) > 1 else 0.0
    st.markdown(
        "<div class='mini-grid'>"
        "<div class='info-card'>"
        "<div class='info-label'>综合多会话总概率</div>"
        f"<div class='info-value'>{overall_confidence:.1%}</div>"
        "</div>"
        "<div class='info-card'>"
        "<div class='info-label'>递进末点概率</div>"
        f"<div class='info-value'>{latest_confidence:.1%}</div>"
        "</div>"
        "<div class='info-card'>"
        "<div class='info-label'>相对首会话变化</div>"
        f"<div class='info-value'>{delta:+.1%}</div>"
        "</div>"
        "<div class='info-card'>"
        "<div class='info-label'>递进更新轮次</div>"
        f"<div class='info-value'>{len(points)}</div>"
        "</div>"
        "</div>",
        unsafe_allow_html=True,
    )

    session_cards: List[str] = []
    for point in points:
        confidence = float(point["minor_confidence"])
        risk_text = point.get("risk_level", "-") or "-"
        stage_text = point.get("education_stage", "未明确") or "未明确"
        session_cards.append(
            "<div class='trajectory-card'>"
            f"<div class='trajectory-session'>{escape(point.get('session_id', '-'))}</div>"
            f"<div class='trajectory-time'>{escape(point.get('session_time', '未标注时间'))}</div>"
            f"<div class='trajectory-probability'>{confidence:.1%}</div>"
            f"<div class='trajectory-meta'>风险等级：{escape(risk_text)}<br/>识别阶段：{escape(stage_text)}</div>"
            f"<div class='trajectory-track'><div class='trajectory-fill' style='width:{confidence * 100:.1f}%'></div></div>"
            "</div>"
        )
    st.markdown(f"<div class='trajectory-grid'>{''.join(session_cards)}</div>", unsafe_allow_html=True)


def render_run_context_strip(payload_dict: Dict[str, Any], source_context: Dict[str, Any]) -> None:
    meta = payload_dict.get("meta", {}) or {}
    mode = payload_dict.get("mode", "-")
    external_rag_count = len(source_context.get("retrieved_cases", []) or [])
    has_profile = bool(source_context.get("prior_profile"))
    input_origin = get_input_origin()
    if input_origin == "upload":
        request_source_meta = str(st.session_state.get("minor_detection_uploaded_name", "") or "").strip() or "未命名文件"
    else:
        request_source_meta = format_source_display(str(meta.get("source") or "minor_detection_frontend"))
    cards = [
        (
            "运行模式",
            format_mode_display(mode),
            "当前结果所对应的输入形态",
        ),
        (
            "用户 ID",
            str(meta.get("user_id") or "未设置"),
            "本次识别所绑定的当前实体",
        ),
        (
            "请求来源",
            format_input_origin_display(input_origin),
            request_source_meta,
        ),
        (
            "外部上下文",
            "历史画像已注入" if has_profile else "未注入历史画像",
            f"外部 RAG {external_rag_count} 条",
        ),
    ]
    cards_html = "".join(
        "<div class='context-summary-card'>"
        f"<div class='context-summary-label'>{escape(label)}</div>"
        f"<div class='context-summary-value'>{escape(value)}</div>"
        f"<div class='context-summary-meta'>{escape(meta_text)}</div>"
        "</div>"
        for label, value, meta_text in cards
    )
    st.markdown(f"<div class='context-summary-grid'>{cards_html}</div>", unsafe_allow_html=True)


def format_icbo_value(label: str, value: str) -> str:
    text = str(value or "").strip()
    if label == "Opportunity Time":
        if text.startswith("归一化标签:"):
            text = text.split(":", 1)[1].strip()
        text = text.replace(",", " · ")
    return text or "未明确"


def render_profile_and_icbo(output: FormalSkillOutput) -> None:
    left, right = st.columns([0.92, 1.08])
    with left:
        chips = "".join(f"<span class='value-chip'>{escape(item)}</span>" for item in output.user_profile.identity_markers)
        chips_html = f"<div class='chip-collection'>{chips}</div>" if chips else "<p class='evidence-empty'>当前输出未给出明确身份标签。</p>"
        if output.user_profile.identity_markers:
            chips_html = f"<div class='chip-collection'>{chips}</div>"
        st.markdown(
            "<div class='section-shell equal-panel' style='min-height:420px;'>"
            "<div class='section-heading'>用户画像</div>"
            "<div class='mini-grid'>"
            "<div class='info-card'>"
            "<div class='info-label'>年龄范围</div>"
            f"<div class='info-value'>{escape(output.user_profile.age_range)}</div>"
            "</div>"
            "<div class='info-card'>"
            "<div class='info-label'>教育阶段</div>"
            f"<div class='info-value'>{escape(output.user_profile.education_stage)}</div>"
            "</div>"
            "</div>"
            f"{chips_html}"
            "</div>",
            unsafe_allow_html=True,
        )

    with right:
        items = [
            ("Intention", output.icbo_features.intention),
            ("Cognition", output.icbo_features.cognition),
            ("Behavior Style", output.icbo_features.behavior_style),
            ("Opportunity Time", output.icbo_features.opportunity_time),
        ]
        info_cards = "".join(
            "<div class='info-card'>"
            f"<div class='info-label'>{escape(label)}</div>"
            f"<div class='info-value'>{escape(format_icbo_value(label, value))}</div>"
            "</div>"
            for label, value in items
        )
        st.markdown(
            "<div class='section-shell equal-panel' style='min-height:420px;'>"
            "<div class='section-heading'>ICBO 结构化分析</div>"
            f"<div class='mini-grid'>{info_cards}</div>"
            "</div>",
            unsafe_allow_html=True,
        )


def build_evidence_card(title: str, items: List[str], klass: str) -> str:
    if items:
        rendered_items = "".join(f"<li>{escape(item)}</li>" for item in items)
        body = f"<ul class='evidence-list'>{rendered_items}</ul>"
    else:
        body = "<p class='evidence-empty'>当前没有命中该类证据。</p>"
    return (
        f"<div class='evidence-card {escape(klass)}'>"
        f"<div class='evidence-header'>"
        f"<div class='evidence-title'>{escape(title)}</div>"
        f"<div class='evidence-count'>{len(items)}</div>"
        f"</div>{body}</div>"
    )


def parse_retrieval_evidence_item(item: str) -> Optional[Dict[str, str]]:
    text = str(item or "").strip()
    if not text.startswith("sample_id="):
        return None

    match = re.match(
        r"sample_id=(?P<sample_id>[^;]+);\s*label=(?P<label>[^;]+);\s*score=(?P<score>[^;]+);\s*summary=(?P<summary>.*)",
        text,
        flags=re.DOTALL,
    )
    if not match:
        return None

    parsed = {key: value.strip() for key, value in match.groupdict().items()}
    summary = re.sub(r"\s+", " ", parsed.get("summary", "")).strip()
    if len(summary) > 180:
        summary = summary[:177].rstrip() + "..."
    parsed["summary"] = summary
    return parsed


def build_retrieval_evidence_card(items: List[str]) -> str:
    structured_items: List[Dict[str, str]] = []
    summary_items: List[str] = []
    for item in items:
        parsed = parse_retrieval_evidence_item(item)
        if parsed:
            structured_items.append(parsed)
        elif item:
            summary_items.append(str(item))

    sections: List[str] = []
    if summary_items:
        rendered_items = "".join(f"<li>{escape(item)}</li>" for item in summary_items)
        sections.append(
            "<div class='evidence-subsection'>"
            "<div class='evidence-subsection-title'>检索结论摘要</div>"
            f"<ul class='evidence-list'>{rendered_items}</ul>"
            "</div>"
        )
    if structured_items:
        cards = []
        for item in structured_items[:4]:
            score_text = item.get("score", "")
            try:
                score_text = f"{float(score_text):.3f}"
            except Exception:
                score_text = str(score_text)
            cards.append(
                "<div class='retrieval-card'>"
                "<div class='retrieval-topline'>"
                f"<div class='retrieval-id'>{escape(item.get('sample_id', 'unknown_case'))}</div>"
                f"<div class='retrieval-meta'>{escape(item.get('label', 'unknown'))} · score {escape(score_text)}</div>"
                "</div>"
                f"<p class='retrieval-summary'>{escape(item.get('summary', ''))}</p>"
                "</div>"
            )
        sections.append(
            "<div class='evidence-subsection'>"
            "<div class='evidence-subsection-title'>命中案例引用</div>"
            f"<div class='retrieval-grid'>{''.join(cards)}</div>"
            "</div>"
        )

    body = "".join(sections) if sections else "<p class='evidence-empty'>当前没有命中该类证据。</p>"
    return (
        "<div class='evidence-card retrieval'>"
        "<div class='evidence-header'>"
        "<div class='evidence-title'>RAG 检索证据</div>"
        f"<div class='evidence-count'>{len(items)}</div>"
        f"</div>{body}</div>"
    )


def render_evidence_chain(output: FormalSkillOutput) -> None:
    cards = [
        build_evidence_card("用户直接证据", output.evidence.direct_evidence, "direct"),
        build_evidence_card("历史画像证据", output.evidence.historical_evidence, "historical"),
        build_retrieval_evidence_card(output.evidence.retrieval_evidence),
        build_evidence_card("时间作息证据", output.evidence.time_evidence, "time"),
        build_evidence_card("反向/冲突信号", output.evidence.conflicting_signals, "conflict"),
    ]
    render_section_shell("多源证据总览", f"<div class='evidence-stack'>{''.join(cards)}</div>")


def render_time_and_runtime(context: Dict[str, Any]) -> None:
    time_features = context.get("time_features", {}) or {}
    runtime_meta = context.get("_formal_runtime", {}) or {}

    badges = []
    if runtime_meta.get("rag_mode_used"):
        rag_label = RAG_MODE_DISPLAY.get(runtime_meta["rag_mode_used"], runtime_meta["rag_mode_used"])
        badges.append((rag_label, "success" if runtime_meta["rag_mode_used"] != "no_rag" else "ghost"))
    if runtime_meta.get("time_mode_used"):
        time_label = TIME_MODE_DISPLAY.get(runtime_meta["time_mode_used"], runtime_meta["time_mode_used"])
        badges.append((time_label, "warning"))
    if runtime_meta.get("rag_reason"):
        badges.append((runtime_meta["rag_reason"], "ghost"))

    content_parts: List[str] = []
    if badges:
        badge_html = "".join(
            f"<span class='badge-pill {escape(tone)}'>{escape(label)}</span>"
            for label, tone in badges
        )
        content_parts.append(f"<div class='badge-row'>{badge_html}</div>")
    if time_features:
        chips = []
        for key, label in TIME_FEATURE_LABELS.items():
            value = time_features.get(key)
            if value in (None, "", []):
                continue
            if value is True:
                display = "是"
            elif value is False:
                display = "否"
            else:
                normalized = str(value).strip().lower()
                if key == "weekday":
                    display = WEEKDAY_DISPLAY.get(normalized, value)
                elif key == "time_bucket":
                    display = TIME_BUCKET_DISPLAY.get(normalized, value)
                elif key == "holiday_label":
                    display = HOLIDAY_DISPLAY.get(normalized, value)
                else:
                    display = value
            chips.append(f"<span class='value-chip'>{escape(label)} · {escape(display)}</span>")
        content_parts.append(f"<div class='chip-collection'>{''.join(chips)}</div>")
    else:
        content_parts.append("<p class='evidence-empty'>当前没有可展示的时间标签特征。</p>")
    render_section_shell("时间标签与运行态", "".join(content_parts))


def render_retrieval_cases(injected_context: Dict[str, Any], output: FormalSkillOutput) -> None:
    retrieved_cases = injected_context.get("retrieved_cases", []) or []
    if not retrieved_cases:
        render_section_shell("外部注入的 RAG 上下文", "<p class='evidence-empty'>当前没有外部注入案例。</p>")
        return

    cards = []
    for case in retrieved_cases[:5]:
        case_id = case.get("sample_id", "unknown_case")
        score = case.get("score", 0)
        label = case.get("label", "unknown")
        summary = case.get("summary", "")
        key_signals = case.get("key_signals", []) or []
        chips = "".join(f"<span class='value-chip'>{escape(item)}</span>" for item in key_signals[:5])
        cards.append(
            "<div class='retrieval-card'>"
            "<div class='retrieval-topline'>"
            f"<div class='retrieval-id'>{escape(case_id)}</div>"
            f"<div class='retrieval-meta'>{escape(label)} · score {float(score):.3f}</div>"
            "</div>"
            f"<p class='retrieval-summary'>{escape(summary)}</p>"
            f"<div class='chip-collection'>{chips}</div>"
            "</div>"
        )
    render_section_shell("外部注入的 RAG 上下文", f"<div class='retrieval-grid'>{''.join(cards)}</div>")


def render_prior_profile(injected_context: Dict[str, Any]) -> None:
    prior_profile = injected_context.get("prior_profile") or {}
    if not prior_profile:
        return

    total_sessions = prior_profile.get("total_sessions", "-")
    age_range = prior_profile.get("estimated_age_range", "未明确")
    stage = prior_profile.get("education_stage", "未明确")
    trend = prior_profile.get("recent_confidence_trend", "未明确")
    summary = prior_profile.get("summary", "")
    identity_markers = prior_profile.get("identity_markers", []) or []

    body = (
        "<div class='mini-grid'>"
        "<div class='info-card'>"
        "<div class='info-label'>历史会话数</div>"
        f"<div class='info-value'>{escape(total_sessions)}</div>"
        "</div>"
        "<div class='info-card'>"
        "<div class='info-label'>历史年龄范围</div>"
        f"<div class='info-value'>{escape(age_range)}</div>"
        "</div>"
        "<div class='info-card'>"
        "<div class='info-label'>历史教育阶段</div>"
        f"<div class='info-value'>{escape(stage)}</div>"
        "</div>"
        "<div class='info-card'>"
        "<div class='info-label'>置信趋势</div>"
        f"<div class='info-value'>{escape(trend)}</div>"
        "</div>"
        "</div>"
    )
    if identity_markers:
        chips = "".join(f"<span class='value-chip'>{escape(item)}</span>" for item in identity_markers)
        body += f"<div class='chip-collection'>{chips}</div>"
    if summary:
        body += f"<div class='reasoning-box'>{escape(summary)}</div>"
    render_section_shell("注入的历史画像", body)


def render_reasoning_and_notes(output: FormalSkillOutput) -> None:
    body = f"<div class='reasoning-box'>{escape(output.reasoning_summary)}</div>"
    if output.uncertainty_notes:
        chips = "".join(f"<span class='value-chip'>{escape(note)}</span>" for note in output.uncertainty_notes)
        body += "<div class='chip-collection'>" + chips + "</div>"
    render_section_shell("结论说明", body)


def render_results_waiting() -> None:
    st.markdown(
        """
        <div class="placeholder-shell">
          <div class="placeholder-panel placeholder-visual">
            <div class="placeholder-glow"></div>
            <div class="placeholder-node primary">
              <div class="placeholder-kicker">Live Analysis</div>
              <span class="placeholder-line long"></span>
              <span class="placeholder-line medium"></span>
              <span class="placeholder-line short"></span>
            </div>
            <div class="placeholder-node secondary">
              <div class="placeholder-kicker">Evidence Chain</div>
              <span class="placeholder-line medium"></span>
              <span class="placeholder-line long"></span>
            </div>
            <div class="placeholder-node tertiary">
              <div class="placeholder-kicker">RAG</div>
              <span class="placeholder-line short"></span>
              <span class="placeholder-line medium"></span>
            </div>
          </div>
          <div class="placeholder-panel">
            <div class="eyebrow">Detection Running</div>
            <div class="placeholder-title">系统正在执行未成年人识别</div>
            <p class="placeholder-copy">
              当前识别任务已启动，系统正在解析输入、注入时间特征与检索增强，并整理结构化识别结果。
              完成后将自动进入结果页，展示风险等级、递进更新概率、时间标签、用户画像与证据链。
            </p>
            <div class="placeholder-mini-grid">
              <div class="placeholder-mini-card">
                <div class="placeholder-kicker">Result</div>
                <span class="placeholder-line medium"></span>
                <span class="placeholder-line short"></span>
              </div>
              <div class="placeholder-mini-card">
                <div class="placeholder-kicker">Profile</div>
                <span class="placeholder-line long"></span>
                <span class="placeholder-line medium"></span>
              </div>
            </div>
            <p class="placeholder-footnote">
              多会话输入会额外计算递进更新概率曲线，用于展示系统如何在连续会话中逐步修正未成年人判断。
            </p>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_results_panel(run_result: Dict[str, Any]) -> None:
    output: FormalSkillOutput = run_result["formal_output"]
    final_payload = run_result["final_payload"]
    payload_dict = final_payload.model_dump() if hasattr(final_payload, "model_dump") else final_payload.dict()
    enriched_context = payload_dict.get("context", {})
    normalized_payload = run_result["normalized_payload"]
    source_context = (
        normalized_payload.context.model_dump()
        if hasattr(normalized_payload.context, "model_dump")
        else normalized_payload.context.dict()
    )
    mode = run_result["normalized_payload"].mode

    render_overview(output, mode)
    render_run_context_strip(payload_dict, source_context)
    st.markdown("<div class='result-panel-shell'></div>", unsafe_allow_html=True)

    summary_tab, evidence_tab, system_tab, json_tab = st.tabs(
        ["识别总览", "证据链", "系统上下文", "原始 JSON"]
    )

    with summary_tab:
        if run_result["normalized_payload"].mode == "multi_session":
            render_probability_trend(run_result)
        render_profile_and_icbo(output)
        render_reasoning_and_notes(output)
        render_time_and_runtime(enriched_context)

    with evidence_tab:
        render_evidence_chain(output)

    with system_tab:
        summary_html = (
            "<div class='mini-grid'>"
            "<div class='info-card'>"
            "<div class='info-label'>外部 RAG</div>"
            f"<div class='info-value'>{len(source_context.get('retrieved_cases', []) or [])} 条</div>"
            "</div>"
            "<div class='info-card'>"
            "<div class='info-label'>历史画像</div>"
            f"<div class='info-value'>{'已注入' if source_context.get('prior_profile') else '未注入'}</div>"
            "</div>"
            "</div>"
        )
        render_section_shell("外部注入上下文", summary_html)
        render_retrieval_cases(source_context, output)
        render_prior_profile(source_context)

    with json_tab:
        output_dict = output.model_dump() if hasattr(output, "model_dump") else output.dict()
        payload_json = pretty_json(payload_dict)
        output_json = pretty_json(output_dict)
        payload_left, payload_right = st.columns([1, 1])
        with payload_left:
            st.markdown("<div class='section-heading'>Formal Payload</div>", unsafe_allow_html=True)
        with payload_right:
            st.download_button(
                "下载 Formal Payload",
                data=payload_json,
                file_name="minor_detection_formal_payload.json",
                mime="application/json",
                use_container_width=True,
            )
        st.text_area(
            "Formal Payload",
            value=payload_json,
            height=420,
            disabled=True,
            label_visibility="collapsed",
            key="minor_detection_payload_viewer",
        )

        output_left, output_right = st.columns([1, 1])
        with output_left:
            st.markdown("<div class='section-heading'>Formal Output</div>", unsafe_allow_html=True)
        with output_right:
            st.download_button(
                "下载 Formal Output",
                data=output_json,
                file_name="minor_detection_formal_output.json",
                mime="application/json",
                use_container_width=True,
            )
        st.text_area(
            "Formal Output",
            value=output_json,
            height=420,
            disabled=True,
            label_visibility="collapsed",
            key="minor_detection_output_viewer",
        )


def main() -> None:
    inject_styles()
    ensure_state()
    sync_input_snapshot()

    preview_payload: Optional[AnalysisPayload] = None
    preview_error: Optional[str] = None
    try:
        preview_payload = AnalysisPayload(**normalize_uploaded_payload(json.loads(get_input_text())))
    except Exception as exc:
        preview_error = str(exc)

    available_steps = get_available_steps(preview_payload)
    current_step = resolve_current_step(available_steps)
    render_hero()
    render_section_banner("识别工作流")
    render_workflow_navigator(current_step, available_steps)

    if current_step == 1:
        render_input_panel()
        if st.button("进入概览", type="primary", use_container_width=True, disabled=preview_payload is None):
            jump_to_step(2)
            st.rerun()
        if preview_payload is None:
            st.markdown(
                f"<p class='tiny-note'>当前输入尚未通过解析：{escape(preview_error or '未知错误')}。请先修正 JSON 后再进入下一步。</p>",
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                "<p class='tiny-note'>当前输入已就绪。确认上传或编辑完成后，进入下一步查看输入概览。</p>",
                unsafe_allow_html=True,
            )

    elif current_step == 2:
        if preview_payload is not None:
            render_input_snapshot(preview_payload)
            render_input_preview(preview_payload)
            render_run_ready_summary(preview_payload)
            left, right = st.columns([1, 1])
            with left:
                if st.button("上一步", use_container_width=True):
                    jump_to_step(1)
                    st.rerun()
            with right:
                if st.button("运行 Minor Detection", type="primary", use_container_width=True):
                    clear_last_result()
                    st.session_state.minor_detection_pending_run = True
                    jump_to_step(3)
                    st.rerun()
        else:
            st.error(f"输入预览失败：{preview_error}")
            if st.button("返回配置输入", use_container_width=True):
                jump_to_step(1)
                st.rerun()

    elif current_step == 3:
        st.markdown(
            """
            <style>
            .preview-list,
            .preview-card,
            .preview-toolbar,
            .session-meta-grid,
            .transcript-shell,
            .turn-bubble,
            .run-meta-grid,
            .stButton,
            [data-baseweb="slider"],
            [data-testid="stSlider"],
            [data-testid="stWidgetLabel"],
            [data-testid="stHorizontalBlock"] {
                display: none !important;
            }
            </style>
            """,
            unsafe_allow_html=True,
        )
        stage_host = st.empty()
        with stage_host.container():
            st.markdown("<div class='progress-stage-shell'>", unsafe_allow_html=True)
            render_results_waiting()
            progress_hint = st.empty()
            progress_bar = st.progress(0)
            st.markdown("</div>", unsafe_allow_html=True)

        def update_progress(progress: int, message: str) -> None:
            progress_hint.markdown(
                f"<div class='progress-card'><div class='section-heading'>检测进度</div><p class='section-lead' style='margin-bottom:0.2rem;'>{escape(message)}</p></div>",
                unsafe_allow_html=True,
            )
            progress_bar.progress(progress)

        if st.session_state.get("minor_detection_pending_run"):
            try:
                payload_dict = json.loads(get_input_text())
                st.session_state.minor_detection_last_result = prepare_and_run(
                    payload_dict,
                    progress_callback=update_progress,
                )
                st.session_state.minor_detection_pending_run = False
                jump_to_step(4)
                st.rerun()
            except Exception as exc:
                st.session_state.minor_detection_pending_run = False
                st.session_state.minor_detection_last_result = {"error": str(exc)}
                jump_to_step(4)
                st.rerun()
        else:
            jump_to_step(2)
            st.rerun()

    else:
        st.markdown("<div class='page-spacer-sm'></div>", unsafe_allow_html=True)
        last_result = st.session_state.get("minor_detection_last_result")
        can_save_result = isinstance(last_result, dict) and "formal_output" in last_result
        save_feedback: Optional[Tuple[str, str]] = None
        with st.container(key="result_action_bar"):
            top_left, top_mid, top_right, top_save = st.columns([1, 1, 1.2, 1.4])
            with top_left:
                if st.button("开始下一次识别", type="primary", use_container_width=True):
                    reset_workflow_to_input()
                    st.rerun()
            with top_mid:
                if st.button("返回配置输入", use_container_width=True):
                    jump_to_step(1)
                    st.rerun()
            with top_right:
                if st.button("回到输入概览", use_container_width=True, disabled=preview_payload is None):
                    jump_to_step(2)
                    st.rerun()
            with top_save:
                if st.button(
                    "写入当前结果到画像库",
                    type="secondary",
                    use_container_width=True,
                    disabled=not can_save_result,
                ):
                    if save_last_result_to_store():
                        save_feedback = ("success", "当前识别结果已写入本地画像库。")
                    else:
                        save_feedback = ("warning", "当前结果暂不可写入，请确认本次识别已完成且输入中带有 user_id。")
        st.markdown("<div class='result-actions-shell'></div>", unsafe_allow_html=True)
        if save_feedback:
            level, message = save_feedback
            with st.container(key="result_feedback_shell"):
                st.markdown(
                    f"<div class='result-feedback {escape(level)}'>{escape(message)}</div>",
                    unsafe_allow_html=True,
                )

        if not st.session_state.minor_detection_last_result:
            render_results_waiting()
        elif "error" in st.session_state.minor_detection_last_result:
            st.error(st.session_state.minor_detection_last_result["error"])
        else:
            render_results_panel(st.session_state.minor_detection_last_result)


if __name__ == "__main__":
    main()

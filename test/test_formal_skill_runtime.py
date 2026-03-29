# 模块说明：
# - 覆盖 formal runtime 适配层和 bundled skill helper 的测试。
# - 偏向调试运行时桥接和 fallback 行为。

import importlib.util
import io
import json
import subprocess
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from src.models import AnalysisPayload, ICBOFeatures, RiskLevel, SkillOutput, UserPersona
from src.executor.executor import _coerce_trend_trajectory
from src.runtime import (
    analyze_formal_payload,
    analyze_multi_session_formal_auto,
    analyze_single_session_formal_auto,
    build_formal_multi_session_payload,
    build_formal_single_session_payload,
    enrich_multi_session_context,
    enrich_single_session_context,
    get_formal_skill_path,
)


ROOT_DIR = Path(__file__).resolve().parents[1]
CLASSIFIER_CLIENT_PATH = ROOT_DIR / "skills" / "minor-detection" / "scripts" / "_classifier_client.py"
RETRIEVE_CASES_PATH = ROOT_DIR / "skills" / "minor-detection" / "scripts" / "retrieve_cases.py"
TIME_SCRIPT_PATH = ROOT_DIR / "skills" / "minor-detection" / "scripts" / "extract_time_features.py"
PIPELINE_SCRIPT_PATH = ROOT_DIR / "skills" / "minor-detection" / "scripts" / "run_minor_detection_pipeline.py"
RETRIEVAL_CORPUS_PATH = ROOT_DIR / "skills" / "minor-detection" / "assets" / "retrieval_assets" / "corpus.jsonl"
PROFILE_MERGE_PATH = ROOT_DIR / "skills" / "minor-detection" / "scripts" / "_profile_merge.py"


def make_skill_output(is_minor: bool, confidence: float) -> SkillOutput:
    return SkillOutput(
        is_minor=is_minor,
        minor_confidence=confidence,
        risk_level=RiskLevel.LOW if is_minor else RiskLevel.MEDIUM,
        icbo_features=ICBOFeatures(
            intention="intent",
            cognition="cognition",
            behavior_style="behavior",
            opportunity_time="time",
        ),
        user_persona=UserPersona(
            age=15 if is_minor else 22,
            age_range="14-16" if is_minor else "21-24",
            education_stage="高中" if is_minor else "大学",
            identity_markers=["marker"],
        ),
        reasoning="ok",
        key_evidence=["evidence"],
    )


def load_retrieve_cases_module():
    spec = importlib.util.spec_from_file_location("minor_detection_retrieve_cases", RETRIEVE_CASES_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def load_classifier_client_module():
    spec = importlib.util.spec_from_file_location("minor_detection_classifier_client", CLASSIFIER_CLIENT_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def load_profile_merge_module():
    spec = importlib.util.spec_from_file_location("minor_detection_profile_merge", PROFILE_MERGE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def run_time_script(timestamp: str) -> dict:
    completed = subprocess.run(
        [sys.executable, str(TIME_SCRIPT_PATH), "--timestamp", timestamp],
        capture_output=True,
        text=True,
        check=True,
    )
    return json.loads(completed.stdout.strip())


class FormalSkillRuntimeTests(unittest.TestCase):
    def test_coerce_trend_trajectory_discards_string_placeholder_items(self):
        parsed = {
            "trend": {
                "trajectory": ["trajectory_1", "trajectory_2", "trajectory_3"],
                "trend_summary": "placeholder",
            }
        }

        coerced = _coerce_trend_trajectory(parsed)

        self.assertEqual(coerced["trend"]["trajectory"], [])

    def test_profile_merge_does_not_fabricate_multi_session_trajectory(self):
        profile_merge = load_profile_merge_module()
        output = {
            "decision": {"is_minor": True, "minor_confidence": 0.9},
            "trend": {"trajectory": [], "trend_summary": ""},
        }
        normalized_payload = {
            "mode": "multi_session",
            "sessions": [
                {"session_id": "s1", "session_time": "2026-03-19T22:18:00+08:00"},
                {"session_id": "s2", "session_time": "2026-03-22T23:36:00+08:00"},
            ],
            "context": {},
            "identity_hints": [],
        }

        merged = profile_merge.merge_output(output, normalized_payload)

        self.assertEqual(merged["trend"]["trajectory"], [])

    def test_formal_skill_path_points_to_new_skill_package(self):
        skill_path = get_formal_skill_path()
        self.assertTrue(skill_path.exists())
        self.assertEqual(skill_path.name, "SKILL.md")
        self.assertEqual(skill_path.parent.name, "minor-detection")

    def test_pipeline_entrypoint_exists_in_skill_bundle(self):
        self.assertTrue(PIPELINE_SCRIPT_PATH.exists())

    def test_classifier_client_retries_transient_timeouts(self):
        classifier_client = load_classifier_client_module()

        class FakeResponse:
            def raise_for_status(self):
                return None

            def json(self):
                return {"choices": [{"message": {"content": "{\"ok\": true}"}}]}

        with mock.patch.object(
            classifier_client.requests,
            "post",
            side_effect=[classifier_client.requests.Timeout("timed out"), FakeResponse()],
        ) as post_mock, mock.patch.object(classifier_client.time, "sleep") as sleep_mock:
            parsed, raw_text = classifier_client.call_chat_completion(
                base_url="https://example.com/v1",
                api_key="test-key",
                model="test-model",
                timeout_sec=5,
                max_retries=2,
                retry_backoff_sec=0.5,
                messages=[{"role": "user", "content": "hello"}],
            )

        self.assertEqual(parsed, {"ok": True})
        self.assertEqual(raw_text, "{\"ok\": true}")
        self.assertEqual(post_mock.call_count, 2)
        sleep_mock.assert_called_once_with(0.5)

    def test_build_formal_single_session_payload_merges_meta_and_context(self):
        payload = build_formal_single_session_payload(
            conversation=[{"role": "user", "content": "我今天上学有点烦"}],
            user_id="u-1",
            request_id="r-1",
            source="web-demo",
            context={
                "channel": "site",
                "locale": "zh-CN",
                "time_features": {"weekday": "Wednesday"},
                "retrieved_cases": [{"sample_id": "s1"}],
                "prior_profile": {"summary": "疑似中学生画像"},
                "raw_time_hint": "2026-03-18 周三 08:59",
            },
        )

        self.assertEqual(payload.mode, "single_session")
        self.assertEqual(payload.meta.user_id, "u-1")
        self.assertEqual(payload.meta.request_id, "r-1")
        self.assertEqual(payload.meta.source, "web-demo")
        self.assertEqual(payload.context.channel, "site")
        self.assertEqual(payload.context.locale, "zh-CN")
        self.assertEqual(payload.context.time_features["weekday"], "Wednesday")
        self.assertEqual(payload.context.retrieved_cases[0]["sample_id"], "s1")
        self.assertEqual(payload.context.prior_profile["summary"], "疑似中学生画像")
        self.assertEqual(payload.context.raw_time_hint, "2026-03-18 周三 08:59")

    def test_build_formal_multi_session_payload_merges_meta_and_context(self):
        payload = build_formal_multi_session_payload(
            sessions=[
                {
                    "session_id": "s1",
                    "session_time": "2026-03-18T23:10:00+08:00",
                    "conversation": [{"role": "user", "content": "我今天补课到很晚。"}],
                }
            ],
            user_id="u-ms-1",
            request_id="r-ms-1",
            source="web-demo",
            context={
                "channel": "site",
                "locale": "zh-CN",
                "prior_profile": {"summary": "疑似初中画像"},
            },
        )

        self.assertEqual(payload.mode, "multi_session")
        self.assertEqual(payload.meta.user_id, "u-ms-1")
        self.assertEqual(payload.meta.request_id, "r-ms-1")
        self.assertEqual(payload.meta.source, "web-demo")
        self.assertEqual(payload.context.channel, "site")
        self.assertEqual(payload.context.locale, "zh-CN")
        self.assertEqual(payload.context.prior_profile["summary"], "疑似初中画像")
        self.assertEqual(payload.sessions[0].session_id, "s1")

    def test_analyze_formal_payload_uses_normalized_payload_boundary(self):
        fake_executor = SimpleNamespace(run_payload=mock.Mock(return_value=make_skill_output(True, 0.88)))

        with mock.patch("src.runtime.skill_runtime_adapter.get_formal_executor", return_value=fake_executor):
            result = analyze_formal_payload(
                {
                    "mode": "single_session",
                    "conversation": [{"role": "user", "content": "我不想写作业"}],
                    "meta": {"user_id": "u-2"},
                }
            )

        forwarded_payload = fake_executor.run_payload.call_args.args[0]
        self.assertIsInstance(forwarded_payload, AnalysisPayload)
        self.assertEqual(forwarded_payload.meta.user_id, "u-2")
        self.assertTrue(result.is_minor)

    def test_enrich_single_session_context_runs_builtin_time_and_rag(self):
        conversation = [
            {
                "role": "user",
                "timestamp": "2026-03-18 周三 08:59",
                "content": "今天上学前又和 AI 聊了半天。",
            }
        ]

        with mock.patch(
            "src.runtime.skill_runtime_adapter._run_skill_script",
            side_effect=[
                {"weekday": "Wednesday", "is_late_night": False},
                {"retrieved_cases": [{"sample_id": "minor_case_1"}]},
            ],
        ) as run_script, mock.patch(
            "src.runtime.skill_runtime_adapter._builtin_retrieval_resources_available",
            return_value=True,
        ):
            enriched = enrich_single_session_context(conversation)

        self.assertEqual(enriched["time_features"]["weekday"], "Wednesday")
        self.assertEqual(enriched["retrieved_cases"][0]["sample_id"], "minor_case_1")
        self.assertEqual(run_script.call_count, 2)
        self.assertIn("--raw-time-hint", run_script.call_args_list[1].args[1])
        self.assertIn("--time-features-json", run_script.call_args_list[1].args[1])

    def test_enrich_single_session_context_uses_raw_time_hint_when_conversation_has_no_timestamp(self):
        conversation = [
            {
                "role": "user",
                "content": "昨晚和 AI 聊到很晚。",
            }
        ]

        with mock.patch(
            "src.runtime.skill_runtime_adapter._run_skill_script",
            side_effect=[
                {"weekday": "Wednesday", "is_late_night": False},
                {"retrieved_cases": [{"sample_id": "minor_case_hint"}]},
            ],
        ) as run_script, mock.patch(
            "src.runtime.skill_runtime_adapter._builtin_retrieval_resources_available",
            return_value=True,
        ):
            enriched = enrich_single_session_context(
                conversation,
                context={"raw_time_hint": "2026-03-18 周三 08:59"},
            )

        self.assertEqual(enriched["time_features"]["weekday"], "Wednesday")
        self.assertEqual(run_script.call_args_list[0].args[1], ["--timestamp", "2026-03-18 周三 08:59"])
        self.assertIn("--raw-time-hint", run_script.call_args_list[1].args[1])
        self.assertIn("2026-03-18 周三 08:59", run_script.call_args_list[1].args[1])

    def test_analyze_single_session_formal_auto_forwards_enriched_context(self):
        fake_executor = SimpleNamespace(run_payload=mock.Mock(return_value=make_skill_output(True, 0.91)))
        conversation = [
            {
                "role": "user",
                "timestamp": "2026-03-18 周三 23:10",
                "content": "明天还要上学，但我还在熬夜和 AI 聊。",
            }
        ]

        with mock.patch(
            "src.runtime.skill_runtime_adapter._run_skill_script",
            side_effect=[
                {"weekday": "Wednesday", "is_late_night": True},
                {"retrieved_cases": [{"sample_id": "minor_case_2"}]},
            ],
        ), mock.patch(
            "src.runtime.skill_runtime_adapter._builtin_retrieval_resources_available",
            return_value=True,
        ), mock.patch("src.runtime.skill_runtime_adapter.get_formal_executor", return_value=fake_executor):
            result = analyze_single_session_formal_auto(conversation, user_id="u-3")

        forwarded_payload = fake_executor.run_payload.call_args.args[0]
        self.assertEqual(forwarded_payload.meta.user_id, "u-3")
        self.assertTrue(forwarded_payload.context.time_features["is_late_night"])
        self.assertEqual(forwarded_payload.context.retrieved_cases[0]["sample_id"], "minor_case_2")
        self.assertTrue(result.is_minor)

    def test_enrich_multi_session_context_uses_session_time_when_turn_has_no_timestamp(self):
        sessions = [
            {
                "session_id": "s1",
                "session_time": "2026-03-18 周三 23:10",
                "conversation": [{"role": "user", "content": "我今天又和 AI 聊到很晚。"}],
            },
            {
                "session_id": "s2",
                "session_time": "2026-03-22 周日 22:40",
                "conversation": [{"role": "user", "content": "明天还要上学。"}],
            },
        ]

        with mock.patch(
            "src.runtime.skill_runtime_adapter._run_skill_script",
            side_effect=[
                {"weekday": "Wednesday", "is_late_night": True},
                {"retrieved_cases": [{"sample_id": "minor_case_multi_1"}]},
            ],
        ) as run_script, mock.patch(
            "src.runtime.skill_runtime_adapter._builtin_retrieval_resources_available",
            return_value=True,
        ):
            enriched = enrich_multi_session_context(sessions)

        self.assertTrue(enriched["time_features"]["is_late_night"])
        self.assertEqual(enriched["retrieved_cases"][0]["sample_id"], "minor_case_multi_1")
        self.assertEqual(run_script.call_args_list[0].args[1], ["--timestamp", "2026-03-18 周三 23:10"])

    def test_analyze_multi_session_formal_auto_forwards_enriched_context(self):
        fake_executor = SimpleNamespace(run_payload=mock.Mock(return_value=make_skill_output(True, 0.89)))
        sessions = [
            {
                "session_id": "s1",
                "session_time": "2026-03-18 周三 23:10",
                "conversation": [{"role": "user", "content": "我初二，最近总是晚上补作业。"}],
            },
            {
                "session_id": "s2",
                "session_time": "2026-03-22 周日 22:40",
                "conversation": [{"role": "user", "content": "明天还要月考，我有点紧张。"}],
            },
        ]

        with mock.patch(
            "src.runtime.skill_runtime_adapter._run_skill_script",
            side_effect=[
                {"weekday": "Wednesday", "is_late_night": True},
                {"retrieved_cases": [{"sample_id": "minor_case_multi_2"}]},
            ],
        ), mock.patch(
            "src.runtime.skill_runtime_adapter._builtin_retrieval_resources_available",
            return_value=True,
        ), mock.patch("src.runtime.skill_runtime_adapter.get_formal_executor", return_value=fake_executor):
            result = analyze_multi_session_formal_auto(sessions, user_id="u-ms-2")

        forwarded_payload = fake_executor.run_payload.call_args.args[0]
        self.assertEqual(forwarded_payload.meta.user_id, "u-ms-2")
        self.assertEqual(forwarded_payload.mode, "multi_session")
        self.assertTrue(forwarded_payload.context.time_features["is_late_night"])
        self.assertEqual(forwarded_payload.context.retrieved_cases[0]["sample_id"], "minor_case_multi_2")
        self.assertEqual(forwarded_payload.sessions[0].session_id, "s1")
        self.assertTrue(result.is_minor)

    def test_analyze_single_session_formal_auto_logs_mode_and_internal_rag(self):
        fake_executor = SimpleNamespace(run_payload=mock.Mock(return_value=make_skill_output(True, 0.83)))
        conversation = [
            {
                "role": "user",
                "timestamp": "2026-03-18 周三 22:05",
                "content": "晚上还在和 AI 聊学校里的事。",
            }
        ]

        stderr_buffer = io.StringIO()
        with mock.patch(
            "src.runtime.skill_runtime_adapter._run_skill_script",
            side_effect=[
                {"weekday": "Wednesday", "is_late_night": True},
                {"status": "ok", "mode": "fallback:RuntimeError", "retrieved_cases": [{"sample_id": "minor_case_3"}]},
            ],
        ), mock.patch(
            "src.runtime.skill_runtime_adapter._builtin_retrieval_resources_available",
            return_value=True,
        ), mock.patch(
            "src.runtime.skill_runtime_adapter.get_formal_executor",
            return_value=fake_executor,
        ), mock.patch("sys.stderr", stderr_buffer):
            analyze_single_session_formal_auto(conversation, user_id="u-4")

        log_output = stderr_buffer.getvalue()
        self.assertIn("mode=single_session", log_output)
        self.assertIn("rag_mode=internal_rag", log_output)

    def test_analyze_single_session_formal_auto_degrades_to_no_rag_when_internal_rag_fails(self):
        fake_executor = SimpleNamespace(run_payload=mock.Mock(return_value=make_skill_output(True, 0.67)))
        conversation = [
            {
                "role": "user",
                "timestamp": "2026-03-18 周三 21:48",
                "content": "今天晚上又在和 AI 聊。",
            }
        ]

        stderr_buffer = io.StringIO()
        with mock.patch(
            "src.runtime.skill_runtime_adapter._run_skill_script",
            side_effect=[
                {"weekday": "Wednesday", "is_late_night": False},
                RuntimeError("retrieval unavailable"),
            ],
        ), mock.patch(
            "src.runtime.skill_runtime_adapter._builtin_retrieval_resources_available",
            return_value=True,
        ), mock.patch(
            "src.runtime.skill_runtime_adapter._run_inprocess_builtin_retrieval_fallback",
            return_value=[],
        ), mock.patch(
            "src.runtime.skill_runtime_adapter.get_formal_executor",
            return_value=fake_executor,
        ), mock.patch("sys.stderr", stderr_buffer):
            result = analyze_single_session_formal_auto(conversation, user_id="u-5")

        forwarded_payload = fake_executor.run_payload.call_args.args[0]
        self.assertEqual(forwarded_payload.context.retrieved_cases, [])
        self.assertTrue(result.is_minor)
        log_output = stderr_buffer.getvalue()
        self.assertIn("mode=single_session", log_output)
        self.assertIn("rag_mode=no_rag", log_output)

    def test_extract_time_features_script_marks_school_holiday_and_late_night(self):
        payload = run_time_script("2026-07-18 周六 23:10")

        self.assertEqual(payload["weekday"], "Saturday")
        self.assertTrue(payload["is_weekend"])
        self.assertTrue(payload["is_late_night"])
        self.assertEqual(payload["time_bucket"], "late_night")
        self.assertEqual(payload["holiday_label"], "summer_vacation")
        self.assertTrue(payload["school_holiday_hint"])

    def test_extract_time_features_script_marks_term_time_morning(self):
        payload = run_time_script("2026-03-18 周三 08:59")

        self.assertEqual(payload["weekday"], "Wednesday")
        self.assertFalse(payload["is_weekend"])
        self.assertFalse(payload["is_late_night"])
        self.assertEqual(payload["time_bucket"], "early_morning")
        self.assertEqual(payload["holiday_label"], "none")
        self.assertFalse(payload["school_holiday_hint"])

    def test_extract_time_features_script_accepts_iso8601_timestamp(self):
        payload = run_time_script("2026-03-19T22:18:00+08:00")

        self.assertEqual(payload["weekday"], "Thursday")
        self.assertFalse(payload["is_weekend"])
        self.assertTrue(payload["is_late_night"])
        self.assertEqual(payload["time_bucket"], "late_night")
        self.assertEqual(payload["local_time"], "2026-03-19 22:18:00")


    def test_extract_time_features_script_emits_ascii_safe_json(self):
        completed = subprocess.run(
            [sys.executable, str(TIME_SCRIPT_PATH), "--timestamp", "2026-03-18 周三 08:59"],
            capture_output=True,
            check=True,
        )

        payload = json.loads(completed.stdout.decode("ascii").strip())
        self.assertEqual(payload["weekday"], "Wednesday")

    def test_retrieve_cases_script_emits_ascii_safe_json(self):
        completed = subprocess.run(
            [sys.executable, str(RETRIEVE_CASES_PATH), "--query", "我初三，最近总是深夜和AI聊作业", "--top-k", "1"],
            capture_output=True,
            check=True,
        )

        payload = json.loads(completed.stdout.decode("ascii").strip())
        self.assertEqual(payload["status"], "ok")
        self.assertIn("retrieved_cases", payload)


class BuiltinRagFallbackTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.retrieve_cases = load_retrieve_cases_module()

    def test_fallback_retrieve_prefers_minor_cases_for_minor_query(self):
        results = self.retrieve_cases._fallback_retrieve(
            RETRIEVAL_CORPUS_PATH,
            "我初三，最近总是深夜和 AI 聊作业和学校里的事",
            3,
        )

        self.assertGreater(len(results), 0)
        self.assertEqual(results[0]["label"], "minor")

    def test_fallback_retrieve_prefers_adult_cases_for_college_query(self):
        results = self.retrieve_cases._fallback_retrieve(
            RETRIEVAL_CORPUS_PATH,
            "我在大学准备实习，最近总在和 AI 聊论文和找工作",
            3,
        )

        self.assertGreater(len(results), 0)
        self.assertEqual(results[0]["label"], "adult")


if __name__ == "__main__":
    unittest.main()

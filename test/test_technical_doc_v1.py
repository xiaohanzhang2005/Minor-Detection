import json
import shutil
import unittest
import warnings
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from scripts.prepare_data import BenchmarkSample, DataPreparer
from scripts.run_pipeline import (
    _build_review_summary,
    _save_run_report,
    _cleanup_pipeline_artifacts,
    _validate_pipeline_config,
    evaluate_rag_mode,
    ensure_benchmark_data,
    ensure_rag_index,
    parse_pipeline_args,
    repair_benchmark_manifest,
    step_evaluate,
    step_evolution,
)
from scripts.run_acceptance_suite import _check_confusion_consistency, _check_rag_gain
from src.config import get_active_skill_path, get_active_skill_version, set_active_skill_version
from src.evolution.evaluator import SkillEvaluator, run_evaluation
from src.evolution.evaluator import EvaluationMetrics, EvaluationReport, EvaluationResult
from src.evolution.optimizer import SkillOptimizer, run_optimization_cycle
from src.executor.executor import ExecutorSkill, analyze_with_memory, analyze_with_rag
from src.memory.user_memory import UserMemory
from src.models import ICBOFeatures, RiskLevel, SkillOutput, UserPersona
from src.retriever.semantic_retriever import SemanticRetriever

WORK_TMP_DIR = Path("tmp") / "unit_test_artifacts"
WORK_TMP_DIR.mkdir(parents=True, exist_ok=True)
warnings.filterwarnings("ignore", category=ResourceWarning)


def fresh_case_dir(name: str) -> Path:
    path = WORK_TMP_DIR / name
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def make_skill_output(is_minor: bool, confidence: float, reasoning: str = "ok") -> SkillOutput:
    return SkillOutput(
        is_minor=is_minor,
        minor_confidence=confidence,
        risk_level=RiskLevel.LOW if is_minor else RiskLevel.MEDIUM,
        icbo_features=ICBOFeatures(
            intention="intent",
            cognition="cognition",
            behavior_style="behavior",
            opportunity_time="night",
        ),
        user_persona=UserPersona(
            age=15 if is_minor else 24,
            age_range="14-16岁" if is_minor else "23岁以上",
            gender=None,
            education_stage="高中" if is_minor else "成人专业",
            identity_markers=["学生"] if is_minor else ["上班族"],
        ),
        reasoning=reasoning,
        key_evidence=["e1"],
    )


class FakeLLMClient:
    def __init__(self, result: SkillOutput):
        self.result = result
        self.messages = []

    def chat(self, messages, response_model=None, temperature=0.7):
        self.messages.append(
            {
                "messages": messages,
                "response_model": response_model,
                "temperature": temperature,
            }
        )
        return self.result

    def get_structured_metrics(self):
        return {"structured_calls": len(self.messages)}


class FakeRetriever:
    def __init__(self):
        self.retrieve_calls = []

    def retrieve(self, conversation, top_k=3, threshold=0.0, raw_time_hint="", time_features=None):
        self.retrieve_calls.append(
            {
                "conversation": conversation,
                "top_k": top_k,
                "threshold": threshold,
                "raw_time_hint": raw_time_hint,
                "time_features": time_features,
            }
        )
        return ["dummy-result"]

    def format_for_prompt(self, results):
        return f"RAG_CONTEXT:{len(results)}"


class SequenceExecutor:
    def __init__(self, outputs):
        self.outputs = list(outputs)
        self.calls = []

    def run(self, conversation, user_id=None, context=None):
        self.calls.append(
            {
                "conversation": conversation,
                "user_id": user_id,
                "context": context,
            }
        )
        return self.outputs.pop(0)


class DataFoundationTests(unittest.TestCase):
    def test_convert_generates_unique_sample_ids_even_when_raw_ids_repeat(self):
        preparer = DataPreparer()

        sample_a = preparer._convert(
            {
                "dataset_id": "dup-1",
                "is_minor": True,
                "conversation": [{"role": "user", "content": "A"}],
                "icbo_features": {},
                "user_persona": {},
            },
            source_tag="social_pos",
            index=0,
        )
        sample_b = preparer._convert(
            {
                "dataset_id": "dup-1",
                "is_minor": True,
                "conversation": [{"role": "user", "content": "B"}],
                "icbo_features": {},
                "user_persona": {},
            },
            source_tag="social_pos",
            index=1,
        )

        self.assertNotEqual(sample_a.sample_id, sample_b.sample_id)
        self.assertEqual(sample_a.extra_info["raw_sample_id"], "dup-1")

    def test_split_data_keeps_splits_disjoint(self):
        preparer = DataPreparer(train_ratio=0.5, val_ratio=0.25, test_ratio=0.25, random_seed=7)
        preparer.samples = [
            BenchmarkSample(f"minor_social_{i}", "social_pos", True, [], {}, {})
            for i in range(4)
        ] + [
            BenchmarkSample(f"adult_social_{i}", "social_neg", False, [], {}, {})
            for i in range(4)
        ]

        with mock.patch("builtins.print"):
            splits = preparer.split_data()

        train_ids = {sample.sample_id for sample in splits["train"]}
        val_ids = {sample.sample_id for sample in splits["val"]}
        test_ids = {sample.sample_id for sample in splits["test"]}

        self.assertFalse(train_ids & val_ids)
        self.assertFalse(train_ids & test_ids)
        self.assertFalse(val_ids & test_ids)

    def test_save_splits_returns_manifest_relative_paths(self):
        case_dir = fresh_case_dir("prepare_data_relative_paths")
        preparer = DataPreparer(output_dir=str(case_dir))
        splits = {
            "train": [BenchmarkSample("t1", "social_pos", True, [], {}, {})],
            "val": [BenchmarkSample("v1", "social_neg", False, [], {}, {})],
            "test": [BenchmarkSample("x1", "knowledge_pos", True, [], {}, {})],
        }

        with mock.patch("builtins.print"):
            paths = preparer.save_splits(splits)

        self.assertEqual(
            paths,
            {
                "train": "train.jsonl",
                "val": "val.jsonl",
                "test": "test.jsonl",
            },
        )

    def test_retriever_manifest_uses_relative_paths(self):
        case_dir = fresh_case_dir("retriever_manifest_relative_paths")
        retriever = SemanticRetriever(index_path=str(case_dir / "retrieval_db" / "index.pkl"))
        retriever.corpus_path = case_dir / "retrieval_corpus" / "cases_v1.jsonl"
        retriever.manifest_path = case_dir / "retrieval_db" / "manifest.json"

        manifest = retriever._build_manifest(case_dir / "benchmark" / "train.jsonl", records=[])

        self.assertEqual(manifest["source_data_path"], "../benchmark/train.jsonl")
        self.assertEqual(manifest["index_path"], "index.pkl")
        self.assertEqual(manifest["corpus_path"], "../retrieval_corpus/cases_v1.jsonl")


class ExecutorAndMemoryTests(unittest.TestCase):
    def _make_skill_file(self, tmpdir: Path) -> Path:
        skill_path = tmpdir / "skill.md"
        skill_path.write_text("# fake skill", encoding="utf-8")
        return skill_path

    def test_analyze_with_rag_injects_retrieved_context(self):
        case_dir = fresh_case_dir("rag_context")
        skill_path = self._make_skill_file(case_dir)
        llm = FakeLLMClient(make_skill_output(True, 0.9))
        executor = ExecutorSkill(skill_path=str(skill_path), llm_client=llm)
        retriever = FakeRetriever()

        with mock.patch("src.executor.executor._executor_instance", executor), mock.patch("builtins.print"):
            analyze_with_rag(
                conversation=[{"role": "user", "content": "明天月考"}],
                retriever=retriever,
                top_k=2,
            )

        user_prompt = llm.messages[0]["messages"][1]["content"]
        self.assertIn("RAG_CONTEXT:1", user_prompt)
        self.assertEqual(retriever.retrieve_calls[0]["top_k"], 2)

    def test_analyze_with_memory_updates_cross_session_profile(self):
        case_dir = fresh_case_dir("memory_update")
        skill_path = self._make_skill_file(case_dir)
        llm = FakeLLMClient(make_skill_output(True, 0.8))
        executor = ExecutorSkill(skill_path=str(skill_path), llm_client=llm)
        memory = UserMemory(db_path=str(case_dir / "memory.db"))

        with mock.patch("src.executor.executor._executor_instance", executor), mock.patch("builtins.print"):
            analyze_with_memory(
                conversation=[{"role": "user", "content": "我不想上学"}],
                user_id="user-1",
                memory=memory,
                retriever=None,
                update_memory=True,
            )
            analyze_with_memory(
                conversation=[{"role": "user", "content": "明天还要考试"}],
                user_id="user-1",
                memory=memory,
                retriever=None,
                update_memory=True,
            )

        profile = memory.get_profile("user-1")
        self.assertIsNotNone(profile)
        self.assertEqual(profile.total_sessions, 2)
        self.assertTrue(profile.is_minor)


class EvaluationTests(unittest.TestCase):
    def test_skill_evaluator_computes_confusion_metrics(self):
        case_dir = fresh_case_dir("evaluator_metrics")
        dataset_path = case_dir / "val.jsonl"
        samples = [
            {"sample_id": "s1", "is_minor": True, "conversation": [{"role": "user", "content": "1"}]},
            {"sample_id": "s2", "is_minor": True, "conversation": [{"role": "user", "content": "2"}]},
            {"sample_id": "s3", "is_minor": False, "conversation": [{"role": "user", "content": "3"}]},
            {"sample_id": "s4", "is_minor": False, "conversation": [{"role": "user", "content": "4"}]},
        ]
        dataset_path.write_text(
            "\n".join(json.dumps(sample, ensure_ascii=False) for sample in samples),
            encoding="utf-8",
        )

        executor = SequenceExecutor(
            [
                make_skill_output(True, 0.9, "tp"),
                make_skill_output(False, 0.2, "fn"),
                make_skill_output(True, 0.8, "fp"),
                make_skill_output(False, 0.1, "tn"),
            ]
        )
        evaluator = SkillEvaluator(executor=executor, skill_version="unit-test")
        report = evaluator.evaluate(dataset_path=str(dataset_path), verbose=False)

        self.assertEqual(report.metrics.true_positive, 1)
        self.assertEqual(report.metrics.false_negative, 1)
        self.assertEqual(report.metrics.false_positive, 1)
        self.assertEqual(report.metrics.true_negative, 1)
        self.assertAlmostEqual(report.metrics.f1_score, 0.5)
        self.assertAlmostEqual(report.metrics.accuracy, 0.5)

    def test_skill_evaluator_routes_formal_skill_through_runtime_adapter(self):
        case_dir = fresh_case_dir("evaluator_formal_runtime")
        dataset_path = case_dir / "val.jsonl"
        samples = [
            {
                "sample_id": "s1",
                "is_minor": True,
                "conversation": [{"role": "user", "content": "1"}],
                "icbo_features": {"opportunity_time": "2026-03-18 周三 08:59"},
            },
        ]
        dataset_path.write_text(
            "\n".join(json.dumps(sample, ensure_ascii=False) for sample in samples),
            encoding="utf-8",
        )

        class FailingExecutor:
            def run(self, conversation, user_id=None, context=None):
                raise AssertionError("legacy executor path should not run for formal skill")

        evaluator = SkillEvaluator(executor=FailingExecutor(), skill_version="minor-detection")

        with mock.patch(
            "src.evolution.evaluator.analyze_single_session_formal_auto",
            return_value=make_skill_output(True, 0.93, "formal"),
        ) as mocked_formal:
            report = evaluator.evaluate(dataset_path=str(dataset_path), verbose=False)

        mocked_formal.assert_called_once()
        self.assertEqual(
            mocked_formal.call_args.kwargs["context"]["raw_time_hint"],
            "2026-03-18 周三 08:59",
        )
        self.assertEqual(report.metrics.true_positive, 1)
        self.assertAlmostEqual(report.metrics.f1_score, 1.0)

    def test_skill_evaluator_keeps_legacy_executor_path_for_old_skills(self):
        case_dir = fresh_case_dir("evaluator_legacy_runtime")
        dataset_path = case_dir / "val.jsonl"
        samples = [
            {"sample_id": "s1", "is_minor": True, "conversation": [{"role": "user", "content": "1"}]},
        ]
        dataset_path.write_text(
            "\n".join(json.dumps(sample, ensure_ascii=False) for sample in samples),
            encoding="utf-8",
        )

        executor = SequenceExecutor([make_skill_output(True, 0.81, "legacy")])
        evaluator = SkillEvaluator(executor=executor, skill_version="teen_detector_v1")

        with mock.patch("src.evolution.evaluator.analyze_single_session_formal_auto") as mocked_formal:
            report = evaluator.evaluate(dataset_path=str(dataset_path), verbose=False)

        mocked_formal.assert_not_called()
        self.assertEqual(len(executor.calls), 1)
        self.assertEqual(report.metrics.true_positive, 1)

    def test_skill_evaluator_legacy_rag_passes_raw_time_hint(self):
        case_dir = fresh_case_dir("evaluator_legacy_rag_time_hint")
        dataset_path = case_dir / "val.jsonl"
        samples = [
            {
                "sample_id": "s1",
                "is_minor": True,
                "conversation": [{"role": "user", "content": "1"}],
                "icbo_features": {"opportunity_time": "2026-03-18 周三 08:59"},
            },
        ]
        dataset_path.write_text(
            "\n".join(json.dumps(sample, ensure_ascii=False) for sample in samples),
            encoding="utf-8",
        )

        executor = SequenceExecutor([make_skill_output(True, 0.81, "legacy-rag")])
        retriever = FakeRetriever()
        evaluator = SkillEvaluator(executor=executor, skill_version="teen_detector_v1", retriever=retriever)

        report = evaluator.evaluate(dataset_path=str(dataset_path), verbose=False)

        self.assertEqual(retriever.retrieve_calls[0]["raw_time_hint"], "2026-03-18 周三 08:59")
        self.assertEqual(report.metrics.true_positive, 1)

    def test_skill_evaluator_memory_isolates_profiles_by_user_id(self):
        case_dir = fresh_case_dir("evaluator_memory_isolation")
        dataset_path = case_dir / "val.jsonl"
        memory_db_path = case_dir / "memory.db"
        samples = [
            {
                "sample_id": "s1",
                "user_id": "user-a",
                "source": "social_pos",
                "is_minor": True,
                "conversation": [{"role": "user", "content": "1"}],
            },
            {
                "sample_id": "s2",
                "user_id": "user-a",
                "source": "social_pos",
                "is_minor": True,
                "conversation": [{"role": "user", "content": "2"}],
            },
            {
                "sample_id": "s3",
                "user_id": "user-b",
                "source": "social_neg",
                "is_minor": False,
                "conversation": [{"role": "user", "content": "3"}],
            },
        ]
        dataset_path.write_text(
            "\n".join(json.dumps(sample, ensure_ascii=False) for sample in samples),
            encoding="utf-8",
        )

        executor = SequenceExecutor(
            [
                make_skill_output(True, 0.9, "u1-1"),
                make_skill_output(True, 0.8, "u1-2"),
                make_skill_output(False, 0.2, "u2-1"),
            ]
        )
        evaluator = SkillEvaluator(
            executor=executor,
            skill_version="unit-test",
            use_memory=True,
            memory_db_path=str(memory_db_path),
        )
        evaluator.evaluate(dataset_path=str(dataset_path), verbose=False)

        memory = UserMemory(db_path=str(memory_db_path))
        profile_a = memory.get_profile("user-a")
        profile_b = memory.get_profile("user-b")
        self.assertIsNotNone(profile_a)
        self.assertIsNotNone(profile_b)
        self.assertEqual(profile_a.total_sessions, 2)
        self.assertEqual(profile_b.total_sessions, 1)

    def test_skill_evaluator_memory_is_disabled_without_user_id(self):
        case_dir = fresh_case_dir("evaluator_memory_requires_user_id")
        dataset_path = case_dir / "val.jsonl"
        memory_db_path = case_dir / "memory.db"
        samples = [
            {
                "sample_id": "s1",
                "source": "social_pos",
                "is_minor": True,
                "conversation": [{"role": "user", "content": "1"}],
            },
            {
                "sample_id": "s2",
                "source": "social_pos",
                "is_minor": True,
                "conversation": [{"role": "user", "content": "2"}],
            },
        ]
        dataset_path.write_text(
            "\n".join(json.dumps(sample, ensure_ascii=False) for sample in samples),
            encoding="utf-8",
        )

        executor = SequenceExecutor(
            [
                make_skill_output(True, 0.9, "a"),
                make_skill_output(True, 0.8, "b"),
            ]
        )
        evaluator = SkillEvaluator(
            executor=executor,
            skill_version="unit-test",
            use_memory=True,
            memory_db_path=str(memory_db_path),
        )
        evaluator.evaluate(dataset_path=str(dataset_path), verbose=False)

        memory = UserMemory(db_path=str(memory_db_path))
        self.assertEqual(memory.get_all_users(), [])

    def test_run_evaluation_defaults_to_validation_set(self):
        self.assertEqual(run_evaluation.__defaults__[1], "val")

    def test_acceptance_gate_level_uses_severity_not_truth_value(self):
        report = SimpleNamespace(
            metrics=SimpleNamespace(
                true_positive=1,
                true_negative=1,
                false_positive=0,
                false_negative=0,
                total_samples=2,
            )
        )
        gate = _check_confusion_consistency(report)
        self.assertTrue(gate.passed)
        self.assertEqual(gate.level, "info")

    def test_rag_regression_is_reported_as_warning(self):
        rag_report = SimpleNamespace(metrics=SimpleNamespace(f1_score=0.90))
        no_rag_report = SimpleNamespace(metrics=SimpleNamespace(f1_score=0.95))
        gate = _check_rag_gain(rag_report, no_rag_report)
        self.assertFalse(gate.passed)
        self.assertEqual(gate.level, "warn")


class OptimizationFlowTests(unittest.TestCase):
    def _create_skill_tree(self, root: Path):
        skills_dir = root / "skills"
        (skills_dir / "teen_detector_v1").mkdir(parents=True, exist_ok=True)
        (skills_dir / "teen_detector_v2").mkdir(parents=True, exist_ok=True)
        (skills_dir / "teen_detector_v1" / "skill.md").write_text("# v1", encoding="utf-8")
        (skills_dir / "teen_detector_v2" / "skill.md").write_text("# v2", encoding="utf-8")
        return skills_dir

    def _create_formal_skill_tree(self, root: Path):
        skills_dir = root / "skills"
        formal_dir = skills_dir / "minor-detection"
        (formal_dir / "references").mkdir(parents=True, exist_ok=True)
        (formal_dir / "SKILL.md").write_text("# formal v1\n- 初中线索要高权重\n", encoding="utf-8")
        (formal_dir / "references" / "evidence-rules.md").write_text("# rules\n", encoding="utf-8")
        (formal_dir / "references" / "icbo-guidelines.md").write_text("# icbo\n", encoding="utf-8")
        (formal_dir / "references" / "output-schema.md").write_text("# schema\n", encoding="utf-8")
        return skills_dir

    def test_optimizer_rollback_updates_active_version_pointer(self):
        root = fresh_case_dir("rollback_pointer")
        skills_dir = self._create_skill_tree(root)
        pointer = skills_dir / "active_version.txt"
        pointer.write_text("teen_detector_v1", encoding="utf-8")

        with mock.patch("src.config.SKILLS_DIR", skills_dir), mock.patch(
            "src.config.ACTIVE_SKILL_POINTER", pointer
        ), mock.patch("src.evolution.optimizer.LLMClient", return_value=SimpleNamespace()):
            optimizer = SkillOptimizer(skills_dir=str(skills_dir))
            with mock.patch("builtins.print"):
                self.assertTrue(optimizer.rollback("teen_detector_v2"))
            self.assertEqual(get_active_skill_version(), "teen_detector_v2")
            self.assertEqual(get_active_skill_path(), skills_dir / "teen_detector_v2" / "skill.md")

    def test_optimizer_creates_new_version_when_report_contains_errors(self):
        root = fresh_case_dir("optimizer_creates_version")
        skills_dir = self._create_skill_tree(root)
        pointer = skills_dir / "active_version.txt"
        pointer.write_text("teen_detector_v1", encoding="utf-8")
        (skills_dir / "teen_detector_v1" / "config.yaml").write_text("version: v1\n", encoding="utf-8")

        metrics = EvaluationMetrics(
            total_samples=4,
            correct=2,
            true_positive=1,
            true_negative=1,
            false_positive=1,
            false_negative=1,
            accuracy=0.5,
            precision=0.5,
            recall=0.5,
            f1_score=0.5,
        )
        report = EvaluationReport(
            skill_version="teen_detector_v1",
            eval_time="2026-03-14T20:00:00",
            dataset="val",
            metrics=metrics,
            results=[],
            errors=[
                {
                    "sample_id": "e1",
                    "ground_truth": "adult",
                    "predicted": "minor",
                    "confidence": 0.88,
                    "reasoning": "误判为未成年",
                }
            ],
        )

        fake_llm = SimpleNamespace(chat=lambda messages, temperature=0.7: "# optimized skill\n")

        with mock.patch("src.config.SKILLS_DIR", skills_dir), mock.patch(
            "src.config.ACTIVE_SKILL_POINTER", pointer
        ):
            optimizer = SkillOptimizer(llm_client=fake_llm, skills_dir=str(skills_dir))
            with mock.patch("builtins.print"):
                result = optimizer.optimize(
                    report=report,
                    current_version="teen_detector_v1",
                    new_version="teen_detector_v2_forced",
                    dry_run=False,
                )

        new_dir = skills_dir / "teen_detector_v2_forced"
        self.assertTrue(result["success"])
        self.assertTrue(new_dir.exists())
        self.assertTrue((new_dir / "skill.md").exists())
        self.assertTrue((new_dir / "optimization_history.json").exists())

    def test_optimizer_creates_new_formal_skill_bundle_when_target_is_skill_folder(self):
        root = fresh_case_dir("optimizer_creates_formal_bundle")
        skills_dir = self._create_formal_skill_tree(root)
        pointer = skills_dir / "active_version.txt"
        pointer.write_text("minor-detection", encoding="utf-8")

        metrics = EvaluationMetrics(
            total_samples=2,
            correct=1,
            true_positive=1,
            true_negative=0,
            false_positive=1,
            false_negative=0,
            accuracy=0.5,
            precision=0.5,
            recall=1.0,
            f1_score=2 / 3,
        )
        report = EvaluationReport(
            skill_version="minor-detection",
            eval_time="2026-03-18T10:00:00",
            dataset="val",
            metrics=metrics,
            results=[],
            errors=[
                {
                    "sample_id": "adult_fp",
                    "ground_truth": "adult",
                    "predicted": "minor",
                    "confidence": 0.79,
                    "reasoning": "adult misread as minor",
                }
            ],
        )

        responses = iter(
            [
                "# formal optimized\n- 深夜时间只能作为辅助证据\n",
                "# rules optimized\n- 深夜证据不能单独定性\n",
            ]
        )
        fake_llm = SimpleNamespace(chat=lambda messages, temperature=0.7: next(responses))

        with mock.patch("src.config.SKILLS_DIR", skills_dir), mock.patch(
            "src.config.ACTIVE_SKILL_POINTER", pointer
        ):
            optimizer = SkillOptimizer(llm_client=fake_llm, skills_dir=str(skills_dir))
            with mock.patch("builtins.print"):
                result = optimizer.optimize(
                    report=report,
                    current_version="minor-detection",
                    new_version="minor-detection-optimized",
                    dry_run=False,
                )

        new_dir = skills_dir / "minor-detection-optimized"
        self.assertTrue(result["success"])
        self.assertEqual(result["target_file"], "SKILL.md")
        self.assertTrue((new_dir / "SKILL.md").exists())
        self.assertTrue((new_dir / "references" / "evidence-rules.md").exists())
        self.assertEqual(sorted(result["edited_files"]), ["SKILL.md", "references/evidence-rules.md"])
        self.assertIn("references/output-schema.md", result["readonly_reference_files"])
        self.assertTrue((new_dir / "optimization_history.json").exists())
        self.assertIn("深夜证据不能单独定性", (new_dir / "references" / "evidence-rules.md").read_text(encoding="utf-8"))

    def test_run_optimization_cycle_can_defer_formal_activation_until_review(self):
        root = fresh_case_dir("formal_activation_deferred")
        skills_dir = self._create_formal_skill_tree(root)
        candidate_dir = skills_dir / "minor-detection-optimized"
        (candidate_dir / "references").mkdir(parents=True, exist_ok=True)
        (candidate_dir / "SKILL.md").write_text("# formal v2\n", encoding="utf-8")
        (candidate_dir / "references" / "evidence-rules.md").write_text("# rules v2\n", encoding="utf-8")
        (candidate_dir / "references" / "icbo-guidelines.md").write_text("# icbo\n", encoding="utf-8")
        (candidate_dir / "references" / "output-schema.md").write_text("# schema\n", encoding="utf-8")
        pointer = skills_dir / "active_version.txt"
        pointer.write_text("minor-detection", encoding="utf-8")

        def fake_optimize(
            self,
            report,
            current_version="minor-detection",
            new_version=None,
            max_errors=None,
            dry_run=False,
        ):
            return {
                "success": True,
                "current_version": current_version,
                "new_version": "minor-detection-optimized",
                "edited_files": ["SKILL.md", "references/evidence-rules.md"],
            }

        def fake_evaluate(
            self,
            dataset_path=None,
            max_samples=None,
            verbose=True,
            use_test_set=False,
            sample_strategy=None,
            sample_seed=None,
        ):
            f1 = 0.6 if self.skill_version == "minor-detection" else 0.9
            return SimpleNamespace(metrics=SimpleNamespace(f1_score=f1))

        class FakeExecutor:
            def __init__(self, skill_path=None, llm_client=None, inference_temperature=0.2):
                self.skill_path = skill_path

        with mock.patch("src.config.SKILLS_DIR", skills_dir), mock.patch(
            "src.evolution.optimizer.SKILLS_DIR", skills_dir
        ), mock.patch("src.config.ACTIVE_SKILL_POINTER", pointer), mock.patch(
            "src.evolution.optimizer.LLMClient", return_value=SimpleNamespace()
        ), mock.patch("src.executor.ExecutorSkill", FakeExecutor), mock.patch(
            "src.evolution.evaluator.SkillEvaluator.evaluate", fake_evaluate
        ), mock.patch(
            "src.evolution.optimizer.SkillOptimizer.optimize", fake_optimize
        ):
            with mock.patch("builtins.print"):
                result = run_optimization_cycle(
                    current_version="minor-detection",
                    max_samples=4,
                    dry_run=False,
                    auto_rollback=True,
                    min_f1_improvement=0.0,
                    retriever=None,
                    activate_accepted_version=False,
                )

        self.assertFalse(result["rolled_back"])
        self.assertTrue(result["review_required"])
        self.assertEqual(result["adoption_status"], "pending_review")
        self.assertEqual(result["active_version"], "minor-detection")
        self.assertEqual(pointer.read_text(encoding="utf-8").strip(), "minor-detection")

    def test_optimizer_prompt_marks_reference_files_read_only_for_formal_skill(self):
        fake_llm = SimpleNamespace()
        optimizer = SkillOptimizer(llm_client=fake_llm, skills_dir=str(fresh_case_dir("optimizer_formal_prompt")))
        prompt = optimizer.generate_optimization_prompt(
            current_skill="# skill",
            optimization_packet={
                "total_errors": 1,
                "analyzed_errors": 1,
                "max_errors": 1,
                "fp_count": 1,
                "fn_count": 0,
                "fp_examples": [{"confidence": 0.91, "reasoning": "adult misread as minor"}],
                "fn_examples": [],
                "error_selection": {"strategy": "all_errors", "budget": 1, "selected_count": 1, "total_count": 1},
                "protected_correct_examples": [],
                "metrics": {
                    "accuracy": 0.5,
                    "precision": 0.5,
                    "recall": 1.0,
                    "f1": 0.66,
                },
            },
            editable_target="SKILL.md",
            reference_materials={
                "references/evidence-rules.md": "# rules",
                "references/output-schema.md": "# schema",
            },
        )

        self.assertIn("Primary editable file: `SKILL.md`", prompt)
        self.assertIn("Formal Skill Editing Rules", prompt)
        self.assertIn("Keep the main language of the markdown as Chinese.", prompt)
        self.assertIn("Preserve the core order: mode detection -> time handling -> retrieval routing -> user modeling + ICBO -> evidence synthesis + final youth/minor judgment -> schema-constrained output.", prompt)
        self.assertIn("Do not remove any supported mode names: `single_session`, `multi_session`, `enriched`.", prompt)
        self.assertIn("These files are read-only for this optimization round.", prompt)
        self.assertIn("Treat `references/output-schema.md` as contract-frozen.", prompt)
        self.assertIn("Do not rewrite bundled reference files in this round.", prompt)

    def test_optimizer_prompt_adds_guardrails_for_fp_only_regression_risk(self):
        fake_llm = SimpleNamespace()
        optimizer = SkillOptimizer(llm_client=fake_llm, skills_dir=str(fresh_case_dir("optimizer_guardrails")))
        prompt = optimizer.generate_optimization_prompt(
            current_skill="# skill",
            optimization_packet={
                "total_errors": 2,
                "analyzed_errors": 2,
                "max_errors": 2,
                "fp_count": 2,
                "fn_count": 0,
                "fp_examples": [{"confidence": 0.91, "reasoning": "adult misread as minor"}],
                "fn_examples": [],
                "error_selection": {"strategy": "all_errors", "budget": 2, "selected_count": 2, "total_count": 2},
                "protected_correct_examples": [
                    {
                        "sample_id": "ok-1",
                        "label": "adult",
                        "confidence": 0.52,
                        "reasoning": "adult boundary case",
                    }
                ],
                "metrics": {
                    "accuracy": 0.99,
                    "precision": 0.98,
                    "recall": 1.0,
                    "f1": 0.99,
                },
            },
        )

        self.assertIn("Optimization Guardrails", prompt)
        self.assertIn("Baseline FN is zero. Preserve recall and avoid introducing new FN.", prompt)
        self.assertIn("Do not solve FP by making the classifier broadly less willing to predict minor.", prompt)
        self.assertIn("Preserve the supplied correct boundary-case examples", prompt)
        self.assertIn("False Positive Examples", prompt)
        self.assertIn("Protected Correct Examples", prompt)

    def test_optimizer_build_optimization_packet_adds_boundary_correct_examples(self):
        fake_llm = SimpleNamespace()
        optimizer = SkillOptimizer(llm_client=fake_llm, skills_dir=str(fresh_case_dir("optimizer_packet")))

        metrics = EvaluationMetrics(
            total_samples=5,
            correct=4,
            true_positive=2,
            true_negative=2,
            false_positive=1,
            false_negative=0,
            accuracy=0.8,
            precision=2 / 3,
            recall=1.0,
            f1_score=0.8,
        )
        report = EvaluationReport(
            skill_version="teen_detector_v1",
            eval_time="2026-03-16T10:00:00",
            dataset="val",
            metrics=metrics,
            results=[
                EvaluationResult(
                    sample_id="minor_boundary",
                    ground_truth=True,
                    predicted=True,
                    confidence=0.51,
                    is_correct=True,
                    reasoning="minor boundary case",
                    latency=1.0,
                ),
                EvaluationResult(
                    sample_id="adult_boundary",
                    ground_truth=False,
                    predicted=False,
                    confidence=0.49,
                    is_correct=True,
                    reasoning="adult boundary case",
                    latency=1.0,
                ),
                EvaluationResult(
                    sample_id="minor_easy",
                    ground_truth=True,
                    predicted=True,
                    confidence=0.95,
                    is_correct=True,
                    reasoning="easy minor case",
                    latency=1.0,
                ),
                EvaluationResult(
                    sample_id="adult_easy",
                    ground_truth=False,
                    predicted=False,
                    confidence=0.05,
                    is_correct=True,
                    reasoning="easy adult case",
                    latency=1.0,
                ),
                EvaluationResult(
                    sample_id="adult_fp",
                    ground_truth=False,
                    predicted=True,
                    confidence=0.83,
                    is_correct=False,
                    reasoning="adult misread as minor",
                    latency=1.0,
                ),
            ],
            errors=[
                {
                    "sample_id": "adult_fp",
                    "ground_truth": "adult",
                    "predicted": "minor",
                    "confidence": 0.83,
                    "reasoning": "adult misread as minor",
                }
            ],
        )

        packet = optimizer.build_optimization_packet(report)

        self.assertEqual(packet["total_errors"], 1)
        self.assertEqual(packet["fp_count"], 1)
        self.assertEqual(packet["fn_count"], 0)
        self.assertEqual(
            [example["sample_id"] for example in packet["protected_correct_examples"][:2]],
            ["adult_boundary", "minor_boundary"],
        )
        self.assertLess(
            packet["protected_correct_examples"][0]["boundary_distance"],
            packet["protected_correct_examples"][-1]["boundary_distance"],
        )

    def test_run_optimization_cycle_activates_accepted_version(self):
        root = fresh_case_dir("accepted_version")
        skills_dir = root / "skills"
        (skills_dir / "teen_detector_v1").mkdir(parents=True, exist_ok=True)
        (skills_dir / "teen_detector_v1" / "skill.md").write_text("# v1", encoding="utf-8")
        pointer = skills_dir / "active_version.txt"
        pointer.write_text("teen_detector_v1", encoding="utf-8")

        def fake_optimize(
            self,
            report,
            current_version="teen_detector_v1",
            new_version=None,
            max_errors=None,
            dry_run=False,
        ):
            new_version = "teen_detector_v2"
            new_dir = skills_dir / new_version
            new_dir.mkdir(parents=True, exist_ok=True)
            (new_dir / "skill.md").write_text("# v2", encoding="utf-8")
            return {
                "success": True,
                "current_version": current_version,
                "new_version": new_version,
            }

        def fake_evaluate(
            self,
            dataset_path=None,
            max_samples=None,
            verbose=True,
            use_test_set=False,
            sample_strategy=None,
            sample_seed=None,
        ):
            f1 = 0.6 if self.skill_version == "teen_detector_v1" else 0.9
            return SimpleNamespace(metrics=SimpleNamespace(f1_score=f1))

        class FakeExecutor:
            def __init__(self, skill_path=None, llm_client=None, inference_temperature=0.2):
                self.skill_path = skill_path

        with mock.patch("src.config.SKILLS_DIR", skills_dir), mock.patch(
            "src.config.ACTIVE_SKILL_POINTER", pointer
        ), mock.patch(
            "src.evolution.optimizer.LLMClient", return_value=SimpleNamespace()
        ), mock.patch("src.executor.ExecutorSkill", FakeExecutor), mock.patch(
            "src.evolution.evaluator.SkillEvaluator.evaluate", fake_evaluate
        ), mock.patch(
            "src.evolution.optimizer.SkillOptimizer.optimize", fake_optimize
        ):
            with mock.patch("builtins.print"):
                result = run_optimization_cycle(
                    current_version="teen_detector_v1",
                    max_samples=4,
                    dry_run=False,
                    auto_rollback=True,
                    min_f1_improvement=0.0,
                    retriever=None,
                )
            self.assertFalse(result["rolled_back"])
            self.assertEqual(result["active_version"], "teen_detector_v2")
            self.assertEqual(pointer.read_text(encoding="utf-8").strip(), "teen_detector_v2")
            self.assertEqual(get_active_skill_version(), "teen_detector_v2")

    def test_run_optimization_cycle_reuses_provided_baseline_report(self):
        root = fresh_case_dir("reuse_baseline_report")
        skills_dir = root / "skills"
        (skills_dir / "teen_detector_v1").mkdir(parents=True, exist_ok=True)
        (skills_dir / "teen_detector_v1" / "skill.md").write_text("# v1", encoding="utf-8")
        pointer = skills_dir / "active_version.txt"
        pointer.write_text("teen_detector_v1", encoding="utf-8")

        baseline_report = SimpleNamespace(metrics=SimpleNamespace(f1_score=0.6))
        evaluate_calls = {"count": 0}

        def fake_optimize(
            self,
            report,
            current_version="teen_detector_v1",
            new_version=None,
            max_errors=None,
            dry_run=False,
        ):
            new_version = "teen_detector_v2"
            new_dir = skills_dir / new_version
            new_dir.mkdir(parents=True, exist_ok=True)
            (new_dir / "skill.md").write_text("# v2", encoding="utf-8")
            return {
                "success": True,
                "current_version": current_version,
                "new_version": new_version,
            }

        def fake_evaluate(
            self,
            dataset_path=None,
            max_samples=None,
            verbose=True,
            use_test_set=False,
            sample_strategy=None,
            sample_seed=None,
        ):
            evaluate_calls["count"] += 1
            return SimpleNamespace(metrics=SimpleNamespace(f1_score=0.8))

        class FakeExecutor:
            def __init__(self, skill_path=None, llm_client=None, inference_temperature=0.2):
                self.skill_path = skill_path

        with mock.patch("src.config.SKILLS_DIR", skills_dir), mock.patch(
            "src.config.ACTIVE_SKILL_POINTER", pointer
        ), mock.patch(
            "src.evolution.optimizer.LLMClient", return_value=SimpleNamespace()
        ), mock.patch("src.executor.ExecutorSkill", FakeExecutor), mock.patch(
            "src.evolution.evaluator.SkillEvaluator.evaluate", fake_evaluate
        ), mock.patch(
            "src.evolution.optimizer.SkillOptimizer.optimize", fake_optimize
        ):
            with mock.patch("builtins.print"):
                result = run_optimization_cycle(
                    current_version="teen_detector_v1",
                    max_samples=4,
                    dry_run=False,
                    auto_rollback=True,
                    min_f1_improvement=0.0,
                    retriever=None,
                    baseline_report=baseline_report,
                )

        self.assertTrue(result["baseline_report_reused"])
        self.assertEqual(evaluate_calls["count"], 1)


class PipelineScriptTests(unittest.TestCase):
    def _make_report_stub(self, f1: float, error_count: int = 1):
        return SimpleNamespace(
            dataset="val",
            errors=[{"sample_id": "e"}] * error_count,
            metrics=SimpleNamespace(
                total_samples=4,
                accuracy=f1,
                precision=f1,
                recall=f1,
                f1_score=f1,
                true_positive=2,
                true_negative=1,
                false_positive=0,
                false_negative=1,
            ),
        )

    def test_build_review_summary_for_pending_formal_review(self):
        summary = _build_review_summary(
            base_version="minor-detection",
            final_version="minor-detection-optimized",
            adopted_version="minor-detection",
            review_required=True,
            review_status="pending",
            review_artifact={
                "review_diff_path": "skills/minor-detection-optimized/review/formal_skill_review_vs_minor-detection.md",
                "review_summary_path": "skills/minor-detection-optimized/review/formal_skill_review_vs_minor-detection.json",
            },
            baseline_f1=0.6,
            best_val_f1=0.8,
        )

        self.assertTrue(summary["needs_human_review"])
        self.assertEqual(summary["current_status"], "候选版本已生成，等待人工审核")
        self.assertEqual(summary["base_version"], "minor-detection")
        self.assertEqual(summary["candidate_version"], "minor-detection-optimized")
        self.assertEqual(summary["adopted_version"], "minor-detection")
        self.assertEqual(summary["f1_change"], 0.2)
        self.assertIn("approve", summary["boss_recommendation"])

    def test_build_review_summary_for_direct_adoption(self):
        summary = _build_review_summary(
            base_version="teen_detector_v1",
            final_version="teen_detector_v2",
            adopted_version="teen_detector_v2",
            review_required=False,
            review_status="not_required",
            review_artifact=None,
            baseline_f1=0.6,
            best_val_f1=0.8,
        )

        self.assertFalse(summary["needs_human_review"])
        self.assertEqual(summary["current_status"], "无需人工审核")
        self.assertEqual(summary["candidate_version"], "teen_detector_v2")
        self.assertEqual(summary["adopted_version"], "teen_detector_v2")
        self.assertEqual(summary["f1_change"], 0.2)
        self.assertIsNone(summary["review_diff_path"])

    def test_pipeline_step_evolution_reuses_accepted_report_in_next_round(self):
        baseline_report = self._make_report_stub(0.6, error_count=2)
        accepted_report = self._make_report_stub(0.8, error_count=0)
        seen_baselines = []

        def fake_cycle(
            current_version="teen_detector_v1",
            max_samples=None,
            optimization_max_errors=None,
            sample_strategy=None,
            sample_seed=None,
            dry_run=False,
            auto_rollback=True,
            min_f1_improvement=0.0,
            retriever=None,
            use_memory=False,
            baseline_report=None,
            activate_accepted_version=True,
        ):
            seen_baselines.append(baseline_report)
            if len(seen_baselines) == 1:
                return {
                    "success": True,
                    "new_version": "teen_detector_v2",
                    "baseline_f1": 0.6,
                    "new_f1": 0.8,
                    "f1_delta": 0.2,
                    "rolled_back": False,
                    "baseline_report_reused": True,
                    "active_version": "teen_detector_v2",
                    "_new_report": accepted_report,
                }

            return {
                "success": True,
                "message": "no errors to optimize",
                "baseline_f1": 0.8,
                "rolled_back": False,
                "baseline_report_reused": True,
            }

        with mock.patch("src.evolution.optimizer.run_optimization_cycle", side_effect=fake_cycle):
            result = step_evolution(
                current_version="teen_detector_v1",
                max_rounds=2,
                max_eval_samples=8,
                retriever=None,
                baseline_report=baseline_report,
                patience=1,
                min_improvement=0.001,
            )

        self.assertIs(seen_baselines[0], baseline_report)
        self.assertIs(seen_baselines[1], accepted_report)
        self.assertEqual(result["final_version"], "teen_detector_v2")
        self.assertIs(result["final_report"], accepted_report)
        self.assertEqual(len(result["round_history"]), 2)
        self.assertTrue(result["round_history"][0]["accepted"])
        self.assertEqual(result["round_history"][0]["version_after"], "teen_detector_v2")
        self.assertFalse(result["round_history"][1]["accepted"])
        self.assertEqual(result["round_history"][1]["version_after"], "teen_detector_v2")
        self.assertEqual(result["stop_reason"], "patience_exhausted")

    def test_pipeline_step_evolution_stops_immediately_when_no_errors_initially(self):
        baseline_report = self._make_report_stub(1.0, error_count=0)

        def fake_cycle(**kwargs):
            return {
                "success": True,
                "message": "no errors to optimize",
                "baseline_f1": 1.0,
                "rolled_back": False,
                "baseline_report_reused": True,
            }

        with mock.patch("src.evolution.optimizer.run_optimization_cycle", side_effect=fake_cycle):
            result = step_evolution(
                current_version="teen_detector_v1",
                max_rounds=5,
                max_eval_samples=8,
                retriever=None,
                baseline_report=baseline_report,
                patience=2,
                min_improvement=0.001,
            )

        self.assertEqual(result["stop_reason"], "no_errors_initially")
        self.assertEqual(len(result["round_history"]), 1)
        self.assertEqual(result["best_version"], "teen_detector_v1")

    def test_pipeline_step_evolution_outputs_formal_review_artifact(self):
        root = fresh_case_dir("pipeline_formal_review_artifact")
        skills_dir = root / "skills"
        formal_dir = skills_dir / "minor-detection"
        candidate_dir = skills_dir / "minor-detection-optimized"
        (formal_dir / "references").mkdir(parents=True, exist_ok=True)
        (candidate_dir / "references").mkdir(parents=True, exist_ok=True)
        (formal_dir / "SKILL.md").write_text("# formal v1\n- 原始流程\n", encoding="utf-8")
        (formal_dir / "references" / "evidence-rules.md").write_text("# rules\n- 原始规则\n", encoding="utf-8")
        (formal_dir / "references" / "icbo-guidelines.md").write_text("# icbo\n", encoding="utf-8")
        (formal_dir / "references" / "output-schema.md").write_text("# schema\n", encoding="utf-8")
        (candidate_dir / "SKILL.md").write_text("# formal v2\n- 更清晰的流程\n", encoding="utf-8")
        (candidate_dir / "references" / "evidence-rules.md").write_text("# rules\n- 新规则\n", encoding="utf-8")
        (candidate_dir / "references" / "icbo-guidelines.md").write_text("# icbo\n", encoding="utf-8")
        (candidate_dir / "references" / "output-schema.md").write_text("# schema\n", encoding="utf-8")

        baseline_report = self._make_report_stub(0.6, error_count=2)
        accepted_report = self._make_report_stub(0.8, error_count=0)

        def fake_cycle(**kwargs):
            return {
                "success": True,
                "new_version": "minor-detection-optimized",
                "baseline_f1": 0.6,
                "new_f1": 0.8,
                "f1_delta": 0.2,
                "rolled_back": False,
                "baseline_report_reused": True,
                "active_version": "minor-detection",
                "review_required": True,
                "adoption_status": "pending_review",
                "_new_report": accepted_report,
            }

        with mock.patch("src.config.SKILLS_DIR", skills_dir), mock.patch(
            "src.evolution.optimizer.SKILLS_DIR", skills_dir
        ), mock.patch("src.evolution.optimizer.run_optimization_cycle", side_effect=fake_cycle):
            result = step_evolution(
                current_version="minor-detection",
                max_rounds=1,
                max_eval_samples=8,
                retriever=None,
                baseline_report=baseline_report,
                patience=1,
                min_improvement=0.001,
            )

        self.assertTrue(result["review_required"])
        self.assertEqual(result["review_status"], "pending")
        self.assertEqual(result["adopted_version"], "minor-detection")
        self.assertIsNotNone(result["review_artifact"])
        review_diff_path = Path(result["review_artifact"]["review_diff_path"])
        self.assertTrue(review_diff_path.exists())
        review_text = review_diff_path.read_text(encoding="utf-8")
        self.assertIn("minor-detection-optimized", review_text)
        self.assertIn("SKILL.md", review_text)

    def test_pipeline_arg_parser_maps_compatibility_flags(self):
        args = parse_pipeline_args(["--max-eval", "123", "--rounds", "4"])

        self.assertTrue(args.full)
        self.assertEqual(args.max_rounds, 4)
        self.assertEqual(args.baseline_max_eval, 123)
        self.assertEqual(args.evolution_max_eval, 123)
        self.assertEqual(args.test_max_eval, 123)
        self.assertEqual(args.baseline_memory_mode, "off")
        self.assertEqual(args.evolution_memory, "off")

    def test_pipeline_arg_parser_supports_formal_skill_mode(self):
        args = parse_pipeline_args(["--formal-skill"])

        self.assertEqual(args.skill_version, "minor-detection")
        self.assertEqual(args.skill_mode, "formal")

    def test_pipeline_arg_parser_rejects_conflicting_formal_skill_and_legacy_version(self):
        with self.assertRaises(ValueError):
            parse_pipeline_args(["--formal-skill", "--skill-version", "teen_detector_v1"])

    def test_pipeline_step_evaluate_resolves_formal_skill_markdown(self):
        root = fresh_case_dir("pipeline_step_evaluate_formal")
        skills_dir = root / "skills"
        formal_dir = skills_dir / "minor-detection"
        formal_dir.mkdir(parents=True, exist_ok=True)
        (formal_dir / "SKILL.md").write_text("# formal", encoding="utf-8")

        fake_report = self._make_report_stub(0.88, error_count=1)
        fake_executor_instances = []

        class FakeExecutor:
            def __init__(self, skill_path=None, llm_client=None, inference_temperature=0.2):
                self.skill_path = skill_path
                fake_executor_instances.append(self)

        class FakeEvaluator:
            def __init__(self, executor=None, skill_version="unknown", retriever=None, use_memory=False):
                self.executor = executor
                self.skill_version = skill_version

            def evaluate(
                self,
                dataset_path=None,
                max_samples=None,
                sample_strategy=None,
                sample_seed=None,
                use_test_set=False,
            ):
                return fake_report

        with mock.patch("src.config.SKILLS_DIR", skills_dir), mock.patch(
            "scripts.run_pipeline.SKILLS_DIR", skills_dir, create=True
        ), mock.patch("src.executor.ExecutorSkill", FakeExecutor), mock.patch(
            "src.evolution.evaluator.SkillEvaluator", FakeEvaluator
        ):
            result = step_evaluate(
                skill_version="minor-detection",
                max_samples=4,
                retriever=None,
                dataset="val",
            )

        self.assertIsNotNone(result)
        self.assertEqual(Path(fake_executor_instances[0].skill_path).name, "SKILL.md")
        self.assertEqual(result["snapshot"]["f1_score"], fake_report.metrics.f1_score)

    def test_pipeline_evaluate_rag_mode_compare_selects_requested_mainline(self):
        rag_report = self._make_report_stub(0.9, error_count=1)
        no_rag_report = self._make_report_stub(0.7, error_count=2)

        with mock.patch(
            "scripts.run_pipeline.step_evaluate",
            side_effect=[
                {"report": rag_report, "snapshot": {"f1_score": 0.9}, "f1": 0.9, "accuracy": 0.9},
                {"report": no_rag_report, "snapshot": {"f1_score": 0.7}, "f1": 0.7, "accuracy": 0.7},
            ],
        ):
            result = evaluate_rag_mode(
                skill_version="teen_detector_v1",
                max_samples=10,
                rag_mode="compare",
                mainline_mode="off",
                dataset="val",
                retriever=object(),
            )

        self.assertEqual(result["selected_mainline"], {"rag": "off", "memory": "off"})
        self.assertEqual(result["selected_f1"], 0.7)
        self.assertAlmostEqual(result["rag_minus_no_rag_f1"], 0.2)

    def test_pipeline_evaluate_rag_mode_can_compare_memory(self):
        no_memory_report = self._make_report_stub(0.7, error_count=2)
        memory_report = self._make_report_stub(0.9, error_count=1)

        with mock.patch(
            "scripts.run_pipeline.step_evaluate",
            side_effect=[
                {"report": memory_report, "snapshot": {"f1_score": 0.9}, "f1": 0.9, "accuracy": 0.9},
                {"report": no_memory_report, "snapshot": {"f1_score": 0.7}, "f1": 0.7, "accuracy": 0.7},
            ],
        ):
            result = evaluate_rag_mode(
                skill_version="teen_detector_v1",
                max_samples=10,
                rag_mode="off",
                mainline_mode="off",
                dataset="val",
                retriever=None,
                memory_mode="compare",
                mainline_memory="on",
            )

        self.assertEqual(result["selected_mainline"], {"rag": "off", "memory": "on"})
        self.assertAlmostEqual(result["memory_minus_no_memory_f1"], 0.2)

    def test_pipeline_cleanup_can_remove_generated_skills_and_restore_active_pointer(self):
        root = fresh_case_dir("pipeline_cleanup")
        data_dir = root / "data"
        benchmark_dir = data_dir / "benchmark"
        retrieval_dir = data_dir / "retrieval_db"
        skills_dir = root / "skills"
        manifest_path = benchmark_dir / "manifest.json"

        benchmark_dir.mkdir(parents=True, exist_ok=True)
        retrieval_dir.mkdir(parents=True, exist_ok=True)
        (benchmark_dir / "train.jsonl").write_text("{}", encoding="utf-8")
        (benchmark_dir / "val.jsonl").write_text("{}", encoding="utf-8")
        (benchmark_dir / "test.jsonl").write_text("{}", encoding="utf-8")
        manifest_path.write_text("{}", encoding="utf-8")
        (retrieval_dir / "index.pkl").write_text("idx", encoding="utf-8")

        (skills_dir / "teen_detector_v1").mkdir(parents=True, exist_ok=True)
        (skills_dir / "teen_detector_v2").mkdir(parents=True, exist_ok=True)
        (skills_dir / "teen_detector_v1" / "skill.md").write_text("# v1", encoding="utf-8")
        (skills_dir / "teen_detector_v2" / "skill.md").write_text("# v2", encoding="utf-8")
        pointer = skills_dir / "active_version.txt"
        pointer.write_text("teen_detector_v2", encoding="utf-8")

        class FakePreparer:
            def __init__(self):
                self.output_dir = benchmark_dir

            def cleanup(self):
                for name in ["train.jsonl", "val.jsonl", "test.jsonl"]:
                    path = benchmark_dir / name
                    if path.exists():
                        path.unlink()
                index_path = retrieval_dir / "index.pkl"
                if index_path.exists():
                    index_path.unlink()

        with mock.patch("scripts.prepare_data.DataPreparer", FakePreparer), mock.patch(
            "src.config.DATA_DIR", data_dir
        ), mock.patch("src.config.SKILLS_DIR", skills_dir), mock.patch(
            "src.config.ACTIVE_SKILL_POINTER", pointer
        ), mock.patch(
            "scripts.run_pipeline.BENCHMARK_MANIFEST_PATH", manifest_path
        ):
            summary = _cleanup_pipeline_artifacts(
                cleanup_assets=False,
                cleanup_generated_skills=True,
                created_versions=["teen_detector_v2"],
                restore_version="teen_detector_v1",
            )

        self.assertTrue(summary["performed"])
        self.assertEqual(summary["restored_active_version"], "teen_detector_v1")
        self.assertIn("teen_detector_v2", summary["removed_generated_skills"])
        self.assertFalse((skills_dir / "teen_detector_v2").exists())
        self.assertTrue((benchmark_dir / "train.jsonl").exists())
        self.assertTrue(manifest_path.exists())
        self.assertTrue((retrieval_dir / "index.pkl").exists())
        self.assertEqual(pointer.read_text(encoding="utf-8").strip(), "teen_detector_v1")

    def test_pipeline_cleanup_flag_removes_benchmark_and_rag_assets(self):
        root = fresh_case_dir("pipeline_cleanup_assets")
        data_dir = root / "data"
        benchmark_dir = data_dir / "benchmark"
        retrieval_dir = data_dir / "retrieval_db"
        manifest_path = benchmark_dir / "manifest.json"

        benchmark_dir.mkdir(parents=True, exist_ok=True)
        retrieval_dir.mkdir(parents=True, exist_ok=True)
        (benchmark_dir / "train.jsonl").write_text("{}", encoding="utf-8")
        (benchmark_dir / "val.jsonl").write_text("{}", encoding="utf-8")
        (benchmark_dir / "test.jsonl").write_text("{}", encoding="utf-8")
        manifest_path.write_text("{}", encoding="utf-8")
        (retrieval_dir / "index.pkl").write_text("idx", encoding="utf-8")

        class FakePreparer:
            def __init__(self):
                self.output_dir = benchmark_dir

        with mock.patch("scripts.prepare_data.DataPreparer", FakePreparer), mock.patch(
            "src.config.DATA_DIR", data_dir
        ), mock.patch(
            "scripts.run_pipeline.BENCHMARK_MANIFEST_PATH", manifest_path
        ):
            summary = _cleanup_pipeline_artifacts(
                cleanup_assets=True,
                cleanup_generated_skills=False,
                created_versions=[],
                restore_version=None,
            )

        self.assertTrue(summary["performed"])
        self.assertFalse((benchmark_dir / "train.jsonl").exists())
        self.assertFalse(manifest_path.exists())
        self.assertFalse((retrieval_dir / "index.pkl").exists())

    def test_pipeline_rejects_invalid_baseline_and_evolution_rag_combination(self):
        with self.assertRaisesRegex(ValueError, "baseline-rag-mode=off cannot feed evolution-rag=on"):
            _validate_pipeline_config(
                baseline_rag_mode="off",
                evolution_rag="on",
                baseline_memory_mode="off",
                evolution_memory="off",
            )

        with self.assertRaisesRegex(ValueError, "baseline-rag-mode=on cannot feed evolution-rag=off"):
            _validate_pipeline_config(
                baseline_rag_mode="on",
                evolution_rag="off",
                baseline_memory_mode="off",
                evolution_memory="off",
            )

        with self.assertRaisesRegex(ValueError, "baseline-memory-mode=off cannot feed evolution-memory=on"):
            _validate_pipeline_config(
                baseline_rag_mode="off",
                evolution_rag="off",
                baseline_memory_mode="off",
                evolution_memory="on",
            )

        with self.assertRaisesRegex(ValueError, "baseline-memory-mode=on cannot feed evolution-memory=off"):
            _validate_pipeline_config(
                baseline_rag_mode="off",
                evolution_rag="off",
                baseline_memory_mode="on",
                evolution_memory="off",
            )

        _validate_pipeline_config(
            baseline_rag_mode="compare",
            evolution_rag="on",
            baseline_memory_mode="compare",
            evolution_memory="on",
        )
        _validate_pipeline_config(
            baseline_rag_mode="compare",
            evolution_rag="off",
            baseline_memory_mode="compare",
            evolution_memory="off",
        )

    def test_pipeline_can_repair_benchmark_manifest_without_touching_existing_splits(self):
        root = fresh_case_dir("repair_benchmark_manifest")
        benchmark_dir = root / "benchmark"
        benchmark_dir.mkdir(parents=True, exist_ok=True)

        train_path = benchmark_dir / "train.jsonl"
        val_path = benchmark_dir / "val.jsonl"
        test_path = benchmark_dir / "test.jsonl"
        manifest_path = benchmark_dir / "manifest.json"

        train_content = "\n".join(
            [
                json.dumps({"sample_id": "t1", "source": "social_pos", "is_minor": True}, ensure_ascii=False),
                json.dumps({"sample_id": "t2", "source": "social_neg", "is_minor": False}, ensure_ascii=False),
            ]
        ) + "\n"
        val_content = json.dumps({"sample_id": "v1", "source": "knowledge_pos", "is_minor": True}, ensure_ascii=False) + "\n"
        test_content = json.dumps({"sample_id": "x1", "source": "knowledge_neg", "is_minor": False}, ensure_ascii=False) + "\n"

        train_path.write_text(train_content, encoding="utf-8")
        val_path.write_text(val_content, encoding="utf-8")
        test_path.write_text(test_content, encoding="utf-8")

        with mock.patch("src.config.BENCHMARK_TRAIN_PATH", train_path), mock.patch(
            "src.config.BENCHMARK_VAL_PATH", val_path
        ), mock.patch("src.config.BENCHMARK_TEST_PATH", test_path), mock.patch(
            "scripts.run_pipeline.BENCHMARK_MANIFEST_PATH", manifest_path
        ):
            result = repair_benchmark_manifest(requested_mode="full", max_per_source=None, seed=42)

        self.assertTrue(result["success"])
        self.assertTrue(manifest_path.exists())
        self.assertEqual(train_path.read_text(encoding="utf-8"), train_content)
        self.assertEqual(val_path.read_text(encoding="utf-8"), val_content)
        self.assertEqual(test_path.read_text(encoding="utf-8"), test_content)

        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        self.assertEqual(payload["mode"], "full")
        self.assertIsNone(payload["quick_n"])
        self.assertEqual(payload["seed"], 42)
        self.assertEqual(payload["splits"], {"train": 2, "val": 1, "test": 1})
        self.assertEqual(payload["statistics"]["total"], 4)
        self.assertEqual(payload["statistics"]["by_is_minor"], {"minor": 2, "adult": 2})
        self.assertEqual(
            payload["paths"],
            {
                "train": "train.jsonl",
                "val": "val.jsonl",
                "test": "test.jsonl",
            },
        )
        self.assertEqual(payload["provenance"], "repaired_from_existing_splits")

    def test_pipeline_run_report_rewrites_absolute_project_paths_as_relative(self):
        case_dir = fresh_case_dir("pipeline_report_relative_paths").resolve()
        data_path = case_dir / "data" / "benchmark" / "val.jsonl"
        data_path.parent.mkdir(parents=True, exist_ok=True)
        data_path.write_text("{}", encoding="utf-8")

        with mock.patch("scripts.run_pipeline.ROOT_DIR", case_dir):
            report_path = _save_run_report(
                {
                    "baseline": {
                        "selected_eval": {
                            "dataset": str(data_path),
                        }
                    }
                }
            )

        payload = json.loads(report_path.read_text(encoding="utf-8"))
        self.assertEqual(payload["baseline"]["selected_eval"]["dataset"], "../../data/benchmark/val.jsonl")

    def test_pipeline_repair_benchmark_manifest_requires_existing_splits(self):
        root = fresh_case_dir("repair_benchmark_manifest_missing")
        benchmark_dir = root / "benchmark"
        benchmark_dir.mkdir(parents=True, exist_ok=True)
        train_path = benchmark_dir / "train.jsonl"
        val_path = benchmark_dir / "val.jsonl"
        test_path = benchmark_dir / "test.jsonl"
        manifest_path = benchmark_dir / "manifest.json"

        train_path.write_text("{}", encoding="utf-8")
        val_path.write_text("{}", encoding="utf-8")

        with mock.patch("src.config.BENCHMARK_TRAIN_PATH", train_path), mock.patch(
            "src.config.BENCHMARK_VAL_PATH", val_path
        ), mock.patch("src.config.BENCHMARK_TEST_PATH", test_path), mock.patch(
            "scripts.run_pipeline.BENCHMARK_MANIFEST_PATH", manifest_path
        ):
            result = repair_benchmark_manifest(requested_mode="full", max_per_source=None, seed=42)

        self.assertFalse(result["success"])
        self.assertEqual(result["error_type"], "missing_assets")
        self.assertFalse(manifest_path.exists())

    def test_pipeline_reuses_existing_verified_full_benchmark_by_default(self):
        case_dir = fresh_case_dir("reuse_benchmark")
        manifest_path = case_dir / "manifest.json"
        manifest_path.write_text(
            json.dumps(
                {
                    "mode": "full",
                    "quick_n": None,
                    "seed": 42,
                    "stratification": {"strategy": "source_age_bucket_v1"},
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

        with mock.patch("scripts.run_pipeline.BENCHMARK_MANIFEST_PATH", manifest_path), mock.patch(
            "scripts.run_pipeline._benchmark_files_exist", return_value=True
        ), mock.patch("scripts.run_pipeline.step_prepare_data") as mocked_prepare:
            result = ensure_benchmark_data(max_per_source=None, seed=42, rebuild=False, requested_mode="full")

        self.assertTrue(result["success"])
        self.assertTrue(result["reused"])
        self.assertTrue(result["verified"])
        mocked_prepare.assert_not_called()

    def test_pipeline_fails_on_benchmark_mode_mismatch_without_rebuild(self):
        case_dir = fresh_case_dir("rebuild_benchmark_mismatch")
        manifest_path = case_dir / "manifest.json"
        manifest_path.write_text(
            json.dumps({"mode": "quick", "quick_n": 50, "seed": 42}, ensure_ascii=False),
            encoding="utf-8",
        )

        with mock.patch("scripts.run_pipeline.BENCHMARK_MANIFEST_PATH", manifest_path), mock.patch(
            "scripts.run_pipeline._benchmark_files_exist", return_value=True
        ), mock.patch(
            "scripts.run_pipeline.step_prepare_data",
            return_value={"success": True, "paths": {}},
        ) as mocked_prepare:
            result = ensure_benchmark_data(max_per_source=None, seed=42, rebuild=False, requested_mode="full")

        self.assertFalse(result["success"])
        self.assertEqual(result["error_type"], "asset_mode_mismatch")
        self.assertIn("manifest mode mismatch", result["reason"])
        mocked_prepare.assert_not_called()

    def test_pipeline_can_rebuild_benchmark_when_manifest_mismatches_requested_mode(self):
        case_dir = fresh_case_dir("rebuild_benchmark_allowed")
        manifest_path = case_dir / "manifest.json"
        manifest_path.write_text(
            json.dumps({"mode": "quick", "quick_n": 50, "seed": 42}, ensure_ascii=False),
            encoding="utf-8",
        )

        with mock.patch("scripts.run_pipeline.BENCHMARK_MANIFEST_PATH", manifest_path), mock.patch(
            "scripts.run_pipeline._benchmark_files_exist", return_value=True
        ), mock.patch(
            "scripts.run_pipeline.step_prepare_data",
            return_value={"success": True, "paths": {}},
        ) as mocked_prepare:
            result = ensure_benchmark_data(max_per_source=None, seed=42, rebuild=True, requested_mode="full")

        self.assertTrue(result["success"])
        self.assertFalse(result["reused"])
        self.assertIn("manifest mode mismatch", result["rebuild_reason"])
        mocked_prepare.assert_called_once()

    def test_pipeline_reuses_existing_rag_index_by_default(self):
        with mock.patch("scripts.run_pipeline._rag_assets_exist", return_value=True), mock.patch(
            "scripts.run_pipeline.step_build_rag_index"
        ) as mocked_build:
            result = ensure_rag_index(max_samples=100, rebuild=False)

        self.assertTrue(result["success"])
        self.assertTrue(result["reused"])
        mocked_build.assert_not_called()

    def test_pipeline_fails_when_rag_index_missing_without_rebuild(self):
        with mock.patch("scripts.run_pipeline._rag_assets_exist", return_value=False), mock.patch(
            "scripts.run_pipeline.step_build_rag_index"
        ) as mocked_build:
            result = ensure_rag_index(max_samples=100, rebuild=False)

        self.assertFalse(result["success"])
        self.assertEqual(result["error_type"], "missing_assets")
        mocked_build.assert_not_called()


if __name__ == "__main__":
    unittest.main()

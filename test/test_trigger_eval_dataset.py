import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from src.trigger_eval import TriggerEvalBuildConfig, TriggerEvalSplitConfig, build_trigger_eval_dataset, split_trigger_eval_dataset


class TriggerEvalDatasetTests(unittest.TestCase):
    def test_build_trigger_eval_dataset_with_synthetic_inputs(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            benchmark_path = root / "benchmark.jsonl"
            quotas_path = root / "quotas.json"
            output_dir = root / "out"
            predicates_path = ROOT_DIR / "data" / "trigger_eval" / "description_v1_predicates.json"

            quotas = {
                "version": "test-v1",
                "window_scan": {
                    "identity_explicit": 1,
                    "school_context_strong": 1,
                    "implicit_minor_signal": 1,
                    "adult_near_miss": 1,
                    "topic_adjacent_not_identity": 1,
                    "insufficient_evidence_window": 1,
                },
                "direct_request": {
                    "explicit_minor_judgment_request": 1,
                    "operator_window_scan_request": 1,
                    "other_profile_request": 1,
                    "analysis_but_not_minor": 1,
                },
            }
            quotas_path.write_text(json.dumps(quotas, ensure_ascii=False, indent=2), encoding="utf-8")

            rows = [
                {
                    "sample_id": "identity-explicit-1",
                    "is_minor": True,
                    "conversation": [
                        {"role": "user", "content": "我初三了，最近月考一塌糊涂。"},
                        {"role": "assistant", "content": "先说说最困扰你的部分。"},
                        {"role": "user", "content": "班主任今天又找我谈成绩。"},
                        {"role": "assistant", "content": "这听起来压力很大。"},
                        {"role": "user", "content": "我妈让我晚自习后继续写作业。"},
                        {"role": "assistant", "content": "我们可以一起拆一下接下来的安排。"},
                    ],
                },
                {
                    "sample_id": "school-context-1",
                    "is_minor": True,
                    "conversation": [
                        {"role": "user", "content": "学校今天一直催我补交作业。"},
                        {"role": "assistant", "content": "你现在最担心哪部分？"},
                        {"role": "user", "content": "老师刚在班群里发了这次测验成绩。"},
                        {"role": "assistant", "content": "你看到成绩后是什么感受？"},
                        {"role": "user", "content": "我妈还说明天要给我签字。"},
                        {"role": "assistant", "content": "这确实会让人紧张。"},
                    ],
                },
                {
                    "sample_id": "implicit-signal-1",
                    "is_minor": True,
                    "conversation": [
                        {"role": "user", "content": "救命，我真的要疯了。"},
                        {"role": "assistant", "content": "发生什么了？"},
                        {"role": "user", "content": "呜呜，我昨晚偷玩手机又被收手机了。"},
                        {"role": "assistant", "content": "这件事让你很难受。"},
                        {"role": "user", "content": "班群里还一直催我补作业。"},
                        {"role": "assistant", "content": "我们一起想想怎么处理。"},
                    ],
                },
                {
                    "sample_id": "adult-near-miss-1",
                    "is_minor": False,
                    "conversation": [
                        {"role": "user", "content": "我大四了，这学期一边实习一边准备教资。"},
                        {"role": "assistant", "content": "听起来任务很多。"},
                        {"role": "user", "content": "学校还催毕业论文，我真的有点顶不住。"},
                        {"role": "assistant", "content": "你想先解决哪一部分？"},
                        {"role": "user", "content": "面试时间又和老板那边安排撞上了。"},
                        {"role": "assistant", "content": "我们可以一起排一下优先级。"},
                    ],
                },
                {
                    "sample_id": "topic-adjacent-1",
                    "is_minor": False,
                    "conversation": [
                        {"role": "user", "content": "我想做一个关于青少年早恋和校园欺凌的科普提纲。"},
                        {"role": "assistant", "content": "你更想覆盖哪些部分？"},
                        {"role": "user", "content": "重点讲中学生心理和学校教育。"},
                        {"role": "assistant", "content": "可以从议题拆分开始。"},
                        {"role": "user", "content": "不用判断说话者年龄，只做内容总结。"},
                        {"role": "assistant", "content": "明白，我们聚焦内容结构。"},
                    ],
                },
                {
                    "sample_id": "insufficient-window-1",
                    "is_minor": True,
                    "conversation": [
                        {"role": "user", "content": "我今天心情很差。"},
                        {"role": "assistant", "content": "愿意说说发生了什么吗？"},
                        {"role": "user", "content": "晚上回家还要处理一堆事。"},
                        {"role": "assistant", "content": "听起来你已经很累了。"},
                        {"role": "user", "content": "明天估计会更忙。"},
                        {"role": "assistant", "content": "我们先把眼前的事理一理。"},
                        {"role": "user", "content": "学校刚通知补课，我妈还要给我签字。"},
                        {"role": "assistant", "content": "这一下子把压力都堆上来了。"},
                    ],
                },
            ]
            with benchmark_path.open("w", encoding="utf-8") as f:
                for row in rows:
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")

            config = TriggerEvalBuildConfig(
                benchmark_path=benchmark_path,
                extra_sources=(),
                quotas_path=quotas_path,
                predicates_path=predicates_path,
                output_dir=output_dir,
                output_stem="synthetic_trigger_eval",
                sample_seed=7,
                window_size_turns_candidates=(6,),
                stride_turns=2,
                min_user_turns=3,
                max_assistant_share=0.6,
                target_window_scan_val_ratio=0.85,
                max_samples_per_base=2,
            )

            result = build_trigger_eval_dataset(config=config)

            samples = result["samples"]
            self.assertEqual(len(samples), 10)
            self.assertEqual(sum(1 for sample in samples if sample["scenario"] == "window_scan"), 6)
            self.assertEqual(sum(1 for sample in samples if sample["scenario"] == "direct_request"), 4)
            self.assertEqual(sum(1 for sample in samples if sample["should_trigger"] is True), 5)
            self.assertEqual(sum(1 for sample in samples if sample["should_trigger"] is False), 5)
            self.assertEqual(result["summary"]["sample_count"], 10)
            self.assertEqual(result["summary"]["scenario_counts"]["window_scan"], 6)
            self.assertEqual(result["summary"]["scenario_counts"]["direct_request"], 4)
            self.assertEqual(result["summary"]["window_size_counts"][6], 6)
            self.assertEqual(result["summary"]["validation_errors"], [])

            for sample in samples:
                self.assertIn("skill_input_turns", sample)
                self.assertIn("skill_input_base_sample_id", sample)
                self.assertIn("expected_is_minor", sample)
                self.assertIsInstance(sample["skill_input_turns"], list)

            for key in [
                "dataset_path",
                "summary_json_path",
                "summary_md_path",
                "review_pack_path",
                "ambiguous_rejects_path",
            ]:
                self.assertTrue(Path(result["paths"][key]).exists(), msg=key)

    def test_negative_direct_request_templates_are_natural_tasks_not_explicit_minor_negations(self):
        predicates_path = ROOT_DIR / "data" / "trigger_eval" / "description_v1_predicates.json"
        payload = json.loads(predicates_path.read_text(encoding="utf-8-sig"))
        templates = payload["direct_request_templates"]
        negative_templates = templates["other_profile_request"] + templates["analysis_but_not_minor"]
        banned_phrases = [
            "不需要判断是否未成年",
            "不要做未成年人识别",
            "不是判断年龄",
            "不需要做未成年人身份判断",
        ]

        for item in negative_templates:
            template = str(item["template"])
            for phrase in banned_phrases:
                self.assertNotIn(phrase, template, msg=item["id"])

    def test_split_trigger_eval_dataset_preserves_strata(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            dataset_path = root / "trigger_eval.json"
            output_dir = root / "split"
            samples = []
            for scenario in ("window_scan", "direct_request"):
                for should_trigger in (True, False):
                    slice_name = f"{scenario}_{'pos' if should_trigger else 'neg'}"
                    for index in range(5):
                        samples.append(
                            {
                                "id": f"{slice_name}_{index}",
                                "scenario": scenario,
                                "should_trigger": should_trigger,
                                "slice": slice_name,
                                "source": "synthetic",
                                "query": f"query-{slice_name}-{index}",
                                "skill_input_turns": [{"role": "user", "content": "x"}],
                                "skill_input_base_sample_id": f"base-{slice_name}-{index}",
                                "expected_is_minor": should_trigger,
                                "label_reason": "ok",
                                "trigger_basis": [slice_name],
                                "generation_rule": "synthetic",
                                "evidence_strength": "medium",
                            }
                        )
            dataset_path.write_text(json.dumps({"metadata": {"version": "test"}, "samples": samples}, ensure_ascii=False, indent=2), encoding="utf-8")

            result = split_trigger_eval_dataset(
                config=TriggerEvalSplitConfig(
                    dataset_path=dataset_path,
                    output_dir=output_dir,
                    output_stem="synthetic_trigger_eval",
                    final_validation_ratio=0.4,
                    sample_seed=7,
                )
            )

            self.assertEqual(len(result["optimization_set"]), 12)
            self.assertEqual(len(result["final_validation_set"]), 8)
            self.assertTrue(Path(result["paths"]["optimization_set_path"]).exists())
            self.assertTrue(Path(result["paths"]["final_validation_set_path"]).exists())
            self.assertTrue(Path(result["paths"]["summary_path"]).exists())
            for stratum, counts in result["summary"]["stratum_counts"].items():
                self.assertEqual(counts["total"], 5)
                self.assertEqual(counts["optimization_set"], 3)
                self.assertEqual(counts["final_validation_set"], 2)


if __name__ == "__main__":
    unittest.main()

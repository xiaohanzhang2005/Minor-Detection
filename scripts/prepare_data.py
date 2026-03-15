"""
数据基座构建脚本
加载全部 4 个数据集（社交正/负 + 知识正/负），
按分层抽样划分为 train(70%) / val(10%) / test(20%)，
输出到 data/benchmark/
"""

import json
import random
import os
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from collections import defaultdict

# 添加项目根目录
ROOT_DIR = Path(__file__).parent.parent.absolute()  # scripts/ -> 项目根目录
sys.path.insert(0, str(ROOT_DIR))

from src.utils.path_utils import to_relative_posix_path


@dataclass
class BenchmarkSample:
    """统一的 Benchmark 样本格式"""
    sample_id: str                      # 唯一标识
    source: str                         # 来源标签
    is_minor: bool                      # 标签：是否未成年人
    conversation: List[Dict[str, str]]  # 对话内容
    icbo_features: Dict[str, str]       # ICBO 特征
    user_persona: Dict[str, Any]        # 用户画像
    extra_info: Optional[Dict] = None   # 额外信息


# ── 4 个数据源定义 ─────────────────────────────────────────
DATA_SOURCES = [
    {
        "name": "社交正样本",
        "tag": "social_pos",
        "path": "社交问答/youth_dialogs.jsonl",
    },
    {
        "name": "社交负样本",
        "tag": "social_neg",
        "path": "社交问答/adult_dialogs.jsonl",
    },
    {
        "name": "知识正样本",
        "tag": "knowledge_pos",
        "path": "知识问答数据库/youth_knowledge_qa.jsonl",
    },
    {
        "name": "知识负样本",
        "tag": "knowledge_neg",
        "path": "知识问答数据库/adult_knowledge_qa.jsonl",
    },
]


class DataPreparer:
    """
    数据准备器
    加载全部 4 个数据源、转换、分层划分
    """

    def __init__(
        self,
        output_dir: Optional[str] = None,
        train_ratio: float = 0.7,
        val_ratio: float = 0.1,
        test_ratio: float = 0.2,
        random_seed: int = 42,
    ):
        self.root_dir = ROOT_DIR
        self.data_dir = self.root_dir / "data"

        # 输出目录
        self.output_dir = Path(output_dir) if output_dir else self.data_dir / "benchmark"
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # 划分比例
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 0.001, "比例之和必须为1"
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio

        # 随机种子
        self.random_seed = random_seed
        random.seed(random_seed)

        # 存储
        self.samples: List[BenchmarkSample] = []

    # ── 通用 JSONL 加载 ──────────────────────────────────
    def _load_jsonl(
        self,
        path: Path,
        source_tag: str,
        max_samples: Optional[int] = None,
    ) -> int:
        """
        从 JSONL 加载样本。
        每条记录必须含 is_minor / conversation / icbo_features / user_persona。
        """
        if not path.exists():
            print(f"[WARN] 文件不存在: {path}")
            return 0

        count = 0
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if max_samples is not None and count >= max_samples:
                    break
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    sample = self._convert(data, source_tag, count)
                    if sample:
                        self.samples.append(sample)
                        count += 1
                except Exception as e:
                    print(f"[WARN] 解析失败 ({path.name}): {e}")
        return count

    def _convert(self, data: Dict, source_tag: str, index: int) -> Optional[BenchmarkSample]:
        """将一条原始 JSON 转为 BenchmarkSample"""
        try:
            raw_sample_id = str(data.get("dataset_id") or "").strip()

            # is_minor —— 直接信任数据文件里已有的标签
            is_minor = data.get("is_minor")
            if is_minor is None:
                # 兜底：从年龄推断
                age = data.get("user_persona", {}).get("age")
                if isinstance(age, str):
                    age = int(age) if age.isdigit() else None
                is_minor = (age < 18) if age is not None else True

            user_persona = data.get("user_persona", {})
            meta = data.get("_meta", {})
            if meta:
                user_persona = {**user_persona}
                user_persona.setdefault("grade", meta.get("_grade_label"))
                user_persona.setdefault("subject", meta.get("_subject"))
                user_persona.setdefault("stage", meta.get("_stage"))

            extra = data.get("extra_info")
            if extra is None and meta:
                extra = {"seed": meta.get("_seed")}

            if extra is None:
                extra = {}
            if raw_sample_id:
                extra = {**extra, "raw_sample_id": raw_sample_id}

            # 基于 source + 行内序号构造稳定唯一 sample_id，避免原始 dataset_id 重复导致 split 假交叉
            sample_id = f"{source_tag}_{index:06d}"
            if raw_sample_id:
                safe_raw_id = raw_sample_id.replace(" ", "_")
                sample_id = f"{sample_id}_{safe_raw_id}"

            return BenchmarkSample(
                sample_id=sample_id,
                source=source_tag,
                is_minor=bool(is_minor),
                conversation=data.get("conversation", []),
                icbo_features=data.get("icbo_features", {}),
                user_persona=user_persona,
                extra_info=extra,
            )
        except Exception as e:
            print(f"[WARN] 转换失败: {e}")
            return None

    # ── 分层划分 ─────────────────────────────────────────
    def split_data(self) -> Dict[str, List[BenchmarkSample]]:
        """
        分层抽样 (Stratified Split)：
        按 (is_minor, source) 分组后各组独立按比例切分，
        保证 train/val/test 中正负样本与场景来源分布一致。
        """
        if not self.samples:
            raise ValueError("没有可划分的样本，请先加载数据")

        # 按 stratum 分组
        strata: Dict[str, List[BenchmarkSample]] = defaultdict(list)
        for s in self.samples:
            key = f"{'minor' if s.is_minor else 'adult'}_{s.source}"
            strata[key].append(s)

        splits: Dict[str, List[BenchmarkSample]] = {"train": [], "val": [], "test": []}

        for stratum_key, group in sorted(strata.items()):
            random.shuffle(group)
            n = len(group)
            t1 = int(n * self.train_ratio)
            t2 = t1 + int(n * self.val_ratio)
            splits["train"].extend(group[:t1])
            splits["val"].extend(group[t1:t2])
            splits["test"].extend(group[t2:])

        # 每个 split 内部再打乱
        for v in splits.values():
            random.shuffle(v)

        total = len(self.samples)
        print(f"\n[INFO] 数据划分结果 (分层抽样):")
        print(f"   总样本数: {total}")
        for split_name, items in splits.items():
            minors = sum(1 for s in items if s.is_minor)
            adults = len(items) - minors
            print(f"   {split_name}: {len(items)} 条 ({len(items)/total*100:.1f}%)  "
                  f"[未成年 {minors} | 成年 {adults}]")

        return splits

    # ── 保存 ─────────────────────────────────────────────
    def save_splits(self, splits: Dict[str, List[BenchmarkSample]]) -> Dict[str, str]:
        paths = {}
        for split_name, samples in splits.items():
            output_path = self.output_dir / f"{split_name}.jsonl"
            with open(output_path, "w", encoding="utf-8") as f:
                for sample in samples:
                    sample_dict = {
                        "sample_id": sample.sample_id,
                        "source": sample.source,
                        "is_minor": sample.is_minor,
                        "conversation": sample.conversation,
                        "icbo_features": sample.icbo_features,
                        "user_persona": sample.user_persona,
                    }
                    if sample.extra_info:
                        sample_dict["extra_info"] = sample.extra_info
                    f.write(json.dumps(sample_dict, ensure_ascii=False) + "\n")
            paths[split_name] = to_relative_posix_path(output_path, self.output_dir)
            print(f"   [OK] {split_name}.jsonl: {len(samples)} 条")
        return paths

    # ── 统计 ─────────────────────────────────────────────
    def get_statistics(self) -> Dict[str, Any]:
        if not self.samples:
            return {"total": 0}

        stats: Dict[str, Any] = {
            "total": len(self.samples),
            "by_source": {},
            "by_is_minor": {"minor": 0, "adult": 0},
        }
        for s in self.samples:
            stats["by_source"][s.source] = stats["by_source"].get(s.source, 0) + 1
            if s.is_minor:
                stats["by_is_minor"]["minor"] += 1
            else:
                stats["by_is_minor"]["adult"] += 1
        return stats

    # ── 主流程 ───────────────────────────────────────────
    def run(self, max_per_source: Optional[int] = None, save: bool = True) -> Dict[str, Any]:
        """
        执行完整数据准备流程。

        Args:
            max_per_source: 每个数据源最多加载的条数。None = 不限。
            save: 是否持久化到 JSONL。
        """
        print("=" * 60)
        print("[DATA] 数据基座构建")
        if max_per_source is not None:
            print(f"   [Quick] 每源最多 {max_per_source} 条")
        else:
            print("   [Full] 加载全量数据")
        print("=" * 60)

        # 加载 4 个数据源
        print("\n[INFO] 加载数据...")
        for src in DATA_SOURCES:
            path = self.data_dir / src["path"]
            n = self._load_jsonl(path, src["tag"], max_per_source)
            print(f"   {src['name']} ({src['tag']}): {n} 条")

        if not self.samples:
            return {"success": False, "error": "没有加载到任何样本"}

        # 统计
        stats = self.get_statistics()
        print(
            f"\n[INFO] 汇总: {stats['total']} 条  "
            f"(未成年 {stats['by_is_minor']['minor']} / "
            f"成年 {stats['by_is_minor']['adult']})"
        )
        for src_tag, cnt in sorted(stats["by_source"].items()):
            print(f"   {src_tag}: {cnt}")

        # 分层划分
        splits = self.split_data()

        # 保存
        paths = {}
        if save:
            print("\n[INFO] 保存文件...")
            paths = self.save_splits(splits)

        return {
            "success": True,
            "statistics": stats,
            "splits": {k: len(v) for k, v in splits.items()},
            "paths": paths,
        }

    def cleanup(self):
        # 清理 benchmark 数据
        for fn in ["train.jsonl", "val.jsonl", "test.jsonl"]:
            fp = self.output_dir / fn
            if fp.exists():
                fp.unlink()
                print(f"[DEL] 已删除: {fp}")
        
        # 清理 RAG 索引
        rag_index = self.data_dir / "retrieval_db" / "index.pkl"
        if rag_index.exists():
            rag_index.unlink()
            print(f"[DEL] 已删除: {rag_index}")
        rag_manifest = self.data_dir / "retrieval_db" / "manifest.json"
        if rag_manifest.exists():
            rag_manifest.unlink()
            print(f"[DEL] Deleted: {rag_manifest}")
        rag_corpus = self.data_dir / "retrieval_corpus" / "cases_v1.jsonl"
        if rag_corpus.exists():
            rag_corpus.unlink()
            print(f"[DEL] Deleted: {rag_corpus}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="数据基座构建 (4 源 → train/val/test)")
    parser.add_argument("--full", action="store_true",
                        help="加载全量数据。默认 Quick 模式每源随机抽 50 条。")
    parser.add_argument("--quick-n", type=int, default=50,
                        help="Quick 模式时每个数据源的样本数 (default: 50)")
    parser.add_argument("--cleanup", action="store_true",
                        help="删除已生成的 benchmark 文件")
    args = parser.parse_args()

    preparer = DataPreparer()

    if args.cleanup:
        preparer.cleanup()
    else:
        max_per_source = None if args.full else args.quick_n
        result = preparer.run(max_per_source=max_per_source, save=True)

        if result["success"]:
            print("\n" + "=" * 60)
            print("[OK] 数据基座构建完成！")
            print(f"   输出目录: {preparer.output_dir}")
            print("=" * 60)

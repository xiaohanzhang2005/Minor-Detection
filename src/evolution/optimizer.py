"""
Skill 优化器
根据评估结果，使用 LLM 优化 skill.md prompt
"""

import json
import random
import shutil
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.config import (
    SKILLS_DIR,
    OPTIMIZER_MODEL,
    BENCHMARK_TRAIN_PATH,
    set_active_skill_version,
)
from src.utils.llm_client import LLMClient
from src.evolution.evaluator import EvaluationReport


OPTIMIZER_SYSTEM_PROMPT = """你是一个专业的 Prompt 工程师，专注于优化青少年识别系统的 Skill Prompt。

你的任务是根据评估报告中的错误案例，分析 skill.md 的不足之处，并提出改进方案。

## 优化原则

1. **保持框架完整**：不要删除 ICBO 框架的核心结构
2. **针对性改进**：根据错误类型添加相应的判断规则
3. **平衡精确与召回**：避免过度优化导致新问题
4. **增量改进**：每次只做小幅改动，避免大改动
5. **保持格式**：输出格式规范与原 skill.md 一致

## 常见错误类型

- FP (误判未成年)：成年人被错误识别为未成年
  - 需要加强成年人特征识别
  - 添加职场、社会身份的判断规则

- FN (漏判未成年)：未成年被错误识别为成年
  - 需要加强校园特征识别
  - 添加隐晦表达的解读规则

## 输出格式

请输出完整的改进后的 skill.md 内容，不要省略任何部分。
不要把改进说明写在 skill.md 内容之外，直接输出完整的 markdown 内容即可。
"""


class SkillOptimizer:
    """
    Skill 优化器
    根据评估报告，使用 LLM 优化 skill prompt
    """
    
    def __init__(
        self,
        llm_client: Optional[LLMClient] = None,
        skills_dir: Optional[str] = None,
    ):
        """
        初始化优化器
        
        Args:
            llm_client: LLM 客户端
            skills_dir: skills 目录路径
        """
        self.llm_client = llm_client or LLMClient(model=OPTIMIZER_MODEL)
        self.skills_dir = Path(skills_dir) if skills_dir else SKILLS_DIR
    
    # ── Train Set 对比检索 ─────────────────────────────
    def _retrieve_contrastive_examples(
        self,
        error_analysis: Dict[str, Any],
        max_per_type: int = 3,
    ) -> str:
        """
        从 Train Set 中检索与错误案例类似的正、负样本，
        帮助 Optimizer 进行归纳推理（Inductive Reasoning）。
        """
        if not BENCHMARK_TRAIN_PATH.exists():
            return ""

        # 一次性加载训练集（已很小，内存安全）
        minor_samples: List[Dict] = []
        adult_samples: List[Dict] = []
        with open(BENCHMARK_TRAIN_PATH, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                s = json.loads(line)
                if s.get("is_minor"):
                    minor_samples.append(s)
                else:
                    adult_samples.append(s)

        parts: List[str] = []

        # 如果有 FP（成年人被误判为未成年），展示真正的成年 vs 未成年对比
        if error_analysis["fp_count"] > 0:
            parts.append("## 训练集对比参考 — FP 修复灵感\n")
            parts.append("以下是训练集中 **真实成年人** 与 **真实未成年人** 的对比案例，")
            parts.append("请从中归纳区分规律并写入 Prompt：\n")
            for label, pool in [("成年人", adult_samples), ("未成年人", minor_samples)]:
                picked = random.sample(pool, min(max_per_type, len(pool)))
                for i, s in enumerate(picked, 1):
                    conv_preview = s.get("conversation", [])[:4]
                    conv_text = "\n".join(
                        f"  {t['role']}: {t['content'][:80]}" for t in conv_preview
                    )
                    icbo = s.get("icbo_features", {})
                    parts.append(f"### {label} 案例 {i}")
                    parts.append(f"对话片段:\n{conv_text}")
                    if icbo:
                        parts.append(f"ICBO: I={icbo.get('intention','?')[:60]} | "
                                     f"C={icbo.get('cognition','?')[:60]}")
                    parts.append("")

        # 如果有 FN（未成年被漏判），展示真正未成年 vs 成年的对比
        if error_analysis["fn_count"] > 0:
            parts.append("## 训练集对比参考 — FN 修复灵感\n")
            parts.append("以下是训练集中 **真实未成年人** 与 **真实成年人** 的对比案例，")
            parts.append("请从中归纳未成年人的隐晦特征并补充进 Prompt：\n")
            for label, pool in [("未成年人", minor_samples), ("成年人", adult_samples)]:
                picked = random.sample(pool, min(max_per_type, len(pool)))
                for i, s in enumerate(picked, 1):
                    conv_preview = s.get("conversation", [])[:4]
                    conv_text = "\n".join(
                        f"  {t['role']}: {t['content'][:80]}" for t in conv_preview
                    )
                    icbo = s.get("icbo_features", {})
                    parts.append(f"### {label} 案例 {i}")
                    parts.append(f"对话片段:\n{conv_text}")
                    if icbo:
                        parts.append(f"ICBO: I={icbo.get('intention','?')[:60]} | "
                                     f"C={icbo.get('cognition','?')[:60]}")
                    parts.append("")

        return "\n".join(parts)

    def analyze_errors(
        self,
        report: EvaluationReport,
        max_errors: int = 10,
    ) -> Dict[str, Any]:
        """
        分析评估报告中的错误
        
        Args:
            report: 评估报告
            max_errors: 最大分析错误数
            
        Returns:
            错误分析结果
        """
        errors = report.errors[:max_errors]
        
        # 统计错误类型
        fp_count = sum(1 for e in errors if e["ground_truth"] == "adult")
        fn_count = sum(1 for e in errors if e["ground_truth"] == "minor")
        
        # 提取错误模式
        fp_examples = [e for e in errors if e["ground_truth"] == "adult"][:3]
        fn_examples = [e for e in errors if e["ground_truth"] == "minor"][:3]
        
        return {
            "total_errors": len(report.errors),
            "analyzed_errors": len(errors),
            "fp_count": fp_count,
            "fn_count": fn_count,
            "fp_examples": fp_examples,
            "fn_examples": fn_examples,
            "metrics": {
                "accuracy": report.metrics.accuracy,
                "precision": report.metrics.precision,
                "recall": report.metrics.recall,
                "f1": report.metrics.f1_score,
            },
        }
    
    def generate_optimization_prompt(
        self,
        current_skill: str,
        error_analysis: Dict[str, Any],
    ) -> str:
        """
        生成优化请求的 prompt
        
        Args:
            current_skill: 当前 skill.md 内容
            error_analysis: 错误分析结果
            
        Returns:
            优化请求 prompt
        """
        prompt_parts = [
            "# 当前 Skill Prompt\n",
            "```markdown",
            current_skill,
            "```\n",
            "# 评估结果\n",
            f"- 准确率: {error_analysis['metrics']['accuracy']:.2%}",
            f"- 精确率: {error_analysis['metrics']['precision']:.2%}",
            f"- 召回率: {error_analysis['metrics']['recall']:.2%}",
            f"- F1 分数: {error_analysis['metrics']['f1']:.4f}",
            f"\n总错误数: {error_analysis['total_errors']}",
            f"- 误判为未成年 (FP): {error_analysis['fp_count']}",
            f"- 漏判未成年 (FN): {error_analysis['fn_count']}",
            "\n# 错误案例\n",
        ]
        
        # 添加 FP 案例
        if error_analysis["fp_examples"]:
            prompt_parts.append("## 误判案例 (成年人被判为未成年)")
            for i, ex in enumerate(error_analysis["fp_examples"], 1):
                prompt_parts.append(f"\n### 案例 {i}")
                prompt_parts.append(f"- 置信度: {ex['confidence']:.2f}")
                prompt_parts.append(f"- 推理: {ex['reasoning'][:300]}")
        
        # 添加 FN 案例
        if error_analysis["fn_examples"]:
            prompt_parts.append("\n## 漏判案例 (未成年被判为成年)")
            for i, ex in enumerate(error_analysis["fn_examples"], 1):
                prompt_parts.append(f"\n### 案例 {i}")
                prompt_parts.append(f"- 置信度: {ex['confidence']:.2f}")
                prompt_parts.append(f"- 推理: {ex['reasoning'][:300]}")
        
        prompt_parts.append("\n# 任务")
        prompt_parts.append("请根据以上错误分析与训练集对比参考，改进 skill.md。输出完整的改进后内容。")
        
        return "\n".join(prompt_parts)
    
    def optimize(
        self,
        report: EvaluationReport,
        current_version: str = "teen_detector_v1",
        new_version: Optional[str] = None,
        dry_run: bool = False,
    ) -> Dict[str, Any]:
        """
        执行优化
        
        Args:
            report: 评估报告
            current_version: 当前版本目录名
            new_version: 新版本目录名（自动生成如果不指定）
            dry_run: 是否只预览，不实际创建文件
            
        Returns:
            优化结果
        """
        print("[INFO] 开始 Skill 优化...")
        
        # 1. 读取当前 skill
        current_skill_dir = self.skills_dir / current_version
        current_skill_path = current_skill_dir / "skill.md"
        
        if not current_skill_path.exists():
            raise FileNotFoundError(f"当前 skill 不存在: {current_skill_path}")
        
        with open(current_skill_path, "r", encoding="utf-8") as f:
            current_skill = f.read()
        
        print(f"[INFO] 加载当前 skill: {current_version}")
        
        # 2. 分析错误
        error_analysis = self.analyze_errors(report)
        print(f"[INFO] 错误分析: FP={error_analysis['fp_count']}, FN={error_analysis['fn_count']}")
        
        if error_analysis["total_errors"] == 0:
            print("[OK] 没有错误，无需优化")
            return {
                "success": True,
                "message": "no errors to optimize",
                "current_version": current_version,
            }
        
        # 3. 从 Train Set 检索对比案例（归纳推理材料）
        contrastive_text = self._retrieve_contrastive_examples(error_analysis)
        if contrastive_text:
            print(f"[INFO] 从 Train Set 检索到对比案例 ({len(contrastive_text)} 字符)")

        # 4. 生成优化 prompt
        optimization_prompt = self.generate_optimization_prompt(current_skill, error_analysis)
        if contrastive_text:
            optimization_prompt += "\n\n" + contrastive_text
        
        # 5. 调用 LLM 优化
        print("[INFO] 调用 LLM 生成优化方案...")
        messages = [
            {"role": "system", "content": OPTIMIZER_SYSTEM_PROMPT},
            {"role": "user", "content": optimization_prompt},
        ]
        
        optimized_skill = self.llm_client.chat(messages, temperature=0.7)
        
        # 清理可能的 markdown 代码块
        if optimized_skill.startswith("```"):
            lines = optimized_skill.split("\n")
            # 移除首尾的 ``` 行
            if lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            optimized_skill = "\n".join(lines)
        
        print(f"[OK] 生成优化 skill ({len(optimized_skill)} 字符)")
        
        if dry_run:
            print("\n[Dry Run] 优化内容预览:")
            print("-" * 50)
            print(optimized_skill[:500] + "..." if len(optimized_skill) > 500 else optimized_skill)
            print("-" * 50)
            return {
                "success": True,
                "dry_run": True,
                "optimized_skill": optimized_skill,
            }
        
        # 6. 创建新版本目录
        if new_version is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            new_version = f"teen_detector_v2_{timestamp}"
        
        new_skill_dir = self.skills_dir / new_version
        new_skill_dir.mkdir(parents=True, exist_ok=True)
        
        # 复制 config.yaml
        old_config = current_skill_dir / "config.yaml"
        if old_config.exists():
            shutil.copy(old_config, new_skill_dir / "config.yaml")
        
        # 写入新 skill.md
        new_skill_path = new_skill_dir / "skill.md"
        with open(new_skill_path, "w", encoding="utf-8") as f:
            f.write(optimized_skill)
        
        print(f"[OK] 新版本已保存: {new_version}")
        
        # 7. 记录优化历史
        history_path = new_skill_dir / "optimization_history.json"
        history = {
            "parent_version": current_version,
            "optimization_time": datetime.now().isoformat(),
            "error_analysis": {
                "total_errors": error_analysis["total_errors"],
                "fp_count": error_analysis["fp_count"],
                "fn_count": error_analysis["fn_count"],
            },
            "parent_metrics": error_analysis["metrics"],
        }
        with open(history_path, "w", encoding="utf-8") as f:
            json.dump(history, f, ensure_ascii=False, indent=2)
        
        return {
            "success": True,
            "current_version": current_version,
            "new_version": new_version,
            "new_skill_path": str(new_skill_path),
            "error_analysis": error_analysis,
        }
    
    def rollback(self, version: str) -> bool:
        """
        回滚到指定版本（更新活跃版本指针）
        
        Args:
            version: 要回滚到的版本
            
        Returns:
            是否成功
        """
        version_dir = self.skills_dir / version
        if not version_dir.exists():
            print(f"[ERROR] 版本不存在: {version}")
            return False
        
        try:
            set_active_skill_version(version)
            print(f"[OK] 已激活 skill 版本: {version}")
            return True
        except FileNotFoundError:
            print(f"[ERROR] 版本缺少 skill.md: {version}")
            return False
    
    def list_versions(self) -> List[Dict[str, Any]]:
        """列出所有 skill 版本"""
        versions = []
        
        for path in self.skills_dir.iterdir():
            if path.is_dir() and path.name.startswith("teen_detector"):
                skill_md = path / "skill.md"
                history = path / "optimization_history.json"
                
                version_info = {
                    "name": path.name,
                    "has_skill": skill_md.exists(),
                    "parent": None,
                }
                
                if history.exists():
                    with open(history, "r", encoding="utf-8") as f:
                        h = json.load(f)
                        version_info["parent"] = h.get("parent_version")
                        version_info["optimization_time"] = h.get("optimization_time")
                
                versions.append(version_info)
        
        return sorted(versions, key=lambda x: x["name"])


def run_optimization_cycle(
    current_version: str = "teen_detector_v1",
    max_samples: Optional[int] = None,
    dry_run: bool = True,
    auto_rollback: bool = True,
    min_f1_improvement: float = 0.0,
    retriever: Any = None,
    baseline_report: Optional[EvaluationReport] = None,
) -> Dict[str, Any]:
    """
    运行一个完整的优化周期: 评估 -> 分析 -> 优化 -> 验证 -> 回滚（如果变差）
    
    Args:
        current_version: 当前版本
        max_samples: 最大评估样本数
        dry_run: 是否只预览
        auto_rollback: 是否启用自动回滚（新版本F1下降时删除新版本）
        min_f1_improvement: 最小 F1 改进阈值（低于此值视为没有改进）
        retriever: SemanticRetriever 实例（可选），传入则评估时带 RAG
        baseline_report: 已有的当前版本评估报告；若提供则直接复用，避免重复评估当前版本
        
    Returns:
        优化结果
    """
    from src.evolution.evaluator import SkillEvaluator
    from src.executor import ExecutorSkill
    from src.config import SKILLS_DIR
    import shutil
    
    print("=" * 60)
    print("[INFO] Skill 优化周期")
    print("=" * 60)
    
    # 1. 获取当前版本的评估结果
    skill_path = SKILLS_DIR / current_version / "skill.md"
    if not skill_path.exists():
        raise FileNotFoundError(f"当前 skill 不存在: {skill_path}")

    if baseline_report is None:
        executor = ExecutorSkill(skill_path=str(skill_path))
        print("\n[INFO] Step 1: 在验证集上评估当前版本...")
        evaluator = SkillEvaluator(executor=executor, skill_version=current_version, retriever=retriever)
        report = evaluator.evaluate(max_samples=max_samples, use_test_set=False)
        baseline_report_reused = False
    else:
        print("\n[INFO] Step 1: 复用上游提供的当前版本评估结果...")
        report = baseline_report
        baseline_report_reused = True

    baseline_f1 = report.metrics.f1_score
    print(f"   当前版本 F1: {baseline_f1:.4f}")
    
    # 3. 优化
    print("\n[INFO] Step 2: 生成优化方案...")
    optimizer = SkillOptimizer()
    result = optimizer.optimize(report, current_version, dry_run=dry_run)
    
    # dry_run 模式下直接返回
    if dry_run or not result.get("success") or "new_version" not in result:
        return result
    
    # 4. 在验证集上评估新版本
    new_version = result["new_version"]
    print(f"\n[INFO] Step 3: 在验证集上评估新版本 ({new_version})...")
    
    new_skill_path = SKILLS_DIR / new_version / "skill.md"
    new_executor = ExecutorSkill(skill_path=str(new_skill_path))
    new_evaluator = SkillEvaluator(executor=new_executor, skill_version=new_version, retriever=retriever)
    new_report = new_evaluator.evaluate(max_samples=max_samples, use_test_set=False)
    
    new_f1 = new_report.metrics.f1_score
    f1_delta = new_f1 - baseline_f1
    
    print(f"   新版本 F1: {new_f1:.4f}")
    print(f"   F1 变化: {f1_delta:+.4f}")
    
    result["baseline_f1"] = baseline_f1
    result["new_f1"] = new_f1
    result["f1_delta"] = f1_delta
    result["baseline_report_reused"] = baseline_report_reused
    result["_new_report"] = new_report
    
    # 5. 自动回滚检查
    if auto_rollback and f1_delta < min_f1_improvement:
        print(f"\n[WARN] Step 4: 自动回滚 - 新版本未能改进 F1 (delta={f1_delta:.4f} < {min_f1_improvement})")
        
        # 删除新版本目录
        new_version_dir = SKILLS_DIR / new_version
        if new_version_dir.exists():
            shutil.rmtree(new_version_dir)
            print(f"   [DEL] 已删除: {new_version}")
        
        result["rolled_back"] = True
        result["rollback_reason"] = f"F1 未改进 (delta={f1_delta:.4f})"
        return result
    
    set_active_skill_version(new_version)
    print(f"\n[OK] Step 4: 新版本已接受 - F1 改进 {f1_delta:+.4f}")
    result["rolled_back"] = False
    result["active_version"] = new_version

    return result


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Skill 优化器")
    parser.add_argument("--version", default="teen_detector_v1", help="当前版本")
    parser.add_argument("--max-samples", type=int, default=10, help="最大评估样本数")
    parser.add_argument("--run", action="store_true", help="实际执行优化（默认 dry-run）")
    parser.add_argument("--list", action="store_true", help="列出所有版本")
    
    args = parser.parse_args()
    
    if args.list:
        optimizer = SkillOptimizer()
        versions = optimizer.list_versions()
        print("[INFO] Skill 版本列表:")
        for v in versions:
            parent = f" (from {v['parent']})" if v.get('parent') else ""
            print(f"  - {v['name']}{parent}")
    else:
        result = run_optimization_cycle(
            current_version=args.version,
            max_samples=args.max_samples,
            dry_run=not args.run,
        )

# 模块说明：
# - 旧执行器，可直接调 LLM，也可转去跑 bundled skill pipeline。
# - 现在主要作为兼容层和调试工具保留。

"""
在线执行体 (Online Executor Skill)
负责接收对话输入，调用 LLM 进行 ICBO 分析，返回结构化判定结果
"""

from pathlib import Path
from typing import Optional, List, Dict, Any
import json
import os
import re
import subprocess
import sys
import tempfile

from src.config import get_active_skill_path
from src.models import (
    SkillOutput,
    ICBOFeatures,
    UserPersona,
    RiskLevel,
    AnalysisPayload,
    FormalSkillOutput,
    FormalDecision,
    FormalUserProfile,
    FormalEvidence,
    FormalTrend,
    formal_to_legacy_output,
    normalize_formal_skill_output,
)
from src.utils.llm_client import LLMClient


PIPELINE_SCRIPT_NAME = "run_minor_detection_pipeline.py"
PIPELINE_OBSERVABILITY_PREFIX = "[MINOR_PIPELINE_OBSERVABILITY]"


def _strip_pipeline_observability(text: str) -> str:
    raw_text = str(text or "")
    lines = [line for line in raw_text.splitlines() if not line.startswith(PIPELINE_OBSERVABILITY_PREFIX)]
    return "\n".join(lines).strip()


def _coerce_trend_trajectory(parsed: Dict[str, Any]) -> Dict[str, Any]:
    """兼容早期/漂移输出，把非法 trajectory 纠正为可消费结构。"""
    trend = parsed.get("trend")
    if not isinstance(trend, dict):
        parsed["trend"] = {"trajectory": [], "trend_summary": ""}
        return parsed

    trajectory = trend.get("trajectory")
    if not isinstance(trajectory, list):
        trend["trajectory"] = []
        return parsed

    normalized_items: List[Dict[str, Any]] = []
    for item in trajectory:
        if isinstance(item, dict):
            confidence = item.get("minor_confidence", 0.5)
            try:
                confidence_value = float(confidence)
            except Exception:
                confidence_value = 0.5
            normalized_items.append(
                {
                    "session_id": item.get("session_id"),
                    "session_time": item.get("session_time"),
                    "minor_confidence": max(0.0, min(1.0, confidence_value)),
                }
            )

    trend["trajectory"] = normalized_items
    return parsed


class ExecutorSkill:
    """
    青少年识别 Skill 执行器

    核心职责：
    1. 加载技能 markdown（优先 SKILL.md，兼容旧版 skill.md）
    2. 将用户对话转换为 LLM 输入
    3. 调用 LLM 获取 ICBO 分析结果
    4. 解析并返回结构化输出
    """

    def __init__(
        self,
        skill_path: Optional[str] = None,
        llm_client: Optional[LLMClient] = None,
        inference_temperature: float = 0.2,
    ):
        """
        初始化执行器

        Args:
            skill_path: 技能 markdown 文件路径，默认使用当前活跃版本
            llm_client: LLM 客户端实例，默认自动创建
            inference_temperature: 结构化推理温度，默认 0.2 以提升 JSON 稳定性
        """
        self.skill_path = skill_path or str(get_active_skill_path())
        self.llm_client = llm_client or LLMClient()
        self.inference_temperature = max(0.0, min(1.0, inference_temperature))

        # 加载 Skill Prompt
        self.system_prompt = self._load_skill_prompt()
        self.skill_dir = Path(self.skill_path).parent
        self.skill_stem = Path(self.skill_path).stem.lower()
        self.skill_dir_name = self.skill_dir.name.lower()
        self.skill_name = self._infer_skill_name()

    def _load_skill_prompt(self) -> str:
        """加载技能 markdown 作为 system prompt"""
        try:
            with open(self.skill_path, "r", encoding="utf-8") as f:
                return f.read()
        except FileNotFoundError:
            raise FileNotFoundError(
                f"Skill 文件未找到: {self.skill_path}\n"
                f"请确保对应目录中存在 SKILL.md 或 skill.md"
            )

    def _infer_skill_name(self) -> str:
        """优先从 frontmatter 读取 skill name，回退到目录名。"""
        lines = self.system_prompt.splitlines()
        if len(lines) >= 3 and lines[0].strip() == "---":
            for line in lines[1:20]:
                if line.strip() == "---":
                    break
                if line.lower().startswith("name:"):
                    return line.split(":", 1)[1].strip().strip('"').strip("'").lower()
        return self.skill_dir_name

    def run(
        self,
        conversation: List[Dict[str, str]],
        user_id: Optional[str] = None,
        context: Optional[str] = None,
    ) -> SkillOutput:
        """
        执行一次完整的青少年识别分析

        Args:
            conversation: 对话列表 [{"role": "user/assistant", "content": "..."}]
            user_id: 用户ID（用于后续记忆模块）
            context: 额外上下文（如 RAG 检索结果）

        Returns:
            SkillOutput: 结构化分析结果
        """
        payload = AnalysisPayload(
            mode="single_session",
            conversation=conversation,
        )
        if user_id:
            payload.meta.user_id = user_id
        if context:
            payload.context.prior_profile = {"summary": context}
        return self.run_payload(payload)

    def run_payload(self, payload: AnalysisPayload) -> SkillOutput:
        """Run a normalized analysis payload."""
        if self._use_pipeline_executor():
            return formal_to_legacy_output(self._run_pipeline_formal_payload(payload))

        messages = self._build_messages(payload)
        response_model = FormalSkillOutput if self._uses_formal_output() else SkillOutput
        result = self._chat_with_fallback(
            messages=messages,
            response_model=response_model,
            payload=payload,
        )
        if isinstance(result, FormalSkillOutput):
            return formal_to_legacy_output(result)
        return result

    def run_formal_payload(self, payload: AnalysisPayload) -> FormalSkillOutput:
        """Run a formal skill payload and return the formal JSON shape."""
        if not self._uses_formal_output():
            raise RuntimeError("Current skill does not expose formal output.")

        if self._use_pipeline_executor():
            return self._run_pipeline_formal_payload(payload)

        messages = self._build_messages(payload)
        result = self._chat_with_fallback(
            messages=messages,
            response_model=FormalSkillOutput,
            payload=payload,
        )
        if not isinstance(result, FormalSkillOutput):
            raise RuntimeError("Formal payload returned a non-formal result.")
        return result

    def get_llm_metrics(self) -> Dict[str, float]:
        """返回 LLM 结构化输出稳定性指标。"""
        if hasattr(self.llm_client, "get_structured_metrics"):
            return self.llm_client.get_structured_metrics()
        return {}

    def run_text(self, text: str, user_id: Optional[str] = None) -> SkillOutput:
        """
        简化接口：直接分析单条文本

        Args:
            text: 用户文本
            user_id: 用户ID

        Returns:
            SkillOutput: 分析结果
        """
        conversation = [{"role": "user", "content": text}]
        return self.run(conversation, user_id)

    def _uses_formal_output(self) -> bool:
        if self.skill_name == "minor-detection" or self.skill_dir_name == "minor-detection":
            return True

        skill_dir = Path(self.skill_path).parent
        return (skill_dir / "references" / "output-schema.md").exists()

    def _pipeline_script_path(self) -> Path:
        return self.skill_dir / "scripts" / PIPELINE_SCRIPT_NAME

    def _use_pipeline_executor(self) -> bool:
        return self._uses_formal_output() and self._pipeline_script_path().exists()

    def _run_pipeline_formal_payload(self, payload: AnalysisPayload) -> FormalSkillOutput:
        payload_dict = payload.model_dump() if hasattr(payload, "model_dump") else payload.dict()
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False, encoding="utf-8") as handle:
            json.dump(payload_dict, handle, ensure_ascii=False, indent=2)
            payload_path = Path(handle.name)

        env = dict(os.environ)
        env.setdefault("PYTHONIOENCODING", "utf-8")
        env.setdefault("PYTHONUTF8", "1")
        try:
            completed = subprocess.run(
                [sys.executable, str(self._pipeline_script_path()), "--payload-file", str(payload_path)],
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                check=False,
                env=env,
            )
        finally:
            payload_path.unlink(missing_ok=True)

        stdout = (completed.stdout or "").strip()
        stderr = (completed.stderr or "").strip()
        if completed.returncode != 0:
            detail = _strip_pipeline_observability(stderr) or _strip_pipeline_observability(stdout) or "pipeline execution failed"
            raise RuntimeError(f"Skill pipeline failed: {detail}")
        if not stdout:
            raise RuntimeError("Skill pipeline returned empty stdout")

        output_text = stdout.splitlines()[-1].strip()
        parsed = json.loads(output_text)
        parsed = _coerce_trend_trajectory(parsed if isinstance(parsed, dict) else {})
        return self._normalize_formal_output(FormalSkillOutput(**parsed), payload=payload)

    def _build_messages(self, payload: AnalysisPayload) -> List[Dict[str, str]]:
        user_content = self._format_payload(payload)
        return [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_content},
        ]

    def _format_payload(self, payload: AnalysisPayload) -> str:
        payload_dict = payload.model_dump() if hasattr(payload, "model_dump") else payload.dict()
        return (
            "下面给出标准化分析输入 analysis_payload。"
            "请严格依据输入内容完成未成年人识别分析，并只输出符合技能要求的 JSON。\n\n"
            f"{json.dumps(payload_dict, ensure_ascii=False, indent=2)}"
        )

    def _chat_with_fallback(
        self,
        *,
        messages: List[Dict[str, str]],
        response_model: Any,
        payload: Optional[AnalysisPayload] = None,
    ) -> Any:
        try:
            result = self.llm_client.chat(
                messages=messages,
                response_model=response_model,
                temperature=self.inference_temperature,
            )
            if isinstance(result, FormalSkillOutput):
                return self._normalize_formal_output(result, payload=payload)
            return result
        except Exception as e:
            return self._fallback_parse(
                messages,
                e,
                response_model=response_model,
                payload=payload,
            )

    def _normalize_formal_output(
        self,
        output: FormalSkillOutput,
        *,
        payload: Optional[AnalysisPayload] = None,
    ) -> FormalSkillOutput:
        normalized = output
        if self._needs_formal_language_repair(output):
            repaired = self._repair_formal_output_language(output)
            if repaired is not None:
                normalized = repaired

        raw_time_hint = ""
        time_features: Dict[str, Any] = {}
        conversation: List[Dict[str, Any]] = []
        if payload is not None:
            raw_time_hint = payload.context.raw_time_hint or ""
            time_features = payload.context.time_features or {}
            conversation = payload.conversation or []
            if not conversation and payload.sessions:
                for session in payload.sessions:
                    if hasattr(session, "conversation"):
                        conversation.extend(session.conversation or [])

        return normalize_formal_skill_output(
            normalized,
            raw_time_hint=raw_time_hint,
            time_features=time_features,
            conversation=conversation,
        )

    def _needs_formal_language_repair(self, output: FormalSkillOutput) -> bool:
        text_fields: List[str] = [
            output.user_profile.age_range,
            output.user_profile.education_stage,
            output.icbo_features.intention,
            output.icbo_features.cognition,
            output.icbo_features.behavior_style,
            output.icbo_features.opportunity_time,
            output.reasoning_summary,
            output.trend.trend_summary,
            output.recommended_next_step,
            *output.user_profile.identity_markers,
            *output.evidence.direct_evidence,
            *output.evidence.historical_evidence,
            *output.evidence.time_evidence,
            *output.evidence.conflicting_signals,
            *output.uncertainty_notes,
        ]
        english_only_count = 0
        for value in text_fields:
            text = str(value or "").strip()
            if not text:
                continue
            if re.search(r"[A-Za-z]", text) and not re.search(r"[\u4e00-\u9fff]", text):
                english_only_count += 1
        return english_only_count >= 2

    def _repair_formal_output_language(self, output: FormalSkillOutput) -> Optional[FormalSkillOutput]:
        try:
            output_dict = output.model_dump() if hasattr(output, "model_dump") else output.dict()
            messages = [
                {
                    "role": "system",
                    "content": (
                        "你是 formal minor-detection 输出规范化助手。"
                        "请把输入 JSON 中所有自然语言说明字段改写为简体中文，"
                        "同时严格保持 JSON 结构、字段名、布尔值、数值不变。"
                        "不要新增字段，不要删除字段。"
                        "`decision.confidence_band` 必须保持 low/medium/high。"
                        "`decision.risk_level` 必须保持 High/Medium/Low。"
                        "`recommended_next_step` 必须保持既有枚举值。"
                        "`evidence.retrieval_evidence` 中的 sample_id、score、来源前缀等检索标识尽量原样保留。"
                        "如果是 single_session 且没有趋势，就让 `trend.trend_summary` 为空字符串。"
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        "请规范化下面的 FormalSkillOutput JSON，只返回修复后的 JSON：\n\n"
                        f"{json.dumps(output_dict, ensure_ascii=False, indent=2)}"
                    ),
                },
            ]
            repaired = self.llm_client.chat(
                messages=messages,
                response_model=FormalSkillOutput,
                temperature=0.0,
            )
            if isinstance(repaired, FormalSkillOutput):
                return repaired
        except Exception:
            return None
        return None

    def _fallback_parse(
        self,
        messages: list,
        original_error: Exception,
        response_model: Any = SkillOutput,
        payload: Optional[AnalysisPayload] = None,
    ) -> Any:
        """
        当标准解析失败时的降级处理

        尝试更宽松的解析，或返回默认值
        """
        print(f"[WARN] 标准解析失败: {original_error}")
        print("尝试降级解析...")

        # 重新调用但不强制 JSON 格式
        try:
            raw_response = self.llm_client.chat(messages=messages, temperature=0.5)
            coerced = self.llm_client.coerce_structured_response(raw_response, response_model)
            if isinstance(coerced, FormalSkillOutput):
                return self._normalize_formal_output(coerced, payload=payload)
            return coerced
        except Exception as e:
            print(f"[WARN] 降级解析也失败: {e}")

        # 最终降级：返回保守的默认值
        if response_model is FormalSkillOutput:
            return self._normalize_formal_output(
                FormalSkillOutput(
                    decision=FormalDecision(
                        is_minor=False,
                        minor_confidence=0.5,
                        confidence_band="medium",
                        risk_level=RiskLevel.MEDIUM,
                    ),
                    user_profile=FormalUserProfile(
                        age_range="未明确",
                        education_stage="未明确",
                        identity_markers=[],
                    ),
                    icbo_features=ICBOFeatures(
                        intention="解析失败",
                        cognition="解析失败",
                        behavior_style="解析失败",
                        opportunity_time="未明确",
                    ),
                    evidence=FormalEvidence(),
                    reasoning_summary=f"LLM 响应解析失败: {original_error}",
                    trend=FormalTrend(trajectory=[], trend_summary=""),
                    uncertainty_notes=[],
                    recommended_next_step="collect_more_context",
                ),
                payload=payload,
            )
        return SkillOutput(
            is_minor=False,
            minor_confidence=0.5,
            risk_level=RiskLevel.MEDIUM,
            icbo_features=ICBOFeatures(
                intention="解析失败",
                cognition="解析失败",
                behavior_style="解析失败",
                opportunity_time="未明确提及",
            ),
            user_persona=UserPersona(
                age_range="未明确",
                education_stage="未明确",
            ),
            reasoning=f"LLM 响应解析失败: {original_error}",
            key_evidence=[],
        )


# === 便捷函数 ===

_executor_instance: Optional[ExecutorSkill] = None


def get_executor() -> ExecutorSkill:
    """获取全局执行器实例（单例）"""
    global _executor_instance
    if _executor_instance is None:
        _executor_instance = ExecutorSkill()
    return _executor_instance


def analyze_conversation(
    conversation: List[Dict[str, str]],
    user_id: Optional[str] = None,
) -> SkillOutput:
    """
    便捷函数：分析对话

    Args:
        conversation: 对话列表
        user_id: 用户ID

    Returns:
        SkillOutput: 分析结果
    """
    return get_executor().run(conversation, user_id)


def analyze_payload(payload: Dict[str, Any]) -> SkillOutput:
    """正式 payload 入口，供后续运行时系统逐步迁移。"""
    return get_executor().run_payload(AnalysisPayload(**payload))


def analyze_text(text: str, user_id: Optional[str] = None) -> SkillOutput:
    """
    便捷函数：分析单条文本

    Args:
        text: 用户文本
        user_id: 用户ID

    Returns:
        SkillOutput: 分析结果
    """
    return get_executor().run_text(text, user_id)


def analyze_with_rag(
    conversation: List[Dict[str, str]],
    retriever: Any = None,
    top_k: int = 3,
    user_id: Optional[str] = None,
) -> SkillOutput:
    """
    带 RAG 语义校准的分析
    
    Args:
        conversation: 对话列表
        retriever: SemanticRetriever 实例（可选）
        top_k: 检索的相似案例数
        user_id: 用户ID
        
    Returns:
        SkillOutput: 分析结果
    """
    context = None
    
    if retriever is not None:
        try:
            from src.config import RAG_THRESHOLD
            results = retriever.retrieve(conversation, top_k=top_k, threshold=RAG_THRESHOLD)
            if results:
                context = retriever.format_for_prompt(results)
                print(f"[RAG] 检索到 {len(results)} 个相似案例")
        except Exception as e:
            print(f"[WARN] RAG 检索失败: {e}")
    
    return get_executor().run(conversation, user_id, context=context)


def analyze_with_memory(
    conversation: List[Dict[str, str]],
    user_id: str,
    memory: Any = None,
    retriever: Any = None,
    top_k: int = 3,
    update_memory: bool = True,
) -> SkillOutput:
    """
    带记忆和 RAG 的完整分析
    
    Args:
        conversation: 对话列表
        user_id: 用户ID（必须）
        memory: UserMemory 实例（可选）
        retriever: SemanticRetriever 实例（可选）
        top_k: RAG 检索的相似案例数
        update_memory: 分析后是否更新用户记忆
        
    Returns:
        SkillOutput: 分析结果
    """
    context_parts = []
    
    # 1. 获取用户历史画像
    if memory is not None:
        try:
            profile = memory.get_profile(user_id)
            if profile is not None:
                context_parts.append(profile.to_context_string())
                print(f"[Memory] 加载用户画像 (会话数: {profile.total_sessions})")
        except Exception as e:
            print(f"[WARN] 记忆加载失败: {e}")
    
    # 2. RAG 检索
    if retriever is not None:
        try:
            from src.config import RAG_THRESHOLD
            results = retriever.retrieve(conversation, top_k=top_k, threshold=RAG_THRESHOLD)
            if results:
                context_parts.append(retriever.format_for_prompt(results))
                print(f"[RAG] 检索到 {len(results)} 个相似案例")
        except Exception as e:
            print(f"[WARN] RAG 检索失败: {e}")
    
    # 合并上下文
    context = "\n\n".join(context_parts) if context_parts else None
    
    # 3. 执行分析
    result = get_executor().run(conversation, user_id, context=context)
    
    # 4. 更新记忆
    if update_memory and memory is not None:
        try:
            updated_profile = memory.update_profile(user_id, result)
            print(f"[Memory] 更新用户画像 (新置信度: {updated_profile.minor_confidence:.2f})")
        except Exception as e:
            print(f"[WARN] 记忆更新失败: {e}")
    
    return result


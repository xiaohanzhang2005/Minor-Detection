"""
在线执行体 (Online Executor Skill)
负责接收对话输入，调用 LLM 进行 ICBO 分析，返回结构化判定结果

MVP 版本：仅包含 Skill Prompt + LLM 调用
后续版本将集成：Memory（长期记忆）+ RAG（参考案例校准）
"""

from typing import Optional, List, Dict, Any
import json

from src.config import get_active_skill_path
from src.models import SkillOutput, ICBOFeatures, UserPersona, RiskLevel
from src.utils.llm_client import LLMClient


class ExecutorSkill:
    """
    青少年识别 Skill 执行器

    核心职责：
    1. 加载 skill.md 中的 Prompt
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
            skill_path: skill.md 文件路径，默认使用当前活跃版本
            llm_client: LLM 客户端实例，默认自动创建
            inference_temperature: 结构化推理温度，默认 0.2 以提升 JSON 稳定性
        """
        self.skill_path = skill_path or str(get_active_skill_path())
        self.llm_client = llm_client or LLMClient()
        self.inference_temperature = max(0.0, min(1.0, inference_temperature))

        # 加载 Skill Prompt
        self.system_prompt = self._load_skill_prompt()

    def _load_skill_prompt(self) -> str:
        """加载 skill.md 作为 system prompt"""
        try:
            with open(self.skill_path, "r", encoding="utf-8") as f:
                return f.read()
        except FileNotFoundError:
            raise FileNotFoundError(
                f"Skill 文件未找到: {self.skill_path}\n"
                f"请确保 skills/teen_detector_v1/skill.md 存在"
            )

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
        # 构建用户输入
        user_content = self._format_conversation(conversation, context)

        # 构建消息
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_content},
        ]

        # 调用 LLM
        try:
            result = self.llm_client.chat(
                messages=messages,
                response_model=SkillOutput,
                temperature=self.inference_temperature,
            )
            return result
        except Exception as e:
            # 解析失败时尝试更宽松的解析
            return self._fallback_parse(messages, e)

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

    def _format_conversation(
        self,
        conversation: List[Dict[str, str]],
        context: Optional[str] = None,
    ) -> str:
        """
        将对话列表格式化为 LLM 输入

        Args:
            conversation: 对话列表
            context: 额外上下文

        Returns:
            格式化的输入字符串
        """
        parts = []

        # 添加额外上下文（如 RAG 检索结果、历史画像）
        if context:
            parts.append(f"# 参考信息\n{context}\n")

        # 格式化对话
        parts.append("# 待分析对话\n")
        for msg in conversation:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            role_label = "用户" if role == "user" else "AI助手"
            parts.append(f"[{role_label}]: {content}")

        parts.append("\n---\n请基于以上对话，按照 ICBO 框架进行分析，输出 JSON 格式结果。")

        return "\n".join(parts)

    def _fallback_parse(self, messages: list, original_error: Exception) -> SkillOutput:
        """
        当标准解析失败时的降级处理

        尝试更宽松的解析，或返回默认值
        """
        print(f"[WARN] 标准解析失败: {original_error}")
        print("尝试降级解析...")

        # 重新调用但不强制 JSON 格式
        try:
            raw_response = self.llm_client.chat(messages=messages, temperature=0.5)

            # 尝试从响应中提取 JSON
            import re
            json_match = re.search(r'\{[\s\S]*\}', raw_response)
            if json_match:
                data = json.loads(json_match.group())

                # 安全提取嵌套字段
                icbo_data = data.get("icbo_features", {})
                if not isinstance(icbo_data, dict):
                    icbo_data = {}
                
                persona_data = data.get("user_persona", {})
                if not isinstance(persona_data, dict):
                    persona_data = {}
                
                key_evidence = data.get("key_evidence", [])
                if not isinstance(key_evidence, list):
                    key_evidence = [key_evidence] if key_evidence else []

                # 手动构建 SkillOutput
                return SkillOutput(
                    is_minor=data.get("is_minor", True),
                    minor_confidence=float(data.get("minor_confidence", 0.5)),
                    risk_level=RiskLevel(data.get("risk_level", "Medium")),
                    icbo_features=ICBOFeatures(
                        intention=icbo_data.get("intention", "无法解析"),
                        cognition=icbo_data.get("cognition", "无法解析"),
                        behavior_style=icbo_data.get("behavior_style", "无法解析"),
                        opportunity_time=icbo_data.get("opportunity_time", "未知"),
                    ),
                    user_persona=UserPersona(
                        age=persona_data.get("age"),
                        age_range=persona_data.get("age_range", "未知"),
                        gender=persona_data.get("gender"),
                        education_stage=persona_data.get("education_stage", "未知"),
                        identity_markers=persona_data.get("identity_markers", []) if isinstance(persona_data.get("identity_markers"), list) else [],
                    ),
                    reasoning=data.get("reasoning", "解析异常，使用降级结果"),
                    key_evidence=key_evidence,
                )
        except Exception as e:
            print(f"[WARN] 降级解析也失败: {e}")

        # 最终降级：返回保守的默认值
        return SkillOutput(
            is_minor=True,  # 保守判定
            minor_confidence=0.5,
            risk_level=RiskLevel.MEDIUM,
            icbo_features=ICBOFeatures(
                intention="解析失败",
                cognition="解析失败",
                behavior_style="解析失败",
                opportunity_time="未知",
            ),
            user_persona=UserPersona(
                age_range="未知",
                education_stage="未知",
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


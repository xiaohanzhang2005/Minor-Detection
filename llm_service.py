"""
LLM 服务模块
封装与 Gemini API 的交互（通过 AiHubMix），提供结构化输出分析功能
"""

import os
import json
import time
from typing import Optional, TypeVar, Union

from openai import OpenAI
from openai._exceptions import APIError, RateLimitError, APIConnectionError

from models import IntentType, KnowledgeAnalysis, SocialAnalysis
from qa_knowledge_calibrator import calibrate_knowledge_analysis
from social_chat_calibrator import calibrate_social_analysis

# 类型变量用于泛型函数
T = TypeVar('T', bound=Union[IntentType, KnowledgeAnalysis, SocialAnalysis])


class LLMService:
    """
    Gemini LLM 服务封装类（通过 AiHubMix）
    提供意图分类和分析功能，支持结构化输出
    """

    def __init__(self, api_key: Optional[str] = None, model: str = "gemini-3-flash-preview"):
        """
        初始化 LLM 服务

        Args:
            api_key: AiHubMix API 密钥，如果为 None 则从环境变量获取
            model: 使用的模型名称，默认为 gemini-3-flash-preview
        """
        # 不再在代码中硬编码真实密钥，统一从环境变量中获取；
        # 如果你在本地调试，可以在系统环境中设置 AIHUBMIX_API_KEY。
        self.api_key = api_key or os.getenv("AIHUBMIX_API_KEY")
        if not self.api_key:
            raise ValueError(
                "AiHubMix API key is required. 请在环境变量 AIHUBMIX_API_KEY 中配置（示例：your-aihublmix-api-key-here）。"
            )

        self.client = OpenAI(
            api_key=self.api_key,
            base_url="https://aihubmix.com/v1"
        )
        self.model = model
        self.max_retries = 3
        self.retry_delay = 1.0

    def _call_with_retry(self, messages: list, response_format: type[T]) -> T:
        """
        带重试机制的 API 调用

        Args:
            messages: 消息列表
            response_format: 响应格式类（Pydantic模型）

        Returns:
            结构化响应结果

        Raises:
            Exception: 当所有重试都失败时抛出异常
        """
        for attempt in range(self.max_retries):
            try:
                # 使用普通chat completions，因为gemini不支持beta.chat.completions.parse
                completion = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    response_format={"type": "json_object"}  # 请求JSON格式响应
                )

                # 解析JSON响应
                response_text = completion.choices[0].message.content
                if not response_text:
                    raise Exception("Empty response from API")

                # 尝试解析JSON
                try:
                    json_data = json.loads(response_text)
                except json.JSONDecodeError as e:
                    raise Exception(f"Failed to parse JSON response: {e}")

                # 根据响应格式类型进行不同的处理
                if response_format == IntentType:
                    # 对于IntentType，返回对应的枚举值
                    intent_str = json_data.get("intent", "").lower()
                    if "knowledge" in intent_str or "qa" in intent_str:
                        return IntentType.KNOWLEDGE_QA
                    elif "social" in intent_str or "chat" in intent_str:
                        return IntentType.SOCIAL_CHAT
                    else:
                        # 默认返回社交聊天
                        return IntentType.SOCIAL_CHAT
                else:
                    # 对于其他Pydantic模型，直接解析
                    return response_format(**json_data)

            except RateLimitError as e:
                if attempt == self.max_retries - 1:
                    raise Exception(f"Rate limit exceeded after {self.max_retries} attempts: {e}")
                wait_time = self.retry_delay * (2 ** attempt)  # 指数退避
                print(f"Rate limit hit, retrying in {wait_time} seconds...")
                time.sleep(wait_time)

            except (APIError, APIConnectionError) as e:
                if attempt == self.max_retries - 1:
                    raise Exception(f"API error after {self.max_retries} attempts: {e}")
                print(f"API error, retrying in {self.retry_delay} seconds...")
                time.sleep(self.retry_delay)

            except Exception as e:
                if attempt == self.max_retries - 1:
                    raise Exception(f"Unexpected error after {self.max_retries} attempts: {e}")
                print(f"Unexpected error, retrying in {self.retry_delay} seconds...")
                time.sleep(self.retry_delay)

    def classify_intent(self, text: str) -> IntentType:
        """
        分类用户意图

        Args:
            text: 用户输入文本

        Returns:
            IntentType: 用户意图类型
        """
        system_prompt = """你是一个意图分类器。判断用户对话是 'knowledge_qa' (事实问答/求知) 还是 'social_chat' (闲聊/情感/求助)。

请根据对话内容的特点进行判断：
- knowledge_qa: 询问事实、概念、原理、方法、数据等客观知识，讨论学习内容
- social_chat: 表达情感、闲聊、寻求建议、倾诉烦恼、社交互动、人际关系

请以JSON格式返回结果：
{"intent": "knowledge_qa"} 或 {"intent": "social_chat"}

不要添加额外解释。"""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": text}
        ]

        return self._call_with_retry(messages, IntentType)

    def analyze_knowledge(self, text: str) -> KnowledgeAnalysis:
        """
        分析用户问题的知识水平

        Args:
            text: 用户输入文本

        Returns:
            KnowledgeAnalysis: 知识分析结果
        """
        system_prompt = """你是一个教育专家。请分析用户对话中的认知深度和逻辑思维，判断其属于哪个教育阶段。

教育阶段分类：
- 小学：基础算术、简单科学、基本常识，对话逻辑简单直接，表达较为幼稚
- 初中：代数、几何、物理化学基础、文学基础，对话有一定逻辑但不够深入，表达较为青涩
- 高中：微积分、三角函数、力学、化学反应、古典文学，对话逻辑较严谨，表达较为成熟
- 大学：高等数学、抽象代数、量子物理、复杂工程问题，对话逻辑严谨深入，表达专业
- 成人专业：行业专有名词、最新研究、复杂商业逻辑、专业工具使用，对话逻辑成熟专业

分析要点：
1. 认知深度：问题的复杂程度和概念理解
2. 逻辑思维：对话是否连贯、是否有逻辑推理、回答是否切题
3. 表达能力：语言组织是否清晰、用词是否准确、前言是否搭后语
4. 学习场景：是否为作业题或学习相关内容

请以JSON格式返回结果：
{
  "predicted_education_level": "小学|初中|高中|大学|成人专业",
  "is_homework": true|false,
  "reasoning": "详细的分析推理过程，包括对话逻辑性和连贯性评估"
}"""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": text}
        ]

        analysis = self._call_with_retry(messages, KnowledgeAnalysis)

        # 在线辅助判定：结合本地知识问答数据库做温和校准
        try:
            analysis = calibrate_knowledge_analysis(analysis, text)
        except Exception as e:
            # 校准失败不影响主流程，仅记录信息
            print(f"⚠️ 知识库校准失败，使用原始分析结果: {e}")

        return analysis

    def analyze_social(self, text: str) -> SocialAnalysis:
        """
        分析用户的情感和社会特征

        Args:
            text: 用户输入文本

        Returns:
            SocialAnalysis: 社交分析结果
        """
        system_prompt = """你是一个心理专家。请分析用户对话的情绪波动、逻辑性和语言风格，特别关注对话的连贯性和思维逻辑。

分析维度：
1. 情绪波动 (emotional_score): 0.0-1.0，评估情绪表达的激烈程度
   - 低分：平静、理性、情绪稳定
   - 高分：激动、焦虑、强烈情感、前言不搭后语

2. 逻辑性 (logic_score): 0.0-1.0，评估表达的条理性
   - 低分：跳跃性思维、缺乏逻辑连接、对AI回复理解不充分、回答偏题
   - 高分：结构清晰、有条理、对AI问题回答切题、思维连贯

3. 未成年人倾向 (minor_tendency_score): 0.0-1.0，基于以下特征评估：
   - 未成年人特征：情绪极端化、逻辑简单、使用校园/网络黑话、易受暗示、表达不成熟、对复杂问题理解困难
   - 成年人特征：情绪稳定、逻辑严谨、专业术语、理性分析、成熟表达、思维深入

重点分析对话特征：
- 是否能理解AI的问题并做出相关回答
- 回答是否逻辑连贯、前后一致
- 是否使用年龄特征明显的语言和思维方式
- 对复杂概念的理解和表达能力

请以JSON格式返回结果：
{
  "emotional_score": 0.0-1.0,
  "logic_score": 0.0-1.0,
  "minor_tendency_score": 0.0-1.0,
  "reasoning": "详细的分析推理过程，包括对话连贯性、逻辑思维和年龄特征评估"
}"""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": text}
        ]

        analysis = self._call_with_retry(messages, SocialAnalysis)

        # 在线辅助判定：结合本地社交对话数据库做温和校准
        try:
            analysis = calibrate_social_analysis(analysis, text)
        except Exception as e:
            # 校准失败不影响主流程，仅记录信息
            print(f"⚠️ 社交库校准失败，使用原始分析结果: {e}")

        return analysis


# 全局服务实例（可选使用）
_default_service: Optional[LLMService] = None


def get_llm_service(api_key: Optional[str] = None, model: str = "gemini-3-flash-preview") -> LLMService:
    """
    获取 LLM 服务实例（单例模式）

    Args:
        api_key: AiHubMix API 密钥
        model: 模型名称

    Returns:
        LLMService: 服务实例
    """
    global _default_service
    if _default_service is None or (api_key and api_key != _default_service.api_key):
        _default_service = LLMService(api_key, model)
    return _default_service

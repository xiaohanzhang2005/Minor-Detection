"""
LLM 客户端封装
提供统一的 API 调用接口，支持重试和错误处理
"""

import json
import time
from typing import Optional, Type, TypeVar, Dict, Any

from openai import OpenAI
from openai._exceptions import APIError, RateLimitError, APIConnectionError
from pydantic import BaseModel, ValidationError

from src.config import (
    API_KEY,
    API_BASE_URL,
    EXECUTOR_MODEL,
    LLM_MAX_RETRIES,
    LLM_RETRY_DELAY,
    validate_config,
)

T = TypeVar('T', bound=BaseModel)


class LLMClient:
    """
    LLM API 客户端
    封装 OpenAI 兼容接口，提供重试机制和结构化输出解析
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
    ):
        """
        初始化 LLM 客户端

        Args:
            api_key: API 密钥，默认从环境变量获取
            base_url: API 基础 URL
            model: 使用的模型名称
        """
        validate_config()

        self.api_key = api_key or API_KEY
        self.base_url = base_url or API_BASE_URL
        self.model = model or EXECUTOR_MODEL

        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url
        )

        self.max_retries = LLM_MAX_RETRIES
        self.retry_delay = LLM_RETRY_DELAY

        # 结构化输出稳定性指标（用于验收门禁）
        self._metrics: Dict[str, int] = {
            "structured_calls": 0,
            "structured_validation_failures": 0,
            "structured_local_repairs": 0,
            "structured_unrecovered_failures": 0,
            "api_retries": 0,
        }

    def chat(
        self,
        messages: list,
        response_model: Optional[Type[T]] = None,
        temperature: float = 0.7,
    ) -> T | str:
        """
        发送聊天请求

        Args:
            messages: 消息列表 [{"role": "system", "content": "..."}, ...]
            response_model: Pydantic 模型类，用于解析 JSON 响应
            temperature: 温度参数

        Returns:
            解析后的 Pydantic 对象或原始字符串
        """
        structured_mode = response_model is not None
        if structured_mode:
            self._metrics["structured_calls"] += 1

        for attempt in range(self.max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=temperature,
                    response_format={"type": "json_object"} if response_model else None,
                )

                content = response.choices[0].message.content
                if not content:
                    raise ValueError("Empty response from API")

                # 如果指定了响应模型，解析 JSON
                if response_model:
                    try:
                        data = json.loads(content)
                    except json.JSONDecodeError as e:
                        if attempt == self.max_retries - 1:
                            self._metrics["structured_unrecovered_failures"] += 1
                            raise RuntimeError(f"Failed to parse JSON after {self.max_retries} attempts: {e}")
                        self._metrics["api_retries"] += 1
                        print(f"⚠️ JSON 解析失败，重试中: {e}")
                        time.sleep(self.retry_delay)
                        continue

                    # 先做轻量清洗
                    data = self._sanitize_response_data(data, response_model)

                    # 先本地修复，再考虑重试 API
                    try:
                        return response_model(**data)
                    except ValidationError as e:
                        self._metrics["structured_validation_failures"] += 1

                        fixed_data = self._repair_response_data(data, response_model)
                        if fixed_data is not None:
                            try:
                                result = response_model(**fixed_data)
                                self._metrics["structured_local_repairs"] += 1
                                return result
                            except ValidationError:
                                pass

                        if attempt == self.max_retries - 1:
                            self._metrics["structured_unrecovered_failures"] += 1
                            raise RuntimeError(
                                f"Structured validation failed after {self.max_retries} attempts: {e}"
                            )

                        self._metrics["api_retries"] += 1
                        print(f"⚠️ Structured 校验失败，本地修复未成功，重试中: {e}")
                        time.sleep(self.retry_delay)
                        continue

                return content

            except RateLimitError as e:
                if attempt == self.max_retries - 1:
                    if structured_mode:
                        self._metrics["structured_unrecovered_failures"] += 1
                    raise RuntimeError(f"Rate limit exceeded after {self.max_retries} attempts: {e}")
                self._metrics["api_retries"] += 1
                wait_time = self.retry_delay * (2 ** attempt)
                print(f"⚠️ Rate limit hit, retrying in {wait_time}s...")
                time.sleep(wait_time)

            except (APIError, APIConnectionError) as e:
                if attempt == self.max_retries - 1:
                    if structured_mode:
                        self._metrics["structured_unrecovered_failures"] += 1
                    raise RuntimeError(f"API error after {self.max_retries} attempts: {e}")
                self._metrics["api_retries"] += 1
                print(f"⚠️ API error, retrying in {self.retry_delay}s...")
                time.sleep(self.retry_delay)

            except Exception as e:
                if attempt == self.max_retries - 1:
                    if structured_mode:
                        self._metrics["structured_unrecovered_failures"] += 1
                    raise RuntimeError(f"Unexpected error: {e}")
                self._metrics["api_retries"] += 1
                print(f"⚠️ Unexpected error, retrying: {e}")
                time.sleep(self.retry_delay)

    def _sanitize_response_data(self, data: dict, response_model: Type[T]) -> dict:
        """
        清洗 LLM 返回的数据，确保嵌套结构类型正确
        
        处理常见问题：
        - 嵌套对象返回为 list 而非 dict
        - 缺失必填字段
        - 类型不匹配
        """
        if not isinstance(data, dict):
            return {}
        
        # 针对 SkillOutput 的特殊处理
        model_name = response_model.__name__ if hasattr(response_model, '__name__') else ''
        
        if model_name == 'SkillOutput':
            # 确保 icbo_features 是 dict
            if 'icbo_features' in data:
                icbo = data['icbo_features']
                if isinstance(icbo, list) and len(icbo) > 0:
                    data['icbo_features'] = icbo[0] if isinstance(icbo[0], dict) else {}
                elif not isinstance(icbo, dict):
                    data['icbo_features'] = {
                        "intention": "解析异常",
                        "cognition": "解析异常",
                        "behavior_style": "解析异常",
                        "opportunity_time": "未知",
                    }
            
            # 确保 user_persona 是 dict
            if 'user_persona' in data:
                persona = data['user_persona']
                if isinstance(persona, list) and len(persona) > 0:
                    data['user_persona'] = persona[0] if isinstance(persona[0], dict) else {}
                elif not isinstance(persona, dict):
                    data['user_persona'] = {
                        "age_range": "未知",
                        "education_stage": "未知",
                    }
            
            # 确保 key_evidence 是 list
            if 'key_evidence' in data and not isinstance(data['key_evidence'], list):
                if isinstance(data['key_evidence'], str):
                    data['key_evidence'] = [data['key_evidence']]
                else:
                    data['key_evidence'] = []
            
            # 确保 identity_markers 在 user_persona 中是 list
            if 'user_persona' in data and isinstance(data['user_persona'], dict):
                markers = data['user_persona'].get('identity_markers')
                if markers is not None and not isinstance(markers, list):
                    if isinstance(markers, str):
                        data['user_persona']['identity_markers'] = [markers]
                    else:
                        data['user_persona']['identity_markers'] = []
        
        return data

    def _repair_response_data(self, data: Dict[str, Any], response_model: Type[T]) -> Optional[Dict[str, Any]]:
        """
        对结构化输出做本地补全修复，尽量避免直接重试 API。
        目前主要针对 SkillOutput。
        """
        model_name = response_model.__name__ if hasattr(response_model, "__name__") else ""
        if model_name != "SkillOutput":
            return None

        repaired: Dict[str, Any] = dict(data) if isinstance(data, dict) else {}

        # 顶层字段兜底
        repaired.setdefault("is_minor", True)
        repaired.setdefault("minor_confidence", 0.5)
        repaired.setdefault("risk_level", "Medium")
        repaired.setdefault("reasoning", "结构化输出缺失字段，已本地修复")
        repaired.setdefault("key_evidence", [])

        # 类型修复
        repaired["is_minor"] = bool(repaired.get("is_minor", True))
        try:
            repaired["minor_confidence"] = float(repaired.get("minor_confidence", 0.5))
        except Exception:
            repaired["minor_confidence"] = 0.5
        repaired["minor_confidence"] = max(0.0, min(1.0, repaired["minor_confidence"]))

        # icbo_features 补全
        icbo = repaired.get("icbo_features")
        if not isinstance(icbo, dict):
            icbo = {}
        icbo.setdefault("intention", "解析异常")
        icbo.setdefault("cognition", "解析异常")
        icbo.setdefault("behavior_style", "解析异常")
        icbo.setdefault("opportunity_time", "未知")
        repaired["icbo_features"] = icbo

        # user_persona 补全
        persona = repaired.get("user_persona")
        if not isinstance(persona, dict):
            persona = {}
        persona.setdefault("age", None)
        persona.setdefault("age_range", "未知")
        persona.setdefault("gender", None)
        persona.setdefault("education_stage", "未知")
        markers = persona.get("identity_markers", [])
        if not isinstance(markers, list):
            markers = [markers] if isinstance(markers, str) else []
        persona["identity_markers"] = markers
        repaired["user_persona"] = persona

        # key_evidence 类型修复
        evidence = repaired.get("key_evidence", [])
        if not isinstance(evidence, list):
            evidence = [evidence] if isinstance(evidence, str) else []
        repaired["key_evidence"] = evidence

        return repaired

    def get_structured_metrics(self) -> Dict[str, float]:
        """返回结构化输出稳定性指标（含每百次失败率）。"""
        calls = self._metrics["structured_calls"]
        validation_failures = self._metrics["structured_validation_failures"]
        unrecovered = self._metrics["structured_unrecovered_failures"]

        validation_rate = (validation_failures / calls * 100.0) if calls > 0 else 0.0
        unrecovered_rate = (unrecovered / calls * 100.0) if calls > 0 else 0.0

        return {
            **self._metrics,
            "validation_failure_rate_per_100": round(validation_rate, 4),
            "unrecovered_failure_rate_per_100": round(unrecovered_rate, 4),
        }

    def chat_raw(self, system_prompt: str, user_content: str, **kwargs) -> str:
        """
        简化的聊天接口

        Args:
            system_prompt: 系统提示词
            user_content: 用户输入
            **kwargs: 传递给 chat() 的额外参数

        Returns:
            响应内容字符串
        """
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ]
        return self.chat(messages, **kwargs)

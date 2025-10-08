"""
LLM集成服务

提供统一的LLM客户端接口，支持OpenAI、Anthropic等多种LLM提供商。
"""

import logging
import asyncio
import json
from typing import Dict, Any, List, Optional, Union, Callable
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
import time

from .config import settings
from .exceptions import LLMServiceError, ExternalServiceError
from .logging import BusinessLogger

logger = logging.getLogger(__name__)
business_logger = BusinessLogger("llm_integration")


class LLMProvider(Enum):
    """LLM提供商"""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GOOGLE = "google"
    LOCAL = "local"


class LLMModel(Enum):
    """LLM模型"""
    # OpenAI模型
    GPT_3_5_TURBO = "gpt-3.5-turbo"
    GPT_4 = "gpt-4"
    GPT_4_TURBO = "gpt-4-turbo"
    GPT_4O = "gpt-4o"

    # Anthropic模型
    CLAUDE_3_HAIKU = "claude-3-haiku-20240307"
    CLAUDE_3_SONNET = "claude-3-sonnet-20240229"
    CLAUDE_3_OPUS = "claude-3-opus-20240229"

    # Google模型
    GEMINI_PRO = "gemini-pro"


@dataclass
class LLMRequest:
    """LLM请求"""
    prompt: str
    model: Union[str, LLMModel]
    provider: Optional[LLMProvider] = None
    temperature: float = 0.7
    max_tokens: Optional[int] = None
    top_p: Optional[float] = None
    frequency_penalty: Optional[float] = None
    presence_penalty: Optional[float] = None
    stop_sequences: Optional[List[str]] = None
    system_prompt: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class LLMResponse:
    """LLM响应"""
    content: str
    model: str
    provider: str
    tokens_used: Optional[int] = None
    cost_usd: Optional[float] = None
    response_time_ms: Optional[int] = None
    metadata: Optional[Dict[str, Any]] = None


class BaseLLMClient(ABC):
    """LLM客户端基类"""

    def __init__(self, provider: LLMProvider):
        self.provider = provider
        self.logger = logger
        self.business_logger = business_logger

    @abstractmethod
    async def generate_completion(self, request: LLMRequest) -> LLMResponse:
        """生成文本完成"""
        pass

    @abstractmethod
    def calculate_cost(self, model: str, tokens: int) -> float:
        """计算API调用成本"""
        pass

    @abstractmethod
    def validate_request(self, request: LLMRequest) -> bool:
        """验证请求参数"""
        pass


class OpenAIClient(BaseLLMClient):
    """OpenAI客户端"""

    def __init__(self):
        super().__init__(LLMProvider.OPENAI)
        self.client = None
        self._initialize_client()

    def _initialize_client(self):
        """初始化OpenAI客户端"""
        try:
            import openai
            self.client = openai.AsyncOpenAI(
                api_key=getattr(settings, 'OPENAI_API_KEY', None),
                timeout=60,
                max_retries=3
            )
        except ImportError as e:
            raise LLMServiceError(
                message="OpenAI package not installed. Install with: pip install openai",
                error_code="OPENAI_NOT_INSTALLED",
                cause=e
            )

    async def generate_completion(self, request: LLMRequest) -> LLMResponse:
        """生成文本完成"""
        if not self.validate_request(request):
            raise LLMServiceError(
                message="Invalid request parameters",
                error_code="INVALID_REQUEST"
            )

        start_time = time.time()

        try:
            # 构建消息
            messages = []
            if request.system_prompt:
                messages.append({"role": "system", "content": request.system_prompt})
            messages.append({"role": "user", "content": request.prompt})

            # 调用API
            response = await self.client.chat.completions.create(
                model=request.model.value if isinstance(request.model, LLMModel) else request.model,
                messages=messages,
                temperature=request.temperature,
                max_tokens=request.max_tokens,
                top_p=request.top_p,
                frequency_penalty=request.frequency_penalty,
                presence_penalty=request.presence_penalty,
                stop=request.stop_sequences
            )

            # 计算响应时间
            response_time = int((time.time() - start_time) * 1000)

            # 提取内容
            content = response.choices[0].message.content
            tokens_used = response.usage.total_tokens if response.usage else None

            # 计算成本
            cost = self.calculate_cost(
                response.model,
                tokens_used or 0
            )

            # 记录业务日志
            self.business_logger.log_system_event(
                event_type="llm_api_call",
                severity="info",
                message=f"OpenAI API调用成功: {request.model}",
                details={
                    "provider": self.provider.value,
                    "model": response.model,
                    "tokens_used": tokens_used,
                    "cost_usd": cost,
                    "response_time_ms": response_time
                }
            )

            return LLMResponse(
                content=content,
                model=response.model,
                provider=self.provider.value,
                tokens_used=tokens_used,
                cost_usd=cost,
                response_time_ms=response_time,
                metadata={
                    "finish_reason": response.choices[0].finish_reason,
                    "usage": response.usage.model_dump() if response.usage else None
                }
            )

        except Exception as e:
            self.logger.error(f"OpenAI API调用失败: {e}")
            self.business_logger.log_system_event(
                event_type="llm_api_error",
                severity="error",
                message=f"OpenAI API调用失败: {str(e)}",
                details={
                    "provider": self.provider.value,
                    "model": request.model,
                    "error": str(e)
                }
            )
            raise LLMServiceError(
                message=f"OpenAI API调用失败: {str(e)}",
                error_code="OPENAI_API_ERROR",
                cause=e
            )

    def calculate_cost(self, model: str, tokens: int) -> float:
        """计算OpenAI API成本"""
        # 成本表（每1K tokens的价格，美元）
        pricing = {
            "gpt-3.5-turbo": {"input": 0.0005, "output": 0.0015},
            "gpt-4": {"input": 0.03, "output": 0.06},
            "gpt-4-turbo": {"input": 0.01, "output": 0.03},
            "gpt-4o": {"input": 0.005, "output": 0.015}
        }

        if model in pricing:
            # 假设输入输出token各占一半
            input_tokens = tokens // 2
            output_tokens = tokens // 2
            cost = (input_tokens * pricing[model]["input"] +
                   output_tokens * pricing[model]["output"]) / 1000
            return round(cost, 6)

        return 0.0

    def validate_request(self, request: LLMRequest) -> bool:
        """验证请求参数"""
        if not request.prompt or len(request.prompt.strip()) == 0:
            return False

        if request.temperature < 0 or request.temperature > 2:
            return False

        if request.max_tokens and request.max_tokens <= 0:
            return False

        return True


class AnthropicClient(BaseLLMClient):
    """Anthropic客户端"""

    def __init__(self):
        super().__init__(LLMProvider.ANTHROPIC)
        self.client = None
        self._initialize_client()

    def _initialize_client(self):
        """初始化Anthropic客户端"""
        try:
            import anthropic
            self.client = anthropic.AsyncAnthropic(
                api_key=getattr(settings, 'ANTHROPIC_API_KEY', None),
                timeout=60
            )
        except ImportError as e:
            raise LLMServiceError(
                message="Anthropic package not installed. Install with: pip install anthropic",
                error_code="ANTHROPIC_NOT_INSTALLED",
                cause=e
            )

    async def generate_completion(self, request: LLMRequest) -> LLMResponse:
        """生成文本完成"""
        if not self.validate_request(request):
            raise LLMServiceError(
                message="Invalid request parameters",
                error_code="INVALID_REQUEST"
            )

        start_time = time.time()

        try:
            # 构建消息
            messages = [{"role": "user", "content": request.prompt}]

            # 调用API
            response = await self.client.messages.create(
                model=request.model.value if isinstance(request.model, LLMModel) else request.model,
                messages=messages,
                temperature=request.temperature,
                max_tokens=request.max_tokens or 1000,
                top_p=request.top_p,
                stop_sequences=request.stop_sequences or [],
                system=request.system_prompt
            )

            # 计算响应时间
            response_time = int((time.time() - start_time) * 1000)

            # 提取内容
            content = response.content[0].text if response.content else ""
            tokens_used = response.usage.input_tokens + response.usage.output_tokens

            # 计算成本
            cost = self.calculate_cost(
                response.model,
                tokens_used
            )

            # 记录业务日志
            self.business_logger.log_system_event(
                event_type="llm_api_call",
                severity="info",
                message=f"Anthropic API调用成功: {request.model}",
                details={
                    "provider": self.provider.value,
                    "model": response.model,
                    "tokens_used": tokens_used,
                    "cost_usd": cost,
                    "response_time_ms": response_time
                }
            )

            return LLMResponse(
                content=content,
                model=response.model,
                provider=self.provider.value,
                tokens_used=tokens_used,
                cost_usd=cost,
                response_time_ms=response_time,
                metadata={
                    "stop_reason": response.stop_reason,
                    "usage": response.usage.model_dump()
                }
            )

        except Exception as e:
            self.logger.error(f"Anthropic API调用失败: {e}")
            self.business_logger.log_system_event(
                event_type="llm_api_error",
                severity="error",
                message=f"Anthropic API调用失败: {str(e)}",
                details={
                    "provider": self.provider.value,
                    "model": request.model,
                    "error": str(e)
                }
            )
            raise LLMServiceError(
                message=f"Anthropic API调用失败: {str(e)}",
                error_code="ANTHROPIC_API_ERROR",
                cause=e
            )

    def calculate_cost(self, model: str, tokens: int) -> float:
        """计算Anthropic API成本"""
        # 成本表（每1K tokens的价格，美元）
        pricing = {
            "claude-3-haiku-20240307": {"input": 0.00025, "output": 0.00125},
            "claude-3-sonnet-20240229": {"input": 0.003, "output": 0.015},
            "claude-3-opus-20240229": {"input": 0.015, "output": 0.075}
        }

        if model in pricing:
            input_tokens = tokens // 2
            output_tokens = tokens // 2
            cost = (input_tokens * pricing[model]["input"] +
                   output_tokens * pricing[model]["output"]) / 1000
            return round(cost, 6)

        return 0.0

    def validate_request(self, request: LLMRequest) -> bool:
        """验证请求参数"""
        if not request.prompt or len(request.prompt.strip()) == 0:
            return False

        if request.temperature < 0 or request.temperature > 1:
            return False

        if request.max_tokens and request.max_tokens <= 0:
            return False

        return True


class LLMIntegrationService:
    """LLM集成服务"""

    def __init__(self):
        self.logger = logger
        self.business_logger = business_logger
        self.clients = {}
        self.default_provider = LLMProvider.OPENAI
        self.default_model = LLMModel.GPT_3_5_TURBO

        # 初始化客户端
        self._initialize_clients()

    def _initialize_clients(self):
        """初始化LLM客户端"""
        # 初始化OpenAI客户端
        try:
            self.clients[LLMProvider.OPENAI] = OpenAIClient()
            self.logger.info("OpenAI客户端初始化成功")
        except Exception as e:
            self.logger.warning(f"OpenAI客户端初始化失败: {e}")

        # 初始化Anthropic客户端
        try:
            self.clients[LLMProvider.ANTHROPIC] = AnthropicClient()
            self.logger.info("Anthropic客户端初始化成功")
        except Exception as e:
            self.logger.warning(f"Anthropic客户端初始化失败: {e}")

    async def generate_completion(
        self,
        prompt: str,
        model: Optional[Union[str, LLMModel]] = None,
        provider: Optional[LLMProvider] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> str:
        """生成文本完成的便捷方法"""
        request = LLMRequest(
            prompt=prompt,
            model=model or self.default_model,
            provider=provider or self.default_provider,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs
        )

        response = await self.generate_completion_with_response(request)
        return response.content

    async def generate_completion_with_response(
        self,
        request: LLMRequest
    ) -> LLMResponse:
        """生成文本完成并返回完整响应"""
        provider = request.provider or self.default_provider

        # 获取客户端
        client = self.clients.get(provider)
        if not client:
            raise LLMServiceError(
                message=f"Provider {provider.value} not available",
                error_code="PROVIDER_NOT_AVAILABLE"
            )

        try:
            return await client.generate_completion(request)
        except Exception as e:
            self.logger.error(f"LLM生成完成失败: {e}")
            raise

    def get_available_providers(self) -> List[LLMProvider]:
        """获取可用的LLM提供商"""
        return list(self.clients.keys())

    def get_provider_status(self) -> Dict[str, Dict[str, Any]]:
        """获取提供商状态"""
        status = {}

        for provider, client in self.clients.items():
            status[provider.value] = {
                "available": True,
                "provider": provider.value,
                "client_type": type(client).__name__
            }

        return status

    async def test_connection(self, provider: LLMProvider) -> bool:
        """测试提供商连接"""
        client = self.clients.get(provider)
        if not client:
            return False

        try:
            test_request = LLMRequest(
                prompt="Hello, please respond with 'OK'",
                model=self.default_model,
                max_tokens=10
            )

            response = await client.generate_completion(test_request)
            return "OK" in response.content

        except Exception as e:
            self.logger.error(f"测试{provider.value}连接失败: {e}")
            return False

    def calculate_total_cost(self, responses: List[LLMResponse]) -> float:
        """计算总成本"""
        return sum(response.cost_usd or 0 for response in responses)


# 全局实例
_llm_service = None


def get_llm_service() -> LLMIntegrationService:
    """获取LLM服务实例"""
    global _llm_service
    if _llm_service is None:
        _llm_service = LLMIntegrationService()
    return _llm_service


# 便捷函数
async def complete_text(
    prompt: str,
    model: Optional[Union[str, LLMModel]] = None,
    provider: Optional[LLMProvider] = None,
    **kwargs
) -> str:
    """文本生成的便捷函数"""
    service = get_llm_service()
    return await service.generate_completion(prompt, model, provider, **kwargs)
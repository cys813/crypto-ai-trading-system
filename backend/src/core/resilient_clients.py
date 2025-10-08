"""
弹性客户端模块

实现LLM服务和交易所API的弹性调用机制，包括重试、降级、熔断等功能。
"""

import asyncio
import logging
import time
from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Union, Callable
from dataclasses import dataclass
import json

from .recovery import health_manager, FailureSeverity, RecoveryAction
from .exceptions import ExternalServiceError, ConfigurationError

logger = logging.getLogger(__name__)


class ClientState(str, Enum):
    """客户端状态"""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    FAILED = "failed"
    CIRCUIT_OPEN = "circuit_open"


@dataclass
class RetryConfig:
    """重试配置"""
    max_attempts: int = 3
    base_delay: float = 1.0
    max_delay: float = 60.0
    exponential_base: float = 2.0
    jitter: bool = True


@dataclass
class CircuitBreakerConfig:
    """熔断器配置"""
    failure_threshold: int = 5
    recovery_timeout: float = 60.0
    success_threshold: int = 3
    monitor_period: float = 60.0


@dataclass
class FallbackConfig:
    """降级配置"""
    enabled: bool = True
    fallback_providers: List[str] = None
    fallback_strategy: str = "sequential"  # sequential, random, priority


class ResilientClient(ABC):
    """弹性客户端基类"""

    def __init__(self, name: str, retry_config: RetryConfig = None,
                 circuit_config: CircuitBreakerConfig = None,
                 fallback_config: FallbackConfig = None):
        self.name = name
        self.retry_config = retry_config or RetryConfig()
        self.circuit_config = circuit_config or CircuitBreakerConfig()
        self.fallback_config = fallback_config or FallbackConfig()

        # 状态管理
        self.state = ClientState.HEALTHY
        self.failure_count = 0
        self.last_failure_time = None
        self.success_count = 0
        self.last_success_time = None

        # 统计信息
        self.total_requests = 0
        self.successful_requests = 0
        self.failed_requests = 0

        # 注册到健康管理器
        health_manager.register_service(name, {
            "retry_config": self.retry_config,
            "circuit_config": self.circuit_config,
            "fallback_config": self.fallback_config,
        })

    async def call(self, method: str, *args, **kwargs) -> Any:
        """弹性调用方法"""
        self.total_requests += 1

        # 检查熔断器状态
        if self.state == ClientState.CIRCUIT_OPEN:
            if self._should_attempt_reset():
                self.state = ClientState.DEGRADED
            else:
                raise ExternalServiceError(f"Service {self.name} circuit breaker is open")

        # 尝试调用
        try:
            result = await self._execute_with_retry(method, *args, **kwargs)
            await self._record_success()
            return result

        except Exception as e:
            await self._record_failure(str(e))
            raise

    async def _execute_with_retry(self, method: str, *args, **kwargs) -> Any:
        """执行带重试的调用"""
        last_exception = None

        for attempt in range(self.retry_config.max_attempts):
            try:
                # 执行实际调用
                if attempt == 0:
                    return await self._do_call(method, *args, **kwargs)
                else:
                    # 使用降级策略
                    return await self._attempt_fallback(method, *args, **kwargs)

            except Exception as e:
                last_exception = e
                if attempt < self.retry_config.max_attempts - 1:
                    delay = self._calculate_retry_delay(attempt)
                    logger.warning(f"调用 {self.name}.{method} 失败 (尝试 {attempt + 1}): {e}, {delay}s后重试")
                    await asyncio.sleep(delay)
                else:
                    logger.error(f"调用 {self.name}.{method} 最终失败: {e}")

        raise last_exception

    @abstractmethod
    async def _do_call(self, method: str, *args, **kwargs) -> Any:
        """执行实际调用 - 子类实现"""
        pass

    async def _attempt_fallback(self, method: str, *args, **kwargs) -> Any:
        """尝试降级调用"""
        if not self.fallback_config.enabled:
            raise ExternalServiceError(f"No fallback configured for {self.name}")

        # 子类可以重写此方法实现具体的降级逻辑
        return await self._do_call(method, *args, **kwargs)

    def _calculate_retry_delay(self, attempt: int) -> float:
        """计算重试延迟"""
        delay = self.retry_config.base_delay * (self.retry_config.exponential_base ** attempt)
        delay = min(delay, self.retry_config.max_delay)

        if self.retry_config.jitter:
            # 添加抖动
            import random
            delay *= (0.5 + random.random() * 0.5)

        return delay

    async def _record_success(self):
        """记录成功"""
        self.successful_requests += 1
        self.success_count += 1
        self.last_success_time = datetime.utcnow()
        self.failure_count = 0

        # 更新状态
        if self.state == ClientState.DEGRADED:
            if self.success_count >= self.circuit_config.success_threshold:
                self.state = ClientState.HEALTHY
                logger.info(f"服务 {self.name} 恢复健康状态")

        # 更新健康管理器
        await health_manager.check_service_health(self.name)

    async def _record_failure(self, error_message: str):
        """记录失败"""
        self.failed_requests += 1
        self.failure_count += 1
        self.last_failure_time = datetime.utcnow()
        self.success_count = 0

        # 更新状态
        if self.failure_count >= self.circuit_config.failure_threshold:
            self.state = ClientState.CIRCUIT_OPEN
            logger.warning(f"服务 {self.name} 熔断器打开")

        # 报告故障
        severity = self._determine_failure_severity()
        health_manager.report_failure(
            service=self.name,
            error_type="service_call_failure",
            severity=severity,
            message=error_message,
            context={
                "failure_count": self.failure_count,
                "state": self.state.value,
                "total_requests": self.total_requests,
                "success_rate": self.get_success_rate()
            }
        )

    def _determine_failure_severity(self) -> FailureSeverity:
        """确定故障严重程度"""
        if self.failure_count >= self.circuit_config.failure_threshold:
            return FailureSeverity.HIGH
        elif self.failure_count >= self.circuit_config.failure_threshold // 2:
            return FailureSeverity.MEDIUM
        else:
            return FailureSeverity.LOW

    def _should_attempt_reset(self) -> bool:
        """是否应该尝试重置熔断器"""
        if self.last_failure_time is None:
            return True

        time_since_failure = time.time() - self.last_failure_time.timestamp()
        return time_since_failure >= self.circuit_config.recovery_timeout

    def get_success_rate(self) -> float:
        """获取成功率"""
        if self.total_requests == 0:
            return 1.0
        return self.successful_requests / self.total_requests

    def get_status(self) -> Dict[str, Any]:
        """获取客户端状态"""
        return {
            "name": self.name,
            "state": self.state.value,
            "success_rate": self.get_success_rate(),
            "total_requests": self.total_requests,
            "successful_requests": self.successful_requests,
            "failed_requests": self.failed_requests,
            "failure_count": self.failure_count,
            "success_count": self.success_count,
            "last_failure_time": self.last_failure_time,
            "last_success_time": self.last_success_time,
        }


class ResilientLLMClient(ResilientClient):
    """弹性LLM客户端"""

    def __init__(self, providers: Dict[str, Any], **kwargs):
        super().__init__("llm_service", **kwargs)
        self.providers = providers
        self.primary_provider = list(providers.keys())[0]
        self.current_provider = self.primary_provider

    async def _do_call(self, method: str, *args, **kwargs) -> Any:
        """执行LLM调用"""
        provider = self.providers.get(self.current_provider)
        if not provider:
            raise ConfigurationError(f"LLM provider {self.current_provider} not found")

        try:
            # 根据方法名调用相应的LLM功能
            if method == "generate_text":
                return await self._generate_text(provider, *args, **kwargs)
            elif method == "summarize":
                return await self._summarize(provider, *args, **kwargs)
            elif method == "analyze_strategy":
                return await self._analyze_strategy(provider, *args, **kwargs)
            else:
                raise ValueError(f"Unknown LLM method: {method}")

        except Exception as e:
            logger.error(f"LLM调用失败 {self.current_provider}.{method}: {e}")
            raise

    async def _attempt_fallback(self, method: str, *args, **kwargs) -> Any:
        """尝试LLM降级调用"""
        # 切换到备用提供商
        if self.current_provider == self.primary_provider:
            # 尝试其他提供商
            for provider_name in self.providers:
                if provider_name != self.primary_provider:
                    try:
                        self.current_provider = provider_name
                        logger.info(f"切换到备用LLM提供商: {provider_name}")
                        return await self._do_call(method, *args, **kwargs)
                    except Exception:
                        continue

        # 如果所有LLM提供商都失败，使用传统算法降级
        return await self._fallback_to_traditional_analysis(method, *args, **kwargs)

    async def _fallback_to_traditional_analysis(self, method: str, *args, **kwargs) -> Any:
        """降级到传统分析"""
        logger.warning(f"LLM服务完全不可用，使用传统算法替代: {method}")

        if method == "summarize":
            return await self._traditional_summarize(*args, **kwargs)
        elif method == "analyze_strategy":
            return await self._traditional_strategy_analysis(*args, **kwargs)
        else:
            raise ExternalServiceError(f"无法为方法 {method} 提供传统算法降级")

    async def _generate_text(self, provider: Any, prompt: str, **kwargs) -> str:
        """生成文本"""
        # 实现具体的文本生成逻辑
        return "Generated text response"

    async def _summarize(self, provider: Any, content: str, **kwargs) -> str:
        """内容概括"""
        # 实现具体的内容概括逻辑
        return "Content summary"

    async def _analyze_strategy(self, provider: Any, market_data: Dict, **kwargs) -> Dict:
        """策略分析"""
        # 实现具体的策略分析逻辑
        return {"recommendation": "hold", "confidence": 0.5}

    async def _traditional_summarize(self, content: str, **kwargs) -> str:
        """传统概括方法"""
        # 简单的文本提取和概括
        sentences = content.split('.')
        summary = '. '.join(sentences[:3])  # 取前3句作为摘要
        return summary.strip()

    async def _traditional_strategy_analysis(self, market_data: Dict, **kwargs) -> Dict:
        """传统策略分析方法"""
        # 基于简单技术指标的分析
        return {
            "recommendation": "hold",
            "confidence": 0.6,
            "reasoning": "Traditional analysis based on basic indicators"
        }


class ResilientExchangeClient(ResilientClient):
    """弹性交易所客户端"""

    def __init__(self, exchange_name: str, api_client: Any, **kwargs):
        super().__init__(f"exchange_{exchange_name}", **kwargs)
        self.exchange_name = exchange_name
        self.api_client = api_client
        self.backup_exchanges = kwargs.get("backup_exchanges", [])

    async def _do_call(self, method: str, *args, **kwargs) -> Any:
        """执行交易所API调用"""
        try:
            # 根据方法名调用相应的交易所功能
            if method == "get_ticker":
                return await self._get_ticker(*args, **kwargs)
            elif method == "get_klines":
                return await self._get_klines(*args, **kwargs)
            elif method == "create_order":
                return await self._create_order(*args, **kwargs)
            elif method == "cancel_order":
                return await self._cancel_order(*args, **kwargs)
            elif method == "get_balance":
                return await self._get_balance(*args, **kwargs)
            else:
                raise ValueError(f"Unknown exchange method: {method}")

        except Exception as e:
            logger.error(f"交易所API调用失败 {self.exchange_name}.{method}: {e}")
            raise

    async def _attempt_fallback(self, method: str, *args, **kwargs) -> Any:
        """尝试交易所API降级调用"""
        # 尝试备用交易所
        for backup_exchange in self.backup_exchanges:
            try:
                logger.info(f"切换到备用交易所: {backup_exchange}")
                # 这里应该实现备用交易所的调用逻辑
                # 暂时返回模拟数据
                return await self._get_mock_data(method, *args, **kwargs)
            except Exception:
                continue

        # 使用缓存数据降级
        return await self._fallback_to_cached_data(method, *args, **kwargs)

    async def _get_mock_data(self, method: str, *args, **kwargs) -> Any:
        """获取模拟数据"""
        if method == "get_ticker":
            return {"symbol": args[0], "price": 50000.0, "volume": 1000.0}
        elif method == "get_klines":
            return [[1609459200000, 50000.0, 50100.0, 49900.0, 50050.0, 100.0]]
        else:
            return {"status": "mock", "message": f"Mock response for {method}"}

    async def _fallback_to_cached_data(self, method: str, *args, **kwargs) -> Any:
        """降级到缓存数据"""
        try:
            cache = health_manager.cache
            cache_key = f"exchange_cache:{self.exchange_name}:{method}:{hash(str(args))}"

            cached_data = cache.get(cache_key)
            if cached_data:
                logger.info(f"使用缓存数据: {cache_key}")
                return cached_data

            logger.warning(f"无可用缓存数据: {cache_key}")
            raise ExternalServiceError(f"No cached data available for {method}")

        except Exception as e:
            logger.error(f"缓存数据访问失败: {e}")
            raise

    async def _get_ticker(self, symbol: str, **kwargs) -> Dict:
        """获取行情数据"""
        # 实现具体的行情获取逻辑
        return {"symbol": symbol, "price": 50000.0, "volume": 1000.0}

    async def _get_klines(self, symbol: str, timeframe: str, limit: int = 100, **kwargs) -> List:
        """获取K线数据"""
        # 实现具体的K线数据获取逻辑
        return [[1609459200000, 50000.0, 50100.0, 49900.0, 50050.0, 100.0]]

    async def _create_order(self, symbol: str, side: str, amount: float, price: float = None, **kwargs) -> Dict:
        """创建订单"""
        # 实现具体的订单创建逻辑
        return {"order_id": "12345", "status": "open"}

    async def _cancel_order(self, order_id: str, **kwargs) -> Dict:
        """取消订单"""
        # 实现具体的订单取消逻辑
        return {"order_id": order_id, "status": "cancelled"}

    async def _get_balance(self, **kwargs) -> Dict:
        """获取账户余额"""
        # 实现具体的余额获取逻辑
        return {"BTC": 1.0, "USDT": 50000.0}


class ResilientClientManager:
    """弹性客户端管理器"""

    def __init__(self):
        self.clients: Dict[str, ResilientClient] = {}

    def register_client(self, name: str, client: ResilientClient):
        """注册客户端"""
        self.clients[name] = client
        logger.info(f"注册弹性客户端: {name}")

    def get_client(self, name: str) -> Optional[ResilientClient]:
        """获取客户端"""
        return self.clients.get(name)

    async def call_client(self, client_name: str, method: str, *args, **kwargs) -> Any:
        """调用客户端"""
        client = self.get_client(client_name)
        if not client:
            raise ConfigurationError(f"Client {client_name} not found")

        return await client.call(method, *args, **kwargs)

    def get_all_clients_status(self) -> Dict[str, Dict[str, Any]]:
        """获取所有客户端状态"""
        return {name: client.get_status() for name, client in self.clients.items()}

    async def start_health_monitoring(self):
        """启动健康监控"""
        asyncio.create_task(self._health_monitoring_loop())

    async def _health_monitoring_loop(self):
        """健康监控循环"""
        while True:
            try:
                for name, client in self.clients.items():
                    await health_manager.check_service_health(name)
                await asyncio.sleep(60)  # 每分钟检查一次
            except Exception as e:
                logger.error(f"健康监控检查失败: {e}")
                await asyncio.sleep(60)


# 全局客户端管理器
client_manager = ResilientClientManager()
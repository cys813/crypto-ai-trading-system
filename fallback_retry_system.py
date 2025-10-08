#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
API调用降级和重试策略系统
支持智能重试、熔断器、降级服务、故障转移等功能
"""

import asyncio
import time
import json
import logging
import random
import hashlib
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Callable, Union
from enum import Enum
from datetime import datetime, timedelta
import redis
from functools import wraps
import httpx

logger = logging.getLogger(__name__)


class RetryStrategy(Enum):
    """重试策略"""
    EXPONENTIAL_BACKOFF = "exponential_backoff"
    LINEAR_BACKOFF = "linear_backoff"
    FIXED_DELAY = "fixed_delay"
    FIBONACCI_BACKOFF = "fibonacci_backoff"
    JITTER_BACKOFF = "jitter_backoff"


class CircuitBreakerState(Enum):
    """熔断器状态"""
    CLOSED = "closed"      # 正常状态
    OPEN = "open"          # 熔断状态
    HALF_OPEN = "half_open"  # 半开状态


class FallbackLevel(Enum):
    """降级级别"""
    NONE = "none"          # 不降级
    PARTIAL = "partial"    # 部分降级
    FULL = "full"          # 完全降级
    EMERGENCY = "emergency"  # 紧急降级


@dataclass
class RetryConfig:
    """重试配置"""
    max_attempts: int = 3
    strategy: RetryStrategy = RetryStrategy.EXPONENTIAL_BACKOFF
    base_delay: float = 1.0
    max_delay: float = 60.0
    multiplier: float = 2.0
    jitter: bool = True
    retry_on: List[Union[int, type]] = field(default_factory=lambda: [429, 500, 502, 503, 504])
    stop_on: List[Union[int, type]] = field(default_factory=lambda: [400, 401, 403, 404])


@dataclass
class CircuitBreakerConfig:
    """熔断器配置"""
    failure_threshold: int = 5
    recovery_timeout: float = 60.0
    expected_exception: type = Exception
    success_threshold: int = 3  # 半开状态下成功次数阈值


@dataclass
class FallbackConfig:
    """降级配置"""
    level: FallbackLevel = FallbackLevel.NONE
    timeout: float = 5.0
    cache_enabled: bool = True
    cache_ttl: int = 300
    alternative_endpoints: List[str] = field(default_factory=list)
    fallback_function: Optional[Callable] = None


class RetryDelayCalculator:
    """重试延迟计算器"""

    @staticmethod
    def calculate_delay(
        attempt: int,
        config: RetryConfig
    ) -> float:
        """计算重试延迟"""
        if config.strategy == RetryStrategy.FIXED_DELAY:
            delay = config.base_delay

        elif config.strategy == RetryStrategy.LINEAR_BACKOFF:
            delay = config.base_delay * attempt

        elif config.strategy == RetryStrategy.EXPONENTIAL_BACKOFF:
            delay = config.base_delay * (config.multiplier ** (attempt - 1))

        elif config.strategy == RetryStrategy.FIBONACCI_BACKOFF:
            delay = config.base_delay * RetryDelayCalculator._fibonacci(attempt)

        elif config.strategy == RetryStrategy.JITTER_BACKOFF:
            exponential_delay = config.base_delay * (config.multiplier ** (attempt - 1))
            jitter_range = exponential_delay * 0.1
            delay = exponential_delay + random.uniform(-jitter_range, jitter_range)

        else:
            delay = config.base_delay

        # 应用最大延迟限制
        delay = min(delay, config.max_delay)

        # 添加随机抖动
        if config.jitter and config.strategy != RetryStrategy.JITTER_BACKOFF:
            jitter = delay * 0.1 * random.random()
            delay += jitter

        return delay

    @staticmethod
    def _fibonacci(n: int) -> int:
        """计算斐波那契数列"""
        if n <= 1:
            return n
        a, b = 0, 1
        for _ in range(2, n + 1):
            a, b = b, a + b
        return b


class CircuitBreaker:
    """熔断器"""

    def __init__(self, name: str, config: CircuitBreakerConfig, redis_client: Optional[redis.Redis] = None):
        self.name = name
        self.config = config
        self.redis = redis_client

        # 本地状态
        self.failure_count = 0
        self.last_failure_time = None
        self.state = CircuitBreakerState.CLOSED
        self.success_count = 0

        # Redis键（用于分布式状态同步）
        if redis_client:
            self.redis_key = f"circuit_breaker:{name}"
            self._load_state_from_redis()

    async def call(self, func: Callable, *args, **kwargs) -> Any:
        """通过熔断器调用函数"""
        if not await self._allow_request():
            raise Exception(f"熔断器开启，拒绝请求: {self.name}")

        try:
            result = await func(*args, **kwargs)
            await self._on_success()
            return result

        except Exception as e:
            await self._on_failure()
            raise

    async def _allow_request(self) -> bool:
        """检查是否允许请求"""
        if self.state == CircuitBreakerState.CLOSED:
            return True

        elif self.state == CircuitBreakerState.OPEN:
            # 检查是否可以转为半开状态
            if time.time() - self.last_failure_time >= self.config.recovery_timeout:
                await self._set_state(CircuitBreakerState.HALF_OPEN)
                return True
            return False

        elif self.state == CircuitBreakerState.HALF_OPEN:
            return True

        return False

    async def _on_success(self):
        """成功时调用"""
        self.success_count += 1

        if self.state == CircuitBreakerState.HALF_OPEN:
            if self.success_count >= self.config.success_threshold:
                await self._reset()

        if self.state == CircuitBreakerState.CLOSED:
            self.failure_count = 0

    async def _on_failure(self):
        """失败时调用"""
        self.failure_count += 1
        self.last_failure_time = time.time()
        self.success_count = 0

        if self.state == CircuitBreakerState.HALF_OPEN:
            await self._set_state(CircuitBreakerState.OPEN)

        elif self.failure_count >= self.config.failure_threshold:
            await self._set_state(CircuitBreakerState.OPEN)

    async def _set_state(self, state: CircuitBreakerState):
        """设置状态"""
        self.state = state
        if self.redis:
            await self._save_state_to_redis()

        logger.info(f"熔断器 {self.name} 状态变更: {state.value}")

    async def _reset(self):
        """重置熔断器"""
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time = None
        await self._set_state(CircuitBreakerState.CLOSED)

    def _load_state_from_redis(self):
        """从Redis加载状态"""
        try:
            state_data = self.redis.get(self.redis_key)
            if state_data:
                data = json.loads(state_data)
                self.state = CircuitBreakerState(data['state'])
                self.failure_count = data['failure_count']
                self.last_failure_time = data['last_failure_time']
        except Exception as e:
            logger.error(f"从Redis加载熔断器状态失败: {e}")

    async def _save_state_to_redis(self):
        """保存状态到Redis"""
        try:
            state_data = {
                'state': self.state.value,
                'failure_count': self.failure_count,
                'last_failure_time': self.last_failure_time,
                'updated_at': time.time()
            }
            self.redis.setex(self.redis_key, 3600, json.dumps(state_data))
        except Exception as e:
            logger.error(f"保存熔断器状态到Redis失败: {e}")

    def get_status(self) -> Dict[str, Any]:
        """获取熔断器状态"""
        return {
            "name": self.name,
            "state": self.state.value,
            "failure_count": self.failure_count,
            "success_count": self.success_count,
            "last_failure_time": self.last_failure_time,
            "config": {
                "failure_threshold": self.config.failure_threshold,
                "recovery_timeout": self.config.recovery_timeout,
                "success_threshold": self.config.success_threshold
            }
        }


class FallbackManager:
    """降级管理器"""

    def __init__(self, redis_client: redis.Redis):
        self.redis = redis_client
        self.cache_key_prefix = "fallback_cache"
        self.metrics_key_prefix = "fallback_metrics"

    async def execute_with_fallback(
        self,
        primary_func: Callable,
        fallback_config: FallbackConfig,
        *args,
        **kwargs
    ) -> Any:
        """执行带降级的函数"""
        cache_key = self._generate_cache_key(primary_func, args, kwargs)

        # 尝试从缓存获取
        if fallback_config.cache_enabled:
            cached_result = await self._get_from_cache(cache_key)
            if cached_result is not None:
                logger.info(f"使用缓存结果: {cache_key}")
                await self._record_metrics("cache_hit", fallback_config.level)
                return cached_result

        # 尝试执行主函数
        try:
            result = await asyncio.wait_for(
                primary_func(*args, **kwargs),
                timeout=fallback_config.timeout
            )

            # 缓存结果
            if fallback_config.cache_enabled:
                await self._set_cache(cache_key, result, fallback_config.cache_ttl)

            await self._record_metrics("primary_success", fallback_config.level)
            return result

        except Exception as e:
            logger.warning(f"主函数执行失败: {e}")

            # 执行降级策略
            return await self._execute_fallback_strategy(
                fallback_config,
                cache_key,
                primary_func,
                args,
                kwargs,
                e
            )

    async def _execute_fallback_strategy(
        self,
        config: FallbackConfig,
        cache_key: str,
        primary_func: Callable,
        args: tuple,
        kwargs: dict,
        error: Exception
    ) -> Any:
        """执行降级策略"""
        if config.level == FallbackLevel.NONE:
            raise error

        await self._record_metrics("fallback_used", config.level)

        # 1. 尝试备用端点
        if config.alternative_endpoints:
            for endpoint in config.alternative_endpoints:
                try:
                    result = await self._call_alternative_endpoint(endpoint, *args, **kwargs)
                    logger.info(f"备用端点成功: {endpoint}")
                    return result
                except Exception as e:
                    logger.warning(f"备用端点失败: {endpoint} - {e}")

        # 2. 尝试降级函数
        if config.fallback_function:
            try:
                result = await config.fallback_function(error, *args, **kwargs)
                logger.info("降级函数执行成功")
                return result
            except Exception as e:
                logger.error(f"降级函数执行失败: {e}")

        # 3. 尝试从缓存获取（即使过期）
        if config.cache_enabled:
            stale_result = await self._get_stale_cache(cache_key)
            if stale_result is not None:
                logger.warning("使用过期缓存数据")
                await self._record_metrics("stale_cache_used", config.level)
                return stale_result

        # 4. 返回默认值或抛出异常
        if config.level in [FallbackLevel.FULL, FallbackLevel.EMERGENCY]:
            logger.error("完全降级，返回默认值")
            return self._get_default_value(config.level)
        else:
            raise error

    async def _call_alternative_endpoint(self, endpoint: str, *args, **kwargs) -> Any:
        """调用备用端点"""
        # 这里需要根据具体业务逻辑实现
        async with httpx.AsyncClient() as client:
            response = await client.get(endpoint, timeout=10.0)
            response.raise_for_status()
            return response.json()

    def _generate_cache_key(self, func: Callable, args: tuple, kwargs: dict) -> str:
        """生成缓存键"""
        key_data = f"{func.__name__}:{str(args)}:{str(sorted(kwargs.items()))}"
        return f"{self.cache_key_prefix}:{hashlib.md5(key_data.encode()).hexdigest()}"

    async def _get_from_cache(self, key: str) -> Optional[Any]:
        """从缓存获取数据"""
        try:
            data = self.redis.get(key)
            if data:
                return json.loads(data)
        except Exception as e:
            logger.error(f"从缓存获取数据失败: {e}")
        return None

    async def _get_stale_cache(self, key: str) -> Optional[Any]:
        """获取过期缓存数据"""
        try:
            # 使用持久化连接获取过期数据
            stale_key = f"{key}:stale"
            data = self.redis.get(stale_key)
            if data:
                return json.loads(data)
        except Exception as e:
            logger.error(f"获取过期缓存失败: {e}")
        return None

    async def _set_cache(self, key: str, value: Any, ttl: int):
        """设置缓存"""
        try:
            data = json.dumps(value)
            self.redis.setex(key, ttl, data)
            # 同时保存一份更长时间的备用数据
            self.redis.setex(f"{key}:stale", ttl * 3, data)
        except Exception as e:
            logger.error(f"设置缓存失败: {e}")

    def _get_default_value(self, level: FallbackLevel) -> Any:
        """获取默认值"""
        defaults = {
            FallbackLevel.PARTIAL: {"status": "partial_degraded", "data": None},
            FallbackLevel.FULL: {"status": "fully_degraded", "data": None},
            FallbackLevel.EMERGENCY: {"status": "emergency_degraded", "data": None}
        }
        return defaults.get(level, {"status": "unknown", "data": None})

    async def _record_metrics(self, action: str, level: FallbackLevel):
        """记录指标"""
        try:
            key = f"{self.metrics_key_prefix}:{action}:{level.value}"
            self.redis.incr(key)
            self.redis.expire(key, 86400)  # 保存24小时
        except Exception as e:
            logger.error(f"记录指标失败: {e}")

    async def get_metrics(self) -> Dict[str, Any]:
        """获取降级指标"""
        metrics = {"timestamp": datetime.now().isoformat()}
        try:
            pattern = f"{self.metrics_key_prefix}:*"
            keys = self.redis.keys(pattern)

            for key in keys:
                parts = key.split(":")
                if len(parts) >= 4:
                    action = parts[2]
                    level = parts[3]
                    count = int(self.redis.get(key) or 0)

                    if action not in metrics:
                        metrics[action] = {}
                    metrics[action][level] = count

        except Exception as e:
            logger.error(f"获取降级指标失败: {e}")

        return metrics


class ResilientAPIClient:
    """具有弹性的API客户端"""

    def __init__(self, redis_client: redis.Redis):
        self.redis = redis_client
        self.circuit_breakers = {}
        self.fallback_manager = FallbackManager(redis_client)
        self.request_metrics = {}

    def get_circuit_breaker(self, name: str, config: Optional[CircuitBreakerConfig] = None) -> CircuitBreaker:
        """获取或创建熔断器"""
        if name not in self.circuit_breakers:
            config = config or CircuitBreakerConfig()
            self.circuit_breakers[name] = CircuitBreaker(name, config, self.redis)
        return self.circuit_breakers[name]

    async def resilient_request(
        self,
        exchange: str,
        endpoint: str,
        method: str = "GET",
        params: Optional[Dict[str, Any]] = None,
        data: Optional[Dict[str, Any]] = None,
        retry_config: Optional[RetryConfig] = None,
        fallback_config: Optional[FallbackConfig] = None,
        circuit_breaker_config: Optional[CircuitBreakerConfig] = None,
        **kwargs
    ) -> Any:
        """弹性API请求"""
        retry_config = retry_config or RetryConfig()
        fallback_config = fallback_config or FallbackConfig()

        # 获取熔断器
        breaker_name = f"{exchange}_{endpoint}"
        circuit_breaker = self.get_circuit_breaker(breaker_name, circuit_breaker_config)

        # 定义主请求函数
        async def make_request():
            return await self._make_single_request(
                exchange, endpoint, method, params, data, **kwargs
            )

        # 定义重试函数
        async def retry_request():
            last_exception = None

            for attempt in range(1, retry_config.max_attempts + 1):
                try:
                    # 通过熔断器执行请求
                    result = await circuit_breaker.call(make_request)

                    # 记录成功指标
                    await self._record_request_metrics(exchange, endpoint, "success", attempt)

                    return result

                except Exception as e:
                    last_exception = e

                    # 检查是否应该重试
                    if not self._should_retry(e, attempt, retry_config):
                        break

                    # 记录失败指标
                    await self._record_request_metrics(exchange, endpoint, "retry", attempt)

                    # 计算延迟并等待
                    delay = RetryDelayCalculator.calculate_delay(attempt, retry_config)
                    logger.info(f"第 {attempt} 次重试，延迟 {delay:.2f}s: {endpoint}")
                    await asyncio.sleep(delay)

            # 所有重试都失败了
            raise last_exception

        # 执行带降级的请求
        try:
            return await self.fallback_manager.execute_with_fallback(
                retry_request,
                fallback_config
            )
        except Exception as e:
            # 最终失败处理
            await self._record_request_metrics(exchange, endpoint, "failed", 0)
            raise

    async def _make_single_request(
        self,
        exchange: str,
        endpoint: str,
        method: str,
        params: Optional[Dict[str, Any]],
        data: Optional[Dict[str, Any]],
        **kwargs
    ) -> Any:
        """执行单次API请求"""
        # 这里实现具体的API调用逻辑
        base_urls = {
            "binance": "https://api.binance.com",
            "coinbase": "https://api.pro.coinbase.com",
            "kraken": "https://api.kraken.com",
            "huobi": "https://api.huobi.pro",
            "okex": "https://www.okex.com"
        }

        base_url = base_urls.get(exchange)
        if not base_url:
            raise ValueError(f"不支持的交易所: {exchange}")

        url = f"{base_url}{endpoint}"

        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.request(
                method=method,
                url=url,
                params=params,
                json=data,
                **kwargs
            )
            response.raise_for_status()
            return response.json()

    def _should_retry(self, exception: Exception, attempt: int, config: RetryConfig) -> bool:
        """判断是否应该重试"""
        if attempt >= config.max_attempts:
            return False

        # 检查停止重试的条件
        for stop_condition in config.stop_on:
            if isinstance(stop_condition, type) and isinstance(exception, stop_condition):
                return False
            elif isinstance(stop_condition, int) and hasattr(exception, 'status_code'):
                if exception.status_code == stop_condition:
                    return False

        # 检查重试条件
        for retry_condition in config.retry_on:
            if isinstance(retry_condition, type) and isinstance(exception, retry_condition):
                return True
            elif isinstance(retry_condition, int) and hasattr(exception, 'status_code'):
                if exception.status_code == retry_condition:
                    return True

        return False

    async def _record_request_metrics(self, exchange: str, endpoint: str, status: str, attempt: int):
        """记录请求指标"""
        try:
            key = f"api_metrics:{exchange}:{endpoint}:{status}"
            self.redis.incr(key)
            self.redis.expire(key, 86400)

            # 记录尝试次数
            if status == "success" and attempt > 0:
                retry_key = f"api_metrics:{exchange}:{endpoint}:retry_success"
                self.redis.incr(retry_key)
                self.redis.expire(retry_key, 86400)

        except Exception as e:
            logger.error(f"记录请求指标失败: {e}")

    async def get_circuit_breaker_status(self) -> Dict[str, Any]:
        """获取所有熔断器状态"""
        status = {"timestamp": datetime.now().isoformat(), "breakers": {}}

        for name, breaker in self.circuit_breakers.items():
            status["breakers"][name] = breaker.get_status()

        return status

    async def get_request_metrics(self) -> Dict[str, Any]:
        """获取请求指标"""
        metrics = {"timestamp": datetime.now().isoformat()}

        try:
            pattern = "api_metrics:*"
            keys = self.redis.keys(pattern)

            for key in keys:
                parts = key.split(":")
                if len(parts) >= 4:
                    exchange = parts[1]
                    endpoint = parts[2]
                    status = parts[3]
                    count = int(self.redis.get(key) or 0)

                    if exchange not in metrics:
                        metrics[exchange] = {}
                    if endpoint not in metrics[exchange]:
                        metrics[exchange][endpoint] = {}

                    metrics[exchange][endpoint][status] = count

        except Exception as e:
            logger.error(f"获取请求指标失败: {e}")

        return metrics


# 装饰器版本
def resilient(
    retry_config: Optional[RetryConfig] = None,
    fallback_config: Optional[FallbackConfig] = None,
    circuit_breaker_config: Optional[CircuitBreakerConfig] = None
):
    """弹性调用装饰器"""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # 这里需要访问全局的客户端实例
            # 实际使用时需要通过依赖注入或其他方式获取
            pass
        return wrapper
    return decorator


# 使用示例
async def example_resilient_client():
    """弹性客户端使用示例"""

    # 初始化Redis客户端
    redis_client = redis.Redis(host='localhost', port=6379, db=0, decode_responses=True)

    # 创建弹性API客户端
    client = ResilientAPIClient(redis_client)

    # 配置重试策略
    retry_config = RetryConfig(
        max_attempts=5,
        strategy=RetryStrategy.EXPONENTIAL_BACKOFF,
        base_delay=1.0,
        max_delay=30.0,
        jitter=True
    )

    # 配置降级策略
    fallback_config = FallbackConfig(
        level=FallbackLevel.PARTIAL,
        timeout=10.0,
        cache_enabled=True,
        cache_ttl=300,
        alternative_endpoints=[
            "https://backup-api.example.com/v1/ticker/BTCUSDT"
        ],
        fallback_function=lambda error, *args, **kwargs: {
            "status": "fallback",
            "price": "50000.00",
            "error": str(error)
        }
    )

    # 配置熔断器
    circuit_breaker_config = CircuitBreakerConfig(
        failure_threshold=3,
        recovery_timeout=60.0,
        success_threshold=2
    )

    try:
        # 执行弹性API请求
        print("=== 测试弹性API请求 ===")

        result = await client.resilient_request(
            exchange="binance",
            endpoint="/api/v3/ticker/price",
            params={"symbol": "BTCUSDT"},
            retry_config=retry_config,
            fallback_config=fallback_config,
            circuit_breaker_config=circuit_breaker_config
        )

        print(f"请求成功: {result}")

        # 模拟多次请求以触发熔断器
        print("\n=== 测试熔断器 ===")
        for i in range(5):
            try:
                # 使用一个不存在的端点来触发失败
                await client.resilient_request(
                    exchange="binance",
                    endpoint="/api/v3/invalid/endpoint",
                    retry_config=retry_config,
                    fallback_config=fallback_config,
                    circuit_breaker_config=circuit_breaker_config
                )
            except Exception as e:
                print(f"请求 {i+1} 失败: {e}")

        # 查看熔断器状态
        print("\n=== 熔断器状态 ===")
        breaker_status = await client.get_circuit_breaker_status()
        print(json.dumps(breaker_status, indent=2, ensure_ascii=False))

        # 查看请求指标
        print("\n=== 请求指标 ===")
        request_metrics = await client.get_request_metrics()
        print(json.dumps(request_metrics, indent=2, ensure_ascii=False))

        # 查看降级指标
        print("\n=== 降级指标 ===")
        fallback_metrics = await client.fallback_manager.get_metrics()
        print(json.dumps(fallback_metrics, indent=2, ensure_ascii=False))

    except Exception as e:
        print(f"请求最终失败: {e}")


if __name__ == "__main__":
    # 运行示例
    asyncio.run(example_resilient_client())
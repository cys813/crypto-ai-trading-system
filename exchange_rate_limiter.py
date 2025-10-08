#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
多交易所API集成限流和并发处理系统
支持5个主流交易所的不同限流规则
"""

import asyncio
import time
import json
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from enum import Enum
import redis
import httpx
from datetime import datetime, timedelta

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ExchangeType(Enum):
    """交易所类型枚举"""
    BINANCE = "binance"
    COINBASE = "coinbase"
    KRAKEN = "kraken"
    HUOBI = "huobi"
    OKEX = "okex"


class RateLimitType(Enum):
    """限流类型枚举"""
    REQUEST_WEIGHT = "request_weight"
    ORDERS = "orders"
    CONNECTIONS = "connections"


class Priority(Enum):
    """请求优先级"""
    HIGH = 1      # 市价单、紧急止损
    MEDIUM = 2    # 限价单、查询余额
    LOW = 3       # 历史数据、统计分析


@dataclass
class RateLimitRule:
    """限流规则配置"""
    limit_type: RateLimitType
    limit: int
    interval: int  # 秒
    interval_num: int = 1

    @property
    def window_seconds(self) -> int:
        """获取窗口时间（秒）"""
        return self.interval * self.interval_num


@dataclass
class ExchangeConfig:
    """交易所配置"""
    name: str
    base_url: str
    api_key: Optional[str] = None
    api_secret: Optional[str] = None
    rate_limits: List[RateLimitRule] = field(default_factory=list)
    max_connections: int = 10
    timeout: int = 30


@dataclass
class APIRequest:
    """API请求数据结构"""
    exchange: ExchangeType
    endpoint: str
    method: str = "GET"
    params: Dict[str, Any] = field(default_factory=dict)
    data: Dict[str, Any] = field(default_factory=dict)
    headers: Dict[str, Any] = field(default_factory=dict)
    priority: Priority = Priority.MEDIUM
    weight: int = 1
    callback: Optional[callable] = None
    retry_count: int = 0
    max_retries: int = 3
    created_at: datetime = field(default_factory=datetime.now)


class RateLimiter(ABC):
    """限流器抽象基类"""

    @abstractmethod
    async def is_allowed(self, key: str, limit: int, window: int) -> bool:
        """检查是否允许请求"""
        pass

    @abstractmethod
    async def consume(self, key: str, tokens: int = 1) -> bool:
        """消费令牌"""
        pass

    @abstractmethod
    async def get_status(self, key: str) -> Dict[str, Any]:
        """获取限流状态"""
        pass


class TokenBucketRateLimiter(RateLimiter):
    """基于Redis的令牌桶限流器"""

    def __init__(self, redis_client: redis.Redis):
        self.redis = redis_client
        self.lua_script = """
        local key = KEYS[1]
        local capacity = tonumber(ARGV[1])
        local tokens = tonumber(ARGV[2])
        local interval = tonumber(ARGV[3])
        local now = tonumber(ARGV[4])

        local bucket_key = "bucket:" .. key
        local last_update_key = "last_update:" .. key

        local current_tokens = tonumber(redis.call("GET", bucket_key) or capacity)
        local last_update = tonumber(redis.call("GET", last_update_key) or now)

        -- 计算应该添加的令牌数
        local time_passed = now - last_update
        local tokens_to_add = time_passed * capacity / interval

        current_tokens = math.min(capacity, current_tokens + tokens_to_add)

        if current_tokens >= tokens then
            current_tokens = current_tokens - tokens
            redis.call("SET", bucket_key, current_tokens)
            redis.call("SET", last_update_key, now)
            redis.call("EXPIRE", bucket_key, interval)
            redis.call("EXPIRE", last_update_key, interval)
            return 1
        else
            redis.call("SET", bucket_key, current_tokens)
            redis.call("SET", last_update_key, now)
            redis.call("EXPIRE", bucket_key, interval)
            redis.call("EXPIRE", last_update_key, interval)
            return 0
        end
        """

    async def is_allowed(self, key: str, limit: int, window: int) -> bool:
        """检查是否允许请求"""
        return await self.consume(key, 1)

    async def consume(self, key: str, tokens: int = 1) -> bool:
        """消费令牌"""
        try:
            result = self.redis.eval(
                self.lua_script,
                1,
                key,
                100,  # capacity
                tokens,  # tokens to consume
                60,  # interval in seconds
                int(time.time())
            )
            return bool(result)
        except Exception as e:
            logger.error(f"令牌桶限流器错误: {e}")
            return True  # 容错：如果Redis失败，允许请求

    async def get_status(self, key: str) -> Dict[str, Any]:
        """获取限流状态"""
        try:
            bucket_key = f"bucket:{key}"
            current_tokens = self.redis.get(bucket_key)
            return {
                "current_tokens": float(current_tokens) if current_tokens else 100,
                "key": key
            }
        except Exception as e:
            logger.error(f"获取限流状态错误: {e}")
            return {"current_tokens": 0, "key": key}


class SlidingWindowRateLimiter(RateLimiter):
    """滑动窗口限流器"""

    def __init__(self, redis_client: redis.Redis):
        self.redis = redis_client
        self.lua_script = """
        local key = KEYS[1]
        local window = tonumber(ARGV[1])
        local limit = tonumber(ARGV[2])
        local now = tonumber(ARGV[3])

        local clear_before = now - window
        redis.call("ZREMRANGEBYSCORE", key, 0, clear_before)

        local current_count = redis.call("ZCARD", key)

        if current_count < limit then
            redis.call("ZADD", key, now, now)
            redis.call("EXPIRE", key, window)
            return 1
        else
            return 0
        end
        """

    async def is_allowed(self, key: str, limit: int, window: int) -> bool:
        """检查是否允许请求"""
        try:
            result = self.redis.eval(
                self.lua_script,
                1,
                key,
                window,
                limit,
                int(time.time())
            )
            return bool(result)
        except Exception as e:
            logger.error(f"滑动窗口限流器错误: {e}")
            return True

    async def consume(self, key: str, tokens: int = 1) -> bool:
        """消费令牌（对于滑动窗口，就是检查是否允许）"""
        return await self.is_allowed(key, tokens, 60)

    async def get_status(self, key: str) -> Dict[str, Any]:
        """获取限流状态"""
        try:
            current_count = self.redis.zcard(key)
            return {
                "current_count": current_count,
                "key": key
            }
        except Exception as e:
            logger.error(f"获取限流状态错误: {e}")
            return {"current_count": 0, "key": key}


class ExchangeRateLimiter:
    """交易所专用限流器"""

    def __init__(self, redis_client: redis.Redis):
        self.redis = redis_client
        self.token_bucket = TokenBucketRateLimiter(redis_client)
        self.sliding_window = SlidingWindowRateLimiter(redis_client)

        # 主流交易所限流配置
        self.exchange_configs = {
            ExchangeType.BINANCE: ExchangeConfig(
                name="Binance",
                base_url="https://api.binance.com",
                rate_limits=[
                    RateLimitRule(RateLimitType.REQUEST_WEIGHT, 6000, 60),  # 6000 weight/minute
                    RateLimitRule(RateLimitType.REQUEST_WEIGHT, 120000, 60, 10),  # 120000 weight/10 minutes
                    RateLimitRule(RateLimitType.ORDERS, 10000, 10),  # 10000 orders/10 seconds
                    RateLimitRule(RateLimitType.ORDERS, 100000, 86400),  # 100000 orders/day
                ]
            ),
            ExchangeType.COINBASE: ExchangeConfig(
                name="Coinbase",
                base_url="https://api.pro.coinbase.com",
                rate_limits=[
                    RateLimitRule(RateLimitType.REQUEST_WEIGHT, 10, 1),  # 10 requests/second
                    RateLimitRule(RateLimitType.REQUEST_WEIGHT, 600, 60),  # 600 requests/minute
                    RateLimitRule(RateLimitType.ORDERS, 50, 10),  # 50 orders/10 seconds
                ]
            ),
            ExchangeType.KRAKEN: ExchangeConfig(
                name="Kraken",
                base_url="https://api.kraken.com",
                rate_limits=[
                    RateLimitRule(RateLimitType.REQUEST_WEIGHT, 15, 1),  # 15 requests/second
                    RateLimitRule(RateLimitType.REQUEST_WEIGHT, 20, 1, 60),  # 20 requests/minute for private endpoints
                    RateLimitRule(RateLimitType.ORDERS, 30, 1),  # 30 orders/second
                ]
            ),
            ExchangeType.HUOBI: ExchangeConfig(
                name="Huobi",
                base_url="https://api.huobi.pro",
                rate_limits=[
                    RateLimitRule(RateLimitType.REQUEST_WEIGHT, 100, 1),  # 100 requests/second
                    RateLimitRule(RateLimitType.REQUEST_WEIGHT, 18000, 60),  # 18000 requests/minute
                    RateLimitRule(RateLimitType.ORDERS, 100, 1),  # 100 orders/second
                ]
            ),
            ExchangeType.OKEX: ExchangeConfig(
                name="OKEx",
                base_url="https://www.okex.com",
                rate_limits=[
                    RateLimitRule(RateLimitType.REQUEST_WEIGHT, 60, 1),  # 60 requests/second
                    RateLimitRule(RateLimitType.REQUEST_WEIGHT, 600, 10),  # 600 requests/10 seconds
                    RateLimitRule(RateLimitType.ORDERS, 60, 1),  # 60 orders/second
                ]
            ),
        }

    async def check_rate_limit(self, exchange: ExchangeType, request: APIRequest) -> Tuple[bool, str]:
        """检查请求是否满足限流要求"""
        config = self.exchange_configs.get(exchange)
        if not config:
            return False, f"不支持的交易所: {exchange}"

        # 检查所有适用的限流规则
        for rule in config.rate_limits:
            key = f"{exchange.value}:{rule.limit_type.value}"

            if rule.limit_type == RateLimitType.REQUEST_WEIGHT:
                # 使用令牌桶处理请求权重
                allowed = await self.token_bucket.is_allowed(
                    key, rule.limit, rule.window_seconds
                )
                if not allowed:
                    return False, f"请求权重超限: {rule.limit}/{rule.window_seconds}s"

            elif rule.limit_type == RateLimitType.ORDERS:
                # 使用滑动窗口处理订单限制
                allowed = await self.sliding_window.is_allowed(
                    key, rule.limit, rule.window_seconds
                )
                if not allowed:
                    return False, f"订单数量超限: {rule.limit}/{rule.window_seconds}s"

        return True, "请求允许"

    async def get_rate_limit_status(self, exchange: ExchangeType) -> Dict[str, Any]:
        """获取交易所限流状态"""
        config = self.exchange_configs.get(exchange)
        if not config:
            return {}

        status = {"exchange": exchange.value, "limits": []}

        for rule in config.rate_limits:
            key = f"{exchange.value}:{rule.limit_type.value}"

            if rule.limit_type == RateLimitType.REQUEST_WEIGHT:
                limit_status = await self.token_bucket.get_status(key)
            else:
                limit_status = await self.sliding_window.get_status(key)

            status["limits"].append({
                "type": rule.limit_type.value,
                "limit": rule.limit,
                "window": rule.window_seconds,
                "current": limit_status.get("current_tokens", 0) or limit_status.get("current_count", 0)
            })

        return status


class RequestQueue:
    """优先级请求队列"""

    def __init__(self):
        self.high_priority_queue = asyncio.Queue()
        self.medium_priority_queue = asyncio.Queue()
        self.low_priority_queue = asyncio.Queue()
        self.active_requests = set()
        self.request_semaphore = asyncio.Semaphore(50)  # 最大并发请求数

    async def add_request(self, request: APIRequest):
        """添加请求到队列"""
        if request.priority == Priority.HIGH:
            await self.high_priority_queue.put(request)
        elif request.priority == Priority.MEDIUM:
            await self.medium_priority_queue.put(request)
        else:
            await self.low_priority_queue.put(request)

    async def get_next_request(self) -> Optional[APIRequest]:
        """获取下一个请求（按优先级）"""
        queues = [
            self.high_priority_queue,
            self.medium_priority_queue,
            self.low_priority_queue
        ]

        for queue in queues:
            if not queue.empty():
                return await queue.get()

        return None

    async def acquire_request_slot(self):
        """获取请求槽位"""
        await self.request_semaphore.acquire()

    def release_request_slot(self):
        """释放请求槽位"""
        self.request_semaphore.release()


class ExchangeAPIClient:
    """交易所API客户端"""

    def __init__(self, rate_limiter: ExchangeRateLimiter, redis_client: redis.Redis):
        self.rate_limiter = rate_limiter
        self.redis = redis_client
        self.request_queue = RequestQueue()
        self.client = httpx.AsyncClient(timeout=30.0)
        self.running = False
        self.circuit_breakers = {}  # 熔断器状态

    async def start(self):
        """启动API客户端"""
        self.running = True
        # 启动多个工作协程处理请求
        tasks = []
        for i in range(5):  # 5个并发工作协程
            task = asyncio.create_task(self._worker(f"worker-{i}"))
            tasks.append(task)

        logger.info("API客户端已启动")
        return tasks

    async def stop(self):
        """停止API客户端"""
        self.running = False
        await self.client.aclose()
        logger.info("API客户端已停止")

    async def submit_request(self, request: APIRequest):
        """提交API请求"""
        await self.request_queue.add_request(request)
        logger.debug(f"请求已加入队列: {request.exchange.value} {request.endpoint}")

    async def _worker(self, worker_name: str):
        """工作协程处理请求"""
        logger.info(f"工作协程 {worker_name} 已启动")

        while self.running:
            try:
                # 获取下一个请求
                request = await self.request_queue.get_next_request()
                if not request:
                    await asyncio.sleep(0.1)
                    continue

                # 获取请求槽位
                await self.request_queue.acquire_request_slot()

                # 处理请求
                asyncio.create_task(self._process_request(request, worker_name))

            except Exception as e:
                logger.error(f"工作协程 {worker_name} 错误: {e}")
                await asyncio.sleep(1)

        logger.info(f"工作协程 {worker_name} 已停止")

    async def _process_request(self, request: APIRequest, worker_name: str):
        """处理单个API请求"""
        request_id = f"{request.exchange.value}_{request.endpoint}_{int(time.time())}"

        try:
            # 检查限流
            allowed, reason = await self.rate_limiter.check_rate_limit(request.exchange, request)
            if not allowed:
                logger.warning(f"请求被限流: {request_id} - {reason}")
                await self._handle_rate_limit_exceeded(request, reason)
                return

            # 检查熔断器
            if self._is_circuit_breaker_open(request.exchange):
                logger.warning(f"熔断器开启，拒绝请求: {request_id}")
                await self._handle_circuit_breaker_open(request)
                return

            # 执行API请求
            url = f"{self.rate_limiter.exchange_configs[request.exchange].base_url}{request.endpoint}"

            logger.debug(f"{worker_name} 执行请求: {request_id}")

            response = await self.client.request(
                method=request.method,
                url=url,
                params=request.params,
                json=request.data,
                headers=request.headers
            )

            # 更新熔断器状态
            self._update_circuit_breaker(request.exchange, success=True)

            # 处理响应
            if response.status_code == 200:
                result = response.json()
                logger.info(f"请求成功: {request_id}")
                if request.callback:
                    await request.callback(result, None)
            else:
                error_msg = f"API错误: {response.status_code} - {response.text}"
                logger.error(f"请求失败: {request_id} - {error_msg}")
                await self._handle_api_error(request, error_msg)

        except httpx.TimeoutException:
            error_msg = "请求超时"
            logger.error(f"请求超时: {request_id}")
            await self._handle_timeout(request)
        except Exception as e:
            error_msg = f"未知错误: {str(e)}"
            logger.error(f"请求异常: {request_id} - {error_msg}")
            await self._handle_unknown_error(request, e)
        finally:
            # 释放请求槽位
            self.request_queue.release_request_slot()

    async def _handle_rate_limit_exceeded(self, request: APIRequest, reason: str):
        """处理限流触发"""
        if request.retry_count < request.max_retries:
            # 指数退避重试
            delay = min(2 ** request.retry_count, 30)
            request.retry_count += 1

            logger.info(f"限流重试 {request.retry_count}/{request.max_retries}: {request.endpoint} - 延迟 {delay}s")

            await asyncio.sleep(delay)
            await self.request_queue.add_request(request)
        else:
            # 超过最大重试次数，调用失败回调
            error = Exception(f"限流重试失败: {reason}")
            if request.callback:
                await request.callback(None, error)

    async def _handle_api_error(self, request: APIRequest, error_msg: str):
        """处理API错误"""
        # 检查是否是429错误（限流）
        if "429" in error_msg or "rate limit" in error_msg.lower():
            self._update_circuit_breaker(request.exchange, success=False)
            await self._handle_rate_limit_exceeded(request, error_msg)
        else:
            # 其他API错误
            if request.callback:
                await request.callback(None, Exception(error_msg))

    async def _handle_timeout(self, request: APIRequest):
        """处理超时"""
        self._update_circuit_breaker(request.exchange, success=False)
        await self._handle_rate_limit_exceeded(request, "请求超时")

    async def _handle_unknown_error(self, request: APIRequest, error: Exception):
        """处理未知错误"""
        self._update_circuit_breaker(request.exchange, success=False)
        if request.callback:
            await request.callback(None, error)

    async def _handle_circuit_breaker_open(self, request: APIRequest):
        """处理熔断器开启"""
        if request.callback:
            await request.callback(None, Exception("熔断器开启，服务暂时不可用"))

    def _is_circuit_breaker_open(self, exchange: ExchangeType) -> bool:
        """检查熔断器是否开启"""
        breaker = self.circuit_breakers.get(exchange)
        if not breaker:
            return False

        # 检查是否需要尝试半开状态
        if breaker["state"] == "open":
            if time.time() - breaker["last_failure"] > breaker["timeout"]:
                breaker["state"] = "half_open"
                return False
            return True

        return False

    def _update_circuit_breaker(self, exchange: ExchangeType, success: bool):
        """更新熔断器状态"""
        if exchange not in self.circuit_breakers:
            self.circuit_breakers[exchange] = {
                "state": "closed",  # closed, open, half_open
                "failure_count": 0,
                "last_failure": 0,
                "timeout": 60,  # 熔断器开启时间
                "failure_threshold": 5  # 失败阈值
            }

        breaker = self.circuit_breakers[exchange]

        if success:
            # 成功请求
            if breaker["state"] == "half_open":
                breaker["state"] = "closed"
            breaker["failure_count"] = 0
        else:
            # 失败请求
            breaker["failure_count"] += 1
            breaker["last_failure"] = time.time()

            if breaker["failure_count"] >= breaker["failure_threshold"]:
                breaker["state"] = "open"
                logger.warning(f"熔断器开启: {exchange.value}")

    async def get_system_status(self) -> Dict[str, Any]:
        """获取系统状态"""
        status = {
            "queue_sizes": {
                "high": self.request_queue.high_priority_queue.qsize(),
                "medium": self.request_queue.medium_priority_queue.qsize(),
                "low": self.request_queue.low_priority_queue.qsize(),
            },
            "circuit_breakers": {},
            "rate_limits": {}
        }

        # 熔断器状态
        for exchange, breaker in self.circuit_breakers.items():
            status["circuit_breakers"][exchange.value] = breaker

        # 限流状态
        for exchange in ExchangeType:
            rate_limit_status = await self.rate_limiter.get_rate_limit_status(exchange)
            status["rate_limits"][exchange.value] = rate_limit_status

        return status


# 使用示例
async def example_usage():
    """使用示例"""

    # 初始化Redis客户端
    redis_client = redis.Redis(host='localhost', port=6379, db=0, decode_responses=True)

    # 创建限流器和API客户端
    rate_limiter = ExchangeRateLimiter(redis_client)
    api_client = ExchangeAPIClient(rate_limiter, redis_client)

    # 启动API客户端
    worker_tasks = await api_client.start()

    try:
        # 示例1：提交Binance价格查询请求（中等优先级）
        price_request = APIRequest(
            exchange=ExchangeType.BINANCE,
            endpoint="/api/v3/ticker/price",
            params={"symbol": "BTCUSDT"},
            priority=Priority.MEDIUM,
            callback=lambda data, error: print(f"价格回调: {data or error}")
        )
        await api_client.submit_request(price_request)

        # 示例2：提交Binance下单请求（高优先级）
        order_request = APIRequest(
            exchange=ExchangeType.BINANCE,
            endpoint="/api/v3/order",
            method="POST",
            data={"symbol": "BTCUSDT", "side": "BUY", "type": "MARKET", "quantity": "0.001"},
            priority=Priority.HIGH,
            callback=lambda data, error: print(f"订单回调: {data or error}")
        )
        await api_client.submit_request(order_request)

        # 示例3：提交历史数据查询请求（低优先级）
        history_request = APIRequest(
            exchange=ExchangeType.BINANCE,
            endpoint="/api/v3/klines",
            params={"symbol": "BTCUSDT", "interval": "1h", "limit": 100},
            priority=Priority.LOW,
            callback=lambda data, error: print(f"历史数据回调: {len(data) if data else error}")
        )
        await api_client.submit_request(history_request)

        # 运行一段时间
        await asyncio.sleep(10)

        # 查看系统状态
        system_status = await api_client.get_system_status()
        print(json.dumps(system_status, indent=2, ensure_ascii=False))

    finally:
        # 停止API客户端
        await api_client.stop()

        # 取消工作任务
        for task in worker_tasks:
            task.cancel()

        # 等待任务完成
        await asyncio.gather(*worker_tasks, return_exceptions=True)


if __name__ == "__main__":
    # 运行示例
    asyncio.run(example_usage())
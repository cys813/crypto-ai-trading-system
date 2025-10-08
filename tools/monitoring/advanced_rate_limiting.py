#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
高级限流算法实现
包含令牌桶、滑动窗口、漏桶等算法的Redis实现
"""

import asyncio
import json
import time
import logging
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum
import redis
from datetime import datetime

logger = logging.getLogger(__name__)


class AlgorithmType(Enum):
    """限流算法类型"""
    TOKEN_BUCKET = "token_bucket"
    SLIDING_WINDOW = "sliding_window"
    FIXED_WINDOW = "fixed_window"
    LEAKY_BUCKET = "leaky_bucket"


@dataclass
class RateLimitResult:
    """限流结果"""
    allowed: bool
    remaining: int
    reset_time: int
    retry_after: Optional[int] = None
    algorithm: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class BaseRateLimiter(ABC):
    """限流器基类"""

    def __init__(self, redis_client: redis.Redis):
        self.redis = redis_client

    @abstractmethod
    async def is_allowed(self, key: str, **kwargs) -> RateLimitResult:
        """检查是否允许请求"""
        pass

    @abstractmethod
    async def consume(self, key: str, tokens: int = 1, **kwargs) -> RateLimitResult:
        """消费令牌"""
        pass

    @abstractmethod
    async def get_status(self, key: str) -> Dict[str, Any]:
        """获取当前状态"""
        pass


class TokenBucketLimiter(BaseRateLimiter):
    """令牌桶限流器"""

    def __init__(self, redis_client: redis.Redis):
        super().__init__(redis_client)
        # 令牌桶Lua脚本
        self.script = """
        local bucket_key = KEYS[1]
        local last_update_key = KEYS[2]
        local capacity = tonumber(ARGV[1])
        local tokens_requested = tonumber(ARGV[2])
        local rate = tonumber(ARGV[3])
        local current_time = tonumber(ARGV[4])

        local current_tokens = tonumber(redis.call('GET', bucket_key) or capacity)
        local last_update = tonumber(redis.call('GET', last_update_key) or current_time)

        local time_diff = current_time - last_update
        local tokens_to_add = (time_diff / 1000) * rate
        current_tokens = math.min(capacity, current_tokens + tokens_to_add)

        if current_tokens >= tokens_requested then
            current_tokens = current_tokens - tokens_requested
            redis.call('SET', bucket_key, current_tokens)
            redis.call('SET', last_update_key, current_time)
            local ttl = math.ceil(capacity / rate) + 1
            redis.call('EXPIRE', bucket_key, ttl)
            redis.call('EXPIRE', last_update_key, ttl)
            return {1, tostring(current_tokens), tostring(current_time)}
        else
            redis.call('SET', bucket_key, current_tokens)
            redis.call('SET', last_update_key, current_time)
            local ttl = math.ceil(capacity / rate) + 1
            redis.call('EXPIRE', bucket_key, ttl)
            redis.call('EXPIRE', last_update_key, ttl)

            local retry_after = math.ceil((tokens_requested - current_tokens) * 1000 / rate)
            return {0, tostring(current_tokens), tostring(retry_after)}
        end
        """
        self.script_sha = None

    async def _ensure_script_loaded(self):
        """确保Lua脚本已加载"""
        if not self.script_sha:
            self.script_sha = self.redis.script_load(self.script)

    async def is_allowed(self, key: str, capacity: int, rate: int, **kwargs) -> RateLimitResult:
        """检查是否允许请求"""
        return await self.consume(key, 1, capacity, rate)

    async def consume(self, key: str, tokens: int = 1, capacity: int = 100, rate: int = 10, **kwargs) -> RateLimitResult:
        """消费令牌"""
        await self._ensure_script_loaded()

        bucket_key = f"token_bucket:{key}"
        last_update_key = f"token_bucket:update:{key}"
        current_time = int(time.time() * 1000)

        try:
            result = self.redis.evalsha(
                self.script_sha,
                2,
                bucket_key,
                last_update_key,
                capacity,
                tokens,
                rate,
                current_time
            )

            allowed = bool(result[0])
            remaining = int(float(result[1]))

            if allowed:
                retry_after = None
                reset_time = current_time + (capacity - remaining) * 1000 // rate
            else:
                retry_after = int(float(result[2]))
                reset_time = current_time + retry_after

            return RateLimitResult(
                allowed=allowed,
                remaining=remaining,
                reset_time=reset_time,
                retry_after=retry_after,
                algorithm="token_bucket",
                metadata={
                    "capacity": capacity,
                    "rate": rate,
                    "tokens_requested": tokens
                }
            )

        except Exception as e:
            logger.error(f"令牌桶限流器错误: {e}")
            # 容错处理
            return RateLimitResult(
                allowed=True,
                remaining=capacity,
                reset_time=int(time.time() * 1000) + 60000,
                algorithm="token_bucket_fallback"
            )

    async def get_status(self, key: str) -> Dict[str, Any]:
        """获取当前状态"""
        bucket_key = f"token_bucket:{key}"
        last_update_key = f"token_bucket:update:{key}"

        try:
            current_tokens = self.redis.get(bucket_key)
            last_update = self.redis.get(last_update_key)

            return {
                "current_tokens": float(current_tokens) if current_tokens else 0,
                "last_update": int(last_update) if last_update else int(time.time() * 1000),
                "algorithm": "token_bucket"
            }
        except Exception as e:
            logger.error(f"获取令牌桶状态错误: {e}")
            return {"current_tokens": 0, "last_update": 0, "algorithm": "token_bucket"}


class SlidingWindowLimiter(BaseRateLimiter):
    """滑动窗口限流器"""

    def __init__(self, redis_client: redis.Redis):
        super().__init__(redis_client)
        self.script = """
        local window_key = KEYS[1]
        local window_size = tonumber(ARGV[1])
        local limit = tonumber(ARGV[2])
        local current_time = tonumber(ARGV[3])
        local request_id = ARGV[4]

        local min_score = current_time - (window_size * 1000)
        redis.call('ZREMRANGEBYSCORE', window_key, 0, min_score)

        local current_count = redis.call('ZCARD', window_key)

        if current_count < limit then
            redis.call('ZADD', window_key, current_time, request_id)
            redis.call('EXPIRE', window_key, window_size + 1)
            return {1, tostring(limit - current_count - 1), tostring(current_time + window_size * 1000)}
        else
            local oldest_request = redis.call('ZRANGE', window_key, 0, 0, 'WITHSCORES')
            local reset_time = 0
            if #oldest_request > 0 then
                reset_time = tonumber(oldest_request[2]) + window_size * 1000
            else
                reset_time = current_time + window_size * 1000
            end
            return {0, tostring(0), tostring(reset_time)}
        end
        """
        self.script_sha = None

    async def _ensure_script_loaded(self):
        """确保Lua脚本已加载"""
        if not self.script_sha:
            self.script_sha = self.redis.script_load(self.script)

    async def is_allowed(self, key: str, limit: int, window: int, **kwargs) -> RateLimitResult:
        """检查是否允许请求"""
        return await self.consume(key, 1, limit, window)

    async def consume(self, key: str, tokens: int = 1, limit: int = 100, window: int = 60, **kwargs) -> RateLimitResult:
        """消费令牌"""
        await self._ensure_script_loaded()

        window_key = f"sliding_window:{key}"
        current_time = int(time.time() * 1000)
        request_id = f"{current_time}_{tokens}"

        try:
            result = self.redis.evalsha(
                self.script_sha,
                1,
                window_key,
                window,
                limit,
                current_time,
                request_id
            )

            allowed = bool(result[0])
            remaining = int(float(result[1]))
            reset_time = int(float(result[2]))

            retry_after = (reset_time - current_time) // 1000 if not allowed and reset_time > current_time else None

            return RateLimitResult(
                allowed=allowed,
                remaining=remaining,
                reset_time=reset_time,
                retry_after=retry_after,
                algorithm="sliding_window",
                metadata={
                    "limit": limit,
                    "window": window,
                    "tokens_requested": tokens
                }
            )

        except Exception as e:
            logger.error(f"滑动窗口限流器错误: {e}")
            return RateLimitResult(
                allowed=True,
                remaining=limit,
                reset_time=int(time.time() * 1000) + window * 1000,
                algorithm="sliding_window_fallback"
            )

    async def get_status(self, key: str) -> Dict[str, Any]:
        """获取当前状态"""
        window_key = f"sliding_window:{key}"

        try:
            current_count = self.redis.zcard(window_key)
            window_start = int(time.time() * 1000) - 60000
            recent_count = self.redis.zcount(window_key, window_start, "+inf")

            return {
                "current_count": current_count,
                "recent_count": recent_count,
                "algorithm": "sliding_window"
            }
        except Exception as e:
            logger.error(f"获取滑动窗口状态错误: {e}")
            return {"current_count": 0, "recent_count": 0, "algorithm": "sliding_window"}


class LeakyBucketLimiter(BaseRateLimiter):
    """漏桶限流器"""

    def __init__(self, redis_client: redis.Redis):
        super().__init__(redis_client)
        self.script = """
        local bucket_key = KEYS[1]
        local last_leak_key = KEYS[2]
        local capacity = tonumber(ARGV[1])
        local leak_rate = tonumber(ARGV[2])
        local current_time = tonumber(ARGV[3])

        local current_volume = tonumber(redis.call('GET', bucket_key) or 0)
        local last_leak = tonumber(redis.call('GET', last_leak_key) or current_time)

        local time_diff = current_time - last_leak
        local leaked = (time_diff / 1000) * leak_rate
        current_volume = math.max(0, current_volume - leaked)

        if current_volume < capacity then
            current_volume = current_volume + 1
            redis.call('SET', bucket_key, current_volume)
            redis.call('SET', last_leak_key, current_time)
            local ttl = math.ceil(capacity / leak_rate) + 1
            redis.call('EXPIRE', bucket_key, ttl)
            redis.call('EXPIRE', last_leak_key, ttl)

            local fill_time = math.ceil((capacity - current_volume) * 1000 / leak_rate)
            return {1, tostring(capacity - current_volume), tostring(fill_time)}
        else
            redis.call('SET', bucket_key, current_volume)
            redis.call('SET', last_leak_key, current_time)
            local ttl = math.ceil(capacity / leak_rate) + 1
            redis.call('EXPIRE', bucket_key, ttl)
            redis.call('EXPIRE', last_leak_key, ttl)

            local retry_after = math.ceil((current_volume - capacity + 1) * 1000 / leak_rate)
            return {0, tostring(0), tostring(retry_after)}
        end
        """
        self.script_sha = None

    async def _ensure_script_loaded(self):
        """确保Lua脚本已加载"""
        if not self.script_sha:
            self.script_sha = self.redis.script_load(self.script)

    async def is_allowed(self, key: str, capacity: int, leak_rate: int, **kwargs) -> RateLimitResult:
        """检查是否允许请求"""
        return await self.consume(key, 1, capacity, leak_rate)

    async def consume(self, key: str, tokens: int = 1, capacity: int = 100, leak_rate: int = 10, **kwargs) -> RateLimitResult:
        """消费令牌"""
        await self._ensure_script_loaded()

        bucket_key = f"leaky_bucket:{key}"
        last_leak_key = f"leaky_bucket:leak:{key}"
        current_time = int(time.time() * 1000)

        try:
            result = self.redis.evalsha(
                self.script_sha,
                2,
                bucket_key,
                last_leak_key,
                capacity,
                leak_rate,
                current_time
            )

            allowed = bool(result[0])
            remaining = int(float(result[1]))

            if allowed:
                reset_time = current_time + int(float(result[2]))
                retry_after = None
            else:
                retry_after = int(float(result[2]))
                reset_time = current_time + retry_after

            return RateLimitResult(
                allowed=allowed,
                remaining=remaining,
                reset_time=reset_time,
                retry_after=retry_after,
                algorithm="leaky_bucket",
                metadata={
                    "capacity": capacity,
                    "leak_rate": leak_rate,
                    "tokens_requested": tokens
                }
            )

        except Exception as e:
            logger.error(f"漏桶限流器错误: {e}")
            return RateLimitResult(
                allowed=True,
                remaining=capacity,
                reset_time=int(time.time() * 1000) + 60000,
                algorithm="leaky_bucket_fallback"
            )

    async def get_status(self, key: str) -> Dict[str, Any]:
        """获取当前状态"""
        bucket_key = f"leaky_bucket:{key}"
        last_leak_key = f"leaky_bucket:leak:{key}"

        try:
            current_volume = self.redis.get(bucket_key)
            last_leak = self.redis.get(last_leak_key)

            return {
                "current_volume": float(current_volume) if current_volume else 0,
                "last_leak": int(last_leak) if last_leak else int(time.time() * 1000),
                "algorithm": "leaky_bucket"
            }
        except Exception as e:
            logger.error(f"获取漏桶状态错误: {e}")
            return {"current_volume": 0, "last_leak": 0, "algorithm": "leaky_bucket"}


class MultiDimensionLimiter:
    """多维度限流器"""

    def __init__(self, redis_client: redis.Redis):
        self.redis = redis_client
        self.limiters = {
            AlgorithmType.TOKEN_BUCKET: TokenBucketLimiter(redis_client),
            AlgorithmType.SLIDING_WINDOW: SlidingWindowLimiter(redis_client),
            AlgorithmType.LEAKY_BUCKET: LeakyBucketLimiter(redis_client),
        }

    async def check_multiple_limits(
        self,
        key: str,
        rules: List[Dict[str, Any]]
    ) -> RateLimitResult:
        """检查多个限流规则"""
        results = []
        overall_allowed = True
        min_remaining = float('inf')
        max_reset_time = 0
        max_retry_after = 0

        for rule in rules:
            algorithm_type = AlgorithmType(rule['type'])
            limiter = self.limiters[algorithm_type]

            params = {k: v for k, v in rule.items() if k != 'type'}
            result = await limiter.is_allowed(key, **params)

            results.append(result)

            if not result.allowed:
                overall_allowed = False
                if result.retry_after and result.retry_after > max_retry_after:
                    max_retry_after = result.retry_after

            if result.remaining < min_remaining:
                min_remaining = result.remaining

            if result.reset_time > max_reset_time:
                max_reset_time = result.reset_time

        return RateLimitResult(
            allowed=overall_allowed,
            remaining=min_remaining if min_remaining != float('inf') else 0,
            reset_time=max_reset_time,
            retry_after=max_retry_after if max_retry_after > 0 else None,
            algorithm="multi_dimension",
            metadata={
                "individual_results": [r.__dict__ for r in results],
                "rules_count": len(rules)
            }
        )


class ExchangeRateLimitManager:
    """交易所限流管理器"""

    def __init__(self, redis_client: redis.Redis):
        self.redis = redis_client
        self.multi_limiter = MultiDimensionLimiter(redis_client)

        # 预定义的交易所限流规则
        self.exchange_rules = {
            "binance": [
                {"type": "token_bucket", "capacity": 6000, "rate": 100},  # 6000 tokens, 100 tokens/sec
                {"type": "sliding_window", "limit": 10000, "window": 10},  # 10000 requests per 10 seconds
                {"type": "sliding_window", "limit": 100000, "window": 86400},  # 100000 requests per day
            ],
            "coinbase": [
                {"type": "token_bucket", "capacity": 600, "rate": 10},  # 600 tokens, 10 tokens/sec
                {"type": "sliding_window", "limit": 50, "window": 10},  # 50 requests per 10 seconds
            ],
            "kraken": [
                {"type": "leaky_bucket", "capacity": 100, "leak_rate": 15},  # 100 capacity, 15/sec leak rate
                {"type": "sliding_window", "limit": 20, "window": 60},  # 20 requests per minute for private endpoints
            ],
            "huobi": [
                {"type": "token_bucket", "capacity": 18000, "rate": 300},  # 18000 tokens, 300 tokens/sec
                {"type": "sliding_window", "limit": 100, "window": 1},  # 100 requests per second
            ],
            "okex": [
                {"type": "sliding_window", "limit": 60, "window": 1},  # 60 requests per second
                {"type": "leaky_bucket", "capacity": 600, "leak_rate": 60},  # 600 capacity, 60/sec leak rate
            ]
        }

    async def check_request(
        self,
        exchange: str,
        endpoint_type: str = "public",
        ip_address: str = "default",
        user_id: str = "default"
    ) -> RateLimitResult:
        """检查请求是否允许"""
        # 构建复合键
        key = f"{exchange}:{endpoint_type}:{ip_address}:{user_id}"

        # 获取交易所规则
        rules = self.exchange_rules.get(exchange, [])

        # 检查多维度限流
        result = await self.multi_limiter.check_multiple_limits(key, rules)

        # 记录请求日志
        await self._log_request(exchange, key, result)

        return result

    async def _log_request(self, exchange: str, key: str, result: RateLimitResult):
        """记录请求日志"""
        try:
            log_key = f"rate_limit_log:{exchange}:{int(time.time() // 60)}"  # 按分钟聚合
            log_data = {
                "timestamp": int(time.time()),
                "key": key,
                "allowed": result.allowed,
                "remaining": result.remaining,
                "algorithm": result.algorithm
            }

            # 使用Redis List存储日志（最近1000条）
            self.redis.lpush(log_key, json.dumps(log_data))
            self.redis.ltrim(log_key, 0, 999)
            self.redis.expire(log_key, 3600)  # 1小时过期

        except Exception as e:
            logger.error(f"记录限流日志错误: {e}")

    async def get_exchange_status(self, exchange: str) -> Dict[str, Any]:
        """获取交易所限流状态"""
        status = {
            "exchange": exchange,
            "timestamp": int(time.time()),
            "rules": [],
            "statistics": {}
        }

        rules = self.exchange_rules.get(exchange, [])
        key = f"{exchange}:public:default:default"

        for i, rule in enumerate(rules):
            algorithm_type = AlgorithmType(rule['type'])
            limiter = self.multi_limiter.limiters[algorithm_type]

            rule_key = f"{key}:rule_{i}"
            rule_status = await limiter.get_status(rule_key)

            status["rules"].append({
                "rule_index": i,
                "type": rule['type'],
                "params": {k: v for k, v in rule.items() if k != 'type'},
                "status": rule_status
            })

        # 获取统计信息
        stats_key = f"rate_limit_stats:{exchange}"
        stats = self.redis.hgetall(stats_key)
        status["statistics"] = {k: int(v) for k, v in stats.items()} if stats else {}

        return status

    async def update_rules(self, exchange: str, new_rules: List[Dict[str, Any]]):
        """更新交易所限流规则"""
        self.exchange_rules[exchange] = new_rules

        # 记录规则变更
        change_log = {
            "timestamp": int(time.time()),
            "exchange": exchange,
            "old_rules": self.exchange_rules.get(exchange, []),
            "new_rules": new_rules
        }

        log_key = f"rule_changes:{exchange}"
        self.redis.lpush(log_key, json.dumps(change_log))
        self.redis.expire(log_key, 86400 * 7)  # 保存7天

    async def get_real_time_metrics(self, exchange: str) -> Dict[str, Any]:
        """获取实时指标"""
        current_minute = int(time.time() // 60)

        # 获取最近5分钟的请求统计
        metrics = {
            "exchange": exchange,
            "timestamp": int(time.time()),
            "recent_requests": {},
            "rate_limit_violations": {},
            "success_rate": 0.0
        }

        total_requests = 0
        total_violations = 0

        for i in range(5):  # 最近5分钟
            minute_key = f"rate_limit_log:{exchange}:{current_minute - i}"
            logs = self.redis.lrange(minute_key, 0, -1)

            requests = 0
            violations = 0

            for log_json in logs:
                try:
                    log = json.loads(log_json)
                    requests += 1
                    if not log.get("allowed", True):
                        violations += 1
                except:
                    continue

            minute_timestamp = (current_minute - i) * 60
            metrics["recent_requests"][minute_timestamp] = requests
            metrics["rate_limit_violations"][minute_timestamp] = violations

            total_requests += requests
            total_violations += violations

        if total_requests > 0:
            metrics["success_rate"] = (total_requests - total_violations) / total_requests

        return metrics


# 使用示例
async def example_advanced_rate_limiting():
    """高级限流使用示例"""

    # 初始化Redis客户端
    redis_client = redis.Redis(host='localhost', port=6379, db=0, decode_responses=True)

    # 创建限流管理器
    rate_manager = ExchangeRateLimitManager(redis_client)

    # 测试Binance请求
    print("=== 测试Binance请求限流 ===")

    for i in range(10):
        result = await rate_manager.check_request("binance", "public")
        status = "✅ 通过" if result.allowed else "❌ 被限流"
        print(f"请求 {i+1}: {status} - 剩余: {result.remaining} - 算法: {result.algorithm}")

        if not result.allowed and result.retry_after:
            print(f"  重试时间: {result.retry_after}秒")

    # 获取交易所状态
    print("\n=== Binance限流状态 ===")
    status = await rate_manager.get_exchange_status("binance")
    print(json.dumps(status, indent=2, ensure_ascii=False))

    # 获取实时指标
    print("\n=== 实时指标 ===")
    metrics = await rate_manager.get_real_time_metrics("binance")
    print(json.dumps(metrics, indent=2, ensure_ascii=False))

    # 测试多交易所
    print("\n=== 多交易所并发测试 ===")

    exchanges = ["binance", "coinbase", "kraken", "huobi", "okex"]
    tasks = []

    for exchange in exchanges:
        for i in range(5):
            task = rate_manager.check_request(exchange, "public")
            tasks.append((exchange, i+1, task))

    for exchange, req_num, task in tasks:
        result = await task
        status = "✅ 通过" if result.allowed else "❌ 被限流"
        print(f"{exchange} 请求 {req_num}: {status} - 剩余: {result.remaining}")


if __name__ == "__main__":
    # 运行示例
    asyncio.run(example_advanced_rate_limiting())
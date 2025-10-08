"""
Redis缓存和连接管理

提供Redis连接池和缓存功能的基础设施。
"""

import json
import logging
from typing import Any, Optional, Union
import redis.asyncio as redis
from redis.asyncio import ConnectionPool
import aioredis
from datetime import timedelta

logger = logging.getLogger(__name__)


class RedisManager:
    """Redis连接管理器"""

    def __init__(self, redis_url: str):
        self.redis_url = redis_url
        self.pool: Optional[ConnectionPool] = None
        self.client: Optional[redis.Redis] = None

    async def connect(self) -> None:
        """建立Redis连接"""
        try:
            self.pool = ConnectionPool.from_url(
                self.redis_url,
                encoding="utf-8",
                decode_responses=True,
                max_connections=20,
                retry_on_timeout=True,
                socket_keepalive=True,
                socket_keepalive_options={},
                health_check_interval=30
            )

            self.client = redis.Redis(connection_pool=self.pool)

            # 测试连接
            await self.client.ping()
            logger.info("Redis连接成功建立")

        except Exception as e:
            logger.error(f"Redis连接失败: {e}")
            raise

    async def disconnect(self) -> None:
        """关闭Redis连接"""
        if self.client:
            await self.client.close()
        if self.pool:
            await self.pool.disconnect()
        logger.info("Redis连接已关闭")

    async def is_connected(self) -> bool:
        """检查Redis连接状态"""
        try:
            if self.client:
                await self.client.ping()
                return True
        except Exception:
            pass
        return False


class CacheService:
    """缓存服务"""

    def __init__(self, redis_manager: RedisManager):
        self.redis = redis_manager
        self.default_ttl = 3600  # 1小时

    async def get(self, key: str) -> Optional[Any]:
        """获取缓存值"""
        try:
            if not await self.redis.is_connected():
                return None

            value = await self.redis.client.get(key)
            if value:
                # 尝试反序列化JSON
                try:
                    return json.loads(value)
                except (json.JSONDecodeError, TypeError):
                    return value
            return None
        except Exception as e:
            logger.error(f"缓存获取失败 {key}: {e}")
            return None

    async def set(
        self,
        key: str,
        value: Any,
        ttl: Optional[int] = None,
        serialize: bool = True
    ) -> bool:
        """设置缓存值"""
        try:
            if not await self.redis.is_connected():
                return False

            ttl = ttl or self.default_ttl

            # 序列化值
            if serialize and not isinstance(value, (str, int, float)):
                value = json.dumps(value, default=str)

            result = await self.redis.client.setex(key, ttl, value)
            return bool(result)
        except Exception as e:
            logger.error(f"缓存设置失败 {key}: {e}")
            return False

    async def delete(self, key: str) -> bool:
        """删除缓存"""
        try:
            if not await self.redis.is_connected():
                return False

            result = await self.redis.client.delete(key)
            return bool(result)
        except Exception as e:
            logger.error(f"缓存删除失败 {key}: {e}")
            return False

    async def exists(self, key: str) -> bool:
        """检查缓存是否存在"""
        try:
            if not await self.redis.is_connected():
                return False

            result = await self.redis.client.exists(key)
            return bool(result)
        except Exception as e:
            logger.error(f"缓存检查失败 {key}: {e}")
            return False

    async def increment(self, key: str, amount: int = 1) -> Optional[int]:
        """递增计数器"""
        try:
            if not await self.redis.is_connected():
                return None

            result = await self.redis.client.incrby(key, amount)
            return result
        except Exception as e:
            logger.error(f"计数器递增失败 {key}: {e}")
            return None

    async def expire(self, key: str, ttl: int) -> bool:
        """设置过期时间"""
        try:
            if not await self.redis.is_connected():
                return False

            result = await self.redis.client.expire(key, ttl)
            return bool(result)
        except Exception as e:
            logger.error(f"设置过期时间失败 {key}: {e}")
            return False

    async def get_keys(self, pattern: str = "*") -> list[str]:
        """获取匹配模式的所有键"""
        try:
            if not await self.redis.is_connected():
                return []

            keys = await self.redis.client.keys(pattern)
            return keys
        except Exception as e:
            logger.error(f"获取键列表失败 {pattern}: {e}")
            return []

    async def flush_all(self) -> bool:
        """清空所有缓存（谨慎使用）"""
        try:
            if not await self.redis.is_connected():
                return False

            result = await self.redis.client.flushdb()
            return bool(result)
        except Exception as e:
            logger.error(f"清空缓存失败: {e}")
            return False


# 全局Redis管理器实例
redis_manager: Optional[RedisManager] = None
cache_service: Optional[CacheService] = None


async def init_redis(redis_url: str) -> None:
    """初始化Redis连接"""
    global redis_manager, cache_service

    redis_manager = RedisManager(redis_url)
    await redis_manager.connect()

    cache_service = CacheService(redis_manager)
    logger.info("Redis缓存服务初始化完成")


async def close_redis() -> None:
    """关闭Redis连接"""
    global redis_manager

    if redis_manager:
        await redis_manager.disconnect()


def get_cache() -> CacheService:
    """获取缓存服务实例"""
    if cache_service is None:
        raise RuntimeError("Redis缓存服务未初始化，请先调用init_redis()")
    return cache_service


# 缓存键前缀常量
class CacheKeys:
    """缓存键常量"""

    # 市场数据
    KLINE_DATA = "kline:{symbol}:{exchange}:{timeframe}"
    TECHNICAL_ANALYSIS = "ta:{symbol}:{exchange}:{timeframe}"

    # 新闻数据
    NEWS_SUMMARY = "news:summary:{period}"
    NEWS_DATA = "news:data:{source}:{date}"

    # 交易数据
    TRADING_STRATEGY = "strategy:{symbol}:{type}"
    OPEN_ORDERS = "orders:open:{user}"
    POSITION_DATA = "position:{user}:{symbol}"

    # 限流
    RATE_LIMIT = "ratelimit:{service}:{identifier}"
    API_QUOTA = "quota:{exchange}:{endpoint}"

    # 会话
    USER_SESSION = "session:{token}"
    LOCK = "lock:{resource}"
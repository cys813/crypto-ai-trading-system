"""
核心模块

包含缓存、配置、数据库连接等核心基础设施。
"""

from .cache import RedisManager, CacheService, init_redis, close_redis, get_cache, CacheKeys

__all__ = [
    "RedisManager",
    "CacheService",
    "init_redis",
    "close_redis",
    "get_cache",
    "CacheKeys",
]
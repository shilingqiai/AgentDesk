"""
缓存服务

支持 Redis 缓存（生产环境）和内存缓存（开发环境）
用于知识库检索结果缓存和会话状态缓存
"""

import json
import hashlib
from typing import Optional, Any
from config.settings import settings


class CacheService:
    """缓存服务，支持 Redis 和内存两种后端"""

    def __init__(self):
        self._redis = None
        self._memory_cache: dict = {}
        self._use_redis = bool(settings.redis_url)

    @property
    def redis(self):
        """懒加载 Redis 连接"""
        if self._redis is None and self._use_redis:
            try:
                import redis
                self._redis = redis.from_url(settings.redis_url)
                self._redis.ping()
            except Exception:
                self._use_redis = False
        return self._redis

    def _make_key(self, prefix: str, data: Any) -> str:
        """生成缓存键"""
        raw = json.dumps(data, sort_keys=True, default=str)
        return f"ticket:{prefix}:{hashlib.md5(raw.encode()).hexdigest()}"

    def get(self, prefix: str, key_data: Any) -> Optional[Any]:
        """获取缓存"""
        key = self._make_key(prefix, key_data)
        if self._use_redis and self.redis:
            try:
                value = self.redis.get(key)
                return json.loads(value) if value else None
            except Exception:
                pass

        return self._memory_cache.get(key)

    def set(self, prefix: str, key_data: Any, value: Any, ttl: int = 3600):
        """设置缓存"""
        key = self._make_key(prefix, key_data)
        if self._use_redis and self.redis:
            try:
                self.redis.setex(key, ttl, json.dumps(value, default=str))
                return
            except Exception:
                pass

        self._memory_cache[key] = value

    def invalidate(self, prefix: str):
        """使指定前缀的缓存失效"""
        if self._use_redis and self.redis:
            try:
                for k in self.redis.scan_iter(f"ticket:{prefix}:*"):
                    self.redis.delete(k)
                return
            except Exception:
                pass

        keys_to_delete = [
            k for k in self._memory_cache
            if k.startswith(f"ticket:{prefix}:")
        ]
        for k in keys_to_delete:
            del self._memory_cache[k]


# 全局缓存服务实例
cache_service = CacheService()

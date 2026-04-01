import os
from typing import Any

from redis.asyncio import ConnectionPool, Redis

_redis_pool: ConnectionPool | None = None
_redis_client: Redis | None = None


def get_redis_url() -> str:
    from apps.api.core.config import get_settings

    return get_settings().redis_url


async def create_redis_pool(url: str | None = None, **kwargs) -> ConnectionPool:
    global _redis_pool
    if _redis_pool is None:
        url = url or get_redis_url()
        _redis_pool = ConnectionPool.from_url(
            url, max_connections=50, decode_responses=True, **kwargs
        )
    return _redis_pool


async def get_redis() -> Redis:
    global _redis_client
    if _redis_client is None:
        pool = await create_redis_pool()
        _redis_client = Redis(connection_pool=pool)
    return _redis_client


async def close_redis():
    global _redis_client, _redis_pool
    if _redis_client:
        await _redis_client.close()
        _redis_client = None
    if _redis_pool:
        await _redis_pool.disconnect()
        _redis_pool = None


class RedisCache:
    def __init__(self, client: Redis):
        self.client = client

    async def get(self, key: str) -> str | None:
        return await self.client.get(key)

    async def set(
        self,
        key: str,
        value: str,
        ex: int | None = None,
        px: int | None = None,
    ) -> bool:
        return await self.client.set(key, value, ex=ex, px=px)

    async def delete(self, key: str) -> int:
        return await self.client.delete(key)

    async def exists(self, key: str) -> bool:
        return await self.client.exists(key) > 0

    async def incr(self, key: str) -> int:
        return await self.client.incr(key)

    async def expire(self, key: str, seconds: int) -> bool:
        return await self.client.expire(key, seconds)

    async def hset(self, name: str, mapping: dict[str, Any]) -> int:
        return await self.client.hset(name, mapping=mapping)

    async def hget(self, name: str, key: str) -> str | None:
        return await self.client.hget(name, key)

    async def hgetall(self, name: str) -> dict[str, str]:
        return await self.client.hgetall(name)

    async def hincrby(self, name: str, key: str, amount: int = 1) -> int:
        return await self.client.hincrby(name, key, amount)

    async def expire_at(self, key: str, timestamp: int) -> bool:
        return await self.client.expireat(key, timestamp)

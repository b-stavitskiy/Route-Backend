import json
from typing import Any

from redis.asyncio import Redis
from packages.redis.client import RedisCache


class CacheService:
    def __init__(self, redis: Redis):
        self.redis = redis
        self.cache = RedisCache(redis)

    async def get_cached_models(self) -> list[dict[str, Any]] | None:
        data = await self.cache.get("models:all")
        if data:
            return json.loads(data)
        return None

    async def set_cached_models(self, models: list[dict[str, Any]], ttl: int = 3600):
        await self.cache.set("models:all", json.dumps(models), ex=ttl)

    async def get_provider_health(self, provider: str) -> dict[str, Any] | None:
        data = await self.cache.get(f"provider:{provider}:health")
        if data:
            return json.loads(data)
        return None

    async def set_provider_health(
        self,
        provider: str,
        health: dict[str, Any],
        ttl: int = 30,
    ):
        await self.cache.set(f"provider:{provider}:health", json.dumps(health), ex=ttl)

    async def get_session(self, user_id: str) -> dict[str, Any] | None:
        data = await self.cache.get(f"session:{user_id}")
        if data:
            return json.loads(data)
        return None

    async def set_session(
        self,
        user_id: str,
        session_data: dict[str, Any],
        ttl: int = 86400,
    ):
        await self.cache.set(f"session:{user_id}", json.dumps(session_data), ex=ttl)

    async def delete_session(self, user_id: str):
        await self.cache.delete(f"session:{user_id}")

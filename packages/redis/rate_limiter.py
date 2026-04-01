import time

from packages.redis.client import RedisCache

from redis.asyncio import Redis


class RateLimiter:
    def __init__(self, redis: Redis):
        self.redis = redis
        self.cache = RedisCache(redis)

    async def check_rate_limit(
        self,
        key: str,
        limit: int,
        window_seconds: int = 3600,
        burst: int = 0,
    ) -> tuple[bool, int, int]:
        current_time = int(time.time())
        window_start = current_time - window_seconds

        pipe = self.redis.pipeline()
        pipe.zremrangebyscore(key, 0, window_start)
        pipe.zcard(key)
        pipe.zadd(key, {str(current_time): current_time})
        pipe.expire(key, window_seconds)
        results = await pipe.execute()

        request_count = results[1]
        remaining = max(0, limit - request_count - 1)

        if burst > 0 and request_count < burst:
            return True, remaining, 0

        if request_count >= limit:
            oldest_timestamp = await self.redis.zrange(key, 0, 0, withscores=True)
            if oldest_timestamp:
                retry_after = int(oldest_timestamp[0][1]) + window_seconds - current_time
                return False, 0, max(0, retry_after)
            return False, 0, window_seconds

        return True, remaining, 0

    async def sliding_window_log(
        self,
        key: str,
        window_seconds: int = 3600,
    ) -> int:
        current_time = int(time.time())
        window_start = current_time - window_seconds

        pipe = self.redis.pipeline()
        pipe.zremrangebyscore(key, 0, window_start)
        pipe.zcard(key)
        results = await pipe.execute()

        return results[1]

    async def get_usage(
        self,
        key: str,
        window_seconds: int = 3600,
    ) -> int:
        current_time = int(time.time())
        window_start = current_time - window_seconds

        await self.redis.zremrangebyscore(key, 0, window_start)
        return await self.redis.zcard(key)

import time
import logging
from redis.asyncio import Redis
from apps.api.core.config import get_provider_config
from packages.redis.client import RedisCache
from packages.shared.exceptions import DailyRequestLimitError

logger = logging.getLogger("routing.run.api")


class RequestManager:
    DAILY_KEY_PREFIX = "requests:daily"
    KEY_EXPIRY = 86400

    def __init__(self, redis: Redis):
        self.redis = redis
        self.cache = RedisCache(redis)
        self.provider_config = get_provider_config()

    def _get_daily_key(self, user_id: str) -> str:
        date = time.strftime("%Y-%m-%d")
        return f"{self.DAILY_KEY_PREFIX}:{user_id}:{date}"

    async def get_daily_request_count(self, user_id: str) -> int:
        daily_key = self._get_daily_key(user_id)
        count = await self.cache.hget(daily_key, "total_requests")
        return int(count) if count else 0

    async def check_daily_limit(self, user_id: str, plan_tier: str) -> None:
        plan_config = self.provider_config.get_plan_config(plan_tier)
        logger.info(
            f"check_daily_limit: user_id={user_id}, plan_tier={plan_tier}, plan_config={plan_config}"
        )
        if not plan_config:
            logger.error(f"check_daily_limit: plan_config is None for plan_tier={plan_tier}")
            raise DailyRequestLimitError(limit=0, used=0)

        daily_limit = plan_config.get("requests_per_day", 0)
        logger.info(f"check_daily_limit: daily_limit={daily_limit}")
        if daily_limit == 0:
            logger.error(f"check_daily_limit: daily_limit is 0 for plan_tier={plan_tier}")
            raise DailyRequestLimitError(limit=0, used=0)

        current_usage = await self.get_daily_request_count(user_id)
        logger.info(f"check_daily_limit: current_usage={current_usage}, limit={daily_limit}")

        if current_usage >= daily_limit:
            logger.error(
                f"check_daily_limit: LIMIT EXCEEDED - user_id={user_id}, current={current_usage}, limit={daily_limit}"
            )
            raise DailyRequestLimitError(
                limit=daily_limit,
                used=current_usage,
            )

    async def increment_request_count(self, user_id: str) -> int:
        daily_key = self._get_daily_key(user_id)
        pipe = self.redis.pipeline()
        pipe.hincrby(daily_key, "total_requests", 1)
        pipe.expire(daily_key, self.KEY_EXPIRY)
        results = await pipe.execute()
        return int(results[0])

    async def get_remaining_requests(self, user_id: str, plan_tier: str) -> dict[str, int]:
        plan_config = self.provider_config.get_plan_config(plan_tier)
        if not plan_config:
            return {"limit": 0, "used": 0, "remaining": 0}

        daily_limit = plan_config.get("requests_per_day", 0)
        current_usage = await self.get_daily_request_count(user_id)
        remaining = max(0, daily_limit - current_usage)

        return {
            "limit": daily_limit,
            "used": current_usage,
            "remaining": remaining,
        }


def get_request_manager(redis: Redis) -> RequestManager:
    return RequestManager(redis)

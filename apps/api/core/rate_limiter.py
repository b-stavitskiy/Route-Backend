import time
import logging

from redis.asyncio import Redis

from apps.api.core.config import get_provider_config
from packages.shared.exceptions import (
    DailyRequestLimitError,
    ModelNotAllowedError,
)

logger = logging.getLogger("routing.run.api")


async def check_model_access(
    user_plan: str,
    model: str,
) -> None:
    provider_config = get_provider_config()
    provider_config._load_config()

    if not provider_config.is_model_allowed(model, user_plan):
        raise ModelNotAllowedError(model, user_plan)


async def check_rate_limit(
    redis: Redis,
    user_plan: str,
    model: str,
    key_hash: str,
) -> None:
    provider_config = get_provider_config()
    provider_config._load_config()

    plan_config = provider_config.get_plan_config(user_plan)
    if not plan_config:
        raise DailyRequestLimitError(limit=0, used=0)

    daily_limit = plan_config.get("requests_per_day", 0)
    if daily_limit == 0:
        raise DailyRequestLimitError(limit=0, used=0)

    date = time.strftime("%Y-%m-%d")
    daily_key = f"ratelimit:daily:{key_hash}:{date}"

    pipe = redis.pipeline()
    pipe.incr(daily_key)
    pipe.expire(daily_key, 86400)
    results = await pipe.execute()

    current_count = results[0]
    if current_count > daily_limit:
        raise DailyRequestLimitError(
            limit=daily_limit,
            used=current_count - 1,
        )

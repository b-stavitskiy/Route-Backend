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
    logger.info(f"check_rate_limit: user_plan={user_plan}, plan_config={plan_config}")
    if not plan_config:
        logger.error(f"check_rate_limit: plan_config is None for user_plan={user_plan}")
        raise DailyRequestLimitError(limit=0, used=0)

    daily_limit = plan_config.get("requests_per_day", 0)
    logger.info(f"check_rate_limit: daily_limit={daily_limit}")
    if daily_limit == 0:
        logger.error(f"check_rate_limit: daily_limit is 0 for user_plan={user_plan}")
        raise DailyRequestLimitError(limit=0, used=0)

    date = time.strftime("%Y-%m-%d")
    daily_key = f"ratelimit:daily:{key_hash}:{date}"

    pipe = redis.pipeline()
    pipe.incr(daily_key)
    pipe.expire(daily_key, 86400)
    results = await pipe.execute()

    current_count = results[0]
    logger.info(
        f"check_rate_limit: INCREMENTED counter to {current_count}, checking {current_count} > {daily_limit}"
    )
    if current_count > daily_limit:
        ttl = await redis.ttl(daily_key)
        logger.error(f"check_rate_limit: RAISING ERROR - {current_count} > {daily_limit}")
        raise DailyRequestLimitError(
            limit=daily_limit,
            used=current_count - 1,
        )
    logger.info(f"check_rate_limit: PASSED - {current_count} <= {daily_limit}")

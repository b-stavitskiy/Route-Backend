from redis.asyncio import Redis
from apps.api.core.config import get_provider_config
from packages.shared.exceptions import (
    ModelNotAllowedError,
)


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
    pass

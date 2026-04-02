from redis.asyncio import Redis
from apps.api.core.config import get_provider_config, get_settings
from packages.redis.rate_limiter import RateLimiter
from packages.shared.exceptions import (
    ModelNotAllowedError,
    RateLimitError,
)


class PlanRateLimiter:
    WINDOW_SECONDS = 3600

    def __init__(self, redis: Redis):
        self.redis = redis
        self.rate_limiter = RateLimiter(redis)
        self.provider_config = get_provider_config()
        self.settings = get_settings()

    async def check_rate_limit(
        self,
        user_plan: str,
        model: str,
        key_hash: str,
    ) -> None:
        is_lite = self._is_lite_model(model)
        tier = "lite" if is_lite else "premium"

        plan_limits = self._get_plan_limits(user_plan)
        limit = plan_limits.get(tier, 0)
        burst = plan_limits.get(f"{tier}_burst", 0)

        if limit == 0:
            if tier == "premium":
                raise ModelNotAllowedError(model, user_plan)
            raise RateLimitError("Rate limit exceeded for your plan")

        key = f"{self.settings.api_key_prefix}ratelimit:{key_hash}"
        allowed, remaining, retry_after = await self.rate_limiter.check_rate_limit(
            key=key,
            limit=limit,
            window_seconds=self.WINDOW_SECONDS,
            burst=burst,
        )

        if not allowed:
            raise RateLimitError(
                message="Rate limit exceeded",
                retry_after=retry_after,
            )

    def _is_lite_model(self, model: str) -> bool:
        model_config = self.provider_config.get_model_config(model)
        if not model_config:
            return False

        models = self.provider_config._config.get("providers", {}).get("models", {})
        return model in models.get("lite", {})

    def _get_plan_limits(self, plan: str) -> dict[str, int]:
        plan_config = self.provider_config.get_plan_config(plan)
        if not plan_config:
            return {"lite": 0, "premium": 0, "lite_burst": 0, "premium_burst": 0}

        rate_limits = plan_config.get("rate_limits", {})

        # Handle simple structure: rate_limits has requests_per_hour and burst at top level
        if "requests_per_hour" in rate_limits:
            requests_per_hour = rate_limits.get("requests_per_hour", 0)
            burst = rate_limits.get("burst", 0)
            return {
                "lite": requests_per_hour,
                "premium": requests_per_hour,
                "lite_burst": burst,
                "premium_burst": burst,
            }

        # Handle nested structure
        lite_config = rate_limits.get("lite", {})
        premium_config = rate_limits.get("premium", {})

        return {
            "lite": lite_config.get("requests_per_hour", 0),
            "premium": premium_config.get("requests_per_hour", 0),
            "lite_burst": lite_config.get("burst", 0),
            "premium_burst": premium_config.get("burst", 0),
        }

    def _get_burst_limits(self, plan: str) -> dict[str, int]:
        burst_map = {
            "free": {"lite": 5, "premium": 0},
            "lite": {"lite": 10, "premium": 0},
            "pro": {"lite": 15, "premium": 0},
            "max": {"lite": 30, "premium": 15},
        }
        return burst_map.get(plan, {"lite": 0, "premium": 0})


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
    limiter = PlanRateLimiter(redis)
    await limiter.check_rate_limit(user_plan, model, key_hash)

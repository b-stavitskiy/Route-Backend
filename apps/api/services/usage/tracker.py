import time
from typing import Any

from redis.asyncio import Redis
from apps.api.core.config import get_provider_config, get_settings
from apps.api.services.credit import get_credit_service
from packages.db.models import User
from packages.db.session import get_db_session
from packages.redis.client import RedisCache
from packages.shared.exceptions import InsufficientCreditsError


class CreditManager:
    def __init__(self, redis: Redis):
        self.redis = redis
        self.cache = RedisCache(redis)
        self.settings = get_settings()
        self.credit_service = get_credit_service()

    async def get_user_credits(self, user_id: str) -> float:
        async with get_db_session() as session:
            from sqlalchemy import select

            result = await session.execute(select(User).where(User.id == user_id))
            user = result.scalar_one_or_none()
            if not user:
                return 0.0
            return user.credits

    async def check_credits_for_request(
        self,
        user_id: str,
        model: str,
        max_tokens: int | None,
    ) -> float:
        estimated_cost = self.estimate_request_cost(model, max_tokens)

        user_credits = await self.get_user_credits(user_id)
        if user_credits < estimated_cost:
            raise InsufficientCreditsError(
                required=estimated_cost,
                available=user_credits,
            )
        return estimated_cost

    async def deduct_credits(
        self,
        user_id: str,
        model: str,
        input_tokens: int,
        output_tokens: int,
    ) -> float:
        actual_cost = self.credit_service.calculate_request_cost(
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        )

        if actual_cost <= 0:
            return 0.0

        async with get_db_session() as session:
            from sqlalchemy import select

            result = await session.execute(select(User).where(User.id == user_id))
            user = result.scalar_one_or_none()
            if not user:
                return 0.0

            new_balance = max(0.0, user.credits - actual_cost)
            user.credits = new_balance
            await session.commit()

        await self._track_credit_usage(
            user_id=user_id,
            model=model,
            cost=actual_cost,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        )

        return actual_cost

    async def deduct_credits_for_image(
        self,
        user_id: str,
        model: str,
    ) -> float:
        pricing = self.credit_service.get_model_pricing(model)
        if not pricing:
            return 0.0

        actual_cost = pricing.get("output_per_million", 0.05)

        if actual_cost <= 0:
            return 0.0

        async with get_db_session() as session:
            from sqlalchemy import select

            result = await session.execute(select(User).where(User.id == user_id))
            user = result.scalar_one_or_none()
            if not user:
                return 0.0

            new_balance = max(0.0, user.credits - actual_cost)
            user.credits = new_balance
            await session.commit()

        await self._track_credit_usage(
            user_id=user_id,
            model=model,
            cost=actual_cost,
            input_tokens=0,
            output_tokens=0,
        )

        return actual_cost

    async def add_credits(self, user_id: str, amount: float) -> float:
        async with get_db_session() as session:
            from sqlalchemy import select

            result = await session.execute(select(User).where(User.id == user_id))
            user = result.scalar_one_or_none()
            if not user:
                return 0.0

            user.credits = user.credits + amount
            await session.commit()
            return user.credits

    async def _track_credit_usage(
        self,
        user_id: str,
        model: str,
        cost: float,
        input_tokens: int,
        output_tokens: int,
    ) -> None:
        daily_key = f"credits:daily:{user_id}:{time.strftime('%Y-%m-%d')}"
        await self.cache.hincrby(daily_key, f"{model}:cost", int(cost * 10000))
        await self.cache.hincrby(daily_key, f"{model}:input_tokens", input_tokens)
        await self.cache.hincrby(daily_key, f"{model}:output_tokens", output_tokens)
        await self.cache.expire(daily_key, 172800)

    async def get_monthly_credits_used(self, user_id: str, month: str | None = None) -> float:
        if month is None:
            month = time.strftime("%Y-%m")

        total_cost = 0.0
        cursor = 0
        keys: list[str] = []

        while True:
            cursor, batch = await self.redis.scan(
                cursor=cursor,
                match=f"credits:daily:{user_id}:{month}*",
                count=100,
            )
            keys.extend(batch)
            if cursor == 0:
                break

        for key in keys:
            usage = await self.cache.hgetall(key)
            for key_name, value in usage.items():
                parts = key_name.split(":")
                if len(parts) >= 2 and parts[1] == "cost":
                    total_cost += int(value) / 10000

        return round(total_cost, 6)

    def estimate_request_cost(self, model: str, max_tokens: int | None) -> float:
        pricing = self.credit_service.get_model_pricing(model)
        if not pricing:
            return 0.0

        estimated_input_tokens = 500
        estimated_output_tokens = max_tokens or 100

        input_cost = (estimated_input_tokens / 1_000_000) * pricing.get("input_per_million", 0)
        output_cost = (estimated_output_tokens / 1_000_000) * pricing.get("output_per_million", 0)

        return round(input_cost + output_cost, 6)


class UsageTracker:
    def __init__(self, redis: Redis):
        self.redis = redis
        self.cache = RedisCache(redis)
        self.settings = get_settings()
        self.provider_config = get_provider_config()

    async def track_request(
        self,
        user_id: str,
        api_key_id: str,
        model: str,
        provider: str,
        input_tokens: int = 0,
        output_tokens: int = 0,
        cost_usd: float = 0.0,
        latency_ms: int = 0,
        status: str = "success",
    ):
        request_id = f"{user_id}:{int(time.time() * 1000)}"

        daily_key = f"usage:daily:{user_id}:{time.strftime('%Y-%m-%d')}"

        await self.cache.hincrby(daily_key, f"{model}:input_tokens", input_tokens)
        await self.cache.hincrby(daily_key, f"{model}:output_tokens", output_tokens)
        request_count = self.provider_config.get_request_count_multiplier(model)

        await self.cache.hincrby(daily_key, f"{model}:requests", request_count)
        await self.cache.expire(daily_key, 172800)

        hourly_key = f"usage:hourly:{user_id}:{time.strftime('%Y-%m-%d:%H')}"
        await self.cache.hincrby(hourly_key, f"{model}:requests", request_count)
        await self.cache.expire(hourly_key, 7200)

        return request_id

    async def track_image_request(
        self,
        user_id: str,
        api_key_id: str,
        model: str,
        provider: str,
        latency_ms: int = 0,
        status: str = "success",
    ):
        request_id = f"{user_id}:{int(time.time() * 1000)}"

        daily_key = f"usage:daily:{user_id}:{time.strftime('%Y-%m-%d')}"

        request_count = self.provider_config.get_request_count_multiplier(model)

        await self.cache.hincrby(daily_key, f"{model}:image_requests", request_count)
        await self.cache.expire(daily_key, 172800)

        hourly_key = f"usage:hourly:{user_id}:{time.strftime('%Y-%m-%d:%H')}"
        await self.cache.hincrby(hourly_key, f"{model}:image_requests", request_count)
        await self.cache.expire(hourly_key, 7200)

        return request_id

    async def get_daily_usage(self, user_id: str, date: str | None = None) -> dict[str, Any]:
        if date is None:
            date = time.strftime("%Y-%m-%d")

        daily_key = f"usage:daily:{user_id}:{date}"
        usage = await self.cache.hgetall(daily_key)

        credits_key = f"credits:daily:{user_id}:{date}"
        credits_data = await self.cache.hgetall(credits_key)

        models_dict: dict[str, dict[str, int]] = {}
        total_requests = 0
        total_input_tokens = 0
        total_output_tokens = 0
        total_cost = 0.0

        for key, value in usage.items():
            parts = key.split(":")
            if len(parts) >= 2:
                model = parts[0]
                metric = parts[1]
                if model not in models_dict:
                    models_dict[model] = {}
                models_dict[model][metric] = int(value)

                if metric == "requests":
                    total_requests += int(value)
                elif metric == "input_tokens":
                    total_input_tokens += int(value)
                elif metric == "output_tokens":
                    total_output_tokens += int(value)

        for key, value in credits_data.items():
            parts = key.split(":")
            if len(parts) >= 2:
                model = parts[0]
                metric = parts[1]
                if metric == "cost":
                    cost = int(value) / 10000
                    total_cost += cost
                    if model not in models_dict:
                        models_dict[model] = {}
                    models_dict[model]["cost"] = cost
                elif metric == "image_requests":
                    if model not in models_dict:
                        models_dict[model] = {}
                    models_dict[model]["image_requests"] = int(value)
                    models_dict[model]["is_image_model"] = True
                    total_requests += int(value)

        result: dict[str, Any] = {
            "date": date,
            "total_requests": total_requests,
            "total_input_tokens": total_input_tokens,
            "total_output_tokens": total_output_tokens,
            "total_cost": round(total_cost, 6),
            "models": models_dict,
        }

        return result

    async def get_hourly_usage(self, user_id: str, hour: str | None = None) -> dict[str, Any]:
        if hour is None:
            hour = time.strftime("%Y-%m-%d:%H")

        hourly_key = f"usage:hourly:{user_id}:{hour}"
        usage = await self.cache.hgetall(hourly_key)

        date_part = hour.split(":")[0]
        credits_key = f"credits:daily:{user_id}:{date_part}"
        credits_data = await self.cache.hgetall(credits_key)

        models_dict: dict[str, dict[str, int]] = {}
        total_requests = 0
        total_input_tokens = 0
        total_output_tokens = 0
        total_cost = 0.0

        for key, value in usage.items():
            parts = key.split(":")
            if len(parts) >= 2:
                model = parts[0]
                metric = parts[1]
                if model not in models_dict:
                    models_dict[model] = {}
                models_dict[model][metric] = int(value)

                if metric == "requests":
                    total_requests += int(value)
                elif metric == "input_tokens":
                    total_input_tokens += int(value)
                elif metric == "output_tokens":
                    total_output_tokens += int(value)

        for key, value in credits_data.items():
            parts = key.split(":")
            if len(parts) >= 2:
                model = parts[0]
                metric = parts[1]
                if metric == "cost":
                    cost = int(value) / 10000
                    total_cost += cost
                    if model not in models_dict:
                        models_dict[model] = {}
                    models_dict[model]["cost"] = cost
                elif metric == "image_requests":
                    if model not in models_dict:
                        models_dict[model] = {}
                    models_dict[model]["image_requests"] = int(value)
                    models_dict[model]["is_image_model"] = True
                    total_requests += int(value)

        result: dict[str, Any] = {
            "hour": hour,
            "total_requests": total_requests,
            "total_input_tokens": total_input_tokens,
            "total_output_tokens": total_output_tokens,
            "total_cost": round(total_cost, 6),
            "models": models_dict,
        }

        return result

    async def get_monthly_usage(self, user_id: str, month: str | None = None) -> dict[str, Any]:
        if month is None:
            month = time.strftime("%Y-%m")

        total_requests = 0
        total_input_tokens = 0
        total_output_tokens = 0
        total_cost = 0.0
        models_dict: dict[str, dict[str, int]] = {}

        date_pattern = month
        cursor = 0
        keys: list[str] = []

        while True:
            cursor, batch = await self.redis.scan(
                cursor=cursor,
                match=f"usage:daily:{user_id}:{date_pattern}*",
                count=100,
            )
            keys.extend(batch)
            if cursor == 0:
                break

        for key in keys:
            usage = await self.cache.hgetall(key)
            for key_name, value in usage.items():
                parts = key_name.split(":")
                if len(parts) >= 2:
                    model = parts[0]
                    metric = parts[1]
                    int_value = int(value)

                    if metric == "requests":
                        total_requests += int_value
                    elif metric == "input_tokens":
                        total_input_tokens += int_value
                    elif metric == "output_tokens":
                        total_output_tokens += int_value

                    if model not in models_dict:
                        models_dict[model] = {}
                    models_dict[model][metric] = models_dict[model].get(metric, 0) + int_value

        cursor = 0
        credit_keys: list[str] = []

        while True:
            cursor, batch = await self.redis.scan(
                cursor=cursor,
                match=f"credits:daily:{user_id}:{date_pattern}*",
                count=100,
            )
            credit_keys.extend(batch)
            if cursor == 0:
                break

        for key in credit_keys:
            credits_data = await self.cache.hgetall(key)
            for key_name, value in credits_data.items():
                parts = key_name.split(":")
                if len(parts) >= 2:
                    model = parts[0]
                    metric = parts[1]
                    if metric == "cost":
                        cost = int(value) / 10000
                        total_cost += cost
                        if model not in models_dict:
                            models_dict[model] = {}
                        models_dict[model]["cost"] = models_dict[model].get("cost", 0) + cost
                    elif metric == "image_requests":
                        if model not in models_dict:
                            models_dict[model] = {}
                        models_dict[model]["image_requests"] = models_dict[model].get(
                            "image_requests", 0
                        ) + int(value)
                        models_dict[model]["is_image_model"] = True
                        total_requests += int(value)

        result: dict[str, Any] = {
            "month": month,
            "total_requests": total_requests,
            "total_input_tokens": total_input_tokens,
            "total_output_tokens": total_output_tokens,
            "total_cost": round(total_cost, 6),
            "models": models_dict,
        }

        return result

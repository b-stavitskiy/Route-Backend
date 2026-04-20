import pytest

from apps.api.core.config import ProviderConfig
from apps.api.services.usage import request_manager as request_manager_module
from apps.api.services.usage.request_manager import RequestManager
from packages.shared.exceptions import DailyRequestLimitError


class FakeProviderConfig:
    def get_plan_config(self, plan_tier: str) -> dict:
        assert plan_tier == "premium"
        return {"requests_per_day": 3}

    def get_request_count_multiplier(self, model: str | None) -> int:
        if model in {"route/qwen3.5-plus", "route/qwen3.6-plus"}:
            return 2
        if model and model.startswith("route/mimo"):
            return 2
        return 1


class FakePipeline:
    def __init__(self, redis: "FakeRedis") -> None:
        self.redis = redis
        self.commands = []

    def hincrby(self, name: str, key: str, amount: int = 1) -> None:
        self.commands.append(("hincrby", name, key, amount))

    def expire(self, name: str, seconds: int) -> None:
        self.commands.append(("expire", name, seconds))

    async def execute(self) -> list[int | bool]:
        results = []
        for command in self.commands:
            if command[0] == "hincrby":
                _, name, key, amount = command
                hash_value = self.redis.hashes.setdefault(name, {})
                hash_value[key] = hash_value.get(key, 0) + amount
                results.append(hash_value[key])
            elif command[0] == "expire":
                results.append(True)
        return results


class FakeRedis:
    def __init__(self) -> None:
        self.hashes: dict[str, dict[str, int]] = {}

    def pipeline(self) -> FakePipeline:
        return FakePipeline(self)

    async def hget(self, name: str, key: str) -> str | None:
        value = self.hashes.get(name, {}).get(key)
        return str(value) if value is not None else None


def test_provider_config_request_count_multiplier_supports_exact_and_wildcard() -> None:
    provider_config = ProviderConfig()
    original_plans_config = provider_config._plans_config
    provider_config._plans_config = {
        "request_count_multipliers": {
            "route/qwen3.5-plus": 2,
            "route/qwen3.6-plus": 2,
            "route/mimo*": 2,
        }
    }

    try:
        assert provider_config.get_request_count_multiplier("route/qwen3.5-plus") == 2
        assert provider_config.get_request_count_multiplier("route/qwen3.6-plus") == 2
        assert provider_config.get_request_count_multiplier("route/mimo-v2-omni") == 2
        assert provider_config.get_request_count_multiplier("route/minimax-m2.5") == 1
    finally:
        provider_config._plans_config = original_plans_config


@pytest.fixture
def request_manager(monkeypatch: pytest.MonkeyPatch) -> RequestManager:
    monkeypatch.setattr(
        request_manager_module,
        "get_provider_config",
        lambda: FakeProviderConfig(),
    )
    return RequestManager(FakeRedis())


@pytest.mark.asyncio
async def test_premium_qwen_plus_counts_as_two_requests(request_manager: RequestManager) -> None:
    current_count = await request_manager.check_and_increment(
        "user_1",
        "premium",
        "route/qwen3.5-plus",
    )

    assert current_count == 2
    assert await request_manager.get_daily_request_count("user_1") == 2


@pytest.mark.asyncio
async def test_mimo_models_count_as_two_requests(request_manager: RequestManager) -> None:
    current_count = await request_manager.check_and_increment(
        "user_1",
        "premium",
        "route/mimo-v2-pro",
    )

    assert current_count == 2
    assert await request_manager.get_daily_request_count("user_1") == 2


@pytest.mark.asyncio
async def test_weighted_request_rolls_back_when_over_limit(
    request_manager: RequestManager,
) -> None:
    await request_manager.check_and_increment("user_1", "premium", "route/minimax-m2.5")
    await request_manager.check_and_increment("user_1", "premium", "route/minimax-m2.5")

    with pytest.raises(DailyRequestLimitError) as exc_info:
        await request_manager.check_and_increment("user_1", "premium", "route/qwen3.6-plus")

    assert exc_info.value.details == {"limit": 3, "used": 2, "remaining": 1}
    assert await request_manager.get_daily_request_count("user_1") == 2

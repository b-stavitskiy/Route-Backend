from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

import pytest
from starlette.requests import Request

from apps.api.api.v1.endpoints import models as models_endpoint
from apps.api.core import plans as plan_helpers
from apps.api.services.llm.router import LLMRouter


class DummyRedis:
    async def get(self, _key: str):
        return None

    async def set(self, _key: str, _value, ex: int | None = None):
        return None


class FakeProviderConfig:
    def __init__(self) -> None:
        self._config = {
            "providers": {
                "models": {
                    "free": {
                        "route/kimi-k2.5": {
                            "provider_chain": [{"provider": "crof"}],
                            "context_size": 32000,
                            "max_output_tokens": 4096,
                        },
                    },
                    "lite": {
                        "route/minimax-m2.5": {
                            "name": "MiniMax M2.5",
                            "provider_chain": [{"provider": "crof"}],
                            "modalities": {"input": ["text"], "output": ["text"]},
                            "options": {"thinking": {"type": "enabled", "budgetTokens": 8192}},
                            "context_size": 128000,
                            "max_output_tokens": 8192,
                        },
                        "route/kimi-k2.5": {
                            "provider_chain": [{"provider": "fallback"}],
                            "context_size": 64000,
                            "max_output_tokens": 4096,
                        },
                    },
                    "premium": {
                        "route/glm-5.1": {
                            "provider_chain": [{"provider": "zai"}],
                            "context_size": 200000,
                            "max_output_tokens": 16000,
                        },
                    },
                    "max": {
                        "route/gpt-5": {
                            "name": "GPT-5",
                            "provider_chain": [{"provider": "openai"}],
                            "modalities": {"input": ["text", "image"], "output": ["text"]},
                            "limit": {"context": 256000, "output": 32000},
                        },
                    },
                }
            }
        }

    def get_allowed_models(self, user_plan: str) -> list[str] | str:
        allowed_by_plan = {
            "free": ["route/kimi-k2.5"],
            "lite": ["route/minimax-m2.5", "route/kimi-k2.5"],
            "premium": ["route/minimax-m2.5", "route/glm-5.1"],
            "max": "all",
            "custom:max:10000": ["route/glm-5.1", "route/gpt-5"],
        }
        return allowed_by_plan[user_plan]

    def _ordered_model_tiers(self) -> list[str]:
        return ["free", "lite", "premium", "max"]

    def get_model_config(self, model: str, user_plan: str = "free") -> dict | None:
        models = self._config["providers"]["models"]
        if model in models.get(user_plan, {}):
            return models[user_plan][model]
        for tier in ["free", "lite", "premium", "max"]:
            if model in models.get(tier, {}):
                return models[tier][model]
        return None

    def is_model_allowed(self, model: str, user_plan: str) -> bool:
        allowed_models = self.get_allowed_models(user_plan)
        return allowed_models == "all" or model in allowed_models


def make_request(headers: list[tuple[bytes, bytes]]) -> Request:
    return Request(
        {
            "type": "http",
            "method": "GET",
            "path": "/v1/models",
            "headers": headers,
        }
    )


@pytest.mark.asyncio
async def test_get_user_plan_accepts_bearer_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    async def fake_resolve_api_key_plan(api_key: str) -> str | None:
        assert api_key == "rr_test_123"
        return "premium"

    monkeypatch.setattr(models_endpoint, "resolve_api_key_plan", fake_resolve_api_key_plan)

    request = make_request([(b"authorization", b"Bearer rr_test_123")])

    assert await models_endpoint.get_user_plan(request) == "premium"


@pytest.mark.asyncio
async def test_get_user_plan_checks_bearer_api_key_without_prefix(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def fake_resolve_api_key_plan(api_key: str) -> str | None:
        assert api_key == "plain_api_key"
        return "max"

    async def fake_verify_access_token(_token: str) -> dict[str, str]:
        raise AssertionError("JWT verification should not run for valid API keys")

    monkeypatch.setattr(models_endpoint, "resolve_api_key_plan", fake_resolve_api_key_plan)
    monkeypatch.setattr(models_endpoint, "verify_access_token", fake_verify_access_token)

    request = make_request([(b"authorization", b"Bearer plain_api_key")])

    assert await models_endpoint.get_user_plan(request) == "max"


@pytest.mark.asyncio
async def test_get_user_plan_falls_back_to_access_token(monkeypatch: pytest.MonkeyPatch) -> None:
    async def fake_resolve_api_key_plan(_api_key: str) -> str | None:
        return None

    async def fake_verify_access_token(token: str) -> dict[str, str]:
        assert token == "jwt_token"
        return {"plan": "lite"}

    monkeypatch.setattr(models_endpoint, "resolve_api_key_plan", fake_resolve_api_key_plan)
    monkeypatch.setattr(models_endpoint, "verify_access_token", fake_verify_access_token)

    request = make_request([(b"authorization", b"Bearer jwt_token")])

    assert await models_endpoint.get_user_plan(request) == "lite"


@pytest.mark.asyncio
async def test_resolve_api_key_plan_prefers_active_user_upgrade(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    upgraded_user = SimpleNamespace(
        plan_tier=SimpleNamespace(value="premium"),
        upgraded_to_tier=SimpleNamespace(value="max"),
        upgraded_until=datetime.now(UTC) + timedelta(hours=1),
    )
    api_key = SimpleNamespace(
        plan_tier=SimpleNamespace(value="premium"),
        user=upgraded_user,
    )

    class FakeResult:
        def scalar_one_or_none(self) -> SimpleNamespace:
            return api_key

    class FakeSession:
        async def execute(self, _query) -> FakeResult:
            return FakeResult()

    class FakeSessionContext:
        async def __aenter__(self) -> FakeSession:
            return FakeSession()

        async def __aexit__(self, exc_type, exc, tb) -> None:
            return None

    monkeypatch.setattr("apps.api.core.security.hash_api_key", lambda key: f"hashed:{key}")
    monkeypatch.setattr("packages.db.session.get_db_session", lambda: FakeSessionContext())

    assert await models_endpoint.resolve_api_key_plan("rk_test") == "max"


@pytest.mark.asyncio
async def test_resolve_api_key_plan_prefers_active_custom_upgrade(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    upgraded_user = SimpleNamespace(
        plan_tier=SimpleNamespace(value="premium"),
        custom_plan_id=None,
        upgraded_to_tier=None,
        upgraded_custom_model_catalog_tier=SimpleNamespace(value="max"),
        upgraded_custom_requests_per_day=10000,
        upgraded_until=datetime.now(UTC) + timedelta(hours=1),
    )
    api_key = SimpleNamespace(
        plan_tier=SimpleNamespace(value="premium"),
        user=upgraded_user,
    )

    class FakeResult:
        def scalar_one_or_none(self) -> SimpleNamespace:
            return api_key

    class FakeSession:
        async def execute(self, _query) -> FakeResult:
            return FakeResult()

    class FakeSessionContext:
        async def __aenter__(self) -> FakeSession:
            return FakeSession()

        async def __aexit__(self, exc_type, exc, tb) -> None:
            return None

    monkeypatch.setattr("apps.api.core.security.hash_api_key", lambda key: f"hashed:{key}")
    monkeypatch.setattr("packages.db.session.get_db_session", lambda: FakeSessionContext())

    assert await models_endpoint.resolve_api_key_plan("rk_test") == "custom:max:10000"


@pytest.mark.asyncio
async def test_list_available_models_uses_configured_tiers_and_deduplicates() -> None:
    router = LLMRouter(DummyRedis())
    router.provider_config = FakeProviderConfig()

    models = await router.list_available_models("max")

    assert [model["id"] for model in models] == [
        "route/kimi-k2.5",
        "route/minimax-m2.5",
        "route/glm-5.1",
        "route/gpt-5",
    ]
    assert [model["tier"] for model in models] == ["free", "lite", "premium", "max"]
    assert [model["owned_by"] for model in models] == ["crof", "crof", "zai", "openai"]
    assert [model["limit"] for model in models] == [
        {"context": 32000, "output": 4096},
        {"context": 128000, "output": 8192},
        {"context": 200000, "output": 16000},
        {"context": 256000, "output": 32000},
    ]
    assert all("context_window" not in model for model in models)
    assert all("max_output_tokens" not in model for model in models)
    assert models[0]["name"] == "route/kimi-k2.5"
    assert models[0]["modalities"] == {"input": ["text"], "output": ["text"]}
    assert models[0]["options"] == {}
    assert models[1]["name"] == "MiniMax M2.5"
    assert models[1]["modalities"] == {"input": ["text"], "output": ["text"]}
    assert models[1]["options"] == {"thinking": {"type": "enabled", "budgetTokens": 8192}}


@pytest.mark.asyncio
async def test_get_model_returns_context_metadata(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(models_endpoint, "get_provider_config", lambda: FakeProviderConfig())

    async def fake_get_user_plan(_request: Request) -> str:
        return "premium"

    monkeypatch.setattr(models_endpoint, "get_user_plan", fake_get_user_plan)

    request = make_request([])
    payload = await models_endpoint.get_model(request, "route/glm-5.1")

    assert payload["id"] == "route/glm-5.1"
    assert payload["owned_by"] == "zai"
    assert payload["allowed"] is True
    assert payload["name"] == "route/glm-5.1"
    assert payload["modalities"] == {"input": ["text"], "output": ["text"]}
    assert payload["options"] == {}
    assert payload["limit"] == {"context": 200000, "output": 16000}
    assert "context_window" not in payload
    assert "max_output_tokens" not in payload


@pytest.mark.asyncio
async def test_get_model_returns_rich_metadata(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(models_endpoint, "get_provider_config", lambda: FakeProviderConfig())

    async def fake_get_user_plan(_request: Request) -> str:
        return "lite"

    monkeypatch.setattr(models_endpoint, "get_user_plan", fake_get_user_plan)

    request = make_request([])
    payload = await models_endpoint.get_model(request, "route/minimax-m2.5")

    assert payload["name"] == "MiniMax M2.5"
    assert payload["modalities"] == {"input": ["text"], "output": ["text"]}
    assert payload["options"] == {"thinking": {"type": "enabled", "budgetTokens": 8192}}
    assert payload["limit"] == {"context": 128000, "output": 8192}


@pytest.mark.asyncio
async def test_list_available_models_supports_custom_plan_configs() -> None:
    router = LLMRouter(DummyRedis())
    router.provider_config = FakeProviderConfig()

    models = await router.list_available_models("custom:max:10000")

    assert [model["id"] for model in models] == ["route/glm-5.1", "route/gpt-5"]


def test_custom_plan_display_name_prefers_named_plan() -> None:
    user = SimpleNamespace(
        plan_tier=SimpleNamespace(value="free"),
        custom_plan_name="enterprise-kira",
        custom_model_catalog_tier=SimpleNamespace(value="max"),
        custom_requests_per_day=5000,
        upgraded_custom_plan_name=None,
        upgraded_custom_model_catalog_tier=None,
        upgraded_custom_requests_per_day=None,
        upgraded_to_tier=None,
        upgraded_until=None,
    )

    assert plan_helpers.get_user_effective_plan_name(user) == "custom:max:5000"
    assert plan_helpers.get_user_effective_plan_display_name(user) == "enterprise-kira"

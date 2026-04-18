from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

import pytest
from starlette.requests import Request

from apps.api.api.v1.endpoints import models as models_endpoint
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
                        },
                    },
                    "lite": {
                        "route/minimax-m2.5": {
                            "provider_chain": [{"provider": "crof"}],
                        },
                        "route/kimi-k2.5": {
                            "provider_chain": [{"provider": "fallback"}],
                        },
                    },
                    "premium": {
                        "route/glm-5.1": {
                            "provider_chain": [{"provider": "zai"}],
                        },
                    },
                    "max": {
                        "route/gpt-5": {
                            "provider_chain": [{"provider": "openai"}],
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
        }
        return allowed_by_plan[user_plan]


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

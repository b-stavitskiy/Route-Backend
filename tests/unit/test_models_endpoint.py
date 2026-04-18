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

    monkeypatch.setattr(
        models_endpoint,
        "get_settings",
        lambda: SimpleNamespace(api_key_prefix="rr_"),
    )
    monkeypatch.setattr(models_endpoint, "resolve_api_key_plan", fake_resolve_api_key_plan)

    request = make_request([(b"authorization", b"Bearer rr_test_123")])

    assert await models_endpoint.get_user_plan(request) == "premium"


@pytest.mark.asyncio
async def test_get_user_plan_falls_back_to_access_token(monkeypatch: pytest.MonkeyPatch) -> None:
    async def fake_verify_access_token(token: str) -> dict[str, str]:
        assert token == "jwt_token"
        return {"plan": "lite"}

    monkeypatch.setattr(
        models_endpoint,
        "get_settings",
        lambda: SimpleNamespace(api_key_prefix="rr_"),
    )
    monkeypatch.setattr(models_endpoint, "verify_access_token", fake_verify_access_token)

    request = make_request([(b"authorization", b"Bearer jwt_token")])

    assert await models_endpoint.get_user_plan(request) == "lite"


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

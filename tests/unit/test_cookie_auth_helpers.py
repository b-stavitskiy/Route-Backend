import pytest
from starlette.requests import Request

from apps.api.api.v1.endpoints import models as models_endpoint
from apps.api.api.v1.endpoints import user as user_endpoint
from apps.api.core.security import get_access_token_from_request


def make_request(path: str, headers: list[tuple[bytes, bytes]]) -> Request:
    return Request(
        {
            "type": "http",
            "method": "GET",
            "path": path,
            "headers": headers,
        }
    )


def test_get_access_token_from_request_falls_back_to_cookie() -> None:
    request = make_request("/v1/user", [(b"cookie", b"access_token=cookie-token")])

    assert get_access_token_from_request(request) == "cookie-token"


@pytest.mark.asyncio
async def test_get_authenticated_user_accepts_access_token_cookie(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def fake_verify_access_token(token: str) -> dict[str, str]:
        assert token == "cookie-token"
        return {"sub": "user-123", "plan": "premium"}

    monkeypatch.setattr(user_endpoint, "verify_access_token", fake_verify_access_token)

    request = make_request("/v1/user", [(b"cookie", b"access_token=cookie-token")])

    assert await user_endpoint.get_authenticated_user(request) == ("user-123", "premium")


@pytest.mark.asyncio
async def test_get_user_plan_accepts_access_token_cookie(monkeypatch: pytest.MonkeyPatch) -> None:
    async def fake_resolve_api_key_plan(_api_key: str) -> str | None:
        return None

    async def fake_verify_access_token(token: str) -> dict[str, str]:
        assert token == "cookie-token"
        return {"plan": "premium"}

    monkeypatch.setattr(models_endpoint, "resolve_api_key_plan", fake_resolve_api_key_plan)
    monkeypatch.setattr(models_endpoint, "verify_access_token", fake_verify_access_token)

    request = make_request("/v1/models", [(b"cookie", b"access_token=cookie-token")])

    assert await models_endpoint.get_user_plan(request) == "premium"

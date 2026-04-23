from types import SimpleNamespace

import pytest
from starlette.requests import Request
from starlette.responses import Response

from apps.api.core.middleware import AuthMiddleware


def make_request(path: str, headers: list[tuple[bytes, bytes]]) -> Request:
    return Request(
        {
            "type": "http",
            "method": "GET",
            "path": path,
            "headers": headers,
            "client": ("127.0.0.1", 12345),
        }
    )


@pytest.mark.asyncio
async def test_auth_middleware_accepts_access_token_cookie(monkeypatch: pytest.MonkeyPatch) -> None:
    async def fake_verify_access_token(token: str) -> dict[str, str]:
        assert token == "cookie-token"
        return {"sub": "user-123"}

    async def call_next(request: Request) -> Response:
        assert request.state.user_id == "user-123"
        return Response(status_code=204)

    middleware = AuthMiddleware(app=SimpleNamespace())
    monkeypatch.setattr("apps.api.core.middleware.verify_access_token", fake_verify_access_token)

    request = make_request(
        "/auth/me",
        [(b"cookie", b"access_token=cookie-token")],
    )

    response = await middleware.dispatch(request, call_next)

    assert response.status_code == 204

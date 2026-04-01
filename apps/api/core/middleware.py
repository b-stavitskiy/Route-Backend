from collections.abc import Callable
import hashlib
import time

from fastapi import Request, Response
from fastapi.responses import JSONResponse
from apps.api.core.config import get_settings
from apps.api.core.security import verify_access_token
from packages.redis.client import get_redis
from packages.shared.exceptions import AppError, AuthenticationError
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp


AUTH_RATE_LIMIT = 10
AUTH_WINDOW = 60

GENERAL_RATE_LIMIT = 100
GENERAL_WINDOW = 60


def get_client_ip(request: Request) -> str:
    forwarded = request.headers.get("x-forwarded-for")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


class AuthMiddleware(BaseHTTPMiddleware):
    def __init__(self, app: ASGIApp):
        super().__init__(app)
        self.settings = get_settings()

    async def dispatch(
        self,
        request: Request,
        call_next: Callable,
    ) -> Response:
        public_paths = {
            "/",
            "/health",
            "/auth/signup",
            "/auth/login",
            "/auth/forgot-password",
            "/auth/reset-password",
            "/auth/verify-email",
            "/auth/oauth/github",
            "/webhooks/whop",
            "/v1/models",
            "/v1/settings",
            "/docs",
            "/openapi.json",
            "/redoc",
        }

        if request.url.path in public_paths or request.url.path.startswith("/auth/callback"):
            return await call_next(request)

        auth_header = request.headers.get("Authorization")

        if not auth_header or not auth_header.startswith("Bearer "):
            return JSONResponse(
                content={"message": "Authentication required"},
                status_code=401,
            )

        token = auth_header[7:]
        try:
            payload = verify_access_token(token)
            request.state.user_id = payload.get("sub")
        except AuthenticationError:
            return JSONResponse(
                content={"message": "Invalid or expired token"},
                status_code=401,
            )

        return await call_next(request)


class RateLimitMiddleware(BaseHTTPMiddleware):
    def __init__(self, app: ASGIApp):
        super().__init__(app)
        self.settings = get_settings()

    async def dispatch(
        self,
        request: Request,
        call_next: Callable,
    ) -> Response:
        client_ip = get_client_ip(request)
        redis = await get_redis()

        auth_paths = {"/auth/signup", "/auth/login", "/auth/refresh", "/auth/forgot-password"}
        is_auth_path = request.url.path in auth_paths

        if is_auth_path:
            limit = AUTH_RATE_LIMIT
            window = AUTH_WINDOW
        else:
            limit = GENERAL_RATE_LIMIT
            window = GENERAL_WINDOW

        key = (
            f"ratelimit:ip:{hashlib.sha256(client_ip.encode()).hexdigest()[:16]}:{request.url.path}"
        )

        try:
            current = await redis.get(key)
            if current is None:
                await redis.setex(key, window, "1")
            else:
                count = int(current)
                if count >= limit:
                    ttl = await redis.ttl(key)
                    return JSONResponse(
                        content={
                            "message": "Rate limit exceeded",
                            "retry_after": ttl if ttl > 0 else window,
                        },
                        status_code=429,
                        headers={"Retry-After": str(ttl if ttl > 0 else window)},
                    )
                await redis.incr(key)
        except Exception:
            pass

        response = await call_next(request)
        return response


class MetricsMiddleware(BaseHTTPMiddleware):
    def __init__(self, app: ASGIApp):
        super().__init__(app)

    async def dispatch(
        self,
        request: Request,
        call_next: Callable,
    ) -> Response:
        start_time = getattr(request.state, "start_time", None)
        if start_time is None:
            start_time = 0

        response = await call_next(request)

        return response


class ExceptionHandlerMiddleware(BaseHTTPMiddleware):
    def __init__(self, app: ASGIApp):
        super().__init__(app)

    async def dispatch(
        self,
        request: Request,
        call_next: Callable,
    ) -> Response:
        try:
            return await call_next(request)
        except AppError as e:
            return Response(
                content=e.message,
                status_code=e.status_code,
                headers={"X-Error-Code": e.error_code},
            )
        except Exception as e:
            import json
            import logging

            logging.error(f"Unhandled exception: {str(e)}", exc_info=True)

            return Response(
                content=json.dumps({"message": "Internal server error"}),
                status_code=500,
                media_type="application/json",
            )

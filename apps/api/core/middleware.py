import hashlib
import logging
from collections.abc import Callable

from fastapi import Request, Response
from fastapi.responses import JSONResponse
from redis.asyncio import Redis
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp

from apps.api.core.config import get_settings
from apps.api.core.security import verify_access_token
from packages.redis.client import get_redis
from packages.shared.exceptions import AppError, AuthenticationError

logger = logging.getLogger("routing.run.api")

RATE_LIMIT_LUA_SCRIPT = """
local key = KEYS[1]
local limit = tonumber(ARGV[1])
local window = tonumber(ARGV[2])

local count = redis.call('INCR', key)
if count == 1 then
    redis.call('EXPIRE', key, window)
end

local ttl = redis.call('TTL', key)
if ttl < 0 then
    ttl = window
end

if count > limit then
    return {0, 0, ttl}
else
    return {1, limit - count, 0}
end
"""

BLACKLIST_VIOLATIONS_LUA_SCRIPT = """
local key = KEYS[1]
local max_violations = tonumber(ARGV[1])
local ban_seconds = tonumber(ARGV[2])
local window = tonumber(ARGV[3])

local count = redis.call('INCR', key)
if count == 1 then
    redis.call('EXPIRE', key, window)
end

if count >= max_violations then
    redis.call('SET', KEYS[2], '1', 'EX', ban_seconds)
    redis.call('DEL', key)
    return {1, ban_seconds}
end

return {0, 0}
"""

RATE_CONFIGS: dict[str, tuple[int, int]] = {
    "auth_strict": (5, 60),
    "auth_general": (30, 60),
    "sensitive": (10, 60),
    "default": (60, 60),
}
GLOBAL_RATE_LIMIT: tuple[int, int] = (300, 60)

BLACKLIST_VIOLATIONS_THRESHOLD = 5
BLACKLIST_WINDOW = 300
BLACKLIST_BAN_SECONDS = 3600

AUTH_STRICT_PATHS = frozenset(
    {
        "/auth/signup/init",
        "/auth/signup/verify",
        "/auth/login/init",
        "/auth/login/verify",
        "/auth/forgot-password",
        "/auth/reset-password",
    }
)

SKIP_PATHS = frozenset(
    {
        "/",
        "/health",
        "/docs",
        "/openapi.json",
        "/redoc",
        "/v1/status",
    }
)


def get_client_ip(request: Request) -> str:
    real_ip = request.headers.get("x-real-ip")
    if real_ip:
        return real_ip.strip()
    cf_ip = request.headers.get("cf-connecting-ip")
    if cf_ip:
        return cf_ip.strip()
    forwarded = request.headers.get("x-forwarded-for")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


def get_ip_debug_info(request: Request) -> str:
    parts = []
    real = request.headers.get("x-real-ip")
    if real:
        parts.append(f"x-real-ip={real}")
    cf = request.headers.get("cf-connecting-ip")
    if cf:
        parts.append(f"cf-connecting-ip={cf}")
    forwarded = request.headers.get("x-forwarded-for")
    if forwarded:
        parts.append(f"x-forwarded-for={forwarded}")
    remote = request.client.host if request.client else "unknown"
    parts.append(f"remote={remote}")
    true_client = request.headers.get("True-Client-IP")
    if true_client:
        parts.append(f"true-client-ip={true_client}")
    cf_ip_class = request.headers.get("CF-IPCountry")
    if cf_ip_class:
        parts.append(f"cf-ipcountry={cf_ip_class}")
    return " | ".join(parts)


def classify_path(path: str) -> str:
    if path in AUTH_STRICT_PATHS:
        return "auth_strict"
    if path.startswith("/auth/"):
        return "auth_general"
    if "/user/keys" in path or path.endswith("/user/password"):
        return "sensitive"
    return "default"


RESTRICTED_ORIGIN_PATHS = (
    "/auth/signup",
    "/auth/login",
    "/auth/forgot-password",
    "/auth/reset-password",
    "/auth/verify-email",
    "/auth/oauth",
    "/auth/refresh",
    "/auth/logout",
    "/v1/user",
)


class OriginRestrictionMiddleware(BaseHTTPMiddleware):
    def __init__(self, app: ASGIApp):
        super().__init__(app)
        self._allowed_origin = get_settings().app_origin.lower().rstrip("/")

    async def dispatch(
        self,
        request: Request,
        call_next: Callable,
    ) -> Response:
        path = request.url.path
        is_restricted = any(path.startswith(p) for p in RESTRICTED_ORIGIN_PATHS)

        if not is_restricted:
            return await call_next(request)

        if request.method == "OPTIONS":
            origin = request.headers.get("origin", "").lower().rstrip("/")
            allowed = self._allowed_origin
            allow_headers = request.headers.get("access-control-request-headers", "*")
            return Response(
                status_code=200,
                headers={
                    "Access-Control-Allow-Origin": allowed,
                    "Access-Control-Allow-Credentials": "true",
                    "Access-Control-Allow-Methods": "GET,POST,PUT,PATCH,DELETE,OPTIONS",
                    "Access-Control-Allow-Headers": allow_headers,
                },
            )

        origin = request.headers.get("origin", "")
        referer = request.headers.get("referer", "")

        if origin:
            if origin.lower().rstrip("/") != self._allowed_origin:
                logger.warning(
                    f"Blocked auth request from disallowed origin: "
                    f"{origin} | {get_ip_debug_info(request)}"
                )
                return JSONResponse(
                    status_code=403,
                    content={"message": "Access denied"},
                )
        elif referer:
            from urllib.parse import urlparse

            referer_origin = urlparse(referer).scheme + "://" + urlparse(referer).netloc
            if referer_origin.lower().rstrip("/") != self._allowed_origin:
                logger.warning(
                    f"Blocked auth request from disallowed referer: "
                    f"{referer} | {get_ip_debug_info(request)}"
                )
                return JSONResponse(
                    status_code=403,
                    content={"message": "Access denied"},
                )
        elif not request.headers.get("authorization") and not request.headers.get("x-api-key"):
            logger.warning(
                f"Blocked auth request with no origin/referer: "
                f"path={path} | {get_ip_debug_info(request)}"
            )
            return JSONResponse(
                status_code=403,
                content={"message": "Access denied"},
            )

        return await call_next(request)


class RateLimitMiddleware(BaseHTTPMiddleware):
    _lua_script = None
    _blacklist_lua_script = None

    @classmethod
    def _get_lua_script(cls, redis: Redis):
        if cls._lua_script is None:
            cls._lua_script = redis.register_script(RATE_LIMIT_LUA_SCRIPT)
        return cls._lua_script

    @classmethod
    def _get_blacklist_lua_script(cls, redis: Redis):
        if cls._blacklist_lua_script is None:
            cls._blacklist_lua_script = redis.register_script(BLACKLIST_VIOLATIONS_LUA_SCRIPT)
        return cls._blacklist_lua_script

    async def _check_blacklist(
        self, redis: Redis, ip_hash: str, client_ip: str, request: Request
    ) -> JSONResponse | None:
        ban_key = f"bl:{ip_hash}"
        is_banned = await redis.exists(ban_key)
        if is_banned:
            ttl = await redis.ttl(ban_key)
            logger.warning(
                f"BLACKLISTED IP blocked: ip={client_ip} "
                f"path={request.url.path} | {get_ip_debug_info(request)}"
            )
            return JSONResponse(
                status_code=403,
                content={"message": "Access denied"},
                headers={"Retry-After": str(max(ttl, 0))},
            )
        return None

    async def _record_violation(
        self, redis: Redis, ip_hash: str, client_ip: str, request: Request, reason: str
    ) -> None:
        script = self._get_blacklist_lua_script(redis)
        violations_key = f"bl:v:{ip_hash}"
        ban_key = f"bl:{ip_hash}"

        result = await script(
            keys=[violations_key, ban_key],
            args=[
                str(BLACKLIST_VIOLATIONS_THRESHOLD),
                str(BLACKLIST_BAN_SECONDS),
                str(BLACKLIST_WINDOW),
            ],
        )

        was_banned = bool(result[0])

        if was_banned:
            logger.warning(
                f"IP BLACKLISTED for {BLACKLIST_BAN_SECONDS}s: "
                f"ip={client_ip} reason={reason} | {get_ip_debug_info(request)}"
            )
        else:
            logger.warning(
                f"Rate limit violation: ip={client_ip} "
                f"reason={reason} | {get_ip_debug_info(request)}"
            )

    async def dispatch(
        self,
        request: Request,
        call_next: Callable,
    ) -> Response:
        path = request.url.path
        method = request.method

        if method == "OPTIONS" or path in SKIP_PATHS or path.startswith("/auth/callback"):
            return await call_next(request)

        client_ip = get_client_ip(request)
        ip_hash = hashlib.sha256(client_ip.encode()).hexdigest()[:16]

        try:
            redis = await get_redis()
        except Exception:
            return await call_next(request)

        ban_response = await self._check_blacklist(redis, ip_hash, client_ip, request)
        if ban_response:
            return ban_response

        path_group = classify_path(path)
        limit, window = RATE_CONFIGS[path_group]

        try:
            script = self._get_lua_script(redis)

            path_key = f"rl:{ip_hash}:{path_group}"
            path_result = await script(keys=[path_key], args=[str(limit), str(window)])

            path_allowed = bool(path_result[0])
            path_remaining = int(path_result[1])
            path_retry_after = int(path_result[2])

            if not path_allowed:
                await self._record_violation(
                    redis, ip_hash, client_ip, request, f"rate_limit:{path_group}"
                )
                return self._build_429_response(limit, path_retry_after)

            global_limit, global_window = GLOBAL_RATE_LIMIT
            global_key = f"rl:{ip_hash}:global"
            global_result = await script(
                keys=[global_key], args=[str(global_limit), str(global_window)]
            )

            global_allowed = bool(global_result[0])
            global_remaining = int(global_result[1])
            global_retry_after = int(global_result[2])

            if not global_allowed:
                await self._record_violation(
                    redis, ip_hash, client_ip, request, "global_rate_limit"
                )
                return self._build_429_response(global_limit, global_retry_after)

            response = await call_next(request)

            remaining = min(path_remaining, global_remaining)
            response.headers["X-RateLimit-Limit"] = str(limit)
            response.headers["X-RateLimit-Remaining"] = str(remaining)
            return response
        except Exception:
            logger.exception("Rate limit check failed, allowing request")
            return await call_next(request)

    def _build_429_response(self, limit: int, retry_after: int) -> JSONResponse:
        return JSONResponse(
            status_code=429,
            content={
                "message": "Rate limit exceeded. Please try again later.",
                "retry_after": retry_after,
            },
            headers={
                "Retry-After": str(retry_after),
                "X-RateLimit-Limit": str(limit),
                "X-RateLimit-Remaining": "0",
            },
        )


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
            "/auth/signup/init",
            "/auth/signup/verify",
            "/auth/login/init",
            "/auth/login/verify",
            "/auth/forgot-password",
            "/auth/reset-password",
            "/auth/verify-email",
            "/auth/oauth/github",
            "/webhooks/whop",
            "/v1/models",
            "/v1/settings",
            "/v1/status",
            "/docs",
            "/openapi.json",
            "/redoc",
        }

        if request.url.path in public_paths or request.url.path.startswith("/auth/callback"):
            return await call_next(request)

        auth_header = request.headers.get("Authorization")
        api_key = request.headers.get("X-API-Key", "")

        if api_key:
            request.state.api_key = api_key
            return await call_next(request)

        if auth_header and auth_header.startswith("Bearer "):
            token = auth_header[7:]
            settings = get_settings()
            if token.startswith(settings.api_key_prefix):
                request.state.api_key = token
                return await call_next(request)

            try:
                payload = await verify_access_token(token)
                request.state.user_id = payload.get("sub")
                return await call_next(request)
            except AuthenticationError:
                return JSONResponse(
                    content={"message": "Invalid or expired token"},
                    status_code=401,
                )

        return JSONResponse(
            content={"message": "Authentication required"},
            status_code=401,
        )


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

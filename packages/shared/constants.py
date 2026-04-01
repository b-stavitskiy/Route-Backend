__all__ = [
    "API_KEY_PREFIX",
    "TOKEN_EXPIRE_MINUTES",
    "REFRESH_TOKEN_EXPIRE_DAYS",
    "RATE_LIMIT_WINDOW_SECONDS",
    "OAUTH_CALLBACK_PATH",
    "REDIS_KEYS",
    "ERROR_MESSAGES",
    "HTTP_STATUS",
]

API_KEY_PREFIX = "rk_"
TOKEN_EXPIRE_MINUTES = 60
REFRESH_TOKEN_EXPIRE_DAYS = 30

RATE_LIMIT_WINDOW_SECONDS = 3600

OAUTH_CALLBACK_PATH = "/auth/callback"

REDIS_KEYS = {
    "rate_limit": "ratelimit:{key_hash}",
    "session": "session:{user_id}",
    "provider_health": "provider:{name}:health",
    "provider_latency": "provider:{name}:latency",
    "usage_daily": "usage:daily:{user_id}:{date}",
    "model_cache": "model:{name}",
    "provider_cache": "provider:{name}",
}

ERROR_MESSAGES = {
    "invalid_api_key": "Invalid or revoked API key",
    "rate_limited": "Rate limit exceeded",
    "model_not_allowed": "This model is not available on your plan",
    "invalid_model": "Invalid model requested",
    "provider_unavailable": "No providers available for this model",
    "timeout": "Request timed out",
    "internal_error": "Internal server error",
}

HTTP_STATUS = {
    "success": 200,
    "created": 201,
    "bad_request": 400,
    "unauthorized": 401,
    "forbidden": 403,
    "not_found": 404,
    "rate_limited": 429,
    "internal_error": 500,
    "bad_gateway": 502,
    "service_unavailable": 503,
}

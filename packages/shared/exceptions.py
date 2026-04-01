from typing import Any

__all__ = [
    "AppError",
    "AuthenticationError",
    "AuthorizationError",
    "CircuitBreakerOpenError",
    "DuplicateResourceError",
    "InsufficientCreditsError",
    "InvalidAPIKeyError",
    "InvalidModelError",
    "ModelNotAllowedError",
    "NotFoundError",
    "ProviderError",
    "ProviderTimeoutError",
    "RateLimitError",
    "ValidationError",
]


class AppError(Exception):
    def __init__(
        self,
        message: str,
        status_code: int = 500,
        error_code: str = "INTERNAL_ERROR",
        details: dict[str, Any] | None = None,
    ):
        self.message = message
        self.status_code = status_code
        self.error_code = error_code
        self.details = details
        super().__init__(self.message)


class AuthenticationError(AppError):
    def __init__(
        self, message: str = "Authentication failed", details: dict[str, Any] | None = None
    ):
        super().__init__(
            message=message,
            status_code=401,
            error_code="AUTHENTICATION_ERROR",
            details=details,
        )


class AuthorizationError(AppError):
    def __init__(self, message: str = "Access denied", details: dict[str, Any] | None = None):
        super().__init__(
            message=message,
            status_code=403,
            error_code="AUTHORIZATION_ERROR",
            details=details,
        )


class RateLimitError(AppError):
    def __init__(
        self,
        message: str = "Rate limit exceeded",
        retry_after: int = 0,
        details: dict[str, Any] | None = None,
    ):
        super().__init__(
            message=message,
            status_code=429,
            error_code="RATE_LIMIT_ERROR",
            details=details or {"retry_after": retry_after},
        )


class InvalidAPIKeyError(AuthenticationError):
    def __init__(self, message: str = "Invalid or revoked API key"):
        super().__init__(
            message=message,
            details={"error_code": "invalid_api_key"},
        )


class ModelNotAllowedError(AppError):
    def __init__(self, model: str, plan: str):
        super().__init__(
            message=f"Model '{model}' is not available on your plan",
            status_code=403,
            error_code="MODEL_NOT_ALLOWED",
            details={"model": model, "plan": plan},
        )


class InvalidModelError(AppError):
    def __init__(self, model: str):
        super().__init__(
            message=f"Invalid model: '{model}'",
            status_code=400,
            error_code="INVALID_MODEL",
            details={"model": model},
        )


class ProviderError(AppError):
    def __init__(
        self,
        message: str,
        provider: str,
        status_code: int = 502,
        details: dict[str, Any] | None = None,
    ):
        super().__init__(
            message=message,
            status_code=status_code,
            error_code="PROVIDER_ERROR",
            details=details or {"provider": provider},
        )


class ProviderTimeoutError(ProviderError):
    def __init__(self, provider: str, timeout: int):
        super().__init__(
            message=f"Provider '{provider}' timed out after {timeout}s",
            provider=provider,
            status_code=504,
            details={"timeout": timeout},
        )


class CircuitBreakerOpenError(AppError):
    def __init__(self, provider: str):
        super().__init__(
            message=f"Circuit breaker is open for provider '{provider}'",
            status_code=503,
            error_code="CIRCUIT_BREAKER_OPEN",
            details={"provider": provider},
        )


class ValidationError(AppError):
    def __init__(self, message: str, field: str | None = None):
        super().__init__(
            message=message,
            status_code=400,
            error_code="VALIDATION_ERROR",
            details={"field": field} if field else None,
        )


class NotFoundError(AppError):
    def __init__(self, resource: str, identifier: str):
        super().__init__(
            message=f"{resource} not found: {identifier}",
            status_code=404,
            error_code="NOT_FOUND",
            details={"resource": resource, "identifier": identifier},
        )


class DuplicateResourceError(AppError):
    def __init__(self, resource: str, identifier: str):
        super().__init__(
            message=f"{resource} already exists: {identifier}",
            status_code=409,
            error_code="DUPLICATE_RESOURCE",
            details={"resource": resource, "identifier": identifier},
        )


class InsufficientCreditsError(AppError):
    def __init__(
        self,
        required: float = 0.0,
        available: float = 0.0,
    ):
        super().__init__(
            message="Insufficient credits for this request",
            status_code=402,
            error_code="INSUFFICIENT_CREDITS",
            details={
                "required": required,
                "available": available,
                "shortfall": max(0.0, required - available),
            },
        )

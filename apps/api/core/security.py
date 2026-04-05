import hashlib
import hmac
import secrets
from datetime import UTC, datetime, timedelta
from typing import Any
from uuid import UUID

from jose import JWTError, jwt
from passlib.context import CryptContext
from apps.api.core.config import get_settings
from packages.redis.client import get_redis
from packages.shared.exceptions import AuthenticationError

pwd_context = CryptContext(
    schemes=["argon2"],
    deprecated="auto",
    argon2__rounds=3,
    argon2__memory_cost=131072,
    argon2__parallelism=4,
)

API_KEY_SECRET = b"routing-run-api-key-hmac-secret-do-not-change"


def generate_api_key(prefix: str | None = None, length: int = 32) -> tuple[str, str]:
    settings = get_settings()
    prefix = prefix or settings.api_key_prefix
    random_bytes = secrets.token_bytes(length)
    key = f"{prefix}{random_bytes.hex()}"
    key_hash = hash_api_key(key)
    return key, key_hash


def hash_api_key(key: str) -> str:
    return hmac.new(API_KEY_SECRET, key.encode(), hashlib.sha256).hexdigest()


def verify_api_key(plain_key: str, hashed_key: str) -> bool:
    return hmac.compare_digest(hash_api_key(plain_key), hashed_key)


def hash_password(password: str) -> str:
    return pwd_context.hash(password)


def verify_password(plain_password: str, hashed_password: str) -> bool:
    return pwd_context.verify(plain_password, hashed_password)


def create_access_token(
    subject: str | UUID,
    expires_delta: timedelta | None = None,
    additional_claims: dict[str, Any] | None = None,
) -> str:
    settings = get_settings()

    if isinstance(subject, UUID):
        subject = str(subject)

    if expires_delta:
        expire = datetime.now(UTC) + expires_delta
    else:
        expire = datetime.now(UTC) + timedelta(minutes=settings.access_token_expire_minutes)

    jti = secrets.token_urlsafe(16)

    to_encode: dict[str, Any] = {
        "exp": expire,
        "sub": subject,
        "type": "access",
        "jti": jti,
    }

    if additional_claims:
        to_encode.update(additional_claims)

    encoded_jwt = jwt.encode(
        to_encode,
        settings.jwt.secret_key,
        algorithm=settings.jwt.algorithm,
    )
    return encoded_jwt


def create_refresh_token(
    subject: str | UUID,
    expires_delta: timedelta | None = None,
) -> str:
    settings = get_settings()

    if isinstance(subject, UUID):
        subject = str(subject)

    if expires_delta:
        expire = datetime.now(UTC) + expires_delta
    else:
        expire = datetime.now(UTC) + timedelta(days=settings.refresh_token_expire_days)

    jti = secrets.token_urlsafe(16)

    to_encode = {
        "exp": expire,
        "sub": subject,
        "type": "refresh",
        "jti": jti,
    }

    encoded_jwt = jwt.encode(
        to_encode,
        settings.jwt.secret_key,
        algorithm=settings.jwt.algorithm,
    )
    return encoded_jwt


def decode_token(token: str) -> dict[str, Any]:
    settings = get_settings()
    try:
        payload = jwt.decode(
            token,
            settings.jwt.secret_key,
            algorithms=[settings.jwt.algorithm],
        )
        return payload
    except JWTError as e:
        raise AuthenticationError(f"Invalid token: {str(e)}")


async def verify_access_token(token: str) -> dict[str, Any]:
    payload = decode_token(token)
    if payload.get("type") != "access":
        raise AuthenticationError("Invalid token type")

    jti = payload.get("jti")
    if jti and await is_token_blacklisted(jti):
        raise AuthenticationError("Token has been revoked")

    return payload


async def is_token_blacklisted(jti: str) -> bool:
    redis = await get_redis()
    return await redis.exists(f"blacklist:token:{jti}") == 1


async def blacklist_token(jti: str, expires_in_seconds: int) -> None:
    redis = await get_redis()
    await redis.setex(f"blacklist:token:{jti}", expires_in_seconds, "1")


async def blacklist_refresh_token(jti: str, user_id: str) -> None:
    redis = await get_redis()
    key = f"used_refresh:{jti}"
    await redis.set(key, user_id, ex=86400 * 30)


async def is_refresh_token_used(jti: str) -> bool:
    redis = await get_redis()
    return await redis.exists(f"used_refresh:{jti}") == 1


def verify_refresh_token(token: str) -> dict[str, Any]:
    payload = decode_token(token)
    if payload.get("type") != "refresh":
        raise AuthenticationError("Invalid token type")
    return payload


def generate_verification_code(length: int = 6) -> str:
    return "".join(secrets.choice("0123456789") for _ in range(length))


def generate_state_token() -> str:
    return secrets.token_urlsafe(32)


async def store_oauth_state(state: str, user_id: str | None = None, expires: int = 600) -> None:
    redis = await get_redis()
    await redis.setex(f"oauth_state:{state}", expires, user_id or "")


async def verify_oauth_state(state: str) -> str | None:
    redis = await get_redis()
    value = await redis.get(f"oauth_state:{state}")
    await redis.delete(f"oauth_state:{state}")
    return value


def generate_csrf_token() -> str:
    return secrets.token_urlsafe(32)


async def store_csrf_token(token: str, user_id: str, expires: int = 3600) -> None:
    redis = await get_redis()
    await redis.setex(f"csrf:{user_id}:{token}", expires, "1")


async def verify_csrf_token(token: str, user_id: str) -> bool:
    redis = await get_redis()
    key = f"csrf:{user_id}:{token}"
    exists = await redis.exists(key)
    if exists:
        await redis.delete(key)
    return exists == 1


def create_token_response(
    access_token: str,
    refresh_token: str,
    user_data: dict,
) -> dict:
    settings = get_settings()
    access_expire = settings.access_token_expire_minutes * 60
    refresh_expire = settings.refresh_token_expire_days * 24 * 60 * 60
    return {
        "access_token": access_token,
        "refresh_token": refresh_token,
        "expires_in": access_expire,
        "user": user_data,
    }

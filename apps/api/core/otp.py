import secrets
from apps.api.core.config import get_settings


def generate_otp(length: int = 6) -> str:
    return "".join(secrets.choice("0123456789") for _ in range(length))


async def store_otp(email: str, otp: str, purpose: str, expires: int = 600) -> None:
    from packages.redis.client import get_redis

    redis = await get_redis()
    await redis.setex(f"otp:{purpose}:{email}", expires, otp)


async def verify_otp(email: str, otp: str, purpose: str) -> bool:
    from packages.redis.client import get_redis

    redis = await get_redis()
    stored = await redis.get(f"otp:{purpose}:{email}")
    if stored and stored == otp:
        await redis.delete(f"otp:{purpose}:{email}")
        return True
    return False

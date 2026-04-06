import secrets

from packages.shared.exceptions import RateLimitError


def generate_otp(length: int = 6) -> str:
    return "".join(secrets.choice("0123456789") for _ in range(length))


async def store_otp(email: str, otp: str, purpose: str, expires: int = 600) -> None:
    from packages.redis.client import get_redis

    redis = await get_redis()
    await redis.setex(f"otp:{purpose}:{email}", expires, otp)


async def get_otp_failures(email: str, purpose: str) -> tuple[int, int]:
    from packages.redis.client import get_redis

    redis = await get_redis()
    key = f"otp_failures:{purpose}:{email}"
    count = await redis.get(key)
    ttl = await redis.ttl(key)
    return int(count) if count else 0, ttl


async def increment_otp_failures(email: str, purpose: str) -> int:
    from packages.redis.client import get_redis

    redis = await get_redis()
    key = f"otp_failures:{purpose}:{email}"
    count = await redis.incr(key)
    if count == 1:
        await redis.expire(key, 1800)
    return count


async def reset_otp_failures(email: str, purpose: str) -> None:
    from packages.redis.client import get_redis

    redis = await get_redis()
    key = f"otp_failures:{purpose}:{email}"
    await redis.delete(key)


OTP_LOCKOUT_THRESHOLDS = [
    (3, 30),
    (5, 120),
    (7, 600),
    (10, 1800),
]


async def check_otp_lockout(email: str, purpose: str) -> tuple[bool, int]:
    failures, ttl = await get_otp_failures(email, purpose)
    for threshold, delay in OTP_LOCKOUT_THRESHOLDS:
        if failures >= threshold:
            remaining = max(ttl, delay) if ttl > 0 else delay
            return True, remaining
    return False, 0


async def verify_otp(email: str, otp: str, purpose: str) -> bool:
    from packages.redis.client import get_redis

    is_locked, retry_after = await check_otp_lockout(email, purpose)
    if is_locked:
        raise RateLimitError(
            message=f"Too many failed attempts. Wait {retry_after} seconds before trying again.",
            retry_after=retry_after,
        )

    redis = await get_redis()
    stored = await redis.get(f"otp:{purpose}:{email}")
    if stored and secrets.compare_digest(stored, otp):
        await redis.delete(f"otp:{purpose}:{email}")
        await reset_otp_failures(email, purpose)
        return True

    await increment_otp_failures(email, purpose)
    return False

import logging

import httpx

from apps.api.core.config import get_settings

logger = logging.getLogger("routing.run.api")

TURNSTILE_VERIFY_URL = "https://challenges.cloudflare.com/turnstile/v0/siteverify"


async def verify_turnstile(token: str | None, remote_ip: str | None = None) -> bool:
    settings = get_settings()

    if not settings.turnstile_secret_key:
        logger.warning("Turnstile secret key not configured, skipping verification")
        return True

    if not token:
        logger.warning("Turnstile token missing")
        return False

    data = {
        "secret": settings.turnstile_secret_key,
        "response": token,
    }
    if remote_ip:
        data["remoteip"] = remote_ip

    try:
        async with httpx.AsyncClient(timeout=10) as client:
            response = await client.post(TURNSTILE_VERIFY_URL, data=data)
            result = response.json()

            if result.get("success"):
                return True

            error_codes = result.get("error-codes", [])
            logger.warning(f"Turnstile verification failed: {error_codes}")
            return False
    except Exception:
        logger.exception("Turnstile verification request failed")
        return False

import httpx
from sqlalchemy import select, text
from packages.db.session import create_session_factory


async def notify_new_users_task():
    session_factory = create_session_factory()
    async with session_factory() as session:
        result = await session.execute(
            text("""
                SELECT id, email, name, plan_tier, created_at 
                FROM users 
                WHERE created_at > NOW() - INTERVAL '1 minute'
                ORDER BY created_at DESC
            """)
        )
        new_users = result.fetchall()

        if new_users:
            webhook_url = _get_webhook_url()
            if webhook_url:
                for user in new_users:
                    await _send_discord_notification(webhook_url, user)

        return {"new_users_count": len(new_users)}


def _get_webhook_url() -> str | None:
    from apps.api.core.config import get_settings

    settings = get_settings()
    return settings.discord_webhook_url or None


async def _send_discord_notification(webhook_url: str, user) -> None:
    import logging

    logger = logging.getLogger("routing.run.worker")

    embed = {
        "title": "🆕 New User Signed Up!",
        "color": 5814783,
        "fields": [
            {"name": "Email", "value": user.email or "N/A", "inline": True},
            {"name": "Name", "value": user.name or "N/A", "inline": True},
            {
                "name": "Plan",
                "value": str(
                    user.plan_tier.value if hasattr(user.plan_tier, "value") else user.plan_tier
                ),
                "inline": True,
            },
            {"name": "User ID", "value": str(user.id), "inline": False},
        ],
        "footer": {"text": "Routing.Run"},
    }

    payload = {"embeds": [embed]}

    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(webhook_url, json=payload, timeout=10.0)
            if response.status_code not in (200, 204):
                logger.error(
                    f"Failed to send Discord notification: {response.status_code} {response.text}"
                )
    except Exception as e:
        logger.error(f"Error sending Discord notification: {e}")


WORKER_FUNCTIONS = []

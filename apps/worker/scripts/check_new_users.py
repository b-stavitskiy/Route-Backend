#!/usr/bin/env python3
"""
Standalone script to check for new users and notify Discord.
Run via cron: * * * * * /path/to/python /path/to/check_new_users.py
"""
import asyncio
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


async def main():
    from packages.db.session import create_session_factory
    import httpx
    from sqlalchemy import text
    from apps.api.core.config import get_settings

    settings = get_settings()
    webhook_url = settings.discord_webhook_url

    if not webhook_url:
        print("DISCORD_WEBHOOK_URL not set, skipping")
        return

    session_factory = create_session_factory()
    async with session_factory() as session:
        result = await session.execute(
            text("SELECT COUNT(*) FROM users")
        )
        total_users = result.scalar()

        result = await session.execute(
            text("""
                SELECT id, email, name, plan_tier, created_at 
                FROM users 
                WHERE created_at > NOW() - INTERVAL '1 minute'
                ORDER BY created_at DESC
            """)
        )
        new_users = result.fetchall()

        if not new_users:
            print("No new users")
            return

        for user in new_users:
            plan = user.plan_tier.value if hasattr(user.plan_tier, 'value') else user.plan_tier
            print(f"New user: {user.email} ({plan})")

            embed = {
                "title": "🆕 New User Signed Up!",
                "color": 5814783,
                "fields": [
                    {"name": "Email", "value": user.email or "N/A", "inline": True},
                    {"name": "Name", "value": user.name or "N/A", "inline": True},
                    {"name": "Plan", "value": str(plan), "inline": True},
                    {"name": "Total Users", "value": str(total_users), "inline": False},
                    {"name": "User ID", "value": str(user.id), "inline": False},
                ],
                "footer": {"text": "Routing.Run"},
            }

            payload = {"embeds": [embed]}

            async with httpx.AsyncClient() as client:
                response = await client.post(webhook_url, json=payload, timeout=10.0)
                if response.status_code in (200, 204):
                    print(f"Notified Discord for {user.email}")
                else:
                    print(f"Failed to notify Discord: {response.status_code}")

    print(f"Processed {len(new_users)} new users")


if __name__ == "__main__":
    asyncio.run(main())

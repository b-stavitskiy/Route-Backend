import hashlib
import hmac
import logging
from datetime import datetime, timedelta, timezone
from typing import Any

import httpx
from fastapi import APIRouter, Header, Request
from pydantic import BaseModel
from sqlalchemy import text
from apps.api.core.config import get_settings
from apps.api.services.auth_service import AuthService
from packages.db.models import PlanTier, User
from packages.db.session import get_db_session
from packages.shared.exceptions import AuthenticationError

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/webhooks", tags=["webhooks"])


class WhopWebhookEvent(BaseModel):
    event: str
    data: dict[str, Any]


def verify_whop_signature(
    payload: bytes,
    signature: str,
    secret: str,
) -> bool:
    expected = hmac.new(
        secret.encode(),
        payload,
        hashlib.sha256,
    ).hexdigest()
    return hmac.compare_digest(f"sha256={expected}", signature)


@router.post("/whop")
async def handle_whop_webhook(
    request: Request,
):
    settings = get_settings()
    body = await request.body()
    signature = request.headers.get("x-whop-signature", "")

    if settings.whop.webhook_secret:
        if not signature:
            logger.warning("Missing webhook signature")
            raise AuthenticationError("Missing webhook signature")
        if not verify_whop_signature(body, signature, settings.whop.webhook_secret):
            logger.warning("Invalid webhook signature")
            raise AuthenticationError("Invalid webhook signature")

    event_data = await request.json()
    event_type = event_data.get("event", "")

    logger.info(f"Received Whop webhook: {event_type}")

    async with get_db_session() as session:
        auth_service = AuthService(session)

        if event_type == "membership.activated":
            await handle_membership_activated(session, auth_service, event_data)

        elif event_type == "membership.deactivated":
            await handle_membership_deactivated(session, auth_service, event_data)

        elif event_type == "membership.cancel_at_period_end_changed":
            await handle_membership_cancel_at_period_end_changed(session, auth_service, event_data)

        elif event_type == "payment.succeeded":
            await handle_payment_succeeded(session, auth_service, event_data)

        elif event_type == "payment.failed":
            await handle_payment_failed(session, auth_service, event_data)

        return {"received": True}


async def handle_membership_activated(
    session,
    auth_service: AuthService,
    event_data: dict,
):
    data = event_data.get("data", {})
    attributes = data.get("attributes", {})

    email = attributes.get("email")
    user_id = attributes.get("user_id")
    plan_id = attributes.get("plan_id")
    steaml = attributes.get("steaml", False)
    expired_at = attributes.get("expired_at")

    if not email and not user_id:
        logger.warning("membership.activated: missing email and user_id")
        return

    plan_tier = get_plan_tier_from_whop(plan_id)
    logger.info(
        f"membership.activated: email={email}, user_id={user_id}, plan={plan_tier}, steaml={steaml}"
    )

    if expired_at:
        try:
            expiry_date = datetime.fromisoformat(expired_at.replace("Z", "+00:00"))
        except (ValueError, AttributeError):
            expiry_date = datetime.now(timezone.utc) + timedelta(days=30)
    else:
        expiry_date = datetime.now(timezone.utc) + timedelta(days=30)

    async def update_user(user):
        user.whop_user_id = user_id
        user.plan_tier = plan_tier
        user.upgraded_to_tier = plan_tier
        user.upgraded_until = expiry_date
        await session.commit()
        logger.info(f"Updated user {user.id} to plan {plan_tier} (expires={expiry_date})")

    if email:
        user = await auth_service.get_user_by_email(email)
        if user:
            await update_user(user)
            return

    if user_id:
        from packages.db.session import get_db_session

        async with get_db_session() as new_session:
            auth_service_new = AuthService(new_session)
            user = await auth_service_new.get_user_by_whop_id(user_id)
            if user:
                user.plan_tier = plan_tier
                user.upgraded_to_tier = plan_tier
                user.upgraded_until = expiry_date
                await new_session.commit()
                logger.info(
                    f"Updated user {user.id} by whop_user_id to plan {plan_tier} (expires={expiry_date})"
                )


async def handle_membership_deactivated(
    session,
    auth_service: AuthService,
    event_data: dict,
):
    data = event_data.get("data", {})
    attributes = data.get("attributes", {})

    email = attributes.get("email")
    user_id = attributes.get("user_id")

    if not email and not user_id:
        logger.warning("membership.deactivated: missing email and user_id")
        return

    logger.info(
        f"membership.deactivated: email={email}, user_id={user_id}. "
        "Access is controlled by upgraded_until, not downgrading here."
    )


async def handle_membership_cancel_at_period_end_changed(
    session,
    auth_service: AuthService,
    event_data: dict,
):
    data = event_data.get("data", {})
    attributes = data.get("attributes", {})

    email = attributes.get("email")
    user_id = attributes.get("user_id")
    cancel_at_period_end = attributes.get("cancel_at_period_end", False)

    logger.info(
        f"membership.cancel_at_period_end_changed: email={email}, cancel={cancel_at_period_end}"
    )


async def handle_payment_succeeded(
    session,
    auth_service: AuthService,
    event_data: dict,
):
    data = event_data.get("data", {})
    attributes = data.get("attributes", {})

    email = attributes.get("email")
    user_id = attributes.get("user_id")
    plan_id = attributes.get("plan_id")

    logger.info(f"payment.succeeded: email={email}, user_id={user_id}, plan_id={plan_id}")

    if not email and not user_id:
        logger.warning("payment.succeeded: missing email and user_id")
        return

    plan_tier = get_plan_tier_from_whop(plan_id)
    new_expiry = datetime.now(timezone.utc) + timedelta(days=30)

    if email:
        user = await auth_service.get_user_by_email(email)
        if user:
            user.plan_tier = plan_tier
            user.upgraded_to_tier = plan_tier
            user.upgraded_until = new_expiry
            await session.commit()
            logger.info(f"Renewed user {user.id} to plan {plan_tier} (expires={new_expiry})")
            return

    if user_id:
        user = await auth_service.get_user_by_whop_id(user_id)
        if user:
            user.plan_tier = plan_tier
            user.upgraded_to_tier = plan_tier
            user.upgraded_until = new_expiry
            await session.commit()
            logger.info(
                f"Renewed user {user.id} by whop_user_id to plan {plan_tier} (expires={new_expiry})"
            )


async def handle_payment_failed(
    session,
    auth_service: AuthService,
    event_data: dict,
):
    data = event_data.get("data", {})
    attributes = data.get("attributes", {})

    email = attributes.get("email")
    user_id = attributes.get("user_id")

    logger.info(f"payment.failed: email={email}, user_id={user_id}")


def get_plan_tier_from_whop(plan_id: str) -> PlanTier:
    settings = get_settings()

    if plan_id == settings.whop.lite_product_id:
        return PlanTier.LITE
    if plan_id == settings.whop.premium_product_id:
        return PlanTier.PREMIUM
    if plan_id == settings.whop.max_product_id:
        return PlanTier.MAX

    logger.warning(f"Unknown plan_id: {plan_id}, defaulting to FREE")
    return PlanTier.FREE


@router.post("/cron/new-users")
async def cron_notify_new_users(
    x_admin_key: str = Header(..., alias="X-Admin-Key"),
):
    settings = get_settings()

    if not settings.admin_api_key or x_admin_key != settings.admin_api_key:
        raise AuthenticationError("Invalid admin key")

    webhook_url = settings.discord_webhook_url
    if not webhook_url:
        return {"status": "skipped", "reason": "No webhook configured"}

    async with get_db_session() as session:
        result = await session.execute(text("SELECT COUNT(*) FROM users"))
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

    async with httpx.AsyncClient() as client:
        if new_users:
            for user in new_users:
                plan = user.plan_tier.value if hasattr(user.plan_tier, "value") else user.plan_tier
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
                await client.post(webhook_url, json={"embeds": [embed]}, timeout=10.0)
        else:
            embed = {
                "title": "✅ Cron Check - No New Users",
                "color": 5814783,
                "fields": [
                    {"name": "Total Users", "value": str(total_users), "inline": False},
                    {"name": "New Users (last min)", "value": "0", "inline": False},
                ],
                "footer": {"text": "Routing.Run - Cron Active"},
            }
            await client.post(webhook_url, json={"embeds": [embed]}, timeout=10.0)

    logger.info(f"Cron: notified {len(new_users)} new users to Discord, total_users={total_users}")
    return {"status": "ok", "new_users": len(new_users), "total_users": total_users}

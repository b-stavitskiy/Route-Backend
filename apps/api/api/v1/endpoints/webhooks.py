import hashlib
import hmac
from typing import Any

from fastapi import APIRouter, Request
from pydantic import BaseModel
from apps.api.core.config import get_settings
from apps.api.services.auth_service import AuthService
from packages.db.models import PlanTier, User
from packages.db.session import get_db_session
from packages.shared.exceptions import AuthenticationError

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

    if not settings.whop.webhook_secret:
        raise AuthenticationError("Webhook secret not configured")
    if not verify_whop_signature(body, signature, settings.whop.webhook_secret):
        raise AuthenticationError("Invalid webhook signature")

    event_data = await request.json()
    event_type = event_data.get("event", "")

    async with get_db_session() as session:
        auth_service = AuthService(session)

        if event_type == "checkout.completed":
            await handle_checkout_completed(session, auth_service, event_data)

        elif event_type == "subscription.updated":
            await handle_subscription_updated(session, auth_service, event_data)

        elif event_type == "subscription.cancelled":
            await handle_subscription_cancelled(session, auth_service, event_data)

        elif event_type == "membership.deleted":
            await handle_membership_deleted(session, auth_service, event_data)

        return {"received": True}


async def handle_checkout_completed(
    session,
    auth_service: AuthService,
    event_data: dict,
):

    data = event_data.get("data", {})
    attributes = data.get("attributes", {})

    email = attributes.get("email")
    whop_user_id = attributes.get("id")
    plan_id = attributes.get("plan_id")

    if not email:
        return

    plan_tier = get_plan_tier_from_whop(plan_id)

    existing_user = await auth_service.get_user_by_email(email)

    if existing_user:
        existing_user.whop_user_id = whop_user_id
        existing_user.plan_tier = plan_tier
    else:
        user = User(
            email=email,
            whop_user_id=whop_user_id,
            plan_tier=plan_tier,
            email_verified=True,
        )
        session.add(user)

    await session.commit()


async def handle_subscription_updated(
    session,
    auth_service: AuthService,
    event_data: dict,
):
    data = event_data.get("data", {})
    attributes = data.get("attributes", {})

    email = attributes.get("email")
    whop_user_id = attributes.get("id")
    plan_id = attributes.get("plan_id")

    if not email:
        return

    plan_tier = get_plan_tier_from_whop(plan_id)

    user = await auth_service.get_user_by_email(email)

    if user:
        user.whop_user_id = whop_user_id
        user.plan_tier = plan_tier
        await session.commit()


async def handle_subscription_cancelled(
    session,
    auth_service: AuthService,
    event_data: dict,
):
    data = event_data.get("data", {})
    attributes = data.get("attributes", {})

    email = attributes.get("email")

    if not email:
        return

    user = await auth_service.get_user_by_email(email)

    if user:
        user.plan_tier = PlanTier.FREE
        await session.commit()


async def handle_membership_deleted(
    session,
    auth_service: AuthService,
    event_data: dict,
):
    data = event_data.get("data", {})
    attributes = data.get("attributes", {})

    email = attributes.get("email")

    if not email:
        return

    user = await auth_service.get_user_by_email(email)

    if user:
        user.plan_tier = PlanTier.FREE
        user.whop_user_id = None
        await session.commit()


def get_plan_tier_from_whop(plan_id: str) -> PlanTier:
    settings = get_settings()

    if plan_id == settings.whop.client_id:
        return PlanTier.FREE
    if plan_id == getattr(settings.whop, "lite_product_id", ""):
        return PlanTier.LITE
    if plan_id == getattr(settings.whop, "premium_product_id", ""):
        return PlanTier.PREMIUM
    if plan_id == getattr(settings.whop, "max_product_id", ""):
        return PlanTier.MAX

    return PlanTier.FREE

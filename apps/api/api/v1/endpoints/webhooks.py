import hashlib
import hmac
import logging
from typing import Any

from fastapi import APIRouter, Request
from pydantic import BaseModel
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

    if settings.whop.webhook_secret and signature:
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

    if not email and not user_id:
        logger.warning("membership.activated: missing email and user_id")
        return

    plan_tier = get_plan_tier_from_whop(plan_id)
    logger.info(
        f"membership.activated: email={email}, user_id={user_id}, plan={plan_tier}, steaml={steaml}"
    )

    if email:
        user = await auth_service.get_user_by_email(email)
        if user:
            user.whop_user_id = user_id
            user.plan_tier = plan_tier
            await session.commit()
            logger.info(f"Updated user {user.id} to plan {plan_tier} (steaml={steaml})")
            return

    if user_id:
        from packages.db.session import get_db_session

        async with get_db_session() as new_session:
            auth_service_new = AuthService(new_session)
            user = await auth_service_new.get_user_by_whop_id(user_id)
            if user:
                user.plan_tier = plan_tier
                await new_session.commit()
                logger.info(
                    f"Updated user {user.id} by whop_user_id to plan {plan_tier} (steaml={steaml})"
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

    logger.info(f"membership.deactivated: email={email}, user_id={user_id}")

    if email:
        user = await auth_service.get_user_by_email(email)
        if user:
            user.plan_tier = PlanTier.FREE
            await session.commit()
            logger.info(f"Downgraded user {user.id} to FREE")
            return

    if user_id:
        user = await auth_service.get_user_by_whop_id(user_id)
        if user:
            user.plan_tier = PlanTier.FREE
            await session.commit()
            logger.info(f"Downgraded user {user.id} by whop_user_id to FREE")


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

from typing import Optional

from fastapi import APIRouter, Depends, Request
from pydantic import BaseModel
from apps.api.core.config import get_settings
from apps.api.services.whop.service import get_whop_service
from packages.shared.exceptions import AppError

router = APIRouter(prefix="/whop", tags=["whop"])


class CreateCheckoutRequest(BaseModel):
    plan_tier: str
    email: Optional[str] = None
    return_url: Optional[str] = None


class CheckoutResponse(BaseModel):
    checkout_url: str
    session_id: str


def get_plan_id_from_tier(tier: str) -> str:
    settings = get_settings()
    tier_to_plan = {
        "lite": settings.whop.lite_product_id,
        "premium": settings.whop.premium_product_id,
        "max": settings.whop.max_product_id,
    }
    plan_id = tier_to_plan.get(tier.lower())
    if not plan_id:
        raise AppError(f"No product configured for tier: {tier}", 400)
    return plan_id


@router.post("/checkout", response_model=CheckoutResponse)
async def create_checkout_session(
    request: CreateCheckoutRequest,
    req: Request,
):
    settings = get_settings()

    if not settings.whop.api_key:
        raise AppError("Whop API key not configured", 500)

    plan_id = get_plan_id_from_tier(request.plan_tier)

    whop_service = get_whop_service()

    return_url = request.return_url or str(req.url_for("checkout_complete"))

    result = await whop_service.create_checkout_session(
        plan_id=plan_id,
        email=request.email,
        return_url=return_url,
    )

    return CheckoutResponse(
        checkout_url=result.get("url", ""),
        session_id=result.get("id", ""),
    )


@router.get("/products")
async def list_whop_products():
    settings = get_settings()

    if not settings.whop.api_key:
        raise AppError("Whop API key not configured", 500)

    whop_service = get_whop_service()
    result = await whop_service.list_products()

    return result


async def checkout_complete(request: Request):
    return {"status": "success"}

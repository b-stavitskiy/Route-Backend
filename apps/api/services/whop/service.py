import httpx
from apps.api.core.config import get_settings, WhopSettings


class WhopService:
    def __init__(self, settings: WhopSettings | None = None):
        if settings is None:
            settings = get_settings().whop
        self.settings = settings
        self.base_url = settings.api_base_url

    async def create_checkout_session(
        self,
        plan_id: str,
        email: str | None = None,
        return_url: str | None = None,
    ) -> dict:
        headers = {
            "Authorization": f"Bearer {self.settings.api_key}",
            "Content-Type": "application/json",
        }

        payload = {
            "plan_id": plan_id,
        }

        if email:
            payload["email"] = email
        if return_url:
            payload["return_url"] = return_url

        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self.base_url}/api/v1/payments/checkout_sessions",
                headers=headers,
                json=payload,
                timeout=30.0,
            )
            response.raise_for_status()
            return response.json()

    async def get_product(self, product_id: str) -> dict:
        headers = {
            "Authorization": f"Bearer {self.settings.api_key}",
        }

        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{self.base_url}/api/v1/products/{product_id}",
                headers=headers,
                timeout=30.0,
            )
            response.raise_for_status()
            return response.json()

    async def list_products(self) -> dict:
        headers = {
            "Authorization": f"Bearer {self.settings.api_key}",
        }

        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{self.base_url}/api/v1/products",
                headers=headers,
                timeout=30.0,
            )
            response.raise_for_status()
            return response.json()

    async def get_plan(self, plan_id: str) -> dict:
        headers = {
            "Authorization": f"Bearer {self.settings.api_key}",
        }

        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{self.base_url}/api/v1/plans/{plan_id}",
                headers=headers,
                timeout=30.0,
            )
            response.raise_for_status()
            return response.json()


def get_whop_service() -> WhopService:
    return WhopService()

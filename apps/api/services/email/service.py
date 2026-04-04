import httpx
from apps.api.core.config import get_settings


class EmailService:
    def __init__(self):
        self.settings = get_settings()
        self.api_key = self.settings.resend_api_key
        self.from_email = self.settings.resend_from_email or self.settings.from_email
        self.from_name = self.settings.from_name

    async def send_email(self, to: list[str], subject: str, html: str) -> dict:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                "https://api.resend.com/emails",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "from": f"{self.from_name} <{self.from_email}>",
                    "to": to,
                    "subject": subject,
                    "html": html,
                },
                timeout=30,
            )
            return response.json()

    async def send_otp(self, email: str, otp: str, purpose: str) -> dict:
        subjects = {
            "signup": "Verify your email - Routing.Run",
            "login": "Your login code - Routing.Run",
        }
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="utf-8">
            <style>
                body {{ font-family: Arial, sans-serif; line-height: 1.6; color: #333; }}
                .container {{ max-width: 600px; margin: 0 auto; padding: 20px; }}
                .header {{ background: #E3A514; color: white; padding: 20px; text-align: center; }}
                .content {{ padding: 30px 20px; background: #f9f9f9; }}
                .code {{ font-size: 32px; font-weight: bold; letter-spacing: 8px; text-align: center; padding: 20px; background: white; border: 2px dashed #E3A514; margin: 20px 0; }}
                .footer {{ padding: 20px; text-align: center; font-size: 12px; color: #666; }}
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1>Routing.Run</h1>
                </div>
                <div class="content">
                    <p>Your verification code is:</p>
                    <div class="code">{otp}</div>
                    <p>This code expires in 10 minutes.</p>
                    <p>If you didn't request this, please ignore this email.</p>
                </div>
                <div class="footer">
                    <p>&copy; 2026 Routing.Run - AI Model Routing Platform</p>
                </div>
            </div>
        </body>
        </html>
        """
        return await self.send_email(
            [email], subjects.get(purpose, "Your code - Routing.Run"), html
        )

    async def send_welcome(self, email: str, name: str | None = None) -> dict:
        greeting = f"Hi {name}," if name else "Hi there,"
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="utf-8">
            <style>
                body {{ font-family: Arial, sans-serif; line-height: 1.6; color: #333; }}
                .container {{ max-width: 600px; margin: 0 auto; padding: 20px; }}
                .header {{ background: #E3A514; color: white; padding: 20px; text-align: center; }}
                .content {{ padding: 30px 20px; background: #f9f9f9; }}
                .footer {{ padding: 20px; text-align: center; font-size: 12px; color: #666; }}
                .btn {{ display: inline-block; background: #E3A514; color: white; padding: 12px 30px; text-decoration: none; border-radius: 5px; margin: 20px 0; }}
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1>Welcome to Routing.Run!</h1>
                </div>
                <div class="content">
                    <p>{greeting}</p>
                    <p>Thanks for signing up! You now have access to our AI model routing platform with 5 free credits to get started.</p>
                    <p>With Routing.Run, you can access multiple AI models through a single API endpoint.</p>
                    <p>Get started by creating your first API key in the dashboard.</p>
                    <a href="https://app.routing.run/dashboard/keys" class="btn">Create API Key</a>
                </div>
                <div class="footer">
                    <p>&copy; 2026 Routing.Run - AI Model Routing Platform</p>
                </div>
            </div>
        </body>
        </html>
        """
        return await self.send_email([email], "Welcome to Routing.Run!", html)

    async def send_login_alert(self, email: str, ip: str, location: str | None = None) -> dict:
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="utf-8">
            <style>
                body {{ font-family: Arial, sans-serif; line-height: 1.6; color: #333; }}
                .container {{ max-width: 600px; margin: 0 auto; padding: 20px; }}
                .header {{ background: #E3A514; color: white; padding: 20px; text-align: center; }}
                .content {{ padding: 30px 20px; background: #f9f9f9; }}
                .details {{ background: white; padding: 15px; border-left: 4px solid #E3A514; margin: 20px 0; }}
                .footer {{ padding: 20px; text-align: center; font-size: 12px; color: #666; }}
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1>New Login Detected</h1>
                </div>
                <div class="content">
                    <p>We noticed a new login to your Routing.Run account:</p>
                    <div class="details">
                        <p><strong>IP Address:</strong> {ip}</p>
                        {f"<p><strong>Location:</strong> {location}</p>" if location else ""}
                        <p><strong>Time:</strong> Just now</p>
                    </div>
                    <p>If this was you, no action needed.</p>
                    <p>If this wasn't you, please secure your account immediately.</p>
                </div>
                <div class="footer">
                    <p>&copy; 2026 Routing.Run - AI Model Routing Platform</p>
                </div>
            </div>
        </body>
        </html>
        """
        return await self.send_email([email], "New Login to Your Account - Routing.Run", html)

    async def send_api_key_notification(
        self, email: str, action: str, key_name: str, ip: str
    ) -> dict:
        actions = {
            "created": ("API Key Created", "A new API key has been created for your account."),
            "deleted": ("API Key Deleted", "An API key has been deleted from your account."),
            "updated": ("API Key Updated", "An API key has been updated."),
        }
        subject, message = actions.get(action, ("API Key Notification", ""))
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="utf-8">
            <style>
                body {{ font-family: Arial, sans-serif; line-height: 1.6; color: #333; }}
                .container {{ max-width: 600px; margin: 0 auto; padding: 20px; }}
                .header {{ background: #E3A514; color: white; padding: 20px; text-align: center; }}
                .content {{ padding: 30px 20px; background: #f9f9f9; }}
                .details {{ background: white; padding: 15px; border-left: 4px solid #E3A514; margin: 20px 0; }}
                .footer {{ padding: 20px; text-align: center; font-size: 12px; color: #666; }}
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1>{subject}</h1>
                </div>
                <div class="content">
                    <p>{message}</p>
                    <div class="details">
                        <p><strong>Key Name:</strong> {key_name}</p>
                        <p><strong>IP Address:</strong> {ip}</p>
                        <p><strong>Time:</strong> Just now</p>
                    </div>
                    <p>If you didn't make this change, please contact support immediately.</p>
                </div>
                <div class="footer">
                    <p>&copy; 2026 Routing.Run - AI Model Routing Platform</p>
                </div>
            </div>
        </body>
        </html>
        """
        return await self.send_email([email], f"{subject} - Routing.Run", html)

    async def send_plan_upgrade(self, email: str, old_plan: str, new_plan: str) -> dict:
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="utf-8">
            <style>
                body {{ font-family: Arial, sans-serif; line-height: 1.6; color: #333; }}
                .container {{ max-width: 600px; margin: 0 auto; padding: 20px; }}
                .header {{ background: #E3A514; color: white; padding: 20px; text-align: center; }}
                .content {{ padding: 30px 20px; background: #f9f9f9; }}
                .plan-badge {{ display: inline-block; background: #E3A514; color: white; padding: 8px 20px; border-radius: 20px; font-weight: bold; }}
                .footer {{ padding: 20px; text-align: center; font-size: 12px; color: #666; }}
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1>Plan Upgraded!</h1>
                </div>
                <div class="content">
                    <p>Your Routing.Run plan has been upgraded:</p>
                    <p style="text-align: center; margin: 30px 0;">
                        <span class="plan-badge">{old_plan.upper()}</span>
                        &rarr;
                        <span class="plan-badge">{new_plan.upper()}</span>
                    </p>
                    <p>You now have access to more models and higher rate limits.</p>
                    <p>Check your new benefits in the dashboard.</p>
                </div>
                <div class="footer">
                    <p>&copy; 2026 Routing.Run - AI Model Routing Platform</p>
                </div>
            </div>
        </body>
        </html>
        """
        return await self.send_email([email], "Plan Upgraded! - Routing.Run", html)


def get_email_service() -> EmailService:
    return EmailService()

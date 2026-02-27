from email.message import EmailMessage

import aiosmtplib
import structlog

from fastapi_ollama_rag.core.config import settings

logger = structlog.get_logger(__name__)


class EmailService:
    """
    Asynchronous Email Service.
    Isolated logic for dispatching emails without blocking the FastAPI event loop.
    """

    def __init__(self):
        self.smtp_server = settings.smtp_server
        self.smtp_port = settings.smtp_port
        self.smtp_username = settings.smtp_username
        self.smtp_password = settings.smtp_password

    async def send_otp_email(self, to_email: str, otp: str) -> None:
        """Sends a 6-digit registration OTP to the specified user."""
        if not self.smtp_username or not self.smtp_password:
            logger.warning(
                "SMTP credentials missing. Mocking email send.",
                otp=otp,
                to_email=to_email,
            )
            return

        message = EmailMessage()
        message["From"] = self.smtp_username
        message["To"] = to_email
        message["Subject"] = "RAG Application Verification Code"

        body = (
            f"Welcome!\n\nYour verification code is: {otp}\n\n"
            f"This code will expire in 10 minutes."
        )
        message.set_content(body)

        try:
            logger.info("Dispatching OTP email via SMTP...", to_email=to_email)
            await aiosmtplib.send(
                message,
                hostname=self.smtp_server,
                port=self.smtp_port,
                start_tls=True,
                username=self.smtp_username,
                password=self.smtp_password,
            )
            logger.info("OTP email successfully dispatched!", to_email=to_email)
        except Exception as e:
            logger.error(
                "Failed to dispatch OTP email!!", error=str(e), to_email=to_email
            )
            raise RuntimeError("Could not dispatch email. Please try again later.")

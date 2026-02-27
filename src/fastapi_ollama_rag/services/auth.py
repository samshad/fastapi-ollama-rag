import secrets
import string
from datetime import UTC, datetime, timedelta

import structlog
from fastapi import HTTPException, status

from fastapi_ollama_rag.core import security
from fastapi_ollama_rag.repository import user_repo
from fastapi_ollama_rag.services.email_service import EmailService

logger = structlog.get_logger(__name__)
email_service = EmailService()


async def request_registration_otp(email: str) -> None:
    """Generates an OTP, saves it, and dispatches an email."""
    existing_user = await user_repo.get_user_by_email(email)
    if existing_user:
        logger.warning("OTP requested for existing email...", email=email)
        return

    otp = "".join(secrets.choice(string.digits) for _ in range(6))
    expires_at = datetime.now(UTC) + timedelta(minutes=10)

    await user_repo.save_otp(email, otp, expires_at)
    await email_service.send_otp_email(to_email=email, otp=otp)
    logger.info("OTP generated and dispatched...", email=email)


async def verify_and_register_user(email: str, otp: str, password: str) -> dict:
    """Verifies the OTP and registers the user in the database."""
    valid_otp = await user_repo.get_valid_otp(email, otp)
    if not valid_otp:
        logger.warning("Invalid or expired verification code...", email=email)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid or expired verification code.",
        )

    if await user_repo.get_user_by_email(email):
        logger.warning("User already registered...", email=email)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="User already registered."
        )

    hashed_pwd = security.get_password_hash(password)
    new_user = await user_repo.create_user(email, hashed_pwd)

    await user_repo.delete_otps_for_email(email)

    logger.info("User registered successfully...", user_id=str(new_user["id"]))
    return {"message": "User registered successfully. You may now log in."}


async def authenticate_user(email: str, password: str) -> str:
    """Validates credentials and returns a JWT."""
    user = await user_repo.get_user_by_email(email)

    invalid_creds_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Incorrect email or password",
        headers={"WWW-Authenticate": "Bearer"},
    )

    if not user or not security.verify_password(password, user["hashed_password"]):
        logger.warning("Failed login attempt", email=email)
        raise invalid_creds_exception

    access_token = security.create_access_token(data={"sub": str(user["id"])})
    logger.info("User authenticated", user_id=str(user["id"]))

    return access_token


async def request_password_reset_otp(email: str) -> None:
    """Generates a password reset OTP if the user exists."""
    user = await user_repo.get_user_by_email(email)

    if not user:
        logger.warning("Password reset requested for unregistered email", email=email)
        return

    otp = "".join(secrets.choice(string.digits) for _ in range(6))
    expires_at = datetime.now(UTC) + timedelta(minutes=10)

    await user_repo.save_otp(email, otp, expires_at)
    await email_service.send_otp_email(to_email=email, otp=otp)
    logger.info("Password reset OTP dispatched...", email=email)


async def reset_password(email: str, otp: str, new_password: str) -> dict:
    """Verifies the OTP and updates the user's password."""
    valid_otp = await user_repo.get_valid_otp(email, otp)

    if not valid_otp:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid or expired verification code.",
        )

    hashed_pwd = security.get_password_hash(new_password)
    await user_repo.update_user_password(email, hashed_pwd)

    await user_repo.delete_otps_for_email(email)

    logger.info("Password successfully reset...", email=email)
    return {"message": "Password has been reset successfully. You may now log in."}

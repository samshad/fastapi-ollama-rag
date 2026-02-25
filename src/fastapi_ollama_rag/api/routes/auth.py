import structlog
from fastapi import APIRouter, Depends, status
from fastapi.security import OAuth2PasswordRequestForm

from fastapi_ollama_rag.models.auth import (
    OTPRequest,
    PasswordResetRequest,
    RegisterRequest,
    TokenResponse,
)
from fastapi_ollama_rag.services import auth as auth_service

logger = structlog.get_logger(__name__)

router = APIRouter(prefix="/auth", tags=["Authentication"])


@router.post("/request-otp", status_code=status.HTTP_200_OK)
async def request_otp(request: OTPRequest):
    """
    Initiates the registration flow by generating and emailing a 6-digit OTP.
    Fails silently if the email is already registered to prevent enumeration.
    """
    email_str = str(request.email)
    logger.info("OTP request received", email=email_str)
    await auth_service.request_registration_otp(email_str)
    return {"message": "If the email is valid and unregistered, an OTP has been sent."}


@router.post("/register", status_code=status.HTTP_201_CREATED)
async def register(request: RegisterRequest):
    """
    Verifies the OTP and registers a new user with a hashed password.
    """
    logger.info("Registration attempt", email=request.email)
    result = await auth_service.verify_and_register_user(
        email=request.email, otp=request.otp, password=request.password
    )
    return result


@router.post("/login", response_model=TokenResponse)
async def login(form_data: OAuth2PasswordRequestForm = Depends()):
    """
    Authenticates a user and returns a JSON Web Token (JWT).
    Note: OAuth2 requires the field to be named 'username', which mapped to email.
    """
    logger.info("Login attempt", email=form_data.username)

    access_token = await auth_service.authenticate_user(
        email=form_data.username, password=form_data.password
    )

    return {"access_token": access_token, "token_type": "bearer"}


@router.post("/request-reset-otp", status_code=status.HTTP_200_OK)
async def request_reset_otp(request: OTPRequest):
    """
    Initiates the password reset flow by sending an OTP.
    Fails silently if the email is not registered.
    """
    email_str = str(request.email)
    logger.info("Password reset OTP request received", email=email_str)
    await auth_service.request_password_reset_otp(email_str)
    return {
        "message": "If the email is registered, a password reset OTP has been sent."
    }


@router.post("/reset-password", status_code=status.HTTP_200_OK)
async def reset_password(request: PasswordResetRequest):
    """
    Verifies the OTP and updates the user's password.
    """
    email_str = str(request.email)
    logger.info("Password reset attempt", email=email_str)
    return await auth_service.reset_password(
        email=email_str, otp=request.otp, new_password=request.new_password
    )

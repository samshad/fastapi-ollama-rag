from datetime import UTC, datetime, timedelta

import jwt
import structlog
from passlib.context import CryptContext

from fastapi_ollama_rag.core.config import settings

logger = structlog.get_logger(__name__)

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verifies a plain text password against the hashed version."""
    password_verified = pwd_context.verify(plain_password, hashed_password)
    logger.info("Password verification: ", password_verified)
    return password_verified


def get_password_hash(password: str) -> str:
    """Hashes a password using bcrypt."""
    logger.info("Hashing password using bcrypt...")
    return pwd_context.hash(password)


def create_access_token(data: dict, expires_delta: timedelta | None = None) -> str:
    """Generates a secure JSON Web Token (JWT)."""
    to_encode = data.copy()

    if expires_delta:
        expire = datetime.now(UTC) + expires_delta
    else:
        expire = datetime.now(UTC) + timedelta(
            minutes=settings.access_token_expire_minutes
        )

    to_encode.update({"exp": expire})

    logger.info("Generating JWT token...")

    encoded_jwt = jwt.encode(
        to_encode, settings.secret_key, algorithm=settings.jwt_algorithm
    )
    return encoded_jwt

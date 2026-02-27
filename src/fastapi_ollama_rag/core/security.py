from datetime import UTC, datetime, timedelta

import bcrypt
import jwt
import structlog
from passlib.context import CryptContext

from fastapi_ollama_rag.core.config import settings

logger = structlog.get_logger(__name__)

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verifies a plain text password against the hashed version."""
    logger.info("Password verifying using bcrypt...")
    return bcrypt.checkpw(
        password=plain_password.encode("utf-8"),
        hashed_password=hashed_password.encode("utf-8"),
    )


def get_password_hash(password: str) -> str:
    """Hashes a password using bcrypt."""
    logger.info("Hashing password using bcrypt...")
    salt = bcrypt.gensalt()
    hashed_bytes = bcrypt.hashpw(password.encode("utf-8"), salt)

    return hashed_bytes.decode("utf-8")


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

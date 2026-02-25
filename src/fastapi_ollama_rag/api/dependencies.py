import jwt
import structlog
from asyncpg import Record
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer

from fastapi_ollama_rag.core.config import settings
from fastapi_ollama_rag.repository import user_repo

logger = structlog.get_logger(__name__)

oauth2_scheme = OAuth2PasswordBearer(tokenUrl=settings.api_v1_prefix + "/auth/login")


async def get_current_user(token: str = Depends(oauth2_scheme)) -> Record:
    """
    Validates the JWT token from the Authorization header
    and retrieves the active user.
    """
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )

    try:
        payload = jwt.decode(
            token, settings.secret_key, algorithms=[settings.jwt_algorithm]
        )

        user_id: str = payload.get("sub")
        if user_id is None:
            logger.warning("Rejected token without user ID")
            raise credentials_exception

    except jwt.ExpiredSignatureError:
        logger.warning("Rejected expired token")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token has expired. Please log in again.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    except jwt.InvalidTokenError:
        logger.warning("Rejected invalid token format")
        raise credentials_exception

    user = await user_repo.get_user_by_id(user_id=user_id)
    if user is None:
        logger.warning("Token valid, but user no longer exists in DB", user_id=user_id)
        raise credentials_exception

    logger.info("Token validated", user_id=user_id)
    return user

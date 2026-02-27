import pytest
import jwt
from datetime import datetime, timedelta, UTC
from unittest.mock import AsyncMock, patch
from fastapi import HTTPException

from fastapi_ollama_rag.api.dependencies import get_current_user, oauth2_scheme
from fastapi_ollama_rag.core.config import settings


# -------------------------------------------------------------------
# Helper to generate real JWTs for testing
# -------------------------------------------------------------------
def create_test_token(
    data: dict,
    expires_delta: timedelta = timedelta(minutes=15),
    secret: str = settings.secret_key,
    algorithm: str = settings.jwt_algorithm,
) -> str:
    """Generates a JWT token signed with the given secret and algorithm."""
    to_encode = data.copy()
    expire = datetime.now(UTC) + expires_delta
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, secret, algorithm=algorithm)


# ===================================================================
# Module-level: oauth2_scheme
# ===================================================================


def test_oauth2_scheme_token_url():
    """L12: tokenUrl should be built from settings.api_v1_prefix + '/auth/login'."""
    expected = settings.api_v1_prefix + "/auth/login"
    assert oauth2_scheme.model.flows.password.tokenUrl == expected


# ===================================================================
# Happy path (L26-53)
# ===================================================================


@pytest.mark.asyncio
@patch("fastapi_ollama_rag.api.dependencies.user_repo.get_user_by_id", new_callable=AsyncMock)
async def test_get_current_user_valid(mock_get_user):
    """Test that a valid token decodes correctly and returns the active user."""
    mock_user = {"id": "user_123", "email": "test@example.com"}
    mock_get_user.return_value = mock_user

    token = create_test_token({"sub": "user_123"})

    user = await get_current_user(token=token)

    assert user == mock_user
    mock_get_user.assert_awaited_once_with(user_id="user_123")


@pytest.mark.asyncio
@patch("fastapi_ollama_rag.api.dependencies.user_repo.get_user_by_id", new_callable=AsyncMock)
async def test_get_current_user_returns_user_object(mock_get_user):
    """L53: The return value should be the exact object from get_user_by_id."""
    mock_user = {"id": "u1", "email": "a@b.com", "is_verified": True}
    mock_get_user.return_value = mock_user

    token = create_test_token({"sub": "u1"})
    result = await get_current_user(token=token)

    assert result is mock_user


@pytest.mark.asyncio
@patch("fastapi_ollama_rag.api.dependencies.user_repo.get_user_by_id", new_callable=AsyncMock)
async def test_get_current_user_extra_claims_in_token(mock_get_user):
    """
    Edge Case: Token with extra claims beyond 'sub' and 'exp'.
    jwt.decode should still work — extra claims are ignored.
    """
    mock_user = {"id": "user_123"}
    mock_get_user.return_value = mock_user

    token = create_test_token({
        "sub": "user_123",
        "email": "extra@example.com",
        "role": "admin",
        "custom_field": 42,
    })

    user = await get_current_user(token=token)
    assert user == mock_user


# ===================================================================
# Expired token (L36-42)
# ===================================================================


@pytest.mark.asyncio
async def test_get_current_user_expired():
    """Test that an expired token is rejected with a helpful message."""
    token = create_test_token({"sub": "user_123"}, expires_delta=timedelta(minutes=-10))

    with pytest.raises(HTTPException) as exc_info:
        await get_current_user(token=token)

    assert exc_info.value.status_code == 401
    assert "Token has expired" in exc_info.value.detail


@pytest.mark.asyncio
async def test_get_current_user_expired_has_www_authenticate_header():
    """
    L41: Expired token HTTPException must have WWW-Authenticate: Bearer header.
    """
    token = create_test_token({"sub": "user_123"}, expires_delta=timedelta(minutes=-10))

    with pytest.raises(HTTPException) as exc_info:
        await get_current_user(token=token)

    assert exc_info.value.headers == {"WWW-Authenticate": "Bearer"}


@pytest.mark.asyncio
async def test_get_current_user_expired_exact_detail():
    """L40: Expired token detail should be the exact fixed string."""
    token = create_test_token({"sub": "user_123"}, expires_delta=timedelta(minutes=-10))

    with pytest.raises(HTTPException) as exc_info:
        await get_current_user(token=token)

    assert exc_info.value.detail == "Token has expired. Please log in again."


# ===================================================================
# Invalid token format (L43-45)
# ===================================================================


@pytest.mark.asyncio
async def test_get_current_user_invalid_format():
    """Complete garbage strings → InvalidTokenError → credentials_exception."""
    with pytest.raises(HTTPException) as exc_info:
        await get_current_user(token="this.is.not.a.real.jwt.token")

    assert exc_info.value.status_code == 401
    assert "Could not validate credentials" in exc_info.value.detail


@pytest.mark.asyncio
async def test_get_current_user_invalid_format_has_www_authenticate_header():
    """L23/L45: Invalid token HTTPException must have WWW-Authenticate: Bearer header."""
    with pytest.raises(HTTPException) as exc_info:
        await get_current_user(token="garbage_token")

    assert exc_info.value.headers == {"WWW-Authenticate": "Bearer"}


@pytest.mark.asyncio
async def test_get_current_user_empty_string_token():
    """Edge Case: Empty string token → jwt.InvalidTokenError → 401."""
    with pytest.raises(HTTPException) as exc_info:
        await get_current_user(token="")

    assert exc_info.value.status_code == 401
    assert "Could not validate credentials" in exc_info.value.detail


@pytest.mark.asyncio
async def test_get_current_user_wrong_secret_key():
    """
    Edge Case: Token signed with a different secret key.
    jwt.decode with the app's secret will fail → InvalidTokenError → 401.
    """
    token = create_test_token({"sub": "user_123"}, secret="wrong-secret-key-that-is-long-enough!!")

    with pytest.raises(HTTPException) as exc_info:
        await get_current_user(token=token)

    assert exc_info.value.status_code == 401
    assert "Could not validate credentials" in exc_info.value.detail


@pytest.mark.asyncio
async def test_get_current_user_wrong_algorithm():
    """
    Edge Case: Token signed with HS384 but app expects HS256.
    jwt.decode with algorithms=[HS256] will reject → InvalidTokenError → 401.
    """
    token = create_test_token({"sub": "user_123"}, algorithm="HS384")

    with pytest.raises(HTTPException) as exc_info:
        await get_current_user(token=token)

    assert exc_info.value.status_code == 401
    assert "Could not validate credentials" in exc_info.value.detail


@pytest.mark.asyncio
async def test_get_current_user_none_algorithm_token():
    """
    Edge Case: Token crafted with algorithm="none" (JWT algorithm confusion attack).
    The app's jwt.decode specifies algorithms=[HS256], so this must be rejected.
    """
    import base64

    def b64url(data: bytes) -> str:
        return base64.urlsafe_b64encode(data).rstrip(b"=").decode()

    header = b64url(b'{"alg":"none","typ":"JWT"}')
    payload_data = b64url(b'{"sub":"attacker","exp":9999999999}')
    forged_token = f"{header}.{payload_data}."

    with pytest.raises(HTTPException) as exc_info:
        await get_current_user(token=forged_token)

    assert exc_info.value.status_code == 401


# ===================================================================
# Missing 'sub' claim (L31-34)
# ===================================================================


@pytest.mark.asyncio
async def test_get_current_user_missing_sub():
    """Token with no 'sub' claim at all → credentials_exception."""
    token = create_test_token({"email": "test@example.com"})

    with pytest.raises(HTTPException) as exc_info:
        await get_current_user(token=token)

    assert exc_info.value.status_code == 401
    assert "Could not validate credentials" in exc_info.value.detail


@pytest.mark.asyncio
async def test_get_current_user_missing_sub_has_www_authenticate_header():
    """L23/L34: Missing-sub HTTPException must have WWW-Authenticate: Bearer header."""
    token = create_test_token({"email": "test@example.com"})

    with pytest.raises(HTTPException) as exc_info:
        await get_current_user(token=token)

    assert exc_info.value.headers == {"WWW-Authenticate": "Bearer"}


@pytest.mark.asyncio
async def test_get_current_user_sub_is_explicit_none():
    """
    Edge Case: Token payload is {"sub": None}.
    payload.get("sub") returns None → L32 triggers credentials_exception.
    """
    token = create_test_token({"sub": None})

    with pytest.raises(HTTPException) as exc_info:
        await get_current_user(token=token)

    assert exc_info.value.status_code == 401
    assert "Could not validate credentials" in exc_info.value.detail


@pytest.mark.asyncio
@patch("fastapi_ollama_rag.api.dependencies.user_repo.get_user_by_id", new_callable=AsyncMock)
async def test_get_current_user_sub_is_empty_string(mock_get_user):
    """
    Edge Case: Token payload is {"sub": ""}.
    "" is falsy in Python, BUT `payload.get("sub")` returns "" which is not None.
    So L32 check passes (user_id is not None), and get_user_by_id is called with "".
    """
    mock_get_user.return_value = None  # DB won't find user with empty id

    token = create_test_token({"sub": ""})

    with pytest.raises(HTTPException) as exc_info:
        await get_current_user(token=token)

    # The function proceeds past L32, calls get_user_by_id(""), gets None, raises at L50
    mock_get_user.assert_awaited_once_with(user_id="")
    assert exc_info.value.status_code == 401


@pytest.mark.asyncio
async def test_get_current_user_sub_is_integer():
    """
    Edge Case: Token payload is {"sub": 12345} (integer, not string).
    PyJWT validates that 'sub' must be a string (RFC 7519 §4.1.2).
    It raises InvalidSubjectError (subclass of InvalidTokenError) → L43-45 → 401.
    get_user_by_id is NEVER reached.
    """
    token = create_test_token({"sub": 12345})

    with pytest.raises(HTTPException) as exc_info:
        await get_current_user(token=token)

    assert exc_info.value.status_code == 401
    assert "Could not validate credentials" in exc_info.value.detail


# ===================================================================
# User not found in DB (L47-50)
# ===================================================================


@pytest.mark.asyncio
@patch("fastapi_ollama_rag.api.dependencies.user_repo.get_user_by_id", new_callable=AsyncMock)
async def test_get_current_user_not_found_in_db(mock_get_user):
    """Token is valid, but the user was deleted from the DB."""
    mock_get_user.return_value = None

    token = create_test_token({"sub": "deleted_user_999"})

    with pytest.raises(HTTPException) as exc_info:
        await get_current_user(token=token)

    assert exc_info.value.status_code == 401
    assert "Could not validate credentials" in exc_info.value.detail


@pytest.mark.asyncio
@patch("fastapi_ollama_rag.api.dependencies.user_repo.get_user_by_id", new_callable=AsyncMock)
async def test_get_current_user_not_found_has_www_authenticate_header(mock_get_user):
    """L23/L50: User-not-found HTTPException must have WWW-Authenticate: Bearer header."""
    mock_get_user.return_value = None

    token = create_test_token({"sub": "deleted_user"})

    with pytest.raises(HTTPException) as exc_info:
        await get_current_user(token=token)

    assert exc_info.value.headers == {"WWW-Authenticate": "Bearer"}


# ===================================================================
# DB error propagation (L47)
# ===================================================================


@pytest.mark.asyncio
@patch(
    "fastapi_ollama_rag.api.dependencies.user_repo.get_user_by_id",
    new_callable=AsyncMock,
    side_effect=RuntimeError("DB connection lost"),
)
async def test_get_current_user_db_error_propagates(mock_get_user):
    """
    Edge Case: get_user_by_id raises an exception (e.g., DB down).
    There is no try/except around L47, so the error propagates uncaught.
    """
    token = create_test_token({"sub": "user_123"})

    with pytest.raises(RuntimeError, match="DB connection lost"):
        await get_current_user(token=token)


# ===================================================================
# Security: all 401 responses are indistinguishable
# ===================================================================


@pytest.mark.asyncio
@patch("fastapi_ollama_rag.api.dependencies.user_repo.get_user_by_id", new_callable=AsyncMock)
async def test_invalid_token_and_missing_user_same_error(mock_get_user):
    """
    Security: Invalid-token and user-not-found errors must produce the
    identical status code + detail to prevent user enumeration.
    """
    # 1. Invalid token error
    with pytest.raises(HTTPException) as exc_invalid:
        await get_current_user(token="garbage")

    # 2. User-not-found error
    mock_get_user.return_value = None
    token = create_test_token({"sub": "deleted_user"})
    with pytest.raises(HTTPException) as exc_not_found:
        await get_current_user(token=token)

    # 3. Missing-sub error
    token_no_sub = create_test_token({"email": "a@b.com"})
    with pytest.raises(HTTPException) as exc_no_sub:
        await get_current_user(token=token_no_sub)

    # All three must have identical detail and status
    assert exc_invalid.value.status_code == exc_not_found.value.status_code == exc_no_sub.value.status_code == 401
    assert exc_invalid.value.detail == exc_not_found.value.detail == exc_no_sub.value.detail
    assert exc_invalid.value.headers == exc_not_found.value.headers == exc_no_sub.value.headers



import pytest
from httpx import AsyncClient, ASGITransport
from unittest.mock import AsyncMock, patch
from fastapi import FastAPI, HTTPException, status

# Import the router directly
from fastapi_ollama_rag.api.routes.auth import router

# -------------------------------------------------------------------
# Setup a "Mini" FastAPI App for Router Testing
# -------------------------------------------------------------------
mini_app = FastAPI()
mini_app.include_router(router)


@pytest.fixture
async def client():
    """Provides an async HTTP client connected to our mini FastAPI app."""
    async with AsyncClient(
        transport=ASGITransport(app=mini_app), base_url="http://testserver"
    ) as ac:
        yield ac


# ===================================================================
# POST /auth/request-otp
# ===================================================================


@pytest.mark.asyncio
@patch(
    "fastapi_ollama_rag.api.routes.auth.auth_service.request_registration_otp",
    new_callable=AsyncMock,
)
async def test_request_otp_route(mock_service, client: AsyncClient):
    """Test the registration OTP request endpoint."""
    payload = {"email": "test@example.com"}

    response = await client.post("/auth/request-otp", json=payload)

    assert response.status_code == 200
    assert "sent" in response.json()["message"]
    mock_service.assert_awaited_once_with("test@example.com")


@pytest.mark.asyncio
@patch(
    "fastapi_ollama_rag.api.routes.auth.auth_service.request_registration_otp",
    new_callable=AsyncMock,
)
async def test_request_otp_exact_response_message(mock_service, client: AsyncClient):
    """Response message should be the exact fixed string from L27."""
    response = await client.post(
        "/auth/request-otp", json={"email": "user@example.com"}
    )

    assert response.status_code == 200
    assert (
        response.json()["message"]
        == "If the email is valid and unregistered, an OTP has been sent."
    )


@pytest.mark.asyncio
async def test_request_otp_invalid_email(client: AsyncClient):
    """Invalid email format → 422 from Pydantic's EmailStr."""
    response = await client.post(
        "/auth/request-otp", json={"email": "not-an-email"}
    )
    assert response.status_code == 422


@pytest.mark.asyncio
async def test_request_otp_missing_email(client: AsyncClient):
    """Missing email field → 422."""
    response = await client.post("/auth/request-otp", json={})
    assert response.status_code == 422


@pytest.mark.asyncio
async def test_request_otp_empty_body(client: AsyncClient):
    """Completely empty request body → 422."""
    response = await client.post(
        "/auth/request-otp",
        content=b"",
        headers={"Content-Type": "application/json"},
    )
    assert response.status_code == 422


@pytest.mark.asyncio
@patch(
    "fastapi_ollama_rag.api.routes.auth.auth_service.request_registration_otp",
    new_callable=AsyncMock,
    side_effect=RuntimeError("SMTP failed"),
)
async def test_request_otp_service_exception(mock_service, client: AsyncClient):
    """
    Edge Case: Service raises an unhandled exception.
    The route has no try/except, so the RuntimeError propagates through FastAPI.
    """
    with pytest.raises(RuntimeError, match="SMTP failed"):
        await client.post(
            "/auth/request-otp", json={"email": "user@example.com"}
        )


@pytest.mark.asyncio
@patch(
    "fastapi_ollama_rag.api.routes.auth.auth_service.request_registration_otp",
    new_callable=AsyncMock,
)
async def test_request_otp_response_is_json(mock_service, client: AsyncClient):
    """Response content-type should be application/json."""
    response = await client.post(
        "/auth/request-otp", json={"email": "user@example.com"}
    )
    assert "application/json" in response.headers["content-type"]


# ===================================================================
# POST /auth/register
# ===================================================================


@pytest.mark.asyncio
@patch(
    "fastapi_ollama_rag.api.routes.auth.auth_service.verify_and_register_user",
    new_callable=AsyncMock,
)
async def test_register_route(mock_service, client: AsyncClient):
    """Test the user registration endpoint."""
    mock_service.return_value = {"message": "User registered successfully."}

    payload = {
        "email": "test@example.com",
        "otp": "123456",
        "password": "SecurePassword123!",
    }

    response = await client.post("/auth/register", json=payload)

    assert response.status_code == 201
    assert response.json()["message"] == "User registered successfully."
    mock_service.assert_awaited_once_with(
        email="test@example.com", otp="123456", password="SecurePassword123!"
    )


@pytest.mark.asyncio
async def test_register_missing_password(client: AsyncClient):
    """Missing the 'password' field → 422."""
    payload = {"email": "test@example.com", "otp": "123456"}
    response = await client.post("/auth/register", json=payload)

    assert response.status_code == 422
    assert response.json()["detail"][0]["loc"] == ["body", "password"]


@pytest.mark.asyncio
async def test_register_missing_email(client: AsyncClient):
    """Missing 'email' → 422."""
    payload = {"otp": "123456", "password": "SecurePassword123!"}
    response = await client.post("/auth/register", json=payload)

    assert response.status_code == 422
    locs = [e["loc"] for e in response.json()["detail"]]
    assert ["body", "email"] in locs


@pytest.mark.asyncio
async def test_register_missing_otp(client: AsyncClient):
    """Missing 'otp' → 422."""
    payload = {"email": "test@example.com", "password": "SecurePassword123!"}
    response = await client.post("/auth/register", json=payload)

    assert response.status_code == 422
    locs = [e["loc"] for e in response.json()["detail"]]
    assert ["body", "otp"] in locs


@pytest.mark.asyncio
async def test_register_empty_body(client: AsyncClient):
    """Completely empty body → 422."""
    response = await client.post("/auth/register", json={})
    assert response.status_code == 422


@pytest.mark.asyncio
async def test_register_invalid_email(client: AsyncClient):
    """Invalid email format → 422."""
    payload = {
        "email": "bad-email",
        "otp": "123456",
        "password": "SecurePassword123!",
    }
    response = await client.post("/auth/register", json=payload)
    assert response.status_code == 422


@pytest.mark.asyncio
async def test_register_otp_too_short(client: AsyncClient):
    """OTP < 6 chars → 422 (min_length=6)."""
    payload = {
        "email": "test@example.com",
        "otp": "12345",
        "password": "SecurePassword123!",
    }
    response = await client.post("/auth/register", json=payload)
    assert response.status_code == 422


@pytest.mark.asyncio
async def test_register_otp_too_long(client: AsyncClient):
    """OTP > 6 chars → 422 (max_length=6)."""
    payload = {
        "email": "test@example.com",
        "otp": "1234567",
        "password": "SecurePassword123!",
    }
    response = await client.post("/auth/register", json=payload)
    assert response.status_code == 422


@pytest.mark.asyncio
async def test_register_password_too_short(client: AsyncClient):
    """Password < 8 chars → 422 (min_length=8)."""
    payload = {
        "email": "test@example.com",
        "otp": "123456",
        "password": "Short1!",
    }
    response = await client.post("/auth/register", json=payload)
    assert response.status_code == 422


@pytest.mark.asyncio
async def test_register_password_too_long(client: AsyncClient):
    """Password > 256 chars → 422 (max_length=256)."""
    payload = {
        "email": "test@example.com",
        "otp": "123456",
        "password": "A" * 257,
    }
    response = await client.post("/auth/register", json=payload)
    assert response.status_code == 422


@pytest.mark.asyncio
@patch(
    "fastapi_ollama_rag.api.routes.auth.auth_service.verify_and_register_user",
    new_callable=AsyncMock,
)
async def test_register_password_exactly_8_chars(mock_service, client: AsyncClient):
    """Boundary: Password with exactly 8 chars → 201 (accepted)."""
    mock_service.return_value = {"message": "OK"}

    payload = {
        "email": "test@example.com",
        "otp": "123456",
        "password": "Exactly8",
    }
    response = await client.post("/auth/register", json=payload)
    assert response.status_code == 201


@pytest.mark.asyncio
@patch(
    "fastapi_ollama_rag.api.routes.auth.auth_service.verify_and_register_user",
    new_callable=AsyncMock,
)
async def test_register_password_exactly_256_chars(mock_service, client: AsyncClient):
    """Boundary: Password with exactly 256 chars → 201 (accepted)."""
    mock_service.return_value = {"message": "OK"}

    payload = {
        "email": "test@example.com",
        "otp": "123456",
        "password": "A" * 256,
    }
    response = await client.post("/auth/register", json=payload)
    assert response.status_code == 201


@pytest.mark.asyncio
@patch(
    "fastapi_ollama_rag.api.routes.auth.auth_service.verify_and_register_user",
    new_callable=AsyncMock,
    side_effect=HTTPException(
        status_code=status.HTTP_400_BAD_REQUEST,
        detail="Invalid or expired verification code.",
    ),
)
async def test_register_service_raises_400(mock_service, client: AsyncClient):
    """
    Edge Case: Service raises HTTPException(400) for invalid OTP.
    FastAPI should propagate the 400 status code.
    """
    payload = {
        "email": "test@example.com",
        "otp": "000000",
        "password": "SecurePassword123!",
    }
    response = await client.post("/auth/register", json=payload)

    assert response.status_code == 400
    assert "Invalid or expired" in response.json()["detail"]


@pytest.mark.asyncio
@patch(
    "fastapi_ollama_rag.api.routes.auth.auth_service.verify_and_register_user",
    new_callable=AsyncMock,
    side_effect=HTTPException(
        status_code=status.HTTP_400_BAD_REQUEST,
        detail="User already registered.",
    ),
)
async def test_register_service_raises_already_registered(
    mock_service, client: AsyncClient
):
    """Edge Case: Service raises 400 for already-registered user."""
    payload = {
        "email": "existing@example.com",
        "otp": "123456",
        "password": "SecurePassword123!",
    }
    response = await client.post("/auth/register", json=payload)

    assert response.status_code == 400
    assert "User already registered" in response.json()["detail"]


# ===================================================================
# POST /auth/login
# ===================================================================


@pytest.mark.asyncio
@patch(
    "fastapi_ollama_rag.api.routes.auth.auth_service.authenticate_user",
    new_callable=AsyncMock,
)
async def test_login_route(mock_service, client: AsyncClient):
    """Test the OAuth2 login endpoint (must use Form Data, not JSON)."""
    mock_service.return_value = "fake_jwt_token_string"

    form_data = {"username": "test@example.com", "password": "SecurePassword123!"}
    response = await client.post("/auth/login", data=form_data)

    assert response.status_code == 200
    assert response.json() == {
        "access_token": "fake_jwt_token_string",
        "token_type": "bearer",
    }
    mock_service.assert_awaited_once_with(
        email="test@example.com", password="SecurePassword123!"
    )


@pytest.mark.asyncio
@patch(
    "fastapi_ollama_rag.api.routes.auth.auth_service.authenticate_user",
    new_callable=AsyncMock,
)
async def test_login_json_body_instead_of_form(mock_service, client: AsyncClient):
    """
    Edge Case: OAuth2PasswordRequestForm expects form data, NOT JSON.
    Sending JSON body should result in 422.
    """
    response = await client.post(
        "/auth/login",
        json={"username": "test@example.com", "password": "SecurePassword123!"},
    )
    assert response.status_code == 422


@pytest.mark.asyncio
async def test_login_missing_username(client: AsyncClient):
    """Missing 'username' in form data → 422."""
    response = await client.post(
        "/auth/login", data={"password": "SecurePassword123!"}
    )
    assert response.status_code == 422


@pytest.mark.asyncio
async def test_login_missing_password(client: AsyncClient):
    """Missing 'password' in form data → 422."""
    response = await client.post(
        "/auth/login", data={"username": "test@example.com"}
    )
    assert response.status_code == 422


@pytest.mark.asyncio
async def test_login_empty_form_data(client: AsyncClient):
    """Completely empty form data → 422."""
    response = await client.post("/auth/login", data={})
    assert response.status_code == 422


@pytest.mark.asyncio
@patch(
    "fastapi_ollama_rag.api.routes.auth.auth_service.authenticate_user",
    new_callable=AsyncMock,
    side_effect=HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Incorrect email or password",
        headers={"WWW-Authenticate": "Bearer"},
    ),
)
async def test_login_service_raises_401(mock_service, client: AsyncClient):
    """
    Edge Case: Service raises HTTPException(401) for bad credentials.
    FastAPI should propagate the 401 status code.
    """
    form_data = {"username": "test@example.com", "password": "WrongPassword!"}
    response = await client.post("/auth/login", data=form_data)

    assert response.status_code == 401
    assert "Incorrect email or password" in response.json()["detail"]


@pytest.mark.asyncio
@patch(
    "fastapi_ollama_rag.api.routes.auth.auth_service.authenticate_user",
    new_callable=AsyncMock,
)
async def test_login_response_matches_token_model(mock_service, client: AsyncClient):
    """Response should match the TokenResponse model: access_token + token_type."""
    mock_service.return_value = "jwt_xyz"

    response = await client.post(
        "/auth/login",
        data={"username": "test@example.com", "password": "SecurePassword123!"},
    )

    body = response.json()
    assert set(body.keys()) == {"access_token", "token_type"}
    assert body["access_token"] == "jwt_xyz"
    assert body["token_type"] == "bearer"


# ===================================================================
# POST /auth/request-reset-otp
# ===================================================================


@pytest.mark.asyncio
@patch(
    "fastapi_ollama_rag.api.routes.auth.auth_service.request_password_reset_otp",
    new_callable=AsyncMock,
)
async def test_request_reset_otp_route(mock_service, client: AsyncClient):
    """Test the password reset OTP request endpoint."""
    payload = {"email": "reset@example.com"}
    response = await client.post("/auth/request-reset-otp", json=payload)

    assert response.status_code == 200
    assert "password reset OTP has been sent" in response.json()["message"]
    mock_service.assert_awaited_once_with("reset@example.com")


@pytest.mark.asyncio
@patch(
    "fastapi_ollama_rag.api.routes.auth.auth_service.request_password_reset_otp",
    new_callable=AsyncMock,
)
async def test_request_reset_otp_exact_response_message(
    mock_service, client: AsyncClient
):
    """Response message should be the exact fixed string from L66-67."""
    response = await client.post(
        "/auth/request-reset-otp", json={"email": "user@example.com"}
    )

    assert response.status_code == 200
    assert (
        response.json()["message"]
        == "If the email is registered, a password reset OTP has been sent."
    )


@pytest.mark.asyncio
async def test_request_reset_otp_invalid_email(client: AsyncClient):
    """Invalid email → 422."""
    response = await client.post(
        "/auth/request-reset-otp", json={"email": "not-email"}
    )
    assert response.status_code == 422


@pytest.mark.asyncio
async def test_request_reset_otp_missing_email(client: AsyncClient):
    """Missing email → 422."""
    response = await client.post("/auth/request-reset-otp", json={})
    assert response.status_code == 422


@pytest.mark.asyncio
@patch(
    "fastapi_ollama_rag.api.routes.auth.auth_service.request_password_reset_otp",
    new_callable=AsyncMock,
    side_effect=RuntimeError("SMTP failed"),
)
async def test_request_reset_otp_service_exception(mock_service, client: AsyncClient):
    """Service raises unhandled exception → propagates through FastAPI."""
    with pytest.raises(RuntimeError, match="SMTP failed"):
        await client.post(
            "/auth/request-reset-otp", json={"email": "user@example.com"}
        )


# ===================================================================
# POST /auth/reset-password
# ===================================================================


@pytest.mark.asyncio
@patch(
    "fastapi_ollama_rag.api.routes.auth.auth_service.reset_password",
    new_callable=AsyncMock,
)
async def test_reset_password_route(mock_service, client: AsyncClient):
    """Test the password reset consumption endpoint."""
    mock_service.return_value = {"message": "Password reset."}

    payload = {
        "email": "reset@example.com",
        "otp": "654321",
        "new_password": "BrandNewPassword999!",
    }
    response = await client.post("/auth/reset-password", json=payload)

    assert response.status_code == 200
    mock_service.assert_awaited_once_with(
        email="reset@example.com",
        otp="654321",
        new_password="BrandNewPassword999!",
    )


@pytest.mark.asyncio
async def test_reset_password_missing_email(client: AsyncClient):
    """Missing email → 422."""
    payload = {"otp": "654321", "new_password": "BrandNewPassword999!"}
    response = await client.post("/auth/reset-password", json=payload)

    assert response.status_code == 422


@pytest.mark.asyncio
async def test_reset_password_missing_otp(client: AsyncClient):
    """Missing OTP → 422."""
    payload = {"email": "reset@example.com", "new_password": "BrandNewPassword999!"}
    response = await client.post("/auth/reset-password", json=payload)

    assert response.status_code == 422


@pytest.mark.asyncio
async def test_reset_password_missing_new_password(client: AsyncClient):
    """Missing new_password → 422."""
    payload = {"email": "reset@example.com", "otp": "654321"}
    response = await client.post("/auth/reset-password", json=payload)

    assert response.status_code == 422


@pytest.mark.asyncio
async def test_reset_password_empty_body(client: AsyncClient):
    """Completely empty body → 422."""
    response = await client.post("/auth/reset-password", json={})
    assert response.status_code == 422


@pytest.mark.asyncio
async def test_reset_password_invalid_email(client: AsyncClient):
    """Invalid email format → 422."""
    payload = {
        "email": "not-an-email",
        "otp": "654321",
        "new_password": "BrandNewPassword999!",
    }
    response = await client.post("/auth/reset-password", json=payload)
    assert response.status_code == 422


@pytest.mark.asyncio
async def test_reset_password_otp_too_short(client: AsyncClient):
    """OTP < 6 chars → 422."""
    payload = {
        "email": "reset@example.com",
        "otp": "12345",
        "new_password": "BrandNewPassword999!",
    }
    response = await client.post("/auth/reset-password", json=payload)
    assert response.status_code == 422


@pytest.mark.asyncio
async def test_reset_password_otp_too_long(client: AsyncClient):
    """OTP > 6 chars → 422."""
    payload = {
        "email": "reset@example.com",
        "otp": "1234567",
        "new_password": "BrandNewPassword999!",
    }
    response = await client.post("/auth/reset-password", json=payload)
    assert response.status_code == 422


@pytest.mark.asyncio
async def test_reset_password_new_password_too_short(client: AsyncClient):
    """new_password < 8 chars → 422."""
    payload = {
        "email": "reset@example.com",
        "otp": "654321",
        "new_password": "Short1!",
    }
    response = await client.post("/auth/reset-password", json=payload)
    assert response.status_code == 422


@pytest.mark.asyncio
async def test_reset_password_new_password_too_long(client: AsyncClient):
    """new_password > 256 chars → 422."""
    payload = {
        "email": "reset@example.com",
        "otp": "654321",
        "new_password": "A" * 257,
    }
    response = await client.post("/auth/reset-password", json=payload)
    assert response.status_code == 422


@pytest.mark.asyncio
@patch(
    "fastapi_ollama_rag.api.routes.auth.auth_service.reset_password",
    new_callable=AsyncMock,
)
async def test_reset_password_new_password_exactly_8_chars(
    mock_service, client: AsyncClient
):
    """Boundary: new_password exactly 8 chars → 200 (accepted)."""
    mock_service.return_value = {"message": "OK"}

    payload = {
        "email": "reset@example.com",
        "otp": "654321",
        "new_password": "Exactly8",
    }
    response = await client.post("/auth/reset-password", json=payload)
    assert response.status_code == 200


@pytest.mark.asyncio
@patch(
    "fastapi_ollama_rag.api.routes.auth.auth_service.reset_password",
    new_callable=AsyncMock,
)
async def test_reset_password_new_password_exactly_256_chars(
    mock_service, client: AsyncClient
):
    """Boundary: new_password exactly 256 chars → 200 (accepted)."""
    mock_service.return_value = {"message": "OK"}

    payload = {
        "email": "reset@example.com",
        "otp": "654321",
        "new_password": "A" * 256,
    }
    response = await client.post("/auth/reset-password", json=payload)
    assert response.status_code == 200


@pytest.mark.asyncio
@patch(
    "fastapi_ollama_rag.api.routes.auth.auth_service.reset_password",
    new_callable=AsyncMock,
    side_effect=HTTPException(
        status_code=status.HTTP_400_BAD_REQUEST,
        detail="Invalid or expired verification code.",
    ),
)
async def test_reset_password_service_raises_400(mock_service, client: AsyncClient):
    """
    Edge Case: Service raises HTTPException(400) for invalid OTP.
    FastAPI should propagate the 400 status code.
    """
    payload = {
        "email": "reset@example.com",
        "otp": "000000",
        "new_password": "BrandNewPassword999!",
    }
    response = await client.post("/auth/reset-password", json=payload)

    assert response.status_code == 400
    assert "Invalid or expired" in response.json()["detail"]


# ===================================================================
# Cross-cutting: wrong HTTP methods
# ===================================================================


@pytest.mark.asyncio
async def test_request_otp_get_not_allowed(client: AsyncClient):
    """GET on a POST-only endpoint → 405 Method Not Allowed."""
    response = await client.get("/auth/request-otp")
    assert response.status_code == 405


@pytest.mark.asyncio
async def test_register_get_not_allowed(client: AsyncClient):
    """GET on a POST-only endpoint → 405 Method Not Allowed."""
    response = await client.get("/auth/register")
    assert response.status_code == 405


@pytest.mark.asyncio
async def test_login_get_not_allowed(client: AsyncClient):
    """GET on a POST-only endpoint → 405 Method Not Allowed."""
    response = await client.get("/auth/login")
    assert response.status_code == 405


@pytest.mark.asyncio
async def test_request_reset_otp_get_not_allowed(client: AsyncClient):
    """GET on a POST-only endpoint → 405 Method Not Allowed."""
    response = await client.get("/auth/request-reset-otp")
    assert response.status_code == 405


@pytest.mark.asyncio
async def test_reset_password_get_not_allowed(client: AsyncClient):
    """GET on a POST-only endpoint → 405 Method Not Allowed."""
    response = await client.get("/auth/reset-password")
    assert response.status_code == 405


# ===================================================================
# Cross-cutting: nonexistent routes
# ===================================================================


@pytest.mark.asyncio
async def test_nonexistent_route_returns_404(client: AsyncClient):
    """A route that doesn't exist → 404."""
    response = await client.post("/auth/does-not-exist", json={})
    assert response.status_code == 404


# ===================================================================
# Cross-cutting: additional edge cases
# ===================================================================


@pytest.mark.asyncio
async def test_request_otp_null_email(client: AsyncClient):
    """Edge Case: email field present but null → 422."""
    response = await client.post("/auth/request-otp", json={"email": None})
    assert response.status_code == 422


@pytest.mark.asyncio
async def test_register_null_fields(client: AsyncClient):
    """Edge Case: All fields present but null → 422."""
    response = await client.post(
        "/auth/register",
        json={"email": None, "otp": None, "password": None},
    )
    assert response.status_code == 422


@pytest.mark.asyncio
async def test_reset_password_null_fields(client: AsyncClient):
    """Edge Case: All fields present but null → 422."""
    response = await client.post(
        "/auth/reset-password",
        json={"email": None, "otp": None, "new_password": None},
    )
    assert response.status_code == 422


@pytest.mark.asyncio
async def test_request_reset_otp_empty_body(client: AsyncClient):
    """Completely empty body → 422."""
    response = await client.post(
        "/auth/request-reset-otp",
        content=b"",
        headers={"Content-Type": "application/json"},
    )
    assert response.status_code == 422


@pytest.mark.asyncio
@patch(
    "fastapi_ollama_rag.api.routes.auth.auth_service.verify_and_register_user",
    new_callable=AsyncMock,
    side_effect=RuntimeError("DB crashed"),
)
async def test_register_service_unhandled_exception(mock_service, client: AsyncClient):
    """
    Edge Case: Service raises an unhandled RuntimeError (not HTTPException).
    The route has no try/except, so it propagates through FastAPI.
    """
    payload = {
        "email": "test@example.com",
        "otp": "123456",
        "password": "SecurePassword123!",
    }
    with pytest.raises(RuntimeError, match="DB crashed"):
        await client.post("/auth/register", json=payload)


@pytest.mark.asyncio
@patch(
    "fastapi_ollama_rag.api.routes.auth.auth_service.authenticate_user",
    new_callable=AsyncMock,
    side_effect=RuntimeError("DB crashed"),
)
async def test_login_service_unhandled_exception(mock_service, client: AsyncClient):
    """
    Edge Case: Service raises an unhandled RuntimeError (not HTTPException).
    The route has no try/except, so it propagates through FastAPI.
    """
    with pytest.raises(RuntimeError, match="DB crashed"):
        await client.post(
            "/auth/login",
            data={"username": "test@example.com", "password": "Pw123!"},
        )


@pytest.mark.asyncio
@patch(
    "fastapi_ollama_rag.api.routes.auth.auth_service.reset_password",
    new_callable=AsyncMock,
    side_effect=RuntimeError("DB crashed"),
)
async def test_reset_password_service_unhandled_exception(
    mock_service, client: AsyncClient
):
    """
    Edge Case: Service raises an unhandled RuntimeError (not HTTPException).
    The route has no try/except, so it propagates through FastAPI.
    """
    payload = {
        "email": "test@example.com",
        "otp": "123456",
        "new_password": "NewPassword123!",
    }
    with pytest.raises(RuntimeError, match="DB crashed"):
        await client.post("/auth/reset-password", json=payload)


@pytest.mark.asyncio
@patch(
    "fastapi_ollama_rag.api.routes.auth.auth_service.request_registration_otp",
    new_callable=AsyncMock,
)
async def test_request_otp_email_passed_as_string(mock_service, client: AsyncClient):
    """
    L24: email_str = str(request.email).
    Verify the service receives a plain string, not an EmailStr object.
    """
    await client.post("/auth/request-otp", json={"email": "test@example.com"})

    call_arg = mock_service.call_args[0][0]
    assert isinstance(call_arg, str)
    assert call_arg == "test@example.com"


@pytest.mark.asyncio
@patch(
    "fastapi_ollama_rag.api.routes.auth.auth_service.reset_password",
    new_callable=AsyncMock,
)
async def test_reset_password_email_passed_as_string(
    mock_service, client: AsyncClient
):
    """
    L76: email_str = str(request.email).
    Verify the service receives a plain string, not an EmailStr object.
    """
    mock_service.return_value = {"message": "OK"}

    await client.post(
        "/auth/reset-password",
        json={
            "email": "test@example.com",
            "otp": "123456",
            "new_password": "NewPassword123!",
        },
    )

    _, kwargs = mock_service.call_args
    assert isinstance(kwargs["email"], str)
    assert kwargs["email"] == "test@example.com"

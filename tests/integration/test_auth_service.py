import uuid
import pytest
from unittest.mock import AsyncMock, patch
from fastapi import HTTPException

from fastapi_ollama_rag.services.auth import (
    request_registration_otp,
    verify_and_register_user,
    authenticate_user,
    request_password_reset_otp,
    reset_password,
)
from fastapi_ollama_rag.repository import user_repo
from fastapi_ollama_rag.core.security import get_password_hash, verify_password


# ===================================================================
# request_registration_otp
# ===================================================================


@pytest.mark.asyncio
@patch("fastapi_ollama_rag.services.auth.email_service.send_otp_email", new_callable=AsyncMock)
async def test_request_registration_otp_success(mock_send_email):
    """Test generating an OTP for a brand new email."""
    email = f"new_{uuid.uuid4()}@example.com"

    await request_registration_otp(email)

    # Verify the email service was actually called once
    mock_send_email.assert_awaited_once()
    args, kwargs = mock_send_email.call_args
    assert kwargs["to_email"] == email
    assert "otp" in kwargs


@pytest.mark.asyncio
@patch("fastapi_ollama_rag.services.auth.email_service.send_otp_email", new_callable=AsyncMock)
async def test_request_registration_otp_existing_user(mock_send_email, test_user):
    """Test that requesting an OTP for an already registered user silently aborts."""
    email = test_user["email"]

    await request_registration_otp(email)

    # It should silently return without sending an email
    mock_send_email.assert_not_awaited()


@pytest.mark.asyncio
@patch("fastapi_ollama_rag.services.auth.email_service.send_otp_email", new_callable=AsyncMock)
async def test_request_registration_otp_returns_none(mock_send_email):
    """Return type should be None (no return value)."""
    email = f"new_{uuid.uuid4()}@example.com"
    result = await request_registration_otp(email)
    assert result is None


@pytest.mark.asyncio
@patch("fastapi_ollama_rag.services.auth.email_service.send_otp_email", new_callable=AsyncMock)
async def test_request_registration_otp_existing_user_returns_none(mock_send_email, test_user):
    """Early-return path for existing user also returns None."""
    result = await request_registration_otp(test_user["email"])
    assert result is None


@pytest.mark.asyncio
@patch("fastapi_ollama_rag.services.auth.email_service.send_otp_email", new_callable=AsyncMock)
async def test_request_registration_otp_generates_6_digit_code(mock_send_email):
    """The generated OTP should be exactly 6 digits."""
    email = f"new_{uuid.uuid4()}@example.com"

    await request_registration_otp(email)

    _, kwargs = mock_send_email.call_args
    otp = kwargs["otp"]
    assert len(otp) == 6
    assert otp.isdigit()


@pytest.mark.asyncio
@patch("fastapi_ollama_rag.services.auth.email_service.send_otp_email", new_callable=AsyncMock)
async def test_request_registration_otp_saves_to_db(mock_send_email):
    """The OTP should be saved in the DB and retrievable via get_valid_otp."""
    email = f"new_{uuid.uuid4()}@example.com"

    await request_registration_otp(email)

    _, kwargs = mock_send_email.call_args
    otp = kwargs["otp"]

    # Should be retrievable from the DB
    valid = await user_repo.get_valid_otp(email, otp)
    assert valid is not None

    # Cleanup
    await user_repo.delete_otps_for_email(email)


@pytest.mark.asyncio
@patch("fastapi_ollama_rag.services.auth.email_service.send_otp_email", new_callable=AsyncMock)
async def test_request_registration_otp_twice_saves_two_otps(mock_send_email):
    """
    Edge Case: Calling twice for the same new email saves two different OTPs.
    Both should be valid (no deduplication).
    """
    email = f"new_{uuid.uuid4()}@example.com"

    await request_registration_otp(email)
    first_otp = mock_send_email.call_args[1]["otp"]

    await request_registration_otp(email)
    second_otp = mock_send_email.call_args[1]["otp"]

    assert mock_send_email.await_count == 2

    # Both OTPs should be retrievable
    assert await user_repo.get_valid_otp(email, first_otp) is not None
    assert await user_repo.get_valid_otp(email, second_otp) is not None

    # Cleanup
    await user_repo.delete_otps_for_email(email)


@pytest.mark.asyncio
@patch(
    "fastapi_ollama_rag.services.auth.email_service.send_otp_email",
    new_callable=AsyncMock,
    side_effect=RuntimeError("SMTP failed"),
)
async def test_request_registration_otp_email_failure_propagates(mock_send_email):
    """
    Edge Case: If send_otp_email raises, the exception should propagate
    (the function has no try/except around it).
    """
    email = f"new_{uuid.uuid4()}@example.com"

    with pytest.raises(RuntimeError, match="SMTP failed"):
        await request_registration_otp(email)

    # OTP was still saved to DB before the email send failed
    # (save_otp is called before send_otp_email at L26-27)


# ===================================================================
# verify_and_register_user
# ===================================================================


@pytest.mark.asyncio
@patch("fastapi_ollama_rag.services.auth.email_service.send_otp_email", new_callable=AsyncMock)
async def test_verify_and_register_user_success(mock_send_email):
    """Test successfully consuming an OTP to register a user."""
    email = f"register_{uuid.uuid4()}@example.com"
    password = "MySecurePassword123!"

    await request_registration_otp(email)

    _, kwargs = mock_send_email.call_args
    otp_code = kwargs["otp"]

    response = await verify_and_register_user(email, otp_code, password)
    assert response["message"] == "User registered successfully. You may now log in."

    user_in_db = await user_repo.get_user_by_email(email)
    assert user_in_db is not None


@pytest.mark.asyncio
async def test_verify_and_register_invalid_otp():
    """Test that a bad OTP triggers a 400 Bad Request HTTPException."""
    email = f"bad_otp_{uuid.uuid4()}@example.com"

    with pytest.raises(HTTPException) as exc_info:
        await verify_and_register_user(email, "000000", "password123")

    assert exc_info.value.status_code == 400
    assert "Invalid or expired" in exc_info.value.detail


@pytest.mark.asyncio
@patch("fastapi_ollama_rag.services.auth.email_service.send_otp_email", new_callable=AsyncMock)
async def test_verify_and_register_user_already_registered(mock_send_email, test_user):
    """
    Edge Case: Valid OTP but user already exists (race condition or replay).
    L41-45: Should raise 400 "User already registered."
    """
    email = test_user["email"]

    # Manually save a valid OTP for this already-registered user
    from datetime import datetime, timedelta, UTC
    expires_at = datetime.now(UTC) + timedelta(minutes=10)
    otp_code = "654321"
    await user_repo.save_otp(email, otp_code, expires_at)

    with pytest.raises(HTTPException) as exc_info:
        await verify_and_register_user(email, otp_code, "password123")

    assert exc_info.value.status_code == 400
    assert "User already registered" in exc_info.value.detail

    # Cleanup
    await user_repo.delete_otps_for_email(email)


@pytest.mark.asyncio
@patch("fastapi_ollama_rag.services.auth.email_service.send_otp_email", new_callable=AsyncMock)
async def test_verify_and_register_password_is_hashed(mock_send_email):
    """
    Edge Case: The stored password should be a bcrypt hash, NOT the plaintext.
    L47: hashed_pwd = security.get_password_hash(password)
    """
    email = f"register_{uuid.uuid4()}@example.com"
    raw_password = "PlainTextPassword123!"

    await request_registration_otp(email)
    _, kwargs = mock_send_email.call_args
    otp_code = kwargs["otp"]

    await verify_and_register_user(email, otp_code, raw_password)

    user = await user_repo.get_user_by_email(email)
    assert user is not None
    # The stored password should NOT be the raw password
    assert user["hashed_password"] != raw_password
    # It should be a valid bcrypt hash that verifies against the original
    assert verify_password(raw_password, user["hashed_password"]) is True


@pytest.mark.asyncio
@patch("fastapi_ollama_rag.services.auth.email_service.send_otp_email", new_callable=AsyncMock)
async def test_verify_and_register_cleans_up_otps(mock_send_email):
    """
    Edge Case: After registration, OTPs for the email should be deleted.
    L50: await user_repo.delete_otps_for_email(email)
    """
    email = f"register_{uuid.uuid4()}@example.com"

    await request_registration_otp(email)
    _, kwargs = mock_send_email.call_args
    otp_code = kwargs["otp"]

    await verify_and_register_user(email, otp_code, "Password123!")

    # The OTP should no longer be valid
    assert await user_repo.get_valid_otp(email, otp_code) is None


@pytest.mark.asyncio
@patch("fastapi_ollama_rag.services.auth.email_service.send_otp_email", new_callable=AsyncMock)
async def test_verify_and_register_wrong_email_for_otp(mock_send_email):
    """
    Edge Case: OTP generated for email A should NOT work for email B.
    """
    email_a = f"user_a_{uuid.uuid4()}@example.com"
    email_b = f"user_b_{uuid.uuid4()}@example.com"

    await request_registration_otp(email_a)
    _, kwargs = mock_send_email.call_args
    otp_code = kwargs["otp"]

    with pytest.raises(HTTPException) as exc_info:
        await verify_and_register_user(email_b, otp_code, "Password123!")

    assert exc_info.value.status_code == 400
    assert "Invalid or expired" in exc_info.value.detail

    # Cleanup
    await user_repo.delete_otps_for_email(email_a)


@pytest.mark.asyncio
@patch("fastapi_ollama_rag.services.auth.email_service.send_otp_email", new_callable=AsyncMock)
async def test_verify_and_register_returns_dict(mock_send_email):
    """Return type should be a dict with a 'message' key."""
    email = f"register_{uuid.uuid4()}@example.com"

    await request_registration_otp(email)
    _, kwargs = mock_send_email.call_args
    otp_code = kwargs["otp"]

    result = await verify_and_register_user(email, otp_code, "Password123!")

    assert isinstance(result, dict)
    assert "message" in result


@pytest.mark.asyncio
async def test_verify_and_register_expired_otp():
    """
    Edge Case: An OTP that has expired (expires_at < NOW()) should be rejected.
    The SQL has `expires_at > NOW()`, so expired OTPs are invisible to get_valid_otp.
    """
    from datetime import datetime, timedelta, UTC

    email = f"expired_{uuid.uuid4()}@example.com"
    otp_code = "987654"
    expired_at = datetime.now(UTC) - timedelta(minutes=5)

    # Manually save an already-expired OTP
    await user_repo.save_otp(email, otp_code, expired_at)

    with pytest.raises(HTTPException) as exc_info:
        await verify_and_register_user(email, otp_code, "Password123!")

    assert exc_info.value.status_code == 400
    assert "Invalid or expired" in exc_info.value.detail

    # Cleanup
    await user_repo.delete_otps_for_email(email)


@pytest.mark.asyncio
@patch("fastapi_ollama_rag.services.auth.email_service.send_otp_email", new_callable=AsyncMock)
async def test_verify_and_register_otp_replay_attack(mock_send_email):
    """
    Edge Case: After successful registration, the OTP is deleted.
    Trying to register again with the same OTP should fail with 400
    (either "Invalid or expired" because OTPs were deleted, or
    "User already registered" if the OTP somehow still exists).
    """
    email = f"replay_{uuid.uuid4()}@example.com"
    password = "ReplayTestPw123!"

    await request_registration_otp(email)
    _, kwargs = mock_send_email.call_args
    otp_code = kwargs["otp"]

    # First registration — succeeds
    await verify_and_register_user(email, otp_code, password)

    # Second attempt with same OTP — should fail
    with pytest.raises(HTTPException) as exc_info:
        await verify_and_register_user(email, otp_code, "DifferentPw456!")

    assert exc_info.value.status_code == 400
    # Could be either error message depending on OTP/user state
    assert ("Invalid or expired" in exc_info.value.detail or
            "User already registered" in exc_info.value.detail)


# ===================================================================
# authenticate_user
# ===================================================================


@pytest.mark.asyncio
async def test_authenticate_user_success(test_user):
    """Test logging in with correct credentials returns a JWT."""
    email = test_user["email"]

    raw_password = "known_password_123"
    await user_repo.update_user_password(email, get_password_hash(raw_password))

    token = await authenticate_user(email, raw_password)

    assert isinstance(token, str)
    assert len(token) > 20


@pytest.mark.asyncio
async def test_authenticate_user_invalid_password(test_user):
    """Test logging in with the wrong password triggers 401 Unauthorized."""
    valid_hash = get_password_hash("ActualCorrectPassword123!")
    await user_repo.update_user_password(test_user["email"], valid_hash)

    with pytest.raises(HTTPException) as exc_info:
        await authenticate_user(test_user["email"], "WrongPassword!!!")

    assert exc_info.value.status_code == 401
    assert "Incorrect email or password" in exc_info.value.detail


@pytest.mark.asyncio
async def test_authenticate_user_nonexistent_email():
    """
    Edge Case: Email doesn't exist in DB at all.
    L66: `not user` is True → 401.
    """
    fake_email = f"ghost_{uuid.uuid4()}@example.com"

    with pytest.raises(HTTPException) as exc_info:
        await authenticate_user(fake_email, "AnyPassword123!")

    assert exc_info.value.status_code == 401
    assert "Incorrect email or password" in exc_info.value.detail


@pytest.mark.asyncio
async def test_authenticate_user_401_has_www_authenticate_header(test_user):
    """
    Edge Case: The 401 response should include WWW-Authenticate: Bearer header.
    L63: headers={"WWW-Authenticate": "Bearer"}
    """
    valid_hash = get_password_hash("CorrectPw123!")
    await user_repo.update_user_password(test_user["email"], valid_hash)

    with pytest.raises(HTTPException) as exc_info:
        await authenticate_user(test_user["email"], "WrongPw!!!")

    assert exc_info.value.headers == {"WWW-Authenticate": "Bearer"}


@pytest.mark.asyncio
async def test_authenticate_user_jwt_contains_sub_claim(test_user):
    """
    Edge Case: JWT should contain a 'sub' claim with the user's ID.
    L70: data={"sub": str(user["id"])}
    """
    import jwt
    from fastapi_ollama_rag.core.config import settings

    email = test_user["email"]
    raw_password = "JwtTestPw123!"
    await user_repo.update_user_password(email, get_password_hash(raw_password))

    token = await authenticate_user(email, raw_password)

    decoded = jwt.decode(
        token, settings.secret_key, algorithms=[settings.jwt_algorithm]
    )
    assert "sub" in decoded
    assert decoded["sub"] == test_user["id"]


@pytest.mark.asyncio
async def test_authenticate_user_jwt_has_exp_claim(test_user):
    """Edge Case: JWT should contain an 'exp' (expiration) claim."""
    import jwt
    from fastapi_ollama_rag.core.config import settings

    email = test_user["email"]
    raw_password = "ExpTestPw123!"
    await user_repo.update_user_password(email, get_password_hash(raw_password))

    token = await authenticate_user(email, raw_password)

    decoded = jwt.decode(
        token, settings.secret_key, algorithms=[settings.jwt_algorithm]
    )
    assert "exp" in decoded


@pytest.mark.asyncio
async def test_authenticate_user_nonexistent_same_error_as_wrong_password(test_user):
    """
    Security: Both non-existent email and wrong password should produce
    the exact same error message (prevent user enumeration).
    """
    valid_hash = get_password_hash("RealPw123!")
    await user_repo.update_user_password(test_user["email"], valid_hash)

    # Wrong password
    with pytest.raises(HTTPException) as exc_wrong_pw:
        await authenticate_user(test_user["email"], "BadPw!!!")

    # Non-existent email
    with pytest.raises(HTTPException) as exc_no_user:
        await authenticate_user(f"ghost_{uuid.uuid4()}@example.com", "AnyPw123!")

    assert exc_wrong_pw.value.status_code == exc_no_user.value.status_code
    assert exc_wrong_pw.value.detail == exc_no_user.value.detail


# ===================================================================
# request_password_reset_otp
# ===================================================================


@pytest.mark.asyncio
@patch("fastapi_ollama_rag.services.auth.email_service.send_otp_email", new_callable=AsyncMock)
async def test_request_password_reset_otp_success(mock_send_email, test_user):
    """Existing user requests a reset → OTP generated and email sent."""
    email = test_user["email"]

    await request_password_reset_otp(email)

    mock_send_email.assert_awaited_once()
    _, kwargs = mock_send_email.call_args
    assert kwargs["to_email"] == email
    assert len(kwargs["otp"]) == 6
    assert kwargs["otp"].isdigit()

    # Cleanup
    await user_repo.delete_otps_for_email(email)


@pytest.mark.asyncio
@patch("fastapi_ollama_rag.services.auth.email_service.send_otp_email", new_callable=AsyncMock)
async def test_request_password_reset_otp_nonexistent_user(mock_send_email):
    """
    Edge Case: Non-existent email → early return, no email sent.
    L80-82: `if not user: return`
    """
    fake_email = f"ghost_{uuid.uuid4()}@example.com"

    await request_password_reset_otp(fake_email)

    mock_send_email.assert_not_awaited()


@pytest.mark.asyncio
@patch("fastapi_ollama_rag.services.auth.email_service.send_otp_email", new_callable=AsyncMock)
async def test_request_password_reset_otp_returns_none(mock_send_email, test_user):
    """Return type should be None."""
    result = await request_password_reset_otp(test_user["email"])
    assert result is None

    # Cleanup
    await user_repo.delete_otps_for_email(test_user["email"])


@pytest.mark.asyncio
@patch("fastapi_ollama_rag.services.auth.email_service.send_otp_email", new_callable=AsyncMock)
async def test_request_password_reset_otp_nonexistent_returns_none(mock_send_email):
    """Early-return path for non-existent user also returns None."""
    result = await request_password_reset_otp(f"ghost_{uuid.uuid4()}@example.com")
    assert result is None


@pytest.mark.asyncio
@patch("fastapi_ollama_rag.services.auth.email_service.send_otp_email", new_callable=AsyncMock)
async def test_request_password_reset_otp_saves_to_db(mock_send_email, test_user):
    """The reset OTP should be saved in the DB and retrievable."""
    email = test_user["email"]

    await request_password_reset_otp(email)

    _, kwargs = mock_send_email.call_args
    otp = kwargs["otp"]

    valid = await user_repo.get_valid_otp(email, otp)
    assert valid is not None

    # Cleanup
    await user_repo.delete_otps_for_email(email)


@pytest.mark.asyncio
@patch(
    "fastapi_ollama_rag.services.auth.email_service.send_otp_email",
    new_callable=AsyncMock,
    side_effect=RuntimeError("SMTP failed"),
)
async def test_request_password_reset_otp_email_failure_propagates(
    mock_send_email, test_user
):
    """
    Edge Case: If send_otp_email raises during password reset, the exception
    should propagate (function has no try/except around it).
    """
    email = test_user["email"]

    with pytest.raises(RuntimeError, match="SMTP failed"):
        await request_password_reset_otp(email)


@pytest.mark.asyncio
@patch("fastapi_ollama_rag.services.auth.email_service.send_otp_email", new_callable=AsyncMock)
async def test_request_password_reset_otp_generates_6_digit_code(
    mock_send_email, test_user
):
    """The generated reset OTP should be exactly 6 digits."""
    await request_password_reset_otp(test_user["email"])

    _, kwargs = mock_send_email.call_args
    otp = kwargs["otp"]
    assert len(otp) == 6
    assert otp.isdigit()

    # Cleanup
    await user_repo.delete_otps_for_email(test_user["email"])


# ===================================================================
# reset_password
# ===================================================================


@pytest.mark.asyncio
@patch("fastapi_ollama_rag.services.auth.email_service.send_otp_email", new_callable=AsyncMock)
async def test_password_reset_flow(mock_send_email, test_user):
    """Test the full lifecycle of requesting a reset and consuming the OTP."""
    email = test_user["email"]
    new_password = "BrandNewPassword456!"

    await request_password_reset_otp(email)
    _, kwargs = mock_send_email.call_args
    otp_code = kwargs["otp"]

    response = await reset_password(email, otp_code, new_password)
    assert "reset successfully" in response["message"]

    token = await authenticate_user(email, new_password)
    assert token is not None


@pytest.mark.asyncio
async def test_reset_password_invalid_otp():
    """
    Edge Case: Invalid OTP for reset should raise 400.
    L96-100: `if not valid_otp: raise HTTPException(400)`
    """
    email = f"reset_{uuid.uuid4()}@example.com"

    with pytest.raises(HTTPException) as exc_info:
        await reset_password(email, "000000", "NewPassword123!")

    assert exc_info.value.status_code == 400
    assert "Invalid or expired" in exc_info.value.detail


@pytest.mark.asyncio
async def test_reset_password_expired_otp():
    """
    Edge Case: An expired OTP for password reset should be rejected.
    The SQL has `expires_at > NOW()`, so expired OTPs return None.
    """
    from datetime import datetime, timedelta, UTC

    email = f"expired_reset_{uuid.uuid4()}@example.com"
    otp_code = "654321"
    expired_at = datetime.now(UTC) - timedelta(minutes=5)

    await user_repo.save_otp(email, otp_code, expired_at)

    with pytest.raises(HTTPException) as exc_info:
        await reset_password(email, otp_code, "NewPassword123!")

    assert exc_info.value.status_code == 400
    assert "Invalid or expired" in exc_info.value.detail

    # Cleanup
    await user_repo.delete_otps_for_email(email)


@pytest.mark.asyncio
@patch("fastapi_ollama_rag.services.auth.email_service.send_otp_email", new_callable=AsyncMock)
async def test_reset_password_wrong_email_for_otp(mock_send_email, test_user):
    """
    Edge Case: OTP generated for user A should NOT work for resetting user B's password.
    """
    email_a = test_user["email"]

    await request_password_reset_otp(email_a)
    _, kwargs = mock_send_email.call_args
    otp_code = kwargs["otp"]

    fake_email = f"other_{uuid.uuid4()}@example.com"

    with pytest.raises(HTTPException) as exc_info:
        await reset_password(fake_email, otp_code, "NewPassword123!")

    assert exc_info.value.status_code == 400
    assert "Invalid or expired" in exc_info.value.detail

    # Cleanup
    await user_repo.delete_otps_for_email(email_a)


@pytest.mark.asyncio
@patch("fastapi_ollama_rag.services.auth.email_service.send_otp_email", new_callable=AsyncMock)
async def test_reset_password_hashes_new_password(mock_send_email, test_user):
    """
    Edge Case: The new password should be bcrypt-hashed before storing.
    L102: hashed_pwd = security.get_password_hash(new_password)
    """
    email = test_user["email"]
    new_password = "HashedCheckPw456!"

    await request_password_reset_otp(email)
    _, kwargs = mock_send_email.call_args
    otp_code = kwargs["otp"]

    await reset_password(email, otp_code, new_password)

    user = await user_repo.get_user_by_email(email)
    assert user["hashed_password"] != new_password
    assert verify_password(new_password, user["hashed_password"]) is True


@pytest.mark.asyncio
@patch("fastapi_ollama_rag.services.auth.email_service.send_otp_email", new_callable=AsyncMock)
async def test_reset_password_cleans_up_otps(mock_send_email, test_user):
    """
    Edge Case: After reset, OTPs for the email should be deleted.
    L105: await user_repo.delete_otps_for_email(email)
    """
    email = test_user["email"]

    await request_password_reset_otp(email)
    _, kwargs = mock_send_email.call_args
    otp_code = kwargs["otp"]

    await reset_password(email, otp_code, "NewPassword123!")

    # OTP should no longer be valid (consumed/deleted)
    assert await user_repo.get_valid_otp(email, otp_code) is None


@pytest.mark.asyncio
@patch("fastapi_ollama_rag.services.auth.email_service.send_otp_email", new_callable=AsyncMock)
async def test_reset_password_otp_cannot_be_reused(mock_send_email, test_user):
    """
    Edge Case: Using the same OTP twice should fail the second time
    because OTPs are deleted after the first successful reset.
    """
    email = test_user["email"]

    await request_password_reset_otp(email)
    _, kwargs = mock_send_email.call_args
    otp_code = kwargs["otp"]

    # First reset — should succeed
    await reset_password(email, otp_code, "FirstNewPw123!")

    # Second reset with same OTP — should fail
    with pytest.raises(HTTPException) as exc_info:
        await reset_password(email, otp_code, "SecondNewPw456!")

    assert exc_info.value.status_code == 400
    assert "Invalid or expired" in exc_info.value.detail


@pytest.mark.asyncio
@patch("fastapi_ollama_rag.services.auth.email_service.send_otp_email", new_callable=AsyncMock)
async def test_reset_password_old_password_no_longer_works(mock_send_email, test_user):
    """
    Edge Case: After reset, the old password should NOT authenticate.
    """
    email = test_user["email"]
    old_password = "OldPassword123!"
    new_password = "NewPassword456!"

    # Set a known old password
    await user_repo.update_user_password(email, get_password_hash(old_password))

    # Verify old password works
    token = await authenticate_user(email, old_password)
    assert token is not None

    # Request and consume reset OTP
    await request_password_reset_otp(email)
    _, kwargs = mock_send_email.call_args
    otp_code = kwargs["otp"]
    await reset_password(email, otp_code, new_password)

    # Old password should no longer work
    with pytest.raises(HTTPException) as exc_info:
        await authenticate_user(email, old_password)
    assert exc_info.value.status_code == 401

    # New password should work
    token = await authenticate_user(email, new_password)
    assert token is not None


@pytest.mark.asyncio
@patch("fastapi_ollama_rag.services.auth.email_service.send_otp_email", new_callable=AsyncMock)
async def test_reset_password_returns_dict(mock_send_email, test_user):
    """Return type should be a dict with a 'message' key."""
    email = test_user["email"]

    await request_password_reset_otp(email)
    _, kwargs = mock_send_email.call_args
    otp_code = kwargs["otp"]

    result = await reset_password(email, otp_code, "Password123!")

    assert isinstance(result, dict)
    assert "message" in result
    assert "reset successfully" in result["message"]



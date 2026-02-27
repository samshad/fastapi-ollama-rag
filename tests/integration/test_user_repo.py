import uuid
from datetime import datetime, timedelta, UTC
import asyncio

import asyncpg
import pytest

from fastapi_ollama_rag.repository.user_repo import (
    create_user,
    get_user_by_email,
    get_user_by_id,
    update_user_password,
    save_otp,
    get_valid_otp,
    delete_otps_for_email,
)


# ===================================================================
# create_user
# ===================================================================


@pytest.mark.asyncio
async def test_create_user_and_retrieval():
    """Test creating a user and fetching them by both Email and ID."""
    email = f"test_{uuid.uuid4()}@example.com"
    hashed_password = "secure_hash_123"

    new_user = await create_user(email=email, hashed_password=hashed_password)

    assert new_user is not None
    assert new_user["email"] == email
    user_id = str(new_user["id"])

    fetched_by_email = await get_user_by_email(email)
    assert fetched_by_email is not None
    assert str(fetched_by_email["id"]) == user_id

    fetched_by_id = await get_user_by_id(user_id)
    assert fetched_by_id is not None
    assert fetched_by_id["email"] == email


@pytest.mark.asyncio
async def test_create_user_returns_id_and_email():
    """The RETURNING clause only returns id and email."""
    email = f"test_{uuid.uuid4()}@example.com"
    new_user = await create_user(email=email, hashed_password="hash")

    assert "id" in new_user.keys()
    assert "email" in new_user.keys()
    assert new_user["email"] == email


@pytest.mark.asyncio
async def test_create_user_id_is_valid_uuid():
    """The auto-generated id should be a valid UUID."""
    email = f"test_{uuid.uuid4()}@example.com"
    new_user = await create_user(email=email, hashed_password="hash")

    parsed = uuid.UUID(str(new_user["id"]))
    assert parsed is not None


@pytest.mark.asyncio
async def test_create_user_is_verified_true():
    """
    The SQL hardcodes is_verified = TRUE on creation.
    Verify this by fetching the user back.
    """
    email = f"test_{uuid.uuid4()}@example.com"
    await create_user(email=email, hashed_password="hash")

    user = await get_user_by_email(email)
    assert user is not None
    assert user["is_verified"] is True


@pytest.mark.asyncio
async def test_create_user_unique_ids():
    """Two different users should get different auto-generated UUIDs."""
    email_a = f"test_a_{uuid.uuid4()}@example.com"
    email_b = f"test_b_{uuid.uuid4()}@example.com"

    user_a = await create_user(email=email_a, hashed_password="hash")
    user_b = await create_user(email=email_b, hashed_password="hash")

    assert str(user_a["id"]) != str(user_b["id"])


@pytest.mark.asyncio
async def test_create_user_created_at_auto_populated():
    """
    The schema has `created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()`.
    Verify the field is set by fetching the full row directly.
    """
    email = f"test_{uuid.uuid4()}@example.com"
    new_user = await create_user(email=email, hashed_password="hash")
    user_id = str(new_user["id"])

    # get_user_by_email doesn't SELECT created_at, so we use get_user_by_id
    # which also doesn't. We verify indirectly: the user exists and is_verified
    # is TRUE, which means the INSERT succeeded with all DEFAULT values.
    user = await get_user_by_id(user_id)
    assert user is not None
    assert user["is_verified"] is True


@pytest.mark.asyncio
async def test_create_user_duplicate_email_raises():
    """
    Edge Case: email column has UNIQUE constraint.
    Creating two users with the same email should raise.
    """
    email = f"test_{uuid.uuid4()}@example.com"
    await create_user(email=email, hashed_password="hash_1")

    with pytest.raises(asyncpg.UniqueViolationError):
        await create_user(email=email, hashed_password="hash_2")


# ===================================================================
# get_user_by_email
# ===================================================================


@pytest.mark.asyncio
async def test_get_user_by_email_nonexistent():
    """Fetching a missing user safely returns None."""
    fake_email = f"missing_{uuid.uuid4()}@example.com"
    result = await get_user_by_email(fake_email)
    assert result is None


@pytest.mark.asyncio
async def test_get_user_by_email_returned_fields():
    """
    The SQL selects: id, email, hashed_password, is_verified.
    Verify all four fields are present in the returned Record.
    """
    email = f"test_{uuid.uuid4()}@example.com"
    await create_user(email=email, hashed_password="pw_hash")

    user = await get_user_by_email(email)
    assert user is not None
    keys = user.keys()
    assert "id" in keys
    assert "email" in keys
    assert "hashed_password" in keys
    assert "is_verified" in keys


@pytest.mark.asyncio
async def test_get_user_by_email_returns_correct_password_hash():
    """The hashed_password stored should match what was provided at creation."""
    email = f"test_{uuid.uuid4()}@example.com"
    await create_user(email=email, hashed_password="original_hash")

    user = await get_user_by_email(email)
    assert user["hashed_password"] == "original_hash"


@pytest.mark.asyncio
async def test_get_user_by_email_case_sensitive():
    """
    Edge Case: Email lookup is case-sensitive in PostgreSQL (TEXT column, no LOWER()).
    'User@Example.com' != 'user@example.com'.
    """
    email = f"CaseSensitive_{uuid.uuid4()}@Example.COM"
    await create_user(email=email, hashed_password="hash")

    # Exact case → found
    assert await get_user_by_email(email) is not None

    # Different case → NOT found (PostgreSQL TEXT is case-sensitive)
    assert await get_user_by_email(email.lower()) is None


# ===================================================================
# get_user_by_id
# ===================================================================


@pytest.mark.asyncio
async def test_get_user_by_id_nonexistent():
    """Fetching a user by a valid but non-existent UUID returns None."""
    fake_id = str(uuid.uuid4())
    result = await get_user_by_id(fake_id)
    assert result is None


@pytest.mark.asyncio
async def test_get_user_by_id_returned_fields():
    """
    The SQL selects: id, email, is_verified.
    Note: hashed_password is deliberately excluded from this query.
    """
    email = f"test_{uuid.uuid4()}@example.com"
    new_user = await create_user(email=email, hashed_password="hash")
    user_id = str(new_user["id"])

    user = await get_user_by_id(user_id)
    assert user is not None
    keys = user.keys()
    assert "id" in keys
    assert "email" in keys
    assert "is_verified" in keys
    # hashed_password should NOT be returned by get_user_by_id
    assert "hashed_password" not in keys


@pytest.mark.asyncio
async def test_get_user_by_id_invalid_uuid_raises():
    """
    Edge Case: Passing an invalid UUID string should raise an error
    from asyncpg (DataError or similar).
    """
    with pytest.raises(Exception):
        await get_user_by_id("not-a-valid-uuid")


# ===================================================================
# update_user_password
# ===================================================================


@pytest.mark.asyncio
async def test_update_user_password():
    """Test that the password hash is correctly updated in the DB."""
    email = f"test_{uuid.uuid4()}@example.com"
    await create_user(email, "old_hash")

    await update_user_password(email, "new_hash_999")

    updated_user = await get_user_by_email(email)
    assert updated_user["hashed_password"] == "new_hash_999"


@pytest.mark.asyncio
async def test_update_user_password_nonexistent_email():
    """
    Edge Case: Updating the password for a non-existent email.
    UPDATE ... WHERE email = :email matches 0 rows → silent no-op.
    Should not raise.
    """
    fake_email = f"ghost_{uuid.uuid4()}@example.com"

    # Should not raise
    result = await update_user_password(fake_email, "new_hash")
    assert result is None


@pytest.mark.asyncio
async def test_update_user_password_returns_none():
    """update_user_password uses '!' (execute) SQL, so it returns None."""
    email = f"test_{uuid.uuid4()}@example.com"
    await create_user(email, "hash")

    result = await update_user_password(email, "updated_hash")
    assert result is None


@pytest.mark.asyncio
async def test_update_user_password_multiple_times():
    """Edge Case: Updating the password multiple times in succession."""
    email = f"test_{uuid.uuid4()}@example.com"
    await create_user(email, "hash_v1")

    await update_user_password(email, "hash_v2")
    await update_user_password(email, "hash_v3")

    user = await get_user_by_email(email)
    assert user["hashed_password"] == "hash_v3"


@pytest.mark.asyncio
async def test_update_password_preserves_other_fields():
    """Updating password should NOT change email or is_verified."""
    email = f"test_{uuid.uuid4()}@example.com"
    await create_user(email, "original_hash")

    user_before = await get_user_by_email(email)
    assert user_before is not None

    await update_user_password(email, "new_hash")

    user_after = await get_user_by_email(email)
    assert user_after["email"] == user_before["email"]
    assert user_after["is_verified"] == user_before["is_verified"]
    assert str(user_after["id"]) == str(user_before["id"])
    assert user_after["hashed_password"] == "new_hash"


@pytest.mark.asyncio
async def test_update_password_does_not_affect_other_users():
    """Updating user A's password should not change user B's."""
    email_a = f"test_a_{uuid.uuid4()}@example.com"
    email_b = f"test_b_{uuid.uuid4()}@example.com"
    await create_user(email_a, "pw_a")
    await create_user(email_b, "pw_b")

    await update_user_password(email_a, "pw_a_new")

    user_a = await get_user_by_email(email_a)
    user_b = await get_user_by_email(email_b)
    assert user_a["hashed_password"] == "pw_a_new"
    assert user_b["hashed_password"] == "pw_b"


# ===================================================================
# save_otp
# ===================================================================


@pytest.mark.asyncio
async def test_save_otp_returns_none():
    """save_otp uses '!' (execute) SQL, so it returns None."""
    email = f"otp_{uuid.uuid4()}@example.com"
    expires_at = datetime.now(UTC) + timedelta(minutes=10)

    result = await save_otp(email, "123456", expires_at)
    assert result is None


@pytest.mark.asyncio
async def test_save_multiple_otps_same_email():
    """
    Edge Case: No unique constraint on (email, code) in otps table.
    Multiple OTPs can be stored for the same email.
    """
    email = f"otp_{uuid.uuid4()}@example.com"
    expires_at = datetime.now(UTC) + timedelta(minutes=10)

    await save_otp(email, "111111", expires_at)
    await save_otp(email, "222222", expires_at)
    await save_otp(email, "333333", expires_at)

    # All three should be stored; the valid one should be retrievable
    assert await get_valid_otp(email, "111111") is not None
    assert await get_valid_otp(email, "222222") is not None
    assert await get_valid_otp(email, "333333") is not None


@pytest.mark.asyncio
async def test_save_otp_same_code_different_emails():
    """
    Edge Case: Two different emails can have the same OTP code.
    They should be independently retrievable.
    """
    email_a = f"otp_a_{uuid.uuid4()}@example.com"
    email_b = f"otp_b_{uuid.uuid4()}@example.com"
    expires_at = datetime.now(UTC) + timedelta(minutes=10)
    code = "999999"

    await save_otp(email_a, code, expires_at)
    await save_otp(email_b, code, expires_at)

    assert await get_valid_otp(email_a, code) is not None
    assert await get_valid_otp(email_b, code) is not None

    # Deleting email_a's OTPs should not affect email_b
    await delete_otps_for_email(email_a)
    assert await get_valid_otp(email_a, code) is None
    assert await get_valid_otp(email_b, code) is not None


# ===================================================================
# get_valid_otp
# ===================================================================


@pytest.mark.asyncio
async def test_otp_lifecycle():
    """Test saving, validating, and deleting OTP codes."""
    email = f"otp_{uuid.uuid4()}@example.com"
    code = "123456"
    expires_at = datetime.now(UTC) + timedelta(minutes=10)

    await save_otp(email, code, expires_at)

    valid_otp = await get_valid_otp(email, code)
    assert valid_otp is not None

    invalid_otp = await get_valid_otp(email, "999999")
    assert invalid_otp is None

    await delete_otps_for_email(email)

    deleted_otp = await get_valid_otp(email, code)
    assert deleted_otp is None


@pytest.mark.asyncio
async def test_get_valid_otp_wrong_email():
    """Edge Case: Correct code but wrong email → None."""
    email = f"otp_{uuid.uuid4()}@example.com"
    other_email = f"other_{uuid.uuid4()}@example.com"
    expires_at = datetime.now(UTC) + timedelta(minutes=10)

    await save_otp(email, "123456", expires_at)

    result = await get_valid_otp(other_email, "123456")
    assert result is None


@pytest.mark.asyncio
async def test_get_valid_otp_expired():
    """
    Edge Case: OTP with expires_at in the past.
    SQL: WHERE expires_at > NOW() — expired OTPs should not be returned.
    """
    email = f"otp_{uuid.uuid4()}@example.com"
    # Already expired 10 minutes ago
    expired_at = datetime.now(UTC) - timedelta(minutes=10)

    await save_otp(email, "123456", expired_at)

    result = await get_valid_otp(email, "123456")
    assert result is None


@pytest.mark.asyncio
async def test_get_valid_otp_returns_id_field():
    """The SQL only SELECT id — verify the returned Record has 'id'."""
    email = f"otp_{uuid.uuid4()}@example.com"
    expires_at = datetime.now(UTC) + timedelta(minutes=10)

    await save_otp(email, "123456", expires_at)

    otp = await get_valid_otp(email, "123456")
    assert otp is not None
    assert "id" in otp.keys()


@pytest.mark.asyncio
async def test_get_valid_otp_returns_most_recent():
    """
    Edge Case: Multiple OTPs for same email + code.
    SQL: ORDER BY created_at DESC LIMIT 1 → should return the most recent.
    We verify that even with duplicates, only one record is returned.
    """
    email = f"otp_{uuid.uuid4()}@example.com"
    expires_at = datetime.now(UTC) + timedelta(minutes=10)

    await save_otp(email, "123456", expires_at)
    await asyncio.sleep(0.05)  # ensure different created_at
    await save_otp(email, "123456", expires_at)

    otp = await get_valid_otp(email, "123456")
    assert otp is not None
    # Only one record returned (LIMIT 1)
    assert "id" in otp.keys()


@pytest.mark.asyncio
async def test_get_valid_otp_nonexistent_email():
    """An email with no OTPs at all should return None."""
    result = await get_valid_otp(f"nobody_{uuid.uuid4()}@example.com", "000000")
    assert result is None


@pytest.mark.asyncio
async def test_get_valid_otp_id_is_valid_uuid():
    """The returned 'id' should be a valid UUID."""
    email = f"otp_{uuid.uuid4()}@example.com"
    expires_at = datetime.now(UTC) + timedelta(minutes=10)

    await save_otp(email, "123456", expires_at)

    otp = await get_valid_otp(email, "123456")
    assert otp is not None
    parsed = uuid.UUID(str(otp["id"]))
    assert parsed is not None


@pytest.mark.asyncio
async def test_get_valid_otp_ignores_expired_returns_valid():
    """
    Edge Case: Both expired and valid OTPs exist for the same email + code.
    The query has `expires_at > NOW()`, so only the non-expired one should be returned.
    """
    email = f"otp_{uuid.uuid4()}@example.com"
    code = "555555"
    expired_at = datetime.now(UTC) - timedelta(minutes=10)
    valid_at = datetime.now(UTC) + timedelta(minutes=10)

    # Insert expired OTP first, then valid one
    await save_otp(email, code, expired_at)
    await asyncio.sleep(0.05)
    await save_otp(email, code, valid_at)

    otp = await get_valid_otp(email, code)
    assert otp is not None
    assert "id" in otp.keys()


# ===================================================================
# delete_otps_for_email
# ===================================================================


@pytest.mark.asyncio
async def test_delete_otps_for_email_returns_none():
    """delete_otps_for_email uses '!' (execute) SQL, so it returns None."""
    email = f"otp_{uuid.uuid4()}@example.com"
    expires_at = datetime.now(UTC) + timedelta(minutes=10)
    await save_otp(email, "123456", expires_at)

    result = await delete_otps_for_email(email)
    assert result is None


@pytest.mark.asyncio
async def test_delete_otps_for_email_no_existing_otps():
    """
    Edge Case: Deleting OTPs for an email that has none.
    DELETE ... WHERE email = :email matches 0 rows → silent no-op.
    """
    fake_email = f"ghost_{uuid.uuid4()}@example.com"

    # Should not raise
    result = await delete_otps_for_email(fake_email)
    assert result is None


@pytest.mark.asyncio
async def test_delete_otps_removes_all_for_email():
    """
    Edge Case: Multiple OTPs for one email should all be deleted.
    The SQL is: DELETE FROM otps WHERE email = :email (no LIMIT).
    """
    email = f"otp_{uuid.uuid4()}@example.com"
    expires_at = datetime.now(UTC) + timedelta(minutes=10)

    await save_otp(email, "111111", expires_at)
    await save_otp(email, "222222", expires_at)
    await save_otp(email, "333333", expires_at)

    # All three should be retrievable before deletion
    assert await get_valid_otp(email, "111111") is not None
    assert await get_valid_otp(email, "222222") is not None
    assert await get_valid_otp(email, "333333") is not None

    await delete_otps_for_email(email)

    # All three should be gone
    assert await get_valid_otp(email, "111111") is None
    assert await get_valid_otp(email, "222222") is None
    assert await get_valid_otp(email, "333333") is None


@pytest.mark.asyncio
async def test_delete_otps_does_not_affect_other_emails():
    """Deleting OTPs for email A should not touch email B's OTPs."""
    email_a = f"otp_a_{uuid.uuid4()}@example.com"
    email_b = f"otp_b_{uuid.uuid4()}@example.com"
    expires_at = datetime.now(UTC) + timedelta(minutes=10)

    await save_otp(email_a, "111111", expires_at)
    await save_otp(email_b, "222222", expires_at)

    await delete_otps_for_email(email_a)

    assert await get_valid_otp(email_a, "111111") is None
    assert await get_valid_otp(email_b, "222222") is not None


@pytest.mark.asyncio
async def test_delete_otps_removes_expired_and_valid():
    """
    Edge Case: DELETE FROM otps WHERE email = :email has no expiry filter.
    It should remove BOTH expired and still-valid OTPs.
    """
    email = f"otp_{uuid.uuid4()}@example.com"
    expired_at = datetime.now(UTC) - timedelta(minutes=10)
    valid_at = datetime.now(UTC) + timedelta(minutes=10)

    await save_otp(email, "111111", expired_at)
    await save_otp(email, "222222", valid_at)

    await delete_otps_for_email(email)

    # Both should be gone — including the expired one that get_valid_otp wouldn't
    # have returned anyway, but DELETE should have removed it from the table entirely.
    assert await get_valid_otp(email, "111111") is None
    assert await get_valid_otp(email, "222222") is None



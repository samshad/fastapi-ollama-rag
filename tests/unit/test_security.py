from datetime import UTC, datetime, timedelta

import jwt
import pytest

from fastapi_ollama_rag.core.config import settings
from fastapi_ollama_rag.core.security import (
    create_access_token,
    get_password_hash,
    verify_password,
)


def test_password_hashing_and_verification():
    """
    Test that a password hashes correctly and can be verified.
    """
    plain_password = "SuperSecretPassword123!"

    hashed_password = get_password_hash(plain_password)

    assert hashed_password != plain_password

    assert hashed_password.startswith("$2")

    is_valid = verify_password(plain_password, hashed_password)
    assert is_valid is True

    is_invalid = verify_password("WrongPassword!", hashed_password)
    assert is_invalid is False


def test_create_access_token_default_expiration():
    """
    Test generating a JWT with the default expiration time from settings.
    """
    data = {"sub": "user_123"}

    token = create_access_token(data=data)
    assert isinstance(token, str)

    decoded_payload = jwt.decode(
        token, settings.secret_key, algorithms=[settings.jwt_algorithm]
    )

    assert decoded_payload["sub"] == "user_123"

    assert "exp" in decoded_payload

    exp_datetime = datetime.fromtimestamp(decoded_payload["exp"], tz=UTC)
    assert exp_datetime > datetime.now(UTC)


def test_create_access_token_custom_expiration():
    """
    Test generating a JWT with a specific custom expiration time.
    """
    data = {"sub": "user_456"}
    custom_delta = timedelta(days=7)

    token = create_access_token(data=data, expires_delta=custom_delta)

    decoded_payload = jwt.decode(
        token, settings.secret_key, algorithms=[settings.jwt_algorithm]
    )

    exp_timestamp = decoded_payload["exp"]
    exp_datetime = datetime.fromtimestamp(exp_timestamp, tz=UTC)

    expected_exp = datetime.now(UTC) + custom_delta
    difference = abs((exp_datetime - expected_exp).total_seconds())

    assert difference < 60


def test_password_hashing_is_salted():
    """
    Test that hashing the same password produces different hashes.
    """
    plain_password = "SuperSecretPassword123!"

    hash_one = get_password_hash(plain_password)
    hash_two = get_password_hash(plain_password)

    assert hash_one != hash_two
    assert verify_password(plain_password, hash_one) is True
    assert verify_password(plain_password, hash_two) is True


def test_verify_password_invalid_hash_raises():
    """
    Test that an invalid bcrypt hash raises an error in verification.
    """
    with pytest.raises(ValueError):
        verify_password("password", "not-a-valid-bcrypt-hash")


def test_create_access_token_does_not_mutate_input_data():
    """
    Test that input data is not mutated when creating a token.
    """
    data = {"sub": "user_789"}
    original = data.copy()

    _ = create_access_token(data=data)

    assert data == original


def test_create_access_token_default_expiration_uses_settings():
    """
    Test that default expiration aligns with settings within a small tolerance.
    """
    data = {"sub": "user_999"}
    token = create_access_token(data=data)

    decoded_payload = jwt.decode(
        token, settings.secret_key, algorithms=[settings.jwt_algorithm]
    )

    exp_datetime = datetime.fromtimestamp(decoded_payload["exp"], tz=UTC)
    expected_exp = datetime.now(UTC) + timedelta(
        minutes=settings.access_token_expire_minutes
    )
    difference = abs((exp_datetime - expected_exp).total_seconds())

    assert difference < 60

import pytest
from email.message import EmailMessage
from unittest.mock import AsyncMock, patch

from fastapi_ollama_rag.services.email_service import EmailService


# -------------------------------------------------------------------
# Fixtures
# -------------------------------------------------------------------
@pytest.fixture
def mock_settings():
    """Mocks the application settings so we can control SMTP variables."""
    with patch("fastapi_ollama_rag.services.email_service.settings") as mock:
        yield mock


@pytest.fixture
def configured_settings(mock_settings):
    """Provides a fully configured mock settings fixture."""
    mock_settings.smtp_username = "bot@myapp.com"
    mock_settings.smtp_password = "super-secret-password"
    mock_settings.smtp_server = "smtp.fake-mail.com"
    mock_settings.smtp_port = 587
    return mock_settings


# ===================================================================
# __init__ — constructor
# ===================================================================


def test_init_stores_all_settings(configured_settings):
    """Verify __init__ correctly reads and stores all 4 settings attributes."""
    service = EmailService()

    assert service.smtp_server == "smtp.fake-mail.com"
    assert service.smtp_port == 587
    assert service.smtp_username == "bot@myapp.com"
    assert service.smtp_password == "super-secret-password"


def test_init_stores_none_when_settings_missing(mock_settings):
    """When settings are None, the instance attributes should also be None."""
    mock_settings.smtp_server = None
    mock_settings.smtp_port = None
    mock_settings.smtp_username = None
    mock_settings.smtp_password = None

    service = EmailService()

    assert service.smtp_server is None
    assert service.smtp_port is None
    assert service.smtp_username is None
    assert service.smtp_password is None


# ===================================================================
# send_otp_email — missing credentials (early return at L25)
# ===================================================================


@pytest.mark.asyncio
@patch("fastapi_ollama_rag.services.email_service.aiosmtplib.send", new_callable=AsyncMock)
async def test_send_otp_email_missing_both_credentials(mock_send, mock_settings):
    """Both username and password are None → early return, no SMTP call."""
    mock_settings.smtp_username = None
    mock_settings.smtp_password = None

    service = EmailService()
    await service.send_otp_email("test@example.com", "123456")

    mock_send.assert_not_called()


@pytest.mark.asyncio
@patch("fastapi_ollama_rag.services.email_service.aiosmtplib.send", new_callable=AsyncMock)
async def test_send_otp_email_missing_username_only(mock_send, mock_settings):
    """
    Edge Case: smtp_username is None but smtp_password is set.
    L25: `not self.smtp_username` is True → early return.
    """
    mock_settings.smtp_username = None
    mock_settings.smtp_password = "has-password"

    service = EmailService()
    await service.send_otp_email("test@example.com", "123456")

    mock_send.assert_not_called()


@pytest.mark.asyncio
@patch("fastapi_ollama_rag.services.email_service.aiosmtplib.send", new_callable=AsyncMock)
async def test_send_otp_email_missing_password_only(mock_send, mock_settings):
    """
    Edge Case: smtp_password is None but smtp_username is set.
    L25: `not self.smtp_password` is True → early return.
    """
    mock_settings.smtp_username = "bot@myapp.com"
    mock_settings.smtp_password = None

    service = EmailService()
    await service.send_otp_email("test@example.com", "123456")

    mock_send.assert_not_called()


@pytest.mark.asyncio
@patch("fastapi_ollama_rag.services.email_service.aiosmtplib.send", new_callable=AsyncMock)
async def test_send_otp_email_empty_string_username(mock_send, mock_settings):
    """
    Edge Case: smtp_username is "" (empty string, falsy).
    `not ""` is True → early return.
    """
    mock_settings.smtp_username = ""
    mock_settings.smtp_password = "has-password"

    service = EmailService()
    await service.send_otp_email("test@example.com", "123456")

    mock_send.assert_not_called()


@pytest.mark.asyncio
@patch("fastapi_ollama_rag.services.email_service.aiosmtplib.send", new_callable=AsyncMock)
async def test_send_otp_email_empty_string_password(mock_send, mock_settings):
    """
    Edge Case: smtp_password is "" (empty string, falsy).
    `not ""` is True → early return.
    """
    mock_settings.smtp_username = "bot@myapp.com"
    mock_settings.smtp_password = ""

    service = EmailService()
    await service.send_otp_email("test@example.com", "123456")

    mock_send.assert_not_called()


@pytest.mark.asyncio
@patch("fastapi_ollama_rag.services.email_service.aiosmtplib.send", new_callable=AsyncMock)
async def test_send_otp_email_missing_credentials_returns_none(mock_send, mock_settings):
    """Early-return path for missing credentials returns None."""
    mock_settings.smtp_username = None
    mock_settings.smtp_password = None

    service = EmailService()
    result = await service.send_otp_email("test@example.com", "123456")

    assert result is None


# ===================================================================
# send_otp_email — success path (L33-54)
# ===================================================================


@pytest.mark.asyncio
@patch("fastapi_ollama_rag.services.email_service.aiosmtplib.send", new_callable=AsyncMock)
async def test_send_otp_email_success(mock_send, configured_settings):
    """Test that a fully configured email service dispatches correctly."""
    service = EmailService()
    await service.send_otp_email("target@example.com", "999999")

    mock_send.assert_awaited_once()

    args, kwargs = mock_send.call_args
    email_message = args[0]

    assert email_message["To"] == "target@example.com"
    assert email_message["From"] == "bot@myapp.com"
    assert email_message["Subject"] == "RAG Application Verification Code"
    assert "999999" in email_message.get_content()

    assert kwargs["hostname"] == "smtp.fake-mail.com"
    assert kwargs["port"] == 587
    assert kwargs["username"] == "bot@myapp.com"
    assert kwargs["password"] == "super-secret-password"


@pytest.mark.asyncio
@patch("fastapi_ollama_rag.services.email_service.aiosmtplib.send", new_callable=AsyncMock)
async def test_send_otp_email_returns_none_on_success(mock_send, configured_settings):
    """Successful send should return None (no return value)."""
    service = EmailService()
    result = await service.send_otp_email("target@example.com", "123456")

    assert result is None


@pytest.mark.asyncio
@patch("fastapi_ollama_rag.services.email_service.aiosmtplib.send", new_callable=AsyncMock)
async def test_send_otp_email_message_is_email_message_type(mock_send, configured_settings):
    """The first positional arg to aiosmtplib.send should be an EmailMessage instance."""
    service = EmailService()
    await service.send_otp_email("target@example.com", "123456")

    args, _ = mock_send.call_args
    assert isinstance(args[0], EmailMessage)


@pytest.mark.asyncio
@patch("fastapi_ollama_rag.services.email_service.aiosmtplib.send", new_callable=AsyncMock)
async def test_send_otp_email_start_tls_always_true(mock_send, configured_settings):
    """L50: start_tls=True should always be passed to aiosmtplib.send."""
    service = EmailService()
    await service.send_otp_email("target@example.com", "123456")

    _, kwargs = mock_send.call_args
    assert kwargs["start_tls"] is True


@pytest.mark.asyncio
@patch("fastapi_ollama_rag.services.email_service.aiosmtplib.send", new_callable=AsyncMock)
async def test_send_otp_email_body_contains_welcome(mock_send, configured_settings):
    """The email body should contain 'Welcome!' greeting."""
    service = EmailService()
    await service.send_otp_email("target@example.com", "654321")

    args, _ = mock_send.call_args
    body = args[0].get_content()
    assert "Welcome!" in body


@pytest.mark.asyncio
@patch("fastapi_ollama_rag.services.email_service.aiosmtplib.send", new_callable=AsyncMock)
async def test_send_otp_email_body_contains_expiry_notice(mock_send, configured_settings):
    """The email body should mention the 10-minute expiry."""
    service = EmailService()
    await service.send_otp_email("target@example.com", "654321")

    args, _ = mock_send.call_args
    body = args[0].get_content()
    assert "expire in 10 minutes" in body


@pytest.mark.asyncio
@patch("fastapi_ollama_rag.services.email_service.aiosmtplib.send", new_callable=AsyncMock)
async def test_send_otp_email_body_contains_otp_value(mock_send, configured_settings):
    """The email body must contain the exact OTP string passed in."""
    service = EmailService()
    await service.send_otp_email("target@example.com", "777888")

    args, _ = mock_send.call_args
    body = args[0].get_content()
    assert "777888" in body


@pytest.mark.asyncio
@patch("fastapi_ollama_rag.services.email_service.aiosmtplib.send", new_callable=AsyncMock)
async def test_send_otp_email_different_otp_produces_different_body(mock_send, configured_settings):
    """Different OTP values should produce different email bodies."""
    service = EmailService()

    await service.send_otp_email("target@example.com", "111111")
    body_1 = mock_send.call_args[0][0].get_content()

    mock_send.reset_mock()

    await service.send_otp_email("target@example.com", "222222")
    body_2 = mock_send.call_args[0][0].get_content()

    assert body_1 != body_2
    assert "111111" in body_1
    assert "222222" in body_2


@pytest.mark.asyncio
@patch("fastapi_ollama_rag.services.email_service.aiosmtplib.send", new_callable=AsyncMock)
async def test_send_otp_email_from_matches_username(mock_send, configured_settings):
    """
    L34: message["From"] = self.smtp_username.
    The From field should always match the configured username.
    """
    service = EmailService()
    await service.send_otp_email("target@example.com", "123456")

    args, _ = mock_send.call_args
    assert args[0]["From"] == service.smtp_username


@pytest.mark.asyncio
@patch("fastapi_ollama_rag.services.email_service.aiosmtplib.send", new_callable=AsyncMock)
async def test_send_otp_email_multiple_calls_fresh_message(mock_send, configured_settings):
    """
    Edge Case: Calling send_otp_email multiple times should construct a fresh
    EmailMessage each time (no stale state from previous calls).
    """
    service = EmailService()

    await service.send_otp_email("user_a@example.com", "111111")
    msg_1 = mock_send.call_args[0][0]

    await service.send_otp_email("user_b@example.com", "222222")
    msg_2 = mock_send.call_args[0][0]

    # Different message objects
    assert msg_1 is not msg_2
    assert msg_1["To"] == "user_a@example.com"
    assert msg_2["To"] == "user_b@example.com"
    assert "111111" in msg_1.get_content()
    assert "222222" in msg_2.get_content()


@pytest.mark.asyncio
@patch("fastapi_ollama_rag.services.email_service.aiosmtplib.send", new_callable=AsyncMock)
async def test_send_otp_email_subject_is_fixed(mock_send, configured_settings):
    """The subject line should always be the fixed string, regardless of input."""
    service = EmailService()
    await service.send_otp_email("anyone@example.com", "000000")

    args, _ = mock_send.call_args
    assert args[0]["Subject"] == "RAG Application Verification Code"


# ===================================================================
# send_otp_email — failure path (L55-59)
# ===================================================================


@pytest.mark.asyncio
@patch("fastapi_ollama_rag.services.email_service.aiosmtplib.send", new_callable=AsyncMock)
async def test_send_otp_email_failure_raises_runtime_error(mock_send, configured_settings):
    """Underlying SMTP errors are caught and re-raised as RuntimeError."""
    mock_send.side_effect = Exception("Connection Timeout")

    service = EmailService()

    with pytest.raises(RuntimeError, match="Could not dispatch email. Please try again later."):
        await service.send_otp_email("target@example.com", "123456")


@pytest.mark.asyncio
@patch("fastapi_ollama_rag.services.email_service.aiosmtplib.send", new_callable=AsyncMock)
async def test_send_otp_email_timeout_error_wrapped(mock_send, configured_settings):
    """
    Edge Case: TimeoutError from aiosmtplib should be wrapped as RuntimeError.
    L55: `except Exception` catches all exception types.
    """
    mock_send.side_effect = TimeoutError("SMTP timed out")

    service = EmailService()

    with pytest.raises(RuntimeError, match="Could not dispatch email"):
        await service.send_otp_email("target@example.com", "123456")


@pytest.mark.asyncio
@patch("fastapi_ollama_rag.services.email_service.aiosmtplib.send", new_callable=AsyncMock)
async def test_send_otp_email_os_error_wrapped(mock_send, configured_settings):
    """Edge Case: OSError (network error) should also be wrapped as RuntimeError."""
    mock_send.side_effect = OSError("Network unreachable")

    service = EmailService()

    with pytest.raises(RuntimeError, match="Could not dispatch email"):
        await service.send_otp_email("target@example.com", "123456")


@pytest.mark.asyncio
@patch("fastapi_ollama_rag.services.email_service.aiosmtplib.send", new_callable=AsyncMock)
async def test_send_otp_email_connection_refused_wrapped(mock_send, configured_settings):
    """Edge Case: ConnectionRefusedError should be wrapped as RuntimeError."""
    mock_send.side_effect = ConnectionRefusedError("Connection refused")

    service = EmailService()

    with pytest.raises(RuntimeError, match="Could not dispatch email"):
        await service.send_otp_email("target@example.com", "123456")


@pytest.mark.asyncio
@patch("fastapi_ollama_rag.services.email_service.aiosmtplib.send", new_callable=AsyncMock)
async def test_send_otp_email_original_exception_not_propagated(mock_send, configured_settings):
    """
    Edge Case: The original exception is swallowed — only RuntimeError is raised.
    L59: `raise RuntimeError(...)` with no `from e`, so the original is lost.
    """
    mock_send.side_effect = ValueError("Some weird SMTP error")

    service = EmailService()

    with pytest.raises(RuntimeError) as exc_info:
        await service.send_otp_email("target@example.com", "123456")

    # The raised exception should be RuntimeError, not ValueError
    assert type(exc_info.value) is RuntimeError
    assert "Could not dispatch email" in str(exc_info.value)


@pytest.mark.asyncio
@patch("fastapi_ollama_rag.services.email_service.aiosmtplib.send", new_callable=AsyncMock)
async def test_send_otp_email_failure_message_is_exact(mock_send, configured_settings):
    """The RuntimeError message should be the exact fixed string."""
    mock_send.side_effect = Exception("anything")

    service = EmailService()

    with pytest.raises(RuntimeError) as exc_info:
        await service.send_otp_email("target@example.com", "123456")

    assert str(exc_info.value) == "Could not dispatch email. Please try again later."

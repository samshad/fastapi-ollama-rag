"""
Shared test utilities.

track_test_email() registers emails created during tests so that the
session teardown in conftest.py can surgically delete ONLY test data.
"""

_test_emails: set[str] = set()


def track_test_email(email: str) -> None:
    """Register an email so teardown knows to clean it up."""
    _test_emails.add(email)


def get_tracked_emails() -> set[str]:
    """Return all tracked emails (read-only view for teardown)."""
    return _test_emails


def clear_tracked_emails() -> None:
    """Reset the registry after teardown."""
    _test_emails.clear()


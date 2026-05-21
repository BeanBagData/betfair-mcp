"""Tests for env-var-only credential loading in betfair_client.BetfairClient."""

import pytest

from betfair_client import BetfairClient, BetfairCredentialError


def test_loads_from_env_vars(monkeypatch):
    monkeypatch.setenv("BETFAIR_USERNAME", "alice")
    monkeypatch.setenv("BETFAIR_PASSWORD", "wonderland")
    monkeypatch.setenv("BETFAIR_APP_KEY", "appkey-123")

    client = BetfairClient()

    assert client.username == "alice"
    assert client.password == "wonderland"
    assert client.app_key == "appkey-123"


def test_raises_when_username_missing(monkeypatch):
    monkeypatch.delenv("BETFAIR_USERNAME", raising=False)
    monkeypatch.setenv("BETFAIR_PASSWORD", "wonderland")
    monkeypatch.setenv("BETFAIR_APP_KEY", "appkey-123")

    with pytest.raises(BetfairCredentialError) as exc_info:
        BetfairClient()

    assert "BETFAIR_USERNAME" in str(exc_info.value)


def test_raises_when_password_missing(monkeypatch):
    monkeypatch.setenv("BETFAIR_USERNAME", "alice")
    monkeypatch.delenv("BETFAIR_PASSWORD", raising=False)
    monkeypatch.setenv("BETFAIR_APP_KEY", "appkey-123")

    with pytest.raises(BetfairCredentialError) as exc_info:
        BetfairClient()

    assert "BETFAIR_PASSWORD" in str(exc_info.value)


def test_raises_when_app_key_missing(monkeypatch):
    monkeypatch.setenv("BETFAIR_USERNAME", "alice")
    monkeypatch.setenv("BETFAIR_PASSWORD", "wonderland")
    monkeypatch.delenv("BETFAIR_APP_KEY", raising=False)

    with pytest.raises(BetfairCredentialError) as exc_info:
        BetfairClient()

    assert "BETFAIR_APP_KEY" in str(exc_info.value)


def test_ignores_credentials_path_argument(monkeypatch, tmp_path):
    """Backwards-compat: callers can still pass credentials_path; it must be ignored."""
    monkeypatch.setenv("BETFAIR_USERNAME", "alice")
    monkeypatch.setenv("BETFAIR_PASSWORD", "wonderland")
    monkeypatch.setenv("BETFAIR_APP_KEY", "appkey-123")

    bogus_path = tmp_path / "does-not-exist.json"
    client = BetfairClient(credentials_path=str(bogus_path))

    assert client.username == "alice"


def test_error_message_lists_all_missing_vars(monkeypatch):
    monkeypatch.delenv("BETFAIR_USERNAME", raising=False)
    monkeypatch.delenv("BETFAIR_PASSWORD", raising=False)
    monkeypatch.delenv("BETFAIR_APP_KEY", raising=False)

    with pytest.raises(BetfairCredentialError) as exc_info:
        BetfairClient()

    message = str(exc_info.value)
    assert "BETFAIR_USERNAME" in message
    assert "BETFAIR_PASSWORD" in message
    assert "BETFAIR_APP_KEY" in message


def test_mcp_server_imports_without_credentials(monkeypatch):
    """Importing mcp_server must NOT instantiate BetfairClient at import time.

    Codex CLI launches the server before the env is fully populated; an
    import-time client construction would crash the server on startup.
    """
    import sys

    monkeypatch.delenv("BETFAIR_USERNAME", raising=False)
    monkeypatch.delenv("BETFAIR_PASSWORD", raising=False)
    monkeypatch.delenv("BETFAIR_APP_KEY", raising=False)

    # Force a cold import: remove any cached module before importing.
    monkeypatch.delitem(sys.modules, "mcp_server", raising=False)

    import mcp_server  # must not raise

    # Register the same cleanup so the test does not leak module state to
    # other tests that import mcp_server later.
    monkeypatch.delitem(sys.modules, "mcp_server", raising=False)

    assert hasattr(mcp_server, "_get_client")
    # No eager construction — _client must still be None after a cold import.
    assert mcp_server._client is None

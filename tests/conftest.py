"""Shared pytest fixtures for the Betfair MCP tests."""

import os
import sys
from pathlib import Path

# Ensure the project root is on sys.path so tests can `import betfair_client` etc.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


import pytest


@pytest.fixture(autouse=True)
def isolate_env(monkeypatch):
    """
    Default fixture: clear Betfair-related env vars before each test.
    Tests that need credentials must set them explicitly via monkeypatch.
    """
    for key in (
        "BETFAIR_USERNAME",
        "BETFAIR_PASSWORD",
        "BETFAIR_APP_KEY",
        "PAPER_MODE",
    ):
        monkeypatch.delenv(key, raising=False)


@pytest.fixture
def fake_creds(monkeypatch):
    """Set valid-shaped Betfair credentials in the environment."""
    monkeypatch.setenv("BETFAIR_USERNAME", "test-user")
    monkeypatch.setenv("BETFAIR_PASSWORD", "test-password")
    monkeypatch.setenv("BETFAIR_APP_KEY", "test-appkey")

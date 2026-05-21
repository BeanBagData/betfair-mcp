# Betfair MCP — Codex CLI bridge + targeted hardening — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Spec:** [`docs/superpowers/specs/2026-05-19-betfair-mcp-codex-bridge-design.md`](../specs/2026-05-19-betfair-mcp-codex-bridge-design.md)

**Goal:** Make the existing Betfair Agent usable from Codex CLI via MCP, fix concrete launch + safety bugs, and add tests on the money-math paths. Do not rebuild anything that works.

**Architecture:** Codex CLI launches `mcp_server.py` over stdio. The server delegates to the existing `BetfairClient`, `staking_engine`, `external_ratings`, and `market_analyser` modules (unchanged except where listed). A `PAPER_MODE` env var gates all write operations inside `betfair_client.py` so write tools never reach Betfair during iteration. Tests cover the engines in `tests/`. The `agent.py` + `main.py` Gemini path stays in the repo as a fallback.

**Tech Stack:** Python 3.10+, existing deps (`betfairlightweight`, `flumine`, `mcp`, `pandas`, `requests`, `python-dotenv`). Adds `pytest` as a dev dependency. No other new dependencies.

---

## Repo constraint — local only

This checkout is a clone of `BeanBagData/betfair-mcp` (third-party public repo, not the user's). **Never `git push`, never open a PR upstream.** Local commits are optional; the plan does not require them. If a checkpoint commit feels useful at the end of a task, do it locally — but do not push.

---

## File structure

**Created (15 files):**
- `.gitignore`
- `.env.example`
- `SECURITY.md`
- `requirements-dev.txt`
- `bin/betfair-mcp.sh`
- `tests/__init__.py`
- `tests/conftest.py`
- `tests/test_credentials.py`
- `tests/test_staking_engine.py`
- `tests/test_external_ratings.py`
- `tests/test_market_analyser.py`
- `tests/test_paper_mode.py`
- `tests/test_mcp_tools.py`
- `docs/superpowers/specs/2026-05-19-betfair-mcp-codex-bridge-design.md` (already exists)
- `docs/superpowers/plans/2026-05-19-betfair-mcp-codex-bridge.md` (this file, already exists)

**Modified (3 files):**
- `betfair_client.py` — credential loader (env-only); PAPER_MODE gating on `place_lay_bet`, `place_back_bet`, `cancel_order`.
- `mcp_server.py` — lazy client construction; 4 new tool declarations + handlers; startup PAPER_MODE log.
- `README.md` — Codex CLI section; `.env`/PAPER_MODE docs; testing section; pointer to `SECURITY.md`.

**Unchanged (everything else):** `agent.py`, `main.py`, `sub_agents.py`, `flumine_engine.py`, `market_analyser.py`, `staking_engine.py`, `external_ratings.py`, `betting_memory.py`, `shared_cache.py`, `requirements.txt`, `LICENSE`. We add tests *of* these modules but do not modify them in this work.

---

## Task ordering rationale

1. **Hygiene first** (T1) — `.gitignore` + `SECURITY.md` before any other file lands, so secrets can't leak during the rest of the work.
2. **Credential loader** (T2) — every downstream task depends on env-var-based credentials.
3. **mcp_server lazy init** (T3) — required for the loader change to actually fix the launch brittleness.
4. **Test scaffolding** (T4) — needs to exist before the test tasks.
5. **Engine tests** (T5–T7) — pure-function tests, no Betfair calls; safe to do any time.
6. **Paper mode** (T8) — gates the write tools; must land before adding `place_back_bet` to MCP.
7. **MCP read tools** (T9–T11) — independent of paper mode.
8. **MCP write tool** (T12) — depends on T8.
9. **Codex wiring** (T13) — last, exercises the whole stack.

---

### Task 1: Repository hygiene scaffolding

Adds `.gitignore`, `.env.example`, and `SECURITY.md`. No code changes. Done first so subsequent tasks can write `.env` without leak risk.

**Files:**
- Create: `.gitignore`
- Create: `.env.example`
- Create: `SECURITY.md`

- [ ] **Step 1: Create `.gitignore`**

Write the following to `/Users/nickjordan/Documents/Betfair MCP/.gitignore`:

```gitignore
# Secrets
.env
.env.local
credentials.json

# Generated state
bet_log.json
agent_memory.json
logs/
*.log

# Python
__pycache__/
*.pyc
*.pyo
.venv/
venv/
.pytest_cache/
.coverage
htmlcov/

# Editor
.vscode/
.idea/
*.swp
.DS_Store
```

- [ ] **Step 2: Create `.env.example`**

Write the following to `/Users/nickjordan/Documents/Betfair MCP/.env.example`:

```dotenv
# Betfair credentials — required.
# Paste your working values into a file named `.env` in this directory.
# .env is gitignored; never commit real values.
BETFAIR_USERNAME=me@example.com
BETFAIR_PASSWORD=change-me
BETFAIR_APP_KEY=change-me

# Paper mode (default: true). Set to false ONLY when you intend to bet real money.
# When true, place_lay_bet / place_back_bet / cancel_order return simulated responses
# and do not call Betfair.
PAPER_MODE=true
```

- [ ] **Step 3: Create `SECURITY.md`**

Write the following to `/Users/nickjordan/Documents/Betfair MCP/SECURITY.md`:

```markdown
# Security notes

## Credential exposure in upstream git history

This repository (`BeanBagData/betfair-mcp` on GitHub) committed `credentials.json` to public history across multiple commits before reverting. The blobs are still recoverable from the public packfile. Anyone who pulled the public repo while those commits were live may have those values.

This local clone is not the upstream — we cannot rewrite the public history. The mitigations applied here are:

1. `credentials.json` is now in `.gitignore` and no longer read by the client code.
2. Credentials are loaded from environment variables (`BETFAIR_USERNAME`, `BETFAIR_PASSWORD`, `BETFAIR_APP_KEY`) read from a local `.env` file.
3. The user's working Betfair credentials are different from the values that were committed upstream, so no rotation is required for *this* user. If you are reading this and the committed values *are* your real credentials: rotate them at https://developer.betfair.com immediately.

## Credential setup

1. Copy `.env.example` to `.env`.
2. Paste your real Betfair username, password, and app key into `.env`.
3. Confirm `.env` is gitignored (`git status` should not show it).
4. Never commit `.env` or `credentials.json`.

## Paper mode

`PAPER_MODE=true` (default) prevents `place_lay_bet`, `place_back_bet`, and `cancel_order` from reaching Betfair. Iterate freely with `PAPER_MODE=true`. Only set `PAPER_MODE=false` when you intend to bet real money.
```

- [ ] **Step 4: Verify files exist and have correct content**

Run:
```bash
ls -la /Users/nickjordan/Documents/Betfair\ MCP/.gitignore /Users/nickjordan/Documents/Betfair\ MCP/.env.example /Users/nickjordan/Documents/Betfair\ MCP/SECURITY.md
```

Expected: all three files listed with non-zero size.

Run:
```bash
grep -c "credentials.json" /Users/nickjordan/Documents/Betfair\ MCP/.gitignore
```

Expected: `1`.

---

### Task 2: Env-var-only credential loader in `betfair_client.py`

Replace `_load_credentials()` so it reads only env vars. Raise a clear error if any of the three required vars is missing. Drop the `credentials_path` constructor argument (callers may still pass it for backward compat, but it is ignored). Add a `BetfairCredentialError` exception class.

**Files:**
- Modify: `betfair_client.py:32-56` (BetfairClient.__init__ and _load_credentials)
- Test: `tests/test_credentials.py`

- [ ] **Step 1: Write the failing tests**

Create `/Users/nickjordan/Documents/Betfair MCP/tests/__init__.py` as an empty file (needed for pytest discovery on some setups).

Create `/Users/nickjordan/Documents/Betfair MCP/tests/test_credentials.py` with:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run:
```bash
cd /Users/nickjordan/Documents/Betfair\ MCP && pytest tests/test_credentials.py -v
```

Expected: tests fail because `BetfairCredentialError` is not defined and the current loader reads `credentials.json`, not env vars.

(If `pytest` is not installed, install it: `pip install pytest`. Task 4 formalises this in `requirements-dev.txt`.)

- [ ] **Step 3: Add `BetfairCredentialError` and rewrite `_load_credentials`**

In `/Users/nickjordan/Documents/Betfair MCP/betfair_client.py`, modify the top of the file and the `BetfairClient.__init__` / `_load_credentials` methods.

After the `import` block (after line 18 where `logger` is defined), add:

```python
class BetfairCredentialError(RuntimeError):
    """Raised when required Betfair credentials are missing from the environment."""
```

Replace the existing `BetfairClient.__init__` (lines 32-39) with:

```python
class BetfairClient:
    def __init__(self, credentials_path: Optional[str] = None):
        # credentials_path is accepted for backwards compatibility but ignored.
        # Credentials are read exclusively from environment variables.
        self.session_token: Optional[str] = None
        self.app_key: Optional[str] = None
        self.username: Optional[str] = None
        self.password: Optional[str] = None
        self._load_credentials()
```

Replace the existing `_load_credentials` method (lines 41-56) with:

```python
    def _load_credentials(self):
        """Load credentials from environment variables. Raises BetfairCredentialError if any are missing."""
        username = os.environ.get("BETFAIR_USERNAME")
        password = os.environ.get("BETFAIR_PASSWORD")
        app_key  = os.environ.get("BETFAIR_APP_KEY")

        missing = [
            name for name, value in (
                ("BETFAIR_USERNAME", username),
                ("BETFAIR_PASSWORD", password),
                ("BETFAIR_APP_KEY",  app_key),
            )
            if not value
        ]
        if missing:
            raise BetfairCredentialError(
                f"Betfair credentials missing — set {', '.join(missing)} in your .env file "
                "(see .env.example)."
            )

        self.username = username
        self.password = password
        self.app_key  = app_key
        logger.info("Betfair credentials loaded from environment for user: %s", self.username)
```

- [ ] **Step 4: Run tests to verify they pass**

Run:
```bash
cd /Users/nickjordan/Documents/Betfair\ MCP && pytest tests/test_credentials.py -v
```

Expected: all six tests pass.

- [ ] **Step 5: Sanity-check the existing entry points still load**

Run:
```bash
cd /Users/nickjordan/Documents/Betfair\ MCP && \
  BETFAIR_USERNAME=test BETFAIR_PASSWORD=test BETFAIR_APP_KEY=test \
  python -c "from betfair_client import BetfairClient; c = BetfairClient(); print('loaded:', c.username)"
```

Expected output: `loaded: test`.

---

### Task 3: Lazy client construction in `mcp_server.py`

The current `mcp_server.py:36` constructs `BetfairClient` at module import. With env-var-only credentials, importing the module from a directory without a loaded `.env` raises immediately. Move construction behind a `_get_client()` helper so importing is side-effect-free.

**Files:**
- Modify: `mcp_server.py:36` (replace module-level `client` with `_get_client()`)
- Modify: `mcp_server.py` (every reference to `client.X(...)` becomes `_get_client().X(...)`)

- [ ] **Step 1: Write a failing import test**

Append to `/Users/nickjordan/Documents/Betfair MCP/tests/test_credentials.py`:

```python


def test_mcp_server_imports_without_credentials(monkeypatch):
    """Importing mcp_server must NOT instantiate BetfairClient at import time.

    Codex CLI launches the server before the env is fully populated; an
    import-time client construction would crash the server on startup.
    """
    monkeypatch.delenv("BETFAIR_USERNAME", raising=False)
    monkeypatch.delenv("BETFAIR_PASSWORD", raising=False)
    monkeypatch.delenv("BETFAIR_APP_KEY", raising=False)

    # Force a fresh import so module-level side effects re-run.
    import importlib
    import sys
    sys.modules.pop("mcp_server", None)

    import mcp_server  # must not raise
    assert hasattr(mcp_server, "_get_client")
```

- [ ] **Step 2: Run the new test to verify it fails**

Run:
```bash
cd /Users/nickjordan/Documents/Betfair\ MCP && pytest tests/test_credentials.py::test_mcp_server_imports_without_credentials -v
```

Expected: fails with `BetfairCredentialError` on import (raised by the module-level `client = BetfairClient(...)`).

- [ ] **Step 3: Replace module-level client with `_get_client()`**

In `/Users/nickjordan/Documents/Betfair MCP/mcp_server.py`, replace line 36:

```python
# Global client instance
client = BetfairClient(credentials_path="credentials.json")
```

with:

```python
# Lazy client — constructed on first tool call, not at module import.
# This means importing mcp_server has no side effects, so Codex CLI can
# launch it from any directory without crashing on missing credentials.
_client: BetfairClient | None = None


def _get_client() -> BetfairClient:
    global _client
    if _client is None:
        _client = BetfairClient()
        logger.info("BetfairClient initialised on first tool call")
    return _client
```

- [ ] **Step 4: Replace every `client.X(...)` call with `_get_client().X(...)`**

Run this to find every reference that needs to change:

```bash
cd /Users/nickjordan/Documents/Betfair\ MCP && grep -n "\bclient\." mcp_server.py
```

For each line returned, replace `client.` with `_get_client().` — except inside string literals or comments. Inspect the diff before saving.

Example transformation:
```python
# Before:
return client.get_account_funds()

# After:
return _get_client().get_account_funds()
```

- [ ] **Step 5: Run the import test to verify it passes**

Run:
```bash
cd /Users/nickjordan/Documents/Betfair\ MCP && pytest tests/test_credentials.py::test_mcp_server_imports_without_credentials -v
```

Expected: passes.

- [ ] **Step 6: Run the full test file to ensure no regressions**

Run:
```bash
cd /Users/nickjordan/Documents/Betfair\ MCP && pytest tests/test_credentials.py -v
```

Expected: all 7 tests pass.

---

### Task 4: Test scaffolding

Add `pytest` as a dev dependency. Create `tests/conftest.py` with shared fixtures.

**Files:**
- Create: `requirements-dev.txt`
- Create: `tests/conftest.py`

- [ ] **Step 1: Create `requirements-dev.txt`**

Write to `/Users/nickjordan/Documents/Betfair MCP/requirements-dev.txt`:

```text
# Development dependencies — install with:
#   pip install -r requirements-dev.txt
# Production dependencies live in requirements.txt.

pytest>=7.4
```

- [ ] **Step 2: Create `tests/conftest.py`**

Write to `/Users/nickjordan/Documents/Betfair MCP/tests/conftest.py`:

```python
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
```

- [ ] **Step 3: Install dev deps and verify pytest discovers tests**

Run:
```bash
cd /Users/nickjordan/Documents/Betfair\ MCP && pip install -r requirements-dev.txt && pytest tests/ --collect-only
```

Expected: pytest lists all tests in `tests/test_credentials.py` without errors.

- [ ] **Step 4: Re-run credential tests to confirm conftest works**

Run:
```bash
cd /Users/nickjordan/Documents/Betfair\ MCP && pytest tests/test_credentials.py -v
```

Expected: all 7 tests pass (same as before — conftest's `isolate_env` is autouse but doesn't interfere with the credential tests which set env vars explicitly).

---

### Task 5: Tests for `staking_engine.py`

Cover Kelly invariants, fixed staking, method comparison. Pure functions; no mocks needed.

**Files:**
- Create: `tests/test_staking_engine.py`

- [ ] **Step 1: Write the tests**

Write to `/Users/nickjordan/Documents/Betfair MCP/tests/test_staking_engine.py`:

```python
"""Tests for staking_engine.py — Kelly invariants, fixed staking, comparison."""

import pytest

from staking_engine import (
    kelly_fraction,
    kelly_lay_stake,
    recommend_stake,
    compare_staking_methods,
)


# ─────────────────────────────────────────────────────────────────────────────
# kelly_fraction — raw fraction calculator
# ─────────────────────────────────────────────────────────────────────────────

def test_kelly_fraction_returns_zero_when_no_edge():
    # win_prob exactly matches market implied prob (1/lay_price) → no edge.
    # lay_price = 2.0 → implied win_prob = 0.5.
    assert kelly_fraction(win_prob=0.5, lay_price=2.0) == pytest.approx(0.0, abs=1e-6)


def test_kelly_fraction_negative_when_market_overestimates_horse():
    # win_prob < market implied → market thinks horse wins more often than we do
    # → lay is positive EV → Kelly should be POSITIVE for a lay bet.
    # (We are betting the horse loses; if we think it loses more often than the
    # market does, that's lay-side value.)
    assert kelly_fraction(win_prob=0.3, lay_price=2.0) > 0


def test_kelly_fraction_negative_when_we_think_horse_wins_more():
    # win_prob > market implied → market thinks horse loses more often than we do
    # → laying is negative EV → Kelly negative.
    assert kelly_fraction(win_prob=0.7, lay_price=2.0) < 0


def test_kelly_fraction_zero_for_invalid_lay_price():
    assert kelly_fraction(win_prob=0.5, lay_price=1.0) == 0.0
    assert kelly_fraction(win_prob=0.5, lay_price=0.5) == 0.0


# ─────────────────────────────────────────────────────────────────────────────
# kelly_lay_stake — full Kelly recommendation
# ─────────────────────────────────────────────────────────────────────────────

def test_kelly_lay_stake_returns_no_bet_for_negative_edge():
    result = kelly_lay_stake(bankroll=1000, win_prob=0.7, lay_price=2.0)
    assert result["success"] is False
    assert "negative kelly" in result.get("reason", "").lower() or \
           "negative kelly" in result.get("message", "").lower()


def test_kelly_lay_stake_rejects_win_prob_out_of_range():
    for bad_p in (0.0, 1.0, -0.1, 1.5):
        result = kelly_lay_stake(bankroll=1000, win_prob=bad_p, lay_price=2.0)
        assert result["success"] is False


def test_kelly_lay_stake_rejects_invalid_lay_price():
    result = kelly_lay_stake(bankroll=1000, win_prob=0.5, lay_price=1.0)
    assert result["success"] is False


def test_kelly_lay_stake_rejects_non_positive_bankroll():
    result = kelly_lay_stake(bankroll=0, win_prob=0.3, lay_price=2.0)
    assert result["success"] is False
    result = kelly_lay_stake(bankroll=-100, win_prob=0.3, lay_price=2.0)
    assert result["success"] is False


def test_kelly_lay_stake_liability_never_exceeds_cap():
    # max_liability_pct=0.15 means liability cannot exceed 15% of bankroll.
    result = kelly_lay_stake(
        bankroll=1000,
        win_prob=0.10,        # huge edge
        lay_price=2.0,
        partial=1.0,
        max_liability_pct=0.15,
    )
    assert result["success"] is True
    assert result["liability"] <= 1000 * 0.15 + 1e-6


# ─────────────────────────────────────────────────────────────────────────────
# Kelly variants ordering: quarter < half < full (positive edge case)
# ─────────────────────────────────────────────────────────────────────────────

def test_kelly_variants_order_by_aggression():
    bankroll, win_prob, lay_price = 1000, 0.30, 2.0  # positive lay edge

    full    = kelly_lay_stake(bankroll, win_prob, lay_price, partial=1.0)
    half    = kelly_lay_stake(bankroll, win_prob, lay_price, partial=0.5)
    quarter = kelly_lay_stake(bankroll, win_prob, lay_price, partial=0.25)

    assert all(r["success"] for r in (full, half, quarter))
    assert quarter["backer_stake"] <= half["backer_stake"] <= full["backer_stake"]


# ─────────────────────────────────────────────────────────────────────────────
# recommend_stake — top-level dispatcher
# ─────────────────────────────────────────────────────────────────────────────

def test_recommend_stake_default_is_half_kelly():
    bankroll, win_prob, lay_price = 1000, 0.30, 2.0
    default = recommend_stake(bankroll, win_prob, lay_price)  # method omitted
    half    = recommend_stake(bankroll, win_prob, lay_price, method="half_kelly")
    assert default["backer_stake"] == half["backer_stake"]


def test_recommend_stake_fixed_method_uses_fixed_liability():
    result = recommend_stake(
        bankroll=1000, win_prob=0.30, lay_price=2.0,
        method="fixed", fixed_liability=20.0,
    )
    assert result["success"] is True
    # Fixed liability of $20 at lay 2.0 → backer_stake = liability / (lay-1) = 20.
    assert result["liability"] == pytest.approx(20.0, abs=0.01)


def test_recommend_stake_unknown_method_falls_back_to_half_kelly():
    a = recommend_stake(1000, 0.30, 2.0, method="not_a_real_method")
    b = recommend_stake(1000, 0.30, 2.0, method="half_kelly")
    assert a["backer_stake"] == b["backer_stake"]


# ─────────────────────────────────────────────────────────────────────────────
# compare_staking_methods — returns multiple methods + recommendation
# ─────────────────────────────────────────────────────────────────────────────

def test_compare_staking_methods_returns_multiple_methods():
    result = compare_staking_methods(bankroll=1000, win_prob=0.30, lay_price=2.0)
    assert "comparison" in result
    assert len(result["comparison"]) >= 3  # at least kelly variants + fixed
    assert isinstance(result.get("recommendation", ""), str)
    assert result["recommendation"]  # non-empty


def test_stake_never_exceeds_bankroll_invariant():
    """For any positive-edge inputs, no recommended stake can risk > bankroll."""
    for win_prob in (0.10, 0.20, 0.30, 0.40):
        for lay_price in (1.5, 2.0, 3.0, 5.0):
            for bankroll in (100, 500, 1000, 10_000):
                result = recommend_stake(bankroll, win_prob, lay_price, method="half_kelly")
                if result.get("success"):
                    assert result["liability"] <= bankroll, (
                        f"Liability {result['liability']} > bankroll {bankroll} "
                        f"for win_prob={win_prob}, lay_price={lay_price}"
                    )
```

- [ ] **Step 2: Run the tests to verify they pass**

Run:
```bash
cd /Users/nickjordan/Documents/Betfair\ MCP && pytest tests/test_staking_engine.py -v
```

Expected: all tests pass. If any fail, the existing engine has a bug worth flagging — pause and report which assertion failed rather than weakening the test.

---

### Task 6: Tests for `external_ratings.py`

Cover `RatingsCache.model_edge` signal classification. Monkey-patch `get_model_price` to avoid hitting the live CSV URLs.

**Files:**
- Create: `tests/test_external_ratings.py`

- [ ] **Step 1: Write the tests**

Write to `/Users/nickjordan/Documents/Betfair MCP/tests/test_external_ratings.py`:

```python
"""Tests for external_ratings.RatingsCache.model_edge — signal classification."""

import pytest

from external_ratings import RatingsCache


@pytest.fixture
def cache_with_model_price(monkeypatch):
    """Return a RatingsCache whose get_model_price is patched to return a fixed value."""

    def _make(model_price):
        cache = RatingsCache()
        monkeypatch.setattr(cache, "get_model_price", lambda *args, **kwargs: model_price)
        return cache

    return _make


def test_lay_signal_when_lay_price_below_model_price(cache_with_model_price):
    cache = cache_with_model_price(model_price=3.0)
    result = cache.model_edge(market_id="1.1", selection_id=42, current_lay=2.5)
    assert result["signal"] == "LAY"
    assert result["lay_value"] is True
    assert result["lay_edge"] == pytest.approx(0.5, abs=0.001)


def test_back_signal_when_back_price_above_model_price(cache_with_model_price):
    cache = cache_with_model_price(model_price=3.0)
    result = cache.model_edge(market_id="1.1", selection_id=42, current_back=3.5)
    assert result["signal"] == "BACK"
    assert result["back_value"] is True
    assert result["back_edge"] == pytest.approx(0.5, abs=0.001)


def test_both_signal_prioritises_lay(cache_with_model_price):
    """If lay < model < back, both signals fire — agent is lay-focused → LAY wins."""
    cache = cache_with_model_price(model_price=3.0)
    result = cache.model_edge(
        market_id="1.1",
        selection_id=42,
        current_lay=2.5,
        current_back=3.5,
    )
    assert result["signal"] == "BOTH"
    assert result["lay_value"] is True
    assert result["back_value"] is True
    # Recommendation text should mention LAY priority.
    assert "lay" in result["recommendation"].lower()


def test_none_signal_when_prices_align_with_model(cache_with_model_price):
    cache = cache_with_model_price(model_price=3.0)
    result = cache.model_edge(
        market_id="1.1",
        selection_id=42,
        current_lay=3.0,
        current_back=3.0,
    )
    assert result["signal"] == "NONE"
    assert result["lay_value"] is False
    assert result["back_value"] is False


def test_no_model_price_signal(cache_with_model_price):
    cache = cache_with_model_price(model_price=None)
    result = cache.model_edge(market_id="1.1", selection_id=42, current_lay=2.5)
    assert result["signal"] == "NO_MODEL_PRICE"
    assert result["edge_pct"] is None


def test_edge_pct_is_relative_to_model_price(cache_with_model_price):
    cache = cache_with_model_price(model_price=4.0)
    result = cache.model_edge(market_id="1.1", selection_id=42, current_lay=3.0)
    # edge = 4.0 - 3.0 = 1.0; edge_pct = 1.0 / 4.0 * 100 = 25.0
    assert result["edge_pct"] == pytest.approx(25.0, abs=0.1)


def test_returns_model_name_uppercased(cache_with_model_price):
    cache = cache_with_model_price(model_price=3.0)
    result = cache.model_edge(market_id="1.1", selection_id=42, current_lay=2.5, model="kash")
    assert result["model"] == "KASH"

    result = cache.model_edge(market_id="1.1", selection_id=42, current_lay=2.5, model="iggy")
    assert result["model"] == "IGGY"
```

- [ ] **Step 2: Run the tests to verify they pass**

Run:
```bash
cd /Users/nickjordan/Documents/Betfair\ MCP && pytest tests/test_external_ratings.py -v
```

Expected: all 7 tests pass.

---

### Task 7: Tests for `market_analyser.py`

Cover `weight_of_money`, `market_spread`, and `timing_advice` classifications.

**Files:**
- Create: `tests/test_market_analyser.py`

- [ ] **Step 1: Write the tests**

Write to `/Users/nickjordan/Documents/Betfair MCP/tests/test_market_analyser.py`:

```python
"""Tests for market_analyser.py — WOM, spread, and timing classifiers."""

import pytest

from market_analyser import weight_of_money, market_spread, timing_advice


# ─────────────────────────────────────────────────────────────────────────────
# weight_of_money — classifies BACK_HEAVY / LAY_HEAVY / BALANCED / NO_LIQUIDITY
#
# atb_rungs / atl_rungs are lists of {"price": float, "size": float} dicts.
# The function takes the first `depth` of each (default 5) and computes the
# fraction of total ladder liquidity on each side.
# ─────────────────────────────────────────────────────────────────────────────

def test_wom_back_heavy_when_back_volume_dominates():
    atb = [{"price": 3.0, "size": 1000}]
    atl = [{"price": 3.1, "size": 200}]
    result = weight_of_money(atb_rungs=atb, atl_rungs=atl)
    assert result["signal"] == "BACK_HEAVY"
    assert result["back_wom"] >= 0.60


def test_wom_lay_heavy_when_lay_volume_dominates():
    atb = [{"price": 3.0, "size": 200}]
    atl = [{"price": 3.1, "size": 1000}]
    result = weight_of_money(atb_rungs=atb, atl_rungs=atl)
    assert result["signal"] == "LAY_HEAVY"
    assert result["lay_wom"] >= 0.60


def test_wom_balanced_when_volumes_close():
    atb = [{"price": 3.0, "size": 500}]
    atl = [{"price": 3.1, "size": 500}]
    result = weight_of_money(atb_rungs=atb, atl_rungs=atl)
    assert result["signal"] == "BALANCED"


def test_wom_no_liquidity_when_total_volume_zero():
    result = weight_of_money(atb_rungs=[], atl_rungs=[])
    assert result["signal"] == "NO_LIQUIDITY"


def test_wom_threshold_is_60_percent():
    """Exactly 60% is BACK_HEAVY; 59% is BALANCED."""
    # 600/(600+400) = 0.60 → BACK_HEAVY
    atb = [{"price": 3.0, "size": 600}]
    atl = [{"price": 3.1, "size": 400}]
    assert weight_of_money(atb_rungs=atb, atl_rungs=atl)["signal"] == "BACK_HEAVY"

    # 590/(590+410) = 0.59 → BALANCED
    atb = [{"price": 3.0, "size": 590}]
    atl = [{"price": 3.1, "size": 410}]
    assert weight_of_money(atb_rungs=atb, atl_rungs=atl)["signal"] == "BALANCED"


# ─────────────────────────────────────────────────────────────────────────────
# market_spread — classifies TIGHT / NORMAL / WIDE / NO_MARKET
# ─────────────────────────────────────────────────────────────────────────────

def test_spread_no_market_when_prices_missing():
    assert market_spread(None, 3.0)["assessment"] == "NO_MARKET"
    assert market_spread(3.0, None)["assessment"] == "NO_MARKET"
    assert market_spread(None, None)["assessment"] == "NO_MARKET"


def test_spread_tight_when_best_back_equals_best_lay():
    """Same price both sides → 0 ticks → TIGHT."""
    result = market_spread(best_back_price=3.0, best_lay_price=3.0)
    assert result["assessment"] == "TIGHT"
    assert result["spread_ticks"] == 0


def test_spread_tight_for_one_tick_gap():
    # At price 3.0 a single tick is 0.05 (Betfair ladder).
    result = market_spread(best_back_price=3.0, best_lay_price=3.05)
    assert result["assessment"] == "TIGHT"
    assert result["spread_ticks"] <= 2


def test_spread_returns_input_prices():
    result = market_spread(best_back_price=3.0, best_lay_price=3.1)
    assert result["best_back"] == 3.0
    assert result["best_lay"] == 3.1


# ─────────────────────────────────────────────────────────────────────────────
# timing_advice — classifies TOO_EARLY / MONITOR / OPTIMAL / LAST_CHANCE / INPLAY
# Boundaries (from the source):
#   > 1800s → TOO_EARLY
#   > 600s  → MONITOR
#   > 120s  → OPTIMAL
#   > 0s    → LAST_CHANCE
#   ≤ 0s    → INPLAY
# ─────────────────────────────────────────────────────────────────────────────

def test_timing_too_early_for_far_future():
    assert timing_advice(seconds_to_jump=3600)["window"] == "TOO_EARLY"


def test_timing_monitor_in_middle_window():
    assert timing_advice(seconds_to_jump=900)["window"] == "MONITOR"


def test_timing_optimal_in_2_to_10_minute_window():
    for sec in (130, 300, 599):
        assert timing_advice(seconds_to_jump=sec)["window"] == "OPTIMAL", \
            f"Expected OPTIMAL at {sec}s"


def test_timing_last_chance_under_2_minutes():
    for sec in (1, 60, 119):
        assert timing_advice(seconds_to_jump=sec)["window"] == "LAST_CHANCE", \
            f"Expected LAST_CHANCE at {sec}s"


def test_timing_inplay_at_zero_or_below():
    assert timing_advice(seconds_to_jump=0)["window"] == "INPLAY"
    assert timing_advice(seconds_to_jump=-60)["window"] == "INPLAY"


def test_timing_returns_minutes_helper():
    result = timing_advice(seconds_to_jump=300)
    assert result["minutes_to_jump"] == pytest.approx(5.0, abs=0.1)
```

- [ ] **Step 2: Run the tests to verify they pass**

Run:
```bash
cd /Users/nickjordan/Documents/Betfair\ MCP && pytest tests/test_market_analyser.py -v
```

Expected: all tests pass. If `test_spread_tight_for_one_tick_gap` fails, inspect Betfair's tick ladder at price 3.0 in `market_analyser.py:build_tick_ladder` — the test assumes a 0.05 tick, which is correct for the 3.0–4.0 range. If wrong, fix the test, not the engine.

---

### Task 8: PAPER_MODE flag in `betfair_client.py`

Add a module-level `PAPER_MODE` boolean read from `os.environ.get("PAPER_MODE", "true")`. Guard `place_lay_bet`, `place_back_bet`, and `cancel_order` so they return simulated responses when paper-mode is on. Append to `bet_log.json` with `paper=true`. Add a startup helper for `main.py`/`mcp_server.py` to print a banner.

**Files:**
- Modify: `betfair_client.py` (add helper + 3 method guards)
- Create: `tests/test_paper_mode.py`

- [ ] **Step 1: Write the failing tests**

Write to `/Users/nickjordan/Documents/Betfair MCP/tests/test_paper_mode.py`:

```python
"""Tests for PAPER_MODE — bet methods must not call Betfair when paper-mode is on."""

import json
import os
from unittest.mock import patch

import pytest

from betfair_client import BetfairClient, is_paper_mode


def _make_client(monkeypatch, paper_mode: str):
    monkeypatch.setenv("BETFAIR_USERNAME", "test")
    monkeypatch.setenv("BETFAIR_PASSWORD", "test")
    monkeypatch.setenv("BETFAIR_APP_KEY", "test")
    monkeypatch.setenv("PAPER_MODE", paper_mode)
    return BetfairClient()


def test_is_paper_mode_default_true(monkeypatch):
    monkeypatch.delenv("PAPER_MODE", raising=False)
    assert is_paper_mode() is True


def test_is_paper_mode_false_when_explicitly_disabled(monkeypatch):
    monkeypatch.setenv("PAPER_MODE", "false")
    assert is_paper_mode() is False


def test_is_paper_mode_accepts_common_truthy_values(monkeypatch):
    for value in ("true", "TRUE", "True", "1", "yes"):
        monkeypatch.setenv("PAPER_MODE", value)
        assert is_paper_mode() is True, f"PAPER_MODE={value!r} should be True"


def test_is_paper_mode_accepts_common_falsy_values(monkeypatch):
    for value in ("false", "FALSE", "0", "no"):
        monkeypatch.setenv("PAPER_MODE", value)
        assert is_paper_mode() is False, f"PAPER_MODE={value!r} should be False"


def test_place_lay_bet_does_not_call_betfair_in_paper_mode(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)  # so bet_log.json writes to tmp dir
    client = _make_client(monkeypatch, "true")

    with patch.object(client, "_betting_request") as mock_request:
        result = client.place_lay_bet(
            market_id="1.1",
            selection_id=42,
            lay_price=3.0,
            stake=10.0,
        )

    mock_request.assert_not_called()
    assert result["success"] is True
    assert result["paper"] is True
    assert result["bet_id"].startswith("paper-")
    assert result["lay_price"] == 3.0
    assert result["stake"] == 10.0
    assert result["liability"] == pytest.approx(20.0, abs=0.01)  # (3.0-1) * 10


def test_place_back_bet_does_not_call_betfair_in_paper_mode(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    client = _make_client(monkeypatch, "true")

    with patch.object(client, "_betting_request") as mock_request:
        result = client.place_back_bet(
            market_id="1.1",
            selection_id=42,
            back_price=3.0,
            stake=10.0,
        )

    mock_request.assert_not_called()
    assert result["success"] is True
    assert result["paper"] is True
    assert result["bet_id"].startswith("paper-")


def test_cancel_order_refuses_real_bet_id_in_paper_mode(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    client = _make_client(monkeypatch, "true")

    with patch.object(client, "_betting_request") as mock_request:
        result = client.cancel_order(market_id="1.1", bet_id="real-betfair-id-12345")

    mock_request.assert_not_called()
    assert result["success"] is False
    assert "paper" in result.get("error", "").lower() or "paper" in result.get("message", "").lower()


def test_cancel_order_removes_paper_bet_from_log(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    client = _make_client(monkeypatch, "true")

    # Place a paper bet first.
    placed = client.place_lay_bet(
        market_id="1.1", selection_id=42, lay_price=3.0, stake=10.0,
    )
    bet_id = placed["bet_id"]
    assert bet_id.startswith("paper-")

    # Now cancel it.
    with patch.object(client, "_betting_request") as mock_request:
        cancel_result = client.cancel_order(market_id="1.1", bet_id=bet_id)

    mock_request.assert_not_called()
    assert cancel_result["success"] is True
    assert cancel_result.get("paper") is True


def test_paper_bet_appended_to_bet_log(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    client = _make_client(monkeypatch, "true")

    client.place_lay_bet(
        market_id="1.1", selection_id=42, lay_price=3.0, stake=10.0,
    )

    log_path = tmp_path / "bet_log.json"
    assert log_path.exists()

    entries = json.loads(log_path.read_text())
    assert isinstance(entries, list)
    assert len(entries) == 1
    assert entries[0]["paper"] is True
    assert entries[0]["market_id"] == "1.1"
    assert entries[0]["bet_id"].startswith("paper-")


def test_place_lay_bet_calls_betfair_when_paper_mode_off(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    client = _make_client(monkeypatch, "false")

    with patch.object(client, "_betting_request") as mock_request:
        mock_request.return_value = {
            "success": True,
            "data": {
                "status": "SUCCESS",
                "instructionReports": [
                    {"betId": "real-12345", "orderStatus": "EXECUTABLE",
                     "placedDate": "2026-05-19T00:00:00Z",
                     "sizeMatched": 0, "averagePriceMatched": 0},
                ],
            },
        }
        result = client.place_lay_bet(
            market_id="1.1", selection_id=42, lay_price=3.0, stake=10.0,
        )

    mock_request.assert_called_once()
    assert result["success"] is True
    assert result.get("paper") is not True
    assert result["bet_id"] == "real-12345"
```

- [ ] **Step 2: Run the tests to verify they fail**

Run:
```bash
cd /Users/nickjordan/Documents/Betfair\ MCP && pytest tests/test_paper_mode.py -v
```

Expected: every test fails — `is_paper_mode` is undefined and the existing bet methods always call `_betting_request`.

- [ ] **Step 3: Add `is_paper_mode()` and `_paper_bet_id()` helpers + a paper bet ledger**

In `/Users/nickjordan/Documents/Betfair MCP/betfair_client.py`, add after the `BetfairCredentialError` class:

```python
import uuid


_PAPER_LOG_PATH = "bet_log.json"


def is_paper_mode() -> bool:
    """Return True iff PAPER_MODE env var is anything other than an explicit false-y value."""
    raw = os.environ.get("PAPER_MODE", "true").strip().lower()
    return raw not in ("false", "0", "no", "off", "")


def _paper_bet_id() -> str:
    return f"paper-{uuid.uuid4().hex[:12]}"


def _append_paper_bet_log(entry: dict) -> None:
    """Append a paper-bet record to bet_log.json (creates the file if needed)."""
    entries: list = []
    if os.path.exists(_PAPER_LOG_PATH):
        try:
            with open(_PAPER_LOG_PATH, "r") as f:
                loaded = json.load(f)
                if isinstance(loaded, list):
                    entries = loaded
        except (json.JSONDecodeError, OSError):
            entries = []
    entries.append(entry)
    try:
        with open(_PAPER_LOG_PATH, "w") as f:
            json.dump(entries, f, indent=2, default=str)
    except OSError as e:
        logger.warning("Could not write paper bet log: %s", e)


def _remove_paper_bet_log(bet_id: str) -> bool:
    """Remove a paper-bet record by bet_id. Returns True if found and removed."""
    if not os.path.exists(_PAPER_LOG_PATH):
        return False
    try:
        with open(_PAPER_LOG_PATH, "r") as f:
            entries = json.load(f)
    except (json.JSONDecodeError, OSError):
        return False
    if not isinstance(entries, list):
        return False
    new_entries = [e for e in entries if e.get("bet_id") != bet_id]
    if len(new_entries) == len(entries):
        return False
    try:
        with open(_PAPER_LOG_PATH, "w") as f:
            json.dump(new_entries, f, indent=2, default=str)
        return True
    except OSError:
        return False
```

- [ ] **Step 4: Guard `place_lay_bet` with paper mode**

Replace the body of `place_lay_bet` (the validation + tick rounding stays; the API call gets gated). Locate the existing method at `betfair_client.py:638` and modify the body so the first thing after `lay_price = self._round_to_nearest_tick(lay_price)` becomes:

```python
        if is_paper_mode():
            liability = round((lay_price - 1) * stake, 2)
            bet_id = _paper_bet_id()
            entry = {
                "bet_id":       bet_id,
                "paper":        True,
                "side":         "LAY",
                "market_id":    market_id,
                "selection_id": selection_id,
                "lay_price":    lay_price,
                "stake":        round(stake, 2),
                "liability":    liability,
                "strategy_ref": strategy_ref[:15],
                "placed_at":    datetime.now(timezone.utc).isoformat(),
            }
            _append_paper_bet_log(entry)
            logger.info("PAPER lay bet: market=%s sel=%s @ %s stake=%s id=%s",
                        market_id, selection_id, lay_price, stake, bet_id)
            return {
                "success":               True,
                "paper":                 True,
                "bet_id":                bet_id,
                "status":                "PAPER_PLACED",
                "placed_date":           entry["placed_at"],
                "size_matched":          0,
                "average_price_matched": 0,
                "market_id":             market_id,
                "selection_id":          selection_id,
                "lay_price":             lay_price,
                "stake":                 round(stake, 2),
                "liability":             liability,
                "potential_profit":      round(stake, 2),
                "message":               f"PAPER lay bet (id={bet_id}). No real money. Risk £{liability:.2f} to win £{stake:.2f}",
            }

        params = {
```

(The existing `params = {...}` line that builds the real Betfair request remains immediately after this insertion. Make sure the indentation aligns with the rest of the method.)

- [ ] **Step 5: Guard `place_back_bet` with paper mode**

Locate `place_back_bet` at `betfair_client.py:952`. Apply the same pattern. After any input validation but before the `_betting_request` call, insert:

```python
        if is_paper_mode():
            potential_profit = round((back_price - 1) * stake, 2)
            bet_id = _paper_bet_id()
            entry = {
                "bet_id":       bet_id,
                "paper":        True,
                "side":         "BACK",
                "market_id":    market_id,
                "selection_id": selection_id,
                "back_price":   back_price,
                "stake":        round(stake, 2),
                "potential_profit": potential_profit,
                "placed_at":    datetime.now(timezone.utc).isoformat(),
            }
            _append_paper_bet_log(entry)
            logger.info("PAPER back bet: market=%s sel=%s @ %s stake=%s id=%s",
                        market_id, selection_id, back_price, stake, bet_id)
            return {
                "success":          True,
                "paper":            True,
                "bet_id":           bet_id,
                "status":           "PAPER_PLACED",
                "placed_date":      entry["placed_at"],
                "size_matched":     0,
                "average_price_matched": 0,
                "market_id":        market_id,
                "selection_id":     selection_id,
                "back_price":       back_price,
                "stake":            round(stake, 2),
                "potential_profit": potential_profit,
                "message":          f"PAPER back bet (id={bet_id}). No real money. Risk £{stake:.2f} to win £{potential_profit:.2f}",
            }
```

If the real `place_back_bet` method takes a different parameter name (e.g. `back_price` vs `price`), align this snippet to the actual signature — read `betfair_client.py:952` first to confirm.

- [ ] **Step 6: Guard `cancel_order` with paper mode**

Locate `cancel_order` at `betfair_client.py:712`. Insert at the top of the method body, before the existing real-cancellation logic:

```python
        if is_paper_mode():
            if bet_id is None:
                return {
                    "success": False,
                    "error":   "PAPER_MODE_CANCEL_ALL_NOT_SUPPORTED",
                    "message": "Cancelling all orders is not supported in paper mode. Pass a specific paper bet_id.",
                }
            if not str(bet_id).startswith("paper-"):
                return {
                    "success": False,
                    "error":   "PAPER_MODE_REAL_BET_ID",
                    "message": f"Refusing to cancel real bet id {bet_id!r} while PAPER_MODE=true. "
                               "Set PAPER_MODE=false to cancel real bets.",
                }
            removed = _remove_paper_bet_log(bet_id)
            if removed:
                return {"success": True, "paper": True, "bet_id": bet_id,
                        "message": f"Paper bet {bet_id} cancelled."}
            return {"success": False, "paper": True, "bet_id": bet_id,
                    "error": "PAPER_BET_NOT_FOUND",
                    "message": f"No paper bet with id {bet_id} found in {_PAPER_LOG_PATH}."}
```

- [ ] **Step 7: Import `datetime` and `timezone` if not already present**

Confirm the top of `betfair_client.py` already imports `from datetime import datetime, timezone` (it does, per line 15). If not, add it.

- [ ] **Step 8: Run the paper-mode tests**

Run:
```bash
cd /Users/nickjordan/Documents/Betfair\ MCP && pytest tests/test_paper_mode.py -v
```

Expected: all tests pass.

- [ ] **Step 9: Run the full test suite to confirm no regressions**

Run:
```bash
cd /Users/nickjordan/Documents/Betfair\ MCP && pytest tests/ -v
```

Expected: all tests in `test_credentials.py`, `test_staking_engine.py`, `test_external_ratings.py`, `test_market_analyser.py`, `test_paper_mode.py` pass.

---

### Task 9: MCP tool — `list_venue_markets`

Add the tool declaration and handler in `mcp_server.py`. Mirrors the `agent.py:606` declaration. Read-only.

**Files:**
- Modify: `mcp_server.py` (add tool to `list_tools()` and handler in `call_tool()`)
- Test: `tests/test_mcp_tools.py`

- [ ] **Step 1: Confirm the underlying client method exists**

Run:
```bash
cd /Users/nickjordan/Documents/Betfair\ MCP && grep -n "def list_venue_markets" betfair_client.py
```

Expected: a single match. Note the exact signature; the MCP handler must match it.

- [ ] **Step 2: Write the failing test**

Create `/Users/nickjordan/Documents/Betfair MCP/tests/test_mcp_tools.py` with:

```python
"""Tests for the MCP tools added to bring mcp_server up to parity with agent.py."""

import json
import sys

import pytest


@pytest.fixture
def fresh_mcp_server(monkeypatch, fake_creds):
    """Import a fresh copy of mcp_server with credentials set."""
    sys.modules.pop("mcp_server", None)
    import mcp_server
    return mcp_server


@pytest.mark.asyncio
async def test_list_venue_markets_tool_registered(fresh_mcp_server):
    tools = await fresh_mcp_server.list_tools()
    names = {t.name for t in tools}
    assert "list_venue_markets" in names
```

If `pytest-asyncio` is not available, add the lighter approach using direct list-tools call. Since `list_tools` is `async`, the simplest cross-version approach is:

```python
import asyncio

def test_list_venue_markets_tool_registered(fresh_mcp_server):
    tools = asyncio.run(fresh_mcp_server.list_tools())
    names = {t.name for t in tools}
    assert "list_venue_markets" in names
```

Use the `asyncio.run` form for all `_tool_registered` tests below to avoid adding a new dependency.

- [ ] **Step 3: Run the test to verify it fails**

Run:
```bash
cd /Users/nickjordan/Documents/Betfair\ MCP && pytest tests/test_mcp_tools.py -v
```

Expected: fails because `list_venue_markets` is not registered.

- [ ] **Step 4: Add the tool declaration to `list_tools()`**

The real signature (per `betfair_client.py:887`) is:
```python
def list_venue_markets(self, venue: str, hours_ahead: int = 12, event_type: str = "horse") -> dict
```

In `/Users/nickjordan/Documents/Betfair MCP/mcp_server.py`, inside the `list_tools()` function, add a new entry to the returned list (insert near the other read tools like `search_horse`):

```python
        types.Tool(
            name="list_venue_markets",
            description=(
                "List every WIN market at a given venue for today (or within the next "
                "`hours_ahead` hours). Returns market_id, market_name, start_time, and "
                "runner list per market. Use this before launching any venue-wide strategy."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "venue": {
                        "type": "string",
                        "description": "Track name, e.g. 'Doomben', 'Flemington', 'Randwick'.",
                    },
                    "event_type": {
                        "type": "string",
                        "enum": ["horse", "thoroughbred", "greyhound", "dog"],
                        "description": "Sport: horse/thoroughbred or greyhound/dog (default 'horse').",
                        "default": "horse",
                    },
                    "hours_ahead": {
                        "type": "integer",
                        "description": "Look-ahead window in hours (default 12).",
                        "default": 12,
                    },
                },
                "required": ["venue"],
            },
        ),
```

- [ ] **Step 5: Add the handler in `call_tool()`**

Inside the `call_tool()` function in `mcp_server.py`, find the dispatch block (the chain of `if name == "...":` clauses) and add:

```python
        elif name == "list_venue_markets":
            result = _get_client().list_venue_markets(
                venue=arguments["venue"],
                hours_ahead=int(arguments.get("hours_ahead", 12)),
                event_type=arguments.get("event_type", "horse"),
            )
            return [types.TextContent(type="text", text=json.dumps(result, default=str))]
```

- [ ] **Step 6: Run the test to verify it passes**

Run:
```bash
cd /Users/nickjordan/Documents/Betfair\ MCP && pytest tests/test_mcp_tools.py::test_list_venue_markets_tool_registered -v
```

Expected: passes.

---

### Task 10: MCP tool — `get_external_ratings`

Adds Kash/Iggy model ratings access via MCP. Calls `external_ratings.get_ratings_cache()` and serialises with `to_dict()`.

**Files:**
- Modify: `mcp_server.py`
- Modify: `tests/test_mcp_tools.py`

- [ ] **Step 1: Write the failing test**

Append to `/Users/nickjordan/Documents/Betfair MCP/tests/test_mcp_tools.py`:

```python


def test_get_external_ratings_tool_registered(fresh_mcp_server):
    tools = asyncio.run(fresh_mcp_server.list_tools())
    names = {t.name for t in tools}
    assert "get_external_ratings" in names
```

(Make sure `import asyncio` is at the top of the file.)

- [ ] **Step 2: Run the test to verify it fails**

Run:
```bash
cd /Users/nickjordan/Documents/Betfair\ MCP && pytest tests/test_mcp_tools.py::test_get_external_ratings_tool_registered -v
```

Expected: fails.

- [ ] **Step 3: Add the tool declaration**

Add to `list_tools()` in `mcp_server.py`:

```python
        types.Tool(
            name="get_external_ratings",
            description=(
                "Get Betfair's free public model ratings for a venue. "
                "`model='kash'` for Australian thoroughbreds; `model='iggy'` for greyhounds. "
                "Returns model price per runner — compare against current LAY/BACK prices "
                "to find edge."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "venue": {
                        "type": "string",
                        "description": "Track name, e.g. 'Doomben'.",
                    },
                    "model": {
                        "type": "string",
                        "enum": ["kash", "iggy"],
                        "description": "Which model: kash (thoroughbreds) or iggy (greyhounds).",
                        "default": "kash",
                    },
                },
                "required": ["venue"],
            },
        ),
```

- [ ] **Step 4: Add the handler**

The right method to call is `RatingsCache.get_venue_markets(venue, model)` — returns `list[dict]` of markets at the requested venue with runner ratings. (`to_dict` is a separate summary helper; do not use it here.) See `external_ratings.py:277`.

Add to `call_tool()` in `mcp_server.py`. Import `get_ratings_cache` at the top of the file if not already present:

```python
from external_ratings import get_ratings_cache
```

Then in the dispatch block:

```python
        elif name == "get_external_ratings":
            model = arguments.get("model", "kash")
            venue = arguments["venue"]
            cache = get_ratings_cache()
            markets = cache.get_venue_markets(venue=venue, model=model)
            summary = cache.to_dict(model=model)
            payload = {
                "success":  bool(markets),
                "model":    model.upper(),
                "venue":    venue,
                "markets":  markets,
                "summary":  summary,
                "message":  (
                    f"No {model.upper()} markets found at {venue!r}. "
                    "Check spelling or try a different model."
                ) if not markets else f"{len(markets)} markets found.",
            }
            return [types.TextContent(type="text", text=json.dumps(payload, default=str))]
```

- [ ] **Step 5: Run the test**

Run:
```bash
cd /Users/nickjordan/Documents/Betfair\ MCP && pytest tests/test_mcp_tools.py -v
```

Expected: both registered-tool tests pass.

---

### Task 11: MCP tool — `get_session_report`

Surfaces orchestrator session state if one is running. Returns "no active session" otherwise.

**Files:**
- Modify: `mcp_server.py`
- Modify: `tests/test_mcp_tools.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_mcp_tools.py`:

```python


def test_get_session_report_tool_registered(fresh_mcp_server):
    tools = asyncio.run(fresh_mcp_server.list_tools())
    names = {t.name for t in tools}
    assert "get_session_report" in names


def test_get_session_report_returns_no_active_when_orchestrator_unset(
    fresh_mcp_server, monkeypatch, fake_creds
):
    """When no orchestration has run, the tool returns a structured 'no active session' response."""
    # Ensure the module-level orchestrator is None.
    monkeypatch.setattr(fresh_mcp_server, "_orchestrator", None, raising=False)

    result = asyncio.run(
        fresh_mcp_server.call_tool("get_session_report", arguments={})
    )
    # call_tool returns a list of TextContent.
    payload = json.loads(result[0].text)
    assert payload["success"] is False
    assert "no active" in payload["message"].lower()
```

- [ ] **Step 2: Run the tests to verify they fail**

Run:
```bash
cd /Users/nickjordan/Documents/Betfair\ MCP && pytest tests/test_mcp_tools.py -v
```

Expected: both new tests fail.

- [ ] **Step 3: Add a module-level `_orchestrator` slot to `mcp_server.py`**

Near the top of `mcp_server.py` (just below `_client: BetfairClient | None = None` added in Task 3), add:

```python
# Orchestrator session — populated by orchestrate_venue_session calls if/when added later.
# For now this stays None; get_session_report exposes whether one is running.
_orchestrator = None
```

- [ ] **Step 4: Add the tool declaration**

In `list_tools()`:

```python
        types.Tool(
            name="get_session_report",
            description=(
                "Return live P&L and progress for the currently-running orchestrator "
                "session, if any. Returns success=false with a clear message when no "
                "session is active."
            ),
            inputSchema={"type": "object", "properties": {}, "required": []},
        ),
```

- [ ] **Step 5: Add the handler**

The `_orchestrator` slot will always be `None` in this plan's scope — orchestration is explicitly deferred and there's no code path that populates it. The handler is a structured no-op that points Codex at the correct workaround:

```python
        elif name == "get_session_report":
            if _orchestrator is None:
                payload = {
                    "success": False,
                    "message": "No active orchestrator session. "
                               "Long-running orchestration is not yet exposed over MCP. "
                               "Start a session with `python -m sub_agents <venue>` from a "
                               "separate terminal; this tool will be expanded in a "
                               "follow-up spec once the long-running pattern is designed.",
                }
            else:
                # Reserved for a future spec that wires up an actual orchestrator.
                # Today this branch is unreachable.
                payload = {
                    "success": False,
                    "message": "Orchestrator slot is populated but get_session_report is "
                               "not implemented yet. See follow-up spec.",
                }
            return [types.TextContent(type="text", text=json.dumps(payload, default=str))]
```

- [ ] **Step 6: Run the tests**

Run:
```bash
cd /Users/nickjordan/Documents/Betfair\ MCP && pytest tests/test_mcp_tools.py -v
```

Expected: all four `tools_registered` tests + the `no_active_when_orchestrator_unset` test pass.

---

### Task 12: MCP tool — `place_back_bet`

Adds back-betting to the MCP surface. Inherits the paper-mode gate from Task 8 (the gate lives in `betfair_client.place_back_bet`, so the MCP handler just forwards arguments).

**Files:**
- Modify: `mcp_server.py`
- Modify: `tests/test_mcp_tools.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_mcp_tools.py`:

```python


def test_place_back_bet_tool_registered(fresh_mcp_server):
    tools = asyncio.run(fresh_mcp_server.list_tools())
    names = {t.name for t in tools}
    assert "place_back_bet" in names


def test_place_back_bet_via_mcp_uses_paper_mode(fresh_mcp_server, monkeypatch, fake_creds, tmp_path):
    """Placing a back bet through MCP must respect PAPER_MODE."""
    monkeypatch.setenv("PAPER_MODE", "true")
    monkeypatch.chdir(tmp_path)

    result = asyncio.run(
        fresh_mcp_server.call_tool(
            "place_back_bet",
            arguments={
                "market_id":   "1.1",
                "selection_id": 42,
                "back_price":  3.0,
                "stake":       10.0,
            },
        )
    )
    payload = json.loads(result[0].text)
    assert payload["success"] is True
    assert payload["paper"] is True
    assert payload["bet_id"].startswith("paper-")
```

- [ ] **Step 2: Run the tests to verify they fail**

Run:
```bash
cd /Users/nickjordan/Documents/Betfair\ MCP && pytest tests/test_mcp_tools.py -v
```

Expected: both new tests fail.

- [ ] **Step 3: Confirm the underlying signature**

Run:
```bash
cd /Users/nickjordan/Documents/Betfair\ MCP && sed -n '952,980p' betfair_client.py
```

Note the parameter names (`back_price` vs `price`, etc.). The MCP handler must match.

- [ ] **Step 4: Add the tool declaration**

In `list_tools()`:

```python
        types.Tool(
            name="place_back_bet",
            description=(
                "Place a BACK bet (betting the runner WILL win). "
                "When PAPER_MODE=true (default) this returns a simulated success with a "
                "paper bet_id and does NOT call Betfair. Set PAPER_MODE=false in your "
                ".env to bet real money."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "market_id":   {"type": "string"},
                    "selection_id": {"type": "integer"},
                    "back_price":  {"type": "number",
                                    "description": "The price to back at."},
                    "stake":       {"type": "number",
                                    "description": "Backer's stake. Risked amount."},
                    "strategy_ref": {"type": "string", "default": "back_agent"},
                },
                "required": ["market_id", "selection_id", "back_price", "stake"],
            },
        ),
```

- [ ] **Step 5: Add the handler**

The real signature (per `betfair_client.py:952`) is:
```python
def place_back_bet(self, market_id: str, selection_id: int, back_price: float,
                   stake: float, strategy_ref: str = "agent_back") -> dict
```

In `call_tool()`:

```python
        elif name == "place_back_bet":
            result = _get_client().place_back_bet(
                market_id=arguments["market_id"],
                selection_id=int(arguments["selection_id"]),
                back_price=float(arguments["back_price"]),
                stake=float(arguments["stake"]),
                strategy_ref=arguments.get("strategy_ref", "agent_back"),
            )
            return [types.TextContent(type="text", text=json.dumps(result, default=str))]
```

- [ ] **Step 6: Run the tests**

Run:
```bash
cd /Users/nickjordan/Documents/Betfair\ MCP && pytest tests/test_mcp_tools.py -v
```

Expected: all tests pass.

- [ ] **Step 7: Add a startup PAPER_MODE banner to `mcp_server.py`**

Near the top of `mcp_server.py`, right after `logger = logging.getLogger("betfair-mcp")` (line 33), add:

```python
from betfair_client import is_paper_mode

if is_paper_mode():
    logger.warning("PAPER_MODE=true — bet tools will return simulated responses, no real money at risk.")
else:
    logger.warning("PAPER_MODE=false — bet tools will hit LIVE Betfair. Real money at risk.")
```

(`logger.warning` is used so the message appears regardless of log level. Both branches print so the operator can never miss which mode they're in.)

- [ ] **Step 8: Add the same banner to `main.py`**

In `/Users/nickjordan/Documents/Betfair MCP/main.py`, inside `main()` just after `print_banner()` (line 152) and before `print("Initialising Gemini agent...")`, add:

```python
    from betfair_client import is_paper_mode
    if is_paper_mode():
        print("⚠️  PAPER_MODE=true — bet tools return simulated responses, no real money at risk.")
    else:
        print("🔴 PAPER_MODE=false — bet tools will hit LIVE Betfair. Real money at risk.")
```

- [ ] **Step 9: Run the full test suite**

Run:
```bash
cd /Users/nickjordan/Documents/Betfair\ MCP && pytest tests/ -v
```

Expected: every test passes.

---

### Task 13: Codex CLI launcher + README

Adds a launcher script that resolves paths correctly when Codex CLI spawns the server, and documents the Codex configuration.

**Files:**
- Create: `bin/betfair-mcp.sh`
- Modify: `README.md`

- [ ] **Step 1: Create the launcher script**

Write to `/Users/nickjordan/Documents/Betfair MCP/bin/betfair-mcp.sh`:

```bash
#!/usr/bin/env bash
# Launcher for Betfair MCP server — resolves project root relative to this script
# so Codex CLI can call it from any working directory.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

cd "$PROJECT_ROOT"

# Activate venv if present.
if [[ -f ".venv/bin/activate" ]]; then
  # shellcheck disable=SC1091
  source ".venv/bin/activate"
elif [[ -f "venv/bin/activate" ]]; then
  # shellcheck disable=SC1091
  source "venv/bin/activate"
fi

# Load .env if present.
if [[ -f ".env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source ".env"
  set +a
fi

exec python "mcp_server.py"
```

- [ ] **Step 2: Make it executable**

Run:
```bash
chmod +x /Users/nickjordan/Documents/Betfair\ MCP/bin/betfair-mcp.sh
```

- [ ] **Step 3: Smoke-test the launcher**

Run (from a directory other than the project root, to verify cwd resolution):
```bash
cd /tmp && /Users/nickjordan/Documents/Betfair\ MCP/bin/betfair-mcp.sh < /dev/null > /tmp/mcp-stderr.log 2>&1 &
SERVER_PID=$!
sleep 2
kill $SERVER_PID 2>/dev/null || true
head -20 /tmp/mcp-stderr.log
```

Expected:
- The server starts without crashing on missing credentials.
- The `PAPER_MODE` banner from Task 12 appears in stderr.
- No `ModuleNotFoundError` or `FileNotFoundError`.

(The server runs over stdio; we kill it after 2 seconds because there's no client. We're verifying it survived startup.)

- [ ] **Step 4: Update `README.md` — add Codex section**

Locate the existing `## Using with Claude Desktop (MCP)` section in `/Users/nickjordan/Documents/Betfair MCP/README.md`. Insert a new section *above* it titled `## Using with Codex CLI`:

```markdown
## Using with Codex CLI

This project's primary entry point is the Codex CLI driving `mcp_server.py` over stdio. The Gemini-based `main.py` CLI remains in the repo as a fallback but is no longer the recommended path.

### Setup

1. Copy `.env.example` to `.env` and fill in your Betfair credentials. See [SECURITY.md](SECURITY.md) for details.
2. Install dependencies: `pip install -r requirements.txt`
3. (Optional) Install dev dependencies for running tests: `pip install -r requirements-dev.txt`

### Configure Codex CLI

Add this block to your `~/.codex/config.toml`:

```toml
[mcp_servers.betfair]
command = "/absolute/path/to/betfair-mcp/bin/betfair-mcp.sh"
```

Restart Codex CLI. The Betfair tools (account balance, market search, ratings, staking calculator, etc.) will be available.

### Paper mode (default ON)

`PAPER_MODE=true` in your `.env` (the default) means `place_lay_bet`, `place_back_bet`, and `cancel_order` return simulated responses and **do not contact Betfair**. Iterate with Codex freely without risk.

When you are ready to bet real money, set `PAPER_MODE=false` and restart the MCP server. The startup log will warn you in both modes — you should always know which mode you're in.

### Running tests

```bash
pip install -r requirements-dev.txt
pytest tests/ -v
```

Tests cover credential loading, staking maths, ratings edge classification, market analysis, paper mode, and MCP tool registration. They never hit live Betfair.
```

- [ ] **Step 5: Update `README.md` — credentials section**

In the existing `### 3. Configure Credentials` section, replace the `credentials.json` instructions with:

```markdown
### 3. Configure Credentials

Copy `.env.example` to `.env` in the project folder and fill in your Betfair credentials:

```dotenv
BETFAIR_USERNAME=your_betfair_username
BETFAIR_PASSWORD=your_betfair_password
BETFAIR_APP_KEY=your_betfair_app_key
PAPER_MODE=true
```

Get your Betfair API app key at: [Betfair API Access](https://developer.betfair.com/getting-started/how-to-create-an-api-app-key/)

> **Note:** The old `credentials.json` file is no longer read by the client code. See [SECURITY.md](SECURITY.md) for context.

If you also intend to use the Gemini CLI (`python main.py`), add to `.env`:

```dotenv
GEMINI_API_KEY=your_gemini_api_key_here
```
```

- [ ] **Step 6: Update `README.md` — add testing section**

Add a new section at the bottom of the README, just before the `## Disclaimer` section:

```markdown
## Testing

```bash
pip install -r requirements-dev.txt
pytest tests/ -v
```

The test suite covers credential loading, staking maths, model edge classification, market analysis, paper-mode safety, and MCP tool registration. It does not hit live Betfair.
```

- [ ] **Step 7: Verify the README updates look right**

Run:
```bash
grep -n "PAPER_MODE\|Codex\|.env" /Users/nickjordan/Documents/Betfair\ MCP/README.md
```

Expected: multiple matches covering the new sections.

- [ ] **Step 8: Run the full test suite one final time**

Run:
```bash
cd /Users/nickjordan/Documents/Betfair\ MCP && pytest tests/ -v
```

Expected: every test passes.

---

## Final checklist

- [ ] All 13 tasks above marked complete.
- [ ] `pytest tests/ -v` shows green across the board.
- [ ] `grep -r "credentials.json" /Users/nickjordan/Documents/Betfair\ MCP/{betfair_client,mcp_server,main}.py` shows zero matches (the file is no longer read).
- [ ] Launching `bin/betfair-mcp.sh` from `/tmp` does not crash.
- [ ] `README.md` has the Codex CLI section, the new credentials section, the testing section, and a pointer to `SECURITY.md`.
- [ ] `.env.example` exists and contains placeholder-only values (no real PII).
- [ ] No `git push` has occurred; no commits to the upstream `BeanBagData/betfair-mcp` remote.

## What is intentionally not in this plan

- Deleting `agent.py` or `main.py`.
- Refactoring any of `staking_engine.py`, `external_ratings.py`, `market_analyser.py`, `sub_agents.py`, `flumine_engine.py`, `betting_memory.py`, `shared_cache.py`. Tests of these modules are in scope; modifications are not.
- Exposing `orchestrate_venue_session` over MCP — deferred. Use `python -m sub_agents <venue>` for now.
- Live integration tests against Betfair.
- Any change to dependency manager, repo layout, or framework choice.
- Rewriting upstream git history (we don't own it).

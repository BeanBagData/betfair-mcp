# Betfair MCP — Codex CLI bridge + targeted hardening

**Date:** 2026-05-19
**Status:** Approved for planning
**Author:** Pair design between Nick + Claude

## Goal

Make the existing Betfair Agent usable from the Codex CLI as the LLM-driven orchestrator, via the project's existing MCP server. Replace `agent.py` (Gemini) and `main.py` (Gemini-only CLI) as the day-to-day entry point. Fix a handful of concrete bugs and safety gaps along the way. Do not rebuild anything that already works.

## Non-goals

- **Not** rewriting the project to a new layout, framework, or programming model. The existing flat-module Python layout is fine.
- **Not** replacing `betfairlightweight` or `flumine`. Those are the Betcode/Betfair-supplied libraries; we build around them.
- **Not** deleting `agent.py` or `main.py`. They remain as Gemini-side fallbacks until the Codex setup has been used long enough to confirm nothing is missing.
- **Not** redesigning sub-agent orchestration, the Flumine streaming pipeline, the betting memory, or the in-memory cache.
- **Not** introducing new analytical capabilities, calibration, or model performance tracking. Those are explicitly future-phase work.
- **Not** rewriting the upstream `BeanBagData/betfair-mcp` git history — the upstream is not under our control. Local hygiene and credential rotation only.

## Guiding principles

1. **Only changes that fix a concrete, defensible problem.** No stylistic rewrites, framework swaps, or "while we're in here" cleanup.
2. **Treat `betfairlightweight` and `flumine` as load-bearing.** Their constraints (sync programming model, specific session semantics) are accepted as-is.
3. **Paper-mode is the default until explicitly disabled.** A bet placed during development is a bug, not a feature.
4. **Match the existing code style.** Same module layout, same MCP SDK (low-level `mcp.server.Server`), same dependency manager (`pip` + `requirements.txt`), same logging (`logging`).

## Scope — six concrete work items

### 1. Credential incident triage

**Problem:** `credentials.json` was committed to the public upstream repo (`BeanBagData/betfair-mcp`) across multiple commits. The values in the local clone include a real-looking Betfair app key and a high-entropy password. Anyone who pulled the public repo while those commits were live has the values.

**Out of our control:** Upstream history rewrite (we don't own the repo). Anyone who already pulled the values.

**In our control:**
- Add `.gitignore` covering `credentials.json`, `.env`, `bet_log.json`, `agent_memory.json`, `__pycache__/`, `*.pyc`, `.venv/`, `logs/`.
- Move secrets to environment variables read from `.env`:
  - `BETFAIR_USERNAME`
  - `BETFAIR_PASSWORD`
  - `BETFAIR_APP_KEY`
- `betfair_client.py` reads only env vars. Matching the loader shape already in use in the user's adjacent Greyhound Modelling project (`core/config.py` + `core/betfair_client.py`):
  - All three env vars required; if any is missing, raise `BetfairCredentialError` with a message naming the missing var(s).
  - **No `credentials.json` fallback.** This is a clean break. The file remains on disk only as a legacy artefact (gitignored); the loader never reads it.
- Not adding `BETFAIR_LOCALE` or `BETFAIR_CERT_PATH` in this work. The current `betfair_client.py` uses a hardcoded `identitysso.betfair.com.au` endpoint and interactive login only, so those env vars would be dead config here. They exist in the Greyhound Modelling project because that project uses `betfairlightweight`'s login wrapper directly.
- Create `.env.example` with placeholder-only values (`me@example.com`, `change-me`).
- Add `SECURITY.md` documenting:
  - The exposure (which commits, which fields).
  - Required user action: rotate the Betfair password and regenerate the API app key at https://developer.betfair.com before running the agent again.
  - That history rewrite is not possible here because the repo is a clone of someone else's public fork.

**Files changed:** `.gitignore` (new), `.env.example` (new), `SECURITY.md` (new), `betfair_client.py` (credential loader).

**User action required (outside this work):** Set `BETFAIR_USERNAME`, `BETFAIR_PASSWORD`, `BETFAIR_APP_KEY` in a local `.env` using working credentials (the user has these in the adjacent Greyhound Modelling project). The previously committed values in `credentials.json` are not used by the user's real Betfair account, so rotation is not strictly required — but the file must not be re-committed.

### 2. MCP parity (5 missing tools)

**Problem:** `agent.py` declares 20 tools to Gemini. `mcp_server.py` exposes 15. Five tools that Codex would need to reach via MCP are missing.

**Missing tools:**

| Tool | Type | Notes |
|---|---|---|
| `list_venue_markets` | read | Direct port of `agent.py:606` declaration. Calls `betfair_client.list_venue_markets`. |
| `get_external_ratings` | read | Calls `external_ratings.get_ratings_cache()`. Returns Kash (thoroughbred) or Iggy (greyhound) ratings + edge analysis vs current market price. |
| `place_back_bet` | write | Port of agent.py's declaration. **Gated behind `PAPER_MODE` per item 5.** |
| `get_session_report` | read | Surfaces orchestrator session state if one is running (returns "no active session" otherwise). |
| `orchestrate_venue_session` | long-running | **Explicitly out of scope for this work.** See below. |

**`orchestrate_venue_session` decision:**
Long-running tools over MCP are an unresolved design question. Three options were considered (split into start/status/stop, single time-boxed tool, or leave out of MCP entirely). For this work we adopt option C: **document `python -m sub_agents` as the way to launch venue sessions** and revisit MCP exposure as a follow-up spec once the read-only surface has been used in practice.

**Files changed:** `mcp_server.py` (4 new tool declarations + handlers).

**Test coverage:** `tests/test_mcp_tools.py` (new) — each new tool gets a golden-path test and a deliberately-bad-input error-path test. These tests are additive to the engine-level tests in item 4 and the paper-mode tests in item 5.

### 3. `mcp_server.py` launch brittleness

**Problem:** Two issues at `mcp_server.py:36`:
- `client = BetfairClient(credentials_path="credentials.json")` runs at module import. If credentials are missing or the file path doesn't resolve, the server crashes on startup — before Codex even calls a tool. Hard to debug from the Codex side because the failure surfaces as "MCP server failed to start" with no detail.
- `credentials_path="credentials.json"` is cwd-relative. Codex CLI launches MCP servers from its own working directory, not the betfair-mcp source directory. The relative path resolves to the wrong place.

**Fixes:**
- Defer client construction. Replace the module-level `client = ...` with a `_get_client()` function that lazily constructs and caches the client on first call. Every tool handler calls `_get_client()` instead of the module-level `client`.
- Remove the `credentials_path="credentials.json"` argument entirely. Credentials come from env vars (per item 1); the file is no longer read.
- Add a startup log line on the first `_get_client()` call confirming which credential source was used.

**Files changed:** `mcp_server.py`.

### 4. Tests for staking + edge + analysis math

**Problem:** No `tests/` directory. The money-math code (Kelly staking, edge calculation, simulation, WOM classification) has zero coverage. Bugs here lose money silently.

**Approach:** Add pytest. Cover the *existing* engines without refactoring them. If a function turns out to be hard to test, that's information — but we do not redesign the engine to make it testable in this work.

**Coverage targets:**

`tests/test_staking_engine.py`:
- Kelly returns 0 stake for negative edge.
- Kelly returns 0 stake when `win_prob` is outside (0, 1).
- Stake never exceeds bankroll for any inputs.
- `half_kelly` < `full_kelly` for the same inputs (with both > 0).
- `quarter_kelly` < `half_kelly`.
- `recommend_stake` with `method="fixed"` honours `fixed_liability` exactly.
- `compare_staking_methods` returns one entry per method and a non-empty recommendation string.
- Simulation parameters round-trip (no NaN/inf in output).

`tests/test_external_ratings.py`:
- `compute_edge` returns LAY signal when market_price < model_price by ≥ 3%.
- `compute_edge` returns BACK signal when market_price > model_price by ≥ 3%.
- `compute_edge` returns NEUTRAL when |edge| < 3%.
- CSV parser handles a row with a missing column (returns valid object, logs warning).
- Ratings cache respects TTL (mock the clock).

`tests/test_market_analyser.py`:
- WOM classification: `BACK_HEAVY` when back_volume > 60% of total, `LAY_HEAVY` when lay_volume > 60%, `BALANCED` otherwise.
- Spread classification: `TIGHT` for 0–2 ticks, `WIDE` for >8 ticks.
- Timing window: `OPTIMAL` for 120–600 seconds-to-jump; `TOO_EARLY` for >1800; `LAST_CHANCE` for <120.

**What we are not doing:**
- Not adding Hypothesis (property-based) tests. Overkill for the current surface.
- Not setting a coverage % target. Aim is to cover the money-math paths, not to chase a number.
- Not adding integration tests against live Betfair. Those need fixtures and live creds; deferred.

**Files changed:** `tests/test_staking_engine.py`, `tests/test_external_ratings.py`, `tests/test_market_analyser.py`, `tests/conftest.py` (fixtures), `tests/__init__.py`, `requirements-dev.txt` (pytest only), `README.md` (one paragraph on running tests).

### 5. Paper-mode safety flag

**Problem:** Every betting tool (`place_lay_bet`, `place_back_bet`, `cancel_order`) hits the live Betfair exchange. During iteration with Codex CLI, a mis-prompted tool call spends real money.

**Behaviour:**
- New env var `PAPER_MODE`, default `"true"` until explicitly set to `"false"`.
- When `PAPER_MODE=true`:
  - `place_lay_bet` and `place_back_bet` do **not** call Betfair. They return a structured success response with a fake `bet_id` of the form `paper-<uuid4>`, the requested price/stake, and `paper=true` in the payload.
  - Paper bets are written to `bet_log.json` with `paper=true` and a `placed_at` timestamp.
  - `cancel_order` for a paper bet_id removes the entry from the paper ledger; for a real bet_id, returns an error `"refusing to cancel real bet in paper mode"`.
  - `orchestrate_venue_session` (if used) honours the same gating.
- When `PAPER_MODE=false`:
  - All tools behave as today.
  - Startup log line prints `"PAPER_MODE=false — bets will hit live Betfair"` in red. Hard to miss.

**Where the gate lives:** Inside `betfair_client.py`'s `place_lay_bet`, `place_back_bet`, and `cancel_order` methods. Single point of control — individual tool handlers can't forget to check it. The check reads from a module-level config object populated at startup, so flipping `PAPER_MODE` mid-process requires a restart (deliberate — we don't want a runtime toggle that could be flipped accidentally).

**Files changed:** `betfair_client.py` (3 method guards + helper for fake bet IDs), `mcp_server.py` (startup log line + warning), `main.py` (startup log line + warning), `README.md` (paragraph documenting the flag), `.env.example` (`PAPER_MODE=true`).

**Test coverage:** `tests/test_paper_mode.py` — paper bets return correct shape, are logged with `paper=true`, never invoke the underlying Betfair API (mock and assert no-call).

### 6. Codex CLI wiring

**Problem:** Nothing in the repo currently tells you how to connect Codex CLI to the MCP server.

**Approach:**
- Add `bin/betfair-mcp.sh` — a thin shell launcher that:
  - Resolves the script directory.
  - Activates `.venv/bin/activate` if present.
  - Loads `.env` from the script directory.
  - `exec`s `python mcp_server.py`.
- Add `README-codex.md` (or a `## Using with Codex CLI` section in the existing `README.md`) with:
  - One-line summary of the architecture (Codex CLI → stdio MCP → mcp_server.py → Betfair).
  - The `~/.codex/config.toml` block to paste:

    ```toml
    [mcp_servers.betfair]
    command = "/absolute/path/to/betfair-mcp/bin/betfair-mcp.sh"
    ```

  - List of `.env` keys the server expects.
  - Reminder that `PAPER_MODE=true` is the default and how to flip it.
  - Pointer to `SECURITY.md` for credential rotation.

**What we will not do:**
- Will not install Codex CLI.
- Will not write to `~/.codex/config.toml` directly — that's user-level agent config and edits should be done by the user.
- Will not provide a Windows launcher in this iteration (user is on macOS).

**Files changed:** `bin/betfair-mcp.sh` (new), `README.md` (Codex section).

## Risks and open questions

| Risk | Mitigation |
|---|---|
| **Paper-mode bypass.** A future tool could forget the gate. | The gate is in `betfair_client.py`, not in tool handlers. Any future tool that needs to bet calls `client.place_*` and inherits the gate. Tests assert no-Betfair-call in paper mode. |
| **Tests don't run on user's machine.** Pytest + missing dev deps. | `requirements-dev.txt` makes the dev path explicit. `README.md` documents `pip install -r requirements-dev.txt && pytest`. |
| **`mcp_server.py` parity drift over time.** Adding tools to `agent.py` but not the MCP server. | Once Codex bridge is working, `agent.py` is deprecated. New tools land in `mcp_server.py` only. We do not delete `agent.py` in this work — that's a follow-up decision after the Codex path has been used. |
| **Credential rotation doesn't happen.** User forgets to rotate before running. | `SECURITY.md` is loud about it. `README.md` Codex section references `SECURITY.md` before the run instructions. |
| **`orchestrate_venue_session` unavailable to Codex.** | Documented workaround: `python -m sub_agents <venue>`. Follow-up spec will design proper long-running tool semantics over MCP. |

## Out of scope (explicitly named)

- Repo reorganisation, `src/` layout, `pyproject.toml`/`uv` migration.
- Async refactor of any module.
- Replacing the low-level `mcp.server.Server` API with FastMCP.
- Replacing `logging` with `structlog`.
- Pydantic models for Betfair payloads.
- Hypothesis or property-based testing.
- Live Betfair integration tests.
- Refactoring `agent.py`, `sub_agents.py`, `flumine_engine.py`, `staking_engine.py`, `external_ratings.py`, `market_analyser.py`, `betting_memory.py`, `shared_cache.py`. Tests *of* these modules are in scope; changes *to* these modules are not (except where item 5 requires touching `betfair_client.py`'s `place_*` methods).
- Deleting `agent.py` or `main.py`.
- Anything touching the upstream `BeanBagData/betfair-mcp` repo.

## Success criteria

The work is done when all of the following hold:

1. `pip install -r requirements.txt -r requirements-dev.txt && pytest` passes from a fresh clone with no live credentials.
2. `git status` after a clean checkout + `.env` setup shows no tracked credential files.
3. Codex CLI configured per `README.md` can call every MCP tool listed in this spec and get a structured response (paper-mode for write tools).
4. Setting `PAPER_MODE=false` and calling `place_lay_bet` with a recognisable test stake places a real bet on Betfair (manual verification, post-credential-rotation).
5. `SECURITY.md` exists and is referenced from `README.md`.

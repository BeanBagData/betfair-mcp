# 🏇 Betfair Agent (Codex CLI)

A sophisticated, autonomous betting agent for Australian horse and greyhound racing — driven by **Codex CLI** through MCP, on top of **Flumine** and the **Betfair API**.

The Codex CLI is the LLM brain. This repo exposes the agent's full capability — multi-agent orchestration, Flumine streaming, market analytics, Kelly staking, Kash/Iggy ratings, and long-term betting memory — as MCP tools that Codex calls.

---

## 🚀 Key Features

### 🧠 Intelligent Orchestration
- **Multi-Agent Architecture:** Decomposes "Make money at Doomben" into specific tasks handled by specialized agents:
  - **Venue Analyst:** Scans every race, calculates edge, and scores opportunities.
  - **Risk Manager:** Allocates bankroll, sets stop-losses, and enforces circuit breakers.
  - **Executor:** Launches a background stream (Flumine) to place bets at the optimal time.
  - **Reporter:** Tracks live P&L and summarizes performance.

### 📊 Advanced Market Analysis
- **External Model Integration:** Automatically checks **Kash** (thoroughbreds) and **Iggy** (greyhounds) model ratings to find value.
- **Weight of Money (WOM):** Analyses market pressure (Back vs Lay volume) to confirm moves.
- **Steam/Drift Detection:** Watches live prices to catch drifters (good lay targets) and avoid steamers.
- **Race Context:** Factors in barrier draw, distance, and track conditions into its confidence score.

### ⚡ Professional Execution
- **Streaming (Flumine):** Uses the Betfair Exchange Stream API for millisecond-latency price monitoring.
- **Smart Timing:** Places bets in the "OPTIMAL" window (2–10 mins before the jump) where liquidity is highest.
- **Kelly Staking:** Sizes bets mathematically based on edge and bankroll, with configurable risk limits.

### 🛡️ Risk Management
- **Circuit Breakers:** Daily stop-loss limits (e.g., stop if down 20%).
- **Exposure Caps:** Never risks more than X% of bankroll on a single race.
- **Live Exposure Tracking:** Checks current market liability before placing new bets.

---

## Quick Start

### 1. Prerequisites

- Python 3.10+
- A Betfair account with API access
- [Codex CLI](https://github.com/openai/codex) installed (`brew install codex` or `npm i -g @openai/codex`)

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

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

### 4. Register the MCP with Codex CLI

```toml
# ~/.codex/config.toml
[mcp_servers.betfair]
command = "/absolute/path/to/betfair-mcp/bin/betfair-mcp.sh"
```

Verify Codex sees it:

```bash
codex mcp list           # betfair should appear with status=enabled
codex mcp get betfair    # show transport=stdio + launcher path
```

### 5. Talk to it

Launch Codex CLI:

```bash
codex
```

Then prompt naturally — Codex will dispatch to the betfair MCP tools as needed.

---

## Usage Examples

In a Codex CLI session:

**Simple Commands:**
```
Lay Winx for $10
Check my balance
Show my P&L for the last 7 days
```

**Strategic Requests (full multi-agent orchestration):**
```
What races are at Doomben today?
Run a venue session at Flemington with a $500 bankroll, polling mode
Scan the next 3 races at Randwick — any good lays?
```

**Market Intelligence:**
```
What does the Kash model say about Race 4 at Sandown?
Is there any steam on the favourite in the next race?
Analyse market 1.22283838 — is it a good time to bet?
```

**Streaming (Flumine):**
```
Start a Flumine stream session at Flemington
Show me events from the active stream
Stop the stream and give me the P&L summary
```

---

## Project Structure

```
betfair-mcp/
├── mcp_server.py            # 🔧 MCP Server — the only entry point. Codex talks to this.
├── sub_agents.py            # 🕵️‍♀️ Specialised Agents (Analyst, Risk, Executor, Reporter)
├── flumine_engine.py        # ⚡ Streaming Execution Engine
├── market_analyser.py       # 📉 Quant analysis (WOM, Spread, Steam)
├── staking_engine.py        # 💰 Kelly Criterion & Simulation
├── external_ratings.py      # 📡 Kash/Iggy Model Data Fetcher
├── betfair_client.py        # 🔌 Betfair API Client (env-only credentials)
├── betting_memory.py        # 🧠 Long-term strategy learning & history
├── shared_cache.py          # ⚡ High-performance API caching (per-prefix TTL)
├── bin/betfair-mcp.sh       # 🚀 Launcher — Codex spawns this over stdio
├── tests/                   # 🧪 pytest suite (credentials, staking, ratings, paper-mode, MCP)
├── requirements.txt         # 📦 Runtime dependencies
├── requirements-dev.txt     # 📦 Dev/test dependencies
├── .env                     # 🔐 Credentials (gitignored)
├── betting_history.json     # 📊 Rolling bet record (written by BettingMemory)
└── bet_log.json             # 📝 Audit log of all bets placed
```

---

## Betting Logic & Safety

The agent is built with a **"Safety First"** philosophy:

1. **Profit Threshold**: By default, it looks for a profit ratio ≥ 1.5 (Lay price ≤ 1.67) unless the model edge is extremely strong.
2. **Model Confirmation**: It prefers lays where the **Kash/Iggy model** agrees the horse is overvalued.
3. **Liquidity Check**: It checks `Market Spread` and `Available Volume` to ensure your bet will be matched.
4. **Interactive Mode**: For large automated sessions, it will explain the plan (venues, budget, strategy) and ask for your confirmation before starting.

---

## Codex CLI integration

Codex CLI drives `mcp_server.py` over stdio. The launcher (`bin/betfair-mcp.sh`) resolves the project root relative to itself, activates a venv if present, sources `.env`, and execs `python mcp_server.py`. See the **Register the MCP with Codex CLI** step in Quick Start for the exact config block.

### Tools exposed through MCP

`mcp_server.py` exposes **30 tools** across nine concerns. Codex calls these directly over MCP; other MCP-compatible clients can use the same surface.

| Concern | Tools |
|---|---|
| **Authentication** | `betfair_login` |
| **Account** | `get_account_balance`, `get_account_statement`, `get_market_profit_and_loss` |
| **Market discovery** | `search_horse`, `list_venue_markets`, `get_external_ratings` |
| **Live market data** | `get_market_book`, `get_weight_of_money`, `get_sp_predictions`, `watch_for_steam`, `get_timing_advice` |
| **Sizing** | `calculate_stake` |
| **Bet placement & history** | `place_back_bet`, `place_lay_bet`, `cancel_order`, `get_current_orders`, `get_performance_summary` |
| **Long-term memory** | `get_strategy_insights`, `log_bet_outcome`, `get_bet_history` |
| **Multi-agent orchestration** | `orchestrate_venue_session`, `quick_venue_report`, `get_session_report`, `list_sessions`, `cancel_session` |
| **Flumine streaming** | `start_stream_session`, `get_stream_events`, `stop_stream_session` |

Reads route through `SharedCache` (per-prefix TTLs from `shared_cache.py`); writes (`place_*`, `cancel_order`) invalidate the relevant cache keys and auto-log placements to `BettingMemory` (`betting_history.json`). The orchestrator + streaming tools spawn background daemon threads inside the MCP process and expose status via `get_session_report` / `get_stream_events`. State files (`betting_history.json`, `bet_log.json`, `orders_agent.csv`, `agent_memory.json`) are local generated state and are gitignored.

### Paper mode (default ON)

`PAPER_MODE=true` in your `.env` (the default) means `place_lay_bet`, `place_back_bet`, and `cancel_order` return simulated responses and **do not contact Betfair**. Iterate with Codex freely without risk.

Flumine streaming is blocked while `PAPER_MODE=true` because Flumine places exchange orders directly through its own client. Use polling mode for paper-mode simulation; only set `PAPER_MODE=false` when you intentionally want live Flumine execution.

When you are ready to bet real money, set `PAPER_MODE=false` and restart the MCP server. The startup log will warn you in both modes — you should always know which mode you're in.

### Daily paper bets

When `PAPER_MODE=true`, the MCP server starts a background daily paper-bet scheduler. For daily automation that does not depend on the MCP server staying open, install the macOS LaunchAgent:

```bash
bin/install-paper-autobet-launchd.sh
```

The LaunchAgent runs `bin/paper-autobet-once.sh --run-due` at load and every 30 minutes. The Python runner checks the configured target time and `paper_autobet_state.json`, so it places at most one paper bet per local day and keeps retrying after missed/failed attempts.

On macOS the installer copies the runtime into `~/Library/Application Support/BetfairMCP` because LaunchAgents may be blocked from reading scripts inside `~/Documents`. It also points `.env` at shared state files in `~/Library/Application Support/BetfairMCP/state`, so the LaunchAgent and MCP tools read the same paper ledger/history.

Configure it with:

```dotenv
PAPER_AUTOBET_ENABLED=true
PAPER_AUTOBET_TIME=08:30
PAPER_AUTOBET_TZ=Australia/Melbourne
PAPER_AUTOBET_MAX_ATTEMPTS=3
PAPER_AUTOBET_RETRY_MINUTES=30
```

Check status with the `get_daily_paper_bet_status` MCP tool, or without MCP via:

```bash
bin/paper-autobet-once.sh --status
```

### Running tests

```bash
pip install -r requirements-dev.txt
pytest tests/ -v
```

Tests cover credential loading, staking maths, ratings edge classification, market analysis, paper mode, and MCP tool registration. They never hit live Betfair.

---

## Using with Claude Desktop (MCP)

This project includes a Model Context Protocol (MCP) server, allowing you to use these tools directly inside **Claude Desktop**.

1.  Add the server config to your Claude Desktop configuration file:
    *   **Windows:** `%APPDATA%\Claude\claude_desktop_config.json`
    *   **Mac:** `~/Library/Application Support/Claude/claude_desktop_config.json`

```json
{
  "mcpServers": {
    "betfair": {
      "command": "python",
      "args": ["C:/path/to/betfair-mcp/mcp_server.py"]
    }
  }
}
```
2.  Restart Claude Desktop. You will now see a 🔌 icon, and Claude can use your Betfair tools!

---

## Testing

```bash
pip install -r requirements-dev.txt
pytest tests/ -v
```

The test suite covers credential loading, staking maths, model edge classification, market analysis, paper-mode safety, and MCP tool registration. It does not hit live Betfair.

---

## Disclaimer

⚠️ **Gambling involves significant financial risk.**

This software is a research project demonstrating autonomous agent capabilities. It is not financial advice.
- **Never bet money you cannot afford to lose.**
- The software may contain bugs that could lead to unintended financial loss.
- Always monitor the agent when it is running in automated mode.

**Responsible Gambling:**
- [Betfair Responsible Gambling](https://www.betfair.com.au/hub/responsible-gambling/)
- **Gambling Helpline (Australia):** 1800 858 858
```

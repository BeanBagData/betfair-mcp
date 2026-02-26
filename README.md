# 🏇 Betfair Agent

A sophisticated, autonomous betting agent for Australian horse and greyhound racing, powered by **Gemini 2.5 Flash**, **Flumine**, and the **Betfair API**.

This agent doesn't just place bets — it acts as a full **trading desk**. It orchestrates specialized sub-agents to scan venues, analyse market liquidity, check external model ratings (Kash/Iggy), manage risk with Kelly staking, and execute trades using real-time streaming data.

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
- A Google Gemini API key

### 2. Install Dependencies


pip install -r requirements.txt
```

### 3. Configure Credentials

**Create `credentials.json`** in the project folder:
```json
{
    "username": "your_betfair_username",
    "password": "your_betfair_password",
    "app_key": "your_betfair_app_key"
}
```
> Get your Betfair API app key at: [Betfair API Access](https://developer.betfair.com/getting-started/how-to-create-an-api-app-key/)

**Create `.env`** in the project folder:
```env
GEMINI_API_KEY=your_gemini_api_key_here
```
> Get your Gemini API key at: [Google AI Studio](https://aistudio.google.com/app/apikey)

### 4. Run the Agent

```bash
python main.py
```

---

## Usage Examples

Once running, speak to the agent naturally. It understands complex instructions.

**Simple Commands:**
```
You: "Lay Winx for $10"
You: "Check my balance"
You: "Show me my P&L for the last 7 days"
```

**Strategic Requests (Orchestration):**
```
You: "What races are at Doomben today?"
You: "Make money at Flemington today. Use a $500 bankroll."
You: "Scan the next 3 races at Randwick. Any good lays?"
```

**Market Intelligence:**
```
You: "What does the Kash model say about Race 4 at Sandown?"
You: "Is there any steam on the favourite in the next race?"
You: "Analyse market 1.22283838. Is it a good time to bet?"
```

---

## Project Structure

```
betfair-mcp/
├── main.py                  # 🚀 Entry point (CLI chat interface)
├── agent.py                 # 🤖 Main Gemini Agent & Tool Definitions
├── sub_agents.py            # 🕵️‍♀️ Specialized Agents (Analyst, Risk, Executor)
├── flumine_engine.py        # ⚡ Streaming Execution Engine
├── market_analyser.py       # 📉 Quant analysis (WOM, Spread, Steam)
├── staking_engine.py        # 💰 Kelly Criterion & Simulation
├── external_ratings.py      # 📡 Kash/Iggy Model Data Fetcher
├── betfair_client.py        # 🔌 Betfair API Client
├── betting_memory.py        # 🧠 Long-term strategy learning & history
├── shared_cache.py          # ⚡ High-performance API caching
├── mcp_server.py            # 🔧 MCP Server for Claude Desktop
├── requirements.txt         # 📦 Dependencies
├── credentials.json         # 🔐 Credentials (user provided)
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

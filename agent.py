"""
Gemini 2.5 Flash Agent for Betfair Lay Betting
Coordinates Betfair API operations using Google Gemini's function calling.
The agent makes autonomous decisions about whether to place lay bets.

Memory features:
  - MemoryStore: persists discovered horses/markets/races to agent_memory.json
    so context carries across restarts and conversation resets.
"""

import json
import logging
import os
import datetime
from typing import Any, Optional

import google.generativeai as genai
from google.generativeai.types import FunctionDeclaration, Tool
from google.generativeai import caching as gemini_caching

from betfair_client import BetfairClient
from market_analyser import analyse_market, timing_advice, venue_profile
from staking_engine import (
    recommend_stake, compare_staking_methods,
    estimate_edge_from_sp, run_simulation, compare_all_simulations,
    SimParams,
)
from external_ratings import get_ratings_cache
from sub_agents import OrchestratorAgent
from betting_memory import get_betting_memory, BettingMemory
from shared_cache import SharedCache

logger = logging.getLogger("gemini-agent")

# ─────────────────────────────────────────────────────────────────────────────
# SYSTEM PROMPT
# ─────────────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are an expert Betfair betting agent specialising in Australian horse and greyhound racing.
Your objective is to generate profit through disciplined, data-driven betting using the full suite
of analysis tools available to you.

## YOUR ROLE
You help the user find horses and dogs, analyse markets, and place bets on Betfair's Australian markets.
You make decisions by combining multiple data signals: external model ratings, market intelligence,
race metadata, and staking analysis. You adapt your strategy to the user's goals and risk appetite.

## LAY BETTING EXPLAINED
In a lay bet, you bet that a runner WILL NOT WIN:
- If the runner LOSES → you collect the backer's stake (your profit)
- If the runner WINS → you pay (lay_price - 1) × stake (your liability)
- Profit Ratio = 1 / (lay_price - 1)
- At lay price 1.5: profit_ratio = 2.0 (risk $1 to win $2 — very favourable)
- At lay price 2.0: profit_ratio = 1.0 (risk $1 to win $1 — fair)
- At lay price 5.0: profit_ratio = 0.25 (risk $4 to win $1 — risky)

## BACK BETTING EXPLAINED
In a back bet, you bet that a runner WILL WIN:
- Use when: market back price > model price (external ratings signal BACK)
- Potential profit = (back_price - 1) × stake

## BETTING GUIDELINES
These are defaults — the user can override any of them:

**Lay bets:** A profit_ratio ≥ 1.5 (lay price ≤ ~1.67) is the conservative default. Higher prices
  can still be good lays if model edge and WOM agree strongly — use your judgement and explain
  the tradeoff clearly. Always calculate liability and confirm the user is comfortable with it.

**Back bets:** Only back when external model shows clear BACK edge (≥3% vs model price).

**Stake sizing:** Kelly criterion (half_kelly) is the recommended default. Never let a single
  bet's liability exceed 10% of the bankroll without explicit user approval.

**Market checks:** Before placing any bet, confirm the market is OPEN, not in-play, and has
  sufficient liquidity at the target price. Always check account balance first.

**Transparency:** Always show your full reasoning — lay price, profit_ratio, model edge,
  WOM signal, stake, liability, and why you are or aren't betting.

## ADVANCED MARKET INTELLIGENCE (NEW TOOLS)
You now have access to quantitative market analysis tools derived from professional
Betfair stream data analysis. Use them to make smarter, better-timed decisions.

### External Model Ratings (get_external_ratings)
Betfair publishes two free public model ratings refreshed daily:
- **Kash Ratings Model** (Australian thoroughbred horse racing): model="kash"
- **Iggy Joey Model** (Australian greyhound racing): model="iggy"

Both provide a **model_price** — what Betfair's own ML model thinks the runner is worth:
- **LAY signal**: LAY price < model_price → market undervalues horse → LAY has positive EV
- **BACK signal**: BACK price > model_price → market overvalues horse → BACK has positive EV
- **edge_pct**: % difference between model and market price; require ≥3% before acting

Always call `get_external_ratings` before any bet. It's a free daily-refreshed sanity check
that adds a completely independent second opinion on top of BSP predictions and WOM.

### Live Exposure & Account History (New Tools)
- **get_market_profit_and_loss**: Returns your current "if_win" and "if_lose" financial exposure for every runner in a specific market. If you are placing multiple bets in the same race, ALWAYS check this first to ensure you aren't accidentally stacking liabilities on a single outcome.
- **get_account_statement**: Returns recent wallet activity, including deposits, withdrawals, and detailed P&L breakdowns of recently settled markets. Use this if the user asks "how did I do yesterday?" or "what were my last 5 bets?".

### Multi-Race Venue Orchestration (orchestrate_venue_session)
For complex goals like "make money at Doomben today", use the full sub-agent pipeline:

  1. **VenueAnalystAgent**: scans every market at the venue, scores opportunities using
     model edge + WOM + race metadata + profit_ratio + Kelly gate
  2. **RiskManagerAgent**: allocates bankroll per race, enforces circuit breakers
     (daily loss limit = 20% of bankroll, max 10% liability per bet, max 30% deployed)
  3. **ExecutionAgent**: launches Flumine streaming OR places bets immediately
  4. **ReporterAgent**: session P&L + narrative summary

**Flumine mode** (mode="flumine"): streaming — receives every market tick in real-time.
Best for all-day automation — runs until all venue races are done, then auto-terminates.

**Polling mode** (mode="polling"): places bets immediately via API, then stops.
Use for "bet the next race at X" not "all day at X".

**CRITICAL**: Always ask the user to confirm before calling orchestrate_venue_session
with mode="flumine" — it will place real bets autonomously for hours.

### Listing Venue Markets (list_venue_markets)
Get every WIN market at a track today — with runner lists and start times.
Use this FIRST to preview what's available before launching orchestration.

### COMPLETE ORCHESTRATION WORKFLOW
For: "Make money at Doomben today" / "Bet all day at Flemington"
1. get_account_balance → confirm bankroll
2. list_venue_markets(venue) → preview races, ask user to confirm
3. get_external_ratings(venue) → load Kash/Iggy model prices
4. **Confirm with user**: "I found 8 races at {{venue}}. OK to start automated lay strategy?"
5. orchestrate_venue_session(venue, bankroll, mode="flumine")



### Weight of Money (get_weight_of_money)
Measures whether the volume of unmatched money is on the BACK or LAY side of the ladder.
- **BACK_HEAVY** (>60% money backing): Market participants expect this horse to win.
  DO NOT lay a BACK_HEAVY horse unless its profit_ratio is compelling AND you have
  a strong fundamental reason the market is wrong.
- **LAY_HEAVY** (>60% money laying): Market agrees the horse will lose. Reinforces a lay.
- **BALANCED**: Neutral signal. Rely on profit_ratio and other factors.

Also check the **spread_ticks**: TIGHT (0–2) = liquid, easy to fill. WIDE (>8) = illiquid, bet may not fill.

### Steam Detection (watch_for_steam)
Detects horses whose price is moving rapidly WHILE the market is live.
- **STEAM** (price shortening fast): Sharp money is BACKING this horse → AVOID LAYING
  (you'd be fighting smart money; the horse may win)
- **DRIFT** (price drifting longer): Market losing confidence → EXCELLENT lay candidate
  (drift horses win less often than their starting price suggests)

Key insight from analysis: reacting to steam AFTER it happens is too late — the value
is already gone. But catching a drift signal IN REAL TIME lets you lay at better odds
BEFORE the price settles. Use watch_for_steam to monitor races approaching the jump.

### Timing (get_timing_advice)
Research shows ~70% of a race's volume trades in the last 5 minutes. Sharp money
typically enters the last 2–10 minutes before the jump.
- **OPTIMAL window**: 2–10 minutes before jump — prices have formed, sharp money active
- **TOO_EARLY**: >30 min out — prices not yet reliable, spreads wide
- **LAST_CHANCE**: <2 min — act now or the market closes

### BSP Predictions (get_sp_predictions)
Betfair publishes its own pre-race BSP model estimates: sp_near (early) and sp_far (later, more accurate).
These represent where the market EXPECTS the horse to BSP at race time.

**The core lay value rule:**
  - LAY price > sp_far  → POSITIVE EV  (you're laying above expected BSP — value bet)
  - LAY price ≈ sp_far  → BREAKEVEN    (no edge; skip unless other signals agree)
  - LAY price < sp_far  → NEGATIVE EV  (you're laying below BSP — avoid)

Always call `get_sp_predictions` to get a data-driven `market_win_prob` before sizing your stake.
This replaces the need to guess win probability — use `edge_analysis.market_win_prob` directly.

### Stake Sizing (calculate_stake)
Never size stakes by feel. Always call `calculate_stake` before `place_lay_bet`.

**Kelly Criterion (recommended default: half_kelly)**
  Kelly formula for lay bets: f = (p_lose - p_win / (lay_price-1))
  - Full Kelly is mathematically optimal but can cause aggressive drawdowns
  - Half-Kelly (partial=0.5) reduces variance significantly with ~75% of the EV
  - Quarter-Kelly for uncertain edges or high lay prices

**Other methods:**
  - Fixed ($20 liability/bet): simple, good for beginners or thin-edge bets
  - Proportional A (2% of bank as liability): auto-scales with bank
  - Proportional B (3% win target): stake adjusts so you always win 3% of bank

**Hard rules:**
  - NEVER let a single bet's liability exceed 10% of bankroll
  - If Kelly signals a negative fraction → NO BET, the edge doesn't justify it
  - After any losing run, check if Kelly has automatically reduced your stake

### UPDATED FULL WORKFLOW
1. Login
2. search_horse → market_id + selection_id
3. get_timing_advice → confirm OPTIMAL window (2–10 min before jump)
4. get_sp_predictions → sp_near/sp_far + data-driven edge per runner
5. If STRONG_VALUE or MARGINAL_VALUE verdict:
   a. get_weight_of_money → confirm WOM signal + spread liquidity
   b. (Optional) watch_for_steam → confirm no counter-steam on lay target
   c. calculate_stake → Kelly-sized backer_stake + liability
6. Confirm balance covers liability
7. place_lay_bet with stake from step 5c and full reasoning

For quick user requests ("just lay X for $5"), skip steps 3–5 but flag if edge is unknown.

## MEMORY & CONTEXT
You have access to a persistent memory store. At the start of each message you will
receive a <memory> block containing recently discovered horses, markets, venues and
race times from this and previous sessions. Use this to answer follow-up questions
without asking the user to repeat details. For example:
- "what about the next race?" → use the venue from memory
- "lay it for $10" → use the horse/market/selection from memory
- "check the price again" → use the market_id from memory

## COMMUNICATION STYLE
- Be direct and data-driven in your analysis
- Always show your calculations (lay_price, profit_ratio, stake, liability)
- Explain why you are or aren't placing a bet
- Be concise but thorough in your reasoning
- If you encounter errors, explain what went wrong and what to do

## YOUR HISTORICAL PERFORMANCE
{strategy_insights}

Today's date and time: {datetime}
Remember: Your goal is PROFIT. Don't bet unless the numbers make sense.
"""


# ─────────────────────────────────────────────────────────────────────────────
# MEMORY STORE
# ─────────────────────────────────────────────────────────────────────────────

class MemoryStore:
    """
    Lightweight persistent memory for the Betfair agent.
    """

    MAX_RECENT_HORSES = 10
    MAX_RECENT_MARKETS = 5

    def __init__(self, path: str = "agent_memory.json"):
        self.path = path
        self._data = self._load()

    def _load(self) -> dict:
        if os.path.exists(self.path):
            try:
                with open(self.path, "r") as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Could not load memory file: {e}")
        return {
            "last_horse": None,
            "last_market": None,
            "recent_horses":[],
            "recent_markets": [],
            "session_notes":[],
        }

    def _save(self):
        try:
            with open(self.path, "w") as f:
                json.dump(self._data, f, indent=2, default=str)
        except Exception as e:
            logger.warning(f"Could not save memory: {e}")

    def record_horse(self, name: str, selection_id: int, market_id: str,
                     market_name: str = "", venue: str = "",
                     start_time: str = "", lay_price: float = None,
                     profit_ratio: float = None):
        entry = {
            "name": name,
            "selection_id": selection_id,
            "market_id": market_id,
            "market_name": market_name,
            "venue": venue,
            "start_time": start_time,
            "lay_price": lay_price,
            "profit_ratio": profit_ratio,
            "recorded_at": datetime.datetime.now().isoformat(),
        }
        self._data["last_horse"] = entry
        recent =[h for h in self._data["recent_horses"] if h.get("selection_id") != selection_id]
        recent.insert(0, entry)
        self._data["recent_horses"] = recent[:self.MAX_RECENT_HORSES]
        self._save()

    def record_market(self, market_id: str, market_name: str = "",
                      venue: str = "", start_time: str = ""):
        entry = {
            "market_id": market_id,
            "market_name": market_name,
            "venue": venue,
            "start_time": start_time,
            "recorded_at": datetime.datetime.now().isoformat(),
        }
        self._data["last_market"] = entry
        recent =[m for m in self._data["recent_markets"] if m.get("market_id") != market_id]
        recent.insert(0, entry)
        self._data["recent_markets"] = recent[:self.MAX_RECENT_MARKETS]
        self._save()

    def add_note(self, note: str):
        self._data["session_notes"].append({
            "note": note,
            "at": datetime.datetime.now().isoformat(),
        })
        self._data["session_notes"] = self._data["session_notes"][-20:]
        self._save()

    def clear(self):
        self._data = {
            "last_horse": None,
            "last_market": None,
            "recent_horses":[],
            "recent_markets": [],
            "session_notes":[],
        }
        self._save()

    @property
    def last_horse(self) -> Optional[dict]:
        return self._data.get("last_horse")

    @property
    def last_market(self) -> Optional[dict]:
        return self._data.get("last_market")

    @property
    def recent_horses(self) -> list:
        return self._data.get("recent_horses",[])

    @property
    def recent_markets(self) -> list:
        return self._data.get("recent_markets",[])

    def build_context_block(self) -> str:
        lines = ["<memory>"]
        if self.last_horse:
            h = self.last_horse
            lines.append(f"LAST HORSE SEARCHED:")
            lines.append(f"  Name:         {h['name']}")
            lines.append(f"  Selection ID: {h['selection_id']}")
            lines.append(f"  Market ID:    {h['market_id']}")
            lines.append(f"  Race:         {h.get('market_name', '')} @ {h.get('venue', '')}")
            lines.append(f"  Start time:   {h.get('start_time', '')}")
            if h.get("lay_price"):
                lines.append(f"  Lay price:    {h['lay_price']}  (profit_ratio: {h.get('profit_ratio', 'N/A')})")

        if self.last_market and (
            not self.last_horse or
            self.last_market["market_id"] != (self.last_horse or {}).get("market_id")
        ):
            m = self.last_market
            lines.append(f"LAST MARKET VIEWED:")
            lines.append(f"  Market ID:  {m['market_id']}")
            lines.append(f"  Race:       {m.get('market_name', '')} @ {m.get('venue', '')}")
            lines.append(f"  Start time: {m.get('start_time', '')}")

        if len(self.recent_horses) > 1:
            lines.append(f"OTHER RECENT HORSES: " + ", ".join(h["name"] for h in self.recent_horses[1:5]))
        if len(self.recent_markets) > 1:
            lines.append(f"OTHER RECENT MARKETS: " + ", ".join(f"{m['market_id']} ({m.get('market_name', '')})" for m in self.recent_markets[1:]))

        lines.append("</memory>")
        return "\n".join(lines) if len(lines) > 2 else ""


# ─────────────────────────────────────────────────────────────────────────────
# FUNCTION DECLARATIONS (Gemini Tool Definitions)
# ─────────────────────────────────────────────────────────────────────────────

TOOLS = Tool(function_declarations=[
    FunctionDeclaration(
        name="betfair_login",
        description="Login to Betfair API. Must be called before any other operations.",
        parameters={"type": "object", "properties": {}, "required": []},
    ),
    FunctionDeclaration(
        name="get_account_balance",
        description="Get current account balance, available funds, and exposure.",
        parameters={"type": "object", "properties": {}, "required":[]},
    ),
    FunctionDeclaration(
        name="get_market_profit_and_loss",
        description="Retrieve live profit/loss exposure per runner for a specific market.",
        parameters={
            "type": "object",
            "properties": {
                "market_id": {"type": "string", "description": "Betfair market ID"},
            },
            "required": ["market_id"],
        },
    ),
    FunctionDeclaration(
        name="get_account_statement",
        description="Get detailed recent account transactions (deposits, withdrawals, settled bets).",
        parameters={
            "type": "object",
            "properties": {
                "record_count": {"type": "integer", "description": "Number of records (default: 20)"},
            },
            "required":[],
        },
    ),
    FunctionDeclaration(
        name="search_horse",
        description=(
            "Search for a horse in upcoming Australian horse racing WIN markets. "
            "Returns market IDs and runner selection IDs needed for betting."
        ),
        parameters={
            "type": "object",
            "properties": {
                "horse_name": {"type": "string", "description": "Name of the horse to search for"},
                "hours_ahead": {"type": "integer", "description": "Hours ahead to search (default: 24)"},
            },
            "required":["horse_name"],
        },
    ),
    FunctionDeclaration(
        name="get_market_book",
        description=(
            "Get current lay/back prices for all runners in a market. "
            "Includes profit_ratio analysis for lay betting decisions."
        ),
        parameters={
            "type": "object",
            "properties": {
                "market_id": {"type": "string", "description": "The Betfair market ID"},
            },
            "required":["market_id"],
        },
    ),
    FunctionDeclaration(
        name="place_lay_bet",
        description=(
            "Place a lay bet. ONLY call this if profit_ratio >= 1.5 and all checks pass."
        ),
        parameters={
            "type": "object",
            "properties": {
                "market_id": {"type": "string", "description": "Betfair market ID"},
                "selection_id": {"type": "integer", "description": "Runner selection ID"},
                "lay_price": {"type": "number", "description": "Price to lay at"},
                "stake": {"type": "number", "description": "Backer's stake (your potential win)"},
                "strategy_ref": {"type": "string", "description": "Strategy tag (max 15 chars)"},
                "reason": {"type": "string", "description": "Reasoning for this bet"},
            },
            "required":["market_id", "selection_id", "lay_price", "stake", "reason"],
        },
    ),
    FunctionDeclaration(
        name="get_current_orders",
        description="Get all current open/unmatched orders.",
        parameters={
            "type": "object",
            "properties": {
                "strategy_ref": {"type": "string", "description": "Filter by strategy (optional)"},
            },
            "required":[],
        },
    ),
    FunctionDeclaration(
        name="cancel_order",
        description="Cancel an open bet order.",
        parameters={
            "type": "object",
            "properties": {
                "market_id": {"type": "string", "description": "The Betfair market ID"},
                "bet_id": {"type": "string", "description": "Specific bet ID (optional)"},
            },
            "required": ["market_id"],
        },
    ),
    FunctionDeclaration(
        name="get_performance_summary",
        description="Get settled bets and overall P&L for the last N days.",
        parameters={
            "type": "object",
            "properties": {
                "days_back": {"type": "integer", "description": "Number of days to look back (default: 7)"},
            },
            "required":[],
        },
    ),
    FunctionDeclaration(
        name="get_weight_of_money",
        description=(
            "Fetch the full price ladder and compute weight-of-money (WOM), tick spread, "
            "VWAP, and a composite lay recommendation score for every runner in a market. "
            "Always call this before placing a bet for the best market read."
        ),
        parameters={
            "type": "object",
            "properties": {
                "market_id": {"type": "string", "description": "Betfair market ID"},
                "seconds_to_jump": {"type": "number", "description": "Seconds until jump"},
                "venue": {"type": "string", "description": "Venue name for volume profile"},
            },
            "required": ["market_id"],
        },
    ),
    FunctionDeclaration(
        name="watch_for_steam",
        description=(
            "Monitor a market in real time for steam (horse firming) or drift (horse drifting). "
            "Blocks for polls × interval_seconds before returning. Default: 12 × 5s = 60s."
        ),
        parameters={
            "type": "object",
            "properties": {
                "market_id": {"type": "string", "description": "Betfair market ID to watch"},
                "polls": {"type": "integer", "description": "Price snapshots to take (default: 12)"},
                "interval_seconds": {"type": "number", "description": "Seconds between polls (default: 5)"},
            },
            "required": ["market_id"],
        },
    ),
    FunctionDeclaration(
        name="get_timing_advice",
        description=(
            "Advise on optimal bet placement timing based on seconds to race start. "
            "Returns TOO_EARLY / MONITOR / OPTIMAL / LAST_CHANCE / INPLAY window."
        ),
        parameters={
            "type": "object",
            "properties": {
                "seconds_to_jump": {"type": "number", "description": "Seconds until scheduled race start"},
                "venue": {"type": "string", "description": "Venue name (e.g. 'Flemington')"},
            },
            "required":["seconds_to_jump"],
        },
    ),
    FunctionDeclaration(
        name="get_sp_predictions",
        description=(
            "Fetch Betfair's own pre-race BSP near/far model estimates for all runners. "
            "Returns market_win_prob per runner for use in calculate_stake."
        ),
        parameters={
            "type": "object",
            "properties": {
                "market_id": {"type": "string", "description": "Betfair market ID"},
            },
            "required": ["market_id"],
        },
    ),
    FunctionDeclaration(
        name="calculate_stake",
        description=(
            "Calculate the optimal lay stake using Kelly criterion (recommended: half_kelly) "
            "or other staking methods. Returns backer_stake, liability, EV, and edge."
        ),
        parameters={
            "type": "object",
            "properties": {
                "bankroll":         {"type": "number", "description": "Available account balance"},
                "win_prob":         {"type": "number", "description": "Estimated horse win probability"},
                "lay_price":        {"type": "number", "description": "Lay price to bet at"},
                "method":           {"type": "string", "description": "half_kelly | kelly | fixed | proportional_a"},
                "fixed_liability":  {"type": "number", "description": "Dollar liability for fixed method (default: $20)"},
                "run_simulation":   {"type": "boolean", "description": "Run Monte Carlo comparison"},
            },
            "required":["bankroll", "win_prob", "lay_price"],
        },
    ),
    FunctionDeclaration(
        name="get_external_ratings",
        description=(
            "Download Betfair's free external model ratings (Kash or Iggy) for a venue or specific market. "
            "Always call before place_lay_bet or place_back_bet as an independent second opinion."
        ),
        parameters={
            "type": "object",
            "properties": {
                "venue":        {"type": "string", "description": "Venue/track name"},
                "market_id":    {"type": "string", "description": "Specific Betfair market ID"},
                "selection_id": {"type": "string", "description": "Specific selection ID to get model edge for"},
                "current_lay":  {"type": "number", "description": "Live best lay price for the runner"},
                "current_back": {"type": "number", "description": "Live best back price for the runner"},
                "model":        {"type": "string", "description": "kash | iggy"},
                "min_edge_pct": {"type": "number", "description": "Minimum % edge to highlight (default: 3.0)"},
            },
            "required":[],
        },
    ),
    FunctionDeclaration(
        name="list_venue_markets",
        description=(
            "List all upcoming WIN markets at a specific venue today."
        ),
        parameters={
            "type": "object",
            "properties": {
                "venue":       {"type": "string", "description": "Track/venue name"},
                "hours_ahead": {"type": "integer", "description": "Hours ahead to search"},
                "event_type":  {"type": "string", "description": "horse | greyhound"},
            },
            "required": ["venue"],
        },
    ),
    FunctionDeclaration(
        name="orchestrate_venue_session",
        description=(
            "Deploy the full multi-agent orchestration pipeline for all-day betting at a venue. "
            "Flumine mode = streaming, runs all day until all races close."
        ),
        parameters={
            "type": "object",
            "properties": {
                "venue":             {"type": "string", "description": "Track name"},
                "bankroll":          {"type": "number", "description": "Account balance to use"},
                "mode":              {"type": "string", "description": "flumine | polling"},
                "min_profit_ratio":  {"type": "number", "description": "Minimum lay profit_ratio gate (default: 1.5)"},
                "loss_limit_pct":    {"type": "number", "description": "Stop if daily loss exceeds X% of bankroll"},
                "min_edge_pct":      {"type": "number", "description": "Minimum % edge vs model price"},
            },
            "required":["venue", "bankroll"],
        },
    ),
    FunctionDeclaration(
        name="get_session_report",
        description="Get the current P&L and status report for a running venue session.",
        parameters={
            "type": "object",
            "properties": {
                "venue": {"type": "string", "description": "Venue name the session is running for"},
            },
            "required":[],
        },
    ),
    FunctionDeclaration(
        name="place_back_bet",
        description="Place a BACK bet on a runner. Use when: market back price > model price.",
        parameters={
            "type": "object",
            "properties": {
                "market_id":    {"type": "string",  "description": "Betfair market ID"},
                "selection_id": {"type": "integer", "description": "Runner selection ID"},
                "back_price":   {"type": "number",  "description": "Price to back at"},
                "stake":        {"type": "number",  "description": "Stake amount ($)"},
                "strategy_ref": {"type": "string",  "description": "Strategy tag (max 15 chars)"},
                "reason":       {"type": "string",  "description": "Reasoning for this back bet"},
            },
            "required":["market_id", "selection_id", "back_price", "stake", "reason"],
        },
    ),
])

def _create_or_refresh_cache(system_prompt: str,
                              tools: Optional[list] = None,
                              ttl_minutes: int = 60) -> Optional[Any]:
    try:
        cache = gemini_caching.CachedContent.create(
            model="models/gemini-2.5-flash",
            display_name="betfair_agent_system_prompt",
            system_instruction=system_prompt,
            tools=tools,
            ttl=datetime.timedelta(minutes=ttl_minutes),
        )
        logger.info(f"Gemini cache created: {cache.name}  (TTL: {ttl_minutes}m)")
        return cache
    except Exception as e:
        logger.warning(f"Gemini context caching unavailable: {e}. ")
        return None

class BetfairGeminiAgent:
    def __init__(self, api_key: str, credentials_path: str = "credentials.json",
                 memory_path: str = "agent_memory.json",
                 enable_cache: bool = False):  # <--- Default cache turned off
        genai.configure(api_key=api_key)

        self.memory = MemoryStore(path=memory_path)

        self.betting_memory = get_betting_memory()
        self.cache = SharedCache.instance()

        try:
            synced = self.betting_memory.sync_outcomes()
            if synced:
                logger.info(f"BettingMemory: synced {synced} new outcomes from settlement files")
        except Exception as e:
            logger.warning(f"BettingMemory sync error (non-fatal): {e}")

        strategy_insights = self.betting_memory.get_strategy_insights()
        if not strategy_insights:
            strategy_insights = (
                "No historical bet data yet — start betting to build your performance profile.\n"
                "Once you have 5+ settled bets, I will surface insights about which price ranges,\n"
                "venues, WOM signals, and timing windows are most profitable for you."
            )

        formatted_prompt = SYSTEM_PROMPT.format(
            datetime=datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S AEST"),
            strategy_insights=strategy_insights,
        )

        self._cache = None
        if enable_cache:
            self._cache = _create_or_refresh_cache(formatted_prompt, tools=[TOOLS])

        model_name = "gemini-2.5-flash"
        
        if self._cache:
            self.model = genai.GenerativeModel.from_cached_content(
                cached_content=self._cache
            )
            if hasattr(self.model, '_prepare_tools'):
                self.model._tools = self.model._prepare_tools([TOOLS])
            print("  💾 Gemini context cache active — system prompt cached server-side")
        else:
            self.model = genai.GenerativeModel(
                model_name=model_name,
                tools=[TOOLS],
                system_instruction=formatted_prompt,
            )

        self.chat = self.model.start_chat(history=[])
        self.betfair = BetfairClient(credentials_path=credentials_path)
        self._logged_in = False
        self._ratings = get_ratings_cache()
        self._orchestrator = None  
        logger.info(f"Gemini agent initialised with model: {model_name} (Cache enabled: {enable_cache})")

    def _update_memory_from_result(self, tool_name: str, tool_args: dict, result: dict):
        if not result.get("success"):
            return
        try:
            if tool_name == "search_horse":
                markets = result.get("markets",[])
                horse_name = tool_args.get("horse_name", "")
                for market in markets:
                    for runner in market.get("matching_runners",[]):
                        self.memory.record_horse(
                            name=runner.get("runner_name", horse_name),
                            selection_id=runner["selection_id"],
                            market_id=market["market_id"],
                            market_name=market.get("market_name", ""),
                            venue=market.get("venue", ""),
                            start_time=str(market.get("start_time", "")),
                        )
                    self.memory.record_market(
                        market_id=market["market_id"],
                        market_name=market.get("market_name", ""),
                        venue=market.get("venue", ""),
                        start_time=str(market.get("start_time", "")),
                    )
            elif tool_name == "get_market_book":
                market_id = tool_args.get("market_id", "")
                for runner in result.get("runners",[]):
                    lay = runner.get("lay_analysis") or {}
                    for h in self.memory.recent_horses:
                        if (h.get("market_id") == market_id and
                                h.get("selection_id") == runner.get("selection_id")):
                            self.memory.record_horse(
                                name=h["name"],
                                selection_id=h["selection_id"],
                                market_id=market_id,
                                market_name=h.get("market_name", ""),
                                venue=h.get("venue", ""),
                                start_time=h.get("start_time", ""),
                                lay_price=lay.get("lay_price"),
                                profit_ratio=lay.get("profit_ratio"),
                            )
                            break
            elif tool_name == "place_lay_bet":
                bet_id = result.get("bet_id", "?")
                horse = self.memory.last_horse
                name = horse["name"] if horse else "Unknown"
                self.memory.add_note(
                    f"Bet placed on {name}: bet_id={bet_id}, "
                    f"market={tool_args.get('market_id')}, "
                    f"price={tool_args.get('lay_price')}, "
                    f"stake={tool_args.get('stake')}"
                )
        except Exception as e:
            logger.warning(f"Memory update error [{tool_name}]: {e}")

    def _execute_tool(self, tool_name: str, tool_args: dict) -> Any:
        logger.info(f"Executing tool: {tool_name}({tool_args})")
        try:
            if tool_name == "betfair_login":
                result = self.betfair.login()
                if result.get("success"):
                    self._logged_in = True
                return result
            elif tool_name == "get_account_balance":
                return self.betfair.get_account_funds()
            elif tool_name == "get_market_profit_and_loss":
                return self.betfair.get_market_profit_and_loss(tool_args["market_id"])
            elif tool_name == "get_account_statement":
                return self.betfair.get_account_statement(int(tool_args.get("record_count", 20)))
            elif tool_name == "search_horse":
                return self.betfair.search_horse_racing_markets(
                    tool_args["horse_name"],
                    tool_args.get("hours_ahead", 24),
                )
            elif tool_name == "get_market_book":
                market_id = tool_args["market_id"]
                return self.cache.market_book(
                    market_id,
                    fetch_fn=lambda: self.betfair.get_market_book(market_id),
                )
            elif tool_name == "place_lay_bet":
                reason = tool_args.pop("reason", "Agent decision")
                logger.info(f"🎯 PLACING BET - Reason: {reason}")
                result = self.betfair.place_lay_bet(
                    market_id=tool_args["market_id"],
                    selection_id=int(tool_args["selection_id"]),
                    lay_price=float(tool_args["lay_price"]),
                    stake=float(tool_args["stake"]),
                    strategy_ref=tool_args.get("strategy_ref", "lay_agent"),
                )
                result["bet_reason"] = reason
                self._log_bet(result, context=self._turn_context)
                return result
            elif tool_name == "get_current_orders":
                return self.betfair.get_current_orders(tool_args.get("strategy_ref"))
            elif tool_name == "cancel_order":
                return self.betfair.cancel_order(
                    market_id=tool_args["market_id"],
                    bet_id=tool_args.get("bet_id"),
                )
            elif tool_name == "get_performance_summary":
                return self.betfair.get_cleared_orders(tool_args.get("days_back", 7))
            elif tool_name == "get_weight_of_money":
                market_id = tool_args["market_id"]
                book = self.cache.market_book(
                    market_id,
                    fetch_fn=lambda: self.betfair.get_market_book(market_id),
                )
                depth_book = self.cache.get_or_fetch(
                    key=f"market_depth:{market_id}",
                    fetch_fn=lambda: self.betfair.get_market_depth(market_id, depth=10),
                    ttl=10,
                )
                if not book.get("success"):
                    return book
                return analyse_market(
                    market_book=book,
                    full_depth_book=depth_book if depth_book.get("success") else None,
                    seconds_to_jump=tool_args.get("seconds_to_jump"),
                    venue=tool_args.get("venue", ""),
                )
            elif tool_name == "watch_for_steam":
                return self.betfair.poll_market_for_steam(
                    market_id=tool_args["market_id"],
                    polls=int(tool_args.get("polls", 12)),
                    interval_seconds=float(tool_args.get("interval_seconds", 5)),
                )
            elif tool_name == "get_timing_advice":
                return {
                    "success": True,
                    "timing": timing_advice(
                        seconds_to_jump=float(tool_args["seconds_to_jump"]),
                        venue=tool_args.get("venue", ""),
                    ),
                    "venue_profile": venue_profile(tool_args["venue"]) if tool_args.get("venue") else None,
                }
            elif tool_name == "get_sp_predictions":
                market_id = tool_args["market_id"]
                return self.cache.bsp_predictions(
                    market_id,
                    fetch_fn=lambda: self.betfair.get_sp_predictions(market_id),
                )
            elif tool_name == "calculate_stake":
                bankroll   = float(tool_args["bankroll"])
                win_prob   = float(tool_args["win_prob"])
                lay_price  = float(tool_args["lay_price"])
                method     = tool_args.get("method", "half_kelly")
                fixed_liab = float(tool_args.get("fixed_liability", 20.0))
                do_sim     = bool(tool_args.get("run_simulation", False))

                stake_result = recommend_stake(
                    bankroll=bankroll,
                    win_prob=win_prob,
                    lay_price=lay_price,
                    method=method,
                    fixed_liability=fixed_liab,
                )
                comparison = compare_staking_methods(bankroll, win_prob, lay_price, fixed_liability=fixed_liab)
                result = {
                    "success":        stake_result.get("success", False),
                    "stake":          stake_result,
                    "comparison":     comparison.get("comparison",[]),
                    "recommendation": comparison.get("recommendation", ""),
                }
                if do_sim:
                    sim_params = SimParams(
                        bankroll=bankroll,
                        min_odds=max(lay_price - 1.0, 1.3),
                        max_odds=min(lay_price + 1.5, 8.0),
                        edge=max(0.01, 1.0 / lay_price - win_prob),
                    )
                    result["simulation"] = compare_all_simulations(sim_params, n_sims=500)
                return result
            elif tool_name == "get_external_ratings":
                venue        = tool_args.get("venue", "")
                market_id    = tool_args.get("market_id", "")
                selection_id = tool_args.get("selection_id", "")
                current_lay  = tool_args.get("current_lay")
                current_back = tool_args.get("current_back")
                model        = tool_args.get("model", "kash")

                self._ratings = get_ratings_cache() 
                if venue:
                    venue_markets = self._ratings.get_venue_markets(venue, model=model)
                    return {
                        "success":       True,
                        "model":         model.upper(),
                        "venue":         venue,
                        "market_count":  len(venue_markets),
                        "total_runners": sum(len(m["runners"]) for m in venue_markets),
                        "markets":       venue_markets,
                        "ratings_status": self._ratings.to_dict(model),
                    }
                elif market_id and selection_id:
                    edge = self._ratings.model_edge(
                        market_id, str(selection_id),
                        current_lay=float(current_lay)   if current_lay  else None,
                        current_back=float(current_back) if current_back else None,
                        model=model,
                    )
                    return {"success": True, "edge_analysis": edge, "model": model.upper()}
                else:
                    return {
                        "success":       True,
                        "kash_status":   self._ratings.to_dict("kash"),
                        "iggy_status":   self._ratings.to_dict("iggy"),
                    }
            elif tool_name == "list_venue_markets":
                return self.betfair.list_venue_markets(
                    venue=tool_args["venue"],
                    hours_ahead=int(tool_args.get("hours_ahead", 12)),
                    event_type=tool_args.get("event_type", "horse"),
                )
            elif tool_name == "place_back_bet":
                reason = tool_args.pop("reason", "Agent decision")
                result = self.betfair.place_back_bet(
                    market_id=tool_args["market_id"],
                    selection_id=int(tool_args["selection_id"]),
                    back_price=float(tool_args["back_price"]),
                    stake=float(tool_args["stake"]),
                    strategy_ref=tool_args.get("strategy_ref", "agent_back"),
                )
                result["bet_reason"] = reason
                result["bet_type"] = "back"
                self._log_bet(result, context=self._turn_context)
                return result
            elif tool_name == "orchestrate_venue_session":
                if self._orchestrator is None:
                    self._orchestrator = OrchestratorAgent(self.betfair, self._ratings)
                return self._orchestrator.run_venue_session(
                    venue=tool_args["venue"],
                    bankroll=float(tool_args["bankroll"]),
                    mode=tool_args.get("mode", "polling"),
                    min_profit_ratio=float(tool_args.get("min_profit_ratio", 1.5)),
                    loss_limit_pct=float(tool_args.get("loss_limit_pct", 20.0)),
                    min_edge_pct=float(tool_args.get("min_edge_pct", 3.0)),
                )
            elif tool_name == "get_session_report":
                if self._orchestrator is None:
                    return {"success": False, "error": "No active orchestration session found."}
                from sub_agents import ReporterAgent, SessionState
                dummy_state = SessionState(venue=tool_args.get("venue", "Unknown"))
                reporter = ReporterAgent(self.betfair, self._ratings, dummy_state)
                return reporter.run()
            else:
                return {"success": False, "error": f"Unknown tool: {tool_name}"}
        except Exception as e:
            logger.error(f"Tool execution error [{tool_name}]: {e}", exc_info=True)
            return {"success": False, "error": str(e)}

    def _log_bet(self, bet_result: dict, context: dict = None):
        log_entry = {
            "timestamp": datetime.datetime.now().isoformat(),
            **bet_result,
        }
        log_path = "bet_log.json"
        try:
            logs =[]
            if os.path.exists(log_path):
                with open(log_path, "r") as f:
                    logs = json.load(f)
            logs.append(log_entry)
            with open(log_path, "w") as f:
                json.dump(logs, f, indent=2, default=str)
        except Exception as e:
            logger.warning(f"Could not write bet log: {e}")

        try:
            bet_id       = str(bet_result.get("bet_id", ""))
            bet_type     = "back" if bet_result.get("bet_type") == "back" else "lay"
            market_id    = bet_result.get("market_id", "")
            selection_id = int(bet_result.get("selection_id", 0))
            price        = float(bet_result.get("lay_price") or bet_result.get("back_price", 0))
            stake        = float(bet_result.get("stake", 0))
            liability    = float(bet_result.get("liability", stake * (price - 1) if price > 1 else 0))

            horse   = self.memory.last_horse or {}
            runner_name  = horse.get("name", bet_result.get("runner_name", "Unknown"))
            venue        = horse.get("venue", "")
            race_name    = horse.get("market_name", "")

            ctx = context or {}
            profit_ratio = (1.0 / (price - 1.0)) if price > 1 else 0

            if bet_id:
                self.betting_memory.record_placement(
                    bet_id           = bet_id,
                    bet_type         = bet_type,
                    market_id        = market_id,
                    selection_id     = selection_id,
                    runner_name      = runner_name,
                    price            = price,
                    stake            = stake,
                    liability        = liability,
                    venue            = ctx.get("venue", venue),
                    race_name        = ctx.get("race_name", race_name),
                    strategy_ref     = bet_result.get("strategy_ref", ""),
                    wom_signal       = ctx.get("wom_signal", "UNKNOWN"),
                    model_signal     = ctx.get("model_signal", "UNKNOWN"),
                    model_edge_pct   = float(ctx.get("model_edge_pct", 0)),
                    timing_window    = ctx.get("timing_window", "UNKNOWN"),
                    profit_ratio     = profit_ratio,
                    sp_far           = ctx.get("sp_far"),
                    edge_vs_sp       = ctx.get("edge_vs_sp"),
                    opportunity_score= int(ctx.get("opportunity_score", 0)),
                )
        except Exception as e:
            logger.warning(f"BettingMemory record error (non-fatal): {e}")

    def chat_turn(self, user_message: str) -> str:
        print(f"\n🤖 Processing: '{user_message}'")
        self._turn_context: dict = {}

        memory_block = self.memory.build_context_block()
        perf_block = self.betting_memory.get_context_block()

        context_parts =[]
        if perf_block:
            context_parts.append(perf_block)
        if memory_block:
            context_parts.append(memory_block)

        if context_parts:
            enriched_message = "\n".join(context_parts) + f"\n\nUser: {user_message}"
        else:
            enriched_message = user_message

        response = self.chat.send_message(enriched_message)

        max_iterations = 10
        iteration = 0

        while iteration < max_iterations:
            iteration += 1

            if not response.candidates:
                return "No response generated (no candidates returned by the model)."
                
            content = response.candidates[0].content
            if not content or not getattr(content, "parts", None):
                finish_reason = response.candidates[0].finish_reason
                return f"No response generated. (Model halted early. Finish reason: {finish_reason})"

            function_calls = []
            for part in response.candidates[0].content.parts:
                if getattr(part, "function_call", None):
                    function_calls.append(part.function_call)

            if not function_calls:
                text_parts = []
                for part in response.candidates[0].content.parts:
                    if getattr(part, "text", None):
                        text_parts.append(part.text)
                return "\n".join(text_parts) if text_parts else "No response generated."

            tool_results =[]
            for fc in function_calls:
                tool_name = fc.name
                tool_args = dict(fc.args)
                print(f"  🔧 Calling: {tool_name}({json.dumps({k: v for k, v in tool_args.items() if k != 'reason'}, default=str)})")

                result = self._execute_tool(tool_name, tool_args)
                self._update_memory_from_result(tool_name, tool_args, result)
                self._capture_turn_signals(tool_name, tool_args, result)

                if result.get("success"):
                    print(f"  ✅ {tool_name}: OK")
                else:
                    print(f"  ❌ {tool_name}: {result.get('error', 'Failed')}")

                tool_results.append({
                    "function_response": {
                        "name": tool_name,
                        "response": result,
                    }
                })

            from google.generativeai.types import content_types
            response = self.chat.send_message([content_types.to_part(tr) for tr in tool_results]
            )

        return "Maximum tool call iterations reached. Please try a simpler request."

    def _capture_turn_signals(self, tool_name: str, tool_args: dict, result: dict):
        if not result.get("success"):
            return
        try:
            if tool_name == "get_weight_of_money":
                runners = result.get("runners", [])
                if runners:
                    top = runners[0]
                    self._turn_context["wom_signal"] = top.get("wom_signal", "UNKNOWN")
            elif tool_name == "get_external_ratings":
                edge = result.get("edge_analysis", {})
                self._turn_context["model_signal"]   = edge.get("signal", "UNKNOWN")
                self._turn_context["model_edge_pct"] = float(edge.get("edge_pct", 0))
            elif tool_name == "get_timing_advice":
                timing = result.get("timing", {})
                self._turn_context["timing_window"] = timing.get("window", "UNKNOWN")
            elif tool_name == "get_sp_predictions":
                runners = result.get("runners",[])
                last_horse = self.memory.last_horse
                if last_horse and runners:
                    sel_id = str(last_horse.get("selection_id", ""))
                    for r in runners:
                        if str(r.get("selection_id", "")) == sel_id:
                            self._turn_context["sp_far"]    = r.get("sp_far")
                            ev = r.get("edge_analysis", {})
                            self._turn_context["edge_vs_sp"] = ev.get("edge_net")
                            break
            elif tool_name == "search_horse":
                markets = result.get("markets",[])
                if markets:
                    self._turn_context["venue"]     = markets[0].get("venue", "")
                    self._turn_context["race_name"] = markets[0].get("market_name", "")
        except Exception as e:
            logger.debug(f"Signal capture error [{tool_name}]: {e}")

    def reset_conversation(self, keep_memory: bool = True):
        self.chat = self.model.start_chat(history=[])
        if not keep_memory:
            self.memory.clear()
            print("🗑️  Memory cleared.")
        logger.info(f"Conversation reset (memory {'kept' if keep_memory else 'cleared'})")

    def show_memory(self) -> str:
        block = self.memory.build_context_block()
        entity_mem = block if block else "Entity memory is empty."
        overall = self.betting_memory.overall_stats()
        if overall.get("total_bets_settled", 0) > 0:
            perf = (
                f"\n📊 BETTING PERFORMANCE\n"
                f"  Total bets placed:  {overall['total_bets_placed']}\n"
                f"  Settled:            {overall['total_bets_settled']}\n"
                f"  Win rate:           {overall['win_rate']}%\n"
                f"  ROI:                {overall['roi']:+.1f}%\n"
                f"  Total P&L:          ${overall['profit']:+.2f}\n"
                f"  Pending settlement: {overall['pending_settlement']}\n"
            )
            cs = self.cache.stats()
            cache_info = (
                f"\n⚡ CACHE (this session)\n"
                f"  Entries: {cs['entries']} | "
                f"Hits: {cs['hits']} | "
                f"Misses: {cs['misses']} | "
                f"Hit rate: {cs['hit_rate']}%\n"
            )
            return entity_mem + perf + cache_info
        return entity_mem

    def show_performance(self) -> str:
        settled = self.betting_memory.settled_records()
        if not settled:
            return "No settled bets yet. Place some bets first to build your performance profile."

        lines =["=" * 50, "  📊 STRATEGY PERFORMANCE ANALYTICS", "=" * 50]
        overall = self.betting_memory.overall_stats()
        lines.append(
            f"\nOverall: {overall['n']} bets | WR {overall['win_rate']}% | "
            f"ROI {overall['roi']:+.1f}% | P&L ${overall['profit']:+.2f}"
        )
        recent = self.betting_memory.recent_form(20)
        if recent.get("n", 0) >= 3:
            lines.append(f"Last {recent['n']} bets: ROI {recent['roi']:+.1f}% | Streak {recent['streak']}")
        lines.append("\n📌 By Lay Price:")
        for bucket, s in sorted(self.betting_memory.performance_by_price_bucket().items()):
            if s["n"] >= 2:
                lines.append(f"  {bucket}: {s['n']} bets | WR {s['win_rate']}% | ROI {s['roi']:+.1f}%")
        lines.append("\n📌 By WOM Signal:")
        for signal, s in sorted(self.betting_memory.performance_by_wom_signal().items()):
            if s["n"] >= 2:
                lines.append(f"  {signal}: {s['n']} bets | WR {s['win_rate']}% | ROI {s['roi']:+.1f}%")
        lines.append("\n📌 By Model Edge:")
        for bucket, s in sorted(self.betting_memory.performance_by_model_edge().items()):
            if s["n"] >= 2:
                lines.append(f"  {bucket}: {s['n']} bets | WR {s['win_rate']}% | ROI {s['roi']:+.1f}%")
        lines.append("\n📌 By Venue:")
        for venue, s in sorted(self.betting_memory.performance_by_venue().items(),
                                key=lambda x: x[1].get("roi") or 0, reverse=True):
            if s["n"] >= 2:
                lines.append(f"  {venue}: {s['n']} bets | ROI {s['roi']:+.1f}%")
        lines.append("\n📌 By Timing Window:")
        for w, s in sorted(self.betting_memory.performance_by_timing().items()):
            if s["n"] >= 2:
                lines.append(f"  {w}: {s['n']} bets | ROI {s['roi']:+.1f}%")
        lines.append("\n" + "=" * 50)
        return "\n".join(lines)
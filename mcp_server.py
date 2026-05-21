"""
Betfair MCP Server
Exposes Betfair API operations as MCP tools for use with Claude Desktop
or any MCP-compatible client.

Run with: python mcp_server.py
"""

import json
import logging
import os
import sys
from typing import Any

# MCP SDK
try:
    from mcp.server import Server
    from mcp.server.stdio import stdio_server
    from mcp import types
except ImportError:
    print("ERROR: mcp package not installed. Run: pip install mcp", file=sys.stderr)
    sys.exit(1)

from betfair_client import BetfairClient, is_paper_mode
from betting_memory import get_betting_memory
from external_ratings import get_ratings_cache
from market_analyser import analyse_market, timing_advice, venue_profile
from paper_autobet import DailyPaperBetScheduler, load_scheduler_config_from_env
from prediction_ledger import get_prediction_ledger
from shared_cache import SharedCache
from staking_engine import (
    recommend_stake, compare_staking_methods,
    estimate_edge_from_sp, run_simulation, compare_all_simulations,
    SimParams,
)

_cache = SharedCache.instance()
_paper_scheduler: DailyPaperBetScheduler | None = None


def _runner_name_from_book(market_id: str, selection_id: int) -> str:
    cached = _cache.get(f"market_book:{market_id}")
    if not cached or not cached.get("success"):
        return "Unknown"
    for r in cached.get("runners", []):
        if int(r.get("selection_id", 0)) == int(selection_id):
            return r.get("runner_name", "Unknown")
    return "Unknown"


def _log_placement(
    bet_type: str,
    result: dict,
    market_id: str,
    selection_id: int,
    price: float,
    stake: float,
    strategy_ref: str,
    context: dict | None,
) -> None:
    if not result.get("success") or not result.get("bet_id"):
        return
    ctx = context or {}
    liability = (price - 1.0) * stake if bet_type == "lay" else stake
    try:
        get_betting_memory().record_placement(
            bet_id=str(result["bet_id"]),
            bet_type=bet_type,
            market_id=market_id,
            selection_id=int(selection_id),
            runner_name=ctx.get("runner_name") or _runner_name_from_book(market_id, selection_id),
            price=float(price),
            stake=float(stake),
            liability=float(liability),
            venue=ctx.get("venue", ""),
            race_name=ctx.get("race_name", ""),
            strategy_ref=strategy_ref,
            wom_signal=ctx.get("wom_signal", "UNKNOWN"),
            model_signal=ctx.get("model_signal", "UNKNOWN"),
            model_edge_pct=float(ctx.get("model_edge_pct", 0.0)),
            timing_window=ctx.get("timing_window", "UNKNOWN"),
            profit_ratio=float(ctx.get("profit_ratio", 0.0)),
            sp_far=ctx.get("sp_far"),
            edge_vs_sp=ctx.get("edge_vs_sp"),
            opportunity_score=int(ctx.get("opportunity_score", 0)),
        )
    except Exception as e:
        logger.warning(f"BettingMemory record_placement failed: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# SESSION REGISTRY  (orchestrator + streaming)
# ─────────────────────────────────────────────────────────────────────────────

import threading
import uuid

_sessions_lock = threading.Lock()
_sessions: dict[str, dict] = {}      # session_id → {kind, state, thread, result, error, cancel, started_at}
_streams_lock = threading.Lock()
_streams: dict[str, dict] = {}       # stream_id → {runner, events, lock, started_at, venue, model}


def _new_id(prefix: str) -> str:
    return f"{prefix}-{uuid.uuid4().hex[:8]}"


def _run_venue_session_thread(
    session_id: str,
    venue: str,
    bankroll: float,
    mode: str,
    min_profit_ratio: float,
    loss_limit_pct: float,
    min_edge_pct: float,
) -> None:
    from sub_agents import OrchestratorAgent
    sess = _sessions[session_id]
    try:
        orch = OrchestratorAgent(_get_client(), get_ratings_cache())
        result = orch.run_venue_session(
            venue=venue,
            bankroll=bankroll,
            mode=mode,
            min_profit_ratio=min_profit_ratio,
            loss_limit_pct=loss_limit_pct,
            min_edge_pct=min_edge_pct,
        )
        with _sessions_lock:
            sess["result"] = result
            sess["status"] = "completed" if result.get("success") else "failed"
    except Exception as e:
        logger.exception(f"Orchestrator session {session_id} crashed")
        with _sessions_lock:
            sess["error"] = str(e)
            sess["status"] = "crashed"


def _start_stream_session(
    stream_id: str,
    venue: str,
    max_liability: float,
    min_profit_ratio: float,
    trigger_seconds: float,
    bankroll: float | None,
) -> None:
    """Launch a FlumineRunner with a venue lay strategy in a background thread.
    Raises ImportError if flumine is not installed."""
    if is_paper_mode():
        raise RuntimeError(
            "PAPER_MODE=true prevents starting Flumine streaming because Flumine "
            "places live exchange orders directly. Use orchestrate_venue_session "
            "with mode='polling' for paper-mode simulation, or set PAPER_MODE=false "
            "only when you intend to bet real money."
        )

    from flumine_engine import FlumineRunner
    import datetime
    runner = FlumineRunner(bankroll=bankroll or 0.0)
    runner.add_venue_strategy(
        venue=venue,
        max_liability=max_liability,
        min_profit_ratio=min_profit_ratio,
        trigger_seconds=trigger_seconds,
    )
    stream = {
        "runner":          runner,
        "started_at":      datetime.datetime.now().isoformat(),
        "venue":           venue,
        "max_liability":   max_liability,
        "min_profit_ratio":min_profit_ratio,
        "csv_offset":      0,
        "status":          "starting",
        "thread":          None,
        "error":           None,
    }
    with _streams_lock:
        _streams[stream_id] = stream

    def _spin():
        try:
            with _streams_lock:
                _streams[stream_id]["status"] = "running"
            runner.start()
            with _streams_lock:
                _streams[stream_id]["status"] = "completed"
        except Exception as e:
            logger.exception(f"Stream {stream_id} crashed")
            with _streams_lock:
                _streams[stream_id]["status"] = "crashed"
                _streams[stream_id]["error"] = str(e)

    t = threading.Thread(target=_spin, daemon=True, name=f"stream-{stream_id}")
    stream["thread"] = t
    t.start()


def _read_stream_events(stream_id: str, limit: int) -> list[dict]:
    """Return new rows in orders_agent.csv since the last poll for this stream."""
    import csv
    with _streams_lock:
        s = _streams.get(stream_id)
        if not s:
            return []
        offset = s.get("csv_offset", 0)
    path = "orders_agent.csv"
    if not os.path.exists(path):
        return []
    rows: list[dict] = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        # DictReader doesn't expose row index reliably with offsets, so iterate fully and slice.
        all_rows = list(reader)
    new_rows = all_rows[offset:]
    with _streams_lock:
        _streams[stream_id]["csv_offset"] = len(all_rows)
    return new_rows[-limit:] if limit and len(new_rows) > limit else new_rows

logging.basicConfig(level=logging.INFO, stream=sys.stderr)
logger = logging.getLogger("betfair-mcp")

if is_paper_mode():
    logger.warning("PAPER_MODE=true — bet tools will return simulated responses, no real money at risk.")
else:
    logger.warning("PAPER_MODE=false — bet tools will hit LIVE Betfair. Real money at risk.")

# Lazy client — constructed on first tool call, not at module import.
# This means importing mcp_server has no side effects, so any MCP client (Claude
# Desktop, Codex CLI, etc.) can launch it without crashing on missing credentials.
_client: BetfairClient | None = None


def _get_client() -> BetfairClient:
    global _client
    if _client is None:
        _client = BetfairClient()
        logger.info("BetfairClient initialised on first tool call")
    return _client


def _get_paper_scheduler() -> DailyPaperBetScheduler:
    global _paper_scheduler
    if _paper_scheduler is None:
        cfg = load_scheduler_config_from_env()
        _paper_scheduler = DailyPaperBetScheduler(_get_client, **cfg)
    return _paper_scheduler


server = Server("betfair-lay-agent")


# ─────────────────────────────────────────────────────────────────────────────
# TOOL DEFINITIONS
# ─────────────────────────────────────────────────────────────────────────────

@server.list_tools()
async def list_tools() -> list[types.Tool]:
    return[
        types.Tool(
            name="betfair_login",
            description=(
                "Login to Betfair API using BETFAIR_* credentials from the environment/.env. "
                "Must be called before any other Betfair operations. "
                "Uses interactive login (no certificates required)."
            ),
            inputSchema={
                "type": "object",
                "properties": {},
                "required": [],
            },
        ),
        types.Tool(
            name="get_account_balance",
            description=(
                "Get current Betfair account balance, available funds, and exposure. "
                "Use this to check funds before placing bets."
            ),
            inputSchema={
                "type": "object",
                "properties": {},
                "required":[],
            },
        ),
        types.Tool(
            name="get_market_profit_and_loss",
            description=(
                "Retrieve your live profit, loss, and exposure on a specific market. "
                "Returns exactly how much you will win or lose if each horse wins. "
                "Use this BEFORE placing additional bets in a market to ensure you "
                "aren't exceeding your risk limits or over-exposed on a single runner."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "market_id": {
                        "type": "string",
                        "description": "The Betfair market ID",
                    },
                },
                "required": ["market_id"],
            },
        ),
        types.Tool(
            name="get_account_statement",
            description=(
                "Get detailed recent account transactions including deposits, withdrawals, "
                "commissions, and settled bet P&L. Useful for tracking bankroll history."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "record_count": {
                        "type": "integer",
                        "description": "Number of recent transactions to return (default: 20)",
                        "default": 20
                    },
                },
                "required":[],
            },
        ),
        types.Tool(
            name="search_horse",
            description=(
                "Search for a horse in upcoming Australian horse racing WIN markets "
                "within the next 24 hours. Returns market details and runner selection IDs. "
                "Use this to find a horse before checking prices or placing bets."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "horse_name": {
                        "type": "string",
                        "description": "Name of the horse to search for (partial match supported)",
                    },
                    "hours_ahead": {
                        "type": "integer",
                        "description": "How many hours ahead to search (default: 24)",
                        "default": 24,
                    },
                },
                "required": ["horse_name"],
            },
        ),
        types.Tool(
            name="get_market_book",
            description=(
                "Get current prices (back and lay odds) for all runners in a market. "
                "Includes lay profitability analysis: profit_ratio = 1/(lay_price-1). "
                "A profit_ratio >= 1.5 means you win 1.5x your liability (lay_price <= 1.67). "
                "IMPORTANT: Only recommend lay bets where the horse has strong lay value."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "market_id": {
                        "type": "string",
                        "description": "The Betfair market ID (e.g., '1.150038686')",
                    },
                },
                "required": ["market_id"],
            },
        ),
        types.Tool(
            name="place_lay_bet",
            description=(
                "Place a LAY bet on a horse. In a lay bet, you are betting the horse WILL NOT win. "
                "You win the stake if the horse loses; you pay liability if it wins. "
                "ONLY place if: (1) profit_ratio >= 1.5 from market book analysis, OR "
                "(2) Codex determines sufficient value based on market analysis. "
                "Liability = (lay_price - 1) * stake. Always confirm sufficient account balance."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "market_id": {
                        "type": "string",
                        "description": "The Betfair market ID",
                    },
                    "selection_id": {
                        "type": "integer",
                        "description": "The runner's selection ID from search_horse or get_market_book",
                    },
                    "lay_price": {
                        "type": "number",
                        "description": "The price to lay at (will be rounded to nearest valid Betfair tick)",
                    },
                    "stake": {
                        "type": "number",
                        "description": "The backer's stake in GBP/AUD (amount you win if horse loses)",
                    },
                    "strategy_ref": {
                        "type": "string",
                        "description": "Strategy reference for tracking (max 15 chars, default: 'lay_agent')",
                        "default": "lay_agent",
                    },
                    "reason": {
                        "type": "string",
                        "description": "Your reasoning for placing this bet (for logging)",
                    },
                    "context": {
                        "type": "object",
                        "description": (
                            "Optional analytic context captured at placement for BettingMemory: "
                            "venue, race_name, runner_name, wom_signal, model_signal, model_edge_pct, "
                            "timing_window, profit_ratio, sp_far, edge_vs_sp, opportunity_score. "
                            "All fields are optional; unprovided fields fall back to UNKNOWN/0."
                        ),
                    },
                },
                "required": ["market_id", "selection_id", "lay_price", "stake"],
            },
        ),
        types.Tool(
            name="get_current_orders",
            description=(
                "Get all current open/unmatched orders. "
                "Use this to monitor active bets and positions."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "strategy_ref": {
                        "type": "string",
                        "description": "Filter by strategy reference (optional)",
                    },
                },
                "required":[],
            },
        ),
        types.Tool(
            name="cancel_order",
            description=(
                "Cancel an open order. Provide market_id to cancel all orders "
                "in that market, or also provide bet_id to cancel a specific bet."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "market_id": {
                        "type": "string",
                        "description": "The Betfair market ID",
                    },
                    "bet_id": {
                        "type": "string",
                        "description": "Specific bet ID to cancel (optional - if omitted, cancels all for market)",
                    },
                },
                "required": ["market_id"],
            },
        ),
        types.Tool(
            name="get_performance_summary",
            description=(
                "Get a summary of past settled bets and overall P&L for the last N days. "
                "Use this to track strategy performance and make informed decisions."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "days_back": {
                        "type": "integer",
                        "description": "Number of days to look back (default: 7)",
                        "default": 7,
                    },
                },
                "required":[],
            },
        ),
        types.Tool(
            name="get_weight_of_money",
            description=(
                "Fetch the full price ladder for a market and compute weight-of-money (WOM), "
                "market spread, and per-runner lay recommendations. "
                "WOM measures whether more money is backing or laying a runner. "
                "BACK_HEAVY WOM means smart money is backing the horse (risky lay). "
                "LAY_HEAVY WOM = market money agrees the horse will lose (good lay candidate). "
                "Also returns tick spread (tightness) and VWAP for each runner. "
                "Use this BEFORE placing any lay bet for a deeper market read."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "market_id": {
                        "type": "string",
                        "description": "The Betfair market ID",
                    },
                    "seconds_to_jump": {
                        "type": "number",
                        "description": "Seconds until scheduled race start (for timing advice)",
                    },
                    "venue": {
                        "type": "string",
                        "description": "Venue name (e.g. Flemington) for volume profile lookup",
                    },
                },
                "required": ["market_id"],
            },
        ),
        types.Tool(
            name="watch_for_steam",
            description=(
                "Monitor a market for a period of time and detect steam (horse firming quickly) "
                "or drift (horse drifting out) signals in real time. "
                "Steam = back price shortening fast — sharp money backing the horse. "
                "Drift = price lengthening — market losing confidence in the selection. "
                "Use this to time lay entries more precisely: lay a drifting horse WHILE it "
                "drifts (don't wait until after), or avoid lays on horses that are steaming. "
                "The tool blocks for (polls × interval_seconds) before returning. "
                "Default: 12 polls × 5s = 60 seconds of observation."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "market_id": {
                        "type": "string",
                        "description": "The Betfair market ID to monitor",
                    },
                    "polls": {
                        "type": "integer",
                        "description": "Number of price snapshots to take (default: 12)",
                        "default": 12,
                    },
                    "interval_seconds": {
                        "type": "number",
                        "description": "Seconds between snapshots (default: 5)",
                        "default": 5,
                    },
                },
                "required": ["market_id"],
            },
        ),
        types.Tool(
            name="get_timing_advice",
            description=(
                "Given the current time and the race start time, advise on the optimal "
                "window for placing a lay bet. Based on analysis of Victorian thoroughbred "
                "markets: ~70% of volume trades in the last 5 minutes before jump. "
                "Returns window classification: TOO_EARLY / MONITOR / OPTIMAL / LAST_CHANCE / INPLAY "
                "and a venue volume profile (Flemington ~$480k vs Bendigo ~$120k average). "
                "Use this when the user asks 'should I bet now' or 'when is the best time'."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "seconds_to_jump": {
                        "type": "number",
                        "description": "Seconds until the scheduled race start",
                    },
                    "venue": {
                        "type": "string",
                        "description": "Venue name for volume profile (e.g. 'Flemington')",
                    },
                },
                "required": ["seconds_to_jump"],
            },
        ),
        types.Tool(
            name="get_sp_predictions",
            description=(
                "Fetch Betfair's own pre-race BSP (Betfair Starting Price) near and far "
                "estimates for all runners in a market, alongside current best lay prices. "
                "sp_near = early-trading BSP model estimate. "
                "sp_far  = later, more accurate BSP estimate (prefer this one). "
                "KEY INSIGHT: if current lay price > sp_far, you have positive lay value "
                "(you are laying at better odds than the expected final BSP). "
                "If lay price < sp_far, you are laying below BSP = negative EV. "
                "Automatically returns edge analysis per runner using the BSP comparison. "
                "Use this to get a DATA-DRIVEN edge estimate instead of guessing win_prob. "
                "Call this before calculate_stake so win_prob is based on real data."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "market_id": {
                        "type": "string",
                        "description": "The Betfair market ID",
                    },
                },
                "required": ["market_id"],
            },
        ),
        types.Tool(
            name="calculate_stake",
            description=(
                "Calculate the optimal lay bet stake using Kelly criterion or other "
                "staking methods. Returns backer_stake, liability, edge metrics, and "
                "a staking method comparison. "
                "Methods: 'half_kelly' (recommended), 'kelly', 'quarter_kelly', "
                "'fixed', 'proportional_a' (% of bank as liability), "
                "'proportional_b' (% of bank as win target). "
                "Set win_prob from get_sp_predictions edge_analysis.market_win_prob "
                "for the most accurate result. "
                "Also runs a quick Monte Carlo simulation to show survival odds "
                "for the chosen strategy. "
                "Always call this BEFORE place_lay_bet to ensure correct stake sizing."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "bankroll": {
                        "type": "number",
                        "description": "Current available account balance",
                    },
                    "win_prob": {
                        "type": "number",
                        "description": (
                            "Estimated probability the horse WINS (0-1). "
                            "Use market_win_prob from get_sp_predictions if available. "
                            "Otherwise use 1/lay_price as a neutral estimate."
                        ),
                    },
                    "lay_price": {
                        "type": "number",
                        "description": "The lay price you intend to bet at",
                    },
                    "method": {
                        "type": "string",
                        "description": (
                            "Staking method: 'half_kelly' (default, recommended), "
                            "'kelly', 'quarter_kelly', 'fixed', "
                            "'proportional_a', 'proportional_b'"
                        ),
                    },
                    "fixed_liability": {
                        "type": "number",
                        "description": "Dollar liability for fixed staking (default: $20)",
                    },
                    "run_simulation": {
                        "type": "boolean",
                        "description": "Run a Monte Carlo simulation for all methods (default: false)",
                    },
                },
                "required": ["bankroll", "win_prob", "lay_price"],
            },
        ),
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
        types.Tool(
            name="get_session_report",
            description=(
                "Return live status, P&L and progress for an orchestrator session. "
                "Pass session_id returned by orchestrate_venue_session; omit to get the "
                "most-recent session."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "session_id": {"type": "string", "description": "Optional — defaults to most recent."},
                },
                "required": [],
            },
        ),
        types.Tool(
            name="get_daily_paper_bet_status",
            description=(
                "Return the status of the daily paper-bet scheduler, including when it "
                "will next run and the last recorded attempt."
            ),
            inputSchema={"type": "object", "properties": {}, "required": []},
        ),
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
                    "strategy_ref": {"type": "string", "default": "agent_back"},
                    "context": {
                        "type": "object",
                        "description": "Optional analytic context for BettingMemory (see place_lay_bet).",
                    },
                },
                "required": ["market_id", "selection_id", "back_price", "stake"],
            },
        ),
        types.Tool(
            name="get_strategy_insights",
            description=(
                "Return a formatted summary of historical bet performance from BettingMemory: "
                "overall ROI, win rate, performance by price bucket / WOM signal / model edge / "
                "venue / timing window, plus recent form. Inject the returned text into your own "
                "system prompt or reasoning to ground decisions in the agent's track record. "
                "Returns empty insight when fewer than 5 bets have settled."
            ),
            inputSchema={"type": "object", "properties": {}, "required": []},
        ),
        types.Tool(
            name="log_bet_outcome",
            description=(
                "Record the settlement outcome of a placed bet into BettingMemory. "
                "Use after a race settles when you know whether the lay/back won and the realised P&L."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "bet_id":      {"type": "string"},
                    "won":         {"type": "boolean",
                                    "description": "True if the BET won (i.e. lay collected stake / back paid out)."},
                    "profit":      {"type": "number", "description": "Realised P&L in account currency."},
                    "settled_at":  {"type": "string",
                                    "description": "Optional ISO-8601 settlement timestamp; defaults to now."},
                    "bsp_actual":  {"type": "number",
                                    "description": "Optional actual BSP for calibration vs sp_far."},
                },
                "required": ["bet_id", "won", "profit"],
            },
        ),
        types.Tool(
            name="get_bet_history",
            description=(
                "Return recorded bets from BettingMemory with optional filters. "
                "Defaults to the most recent 50 settled bets across all venues."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "venue":         {"type": "string"},
                    "settled_only":  {"type": "boolean", "default": True},
                    "limit":         {"type": "integer", "default": 50},
                    "sync_first":    {"type": "boolean", "default": False,
                                      "description": "Run sync_outcomes() first to pull recent settlements from bet_log.json / orders_agent.csv."},
                },
                "required": [],
            },
        ),
        types.Tool(
            name="orchestrate_venue_session",
            description=(
                "Kick off the full multi-agent venue session (Analyst → Risk → Executor → Reporter) "
                "in the background and return immediately with a session_id. Poll get_session_report "
                "with the returned session_id for live progress. Mirrors the OrchestratorAgent pipeline "
                "used by this Codex MCP server."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "venue":            {"type": "string", "description": "Track name, e.g. 'Doomben'."},
                    "bankroll":         {"type": "number",
                                         "description": "Bankroll for this session. Falls back to current available funds when omitted."},
                    "mode":             {"type": "string", "enum": ["flumine", "polling"], "default": "flumine"},
                    "min_profit_ratio": {"type": "number", "default": 1.5},
                    "loss_limit_pct":   {"type": "number", "default": 20.0},
                    "min_edge_pct":     {"type": "number", "default": 3.0},
                },
                "required": ["venue"],
            },
        ),
        types.Tool(
            name="quick_venue_report",
            description=(
                "Run the Analyst sub-agent only (no risk allocation, no execution) — returns a "
                "ranked shortlist of opportunities at a venue. Synchronous; suitable for "
                "'what's worth backing/laying at Doomben right now?' style queries."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "venue":    {"type": "string"},
                    "bankroll": {"type": "number"},
                },
                "required": ["venue"],
            },
        ),
        types.Tool(
            name="list_sessions",
            description="List every orchestrator session known to this MCP process (active + finished).",
            inputSchema={"type": "object", "properties": {}, "required": []},
        ),
        types.Tool(
            name="cancel_session",
            description=(
                "Mark a running orchestrator session as cancelled. The background thread checks the "
                "cancel flag between sub-agent steps; mid-step calls cannot be interrupted."
            ),
            inputSchema={
                "type": "object",
                "properties": {"session_id": {"type": "string"}},
                "required": ["session_id"],
            },
        ),
        types.Tool(
            name="start_stream_session",
            description=(
                "Start a Flumine venue-lay streaming session for a track. Returns a stream_id you "
                "can poll with get_stream_events. Uses the same VenueLayStrategy as the CLI; "
                "requires flumine + betfairlightweight installed. Refuses to start while "
                "PAPER_MODE=true because Flumine places live exchange orders directly."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "venue":           {"type": "string"},
                    "max_liability":   {"type": "number", "default": 50.0},
                    "min_profit_ratio":{"type": "number", "default": 1.5},
                    "trigger_seconds": {"type": "number", "default": 120.0},
                    "bankroll":        {"type": "number"},
                },
                "required": ["venue"],
            },
        ),
        types.Tool(
            name="get_stream_events",
            description=(
                "Pop queued events from a running stream session (price ticks, steam/drift signals, "
                "executed orders). Returns the events emitted since the last call."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "stream_id": {"type": "string"},
                    "limit":     {"type": "integer", "default": 200},
                },
                "required": ["stream_id"],
            },
        ),
        types.Tool(
            name="stop_stream_session",
            description="Stop a running Flumine stream session and return its final summary.",
            inputSchema={
                "type": "object",
                "properties": {"stream_id": {"type": "string"}},
                "required": ["stream_id"],
            },
        ),
    ]


# ─────────────────────────────────────────────────────────────────────────────
# TOOL HANDLERS
# ─────────────────────────────────────────────────────────────────────────────

@server.call_tool()
async def call_tool(name: str, arguments: dict) -> list[types.TextContent]:
    """Handle tool calls from the MCP client."""

    logger.info(f"Tool called: {name} with args: {arguments}")

    try:
        if name == "betfair_login":
            result = _get_client().login()

        elif name == "get_account_balance":
            result = _cache.account_balance(lambda: _get_client().get_account_funds())

        elif name == "get_market_profit_and_loss":
            result = _get_client().get_market_profit_and_loss(arguments["market_id"])

        elif name == "get_account_statement":
            record_count = arguments.get("record_count", 20)
            result = _get_client().get_account_statement(record_count=record_count)

        elif name == "search_horse":
            horse_name = arguments["horse_name"]
            hours_ahead = arguments.get("hours_ahead", 24)
            result = _get_client().search_horse_racing_markets(horse_name, hours_ahead)

        elif name == "get_market_book":
            market_id = arguments["market_id"]
            result = _cache.market_book(
                market_id,
                lambda: _get_client().get_market_book(market_id),
            )

        elif name == "place_lay_bet":
            reason = arguments.pop("reason", "No reason provided")
            context = arguments.pop("context", None)
            logger.info(f"BET REASON: {reason}")
            mid = arguments["market_id"]
            sel_id = int(arguments["selection_id"])
            lay_price = float(arguments["lay_price"])
            stake = float(arguments["stake"])
            strategy_ref = arguments.get("strategy_ref", "lay_agent")
            result = _get_client().place_lay_bet(
                market_id=mid, selection_id=sel_id,
                lay_price=lay_price, stake=stake, strategy_ref=strategy_ref,
                context=context,
            )
            result["bet_reason"] = reason
            if result.get("success"):
                _cache.invalidate(f"market_book:{mid}")
                _cache.invalidate(f"market_depth:{mid}")
                _cache.invalidate("account_balance")
                _log_placement("lay", result, mid, sel_id, lay_price, stake, strategy_ref, context)

        elif name == "get_current_orders":
            strategy_ref = arguments.get("strategy_ref")
            result = _get_client().get_current_orders(strategy_ref)

        elif name == "cancel_order":
            mid = arguments["market_id"]
            result = _get_client().cancel_order(
                market_id=mid,
                bet_id=arguments.get("bet_id"),
            )
            if result.get("success"):
                _cache.invalidate(f"market_book:{mid}")
                _cache.invalidate(f"market_depth:{mid}")
                _cache.invalidate("account_balance")

        elif name == "get_performance_summary":
            days_back = arguments.get("days_back", 7)
            cleared = _get_client().get_cleared_orders(days_back)
            try:
                mem = get_betting_memory()
                mem.sync_outcomes()
                memory_summary = {
                    "overall":             mem.overall_stats() if hasattr(mem, "overall_stats") else None,
                    "recent_form":         mem.recent_form(20),
                    "by_price_bucket":     mem.performance_by_price_bucket(),
                    "by_wom_signal":       mem.performance_by_wom_signal(),
                    "by_model_edge":       mem.performance_by_model_edge(),
                    "by_venue":            mem.performance_by_venue(),
                    "by_timing":           mem.performance_by_timing(),
                    "by_model_signal":     mem.performance_by_model_signal(),
                }
                try:
                    memory_summary["prediction_calibration"] = get_prediction_ledger().summary()
                except Exception as ledger_exc:
                    memory_summary["prediction_calibration"] = {"error": str(ledger_exc)}
            except Exception as e:
                logger.warning(f"BettingMemory summary failed: {e}")
                memory_summary = {"error": str(e)}
            result = {
                "success":         cleared.get("success", False),
                "cleared_orders":  cleared,
                "memory_summary":  memory_summary,
                "days_back":       days_back,
            }

        elif name == "get_weight_of_money":
            market_id = arguments["market_id"]
            book = _cache.market_book(
                market_id,
                lambda: _get_client().get_market_book(market_id),
            )
            depth_book = _cache.get_or_fetch(
                key=f"market_depth:{market_id}",
                ttl=5,
                fetch_fn=lambda: _get_client().get_market_depth(market_id, depth=10),
                skip_cache_if=lambda r: not r.get("success"),
            )
            if not book.get("success"):
                result = book
            else:
                result = analyse_market(
                    market_book=book,
                    full_depth_book=depth_book if depth_book.get("success") else None,
                    seconds_to_jump=arguments.get("seconds_to_jump"),
                    venue=arguments.get("venue", ""),
                )

        elif name == "watch_for_steam":
            result = _get_client().poll_market_for_steam(
                market_id=arguments["market_id"],
                polls=arguments.get("polls", 12),
                interval_seconds=float(arguments.get("interval_seconds", 5)),
            )

        elif name == "get_timing_advice":
            result = {
                "success": True,
                "timing":  timing_advice(
                    seconds_to_jump=float(arguments["seconds_to_jump"]),
                    venue=arguments.get("venue", ""),
                ),
                "venue_profile": venue_profile(arguments.get("venue", "")) if arguments.get("venue") else None,
            }

        elif name == "get_sp_predictions":
            mid = arguments["market_id"]
            result = _cache.bsp_predictions(mid, lambda: _get_client().get_sp_predictions(mid))

        elif name == "calculate_stake":
            bankroll   = float(arguments["bankroll"])
            win_prob   = float(arguments["win_prob"])
            lay_price  = float(arguments["lay_price"])
            method     = arguments.get("method", "half_kelly")
            fixed_liab = float(arguments.get("fixed_liability", 20.0))
            do_sim     = bool(arguments.get("run_simulation", False))

            stake_result = recommend_stake(
                bankroll=bankroll,
                win_prob=win_prob,
                lay_price=lay_price,
                method=method,
                fixed_liability=fixed_liab,
            )
            comparison = compare_staking_methods(bankroll, win_prob, lay_price,
                                                 fixed_liability=fixed_liab)

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

        elif name == "list_venue_markets":
            venue = arguments["venue"]
            hours_ahead = int(arguments.get("hours_ahead", 12))
            event_type = arguments.get("event_type", "horse")
            result = _cache.venue_markets(
                f"{venue}:{event_type}:{hours_ahead}",
                lambda: _get_client().list_venue_markets(
                    venue=venue, hours_ahead=hours_ahead, event_type=event_type,
                ),
            )

        elif name == "get_external_ratings":
            model = arguments.get("model", "kash")
            venue = arguments["venue"]
            cache = get_ratings_cache()
            markets = cache.get_venue_markets(venue=venue, model=model)
            summary = cache.to_dict(model=model)
            result = {
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

        elif name == "get_session_report":
            sid = arguments.get("session_id")
            with _sessions_lock:
                if sid:
                    sess = _sessions.get(sid)
                    if not sess:
                        result = {"success": False, "error": f"Unknown session_id {sid}"}
                    else:
                        result = {
                            "success":    True,
                            "session_id": sid,
                            "status":     sess.get("status"),
                            "venue":      sess.get("venue"),
                            "started_at": sess.get("started_at"),
                            "result":     sess.get("result"),
                            "error":      sess.get("error"),
                        }
                else:
                    if not _sessions:
                        result = {"success": False, "message": "No active orchestrator sessions in this process."}
                    else:
                        latest = max(_sessions.values(), key=lambda s: s.get("started_at", ""))
                        result = {
                            "success":    True,
                            "session_id": latest.get("session_id"),
                            "status":     latest.get("status"),
                            "venue":      latest.get("venue"),
                            "started_at": latest.get("started_at"),
                            "result":     latest.get("result"),
                            "error":      latest.get("error"),
                            "note":       "No session_id supplied; returned most-recent session.",
                        }

        elif name == "get_daily_paper_bet_status":
            result = {
                "success": True,
                "scheduler": _get_paper_scheduler().status(),
            }

        elif name == "place_back_bet":
            context = arguments.pop("context", None)
            mid = arguments["market_id"]
            sel_id = int(arguments["selection_id"])
            back_price = float(arguments["back_price"])
            stake = float(arguments["stake"])
            strategy_ref = arguments.get("strategy_ref", "agent_back")
            result = _get_client().place_back_bet(
                market_id=mid, selection_id=sel_id,
                back_price=back_price, stake=stake, strategy_ref=strategy_ref,
            )
            if result.get("success"):
                _cache.invalidate(f"market_book:{mid}")
                _cache.invalidate(f"market_depth:{mid}")
                _cache.invalidate("account_balance")
                _log_placement("back", result, mid, sel_id, back_price, stake, strategy_ref, context)

        elif name == "get_strategy_insights":
            mem = get_betting_memory()
            mem.sync_outcomes()
            insights = mem.get_strategy_insights()
            try:
                prediction_calibration = get_prediction_ledger().summary()
            except Exception as ledger_exc:
                prediction_calibration = {"error": str(ledger_exc)}
            result = {
                "success":          True,
                "insights_text":    insights,
                "have_insights":    bool(insights),
                "settled_bets":     len(mem.settled_records()),
                "total_recorded":   len(mem.all_records()),
                "prediction_calibration": prediction_calibration,
            }

        elif name == "log_bet_outcome":
            mem = get_betting_memory()
            mem.update_outcome(
                bet_id=str(arguments["bet_id"]),
                won=bool(arguments["won"]),
                profit=float(arguments["profit"]),
                settled_at=arguments.get("settled_at"),
                bsp_actual=arguments.get("bsp_actual"),
                result_source="mcp_log_bet_outcome",
            )
            prediction_updated = False
            try:
                runner_won = bool(arguments["won"])
                try:
                    matching_record = next(
                        (
                            r for r in mem.all_records()
                            if str(getattr(r, "bet_id", "")) == str(arguments["bet_id"])
                        ),
                        None,
                    )
                    if matching_record and getattr(matching_record, "bet_type", "") == "lay":
                        runner_won = not bool(arguments["won"])
                except Exception:
                    pass
                prediction_updated = get_prediction_ledger().update_outcome_by_bet_id(
                    str(arguments["bet_id"]),
                    won=runner_won,
                    profit=float(arguments["profit"]),
                    settled_at=arguments.get("settled_at"),
                    bsp_actual=arguments.get("bsp_actual"),
                    result_source="mcp_log_bet_outcome",
                )
            except Exception as ledger_exc:
                logger.warning(f"PredictionLedger update_outcome_by_bet_id failed: {ledger_exc}")
            result = {
                "success": True,
                "bet_id": str(arguments["bet_id"]),
                "prediction_ledger_updated": prediction_updated,
            }

        elif name == "get_bet_history":
            mem = get_betting_memory()
            if arguments.get("sync_first"):
                mem.sync_outcomes()
            settled_only = bool(arguments.get("settled_only", True))
            venue_filter = arguments.get("venue")
            limit = int(arguments.get("limit", 50))
            records = mem.settled_records() if settled_only else mem.all_records()
            if venue_filter:
                records = [r for r in records if (r.venue or "").lower() == venue_filter.lower()]
            records = sorted(records, key=lambda r: r.placed_at or "", reverse=True)[:limit]
            result = {
                "success":  True,
                "count":    len(records),
                "records":  [r.to_dict() for r in records],
            }

        elif name == "orchestrate_venue_session":
            venue = arguments["venue"]
            bankroll = arguments.get("bankroll")
            if bankroll is None:
                funds = _cache.account_balance(lambda: _get_client().get_account_funds())
                bankroll = float(funds.get("available_to_bet", 0.0)) if funds.get("success") else 0.0
            sid = _new_id("orch")
            sess = {
                "session_id": sid,
                "kind":       "orchestrator",
                "venue":      venue,
                "status":     "starting",
                "started_at": __import__("datetime").datetime.now().isoformat(),
                "result":     None,
                "error":      None,
            }
            with _sessions_lock:
                _sessions[sid] = sess
            t = threading.Thread(
                target=_run_venue_session_thread,
                args=(
                    sid, venue, float(bankroll),
                    arguments.get("mode", "flumine"),
                    float(arguments.get("min_profit_ratio", 1.5)),
                    float(arguments.get("loss_limit_pct", 20.0)),
                    float(arguments.get("min_edge_pct", 3.0)),
                ),
                daemon=True,
                name=f"orch-{sid}",
            )
            sess["thread"] = t
            t.start()
            with _sessions_lock:
                _sessions[sid]["status"] = "running"
            result = {
                "success":    True,
                "session_id": sid,
                "venue":      venue,
                "bankroll":   bankroll,
                "status":     "running",
                "poll_with":  "get_session_report",
            }

        elif name == "quick_venue_report":
            from sub_agents import OrchestratorAgent
            venue = arguments["venue"]
            bankroll = arguments.get("bankroll")
            if bankroll is None:
                funds = _cache.account_balance(lambda: _get_client().get_account_funds())
                bankroll = float(funds.get("available_to_bet", 0.0)) if funds.get("success") else 0.0
            orch = OrchestratorAgent(_get_client(), get_ratings_cache())
            result = orch.quick_report(venue=venue, bankroll=float(bankroll))

        elif name == "list_sessions":
            with _sessions_lock:
                rows = [
                    {
                        "session_id": sid,
                        "kind":       s.get("kind"),
                        "venue":      s.get("venue"),
                        "status":     s.get("status"),
                        "started_at": s.get("started_at"),
                        "error":      s.get("error"),
                    }
                    for sid, s in _sessions.items()
                ]
            with _streams_lock:
                streams = [
                    {
                        "stream_id":  sid,
                        "kind":       "stream",
                        "venue":      s.get("venue"),
                        "status":     s.get("status"),
                        "started_at": s.get("started_at"),
                        "error":      s.get("error"),
                    }
                    for sid, s in _streams.items()
                ]
            result = {"success": True, "orchestrator_sessions": rows, "stream_sessions": streams}

        elif name == "cancel_session":
            sid = arguments["session_id"]
            with _sessions_lock:
                sess = _sessions.get(sid)
                if not sess:
                    result = {"success": False, "error": f"Unknown session_id {sid}"}
                else:
                    sess["cancel"] = True
                    sess["status"] = "cancel_requested"
                    result = {
                        "success":    True,
                        "session_id": sid,
                        "note":       "Cancel flag set. Background thread checks between sub-agent steps; current in-flight Betfair calls run to completion.",
                    }

        elif name == "start_stream_session":
            try:
                stream_id = _new_id("stream")
                _start_stream_session(
                    stream_id=stream_id,
                    venue=arguments["venue"],
                    max_liability=float(arguments.get("max_liability", 50.0)),
                    min_profit_ratio=float(arguments.get("min_profit_ratio", 1.5)),
                    trigger_seconds=float(arguments.get("trigger_seconds", 120.0)),
                    bankroll=arguments.get("bankroll"),
                )
                result = {"success": True, "stream_id": stream_id, "venue": arguments["venue"]}
            except ImportError as e:
                result = {"success": False, "error": str(e),
                          "hint": "Install Flumine: pip install flumine betfairlightweight"}
            except RuntimeError as e:
                result = {
                    "success": False,
                    "error": str(e),
                    "hint": "Keep PAPER_MODE=true for dry runs; use polling mode for simulated execution.",
                }

        elif name == "get_stream_events":
            stream_id = arguments["stream_id"]
            with _streams_lock:
                s = _streams.get(stream_id)
            if not s:
                result = {"success": False, "error": f"Unknown stream_id {stream_id}"}
            else:
                events = _read_stream_events(stream_id, int(arguments.get("limit", 200)))
                result = {
                    "success":     True,
                    "stream_id":   stream_id,
                    "status":      s.get("status"),
                    "venue":       s.get("venue"),
                    "event_count": len(events),
                    "events":      events,
                }

        elif name == "stop_stream_session":
            stream_id = arguments["stream_id"]
            with _streams_lock:
                s = _streams.get(stream_id)
            if not s:
                result = {"success": False, "error": f"Unknown stream_id {stream_id}"}
            else:
                summary = {}
                try:
                    summary = s["runner"].get_p_and_l()
                except Exception as e:
                    summary = {"success": False, "error": str(e)}
                # Flumine has no clean public stop hook on FlumineRunner; the framework's run()
                # exits when all markets close. We mark the slot stopped from our side and let
                # the daemon thread exit naturally. If a hard stop is needed, kill the process.
                with _streams_lock:
                    _streams[stream_id]["status"] = "stop_requested"
                result = {
                    "success":   True,
                    "stream_id": stream_id,
                    "summary":   summary,
                    "note":      "Flumine framework exits when all watched markets close. The MCP marks the slot stopped; the daemon thread will end shortly.",
                }

        else:
            result = {"success": False, "error": f"Unknown tool: {name}"}

    except Exception as e:
        logger.error(f"Error in tool {name}: {e}", exc_info=True)
        result = {"success": False, "error": str(e)}

    return[types.TextContent(type="text", text=json.dumps(result, indent=2, default=str))]


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

async def main():
    logger.info("Starting Betfair MCP Server...")
    _get_paper_scheduler().start()
    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            server.create_initialization_options(),
        )

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())

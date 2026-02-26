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

from betfair_client import BetfairClient
from market_analyser import analyse_market, timing_advice, venue_profile
from staking_engine import (
    recommend_stake, compare_staking_methods,
    estimate_edge_from_sp, run_simulation, compare_all_simulations,
    SimParams,
)

logging.basicConfig(level=logging.INFO, stream=sys.stderr)
logger = logging.getLogger("betfair-mcp")

# Global client instance
client = BetfairClient(credentials_path="credentials.json")
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
                "Login to Betfair API using credentials from credentials.json. "
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
                "(2) the Gemini agent determines sufficient value based on market analysis. "
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
            result = client.login()

        elif name == "get_account_balance":
            result = client.get_account_funds()

        elif name == "get_market_profit_and_loss":
            result = client.get_market_profit_and_loss(arguments["market_id"])
            
        elif name == "get_account_statement":
            record_count = arguments.get("record_count", 20)
            result = client.get_account_statement(record_count=record_count)

        elif name == "search_horse":
            horse_name = arguments["horse_name"]
            hours_ahead = arguments.get("hours_ahead", 24)
            result = client.search_horse_racing_markets(horse_name, hours_ahead)

        elif name == "get_market_book":
            market_id = arguments["market_id"]
            result = client.get_market_book(market_id)

        elif name == "place_lay_bet":
            reason = arguments.pop("reason", "No reason provided")
            logger.info(f"BET REASON: {reason}")
            result = client.place_lay_bet(
                market_id=arguments["market_id"],
                selection_id=int(arguments["selection_id"]),
                lay_price=float(arguments["lay_price"]),
                stake=float(arguments["stake"]),
                strategy_ref=arguments.get("strategy_ref", "lay_agent"),
            )
            result["bet_reason"] = reason

        elif name == "get_current_orders":
            strategy_ref = arguments.get("strategy_ref")
            result = client.get_current_orders(strategy_ref)

        elif name == "cancel_order":
            result = client.cancel_order(
                market_id=arguments["market_id"],
                bet_id=arguments.get("bet_id"),
            )

        elif name == "get_performance_summary":
            days_back = arguments.get("days_back", 7)
            result = client.get_cleared_orders(days_back)

        elif name == "get_weight_of_money":
            market_id = arguments["market_id"]
            # Fetch both standard book (for lay_analysis) and full depth (for WOM)
            book       = client.get_market_book(market_id)
            depth_book = client.get_market_depth(market_id, depth=10)
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
            result = client.poll_market_for_steam(
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
            result = client.get_sp_predictions(arguments["market_id"])

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
    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            server.create_initialization_options(),
        )

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
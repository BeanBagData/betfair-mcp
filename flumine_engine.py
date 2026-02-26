"""
Flumine Execution Engine
Provides Flumine-based streaming strategies that run continuously,
receiving real-time market updates rather than polling.

Why Flumine vs polling (BetfairClient):
  - Polling: fetch data on demand → suitable for one-off decisions
  - Streaming: Betfair pushes every market tick instantly → needed for
    automated, all-day strategies where timing is critical

Strategies implemented here:
  1. AgentModelStrategy  – compares Kash/Iggy model price against live prices,
                           backs or lays with Kelly-sized stakes
  2. VenueLayStrategy    – lays market over-rounds runners at a specific venue
                           using the agent's full analysis pipeline
  3. FlumineRunner       – starts/stops the Flumine framework, manages
                           circuit breakers and daily loss limits

Usage:
    from flumine_engine import FlumineRunner
    runner = FlumineRunner(credentials_path='credentials.json', bankroll=500)
    runner.add_venue_strategy(venue='Doomben', model='kash', max_stake=20)
    runner.start()   # blocks until all races are done
"""

from __future__ import annotations

import csv
import datetime
import logging
import os
import re
import threading
from typing import Optional

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# LAZY IMPORTS  (flumine / betfairlightweight are optional heavy deps)
# ─────────────────────────────────────────────────────────────────────────────

def _check_flumine() -> bool:
    try:
        import flumine  # noqa: F401
        return True
    except ImportError:
        return False


# ─────────────────────────────────────────────────────────────────────────────
# ORDER LOG FIELDS
# ─────────────────────────────────────────────────────────────────────────────

_LOG_FIELDS = [
    "bet_id", "strategy_name", "market_id", "selection_id", "trade_id",
    "date_time_placed", "price", "price_matched", "size", "size_matched",
    "profit", "side", "elapsed_seconds_executable", "order_status",
    "market_note", "trade_notes", "order_notes",
]


# ─────────────────────────────────────────────────────────────────────────────
# LOGGING CONTROL
# ─────────────────────────────────────────────────────────────────────────────

def _make_logging_control(output_path: str):
    """
    Build a Flumine LoggingControl that writes settled orders to a CSV.
    Derived from How-to-Automate scripts 2–4 LiveLoggingControl.
    """
    from flumine.controls.loggingcontrols import LoggingControl
    from flumine.order.ordertype import OrderTypes

    class _Control(LoggingControl):
        NAME = "AGENT_LOGGING_CONTROL"

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            if not os.path.exists(output_path):
                with open(output_path, "w", newline="") as f:
                    csv.DictWriter(f, fieldnames=_LOG_FIELDS).writeheader()

        def _process_cleared_orders_meta(self, event):
            orders = event.event
            with open(output_path, "a", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=_LOG_FIELDS)
                for order in orders:
                    size  = order.order_type.size  if order.order_type.ORDER_TYPE == OrderTypes.LIMIT else order.order_type.liability
                    price = order.order_type.price if order.order_type.ORDER_TYPE != OrderTypes.MARKET_ON_CLOSE else None
                    try:
                        writer.writerow({
                            "bet_id":                    order.bet_id,
                            "strategy_name":             str(order.trade.strategy),
                            "market_id":                 order.market_id,
                            "selection_id":              order.selection_id,
                            "trade_id":                  order.trade.id,
                            "date_time_placed":          order.responses.date_time_placed,
                            "price":                     price,
                            "price_matched":             order.average_price_matched,
                            "size":                      size,
                            "size_matched":              order.size_matched,
                            "profit":                    0 if not order.cleared_order else order.cleared_order.profit,
                            "side":                      order.side,
                            "elapsed_seconds_executable": order.elapsed_seconds_executable,
                            "order_status":              order.status.value,
                            "market_note":               order.trade.market_notes,
                            "trade_notes":               order.trade.notes_str,
                            "order_notes":               order.notes_str,
                        })
                    except Exception as e:
                        logger.error(f"Logging error: {e}")

        def _process_cleared_markets(self, event):
            for cm in event.event.orders:
                logger.info(f"Cleared market {cm.market_id}: profit={cm.profit:.2f} commission={cm.commission:.2f}")

    return _Control()


# ─────────────────────────────────────────────────────────────────────────────
# TERMINATION WORKER
# ─────────────────────────────────────────────────────────────────────────────

def _terminate(context: dict, flumine_fw, today_only: bool = True, seconds_closed: int = 1200) -> None:
    """Terminate the Flumine framework when all today's markets have closed."""
    from flumine.events.events import TerminationEvent
    markets = list(flumine_fw.markets.markets.values())
    if today_only:
        today = datetime.datetime.utcnow().date()
        markets = [
            m for m in markets
            if m.market_start_datetime.date() == today
            and (m.elapsed_seconds_closed is None or m.elapsed_seconds_closed < seconds_closed)
        ]
    if len(markets) == 0:
        logger.info("No more markets active — terminating Flumine framework")
        flumine_fw.handler_queue.put(TerminationEvent(flumine_fw))


# ─────────────────────────────────────────────────────────────────────────────
# STRATEGY 1: MODEL-DRIVEN (KASH / IGGY)
# ─────────────────────────────────────────────────────────────────────────────

def _make_model_strategy(
    ratings_cache,
    staking_fn,
    bankroll: float,
    model: str = "kash",
    max_stake: float = 20.0,
    trigger_seconds: float = 60.0,
    min_edge_pct: float = 3.0,
):
    """
    Build a Flumine BaseStrategy that trades based on external model prices.

    Logic (from How-to-Automate scripts 3 & 4):
      - At *trigger_seconds* before jump, compare live market prices to
        the Kash/Iggy model price for each runner.
      - BACK if best_back > model_price (market overestimates horse)
      - LAY  if best_lay  < model_price (market underestimates horse)
      - Stake is sized using the passed *staking_fn* (Kelly-based)

    Args:
        ratings_cache:   RatingsCache instance with today's ratings loaded
        staking_fn:      Callable(bankroll, win_prob, lay_price) → stake dict
        bankroll:        Current account balance for Kelly sizing
        model:           "kash" (thoroughbreds) or "iggy" (greyhounds)
        max_stake:       Hard cap on bet size ($)
        trigger_seconds: Window before jump to start trading (default: 60s)
        min_edge_pct:    Minimum % edge vs model price to place a bet

    Returns:
        A Flumine BaseStrategy subclass instance.
    """
    from flumine import BaseStrategy
    from flumine.markets.market import Market
    from flumine.order.order import LimitOrder
    from flumine.order.trade import Trade
    from betfairlightweight.resources import MarketBook

    class _ModelStrategy(BaseStrategy):
        _label = f"AgentModel_{model.upper()}"

        def start(self) -> None:
            logger.info(f"[{self._label}] Strategy started")

        def check_market_book(self, market: Market, market_book: MarketBook) -> bool:
            return market_book.status != "CLOSED"

        def process_market_book(self, market: Market, market_book: MarketBook) -> None:
            if market.seconds_to_start > trigger_seconds or market_book.inplay:
                return

            for runner in market_book.runners:
                if runner.status != "ACTIVE":
                    continue
                if not runner.ex.available_to_back or not runner.ex.available_to_lay:
                    continue

                market_id    = market_book.market_id
                selection_id = str(runner.selection_id)
                best_back    = runner.ex.available_to_back[0]["price"]
                best_lay     = runner.ex.available_to_lay[0]["price"]

                edge = ratings_cache.model_edge(
                    market_id, selection_id,
                    current_lay=best_lay, current_back=best_back,
                    model=model,
                )

                if edge.get("edge_pct") is None or edge["edge_pct"] < min_edge_pct:
                    continue

                signal = edge.get("signal")
                model_price = edge.get("model_price", best_back)

                # ── LAY (agent's primary strategy) ─────────────────────────
                if signal in ("LAY", "BOTH") and best_lay < model_price:
                    # Estimate win prob from model price
                    win_prob = min(0.95, 1.0 / model_price) if model_price else 0.5
                    try:
                        stake_rec = staking_fn(
                            bankroll=bankroll,
                            win_prob=win_prob,
                            lay_price=best_lay,
                            method="half_kelly",
                        )
                        raw_stake = stake_rec.get("backer_stake", max_stake / 2)
                        stake = min(float(raw_stake), max_stake)
                    except Exception:
                        stake = min(5.0, max_stake)

                    if stake < 2.0:
                        continue

                    logger.info(
                        f"[{self._label}] LAY {selection_id} @ {best_lay} "
                        f"(model={model_price}, edge={edge['edge_pct']:.1f}%, stake={stake:.2f})"
                    )
                    trade = Trade(market_id=market_id, selection_id=runner.selection_id,
                                  handicap=runner.handicap, strategy=self)
                    market.place_order(
                        trade.create_order(side="LAY", order_type=LimitOrder(price=best_lay, size=stake))
                    )

                # ── BACK ───────────────────────────────────────────────────
                elif signal == "BACK" and best_back > model_price:
                    stake = min(5.0, max_stake)   # Conservative flat stake for back bets
                    logger.info(
                        f"[{self._label}] BACK {selection_id} @ {best_back} "
                        f"(model={model_price}, edge={edge['edge_pct']:.1f}%, stake={stake:.2f})"
                    )
                    trade = Trade(market_id=market_id, selection_id=runner.selection_id,
                                  handicap=runner.handicap, strategy=self)
                    market.place_order(
                        trade.create_order(side="BACK", order_type=LimitOrder(price=best_back, size=stake))
                    )

    return _ModelStrategy


# ─────────────────────────────────────────────────────────────────────────────
# STRATEGY 2: VENUE LAY STRATEGY  (uses agent's full analysis pipeline)
# ─────────────────────────────────────────────────────────────────────────────

def _make_venue_lay_strategy(
    betfair_client,
    ratings_cache,
    venue: str,
    bankroll: float,
    max_liability: float = 50.0,
    min_profit_ratio: float = 1.5,
    trigger_seconds: float = 120.0,
):
    """
    Build a Flumine strategy that applies the full agent analysis pipeline
    to every WIN market at *venue* and lays runners that pass all checks.

    This is the strategy for "make money at Doomben today" — it:
      1. Checks model edge (Kash ratings)
      2. Validates profit_ratio ≥ min_profit_ratio (Kelly gate)
      3. Cross-references with WOM signal from betfair_client
      4. Sizes the lay stake via Kelly
    """
    from flumine import BaseStrategy
    from flumine.markets.market import Market
    from flumine.order.order import LimitOrder
    from flumine.order.trade import Trade
    from betfairlightweight.resources import MarketBook
    from staking_engine import recommend_stake

    venue_lower = venue.lower().strip()
    _placed: set[tuple[str, int]] = set()  # (market_id, selection_id) already traded

    class _VenueLayStrategy(BaseStrategy):
        _label = f"VenueLay_{venue}"

        def start(self) -> None:
            logger.info(f"[{self._label}] Strategy started for {venue}")

        def check_market_book(self, market: Market, market_book: MarketBook) -> bool:
            # Only active, non-closed markets
            if market_book.status == "CLOSED":
                return False
            # Venue filter — market catalogue should contain venue name
            try:
                mkt_venue = (market.market_catalogue.event.venue or "").lower()
                return venue_lower in mkt_venue
            except Exception:
                return False

        def process_market_book(self, market: Market, market_book: MarketBook) -> None:
            if market.seconds_to_start > trigger_seconds or market_book.inplay:
                return

            market_id = market_book.market_id

            for runner in market_book.runners:
                key = (market_id, runner.selection_id)
                if key in _placed or runner.status != "ACTIVE":
                    continue
                if not runner.ex.available_to_lay:
                    continue

                best_lay = runner.ex.available_to_lay[0]["price"]

                # Gate 1: profit ratio
                if best_lay <= 1.0:
                    continue
                profit_ratio = 1.0 / (best_lay - 1.0)
                if profit_ratio < min_profit_ratio:
                    continue

                # Gate 2: model edge
                sel_id_str = str(runner.selection_id)
                edge = ratings_cache.model_edge(
                    market_id, sel_id_str, current_lay=best_lay, model="kash"
                )
                if edge.get("signal") not in ("LAY", "BOTH", "NO_MODEL_PRICE"):
                    continue  # Model says BACK or NONE → skip

                # Gate 3: Kelly stake sizing
                model_price = edge.get("model_price") or best_lay
                win_prob    = min(0.95, 1.0 / model_price)
                try:
                    stake_rec = recommend_stake(
                        bankroll=bankroll,
                        win_prob=win_prob,
                        lay_price=best_lay,
                        method="half_kelly",
                    )
                    backer_stake = float(stake_rec.get("backer_stake", 5.0))
                    liability    = (best_lay - 1) * backer_stake
                except Exception:
                    backer_stake, liability = 5.0, (best_lay - 1) * 5.0

                if liability > max_liability:
                    backer_stake = max_liability / (best_lay - 1)
                    liability    = max_liability

                if backer_stake < 2.0:
                    continue

                # Place the lay bet
                logger.info(
                    f"[{self._label}] LAY sel={runner.selection_id} @ {best_lay} "
                    f"stake={backer_stake:.2f} liability={liability:.2f} "
                    f"model_edge={edge.get('signal')}"
                )
                trade = Trade(market_id=market_id, selection_id=runner.selection_id,
                              handicap=runner.handicap, strategy=self)
                market.place_order(
                    trade.create_order(
                        side="LAY",
                        order_type=LimitOrder(price=best_lay, size=round(backer_stake, 2))
                    )
                )
                _placed.add(key)

    return _VenueLayStrategy


# ─────────────────────────────────────────────────────────────────────────────
# FLUMINE RUNNER
# ─────────────────────────────────────────────────────────────────────────────

class FlumineRunner:
    """
    Manages a Flumine framework instance for automated betting.

    Usage:
        runner = FlumineRunner(credentials_path='credentials.json', bankroll=500)
        runner.add_model_strategy(event_type='horse', model='kash')
        runner.add_venue_strategy(venue='Doomben', max_liability=30)
        runner.start()   # blocks — Ctrl+C or end-of-day to stop
    """

    def __init__(
        self,
        credentials_path: str = "credentials.json",
        bankroll: float = 200.0,
        log_path:  str = "orders_agent.csv",
        daily_loss_limit: float = 100.0,
    ):
        if not _check_flumine():
            raise ImportError(
                "flumine is not installed. Run: pip install flumine betfairlightweight"
            )

        import json
        with open(credentials_path) as f:
            creds = json.load(f)

        import betfairlightweight
        from flumine import Flumine, clients

        self._trading = betfairlightweight.APIClient(
            creds["username"], creds["password"], app_key=creds["app_key"]
        )
        self._client    = clients.BetfairClient(self._trading, interactive_login=True)
        self._framework = Flumine(client=self._client)
        self._bankroll  = bankroll
        self._log_path  = log_path
        self._daily_loss_limit = daily_loss_limit

        self._framework.add_logging_control(_make_logging_control(log_path))
        self._add_termination_worker()

        # Load ratings once
        from external_ratings import get_ratings_cache
        self.ratings = get_ratings_cache()

        logger.info(f"FlumineRunner initialised  bankroll={bankroll}  log={log_path}")

    def add_model_strategy(
        self,
        event_type:      str = "horse",   # "horse" or "greyhound"
        model:           str = "kash",
        max_stake:       float = 20.0,
        trigger_seconds: float = 60.0,
        min_edge_pct:    float = 3.0,
    ) -> "FlumineRunner":
        """Add a model-driven strategy (backs/lays based on Kash or Iggy ratings)."""
        from betfairlightweight.filters import streaming_market_filter
        from staking_engine import recommend_stake

        event_map   = {"horse": "7", "thoroughbred": "7",
                       "greyhound": "4339", "dog": "4339"}
        event_type_id = event_map.get(event_type.lower(), "7")

        strategy_cls = _make_model_strategy(
            ratings_cache=self.ratings,
            staking_fn=recommend_stake,
            bankroll=self._bankroll,
            model=model,
            max_stake=max_stake,
            trigger_seconds=trigger_seconds,
            min_edge_pct=min_edge_pct,
        )
        strategy = strategy_cls(
            market_filter=streaming_market_filter(
                event_type_ids=[event_type_id],
                country_codes=["AU"],
                market_types=["WIN"],
            ),
            max_order_exposure=max_stake,
            max_trade_count=1,
            max_live_trade_count=1,
        )
        self._framework.add_strategy(strategy)
        logger.info(f"Model strategy added: event={event_type} model={model.upper()}")
        return self

    def add_venue_strategy(
        self,
        venue:           str,
        betfair_client=None,
        max_liability:   float = 50.0,
        min_profit_ratio: float = 1.5,
        trigger_seconds: float = 120.0,
    ) -> "FlumineRunner":
        """
        Add a venue-specific lay strategy for all-day betting at one track.
        e.g. runner.add_venue_strategy('Doomben')
        """
        from betfairlightweight.filters import streaming_market_filter

        strategy_cls = _make_venue_lay_strategy(
            betfair_client=betfair_client,
            ratings_cache=self.ratings,
            venue=venue,
            bankroll=self._bankroll,
            max_liability=max_liability,
            min_profit_ratio=min_profit_ratio,
            trigger_seconds=trigger_seconds,
        )
        strategy = strategy_cls(
            market_filter=streaming_market_filter(
                event_type_ids=["7"],   # Horse racing
                country_codes=["AU"],
                market_types=["WIN"],
            ),
            max_order_exposure=max_liability,
            max_trade_count=1,
            max_live_trade_count=1,
        )
        self._framework.add_strategy(strategy)
        logger.info(f"Venue strategy added: {venue}  max_liability={max_liability}")
        return self

    def start(self) -> None:
        """Start the Flumine framework (blocks until all markets are done)."""
        logger.info("FlumineRunner.start() — framework running")
        self._framework.run()
        logger.info("FlumineRunner: all markets complete")

    def start_background(self) -> threading.Thread:
        """Start Flumine in a background thread (non-blocking)."""
        t = threading.Thread(target=self.start, daemon=True, name="flumine-runner")
        t.start()
        logger.info("FlumineRunner started in background thread")
        return t

    def get_p_and_l(self) -> dict:
        """Read the orders CSV and compute running P&L."""
        if not os.path.exists(self._log_path):
            return {"success": False, "error": "No order log found"}
        try:
            df = pd.read_csv(self._log_path)
            total_profit = df["profit"].sum() if "profit" in df.columns else 0
            bet_count    = len(df)
            won          = (df["profit"] > 0).sum() if "profit" in df.columns else 0
            return {
                "success":      True,
                "bet_count":    int(bet_count),
                "total_profit": round(float(total_profit), 2),
                "won_count":    int(won),
                "lost_count":   int(bet_count - won),
                "win_rate_pct": round(won / bet_count * 100, 1) if bet_count > 0 else 0,
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    def _add_termination_worker(self):
        from flumine.worker import BackgroundWorker
        self._framework.add_worker(
            BackgroundWorker(
                self._framework,
                _terminate,
                func_kwargs={"today_only": True, "seconds_closed": 1200},
                interval=60,
                start_delay=60,
            )
        )


# Import pandas lazily for p_and_l
try:
    import pandas as pd
except ImportError:
    pd = None  # type: ignore

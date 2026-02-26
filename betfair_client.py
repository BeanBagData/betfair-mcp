"""
Betfair API Client - handles all communication with Betfair's REST API.
Uses interactive login (no certificates required).

Endpoint note from Betfair docs:
  - Global endpoints (api.betfair.com) work for AU accounts.
  - NZ customers MUST use .com.au endpoints.
  - If the global endpoint returns 400/403, we fall back to .com.au automatically.
"""

import json
import os
import logging
import requests
from datetime import datetime, timezone
from typing import Optional

logger = logging.getLogger(__name__)

# Primary endpoints (global — work for AU/NZ)
BETTING_ENDPOINT  = "https://api.betfair.com/exchange/betting/rest/v1.0"
ACCOUNT_ENDPOINT  = "https://api.betfair.com/exchange/account/rest/v1.0"

# Fallback AU endpoints
BETTING_ENDPOINT_AU = "https://api.betfair.com.au/exchange/betting/rest/v1.0"
ACCOUNT_ENDPOINT_AU = "https://api.betfair.com.au/exchange/account/rest/v1.0"

# Interactive login (no certs needed)
INTERACTIVE_LOGIN_ENDPOINT = "https://identitysso.betfair.com.au/api/login"


class BetfairClient:
    def __init__(self, credentials_path: str = "credentials.json"):
        self.session_token: Optional[str] = None
        self.app_key: Optional[str] = None
        self.username: Optional[str] = None
        self.password: Optional[str] = None
        self.credentials_path = credentials_path
        self._load_credentials()

    def _load_credentials(self):
        """Load credentials from JSON file."""
        try:
            with open(self.credentials_path, "r") as f:
                creds = json.load(f)
            self.username = creds["username"]
            self.password = creds["password"]
            self.app_key = creds["app_key"]
            logger.info(f"Credentials loaded for user: {self.username}")
        except FileNotFoundError:
            raise FileNotFoundError(
                f"credentials.json not found at '{self.credentials_path}'. "
                "Please create it with: {\"username\": \"...\", \"password\": \"...\", \"app_key\": \"...\"}"
            )
        except KeyError as e:
            raise ValueError(f"Missing key in credentials.json: {e}")

    def login(self) -> dict:
        """Login to Betfair using interactive login (no certificates)."""
        payload = {
            "username": self.username,
            "password": self.password,
        }
        headers = {
            "X-Application": self.app_key,
            "Content-Type": "application/x-www-form-urlencoded",
            "Accept": "application/json",
        }
        try:
            response = requests.post(
                INTERACTIVE_LOGIN_ENDPOINT,
                data=payload,
                headers=headers,
                timeout=30,
            )
            response.raise_for_status()
            result = response.json()

            if result.get("status") == "SUCCESS":
                self.session_token = result["token"]
                logger.info("Successfully logged into Betfair API")
                return {"success": True, "message": "Logged in successfully", "token": self.session_token[:10] + "..."}
            else:
                error = result.get("error", "UNKNOWN_ERROR")
                logger.error(f"Login failed: {error}")
                return {"success": False, "error": error, "message": self._explain_error(error)}
        except requests.RequestException as e:
            logger.error(f"Network error during login: {e}")
            return {"success": False, "error": "NETWORK_ERROR", "message": str(e)}

    def _explain_error(self, error_code: str) -> str:
        """Human-readable explanation of Betfair error codes."""
        explanations = {
            "ACCOUNT_ALREADY_LOCKED": "Account is locked. Contact Betfair customer service.",
            "ACCOUNT_NOW_LOCKED": "Account has been locked. Contact Betfair customer service.",
            "ACCOUNT_PENDING_PASSWORD_CHANGE": "You need to reset your password on the Betfair desktop site.",
            "ACTIONS_REQUIRED": "Login to Betfair desktop site and process any pop-ups.",
            "BETTING_RESTRICTED_LOCATION": "Betfair is not licensed in your current location.",
            "CERT_AUTH_REQUIRED": "Certificate authentication required. Check your certificates.",
            "CLOSED": "Account is closed. Contact Betfair customer service.",
            "EMAIL_LOGIN_NOT_ALLOWED": "Use your username (not email address) to login.",
            "INVALID_USERNAME_OR_PASSWORD": "Check your username/password. Watch for special characters.",
            "SELF_EXCLUDED": "Account is self-excluded.",
            "SUSPENDED": "Account is suspended. Contact Betfair customer service.",
            "TEMPORARY_BAN_TOO_MANY_REQUESTS": "Too many login attempts. Wait 20 minutes before trying again.",
        }
        return explanations.get(error_code, f"Unknown error: {error_code}")

    def _get_headers(self) -> dict:
        """Get standard request headers with session token."""
        if not self.session_token:
            raise RuntimeError("Not logged in. Call login() first.")
        return {
            "X-Application":   self.app_key,
            "X-Authentication": self.session_token,
            "Content-Type":    "application/json",
            "Accept":          "application/json",
        }

    def _post(self, url: str, params: dict) -> dict:
        """
        Internal POST. Returns {"success": True, "data": ...} on 200,
        or {"success": False, "error": ..., "error_body": ..., "status_code": ...}.
        Always captures the response body on errors for clear diagnostics.
        """
        try:
            response = requests.post(
                url,
                headers=self._get_headers(),
                json=params,
                timeout=30,
            )
            if response.status_code == 200:
                return {"success": True, "data": response.json()}
            # Capture error body — the Betfair API always returns useful JSON errors
            try:
                error_body = response.json()
            except Exception:
                error_body = response.text[:500]
            logger.warning(f"Betfair API {response.status_code} at {url}: {error_body}")
            return {
                "success":     False,
                "error":       f"{response.status_code} {response.reason}",
                "error_body":  error_body,
                "status_code": response.status_code,
            }
        except requests.RequestException as e:
            return {"success": False, "error": str(e)}

    def _betting_request(self, endpoint: str, params: dict) -> dict:
        """
        POST to the Betfair Betting API.
        Tries the global endpoint first; falls back to .com.au on 4xx.
        """
        result = self._post(f"{BETTING_ENDPOINT}/{endpoint}/", params)
        if not result["success"] and result.get("status_code") in (400, 403, 404):
            logger.info(f"Retrying {endpoint} on AU endpoint")
            result = self._post(f"{BETTING_ENDPOINT_AU}/{endpoint}/", params)
        return result

    def _account_request(self, endpoint: str, params: dict) -> dict:
        """
        POST to the Betfair Account API.
        Tries the global endpoint first; falls back to .com.au on 4xx.
        """
        result = self._post(f"{ACCOUNT_ENDPOINT}/{endpoint}/", params)
        if not result["success"] and result.get("status_code") in (400, 403, 404):
            logger.info(f"Retrying {endpoint} on AU account endpoint")
            result = self._post(f"{ACCOUNT_ENDPOINT_AU}/{endpoint}/", params)
        return result

    # ─────────────────────────────────────────────────────────────
    # ACCOUNT OPERATIONS
    # ─────────────────────────────────────────────────────────────

    def get_account_funds(self) -> dict:
        """
        Get account balance and available funds.

        Note: 'wallet' parameter is NOT sent — it causes 400 on AU accounts.
        The Betfair docs list it as optional and AU accounts don't use wallet
        segmentation, so an empty params dict is correct.
        """
        result = self._account_request("getAccountFunds", {})
        if result["success"]:
            data = result["data"]
            available = data.get("availableToBetBalance", 0)
            balance   = data.get("balance", 0)
            exposure  = data.get("exposure", 0)
            return {
                "success":            True,
                "available_to_bet":   available,
                "balance":            balance,
                "exposure":           exposure,
                "retained_commission": data.get("retainedCommission", 0),
                "summary": (
                    f"Balance: ${balance:.2f} | "
                    f"Available to bet: ${available:.2f} | "
                    f"Exposure: ${abs(exposure):.2f}"
                ),
            }
        # Enrich the error message
        err = result.get("error", "Unknown error")
        body = result.get("error_body", "")
        return {
            "success": False,
            "error":   err,
            "detail":  body,
            "message": (
                f"Could not retrieve account balance: {err}. "
                "If you just logged in this may be a session issue — try 'login' again."
            ),
        }

    def get_account_details(self) -> dict:
        """Get account details including currency, timezone and account status."""
        result = self._account_request("getAccountDetails", {})
        if result["success"]:
            data = result["data"]
            return {
                "success":      True,
                "first_name":   data.get("firstName", ""),
                "last_name":    data.get("lastName", ""),
                "currency_code": data.get("currencyCode", "AUD"),
                "locale":       data.get("locale", ""),
                "timezone":     data.get("timezone", ""),
                "discount_rate": data.get("discountRate", 0),
                "points_balance": data.get("pointsBalance", 0),
            }
        return result

    # ─────────────────────────────────────────────────────────────
    # MARKET DISCOVERY
    # ─────────────────────────────────────────────────────────────

    def search_horse_racing_markets(self, horse_name: str, hours_ahead: int = 24) -> dict:
        """
        Search for Australian horse racing markets containing the specified horse.
        Returns WIN markets where the horse is listed as a runner.
        """
        from datetime import timedelta
        now = datetime.now(timezone.utc)
        end_time = now + timedelta(hours=hours_ahead)

        # Search WIN markets for Australian thoroughbred racing.
        # NOTE: textQuery searches market/event names, NOT runner names — so we
        # intentionally omit it here and filter runners by name ourselves below.
        params = {
            "filter": {
                "eventTypeIds": ["7"],  # Horse Racing
                "marketCountries": ["AU"],
                "marketTypes": ["WIN"],
                "marketStartTime": {
                    "from": now.strftime("%Y-%m-%dT%H:%M:%SZ"),
                    "to": end_time.strftime("%Y-%m-%dT%H:%M:%SZ"),
                },
            },
            "marketProjection": ["RUNNER_DESCRIPTION", "EVENT", "MARKET_START_TIME", "RUNNER_METADATA"],
            "maxResults": "200",  # Fetch all AU markets in the window; filter runners below
            "sort": "FIRST_TO_START",
        }

        result = self._betting_request("listMarketCatalogue", params)
        if not result["success"]:
            return result

        markets = result["data"]
        found = []
        for market in markets:
            runners = market.get("runners", [])
            matching = [
                r for r in runners
                if horse_name.lower() in r.get("runnerName", "").lower()
            ]
            if matching:
                found.append({
                    "market_id": market["marketId"],
                    "market_name": market["marketName"],
                    "event_name": market.get("event", {}).get("name", ""),
                    "venue": market.get("event", {}).get("venue", ""),
                    "start_time": market.get("marketStartTime", ""),
                    "total_matched": market.get("totalMatched", 0),
                    "matching_runners": [
                        {
                            "selection_id": r["selectionId"],
                            "runner_name": r["runnerName"],
                            "sort_priority": r.get("sortPriority", 99),
                        }
                        for r in matching
                    ],
                })

        return {
            "success": True,
            "horse_name": horse_name,
            "markets_found": len(found),
            "markets": found,
        }

    def get_market_book(self, market_id: str, runner_name_map: Optional[dict] = None) -> dict:
        """
        Get the current market book (prices/liquidity) for a market.
        Returns best back and lay prices for all runners.

        Args:
            market_id: The Betfair market ID
            runner_name_map: Optional dict of {selection_id: runner_name} to add
                             horse names to the response. If omitted, the method
                             fetches runner names from listMarketCatalogue automatically.
        """
        # If no name map provided, fetch runner names via listMarketCatalogue
        if runner_name_map is None:
            cat_params = {
                "filter": {"marketIds": [market_id]},
                "marketProjection": ["RUNNER_DESCRIPTION"],
                "maxResults": "1",
            }
            cat_result = self._betting_request("listMarketCatalogue", cat_params)
            if cat_result["success"] and cat_result["data"]:
                runners_meta = cat_result["data"][0].get("runners", [])
                runner_name_map = {r["selectionId"]: r.get("runnerName", "Unknown") for r in runners_meta}
            else:
                runner_name_map = {}

        params = {
            "marketIds": [market_id],
            "priceProjection": {
                "priceData": ["EX_BEST_OFFERS", "EX_TRADED"],
                "exBestOffersOverrides": {"bestPricesDepth": 3},
            },
            "orderProjection": "ALL",
            "matchProjection": "NO_ROLLUP",
        }

        result = self._betting_request("listMarketBook", params)
        if not result["success"]:
            return result

        books = result["data"]
        if not books:
            return {"success": False, "error": "No market book data returned"}

        book = books[0]
        runners = []
        for runner in book.get("runners", []):
            back_prices = runner.get("ex", {}).get("availableToBack", [])
            lay_prices = runner.get("ex", {}).get("availableToLay", [])

            best_back = back_prices[0] if back_prices else None
            best_lay = lay_prices[0] if lay_prices else None

            # Calculate lay profitability
            lay_analysis = None
            if best_lay:
                lay_price = best_lay["price"]
                lay_size_available = best_lay["size"]
                # If you lay at this price:
                # - Win (horse loses): profit = your_stake (backer's money)
                # - Loss (horse wins): liability = (lay_price - 1) * your_stake
                # Profit ratio = your_stake / liability = 1 / (lay_price - 1)
                if lay_price > 1:
                    profit_ratio = 1 / (lay_price - 1)
                    lay_analysis = {
                        "lay_price": lay_price,
                        "available_size": lay_size_available,
                        "profit_ratio": round(profit_ratio, 4),
                        "meets_1_5x_threshold": profit_ratio >= 1.5,
                        "implied_probability_pct": round((1 / lay_price) * 100, 2),
                        # For a £10 lay stake: win = £10, liability = (price-1)*10
                        "example_10_stake": {
                            "potential_profit": 10.0,
                            "liability": round((lay_price - 1) * 10, 2),
                            "profit_if_loses": 10.0,
                        },
                    }

            runners.append({
                "selection_id": runner["selectionId"],
                "runner_name": runner_name_map.get(runner["selectionId"], "Unknown"),
                "status": runner.get("status", "UNKNOWN"),
                "last_price_traded": runner.get("lastPriceTraded"),
                "total_matched": runner.get("totalMatched", 0),
                "best_back": best_back,
                "best_lay": best_lay,
                "lay_analysis": lay_analysis,
                "back_prices": back_prices[:3],
                "lay_prices": lay_prices[:3],
            })

        return {
            "success": True,
            "market_id": market_id,
            "status": book.get("status", "UNKNOWN"),
            "inplay": book.get("inplay", False),
            "total_matched": book.get("totalMatched", 0),
            "runners": runners,
        }

    def get_market_depth(self, market_id: str, depth: int = 10) -> dict:
        """
        Fetch the full price ladder (up to *depth* rungs) for all runners.
        Used for weight-of-money and market-spread analysis.

        Unlike get_market_book (3-rung depth), this returns the complete
        available-to-back and available-to-lay ladders needed by the
        market_analyser module.

        Args:
            market_id: The Betfair market ID
            depth:     How many ladder rungs to fetch (max 10)
        """
        # Fetch runner names first
        cat_params = {
            "filter": {"marketIds": [market_id]},
            "marketProjection": ["RUNNER_DESCRIPTION"],
            "maxResults": "1",
        }
        cat_result = self._betting_request("listMarketCatalogue", cat_params)
        runner_name_map = {}
        if cat_result["success"] and cat_result["data"]:
            runners_meta = cat_result["data"][0].get("runners", [])
            runner_name_map = {r["selectionId"]: r.get("runnerName", "Unknown")
                               for r in runners_meta}

        params = {
            "marketIds": [market_id],
            "priceProjection": {
                "priceData": ["EX_BEST_OFFERS", "EX_TRADED"],
                "exBestOffersOverrides": {
                    "bestPricesDepth": min(depth, 10),
                    "rollupModel": "STAKE",
                    "rollupLimit": 0,
                },
            },
            "orderProjection": "ALL",
            "matchProjection": "NO_ROLLUP",
        }

        result = self._betting_request("listMarketBook", params)
        if not result["success"]:
            return result

        books = result["data"]
        if not books:
            return {"success": False, "error": "No market book data returned"}

        book = books[0]
        runners = []
        for runner in book.get("runners", []):
            back_prices = runner.get("ex", {}).get("availableToBack", [])
            lay_prices  = runner.get("ex", {}).get("availableToLay", [])
            traded_vol  = runner.get("ex", {}).get("tradedVolume", [])

            runners.append({
                "selection_id":  runner["selectionId"],
                "runner_name":   runner_name_map.get(runner["selectionId"], "Unknown"),
                "status":        runner.get("status", "UNKNOWN"),
                "last_price_traded": runner.get("lastPriceTraded"),
                "total_matched":     runner.get("totalMatched", 0),
                "back_prices":   back_prices,   # full depth list of {price, size}
                "lay_prices":    lay_prices,
                "traded_volume": traded_vol,    # list of {price, size} traded rungs
            })

        return {
            "success":       True,
            "market_id":     market_id,
            "status":        book.get("status", "UNKNOWN"),
            "inplay":        book.get("inplay", False),
            "total_matched": book.get("totalMatched", 0),
            "runners":       runners,
        }

    def get_sp_predictions(self, market_id: str) -> dict:
        """
        Fetch Betfair Starting Price (BSP) near and far estimates for all runners.

        Discovered in the stream parser (Document 5):
          sp_near = BSP estimate early in pre-race trading
          sp_far  = BSP estimate later in pre-race trading (more accurate, closer to jump)

        These are Betfair's own model-generated expected BSP values, available
        before the race starts.  They are invaluable for lay betting because:

          - If current LAY price > sp_far  → you are laying above expected BSP → VALUE
          - If current LAY price ≈ sp_far  → fair price, thin or no edge
          - If current LAY price < sp_far  → laying below expected BSP → AVOID

        The sp_near/far values update throughout pre-race trading and converge
        to the actual BSP as the market approaches the jump.

        Returns runner-level SP estimates alongside current best prices so the
        staking_engine.estimate_edge_from_sp() function can compute a data-driven
        edge estimate without the user needing to guess their win probability.
        """
        # Fetch runner names
        cat_params = {
            "filter": {"marketIds": [market_id]},
            "marketProjection": ["RUNNER_DESCRIPTION"],
            "maxResults": "1",
        }
        cat_result = self._betting_request("listMarketCatalogue", cat_params)
        runner_name_map = {}
        if cat_result["success"] and cat_result["data"]:
            for r in cat_result["data"][0].get("runners", []):
                runner_name_map[r["selectionId"]] = r.get("runnerName", "Unknown")

        params = {
            "marketIds": [market_id],
            "priceProjection": {
                "priceData": ["EX_BEST_OFFERS", "SP_TRADED"],
                "exBestOffersOverrides": {"bestPricesDepth": 3},
            },
        }

        result = self._betting_request("listMarketBook", params)
        if not result["success"]:
            return result

        books = result["data"]
        if not books:
            return {"success": False, "error": "No market book data returned"}

        book = books[0]
        runners_out = []

        for runner in book.get("runners", []):
            sel_id    = runner["selectionId"]
            sp_data   = runner.get("sp", {})
            sp_near   = sp_data.get("nearPrice")
            sp_far    = sp_data.get("farPrice")

            back_prices = runner.get("ex", {}).get("availableToBack", [])
            lay_prices  = runner.get("ex", {}).get("availableToLay",  [])
            best_lay    = lay_prices[0]["price"]  if lay_prices  else None
            best_back   = back_prices[0]["price"] if back_prices else None

            # Edge analysis using sp_far (or sp_near as fallback)
            edge_analysis = None
            if best_lay and (sp_near or sp_far):
                from staking_engine import estimate_edge_from_sp
                edge_analysis = estimate_edge_from_sp(best_lay, sp_near, sp_far)

            runners_out.append({
                "selection_id":   sel_id,
                "runner_name":    runner_name_map.get(sel_id, "Unknown"),
                "status":         runner.get("status", "UNKNOWN"),
                "best_lay":       best_lay,
                "best_back":      best_back,
                "sp_near":        sp_near,
                "sp_far":         sp_far,
                "last_price_traded": runner.get("lastPriceTraded"),
                "total_matched":  runner.get("totalMatched", 0),
                "edge_analysis":  edge_analysis,
            })

        # Sort favourites first
        runners_out.sort(key=lambda r: r.get("best_lay") or 9999)

        return {
            "success":        True,
            "market_id":      market_id,
            "status":         book.get("status", "UNKNOWN"),
            "inplay":         book.get("inplay", False),
            "total_matched":  book.get("totalMatched", 0),
            "runners":        runners_out,
            "note": (
                "sp_near/sp_far are Betfair's own pre-race BSP model estimates. "
                "Lay prices above sp_far represent positive expected value. "
                "Use staking_engine.estimate_edge_from_sp() for full analysis."
            ),
        }

    def poll_market_for_steam(
        self,
        market_id: str,
        polls: int = 12,
        interval_seconds: float = 5.0,
    ) -> dict:
        """
        Poll a market repeatedly and feed data into a SteamDetector.
        Returns steam/drift signals detected during the observation window.

        Args:
            market_id:        Betfair market ID to watch
            polls:            How many times to fetch the market book
            interval_seconds: Seconds between polls

        Total watch time ≈ polls × interval_seconds (default: 60 seconds)
        """
        import time as _time
        from market_analyser import SteamDetector, analyse_market

        detector = SteamDetector(market_id=market_id)
        errors   = []

        logger.info(f"Steam watch: {market_id} — {polls} polls × {interval_seconds}s")

        for i in range(polls):
            try:
                book = self.get_market_book(market_id)
                if book.get("success"):
                    detector.update(book)
                    if book.get("inplay"):
                        errors.append("Market went in-play during monitoring")
                        break
                else:
                    errors.append(f"Poll {i+1}: {book.get('error', 'failed')}")
            except Exception as e:
                errors.append(f"Poll {i+1}: {e}")

            if i < polls - 1:
                _time.sleep(interval_seconds)

        signals  = detector.scan(window_seconds=polls * interval_seconds)
        snapshot = detector.snapshot()

        return {
            "success":            True,
            "market_id":          market_id,
            "observation_seconds": int(polls * interval_seconds),
            "polls_completed":    polls,
            "steam_signals":      signals,
            "signal_count":       len(signals),
            "runner_snapshot":    snapshot["runners"],
            "errors":             errors,
            "interpretation": (
                f"Monitored {market_id} for {int(polls * interval_seconds)}s. "
                f"Found {len(signals)} movement signal(s). "
                + ("See steam_signals for detail." if signals else "No notable price moves detected.")
            ),
        }

    # ─────────────────────────────────────────────────────────────
    # ORDER MANAGEMENT
    # ─────────────────────────────────────────────────────────────

    def place_lay_bet(
        self,
        market_id: str,
        selection_id: int,
        lay_price: float,
        stake: float,
        strategy_ref: str = "lay_agent",
    ) -> dict:
        """
        Place a lay bet on a selection.

        Args:
            market_id: The Betfair market ID (e.g., "1.150038686")
            selection_id: The runner's selection ID
            lay_price: The price to lay at
            stake: The backer's stake (amount you'll win if the horse loses)
            strategy_ref: Strategy reference for tracking (max 15 chars)

        Returns:
            Bet placement result with bet ID and status
        """
        # Validate price is a valid Betfair tick
        lay_price = self._round_to_nearest_tick(lay_price)

        params = {
            "marketId": market_id,
            "instructions": [
                {
                    "orderType": "LIMIT",
                    "selectionId": selection_id,
                    "side": "LAY",
                    "limitOrder": {
                        "size": round(stake, 2),
                        "price": lay_price,
                        "persistenceType": "LAPSE",
                    },
                }
            ],
            "customerStrategyRef": strategy_ref[:15],
        }

        result = self._betting_request("placeOrders", params)
        if not result["success"]:
            return result

        data = result["data"]
        if data.get("status") == "SUCCESS":
            report = data["instructionReports"][0]
            liability = round((lay_price - 1) * stake, 2)
            return {
                "success": True,
                "bet_id": report.get("betId"),
                "status": report.get("orderStatus"),
                "placed_date": report.get("placedDate"),
                "size_matched": report.get("sizeMatched", 0),
                "average_price_matched": report.get("averagePriceMatched", 0),
                "market_id": market_id,
                "selection_id": selection_id,
                "lay_price": lay_price,
                "stake": stake,
                "liability": liability,
                "potential_profit": stake,
                "message": f"Lay bet placed. Bet ID: {report.get('betId')}. "
                           f"Risk £{liability:.2f} to win £{stake:.2f}",
            }
        else:
            errors = [r.get("errorCode") for r in data.get("instructionReports", [])]
            return {
                "success": False,
                "error": data.get("status"),
                "instruction_errors": errors,
                "message": f"Bet placement failed: {errors}",
            }

    def cancel_order(self, market_id: str, bet_id: Optional[str] = None) -> dict:
        """Cancel an order. If bet_id is None, cancels all orders for the market."""
        params = {"marketId": market_id}
        if bet_id:
            params["instructions"] = [{"betId": bet_id}]

        result = self._betting_request("cancelOrders", params)
        if result["success"]:
            data = result["data"]
            reports = data.get("instructionReports", [])
            cancelled = sum(1 for r in reports if r.get("status") == "SUCCESS")
            return {
                "success": True,
                "cancelled_count": cancelled,
                "total_size_cancelled": sum(r.get("sizeCancelled", 0) for r in reports),
                "reports": reports,
            }
        return result

    def get_current_orders(self, strategy_ref: Optional[str] = None) -> dict:
        """Get all current (unmatched/partially matched) orders."""
        params = {}
        if strategy_ref:
            params["customerStrategyRefs"] = [strategy_ref]

        result = self._betting_request("listCurrentOrders", params)
        if not result["success"]:
            return result

        data = result["data"]
        orders = data.get("currentOrders", [])
        formatted = []
        for order in orders:
            price_size = order.get("priceSize", {})
            formatted.append({
                "bet_id": order.get("betId"),
                "market_id": order.get("marketId"),
                "selection_id": order.get("selectionId"),
                "side": order.get("side"),
                "order_type": order.get("orderType"),
                "status": order.get("status"),
                "price": price_size.get("price"),
                "size": price_size.get("size"),
                "size_matched": order.get("sizeMatched", 0),
                "size_remaining": order.get("sizeRemaining", 0),
                "size_cancelled": order.get("sizeCancelled", 0),
                "placed_date": order.get("placedDate"),
                "strategy_ref": order.get("customerStrategyRef"),
            })

        return {
            "success": True,
            "order_count": len(formatted),
            "orders": formatted,
        }

    def get_cleared_orders(self, days_back: int = 7) -> dict:
        """Get settled orders from the past N days."""
        from datetime import timedelta
        start = (datetime.now(timezone.utc) - timedelta(days=days_back)).strftime("%Y-%m-%dT%H:%M:%SZ")

        params = {
            "betStatus": "SETTLED",
            "settledDateRange": {"from": start},
        }

        result = self._betting_request("listClearedOrders", params)
        if not result["success"]:
            return result

        data = result["data"]
        orders = data.get("clearedOrders", [])
        total_pl = sum(o.get("profit", 0) for o in orders)

        return {
            "success": True,
            "order_count": len(orders),
            "total_profit_loss": round(total_pl, 2),
            "orders": [
                {
                    "bet_id": o.get("betId"),
                    "market_id": o.get("marketId"),
                    "selection_id": o.get("selectionId"),
                    "side": o.get("side"),
                    "outcome": o.get("betOutcome"),
                    "profit": o.get("profit", 0),
                    "price_matched": o.get("priceMatched"),
                    "size_settled": o.get("sizeSettled"),
                    "placed_date": o.get("placedDate"),
                    "settled_date": o.get("settledDate"),
                }
                for o in orders
            ],
        }

    # ─────────────────────────────────────────────────────────────
    # RACE METADATA (Betfair Hub API — no auth required)
    # ─────────────────────────────────────────────────────────────

    def get_race_metadata(self, market_id: str) -> dict:
        """
        Fetch rich race metadata from Betfair's public Hub API.
        Returns jockey, trainer, barrier, weight, track condition,
        race distance and finishing place for each non-scratched runner.
        """
        url = f"https://apigateway.betfair.com.au/hub/raceevent/{market_id}"
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            data = response.json()
        except requests.RequestException as e:
            logger.error(f"Race metadata fetch error for {market_id}: {e}")
            return {"success": False, "error": str(e), "market_id": market_id}

        if "error" in data:
            return {
                "success": False, "error": data["error"], "market_id": market_id,
                "note": "Market may not be available in Hub API (try closer to race time)",
            }

        def _barrier_group(barrier) -> str:
            try:
                b = int(barrier)
            except (TypeError, ValueError):
                return "unknown"
            return "inside" if b <= 3 else "mid_field" if b <= 8 else "outside"

        def _distance_group(distance) -> str:
            try:
                d = int(distance)
            except (TypeError, ValueError):
                return "unknown"
            if d < 1100: return "sprint"
            if d < 1400: return "mid_short"
            if d < 1800: return "mid_long"
            return "long"

        race_distance = data.get("raceLength")
        dist_group    = _distance_group(race_distance)
        runners_out   = []

        for runner in data.get("runners", []):
            if runner.get("isScratched"):
                continue
            try:
                place = int(runner.get("placedResult", 0) or 0)
            except (TypeError, ValueError):
                place = 0
            barrier = runner.get("barrierNo")
            runners_out.append({
                "selection_id":  runner.get("selectionId"),
                "runner_name":   runner.get("runnerName", ""),
                "barrier":       barrier,
                "barrier_group": _barrier_group(barrier),
                "jockey":        runner.get("jockeyName", "Unknown"),
                "trainer":       runner.get("trainerName", "Unknown"),
                "weight":        runner.get("weight"),
                "place":         place,
                "distance_group": dist_group,
            })

        return {
            "success":         True,
            "market_id":       market_id,
            "weather":         data.get("weather", "Unknown"),
            "track_condition": data.get("trackCondition", "Unknown"),
            "race_distance":   race_distance,
            "distance_group":  dist_group,
            "runners":         runners_out,
        }

    # ─────────────────────────────────────────────────────────────
    # VENUE & MULTI-MARKET DISCOVERY
    # ─────────────────────────────────────────────────────────────

    def list_venue_markets(
        self,
        venue:       str,
        hours_ahead: int = 12,
        event_type:  str = "horse",
    ) -> dict:
        """
        List all upcoming WIN markets at a specific venue today.
        Essential for 'make money at Doomben today' type requests.

        Args:
            venue:       Track/venue name (e.g. "Doomben", "Eagle Farm")
            hours_ahead: How far ahead to look (default 12 hours)
            event_type:  "horse" or "greyhound"
        """
        from datetime import timedelta
        now      = datetime.now(timezone.utc)
        end_time = now + timedelta(hours=hours_ahead)
        et_ids   = {"horse": ["7"], "thoroughbred": ["7"],
                    "greyhound": ["4339"], "dog": ["4339"]}.get(event_type.lower(), ["7"])

        params = {
            "filter": {
                "eventTypeIds": et_ids,
                "marketCountries": ["AU"],
                "marketTypes": ["WIN"],
                "venues": [venue],
                "marketStartTime": {
                    "from": now.strftime("%Y-%m-%dT%H:%M:%SZ"),
                    "to":   end_time.strftime("%Y-%m-%dT%H:%M:%SZ"),
                },
            },
            "marketProjection": ["RUNNER_DESCRIPTION", "EVENT", "MARKET_START_TIME"],
            "maxResults": "100",
            "sort": "FIRST_TO_START",
        }

        result = self._betting_request("listMarketCatalogue", params)
        if not result["success"]:
            return result

        markets_out = []
        for m in result["data"]:
            markets_out.append({
                "market_id":    m["marketId"],
                "market_name":  m.get("marketName", ""),
                "event_name":   m.get("event", {}).get("name", ""),
                "start_time":   m.get("marketStartTime", ""),
                "runner_count": len(m.get("runners", [])),
                "runners": [
                    {"selection_id": r["selectionId"],
                     "runner_name":  r.get("runnerName", ""),
                     "sort_priority": r.get("sortPriority", 99)}
                    for r in m.get("runners", [])
                ],
            })

        return {
            "success":      True,
            "venue":        venue,
            "event_type":   event_type,
            "market_count": len(markets_out),
            "markets":      markets_out,
        }

    def place_back_bet(
        self,
        market_id:    str,
        selection_id: int,
        back_price:   float,
        stake:        float,
        strategy_ref: str = "agent_back",
    ) -> dict:
        """
        Place a BACK bet. Used when market price > model price (horse over-valued
        by the market). Complements place_lay_bet for full model-driven execution.
        """
        back_price = self._round_to_nearest_tick(back_price)
        params = {
            "marketId": market_id,
            "instructions": [{
                "orderType":   "LIMIT",
                "selectionId": selection_id,
                "side":        "BACK",
                "limitOrder":  {
                    "size":            round(stake, 2),
                    "price":           back_price,
                    "persistenceType": "LAPSE",
                },
            }],
            "customerStrategyRef": strategy_ref[:15],
        }
        result = self._betting_request("placeOrders", params)
        if not result["success"]:
            return result
        data = result["data"]
        if data.get("status") == "SUCCESS":
            report           = data["instructionReports"][0]
            potential_profit = round((back_price - 1) * stake, 2)
            return {
                "success":              True,
                "bet_id":               report.get("betId"),
                "status":               report.get("orderStatus"),
                "placed_date":          report.get("placedDate"),
                "size_matched":         report.get("sizeMatched", 0),
                "average_price_matched": report.get("averagePriceMatched", 0),
                "market_id":            market_id,
                "selection_id":         selection_id,
                "back_price":           back_price,
                "stake":                stake,
                "potential_profit":     potential_profit,
                "message": f"Back bet placed. ID={report.get('betId')}. Stake ${stake:.2f}, profit if wins ${potential_profit:.2f}.",
            }
        errors = [r.get("errorCode") for r in data.get("instructionReports", [])]
        return {"success": False, "error": data.get("status"), "instruction_errors": errors}

    # ─────────────────────────────────────────────────────────────
    # ADVANCED ACCOUNT & EXPOSURE (Derived from betfairlightweight)
    # ─────────────────────────────────────────────────────────────

    def get_market_profit_and_loss(self, market_id: str) -> dict:
        """
        Retrieve profit and loss (exposure) for a given OPEN market.
        Crucial for Risk Management: tells the agent exactly what happens to the
        bankroll if a specific runner wins or loses.
        """
        params = {
            "marketIds": [market_id],
            "includeSettledBets": True,
            "includeBspBets": True,
            "netOfCommission": True
        }
        result = self._betting_request("listMarketProfitAndLoss", params)
        if not result["success"]:
            return result
            
        data = result["data"]
        if not data:
            return {"success": False, "error": "No P&L data returned"}
            
        market_pnl = data[0]
        formatted_runners =[]
        
        for runner in market_pnl.get("profitAndLosses",[]):
            formatted_runners.append({
                "selection_id": runner.get("selectionId"),
                "if_win": round(runner.get("ifWin", 0), 2),
                "if_lose": round(runner.get("ifLose", 0), 2),
                # Note: if_win < 0 means liability. if_win > 0 means profit.
            })
            
        return {
            "success": True,
            "market_id": market_id,
            "commission_applied": market_pnl.get("commissionApplied"),
            "exposure_by_runner": formatted_runners
        }

    def get_account_statement(self, record_count: int = 20) -> dict:
        """
        Get detailed account statement (deposits, withdrawals, settled bet transactions).
        """
        params = {
            "fromRecord": 0,
            "recordCount": record_count
        }
        result = self._account_request("getAccountStatement", params)
        if not result["success"]:
            return result
            
        data = result["data"]
        statements = []
        
        for item in data.get("accountStatement",[]):
            legacy = item.get("legacyData", {})
            statements.append({
                "date": item.get("itemDate"),
                "amount": round(item.get("amount", 0), 2),
                "running_balance": round(item.get("balance", 0), 2),
                "item_class": item.get("itemClass"),
                "details": {
                    "market": legacy.get("fullMarketName") or legacy.get("marketName"),
                    "selection": legacy.get("selectionName"),
                    "bet_type": legacy.get("betType"),
                    "avg_price": legacy.get("avgPrice"),
                    "bet_size": legacy.get("betSize"),
                    "win_lose": legacy.get("winLose")
                }
            })
            
        return {
            "success": True,
            "more_available": data.get("moreAvailable", False),
            "transactions": statements
        }


    # ─────────────────────────────────────────────────────────────
    # HELPERS
    # ─────────────────────────────────────────────────────────────

    def _round_to_nearest_tick(self, price: float) -> float:
        """
        Round a price to the nearest valid Betfair tick.
        Betfair tick increments:
        1.0 - 2.0: 0.01
        2.0 - 3.0: 0.02
        3.0 - 4.0: 0.05
        4.0 - 6.0: 0.1
        6.0 - 10.0: 0.2
        10.0 - 20.0: 0.5
        20.0 - 30.0: 1.0
        30.0 - 50.0: 2.0
        50.0 - 100.0: 5.0
        100.0+: 10.0
        """
        if price < 2.0:
            return round(round(price / 0.01) * 0.01, 2)
        elif price < 3.0:
            return round(round(price / 0.02) * 0.02, 2)
        elif price < 4.0:
            return round(round(price / 0.05) * 0.05, 2)
        elif price < 6.0:
            return round(round(price / 0.1) * 0.1, 2)
        elif price < 10.0:
            return round(round(price / 0.2) * 0.2, 2)
        elif price < 20.0:
            return round(round(price / 0.5) * 0.5, 2)
        elif price < 30.0:
            return round(round(price / 1.0) * 1.0, 2)
        elif price < 50.0:
            return round(round(price / 2.0) * 2.0, 2)
        elif price < 100.0:
            return round(round(price / 5.0) * 5.0, 2)
        else:
            return round(round(price / 10.0) * 10.0, 2)

# End of betfair_client.py
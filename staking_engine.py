"""
Staking Engine
==============
Implements and compares four staking strategies for Betfair lay betting:

  1. Fixed Stakes         – constant liability per bet
  2. Proportional A       – stake a fixed % of dynamic bankroll
  3. Proportional B       – stake to WIN a fixed % of dynamic bankroll
  4. Kelly                – mathematically optimal fraction of bankroll
  5. Martingale           – chase losses until target profit achieved (risky)

Core public API
---------------
  kelly_lay_stake(bankroll, win_prob, lay_price, partial=0.5) → float
  fixed_lay_stake(liability_amount) → float
  proportional_a_stake(bankroll, pct=0.02, lay_price=...) → float
  proportional_b_stake(bankroll, win_pct=0.05, lay_price=...) → float
  recommend_stake(bankroll, win_prob, lay_price, method, ...) → dict
  compare_staking_methods(bankroll, win_prob, lay_price, ...) → dict
  estimate_edge_from_sp(lay_price, sp_near, sp_far) → dict
  run_simulation(method, params, n_sims=1000) → dict

All monetary values are in the same currency as the bankroll (AUD/GBP).
"""

from __future__ import annotations

import math
import random
import logging
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# 1.  KELLY CRITERION FOR LAY BETTING
# ─────────────────────────────────────────────────────────────────────────────

def kelly_fraction(win_prob: float, lay_price: float) -> float:
    """
    Kelly fraction for a LAY bet (expressed as a fraction of bankroll to
    risk as liability).

    For a lay bet:
      - You WIN (horse loses) with probability p_lose = 1 - win_prob
        → profit = backer_stake
      - You LOSE (horse wins) with probability p_win = win_prob
        → loss   = (lay_price - 1) * backer_stake  [your liability]

    Treating the lay as a "back on the horse losing" with effective decimal
    odds of lay_price / 1 (risk 1 unit to profit lay_price-1 / lay_price-1):

    Standard Kelly: f* = (b*p - q) / b
      where b  = net odds = 1 / (lay_price - 1)   (win $1 for every $(lay_price-1) risked)
            p  = p_lose = 1 - win_prob
            q  = p_win  = win_prob

    f* expressed as a fraction of BANKROLL to stake as backer's stake.
    Multiply by (lay_price-1) to convert to liability fraction.

    Returns the raw Kelly fraction (can be negative → no bet).
    """
    if lay_price <= 1.0:
        return 0.0

    p_lose = 1.0 - win_prob          # probability we win the lay
    p_win  = win_prob                 # probability we lose the lay
    b      = 1.0 / (lay_price - 1.0) # net odds per unit risked as liability

    # Kelly: f* (as fraction of bankroll staked as backer's stake)
    kelly = (b * p_lose - p_win) / b

    return kelly


def kelly_lay_stake(
    bankroll:  float,
    win_prob:  float,
    lay_price: float,
    partial:   float = 0.5,
    min_stake: float = 2.0,
    max_liability_pct: float = 0.15,
) -> dict:
    """
    Compute the optimal Kelly stake for a lay bet.

    Args:
        bankroll:          Current available balance
        win_prob:          Estimated probability the horse WINS (0-1)
        lay_price:         Current best lay price
        partial:           Fraction of full Kelly to bet (default: 0.5 = half-Kelly)
                           Half-Kelly is strongly recommended to reduce variance
        min_stake:         Minimum backer stake (default: $2)
        max_liability_pct: Hard cap: liability cannot exceed this % of bankroll

    Returns dict with backer_stake, liability, reasoning, and edge metrics.
    """
    if win_prob <= 0 or win_prob >= 1:
        return _no_bet("win_prob must be between 0 and 1 exclusive")
    if lay_price <= 1.0:
        return _no_bet("lay_price must be > 1.0")
    if bankroll <= 0:
        return _no_bet("bankroll must be positive")

    p_lose = 1.0 - win_prob
    implied_prob = 1.0 / lay_price

    # Edge = our estimated p_lose vs the market's implied p_lose
    market_p_lose = 1.0 - implied_prob
    edge = p_lose - market_p_lose

    raw_kelly = kelly_fraction(win_prob, lay_price)

    if raw_kelly <= 0:
        return _no_bet(
            f"Negative Kelly ({raw_kelly:.4f}): no edge at this price. "
            f"Market implies win_prob={implied_prob:.3f}, your estimate={win_prob:.3f}"
        )

    # Backer's stake as fraction of bankroll
    stake_frac   = raw_kelly * partial
    backer_stake = round(max(stake_frac * bankroll, min_stake), 2)
    liability    = round((lay_price - 1.0) * backer_stake, 2)

    # Hard liability cap
    max_liability = bankroll * max_liability_pct
    if liability > max_liability:
        liability    = round(max_liability, 2)
        backer_stake = round(liability / (lay_price - 1.0), 2)

    profit_ratio = round(backer_stake / liability, 4) if liability > 0 else 0

    return {
        "success":       True,
        "method":        f"Kelly ({int(partial*100)}%)",
        "backer_stake":  backer_stake,
        "liability":     liability,
        "lay_price":     lay_price,
        "bankroll":      bankroll,
        "edge":          round(edge, 4),
        "edge_pct":      f"{edge*100:.2f}%",
        "raw_kelly_frac": round(raw_kelly, 4),
        "partial_kelly":  partial,
        "estimated_win_prob":  win_prob,
        "implied_win_prob":    round(implied_prob, 4),
        "profit_ratio":        profit_ratio,
        "expected_value_per_bet": round(backer_stake * p_lose - liability * win_prob, 2),
        "note": (
            f"Partial Kelly ({int(partial*100)}%) recommended to manage variance. "
            f"Full Kelly would stake ${round(raw_kelly * bankroll, 2)} backer stake."
        ),
    }


# ─────────────────────────────────────────────────────────────────────────────
# 2.  OTHER STAKING METHODS
# ─────────────────────────────────────────────────────────────────────────────

def fixed_lay_stake(
    liability_amount: float,
    lay_price:        float,
    bankroll:         float,
    max_liability_pct: float = 0.10,
) -> dict:
    """
    Fixed liability staking: risk the same dollar amount each bet.

    Args:
        liability_amount: Dollar amount to risk (your loss if horse wins)
        lay_price:        Current best lay price
        bankroll:         Current available balance
    """
    if lay_price <= 1.0:
        return _no_bet("lay_price must be > 1.0")

    max_liability = bankroll * max_liability_pct
    capped = False
    if liability_amount > max_liability:
        liability_amount = round(max_liability, 2)
        capped = True

    backer_stake = round(liability_amount / (lay_price - 1.0), 2)
    return {
        "success":      True,
        "method":       "Fixed Stake",
        "backer_stake": backer_stake,
        "liability":    round(liability_amount, 2),
        "lay_price":    lay_price,
        "bankroll":     bankroll,
        "note": ("Liability capped at 10% of bankroll. " if capped else "") +
                "Simple and predictable but does not scale with bankroll growth.",
    }


def proportional_a_stake(
    bankroll:   float,
    stake_pct:  float,
    lay_price:  float,
    min_stake:  float = 2.0,
) -> dict:
    """
    Proportional A: stake a fixed % of current bankroll as LIABILITY each bet.
    Stakes grow as bankroll grows, shrink during drawdowns.

    Args:
        stake_pct: Fraction of bankroll to risk as liability (e.g. 0.02 = 2%)
    """
    if lay_price <= 1.0:
        return _no_bet("lay_price must be > 1.0")

    liability    = round(max(bankroll * stake_pct, min_stake), 2)
    backer_stake = round(liability / (lay_price - 1.0), 2)

    return {
        "success":      True,
        "method":       f"Proportional A ({stake_pct*100:.1f}% of bank as liability)",
        "backer_stake": backer_stake,
        "liability":    liability,
        "lay_price":    lay_price,
        "bankroll":     bankroll,
        "note": f"Risking {stake_pct*100:.1f}% of bankroll (${liability:.2f}) per bet. "
                "Automatically adjusts with bankroll size.",
    }


def proportional_b_stake(
    bankroll:  float,
    win_pct:   float,
    lay_price: float,
    min_stake: float = 2.0,
) -> dict:
    """
    Proportional B: stake to WIN a fixed % of current bankroll each bet.
    The backer stake = bankroll * win_pct, regardless of lay price.
    Liability scales with lay price.

    Args:
        win_pct: Fraction of bankroll you want to win if the horse loses
    """
    if lay_price <= 1.0:
        return _no_bet("lay_price must be > 1.0")

    backer_stake = round(max(bankroll * win_pct, min_stake), 2)
    liability    = round(backer_stake * (lay_price - 1.0), 2)

    return {
        "success":      True,
        "method":       f"Proportional B ({win_pct*100:.1f}% win target)",
        "backer_stake": backer_stake,
        "liability":    liability,
        "lay_price":    lay_price,
        "bankroll":     bankroll,
        "note": f"Targeting a win of ${backer_stake:.2f} ({win_pct*100:.1f}% of bank). "
                f"Liability is ${liability:.2f} at price {lay_price}. "
                "Liability grows quickly at high lay prices — use caution.",
    }


# ─────────────────────────────────────────────────────────────────────────────
# 3.  EDGE ESTIMATION FROM BSP NEAR / FAR PRICES
# ─────────────────────────────────────────────────────────────────────────────

def estimate_edge_from_sp(
    lay_price: float,
    sp_near:   Optional[float],
    sp_far:    Optional[float],
    commission: float = 0.05,
) -> dict:
    """
    Estimate lay edge by comparing the current lay price to Betfair's
    own pre-race BSP estimates (sp_near and sp_far).

    From the stream parser (Document 5):
      sp_near = BSP estimate early in trading
      sp_far  = BSP estimate later in trading (more accurate, closer to jump)

    If the current LAY price > expected BSP, you are laying at BETTER than
    BSP → positive edge.  If LAY price < BSP, you're getting worse than BSP.

    Key insight: the BSP is the "true" settled price after sharp money has
    fully corrected the market. Laying above BSP = value. Laying below = trap.

    Args:
        lay_price:  Current best available lay price
        sp_near:    BSP near estimate (early trading)
        sp_far:     BSP far estimate  (later trading, preferred)
        commission: Betfair commission rate (default 5%)

    Returns edge estimates and a BET / NO BET recommendation.
    """
    result: dict = {
        "lay_price":  lay_price,
        "sp_near":    sp_near,
        "sp_far":     sp_far,
        "commission": commission,
    }

    if sp_far is None and sp_near is None:
        result.update({"verdict": "UNKNOWN", "edge": None,
                       "note": "No BSP estimates available yet. Market may be too early."})
        return result

    # Prefer sp_far (closer to jump, more accurate)
    reference_sp = sp_far if sp_far is not None else sp_near
    reference_label = "sp_far" if sp_far is not None else "sp_near"

    # Implied win probabilities (after commission adjustment)
    # BSP: market prices include commission — effective BSP win prob is higher
    # Our lay: we pay commission on winnings = our effective win prob is lower
    market_win_prob = 1.0 / reference_sp
    our_lay_win_prob = 1.0 / lay_price    # the backer's implied win prob at our price

    # Net probability advantage for layer
    # We win if horse LOSES: p_lose = 1 - market_win_prob
    # Our lay is priced at: implied p_win_for_backer = 1/lay_price
    # Edge = difference between market's implied win prob and our priced win prob
    edge_raw = market_win_prob - our_lay_win_prob

    # After commission: our net winnings are reduced by commission
    # Effective edge accounting for commission
    edge_net = edge_raw - (commission * (1.0 - our_lay_win_prob))

    # Tick difference
    from market_analyser import tick_delta
    tick_diff = tick_delta(lay_price, reference_sp)

    # Win probability for Kelly input
    estimated_win_prob = market_win_prob  # use BSP as best win probability estimate

    if edge_net > 0.02:
        verdict = "STRONG_VALUE"
        note    = (f"Lay price {lay_price} is {tick_diff} ticks ABOVE {reference_label} "
                   f"({reference_sp}). Clear lay value — BSP expected to be lower. "
                   f"Use this win_prob={market_win_prob:.3f} for Kelly sizing.")
    elif edge_net > 0:
        verdict = "MARGINAL_VALUE"
        note    = (f"Lay price {lay_price} is slightly above {reference_label} ({reference_sp}). "
                   f"Small edge of {edge_net:.3f} after commission. Proceed cautiously.")
    elif edge_net > -0.02:
        verdict = "BREAKEVEN"
        note    = (f"Lay price {lay_price} is near {reference_label} ({reference_sp}). "
                   f"Near zero edge. Not worth the risk.")
    else:
        verdict = "NO_VALUE"
        note    = (f"Lay price {lay_price} is BELOW {reference_label} ({reference_sp}). "
                   f"You would be laying below BSP — negative expected value. Do not bet.")

    result.update({
        "verdict":              verdict,
        "reference_sp":         reference_sp,
        "reference_label":      reference_label,
        "market_win_prob":      round(market_win_prob,    4),
        "our_implied_win_prob": round(our_lay_win_prob,   4),
        "edge_raw":             round(edge_raw,           4),
        "edge_net":             round(edge_net,           4),
        "edge_pct":             f"{edge_net * 100:.2f}%",
        "tick_delta":           tick_diff,
        "estimated_win_prob_for_kelly": round(estimated_win_prob, 4),
        "note":                 note,
    })
    return result


# ─────────────────────────────────────────────────────────────────────────────
# 4.  COMPARISON AND RECOMMENDATION
# ─────────────────────────────────────────────────────────────────────────────

def compare_staking_methods(
    bankroll:         float,
    win_prob:         float,
    lay_price:        float,
    fixed_liability:  float = 20.0,
    prop_a_pct:       float = 0.02,
    prop_b_win_pct:   float = 0.03,
    partial_kelly:    float = 0.5,
) -> dict:
    """
    Run all four staking methods and return a side-by-side comparison.
    Useful for letting the agent explain the tradeoffs to the user.
    """
    methods = {
        "kelly":           kelly_lay_stake(bankroll, win_prob, lay_price, partial_kelly),
        "fixed":           fixed_lay_stake(fixed_liability, lay_price, bankroll),
        "proportional_a":  proportional_a_stake(bankroll, prop_a_pct, lay_price),
        "proportional_b":  proportional_b_stake(bankroll, prop_b_win_pct, lay_price),
    }

    # Build comparison table
    comparison = []
    for name, result in methods.items():
        if result.get("success"):
            comparison.append({
                "method":       result["method"],
                "backer_stake": result["backer_stake"],
                "liability":    result["liability"],
                "pct_of_bank":  round(result["liability"] / bankroll * 100, 2),
                "note":         result.get("note", ""),
            })

    return {
        "success":       True,
        "bankroll":      bankroll,
        "win_prob":      win_prob,
        "lay_price":     lay_price,
        "methods":       methods,
        "comparison":    comparison,
        "recommendation": _pick_recommendation(methods, bankroll, win_prob),
    }


def _pick_recommendation(methods: dict, bankroll: float, win_prob: float) -> str:
    """Choose and justify the best method given the situation."""
    kelly = methods.get("kelly", {})
    if not kelly.get("success"):
        return (
            "Kelly signals NO BET (negative edge). Do not bet at this price. "
            "Only proceed if you have a strong fundamental reason the BSP will be lower."
        )

    kelly_pct = kelly["liability"] / bankroll * 100
    if kelly_pct > 8:
        return (
            f"Full Kelly ({kelly_pct:.1f}% of bank) is aggressive. "
            "Use Half-Kelly or Proportional A (2% of bank) to reduce ruin risk. "
            "High variance at this Kelly size."
        )
    elif kelly_pct < 1:
        return (
            "Kelly stake is very small — edge is thin. "
            "Fixed stake or skip is appropriate. "
            "Only bet if confidence in edge is high."
        )
    else:
        return (
            f"Kelly ({kelly_pct:.1f}% of bank liability) looks reasonable. "
            "Half-Kelly is recommended for robustness. "
            "Proportional A (2%) is the conservative choice."
        )


def recommend_stake(
    bankroll:         float,
    win_prob:         float,
    lay_price:        float,
    method:           str   = "half_kelly",
    ruin_threshold:   float = 50.0,
    partial_kelly:    float = 0.5,
    fixed_liability:  float = 20.0,
    prop_a_pct:       float = 0.02,
    prop_b_win_pct:   float = 0.03,
) -> dict:
    """
    Recommend a single stake using the specified method.

    Args:
        method: "kelly" | "half_kelly" | "quarter_kelly" |
                "fixed" | "proportional_a" | "proportional_b"
        ruin_threshold: Warn if liability would put bankroll dangerously low
    """
    method = method.lower().replace("-", "_")

    if method in ("kelly", "full_kelly"):
        result = kelly_lay_stake(bankroll, win_prob, lay_price, partial=1.0)
    elif method in ("half_kelly", "default"):
        result = kelly_lay_stake(bankroll, win_prob, lay_price, partial=0.5)
    elif method == "quarter_kelly":
        result = kelly_lay_stake(bankroll, win_prob, lay_price, partial=0.25)
    elif method == "fixed":
        result = fixed_lay_stake(fixed_liability, lay_price, bankroll)
    elif method == "proportional_a":
        result = proportional_a_stake(bankroll, prop_a_pct, lay_price)
    elif method == "proportional_b":
        result = proportional_b_stake(bankroll, prop_b_win_pct, lay_price)
    else:
        result = kelly_lay_stake(bankroll, win_prob, lay_price, partial=0.5)

    if result.get("success"):
        remaining = bankroll - result["liability"]
        if remaining < ruin_threshold:
            result["warning"] = (
                f"⚠️ This bet would leave only ${remaining:.2f} in your bankroll "
                f"(below ruin threshold of ${ruin_threshold:.2f}). Consider reducing stake."
            )

    return result


# ─────────────────────────────────────────────────────────────────────────────
# 5.  MONTE CARLO SIMULATION
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class SimParams:
    bankroll:    float = 1000.0
    ruin:        float = 50.0
    min_odds:    float = 1.5
    max_odds:    float = 5.0
    edge:        float = 0.05
    bank_target: float = 4.0      # multiple of starting bankroll
    n_bets:      int   = 50_000
    n_sims:      int   = 1_000


def _simulate_one(params: SimParams, method: str, **kwargs) -> tuple[str, int]:
    """Run a single simulation path. Returns (outcome, bets_taken)."""
    bank     = params.bankroll
    bet_range = params.max_odds - params.min_odds

    for i in range(params.n_bets):
        lay_price  = round(random.uniform(0, 1) * bet_range + params.min_odds, 2)
        win_prob   = (1.0 + params.edge) / lay_price   # true win probability
        p_lose     = 1.0 - win_prob

        # Determine stake by method
        if method == "fixed":
            stake = kwargs.get("liability", 20.0)
        elif method == "proportional_a":
            pct   = kwargs.get("stake_pct", 0.02)
            stake = max(pct * bank, params.ruin)
        elif method == "proportional_b":
            wpct  = kwargs.get("win_pct", 0.03)
            backer = max(bank * wpct, params.ruin)
            stake = backer * (lay_price - 1.0)
        elif method in ("kelly", "half_kelly", "quarter_kelly"):
            partial = {"kelly": 1.0, "half_kelly": 0.5, "quarter_kelly": 0.25}[method]
            frac    = kelly_fraction(win_prob, lay_price)
            if frac <= 0:
                continue
            backer = max(frac * partial * bank, params.ruin / (lay_price - 1.0))
            stake  = backer * (lay_price - 1.0)
        else:
            stake = kwargs.get("liability", 20.0)

        # Simulate outcome
        if random.random() < p_lose:
            pnl = stake / (lay_price - 1.0)   # backer stake = our win
        else:
            pnl = -stake                        # liability lost

        bank += pnl

        if bank >= params.bank_target * params.bankroll:
            return "Objective Achieved", i + 1
        if bank < params.ruin:
            return "Ruined", i + 1

    return "Bets Exhausted", params.n_bets


def run_simulation(
    method: str,
    params: Optional[SimParams] = None,
    n_sims: int = 1_000,
    **kwargs,
) -> dict:
    """
    Monte Carlo simulation of a staking strategy over n_sims paths.

    Args:
        method: "fixed" | "proportional_a" | "proportional_b" |
                "kelly" | "half_kelly" | "quarter_kelly"
        params: SimParams instance (uses defaults if None)
        n_sims: Number of simulation runs (default 1,000; use ≤5,000 for speed)
        **kwargs: Method-specific parameters (e.g. liability=20, stake_pct=0.02)

    Returns summary statistics across all simulation runs.
    """
    if params is None:
        params = SimParams()

    results = [_simulate_one(params, method, **kwargs) for _ in range(n_sims)]

    outcomes  = [r[0] for r in results]
    bet_counts = [r[1] for r in results]

    success_runs = [bc for o, bc in results if o == "Objective Achieved"]
    ruin_runs    = [bc for o, bc in results if o == "Ruined"]

    prob_success = len(success_runs) / n_sims
    prob_ruin    = len(ruin_runs)    / n_sims

    def _median(lst):
        if not lst:
            return None
        s = sorted(lst)
        n = len(s)
        return s[n // 2] if n % 2 else (s[n // 2 - 1] + s[n // 2]) / 2

    return {
        "success":          True,
        "method":           method,
        "n_sims":           n_sims,
        "bankroll":         params.bankroll,
        "ruin_threshold":   params.ruin,
        "bank_target_x":    params.bank_target,
        "odds_range":       [params.min_odds, params.max_odds],
        "edge":             params.edge,
        "prob_success":     round(prob_success, 4),
        "prob_success_pct": f"{prob_success*100:.1f}%",
        "prob_ruin":        round(prob_ruin,    4),
        "prob_ruin_pct":    f"{prob_ruin*100:.1f}%",
        "median_bets_to_success": _median(success_runs),
        "median_bets_to_ruin":    _median(ruin_runs),
        "kwargs_used":      kwargs,
        "interpretation": (
            f"Over {n_sims:,} simulations of a {method} strategy with "
            f"{params.edge*100:.0f}% edge and odds between {params.min_odds}–{params.max_odds}: "
            f"{prob_success*100:.1f}% chance of reaching {params.bank_target}x bankroll target, "
            f"{prob_ruin*100:.1f}% chance of ruin."
        ),
    }


def compare_all_simulations(
    params: Optional[SimParams] = None,
    n_sims: int = 500,
) -> dict:
    """
    Run all five staking strategies through Monte Carlo and return a
    ranked comparison. Useful for the agent to explain which strategy
    best matches the user's risk/reward profile.
    """
    if params is None:
        params = SimParams()

    strategies = [
        ("fixed",          {"liability": 20}),
        ("proportional_a", {"stake_pct": 0.02}),
        ("proportional_b", {"win_pct": 0.03}),
        ("half_kelly",     {}),
        ("quarter_kelly",  {}),
    ]

    results = {}
    for method, kwargs in strategies:
        results[method] = run_simulation(method, params, n_sims, **kwargs)

    # Rank by probability of success
    ranked = sorted(
        results.items(),
        key=lambda x: x[1].get("prob_success", 0),
        reverse=True,
    )

    return {
        "success":         True,
        "simulation_params": {
            "bankroll":    params.bankroll,
            "ruin":        params.ruin,
            "target_x":   params.bank_target,
            "min_odds":    params.min_odds,
            "max_odds":    params.max_odds,
            "edge":        params.edge,
            "n_sims":      n_sims,
        },
        "results":  results,
        "ranked_by_success": [
            {
                "rank":             i + 1,
                "method":           name,
                "prob_success_pct": res["prob_success_pct"],
                "prob_ruin_pct":    res["prob_ruin_pct"],
                "median_bets_to_success": res["median_bets_to_success"],
            }
            for i, (name, res) in enumerate(ranked)
        ],
        "recommendation": _sim_recommendation(ranked),
    }


def _sim_recommendation(ranked: list) -> str:
    if not ranked:
        return "No simulation data."
    best_name, best = ranked[0]
    worst_name, worst = ranked[-1]
    return (
        f"Best strategy for this scenario: {best_name} "
        f"({best['prob_success_pct']} success, {best['prob_ruin_pct']} ruin). "
        f"Worst: {worst_name} ({worst['prob_success_pct']} success, {worst['prob_ruin_pct']} ruin). "
        "Kelly variants typically dominate when edge is positive and well-estimated. "
        "Fixed staking is safest when edge is uncertain."
    )


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _no_bet(reason: str) -> dict:
    return {"success": False, "verdict": "NO_BET", "reason": reason}

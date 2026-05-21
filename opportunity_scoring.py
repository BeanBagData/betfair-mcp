"""
Unified opportunity scoring for lay candidates.

This module is pure: it does not call Betfair, touch disk, or mutate shared
state. Sub-agents, paper automation, and streaming strategies use it so the
same signals produce the same verdicts everywhere.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional


@dataclass(frozen=True)
class ScoreConfig:
    """Runtime scoring gates for lay opportunities."""

    min_profit_ratio: float = 1.5
    min_edge_pct: float = 3.0
    strong_lay_score: int = 70
    lay_score: int = 40
    marginal_score: int = 20


def _safe_float(value: Any, default: Optional[float] = None) -> Optional[float]:
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _bounded(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def _profit_ratio(lay_price: Optional[float]) -> float:
    if lay_price is None or lay_price <= 1.0:
        return 0.0
    return round(1.0 / (lay_price - 1.0), 3)


def _estimated_win_prob(
    lay_price: Optional[float],
    model_edge: dict[str, Any],
    sp_edge: dict[str, Any],
) -> tuple[float, str]:
    sp_prob = _safe_float(sp_edge.get("estimated_win_prob_for_kelly"))
    if sp_prob is None:
        sp_prob = _safe_float(sp_edge.get("market_win_prob"))
    if sp_prob is not None and 0 < sp_prob < 1:
        return round(sp_prob, 4), "sp"

    model_price = _safe_float(model_edge.get("model_price"))
    if model_price is not None and model_price > 1.0:
        return round(_bounded(1.0 / model_price, 0.01, 0.95), 4), "model"

    if lay_price is not None and lay_price > 1.0:
        return round(_bounded(1.0 / lay_price, 0.01, 0.95), 4), "market"

    return 0.5, "fallback"


def _score_sp_value(sp_edge: dict[str, Any], issues: list[str]) -> int:
    verdict = str(sp_edge.get("verdict") or "UNKNOWN").upper()
    edge_net = _safe_float(sp_edge.get("edge_net"), 0.0) or 0.0

    if verdict == "STRONG_VALUE":
        return 30 + int(_bounded(edge_net * 100, 0, 10))
    if verdict == "MARGINAL_VALUE":
        return 15 + int(_bounded(edge_net * 100, 0, 5))
    if verdict == "BREAKEVEN":
        issues.append("sp_edge_breakeven")
        return -5
    if verdict == "NO_VALUE":
        issues.append("sp_edge_no_value")
        return -30
    return 0


def _score_model_edge(
    model_edge: dict[str, Any],
    config: ScoreConfig,
    issues: list[str],
) -> int:
    signal = str(model_edge.get("signal") or "UNKNOWN").upper()
    edge_pct = _safe_float(model_edge.get("edge_pct"), 0.0) or 0.0

    if signal in {"LAY", "BOTH"}:
        if edge_pct >= config.min_edge_pct:
            return 25 + int(_bounded(edge_pct - config.min_edge_pct, 0, 10))
        issues.append("model_edge_below_minimum")
        return 0
    if signal == "BACK":
        issues.append("model_signal_back")
        return -25
    if signal == "NONE":
        issues.append("model_signal_none")
        return -5
    return 0


def _score_wom(wom_signal: str, issues: list[str]) -> int:
    signal = (wom_signal or "UNKNOWN").upper()
    if signal == "LAY_HEAVY":
        return 15
    if signal == "BALANCED":
        return 5
    if signal == "BACK_HEAVY":
        issues.append("wom_back_heavy")
        return -20
    if signal == "NO_LIQUIDITY":
        issues.append("wom_no_liquidity")
        return -10
    return 0


def _score_movement(movement_signal: Optional[dict[str, Any]], issues: list[str]) -> int:
    if not movement_signal:
        return 0

    signal = str(movement_signal.get("signal") or "UNKNOWN").upper()
    ticks = abs(int(_safe_float(movement_signal.get("delta_ticks"), 0) or 0))

    if signal in {"DRIFT", "DRIFTED"}:
        return 10 + min(ticks, 10)
    if signal in {"STEAM", "SUPPORTED"}:
        issues.append("market_supported_or_steaming")
        return -15 - min(ticks, 10)
    return 0


def _score_spread(spread: Optional[dict[str, Any]], issues: list[str]) -> int:
    assessment = str((spread or {}).get("assessment") or "UNKNOWN").upper()
    if assessment == "TIGHT":
        return 10
    if assessment == "NORMAL":
        return 5
    if assessment == "WIDE":
        issues.append("wide_spread")
        return -10
    if assessment == "NO_MARKET":
        issues.append("no_market")
        return -15
    return 0


def _verdict(score: int, config: ScoreConfig, issues: list[str]) -> str:
    hard_negative = {"sp_edge_no_value", "model_signal_back", "no_market"}
    if any(issue in hard_negative for issue in issues):
        return "SKIP"
    if score >= config.strong_lay_score:
        return "STRONG_LAY"
    if score >= config.lay_score:
        return "LAY"
    if score >= config.marginal_score:
        return "MARGINAL"
    return "SKIP"


def score_lay_opportunity(
    *,
    lay_price: Any,
    best_back: Any = None,
    model_edge: Optional[dict[str, Any]] = None,
    wom_signal: str = "UNKNOWN",
    context_score: int = 0,
    sp_edge: Optional[dict[str, Any]] = None,
    movement_signal: Optional[dict[str, Any]] = None,
    spread: Optional[dict[str, Any]] = None,
    historical_adjustments: Optional[dict[str, Any]] = None,
    config: Optional[ScoreConfig] = None,
) -> dict[str, Any]:
    """
    Score a lay candidate and return a stable decision payload.

    `estimated_win_prob` is the horse-win probability used for lay Kelly sizing.
    Lower values favour laying; it prefers BSP-derived estimates, then model
    price, then the live lay price.
    """
    config = config or ScoreConfig()
    model_edge = model_edge or {}
    sp_edge = sp_edge or {}
    historical_adjustments = historical_adjustments or {}

    issues: list[str] = []
    components: dict[str, int] = {}

    lay_price_f = _safe_float(lay_price)
    best_back_f = _safe_float(best_back)
    profit_ratio = _profit_ratio(lay_price_f)

    if profit_ratio >= config.min_profit_ratio:
        components["profit_ratio"] = 30
    elif profit_ratio >= 1.0:
        components["profit_ratio"] = 5
        issues.append("profit_ratio_below_minimum")
    else:
        components["profit_ratio"] = -15
        issues.append("profit_ratio_below_minimum")

    components["sp_value"] = _score_sp_value(sp_edge, issues)
    components["model_edge"] = _score_model_edge(model_edge, config, issues)
    components["wom"] = _score_wom(wom_signal, issues)
    components["context"] = int(context_score or 0)
    components["movement"] = _score_movement(movement_signal, issues)
    components["spread"] = _score_spread(spread, issues)

    hist_score = 0
    for key in ("venue_roi", "wom_roi", "price_bucket_roi"):
        roi = _safe_float(historical_adjustments.get(key), 0.0) or 0.0
        if roi > 15:
            hist_score += 8
        elif roi > 5:
            hist_score += 4
        elif roi < -15:
            hist_score -= 8
        elif roi < -5:
            hist_score -= 4
    components["historical"] = hist_score

    score = int(round(_bounded(sum(components.values()), 0, 100)))
    estimated_win_prob, estimated_source = _estimated_win_prob(
        lay_price_f,
        model_edge,
        sp_edge,
    )

    edge_pct = _safe_float(model_edge.get("edge_pct"), 0.0) or 0.0
    edge_net = _safe_float(sp_edge.get("edge_net"))

    return {
        "score": score,
        "verdict": _verdict(score, config, issues),
        "profit_ratio": profit_ratio,
        "estimated_win_prob": estimated_win_prob,
        "estimated_win_prob_source": estimated_source,
        "components": components,
        "issues": issues,
        "lay_price": lay_price_f,
        "best_back": best_back_f,
        "model_signal": str(model_edge.get("signal") or "UNKNOWN").upper(),
        "model_edge_pct": round(edge_pct, 4),
        "wom_signal": (wom_signal or "UNKNOWN").upper(),
        "sp_verdict": str(sp_edge.get("verdict") or "UNKNOWN").upper(),
        "sp_edge_net": round(edge_net, 6) if edge_net is not None else None,
    }

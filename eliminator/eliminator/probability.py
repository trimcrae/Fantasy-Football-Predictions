"""Conversions between market prices, point spreads and win probabilities."""
from __future__ import annotations

import math

import numpy as np
from scipy.stats import norm

# Standard deviation of (actual margin - spread). Calibrated by `eliminator calibrate`;
# this default is the long-run NFL value and is overridden by calibration.json.
DEFAULT_SIGMA = 13.45


def american_to_prob(ml: float) -> float:
    """Implied probability (with vig) of an American moneyline."""
    ml = float(ml)
    if ml < 0:
        return -ml / (-ml + 100.0)
    return 100.0 / (ml + 100.0)


def devig(p_a: float, p_b: float) -> tuple[float, float]:
    """Remove the bookmaker margin multiplicatively (the standard, well-calibrated choice
    for two-way NFL moneylines)."""
    s = p_a + p_b
    if s <= 0:
        raise ValueError("implied probabilities must be positive")
    return p_a / s, p_b / s


def moneyline_home_prob(home_ml: float, away_ml: float) -> float:
    ph, pa = devig(american_to_prob(home_ml), american_to_prob(away_ml))
    return ph


def spread_to_prob(spread: float | np.ndarray, sigma: float = DEFAULT_SIGMA):
    """P(home wins) given the expected home margin `spread` (positive = home favoured)."""
    return norm.cdf(np.asarray(spread, dtype=float) / sigma)


def prob_to_spread(p: float | np.ndarray, sigma: float = DEFAULT_SIGMA):
    """Inverse of spread_to_prob: the margin that corresponds to win probability p."""
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    return sigma * norm.ppf(p)


def game_home_prob(home_ml, away_ml, spread, sigma: float = DEFAULT_SIGMA) -> float | None:
    """Best available market probability for the home team.

    The moneyline is the direct price on the outcome we care about, so it wins when
    present; the spread is the fallback (pre-2010 history and some sportsbooks feeds).
    """
    if _is_num(home_ml) and _is_num(away_ml):
        return moneyline_home_prob(home_ml, away_ml)
    if _is_num(spread):
        return float(spread_to_prob(spread, sigma))
    return None


def _is_num(x) -> bool:
    try:
        return x is not None and not (isinstance(x, float) and math.isnan(x)) and str(x) != ""
    except Exception:  # pragma: no cover
        return False

"""Market-implied team ratings.

``fit_market_ratings`` backs team strengths out of posted spreads, the same idea as
inpredictable's GPF: spread = r_home - r_away + HFA + rest effect (+ QB effect). It is a
weighted ridge least squares with

* recency weighting of past lines (half-life in weeks),
* a preseason prior (last season's rating regressed toward zero) that carries the early
  weeks and fades as lines accumulate,
* optional QB residualisation, so ratings describe the *healthy* team and the projection
  layer can add the expected QB effect back per week.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..teams import TEAMS, TEAM_INDEX

N = len(TEAMS)


@dataclass
class RatingFit:
    ratings: pd.Series          # team -> points vs average opponent, neutral field
    n_lines: int
    weight_total: float
    residual_sd: float          # sd of (spread - fitted) over the lines used


def _design(home: np.ndarray, away: np.ndarray) -> np.ndarray:
    X = np.zeros((len(home), N))
    X[np.arange(len(home)), [TEAM_INDEX[t] for t in home]] = 1.0
    X[np.arange(len(away)), [TEAM_INDEX[t] for t in away]] = -1.0
    return X


def fit_market_ratings(lines: pd.DataFrame, hfa: float, rest_per_day: float,
                       prior: pd.Series | None = None, prior_weight: float = 6.0,
                       weights: np.ndarray | None = None, qb_adj: np.ndarray | None = None) -> RatingFit:
    """Fit ratings to posted spreads.

    lines: DataFrame with columns home, away, spread_line, neutral, rest_diff.
    weights: per-line weights (recency); default 1.
    qb_adj: net home-side QB effect (points) already inside each spread, removed before fitting.
    prior: team -> prior rating (centred); prior_weight is its weight in line-equivalents.
    """
    lines = lines.dropna(subset=["spread_line"])
    n = len(lines)
    X = _design(lines["home"].to_numpy(), lines["away"].to_numpy())
    y = lines["spread_line"].to_numpy(dtype=float).copy()
    y -= hfa * (~lines["neutral"].to_numpy(dtype=bool))
    y -= rest_per_day * lines["rest_diff"].to_numpy(dtype=float)
    if qb_adj is not None:
        y -= np.asarray(qb_adj, dtype=float)
    w = np.ones(n) if weights is None else np.asarray(weights, dtype=float)

    rows_X = [X]
    rows_y = [y]
    rows_w = [w]
    # Prior pseudo-observations: one identity row per team.
    p = np.zeros(N) if prior is None else np.array([float(prior.get(t, 0.0)) for t in TEAMS])
    rows_X.append(np.eye(N))
    rows_y.append(p)
    rows_w.append(np.full(N, max(prior_weight, 1e-3)))
    # Sum-to-zero constraint (soft, heavy weight).
    rows_X.append(np.ones((1, N)))
    rows_y.append(np.zeros(1))
    rows_w.append(np.array([1e4]))

    XA = np.vstack(rows_X)
    yA = np.concatenate(rows_y)
    wA = np.concatenate(rows_w)
    sw = np.sqrt(wA)[:, None]
    beta, *_ = np.linalg.lstsq(XA * sw, yA * sw[:, 0], rcond=None)
    beta -= beta.mean()
    fitted = X @ beta
    resid_sd = float(np.sqrt(np.average((y - fitted) ** 2, weights=w))) if n else float("nan")
    return RatingFit(ratings=pd.Series(beta, index=TEAMS), n_lines=n, weight_total=float(w.sum()), residual_sd=resid_sd)


def recency_weights(line_weeks: np.ndarray, as_of_week: int, half_life: float) -> np.ndarray:
    """Weight lines by age; lines for the current and future weeks count fully."""
    age = np.clip(as_of_week - np.asarray(line_weeks, dtype=float), 0, None)
    if not np.isfinite(half_life) or half_life <= 0:
        return np.ones_like(age)
    return 0.5 ** (age / half_life)


def season_final_ratings(games: pd.DataFrame, hfa: float, rest_per_day: float) -> pd.Series:
    """Unweighted fit over a whole regular season's closing spreads (used for priors)."""
    fit = fit_market_ratings(games, hfa=hfa, rest_per_day=rest_per_day, prior=None, prior_weight=0.05)
    return fit.ratings


def preseason_prior(prev_final: pd.Series | None, regression: float) -> pd.Series:
    if prev_final is None:
        return pd.Series(0.0, index=TEAMS)
    return (prev_final.reindex(TEAMS).fillna(0.0) * regression)


def as_of_ratings(season_games: pd.DataFrame, as_of_week: int, hfa: float, rest_per_day: float,
                  prior: pd.Series | None, prior_weight: float, half_life: float,
                  include_future_lines: bool = True, qb_adj: np.ndarray | None = None) -> RatingFit:
    """Ratings using lines posted for weeks <= as_of_week plus any posted future lines.

    In a backtest only closing lines exist, so ``include_future_lines=False`` restricts the
    fit to weeks <= as_of_week, which is what would have been known at the time.
    """
    g = season_games.dropna(subset=["spread_line"])
    if not include_future_lines:
        g = g[g["week"] <= as_of_week]
    adj = None
    if qb_adj is not None:
        adj = np.asarray(pd.Series(qb_adj, index=season_games.index).reindex(g.index).fillna(0.0))
    w = recency_weights(g["week"].to_numpy(), as_of_week, half_life)
    return fit_market_ratings(g, hfa=hfa, rest_per_day=rest_per_day, prior=prior, prior_weight=prior_weight,
                              weights=w, qb_adj=adj)

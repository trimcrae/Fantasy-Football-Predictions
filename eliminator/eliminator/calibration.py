"""Fit every statistical parameter of the model from nflverse history.

Outputs calibration.json; config.py merges it under ``model``. Run ``eliminator calibrate``.

What gets estimated and how:

* sigma            MLE of the margin sd around closing spreads (spread -> win probability).
* hfa, rest        joint least squares of closing spreads on team dummies, home field and
                   rest differential, season by season; HFA default = recent-seasons mean.
* prior_regression regression of early-season market ratings on last season's final ratings.
* prior_weight / recency_half_life   grid search minimising one-week-ahead projection error.
* horizon variance var(h, k) = a + b*h + c/(1+k): squared error of projected vs closing
                   spreads as a function of horizon h (weeks ahead) and as-of week k.
* week-18 rest     points a settled playoff seed (bye / wild-card game) or elimination takes off a
                   team's week-18 line (standings after week 17 vs closing lines); then the
                   residual flattening and extra variance of week-18 lines.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.stats import norm

from .config import ROOT
from .data.schedule import regular_season
from .model.ratings import as_of_ratings, preseason_prior, season_final_ratings
from .probability import moneyline_home_prob
from .teams import TEAMS

CAL_PATH = ROOT / "calibration.json"


def fit_sigma(df: pd.DataFrame) -> float:
    g = df[(df["game_type"] == "REG") & df["played"] & df["spread_line"].notna()]
    s = g["spread_line"].to_numpy(float)
    y = g["home_win"].to_numpy(float)

    def nll(sigma):
        p = np.clip(norm.cdf(s / sigma), 1e-9, 1 - 1e-9)
        return -np.sum(y * np.log(p) + (1 - y) * np.log(1 - p))

    res = minimize_scalar(nll, bounds=(9.0, 20.0), method="bounded")
    return float(res.x)


def moneyline_vs_spread_logloss(df: pd.DataFrame, sigma: float) -> dict:
    g = df[(df["game_type"] == "REG") & df["played"] & df["home_moneyline"].notna() & df["spread_line"].notna()]
    y = g["home_win"].to_numpy(float)
    p_ml = np.array([moneyline_home_prob(h, a) for h, a in zip(g["home_moneyline"], g["away_moneyline"])])
    p_sp = norm.cdf(g["spread_line"].to_numpy(float) / sigma)

    def ll(p):
        p = np.clip(p, 1e-9, 1 - 1e-9)
        return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))

    return {"n": int(len(g)), "logloss_moneyline": ll(p_ml), "logloss_spread": ll(p_sp),
            "logloss_blend": ll(0.5 * p_ml + 0.5 * p_sp), "mean_abs_diff": float(np.mean(np.abs(p_ml - p_sp)))}


def fit_hfa_rest(df: pd.DataFrame, seasons: list[int]) -> dict:
    """Per-season HFA with a shared rest coefficient (least squares with team dummies)."""
    rows_X, rows_y = [], []
    n_teams = len(TEAMS)
    idx = {t: i for i, t in enumerate(TEAMS)}
    for si, s in enumerate(seasons):
        g = regular_season(df, s).dropna(subset=["spread_line"])
        base = si * n_teams
        for r in g.itertuples(index=False):
            x = np.zeros(len(seasons) * n_teams + len(seasons) + 1)
            x[base + idx[r.home]] = 1.0
            x[base + idx[r.away]] = -1.0
            if not r.neutral:
                x[len(seasons) * n_teams + si] = 1.0
            x[-1] = r.rest_diff
            rows_X.append(x)
            rows_y.append(r.spread_line)
        # sum-to-zero per season
        x = np.zeros(len(seasons) * n_teams + len(seasons) + 1)
        x[base:base + n_teams] = 100.0
        rows_X.append(x)
        rows_y.append(0.0)
    X = np.array(rows_X)
    y = np.array(rows_y)
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    hfa = {s: float(beta[len(seasons) * n_teams + i]) for i, s in enumerate(seasons)}
    return {"hfa_by_season": hfa, "rest_per_day": float(beta[-1])}


def fit_prior_regression(df: pd.DataFrame, seasons: list[int], hfa: float, rest: float) -> dict:
    xs, ys = [], []
    for s in seasons:
        prev = regular_season(df, s - 1)
        cur = regular_season(df, s)
        if prev["spread_line"].notna().sum() < 200 or cur["spread_line"].notna().sum() < 50:
            continue
        r_prev = season_final_ratings(prev, hfa, rest)
        early = cur[cur["week"] <= 4]
        r_early = season_final_ratings(early, hfa, rest)
        xs.append(r_prev.reindex(TEAMS).to_numpy())
        ys.append(r_early.reindex(TEAMS).to_numpy())
    x = np.concatenate(xs)
    y = np.concatenate(ys)
    c = float(np.sum(x * y) / np.sum(x * x))
    resid_sd = float(np.std(y - c * x))
    return {"prior_regression": c, "prior_resid_sd": resid_sd, "n_seasons": len(xs)}


def projection_errors(df: pd.DataFrame, seasons: list[int], hfa: float, rest: float, regression: float,
                      prior_weight: float, half_life: float) -> pd.DataFrame:
    """For every (season, as-of week k, game in week k+h) the projected vs closing spread."""
    recs = []
    for s in seasons:
        cur = regular_season(df, s)
        prev = regular_season(df, s - 1)
        prior = preseason_prior(season_final_ratings(prev, hfa, rest) if prev["spread_line"].notna().sum() > 200 else None,
                                regression)
        max_week = int(cur["week"].max())
        for k in range(0, max_week):
            fit = as_of_ratings(cur, k, hfa, rest, prior, prior_weight, half_life, include_future_lines=False)
            r = fit.ratings
            fut = cur[(cur["week"] > k) & cur["spread_line"].notna()]
            proj = (r[fut["home"]].to_numpy() - r[fut["away"]].to_numpy()
                    + hfa * (~fut["neutral"].to_numpy(bool)) + rest * fut["rest_diff"].to_numpy(float))
            for wk, sp, pj in zip(fut["week"], fut["spread_line"], proj):
                recs.append((s, k, int(wk), int(wk) - k, float(sp), float(pj)))
    out = pd.DataFrame(recs, columns=["season", "k", "week", "h", "spread", "proj"])
    out["err"] = out["spread"] - out["proj"]
    return out


def fit_horizon_variance(errs: pd.DataFrame) -> dict:
    """var(h, k) = a + b*h + c/(1+k), fitted on squared errors for h >= 1."""
    e = errs[errs["h"] >= 1]
    X = np.column_stack([np.ones(len(e)), e["h"].to_numpy(float), 1.0 / (1.0 + e["k"].to_numpy(float))])
    beta, *_ = np.linalg.lstsq(X, e["err"].to_numpy(float) ** 2, rcond=None)
    by_h = e.groupby("h")["err"].apply(lambda v: float(np.mean(v ** 2))).to_dict()
    return {"horizon_var_a": float(beta[0]), "horizon_var_b": float(beta[1]), "horizon_var_c": float(beta[2]),
            "mse_by_horizon": {int(k): v for k, v in by_h.items()}}


def week18_effect(errs: pd.DataFrame, last_week: int = 18) -> dict:
    """How much worse and how much flatter week-18 lines are relative to projections."""
    e = errs[(errs["h"] >= 1) & (errs["k"] >= 12)]
    def slope(sub):
        x, y = sub["proj"].to_numpy(float), sub["spread"].to_numpy(float)
        return float(np.sum(x * y) / max(np.sum(x * x), 1e-9))
    w18 = e[e["week"] == last_week]
    other = e[(e["week"] >= last_week - 3) & (e["week"] < last_week)]
    return {"week18_slope": slope(w18), "late_weeks_slope": slope(other),
            "week18_mse": float(np.mean(w18["err"] ** 2)), "late_weeks_mse": float(np.mean(other["err"] ** 2)),
            "n_week18": int(len(w18))}


def fit_week18_rest(df: pd.DataFrame, seasons: list[int], hfa: float, rest: float, regression: float,
                    prior_weight: float, half_life: float, late_mse: float | None = None) -> dict:
    """Points a settled playoff seed (bye / wild-card game to play) or elimination takes off a
    team's week-18 line, from closing lines against the as-of-week-17 projection, plus what is
    left of the week-18 flattening and extra variance once the flags are in."""
    from .model.standings import CONF_OF, DIV_OF, Record, week18_flags
    from .teams import TEAM_INDEX
    rows = []
    for s in seasons:
        cur = regular_season(df, s)
        prev = regular_season(df, s - 1)
        last = int(cur["week"].max())
        prior = preseason_prior(season_final_ratings(prev, hfa, rest) if prev["spread_line"].notna().sum() > 200 else None, regression)
        r = as_of_ratings(cur, last - 1, hfa, rest, prior, prior_weight, half_life, include_future_lines=False).ratings
        before = cur[(cur["week"] < last) & cur["played"]]
        wins = np.zeros((1, len(TEAMS))); conf = np.zeros((1, len(TEAMS))); div = np.zeros((1, len(TEAMS)))
        for g in before.itertuples(index=False):
            ih, ia = TEAM_INDEX[g.home], TEAM_INDEX[g.away]
            win = ih if g.result > 0 else (ia if g.result < 0 else None)
            if win is None:
                continue
            wins[0, win] += 1
            conf[0, win] += CONF_OF[ih] == CONF_OF[ia]
            div[0, win] += DIV_OF[ih] == DIV_OF[ia]
        w18 = cur[cur["week"] == last]
        pairs = [(TEAM_INDEX[g.home], TEAM_INDEX[g.away]) for g in w18.itertuples(index=False)]
        fl = week18_flags(Record(wins, conf, div), pairs, s, draws=256, rng=np.random.default_rng(s))
        for g in w18[w18["spread_line"].notna()].itertuples(index=False):
            ih, ia = TEAM_INDEX[g.home], TEAM_INDEX[g.away]
            proj = r[g.home] - r[g.away] + hfa * (not g.neutral) + rest * g.rest_diff
            rows.append((float(g.spread_line), float(proj), *(int(fl[k][0, ih]) - int(fl[k][0, ia]) for k in ("bye", "locked", "out"))))
    d = np.array(rows, float)
    spread, proj, X = d[:, 0], d[:, 1], d[:, 2:]
    err = spread - proj
    beta, *_ = np.linalg.lstsq(X, err, rcond=None)
    resid = err - X @ beta
    se = np.sqrt(np.diag(np.linalg.inv(X.T @ X)) * resid.var())
    adj = proj + X @ beta
    slope_after = float(np.sum(adj * spread) / max(np.sum(adj * adj), 1e-9))
    out = {"bye": float(-beta[0]), "locked": float(-beta[1]), "out": float(-beta[2]),
           "se": {"bye": float(se[0]), "locked": float(se[1]), "out": float(se[2])},
           "n_games": int(len(d)), "n_flagged": int(np.sum(np.any(X[:, :2] != 0, axis=1))),
           "mse_before": float(np.mean(err ** 2)), "mse_after": float(np.mean(resid ** 2)),
           "slope_after": slope_after}
    if late_mse is not None:
        out["extra_var_after"] = float(max(out["mse_after"] - late_mse, 0.0))
    return out


def calibrate(df: pd.DataFrame, seasons: list[int] | None = None, verbose: bool = True) -> dict:
    seasons = seasons or list(range(2012, int(df["season"].max()) + 1))
    played_seasons = [s for s in seasons if regular_season(df, s)["played"].sum() > 200]
    sigma = fit_sigma(df)
    ml = moneyline_vs_spread_logloss(df, sigma)
    hr = fit_hfa_rest(df, played_seasons)
    recent = [s for s in played_seasons if s >= played_seasons[-1] - 3 and s != 2020]
    hfa = float(np.mean([hr["hfa_by_season"][s] for s in recent]))
    rest = hr["rest_per_day"]
    pr = fit_prior_regression(df, played_seasons, hfa, rest)
    # grid search for prior weight and half-life on one-week-ahead error
    best = None
    grid_log = []
    for pw in (0.5, 1.0, 2.0, 4.0, 8.0):
        for hl in (4.0, 8.0, 12.0, 100.0):
            errs = projection_errors(df, played_seasons, hfa, rest, pr["prior_regression"], pw, hl)
            mse1 = float(np.mean(errs[errs["h"] == 1]["err"] ** 2))
            mse_all = float(np.mean(errs["err"] ** 2))
            grid_log.append({"prior_weight": pw, "half_life": hl, "mse_h1": mse1, "mse_all": mse_all})
            if best is None or mse_all < best[0]:
                best = (mse_all, pw, hl, errs)
    _, prior_weight, half_life, errs = best
    hv = fit_horizon_variance(errs)
    w18 = week18_effect(errs, last_week=int(regular_season(df, played_seasons[-1])["week"].max()))
    w18_rest = fit_week18_rest(df, played_seasons, hfa, rest, pr["prior_regression"], prior_weight, half_life,
                               late_mse=w18["late_weeks_mse"])
    model = {
        "sigma": sigma, "hfa": hfa, "rest_per_day": rest,
        "prior_regression": pr["prior_regression"], "prior_weight": prior_weight,
        "recency_half_life": half_life,
        "horizon_var_a": hv["horizon_var_a"], "horizon_var_b": hv["horizon_var_b"], "horizon_var_c": hv["horizon_var_c"],
        # week 18: the rest flags carry the effect; what is left is a residual flattening and variance
        "week18_shrink": float(np.clip(w18_rest["slope_after"] / max(w18["late_weeks_slope"], 1e-6), 0.3, 1.0)),
        "week18_extra_var": float(w18_rest.get("extra_var_after", 0.0)),
        "week18_rest": {k: round(w18_rest[k], 2) for k in ("bye", "locked", "out")},
    }
    out = {"model": model, "diagnostics": {"moneyline_vs_spread": ml, "hfa_by_season": hr["hfa_by_season"],
                                           "prior": pr, "grid": grid_log, "horizon": hv, "week18": w18,
                                           "week18_rest": w18_rest,
                                           "seasons": played_seasons}}
    if verbose:
        print(json.dumps(out, indent=1, default=float))
    return out


def save_calibration(cal: dict, path: Path = CAL_PATH) -> Path:
    path.write_text(json.dumps(cal, indent=1, default=float))
    return path

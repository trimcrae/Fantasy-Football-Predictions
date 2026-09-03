"""Assemble the healthy-baseline rating for every team from the chosen source(s).

Sources
* market        ratings fitted to posted spreads (see ratings.py). QB absences that are in
                the ledger are residualised out, so the result is the healthy team.
* inpredictable GPF from inpredictable.com. It is a blend of the season's lines, so a QB
                who has been out for a while is partly priced in; that share is added back
                (``reflected_fraction``) and a small correction removes the boost a team gets
                from this week's line against an opponent missing its QB.
* blend         weighted mean of both healthy series.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ..data.schedule import regular_season
from ..teams import TEAMS
from .qb import QBSituation, p_out_by_week, reflected_fraction
from .ratings import as_of_ratings, preseason_prior, season_final_ratings


@dataclass
class Strength:
    healthy: pd.Series                 # team -> points vs average, neutral field, starter healthy
    source: str
    detail: dict = field(default_factory=dict)
    market: pd.Series | None = None
    inpredictable: pd.Series | None = None
    qb_effect: np.ndarray | None = None  # [season_weeks+1, 32] expected QB points effect (<= 0)
    p_out: np.ndarray | None = None      # [season_weeks+1, 32] P(starter out)
    penalty: np.ndarray | None = None    # [32]


def team_game_weeks(games: pd.DataFrame, team: str) -> list[int]:
    g = games[(games["home"] == team) | (games["away"] == team)]
    return sorted(int(w) for w in g["week"].unique())


def qb_matrices(games: pd.DataFrame, ledger: list[QBSituation], current_week: int, season_weeks: int,
                cfg_qb: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """P(out), penalty and expected effect (= -penalty * P(out)) per week and team."""
    p_out = np.zeros((season_weeks + 1, len(TEAMS)))
    penalty = np.zeros(len(TEAMS))
    for sit in ledger:
        i = TEAMS.index(sit.team)
        weeks = team_game_weeks(games, sit.team)
        p = p_out_by_week(sit, current_week, season_weeks, cfg_qb, game_weeks=weeks)
        # multiple situations on one team (e.g. QB1 and QB2 both hurt) - keep the larger effect
        eff_new = sit.penalty * p
        eff_old = penalty[i] * p_out[:, i]
        mask = eff_new > eff_old
        p_out[mask, i] = p[mask]
        if eff_new.max() >= eff_old.max():
            penalty[i] = sit.penalty
    effect = -(penalty[None, :] * p_out)
    return p_out, penalty, effect


def game_qb_adj(games: pd.DataFrame, effect: np.ndarray) -> np.ndarray:
    """Net home-side QB effect inside each game's spread (home effect - away effect)."""
    idx = {t: i for i, t in enumerate(TEAMS)}
    w = games["week"].to_numpy(int)
    h = np.array([idx[t] for t in games["home"]])
    a = np.array([idx[t] for t in games["away"]])
    return effect[w, h] - effect[w, a]


def assemble(games_all: pd.DataFrame, season: int, current_week: int, cfg: dict,
             ledger: list[QBSituation], inpredictable: pd.DataFrame | None, source: str = "auto") -> Strength:
    m = cfg["model"]
    season_weeks = int(cfg.get("season_weeks") or 18)
    games = regular_season(games_all, season)
    p_out, penalty, effect = qb_matrices(games, ledger, current_week, season_weeks, cfg["qb"])

    # --- market-implied fit (always computed: it is the fallback and the blend partner)
    prev = regular_season(games_all, season - 1)
    prev_final = season_final_ratings(prev, m["hfa"], m["rest_per_day"]) if prev["spread_line"].notna().sum() > 200 else None
    prior = preseason_prior(prev_final, m["prior_regression"])
    qb_adj = game_qb_adj(games, effect)
    fit = as_of_ratings(games, current_week, m["hfa"], m["rest_per_day"], prior, m["prior_weight"],
                        m["recency_half_life"], include_future_lines=True, qb_adj=qb_adj)
    market = fit.ratings

    inp_healthy = None
    if inpredictable is not None:
        gpf = inpredictable.set_index("team")["gpf"].reindex(TEAMS).astype(float)
        inp_healthy = gpf.copy()
        cur_w = float(m.get("inpredictable_current_line_weight", 0.15))
        for sit in ledger:
            i = TEAMS.index(sit.team)
            weeks = team_game_weeks(games, sit.team)
            frac = reflected_fraction(sit, current_week, weeks, cfg["qb"])
            inp_healthy[sit.team] += penalty[i] * p_out[current_week, i] * frac
        # opponent-contamination: this week's line vs a QB-less opponent flatters the team
        wk = games[games["week"] == current_week]
        for r in wk.itertuples(index=False):
            ih, ia = TEAMS.index(r.home), TEAMS.index(r.away)
            inp_healthy[r.home] += cur_w * effect[current_week, ia]   # effect <= 0 -> reduces home
            inp_healthy[r.away] += cur_w * effect[current_week, ih]
        inp_healthy -= inp_healthy.mean()

    inp_check = None
    if inp_healthy is not None:
        rmse = float(np.sqrt(((inp_healthy - market) ** 2).mean()))
        played = inpredictable.attrs.get("games_played")
        stale = current_week == 1 and played is not None and played > 0
        max_rmse = float(m.get("inpredictable_max_rmse", 2.0))
        reason = None
        if stale:
            reason = f"page still shows last season ({played} games played before week 1)"
        elif rmse > max_rmse:
            reason = f"disagrees with this season's lines (rmse {rmse:.1f} pts vs {max_rmse:.1f} allowed)"
        inp_check = {"rmse": rmse, "games_played": played, "reason": reason}
    if source == "auto":
        if inp_healthy is None:
            source = "market"
        elif inp_check["reason"]:
            print(f"[strength] inpredictable ratings rejected: {inp_check['reason']}; using market-implied ratings")
            source = "market"
        else:
            source = "inpredictable"
    elif source in ("inpredictable", "blend") and inp_check and inp_check["reason"]:
        print(f"[strength] warning: inpredictable ratings {inp_check['reason']} (used anyway: ratings_source={source})")
    if source == "inpredictable" and inp_healthy is None:
        print("[strength] inpredictable ratings unavailable; using market-implied ratings")
        source = "market"
    if source == "market":
        healthy = market
    elif source == "inpredictable":
        healthy = inp_healthy
    elif source == "blend":
        w = float(m["inpredictable_weight"]) if inp_healthy is not None else 0.0
        healthy = (w * inp_healthy + (1 - w) * market) if inp_healthy is not None else market
    else:
        raise ValueError(f"unknown ratings source {source!r}")
    healthy = healthy - healthy.mean()
    return Strength(healthy=healthy, source=source, market=market, inpredictable=inp_healthy,
                    qb_effect=effect, p_out=p_out, penalty=penalty,
                    detail={"n_lines": fit.n_lines, "fit_resid_sd": fit.residual_sd,
                            "prior_from": season - 1 if prev_final is not None else None,
                            "inpredictable_check": inp_check})

"""Win probability for every (week, team) from now to the end of the regular season.

Rules, in priority order:
1. Played game            -> 1 / 0 (a tie counts as a loss in ESPN and Yahoo pools).
2. Current week with line -> de-vigged moneyline (spread fallback). This is the truth.
3. Otherwise              -> model spread from healthy ratings + home field + rest + expected
                             QB effect, blended with any posted line (weight falls with
                             horizon), converted with an inflated sigma:
                             sigma_h^2 = sigma^2 + discount * (a + b*h + c/(1+k)).
                             Week 18 additionally gets its spread shrunk and extra variance.
Overrides (state/overrides.yaml) can pin a spread or moneyline on any game and add a
week-18 rest risk per team.
"""
from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy.stats import norm

from ..data.schedule import ET
from ..probability import moneyline_home_prob, prob_to_spread
from ..teams import TEAMS, normalize
from .strength import Strength

NT = len(TEAMS)


@dataclass
class Projection:
    season: int
    current_week: int
    season_weeks: int
    weeks: list[int]                # planning weeks: current_week .. season_weeks
    prob: np.ndarray                # [n_weeks, 32] P(win); 0 where no game
    spread: np.ndarray              # [n_weeks, 32] team-perspective point estimate
    line_var: np.ndarray            # [n_weeks, 32] variance of the projected line (0 = known)
    sigma: float                    # game sigma
    pickable: np.ndarray            # [n_weeks, 32] game exists and has not kicked off
    has_game: np.ndarray            # [n_weeks, 32]
    table: pd.DataFrame             # one row per (week, team) with the explanation
    opponent: np.ndarray            # [n_weeks, 32] opponent index or -1
    week18_shrink: float

    def week_index(self, week: int) -> int:
        return self.weeks.index(week)

    def row(self, week: int, team: str) -> pd.Series:
        t = self.table
        return t[(t["week"] == week) & (t["team"] == team)].iloc[0]


def load_overrides(path: Path) -> dict:
    if not path.exists():
        return {}
    return yaml.safe_load(path.read_text()) or {}


def _override_for(overrides: dict, week: int, home: str, away: str) -> dict | None:
    for rec in overrides.get("lines", []) or []:
        try:
            if int(rec["week"]) == week and normalize(rec["home"]) == home and normalize(rec["away"]) == away:
                return rec
        except (KeyError, ValueError):
            continue
    return None


def build_projection(games: pd.DataFrame, season: int, current_week: int, strength: Strength, cfg: dict,
                     now: dt.datetime | None = None, overrides: dict | None = None) -> Projection:
    m = cfg["model"]
    now = now or dt.datetime.now(tz=ET)
    overrides = overrides or {}
    season_weeks = int(cfg.get("season_weeks") or int(games["week"].max()))
    weeks = list(range(current_week, season_weeks + 1))
    nW = len(weeks)
    sigma = float(m["sigma"])
    disc = float(m.get("future_discount", 1.0))
    a, b, c = float(m["horizon_var_a"]), float(m["horizon_var_b"]), float(m.get("horizon_var_c", 0.0))
    la, lb = float(m.get("posted_line_var_a", 1.0)), float(m.get("posted_line_var_b", 1.0))
    w18_shrink = float(m["week18_shrink"])
    w18_var = float(m["week18_extra_var"])
    rest_risk = {normalize(k): float(v) for k, v in (overrides.get("week18_rest_risk") or {}).items()}
    rest_pen = float(overrides.get("week18_rest_penalty", m.get("week18_rest_penalty", 6.0)))

    prob = np.zeros((nW, NT)); spread = np.zeros((nW, NT)); lvar = np.zeros((nW, NT))
    pickable = np.zeros((nW, NT), bool); has_game = np.zeros((nW, NT), bool)
    opp = np.full((nW, NT), -1, int)
    R = strength.healthy.reindex(TEAMS).to_numpy(float)
    eff = strength.qb_effect if strength.qb_effect is not None else np.zeros((season_weeks + 1, NT))
    rows = []
    idx = {t: i for i, t in enumerate(TEAMS)}
    sub = games[(games["week"] >= current_week) & (games["week"] <= season_weeks)]
    for g in sub.itertuples(index=False):
        w = int(g.week); wi = weeks.index(w); h = w - current_week
        ih, ia = idx[g.home], idx[g.away]
        ov = _override_for(overrides, w, g.home, g.away)
        # ---- model spread (home perspective)
        model_sp = (R[ih] + eff[w, ih]) - (R[ia] + eff[w, ia]) + float(m["hfa"]) * (not g.neutral) \
            + float(m["rest_per_day"]) * float(g.rest_diff)
        # ---- market line (converted to the sigma scale so blending is consistent)
        line_sp = None; line_src = None; p_ml = None
        if ov and ("home_ml" in ov and "away_ml" in ov):
            p_ml = moneyline_home_prob(ov["home_ml"], ov["away_ml"]); line_sp = float(prob_to_spread(p_ml, sigma)); line_src = "override-ml"
        elif ov and "spread" in ov:
            line_sp = float(ov["spread"]); line_src = "override-spread"
        elif pd.notna(g.home_moneyline) and pd.notna(g.away_moneyline):
            p_ml = moneyline_home_prob(g.home_moneyline, g.away_moneyline); line_sp = float(prob_to_spread(p_ml, sigma)); line_src = "moneyline"
        elif pd.notna(g.spread_line):
            line_sp = float(g.spread_line); line_src = "spread"
        kicked = g.kickoff <= now
        note = ""
        p_away = None
        # the market's price on the game whatever its state: kept for the record once it is played
        p_line = p_ml if p_ml is not None else (float(norm.cdf(line_sp / sigma)) if line_sp is not None else None)
        if g.played:
            p_home = 1.0 if g.result > 0 else 0.0
            p_away = 1.0 if g.result < 0 else 0.0          # a tie is a loss for both sides
            s_home = line_sp if line_sp is not None else model_sp
            var_h = 0.0; src = "result"
        elif h == 0 and line_sp is not None:
            p_home = p_ml if p_ml is not None else float(norm.cdf(line_sp / sigma))
            s_home = line_sp; var_h = 0.0; src = line_src
        elif line_sp is not None:
            # a posted line is the truth at any horizon; the only uncertainty left is how far the
            # line can still move before kickoff (injuries, news), which grows with the horizon
            s_home = line_sp
            var_h = disc * max(la + lb * h, 0.0)
            src = f"posted-{line_src}"
            if w == season_weeks and h >= 1:
                # a week-18 line posted early cannot know who will rest starters
                s_home *= w18_shrink; var_h += w18_var; src += "+wk18"
                s_home -= rest_pen * rest_risk.get(g.home, 0.0)
                s_home += rest_pen * rest_risk.get(g.away, 0.0)
            p_home = float(norm.cdf(s_home / np.sqrt(sigma ** 2 + var_h)))
        else:
            s_home = model_sp
            var_h = disc * max(a + b * h + c / (1.0 + current_week), 0.0) if h >= 1 else max(a + c / (1.0 + current_week), 0.0)
            src = "model"
            if w == season_weeks and h >= 1:
                s_home *= w18_shrink; var_h += w18_var; src += "+wk18"
                # explicit rest risk: expected points lost by a team likely to sit starters
                s_home -= rest_pen * rest_risk.get(g.home, 0.0)
                s_home += rest_pen * rest_risk.get(g.away, 0.0)
            if abs(eff[w, ih]) > 0.05 or abs(eff[w, ia]) > 0.05:
                note = f"qb {g.home}:{eff[w, ih]:+.1f} {g.away}:{eff[w, ia]:+.1f}"
            p_home = float(norm.cdf(s_home / np.sqrt(sigma ** 2 + var_h)))
        if p_away is None:
            p_away = 1.0 - p_home
        for team, other, is_home, p, s in ((g.home, g.away, True, p_home, s_home), (g.away, g.home, False, p_away, -s_home)):
            ti = idx[team]
            prob[wi, ti] = p; spread[wi, ti] = s; lvar[wi, ti] = var_h
            has_game[wi, ti] = True; pickable[wi, ti] = not kicked; opp[wi, ti] = idx[other]
            rows.append({"week": w, "team": team, "opp": other, "home": is_home, "neutral": bool(g.neutral),
                         "kickoff": g.kickoff, "horizon": h, "spread": s, "line_var": var_h,
                         "prob": p, "source": src, "locked": bool(kicked), "played": bool(g.played),
                         "line_prob": (p_line if is_home else 1.0 - p_line) if p_line is not None else np.nan,
                         "model_spread": model_sp if is_home else -model_sp,
                         "line_spread": (line_sp if is_home else -line_sp) if line_sp is not None else np.nan,
                         "qb_note": note})
    table = pd.DataFrame(rows).sort_values(["week", "prob"], ascending=[True, False]).reset_index(drop=True)
    return Projection(season=season, current_week=current_week, season_weeks=season_weeks, weeks=weeks,
                      prob=prob, spread=spread, line_var=lvar, sigma=sigma, pickable=pickable,
                      has_game=has_game, table=table, opponent=opp, week18_shrink=w18_shrink)

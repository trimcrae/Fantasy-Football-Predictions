"""Replay past seasons with only the information available at the time.

For season S and week k the planner sees: closing lines for weeks <= k (the current week's
line is the price you could still act on before kickoff), results for weeks < k, market
ratings fitted from those lines, and nothing about the future. No QB ledger is used, so
the live tool (with a maintained ledger and lookahead lines) has strictly more information.

Two scores per season:
* realised   - did the entry / any entry actually survive, and when it was eliminated;
* expected   - survival probability of the picks *at closing prices*, a low-variance
               measure of pick quality that does not depend on how the coin flips landed.
"""
from __future__ import annotations

import copy
import datetime as dt

import numpy as np
import pandas as pd

from .data.schedule import regular_season
from .optimize.single import survival_prob
from .plan import commit_picks, make_plan
from .probability import game_home_prob
from .state import PoolState
from .teams import TEAMS


def as_of_view(games_all: pd.DataFrame, season: int, k: int) -> pd.DataFrame:
    """Hide everything the planner could not have known before week k kicked off."""
    df = games_all.copy()
    sel = (df["season"] == season) & (df["game_type"] == "REG")
    fut = sel & (df["week"] > k)
    df.loc[fut, ["spread_line", "home_moneyline", "away_moneyline"]] = np.nan
    cur = sel & (df["week"] >= k)
    df.loc[cur, ["result", "home_score", "away_score"]] = np.nan
    df.loc[cur, "home_win"] = np.nan
    df.loc[cur, "played"] = False
    # later seasons are irrelevant but harmless; drop them to keep the fit honest
    return df[df["season"] <= season]


def closing_prob(games: pd.DataFrame, week: int, team: str, sigma: float) -> float | None:
    g = games[(games["week"] == week) & ((games["home"] == team) | (games["away"] == team))]
    if g.empty:
        return None
    r = g.iloc[0]
    p = game_home_prob(r["home_moneyline"], r["away_moneyline"], r["spread_line"], sigma)
    if p is None:
        return None
    return p if r["home"] == team else 1 - p


def won(games: pd.DataFrame, week: int, team: str) -> bool | None:
    g = games[(games["week"] == week) & ((games["home"] == team) | (games["away"] == team))]
    if g.empty or not g.iloc[0]["played"]:
        return None
    r = g.iloc[0]
    return bool((r["result"] > 0 and r["home"] == team) or (r["result"] < 0 and r["away"] == team))


def default_scenarios(cfg: dict) -> int:
    """Policy planning chooses by simulation, so it needs more scenarios than fixed-path planning."""
    return 2000 if str((cfg.get("planning") or {}).get("mode", "policy")) == "policy" else 400


def run_season(games_all: pd.DataFrame, season: int, cfg: dict, mode: str = "single", strikes: int = 0,
               n_entries: int = 1, scenarios: int | None = None, policy: str = "plan", verbose: bool = False) -> dict:
    cfg = copy.deepcopy(cfg)
    scenarios = int(scenarios or default_scenarios(cfg))
    games = regular_season(games_all, season)
    W = int(games["week"].max())
    cfg["season_weeks"] = W
    state = PoolState(name=f"bt-{season}", mode="strikes" if mode == "strikes" else "multi",
                      n_entries=n_entries, strikes=strikes, season=season)
    sigma = float(cfg["model"]["sigma"])
    picks_log = []
    for k in range(1, W + 1):
        view = as_of_view(games_all, season, k)
        wk = games[games["week"] == k]
        now = wk["kickoff"].min() - dt.timedelta(minutes=1)
        if policy == "greedy":
            res = make_plan(state, view, cfg, [], None, now=now, season=season, week=k, source="market",
                            scenarios=64, compute_options=False, greedy=True, ignore_elimination=(mode != "multi"))
        else:
            res = make_plan(state, view, cfg, [], None, now=now, season=season, week=k, source="market",
                            scenarios=scenarios, compute_options=False, ignore_elimination=(mode != "multi"))
        commit_picks(res)
        for e in res.entries:
            if not e.alive or e.path is None:
                continue
            team = TEAMS[e.path.teams[0]]
            picks_log.append({"season": season, "week": k, "entry": e.entry_id, "team": team,
                              "p_model": float(e.path.probs[0]), "p_close": closing_prob(games, k, team, sigma),
                              "won": won(games, k, team)})
        if verbose:
            print(f"  {season} w{k}: " + ", ".join(f"#{r['entry']} {r['team']} ({r['p_close']:.2f}) {'W' if r['won'] else 'L'}" for r in picks_log if r["week"] == k))
    log = pd.DataFrame(picks_log)
    out = {"season": season, "mode": mode, "weeks": W}
    per_entry = []
    for eid, g in log.groupby("entry"):
        g = g.sort_values("week")
        losses = (~g["won"].astype(bool)).cumsum()
        elim_week = int(g["week"][losses > strikes].min()) if (losses > strikes).any() else None
        p_close = g["p_close"].astype(float).to_numpy()
        expected = float(np.prod(p_close)) if strikes == 0 else survival_prob(p_close, strikes)
        per_entry.append({"entry": eid, "survived": elim_week is None, "elim_week": elim_week,
                          "expected": expected, "log_expected": float(np.log(max(expected, 1e-12)))})
    pe = pd.DataFrame(per_entry)
    alive_by_week = {w: int(sum(1 for r in per_entry if r["elim_week"] is None or r["elim_week"] > w)) for w in (4, 8, 12, 16, W)}
    out.update({"any_survived": bool(pe["survived"].any()), "n_survived": int(pe["survived"].sum()),
                "mean_log_expected": float(pe["log_expected"].mean()), "expected_survivors": float(pe["expected"].sum()),
                "first_elim_week": int(pe["elim_week"].dropna().min()) if pe["elim_week"].notna().any() else None,
                "last_elim_week": int(pe["elim_week"].dropna().max()) if pe["elim_week"].notna().any() else None,
                "alive_by_week": alive_by_week})
    if mode == "multi":   # entries stop picking when eliminated, so per-entry expectations are not comparable
        out["mean_log_expected"] = float("nan"); out["expected_survivors"] = float("nan")
    out["picks"] = log
    return out


def run_backtest(games_all: pd.DataFrame, seasons: list[int], cfg: dict, mode: str, strikes: int = 0,
                 n_entries: int = 1, scenarios: int | None = None, policy: str = "plan", verbose: bool = False) -> pd.DataFrame:
    rows = []
    for s in seasons:
        r = run_season(games_all, s, cfg, mode=mode, strikes=strikes, n_entries=n_entries, scenarios=scenarios,
                       policy=policy, verbose=verbose)
        picks = r.pop("picks")
        r["picks"] = " ".join(f"{t}" for t in picks[picks["entry"] == picks["entry"].iloc[0]]["team"])
        rows.append(r)
        if verbose:
            print(f"{s}: survived={r['n_survived']} expected={np.exp(r['mean_log_expected']):.4f} elim_week={r['first_elim_week']}")
    return pd.DataFrame(rows)


def discount_sweep(games_all: pd.DataFrame, seasons: list[int], cfg: dict, mode: str, strikes: int,
                   discounts: list[float], verbose: bool = True) -> pd.DataFrame:
    rows = []
    for d in discounts:
        c = copy.deepcopy(cfg); c["model"]["future_discount"] = d
        bt = run_backtest(games_all, seasons, c, mode=mode, strikes=strikes)
        rows.append({"future_discount": d, "mean_log_expected": bt["mean_log_expected"].mean(),
                     "geo_mean_expected": float(np.exp(bt["mean_log_expected"].mean())),
                     "seasons_survived": int(bt["any_survived"].sum()), "n_seasons": len(bt)})
        if verbose:
            print(rows[-1])
    bt = run_backtest(games_all, seasons, cfg, mode=mode, strikes=strikes, policy="greedy")
    rows.append({"future_discount": "greedy (no lookahead)", "mean_log_expected": bt["mean_log_expected"].mean(),
                 "geo_mean_expected": float(np.exp(bt["mean_log_expected"].mean())),
                 "seasons_survived": int(bt["any_survived"].sum()), "n_seasons": len(bt)})
    if verbose:
        print(rows[-1])
    return pd.DataFrame(rows)


def horizon_sweep(games_all: pd.DataFrame, seasons: list[int], cfg: dict, mode: str, strikes: int,
                  horizons: list[int], n_entries: int = 1, scenarios: int | None = None, verbose: bool = True) -> pd.DataFrame:
    """Policy planning with each horizon, plus the fixed-path discount planner and greedy for reference."""
    rows = []

    def summarise(label, bt):
        row = {"planning": label, "seasons_survived": int(bt["any_survived"].sum()), "n_seasons": len(bt),
               "mean_survivors": float(bt["n_survived"].mean())}
        if mode != "multi":
            row["geo_mean_expected"] = float(np.exp(bt["mean_log_expected"].mean()))
        rows.append(row)
        if verbose:
            print(row)

    for h in horizons:
        c = copy.deepcopy(cfg); c["planning"] = dict(c.get("planning") or {}, mode="policy", horizon=int(h))
        summarise(f"policy h={h}", run_backtest(games_all, seasons, c, mode=mode, strikes=strikes, n_entries=n_entries, scenarios=scenarios))
    c = copy.deepcopy(cfg); c["planning"] = dict(c.get("planning") or {}, mode="discount")
    summarise(f"discount x{c['model'].get('future_discount', 1)}", run_backtest(games_all, seasons, c, mode=mode, strikes=strikes, n_entries=n_entries))
    summarise("greedy (no lookahead)", run_backtest(games_all, seasons, cfg, mode=mode, strikes=strikes, n_entries=n_entries, policy="greedy"))
    return pd.DataFrame(rows)

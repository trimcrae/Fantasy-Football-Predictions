"""Weekly planning: project, simulate, optimise, and explain."""
from __future__ import annotations

import copy
import datetime as dt
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .data.schedule import ET, current_week as _current_week, latest_season, regular_season
from .model.projection import Projection, build_projection
from .model.qb import QBSituation
from .model.strength import Strength, assemble
from .optimize.portfolio import EntryPlan, build_portfolio, portfolio_summary
from .optimize.simulate import Sim, path_alive, simulate_season, simulate_wins
from .optimize.single import Path, best_path, best_path_strikes, current_week_options, policy_options
from .state import EntryStatus, PoolState, evaluate_entries
from .teams import TEAMS


@dataclass
class PlanResult:
    season: int
    week: int
    state: PoolState
    projection: Projection
    strength: Strength
    entries: list[EntryPlan]
    statuses: list[EntryStatus]
    summary: dict
    options: list[Path] = field(default_factory=list)
    wins: np.ndarray | None = None            # calibrated simulation (reporting)
    wins_policy: np.ndarray | None = None     # discounted simulation (planning view; discount mode only)
    allocation_view: str = "planning"         # policy | planning | calibrated: what scored the split
    sim: Sim | None = None                    # calibrated simulation with per-scenario closing probabilities
    horizon: int | None = None                # policy mode: weeks that are a commitment; later weeks re-pick
    planning: dict = field(default_factory=dict)  # the planning config the result was built with

    # ---- convenience -------------------------------------------------------------
    def this_week(self) -> pd.DataFrame:
        rows = []
        status = {s.entry_id: s for s in self.statuses}
        for e in self.entries:
            st = status[e.entry_id]
            if e.alive and e.path is not None:
                t = TEAMS[e.path.teams[0]]
            elif not e.alive and st.locked_now is not None and self._died_this_week(st):
                t = st.locked_now          # the pick went in and lost: it stays on this week's record
            else:
                continue
            r = self.projection.row(self.week, t)
            rows.append({"entry": e.entry_id, "team": t, "opp": ("" if r["home"] else "@") + r["opp"],
                         "p_win": r["prob"], "p_line": r["line_prob"], "spread": r["spread"], "source": r["source"],
                         "kickoff": r["kickoff"].strftime("%a %m/%d %H:%M"),
                         "status": "locked" if st.locked_now == t else ("keep" if st.provisional_now == t else ("change" if st.provisional_now else "new")),
                         "p_season": e.path.value if e.path is not None else np.nan,
                         "p_season_sim": e.p_season() if e.path is not None and e.p_season() is not None else np.nan})
        return pd.DataFrame(rows)

    def status_of(self, entry_id: str) -> EntryStatus:
        return next(x for x in self.statuses if x.entry_id == entry_id)

    def strikes_left_of(self, entry_id: str) -> int:
        """Strikes the entry has left after every result in so far: 2 of 2 at the start of a
        two-strike pool, 1 after a loss, 0 when it is out. What to show, not what the optimiser
        plans with (``EntryPlan.strikes_left`` is the losses it can still take, see ``make_plan``)."""
        return max(self.state.strikes - self.status_of(entry_id).losses, 0)

    def decided_now(self, entry_id: str) -> bool:
        """This week's pick has kicked off and its result is in."""
        st = self.status_of(entry_id)
        return st.locked_now is not None and st.results.get(self.week) in ("win", "loss")

    def _died_this_week(self, st: EntryStatus) -> bool:
        """An entry that was still alive coming into this week and lost this week's game."""
        lost_now = st.results.get(self.week) in ("loss", "missed")
        return lost_now and st.losses - 1 <= self.state.lives

    def picks_by_team(self) -> pd.DataFrame:
        tw = self.this_week()
        if tw.empty:
            return tw
        g = tw.groupby(["team", "opp", "p_win", "spread", "source", "kickoff"]).agg(entries=("entry", "count")).reset_index()
        return g.sort_values(["entries", "p_win"], ascending=False).reset_index(drop=True)


def make_plan(state: PoolState, games_all: pd.DataFrame, cfg: dict, ledger: list[QBSituation],
              inpredictable: pd.DataFrame | None, now: dt.datetime | None = None, season: int | None = None,
              week: int | None = None, source: str | None = None, scenarios: int | None = None,
              objective: str = "any", overrides: dict | None = None, keep_wins: bool = False,
              compute_options: bool = True, greedy: bool = False, ignore_elimination: bool = False) -> PlanResult:
    season = int(season or state.season or cfg.get("season") or latest_season(games_all))
    games = regular_season(games_all, season)
    now = now or dt.datetime.now(tz=ET)
    week = int(week or _current_week(games, now))
    cfg = copy.deepcopy(cfg); cfg["season_weeks"] = min(int(cfg.get("season_weeks") or 18), int(games["week"].max()))
    planning = cfg.get("planning") or {}
    policy = str(planning.get("mode", "policy")) == "policy"
    horizon = max(int(planning.get("horizon", 1)), 1)
    if policy:
        # plug-in probabilities are the calibrated ones; the future is valued by re-picking, not by a variance hack
        cfg["model"]["future_discount"] = 1.0
    strength = assemble(games_all, season, week, cfg, ledger, inpredictable, source or cfg["model"]["ratings_source"])
    proj = build_projection(games, season, week, strength, cfg, now=now, overrides=overrides)
    statuses = evaluate_entries(state, games, week, now)

    P = proj.prob.copy()
    pickable_now = proj.pickable[0].copy()
    entries: list[EntryPlan] = []
    for s in statuses:
        available = np.array([t not in s.used for t in TEAMS])
        fixed = {}
        if s.locked_now is not None:
            fixed[0] = TEAMS.index(s.locked_now)
            available[fixed[0]] = True
        # A locked pick whose result is already in sits at P = 1 or 0 in the projection and the
        # simulation, so a loss there is charged by the simulation itself; the strikes the
        # optimiser plans with must not charge it a second time. (The strikes shown to the user
        # come from the results: PlanResult.strikes_left_of.)
        losses_ahead = s.losses - (1 if s.locked_now is not None and s.results.get(week) == "loss" else 0)
        entries.append(EntryPlan(entry_id=s.entry_id, available=available, fixed=fixed,
                                 strikes_left=max(state.lives - losses_ahead, 0), alive=s.alive or ignore_elimination))
    plan_disc = float(cfg["model"].get("future_discount", 1.0))
    real_disc = float(cfg["simulation"].get("discount", 1.0))
    sim = simulate_season(proj, cfg, n=scenarios, discount=real_disc)            # calibrated: reporting, and choosing in policy mode
    wins = sim.wins
    wins_policy = wins if policy or abs(plan_disc - real_disc) < 1e-9 else simulate_wins(proj, cfg, n=scenarios, discount=plan_disc)
    live = [e for e in entries if e.alive]
    options: list[Path] = []
    if greedy:
        # baseline policy: highest available probability this week, no lookahead
        for e in live:
            cand = np.where(e.available & pickable_now & (P[0] > 0))[0]
            if 0 in e.fixed:
                cand = np.array([e.fixed[0]])
            if len(cand) == 0:
                continue
            t = int(cand[np.argmax(P[0, cand])])
            e.path = Path(teams=[t] + [0] * (P.shape[0] - 1), probs=[float(P[0, t])] + [1.0] * (P.shape[0] - 1), value=float(P[0, t]))
            e.alive_mask = wins[:, 0, t]
    elif live and (compute_options or state.mode == "strikes" or len(live) == 1):
        e0 = live[0]
        if policy:
            options = policy_options(P, sim, e0.available, pickable_now, e0.strikes_left, e0.fixed, horizon,
                                     min_prob=float(cfg["model"].get("min_pick_prob", 0.0)))
        elif compute_options:
            options = current_week_options(P, e0.available, pickable_now, e0.strikes_left, e0.fixed,
                                           min_prob=float(cfg["model"].get("min_pick_prob", 0.0)))
            for o in options:
                o.detail["sim"] = float(path_alive(wins, o.teams, e0.strikes_left).mean())
    if greedy:
        pass
    elif state.mode == "strikes" or len(live) == 1:
        if live:
            e0 = live[0]
            if options:
                e0.path = options[0]
            else:
                e0.path = best_path_strikes(P, e0.available, e0.strikes_left, e0.fixed, pickable_now=pickable_now) if e0.strikes_left \
                    else best_path(P, e0.available, e0.fixed, pickable_now=pickable_now)
            if e0.path is not None:
                if policy and "mask" in e0.path.detail:
                    d = e0.path.detail
                    e0.alive_mask, e0.surv, e0.won, e0.picks = d["mask"], d["surv"], d["won"], d["picks"]
                else:
                    e0.alive_mask = path_alive(wins, e0.path.teams, e0.strikes_left)
    elif policy:
        build_portfolio(P, pickable_now, entries, sim, cfg, objective=objective, horizon=horizon)
    else:
        # allocation view: the discounted simulation (planning view, default) or the calibrated one
        alloc = wins if str(cfg.get("portfolio", {}).get("allocation_view", "planning")) == "calibrated" else wins_policy
        build_portfolio(P, pickable_now, entries, alloc, cfg, objective=objective)
        for e in entries:
            if e.alive and e.path is not None:
                e.alive_mask = path_alive(wins, e.path.teams, e.strikes_left)
    summary = portfolio_summary(entries, wins)
    if live and live[0].path is not None:
        summary["p_plugin_first"] = live[0].path.value
    return PlanResult(season=season, week=week, state=state, projection=proj, strength=strength, entries=entries,
                      statuses=statuses, summary=summary, options=options, wins=wins if keep_wins else None,
                      wins_policy=wins_policy if keep_wins and not policy else None,
                      allocation_view="policy" if policy else str(cfg.get("portfolio", {}).get("allocation_view", "planning")),
                      sim=sim if keep_wins else None, horizon=horizon if policy else None, planning=dict(planning, seed=int(cfg["simulation"]["seed"]) + 2))


def commit_picks(result: PlanResult) -> int:
    """Write this week's recommended picks into the state (locked picks are left alone)."""
    n = 0
    status = {s.entry_id: s for s in result.statuses}
    for e in result.entries:
        if not e.alive or e.path is None:
            continue
        if status[e.entry_id].locked_now is not None:
            continue
        team = TEAMS[e.path.teams[0]]
        result.state.picks.setdefault(e.entry_id, {})[result.week] = team
        n += 1
    return n


# ---------------------------------------------------------------------------------
# rendering
# ---------------------------------------------------------------------------------

def render(result: PlanResult, show_paths: bool = True, top_options: int = 12) -> str:
    out = []
    p = result.projection; st = result.strength; s = result.state
    n_lines = int(p.table[(p.table["source"].str.contains("moneyline|spread")) & p.table["home"]].shape[0])
    out.append(f"=== {s.name} | season {result.season}, week {result.week} (planning weeks {p.weeks[0]}-{p.weeks[-1]}) ===")
    out.append(f"ratings: {st.source} (market fit on {st.detail.get('n_lines')} posted lines, resid sd {st.detail.get('fit_resid_sd', float('nan')):.2f}); "
               f"games with a market price from here on: {n_lines}; QB situations: {int((np.abs(st.penalty) > 0).sum()) if st.penalty is not None else 0}")
    live = [e for e in result.entries if e.alive]
    dead = [e for e in result.entries if not e.alive]
    if s.mode == "strikes":
        e0 = live[0] if live else None
        if e0 is None:
            out.append("entry is eliminated.")
            return "\n".join(out)
        out.append(f"strikes left: {result.strikes_left_of(e0.entry_id)} of {s.strikes}")
        out.append(f"P(survive season) following the plan: {result.summary['p_each'][0]:.3f} (simulated; plan score {result.summary.get('p_plugin_first', float('nan')):.3f})")
    else:
        out.append(f"entries alive: {len(live)} of {len(result.entries)}; eliminated: {len(dead)}")
        out.append(f"P(at least one entry survives the season) = {result.summary['p_any']:.3f}; expected survivors = {result.summary['expected_survivors']:.2f}")
    # this week's board
    board = p.table[(p.table["week"] == result.week)].copy()
    board = board.sort_values("prob", ascending=False)
    out.append("")
    out.append(f"-- week {result.week} board (win probability, source) --")
    for r in board.head(16).itertuples(index=False):
        loc = "" if r.home else "@"
        flag = " (kicked off)" if r.locked else ""
        out.append(f"  {r.team:<4} {loc}{r.opp:<4} p={r.prob:.3f} spread {r.spread:+5.1f} [{r.source}]{flag} {r.qb_note}")
    # options: value of using each team now
    if result.options:
        out.append("")
        out.append("-- this week's options: use the team now, then the best available each week --" if result.horizon
                   else "-- this week's options: use the team now, play the rest optimally --")
        out.append("  team  p_now   score    P(season)  plan")
        for o in result.options[:top_options]:
            path = " ".join(TEAMS[t] for t in o.teams[1:])
            out.append(f"  {TEAMS[o.teams[0]]:<4}  {o.detail.get('now_prob', 0):.3f}  {o.value:.4f}   {o.detail.get('sim', float('nan')):.4f}   {path}")
    # picks
    tw = result.this_week()
    out.append("")
    if s.mode == "strikes":
        if not tw.empty:
            r = tw.iloc[0]
            out.append(f"PICK week {result.week}: {r['team']} {r['opp']} p={r['p_win']:.3f} ({r['status']}, kickoff {r['kickoff']})")
    else:
        out.append(f"-- week {result.week} picks by team --")
        for r in result.picks_by_team().itertuples(index=False):
            out.append(f"  {r.team:<4} x{r.entries:<3} {r.opp:<5} p={r.p_win:.3f} spread {r.spread:+5.1f} [{r.source}] {r.kickoff}")
        status = {st.entry_id: st for st in result.statuses}
        changes = [(r["entry"], r["team"], status[r["entry"]].provisional_now) for r in tw.to_dict(orient="records") if r["status"] == "change"]
        locked = [(r["entry"], r["team"]) for r in tw.to_dict(orient="records") if r["status"] == "locked"]
        if changes:
            out.append("  changes vs the picks on file: " + ", ".join(f"#{e} {old}->{new}" for e, new, old in changes))
        if locked:
            out.append("  locked (already kicked off): " + ", ".join(f"#{e} {t}" for e, t in locked))
        out.append("  entries: " + " ".join(f"#{r['entry']}:{r['team']}" for r in tw.to_dict(orient="records")))
    if show_paths:
        out.append("")
        out.append("-- per-entry plan (this week first; later weeks are re-optimised every run) --")
        for e in result.entries:
            if not e.alive:
                out.append(f"  #{e.entry_id}: eliminated")
                continue
            if e.path is None:
                out.append(f"  #{e.entry_id}: no feasible path")
                continue
            parts = []
            for wi, (t, pr) in enumerate(zip(e.path.teams, e.path.probs)):
                parts.append(f"w{p.weeks[wi]}:{TEAMS[t]}({pr:.2f})")
            sim = e.p_season() if e.p_season() is not None else float("nan")
            out.append(f"  #{e.entry_id} [P(season)={sim:.3f} score={e.path.value:.3f}] " + " ".join(parts))
    return "\n".join(out)

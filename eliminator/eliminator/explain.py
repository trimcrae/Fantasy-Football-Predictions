"""Plain-language explanations for the numbers in a plan.

Everything here is derived from the plan result itself (projection table, ratings, options,
per-entry paths, simulation), so the sentences say what actually drove each number rather
than restating a formula. They are stored in the snapshot and shown on the site.
"""
from __future__ import annotations

import numpy as np

from .plan import PlanResult
from .teams import TEAMS


def _pct(x: float, d: int = 0) -> str:
    return f"{100 * float(x):.{d}f}%"


def _src(source: str) -> str:
    s = str(source)
    if s == "moneyline":
        return "From the betting odds"
    if s == "spread":
        return "From the point spread"
    if s == "result":
        return "From the final score"
    if s.startswith("posted-moneyline"):
        return "From the posted odds"
    if s.startswith("posted-"):
        return "From the posted spread"
    if s.startswith("model"):
        return "No line posted yet, so from our ratings" + (" (week 18, starters may rest)" if "wk18" in s else "")
    return s


def _home_text(row) -> str:
    if row["neutral"]:
        return "neutral site"
    return "at home" if row["home"] else "on the road"


def best_later_spot(res: PlanResult, team: str) -> tuple[int, str, float] | None:
    """Best pickable spot for the team after this week: (week, opponent, prob)."""
    p = res.projection
    ti = TEAMS.index(team)
    best = None
    for wi in range(1, len(p.weeks)):
        if not p.has_game[wi, ti]:
            continue
        pr = float(p.prob[wi, ti])
        if best is None or pr > best[2]:
            best = (int(p.weeks[wi]), TEAMS[int(p.opponent[wi, ti])] if p.opponent[wi, ti] >= 0 else "?", pr)
    return best


def team_rank(res: PlanResult, team: str) -> tuple[int, int]:
    """(rank of this week's spot among the team's own remaining games, number of those games)."""
    p = res.projection
    ti = TEAMS.index(team)
    probs = [float(p.prob[wi, ti]) for wi in range(len(p.weeks)) if p.has_game[wi, ti]]
    now = float(p.prob[0, ti])
    return 1 + sum(1 for x in probs if x > now), len(probs)


def _ordinal(n: int) -> str:
    return f"{n}{'th' if 10 <= n % 100 <= 20 else {1: 'st', 2: 'nd', 3: 'rd'}.get(n % 10, 'th')}"


def explain_probability(res: PlanResult, team: str, cfg: dict | None = None) -> str:
    """Why this week's win probability is what it is: the price, and any gap to the ratings."""
    r = res.projection.row(res.week, team)
    where = "neutral site" if r["neutral"] else ("at home" if r["home"] else "on the road")
    sp = float(r["spread"])
    side = f"favoured by {abs(sp):.1f}" if sp > 0 else (f"an underdog by {abs(sp):.1f}" if sp < 0 else "a pick'em")
    out = f"{_src(r['source'])}: {team} {side} {where}."
    if r.get("qb_note"):
        out += f" QB: {r['qb_note']}."
    return out


def explain_timing(res: PlanResult, team: str) -> str:
    """Why use the team now: where this week sits among the team's own remaining games."""
    later = best_later_spot(res, team)
    rank, n = team_rank(res, team)
    if later is None:
        return f"{team}'s last game of the season."
    w, opp, pr = later
    if rank == 1:
        return f"{team}'s best spot all season; next best is {_pct(pr)} (wk {w} vs {opp})."
    if res.horizon:
        return f"{team}'s {_ordinal(rank)}-best spot of {n}; best is {_pct(pr)} in wk {w} vs {opp}, but later weeks are worth whoever is best available then, not a named team."
    return f"{team}'s {_ordinal(rank)}-best spot of {n}; best is {_pct(pr)} in wk {w} vs {opp}, but far-off games are deliberately trusted less."


def explain_not_used(res: PlanResult, entry, team: str) -> str:
    """Bigger favourites this entry passed over, and where they went."""
    p = res.projection
    ti = TEAMS.index(team)
    pr = float(p.prob[0, ti])
    better = sorted([(TEAMS[j], float(p.prob[0, j])) for j in range(32)
                     if p.pickable[0, j] and entry.available[j] and p.prob[0, j] > pr + 0.02 and j != ti], key=lambda x: -x[1])
    if not better:
        return ""
    later_of = {TEAMS[t]: int(p.weeks[i]) for i, t in enumerate(entry.path.teams)} if entry.path else {}
    bits = [f"{t} {_pct(q)} saved for wk {later_of[t]}" if t in later_of else f"{t} {_pct(q)} left to other entries" for t, q in better[:3]]
    return "Passed: " + "; ".join(bits) + "."


def pool_add_values(res: PlanResult, detail: bool = False) -> dict:
    """What each option's team would add to the pool as its last entry: how often an entry on
    that team survives while every other entry is dead, scored exactly as the split was built.
    With ``detail`` each value is (mean, standard error)."""
    if res.horizon and res.sim is not None:
        out = _pool_add_values_policy(res)
    else:
        out = {i: (v, None) for i, v in _pool_add_values_paths(res).items()}
    return out if detail else {i: v for i, (v, _) in out.items()}


def _pool_add_values_policy(res: PlanResult) -> dict[int, tuple[float, float]]:
    from .optimize.portfolio import EntryPlan, _candidates, dead_given, make_scorer
    live = [e for e in res.entries if e.alive and e.path is not None and e.alive_mask is not None]
    if len(live) < 2:
        return {}
    counts: dict[int, int] = {}
    for e in live:
        counts[e.path.teams[0]] = counts.get(e.path.teams[0], 0) + 1
    star = min(live, key=lambda e: counts[e.path.teams[0]])
    others = [e for e in live if e is not star]
    dead_plain = ~np.vstack([e.alive_mask for e in others]).any(axis=0)
    scorer = make_scorer(res.sim, res.planning, len(live), res.planning.get("seed", 2))
    # the probe takes the star's place: same later-week pattern
    P = res.projection.prob
    usage = np.zeros(P.shape)
    for e in others:
        for w, t in enumerate(e.path.teams):
            usage[w, t] += 1.0
    rng = np.random.default_rng(11)
    out: dict[int, float] = {}
    for i, o in enumerate(res.options[:16]):
        t = o.teams[0]
        if not star.available[t]:
            continue
        probe = EntryPlan(entry_id="probe", available=star.available, fixed={**star.fixed, 0: int(t)}, strikes_left=star.strikes_left, slot=star.slot)
        cands = [o] if res.horizon == 1 else _candidates(P, res.projection.pickable[0], probe, usage, 40, rng, res.horizon)
        best = None
        for c in cands:
            picks, surv, won = scorer.run(probe, c)
            if probe.strikes_left == 0:
                x = surv * dead_given(others, picks)
            else:
                x = (((~won).sum(axis=1) <= probe.strikes_left) & dead_plain).astype(float)
            if best is None or x.mean() > best.mean():
                best = x
        if best is not None:
            out[i] = (float(best.mean()), float(best.std() / np.sqrt(len(best))))
    return out


def _pool_add_values_paths(res: PlanResult) -> dict[int, float]:
    """Fixed-path (discount) planning: what each option's team would add to the pool as its last entry.

    For the entry in the smallest group this week, and for each team on the options list,
    the portfolio's own candidate generator proposes diversified paths that start with that
    team; the value is the best candidate's survival in the seasons where every other entry
    is dead. (The single-entry-optimal path is not used on its own: it can coincide with a
    path an existing entry already follows, which would score zero by construction.)
    """
    from .optimize.portfolio import EntryPlan, _candidates
    from .optimize.simulate import path_alive
    # score on the same simulation the portfolio was built on, so the column agrees with the split
    W = res.wins if res.allocation_view == "calibrated" or res.wins_policy is None else res.wins_policy
    if W is None:
        return {}
    live = [e for e in res.entries if e.alive and e.path is not None]
    if len(live) < 2:
        return {}
    counts: dict[int, int] = {}
    for e in live:
        counts[e.path.teams[0]] = counts.get(e.path.teams[0], 0) + 1
    star = min(live, key=lambda e: counts[e.path.teams[0]])
    others = [e for e in live if e is not star]
    dead = ~np.vstack([path_alive(W, e.path.teams, e.strikes_left) for e in others]).any(axis=0)
    P = res.projection.prob
    usage = np.zeros(P.shape)
    for e in others:
        for w, t in enumerate(e.path.teams):
            usage[w, t] += 1.0
    rng = np.random.default_rng(11)
    out: dict[int, float] = {}
    for i, o in enumerate(res.options[:16]):
        t = o.teams[0]
        if not star.available[t]:
            continue
        fixed = dict(star.fixed); fixed[0] = int(t)
        probe = EntryPlan(entry_id="probe", available=star.available, fixed=fixed, strikes_left=star.strikes_left)
        cands = _candidates(P, res.projection.pickable[0], probe, usage, 40, rng)
        if not cands:
            continue
        out[i] = max(float((path_alive(W, c.teams, star.strikes_left) & dead).mean()) for c in cands)
    return out


def explain_hedge(res: PlanResult, entry, team: str, adds: dict[int, float] | None = None) -> str:
    """Why this team for an entry on its own: what its best path adds when every other entry is dead."""
    if res.state.mode == "strikes" or entry.path is None:
        return ""
    adds = pool_add_values(res, detail=True) if adds is None else adds
    if not adds:
        return ""
    by_team = {TEAMS[res.options[i].teams[0]]: (v if isinstance(v, tuple) else (v, None)) for i, v in adds.items()}
    if team not in by_team:
        return ""
    mine, mine_se = by_team[team]
    by_team = {t: v for t, (v, _) in by_team.items()}
    n = res.sim.n if res.sim is not None else (res.wins_policy if res.allocation_view != "calibrated" and res.wins_policy is not None else res.wins).shape[0]
    se = float(np.sqrt(2) * mine_se) if mine_se is not None else float(np.sqrt(2 * max(mine * (1 - mine), 1e-12) / n))
    others = sorted(((t, v) for t, v in by_team.items() if t != team), key=lambda x: -x[1])
    n_others = sum(1 for x in res.entries if x.alive and x is not entry)
    mine_alone = next((float(o.detail.get("sim") or 0) for o in res.options if TEAMS[o.teams[0]] == team),
                      float(entry.alive_mask.mean()) if entry.alive_mask is not None else 0.0)
    alone_rank = 1 + sum(1 for o in res.options if TEAMS[o.teams[0]] != team and (o.detail.get("sim") or 0) > mine_alone)
    how = "it survives" if res.horizon else "its best path survives"
    head = f"As the pool's last entry it adds {_pct(mine, 2)}: how often {how} while the other {n_others} entries are all dead"
    close = [f"{t} {_pct(v, 2)}" for t, v in others[:3] if abs(mine - v) < 2 * se]
    if close:
        return f"{head}, about level with {', '.join(close)}; a coin flip between them. On its own it would rank {_ordinal(alone_rank)}."
    if others and mine >= others[0][1]:
        return f"{head}, the most of any team ({others[0][0]} {_pct(others[0][1], 2)}). On its own it would rank {_ordinal(alone_rank)}."
    return f"{head}; {others[0][0]} would add {_pct(others[0][1], 2)}." if others else head + "."


def explain_exposure(res: PlanResult) -> str:
    """Multi-entry: why the entries are spread the way they are."""
    tw = res.this_week()
    if tw.empty:
        return ""
    counts = tw.groupby("team").size().sort_values(ascending=False)
    p = res.projection
    probs = p.table[(p.table["week"] == res.week) & (p.table["prob"] > 0)].sort_values("prob", ascending=False)["prob"].to_list()
    n_strong = sum(1 for x in probs if x >= 0.75)
    top = counts.index[0]
    r = p.row(res.week, top)
    board = (f"{n_strong} team{'s' if n_strong != 1 else ''} at 75%+, then {_pct(probs[n_strong])}" if n_strong and len(probs) > n_strong
             else f"best on the board {_pct(probs[0])}")
    return f"Board: {board}. Spreading further would lower P(at least one survives). If {r['opp']} wins, {counts.iloc[0]} entries go out together."


def explain_summary(res: PlanResult, cfg: dict | None = None) -> str:
    s = res.state
    live = [e for e in res.entries if e.alive and e.path is not None]
    if not live:
        return "No entries alive."
    n_weeks = len(res.projection.weeks)
    if s.mode == "strikes":
        e = live[0]
        if res.horizon:
            return f"Chance of at most {e.strikes_left} loss{'es' if e.strikes_left != 1 else ''} in {n_weeks} picks, from simulated seasons in which the entry takes the best team still available each later week."
        return f"Chance of at most {e.strikes_left} loss{'es' if e.strikes_left != 1 else ''} in {n_weeks} picks, from simulated seasons. The score treats later weeks as far less certain; it ranks plans and is not a forecast."
    sims = [e.p_season() for e in live if e.p_season() is not None]
    per = float(np.mean(sims)) if sims else float("nan")
    indep = 1 - (1 - per) ** len(live)
    if res.horizon:
        return (f"Chance that one of {len(live)} entries wins {n_weeks} straight, from simulated seasons in which the pool is re-split over the best available teams every week. "
                f"Each entry survives {_pct(per, 2)} on average; if they were independent that would be {_pct(indep, 1)}, but they share teams, so {_pct(res.summary['p_any'], 1)}.")
    return (f"Chance that one of {len(live)} entries wins {n_weeks} straight, from simulated seasons. Alone each survives {_pct(per, 2)}; "
            f"if they were independent that would be {_pct(indep, 1)}, but they share teams, so {_pct(res.summary['p_any'], 1)}.")


def explain_option(res: PlanResult, option, best, cfg: dict | None = None) -> str:
    """Why an alternative ranks where it does."""
    p = res.projection
    now, bnow = float(option.detail.get("now_prob", 0)), float(best.detail.get("now_prob", 0))
    if res.horizon:
        if option is best:
            return "Best season odds for a single entry: this week's win chance times what is left to pick from afterwards."
        sim, bsim = float(option.detail.get("sim") or 0), float(best.detail.get("sim") or 0)
        if option.detail.get("surv") is not None and best.detail.get("surv") is not None:
            d = option.detail["surv"] - best.detail["surv"]            # common random numbers
            se = float(d.std() / np.sqrt(len(d)))
        else:
            n = res.sim.n if res.sim is not None else float(((cfg or {}).get("simulation") or {}).get("scenarios", 20000))
            se = np.sqrt(max(sim * (1 - sim), 1e-12) / n + max(bsim * (1 - bsim), 1e-12) / n)
        out = f"{_pct(now)} now vs {_pct(bnow)}."
        if abs(sim - bsim) < 2 * se:
            return out + " Season odds within simulation noise of the top row."
        if now > bnow:
            return out + " Likelier now, but using this team costs more later than the top row's does."
        return out + " Less likely now, and what is left afterwards does not make up for it."
    if option is best:
        return "Best now and best schedule after for a single entry."
    diffs = [(i, option.teams[i], best.teams[i]) for i in range(1, min(len(option.teams), len(best.teams))) if option.teams[i] != best.teams[i]]
    out = f"{_pct(now)} now vs {_pct(bnow)}."
    if diffs:
        i, a, b = diffs[0]
        out += f" Later plan differs in {len(diffs)} week{'s' if len(diffs) != 1 else ''} (wk {int(p.weeks[i])}: {TEAMS[a]} {_pct(p.prob[i, a])} for {TEAMS[b]} {_pct(p.prob[i, b])}" + (", ...)." if len(diffs) > 1 else ").")
    else:
        out += " Same plan after."
    sim, bsim = option.detail.get("sim"), best.detail.get("sim")
    if sim is not None and bsim is not None:
        n = float(((cfg or {}).get("simulation") or {}).get("scenarios", 20000))
        se = np.sqrt(max(sim * (1 - sim), 1e-12) / n + max(bsim * (1 - bsim), 1e-12) / n)
        if abs(sim - bsim) < 2 * se:
            out += " P(season) is within simulation noise of the top row."
        elif sim > bsim:
            out += " Higher P(season) yet ranked lower: the score trusts this week's line more than later projections."
    return out


def explain_entry(res: PlanResult, entry) -> str:
    if not entry.alive or entry.path is None:
        return ""
    p = res.projection
    probs = np.array(entry.path.probs)
    i_min = int(np.argmin(probs))
    return f"Avg {_pct(probs.mean())} per pick; weakest wk {p.weeks[i_min]} {TEAMS[entry.path.teams[i_min]]} {_pct(probs[i_min])}."


def notes(res: PlanResult, cfg: dict | None = None) -> list[str]:
    """Counterintuitive things worth knowing, one line each."""
    out: list[str] = []
    p, st = res.projection, res.strength
    m = (cfg or {}).get("model", {})
    chk = (st.detail or {}).get("inpredictable_check")
    if chk and chk.get("reason"):
        out.append(f"Games with no line yet use team ratings backed out of the posted Vegas spreads; inpredictable's ratings were not used because the {chk['reason']}.")
    elif st.source == "inpredictable":
        out.append("Games with no line yet use inpredictable's betting-market power ratings.")
    else:
        out.append(f"Posted lines are used as-is at any horizon. Games with no line yet use team ratings backed out of the {st.detail.get('n_lines')} Vegas spreads posted so far.")
    tw = res.this_week()
    if not tw.empty:
        road = sorted({r["team"] for r in tw.to_dict(orient="records") if str(r["opp"]).startswith("@")})
        if road:
            out.append(f"Road pick{'s' if len(road) != 1 else ''} {', '.join(road)}: the line already includes home field.")
        for team in sorted(set(tw["team"])):
            r = p.row(res.week, team)
            if r.get("qb_note"):
                out.append(f"{team}: {r['qb_note']}.")
    if 18 in p.weeks:
        out.append(f"Week 18 spots are shrunk {int(round(100 * (1 - float(m.get('week18_shrink', 0.8)))))}% toward 50% for resting starters.")
    fit = m.get("posted_line_fit")
    if fit:
        out.append(f"How much a posted line can still move is fitted from {fit['n_obs']} archived lines (about {np.sqrt(fit['posted_line_var_a'] + fit['posted_line_var_b']):.1f} pts one week out).")
    else:
        out.append(f"How much a posted line can still move is a default (about {np.sqrt(float(m.get('posted_line_var_a', 1.0)) + float(m.get('posted_line_var_b', 1.0))):.1f} pts one week out) until enough archived lines have closed to fit it.")
    if res.horizon:
        out.append("Later weeks are not planned as named teams: season odds assume the entry takes the best team still available each week at the line at the time. "
                   "Simulated spreads widen as the season goes on, as real ones do, so the later menu is richer than today's board.")
    else:
        out.append("Percentages for later weeks are deliberately pulled toward 50% because far-off games are trusted much less; the season survival percentages are not.")
    n = res.sim.n if res.sim is not None else int(((cfg or {}).get("simulation") or {}).get("scenarios", 20000))
    surv = res.options[0].detail.get("surv") if res.options else None
    if surv is not None:
        se = 2 * 100 * float(np.sqrt(2) * surv.std() / np.sqrt(len(surv)))
        out.append(f"Season odds come from {n:,} simulated seasons, each scored exactly from its lines; differences under about {se:.2f} points are noise.")
    else:
        se = 2 * 100 * np.sqrt(0.01 * 0.99 / n)
        out.append(f"Survival percentages come from {n:,} simulated seasons; differences under about {se:.2f} points are noise.")
    out.append("A tie is a loss. Only this week's pick is a recommendation; later weeks re-solve every run.")
    return out


def explain_all(res: PlanResult, cfg: dict | None = None) -> dict:
    """Everything the site shows, keyed the way the snapshot stores it."""
    tw = res.this_week()
    picks: dict[str, dict] = {}
    entries_by_team = {}
    for r in tw.to_dict(orient="records"):
        entries_by_team.setdefault(r["team"], []).append(r["entry"])
    ent = {e.entry_id: e for e in res.entries}
    top = max(entries_by_team, key=lambda t: len(entries_by_team[t])) if entries_by_team else None
    adds = pool_add_values(res, detail=True) if res.state.mode != "strikes" and any(len(v) <= 2 for v in entries_by_team.values()) else {}
    for team, eids in entries_by_team.items():
        e0 = ent[str(eids[0])]
        picks[team] = {"probability": explain_probability(res, team, cfg), "timing": explain_timing(res, team),
                       "not_used": explain_not_used(res, e0, team),
                       "hedge": explain_hedge(res, e0, team, adds) if team != top and len(eids) <= 2 else ""}
    best = res.options[0] if res.options else None
    return {
        "summary": explain_summary(res, cfg),
        "exposure": explain_exposure(res) if res.state.mode != "strikes" else "",
        "picks": picks,
        "options": [explain_option(res, o, best, cfg) for o in res.options[:16]] if best else [],
        "entries": {e.entry_id: explain_entry(res, e) for e in res.entries},
        "notes": notes(res, cfg),
    }

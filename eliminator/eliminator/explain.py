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
        return "moneyline"
    if s == "spread":
        return "posted spread"
    if s == "result":
        return "final score"
    if s.startswith("blend"):
        return "posted line blended with the model"
    if s.startswith("model"):
        return "model only, no line yet" + (", week-18 rest shrink" if "wk18" in s else "")
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


def season_rank(res: PlanResult, week: int, team: str) -> tuple[int, int, int]:
    """(rank of this spot among all remaining spots, spots >= 75%, spots >= 70%)."""
    t = res.projection.table
    t = t[t["prob"] > 0]
    pr = float(t[(t["week"] == week) & (t["team"] == team)]["prob"].iloc[0])
    rank = int((t["prob"] > pr).sum()) + 1
    return rank, int((t["prob"] >= 0.75).sum()), int((t["prob"] >= 0.70).sum())


def explain_probability(res: PlanResult, team: str, cfg: dict | None = None) -> str:
    """Why this week's win probability is what it is: the price, and any gap to the ratings."""
    r = res.projection.row(res.week, team)
    where = "neutral site" if r["neutral"] else ("at home" if r["home"] else "on the road")
    out = f"{_src(r['source'])}, {team} {r['spread']:+.1f} {where}."
    if not np.isnan(r.get("line_spread", np.nan)) and abs(float(r["model_spread"]) - float(r["line_spread"])) >= 2.0:
        d = float(r["line_spread"]) - float(r["model_spread"])
        out += f" Market {abs(d):.1f} pts {'above' if d > 0 else 'below'} the ratings."
    if r.get("qb_note"):
        out += f" QB: {r['qb_note']}."
    return out


def explain_timing(res: PlanResult, team: str) -> str:
    """Why use the team now: rank of this spot, and the team's best later spot."""
    r = res.projection.row(res.week, team)
    later = best_later_spot(res, team)
    rank, _, _ = season_rank(res, res.week, team)
    head = "Best spot on the remaining schedule" if rank == 1 else f"#{rank} spot on the remaining schedule"
    if later is None:
        return f"{head}; no later game."
    w, opp, pr = later
    gap = float(r["prob"]) - pr
    if gap >= 0:
        return f"{head}; {team}'s next best is {_pct(pr)} (wk {w} vs {opp})."
    return f"{head}; {team} projects {_pct(pr)} in wk {w} vs {opp}, but that is discounted as far off."


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
        return f"Simulated chance of at most {e.strikes_left} loss{'es' if e.strikes_left != 1 else ''} in {n_weeks} picks. Score {_pct(e.path.value, 1)} is the same under the 16x future discount; it ranks plans, it is not a forecast."
    sims = [float(e.alive_mask.mean()) for e in live if e.alive_mask is not None]
    per = float(np.mean(sims)) if sims else float("nan")
    indep = 1 - (1 - per) ** len(live)
    return (f"Simulated chance one of {len(live)} entries wins {n_weeks} straight. Alone each survives {_pct(per, 2)}; "
            f"independent that would be {_pct(indep, 1)}, shared teams pull it to {_pct(res.summary['p_any'], 1)}.")


def explain_option(res: PlanResult, option, best) -> str:
    """Why an alternative ranks where it does."""
    p = res.projection
    if option is best:
        return "Best now and best schedule after."
    now, bnow = float(option.detail.get("now_prob", 0)), float(best.detail.get("now_prob", 0))
    diffs = [(i, option.teams[i], best.teams[i]) for i in range(1, min(len(option.teams), len(best.teams))) if option.teams[i] != best.teams[i]]
    out = f"{_pct(now)} now vs {_pct(bnow)}."
    if diffs:
        i, a, b = diffs[0]
        out += f" Later plan differs in {len(diffs)} week{'s' if len(diffs) != 1 else ''} (wk {int(p.weeks[i])}: {TEAMS[a]} {_pct(p.prob[i, a])} for {TEAMS[b]} {_pct(p.prob[i, b])}" + (", ...)." if len(diffs) > 1 else ").")
    else:
        out += " Same plan after."
    sim, bsim = option.detail.get("sim"), best.detail.get("sim")
    if sim is not None and bsim is not None and sim > bsim:
        out += " Higher P(season): plans are ranked on the discounted score."
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
        out.append(f"Ratings: market fit; inpredictable rejected ({chk['reason']}).")
    elif st.source == "inpredictable":
        out.append("Ratings: inpredictable.")
    else:
        out.append(f"Ratings: fitted from {st.detail.get('n_lines')} posted lines, seeded by last season regressed to average.")
    board = p.table[(p.table["week"] == res.week) & p.table["home"]]
    big = board[(board["line_spread"].notna()) & ((board["line_spread"] - board["model_spread"]).abs() >= 3.0)]
    for r in big.itertuples(index=False):
        d = float(r.line_spread) - float(r.model_spread)
        out.append(f"{r.team} vs {r.opp}: line {abs(d):.1f} pts {'above' if d > 0 else 'below'} the ratings; the line rules this week, ratings rule later weeks.")
    tw = res.this_week()
    if not tw.empty:
        board_all = p.table[(p.table["week"] == res.week) & (p.table["prob"] > 0)].sort_values("prob", ascending=False)
        top_team, top_p = board_all.iloc[0]["team"], float(board_all.iloc[0]["prob"])
        ent = {e.entry_id: e for e in res.entries}
        for team, g in tw.groupby("team"):
            if team == top_team:
                continue
            later, n_none = [], 0
            for eid in g["entry"]:
                e = ent[str(eid)]
                weeks = [int(p.weeks[i]) for i, t in enumerate(e.path.teams) if i > 0 and TEAMS[t] == top_team] if e.path else []
                if weeks:
                    later.append(weeks[0])
                else:
                    n_none += 1
            bits = []
            if later:
                bits.append(f"{len(later)} save{'s' if len(later) == 1 else ''} {top_team} for wk {', '.join(map(str, sorted(set(later))))}")
            if n_none:
                bits.append(f"{n_none} leave{'s' if n_none == 1 else ''} {top_team} to other entries")
            out.append(f"{len(g)} on {team} ({_pct(g['p_win'].iloc[0])}) not {top_team} ({_pct(top_p)}): " + "; ".join(bits) + ".")
        road = sorted({r["team"] for r in tw.to_dict(orient="records") if str(r["opp"]).startswith("@")})
        if road:
            out.append(f"Road pick{'s' if len(road) != 1 else ''} {', '.join(road)}: the line already includes home field.")
        for team in sorted(set(tw["team"])):
            r = p.row(res.week, team)
            if r.get("qb_note"):
                out.append(f"{team}: {r['qb_note']}.")
    if 18 in p.weeks:
        out.append(f"Week 18 spots are shrunk {int(round(100 * (1 - float(m.get('week18_shrink', 0.8)))))}% toward 50% for resting starters.")
    out.append(f"Later-week percentages are the planning view, discounted {float(m.get('future_discount', 16.0)):.0f}x toward 50%; survival percentages are simulated without that discount.")
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
    for team, eids in entries_by_team.items():
        e0 = ent[str(eids[0])]
        picks[team] = {"probability": explain_probability(res, team, cfg), "timing": explain_timing(res, team),
                       "not_used": explain_not_used(res, e0, team)}
    best = res.options[0] if res.options else None
    return {
        "summary": explain_summary(res, cfg),
        "exposure": explain_exposure(res) if res.state.mode != "strikes" else "",
        "picks": picks,
        "options": [explain_option(res, o, best) for o in res.options[:16]] if best else [],
        "entries": {e.entry_id: explain_entry(res, e) for e in res.entries},
        "notes": notes(res, cfg),
    }

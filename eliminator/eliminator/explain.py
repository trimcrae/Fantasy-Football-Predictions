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
        return f"Chance of at most {e.strikes_left} loss{'es' if e.strikes_left != 1 else ''} in {n_weeks} picks, from simulated seasons. The score treats later weeks as far less certain; it ranks plans and is not a forecast."
    sims = [float(e.alive_mask.mean()) for e in live if e.alive_mask is not None]
    per = float(np.mean(sims)) if sims else float("nan")
    indep = 1 - (1 - per) ** len(live)
    return (f"Chance that one of {len(live)} entries wins {n_weeks} straight, from simulated seasons. Alone each survives {_pct(per, 2)}; "
            f"if they were independent that would be {_pct(indep, 1)}, but they share teams, so {_pct(res.summary['p_any'], 1)}.")


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
    out.append("Percentages for later weeks are deliberately pulled toward 50% because far-off games are trusted much less; the season survival percentages are not.")
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

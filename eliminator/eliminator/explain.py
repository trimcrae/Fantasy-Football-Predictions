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
        return "the de-vigged moneyline"
    if s == "spread":
        return "the posted spread"
    if s == "result":
        return "the final score"
    if s.startswith("blend"):
        w = s[s.find(",") + 1:s.find(")")] if "," in s else ""
        return f"a posted line blended with the model (line weight {w})" if w else "a posted line blended with the model"
    if s.startswith("model"):
        return "the model alone (no line posted yet)" + (", shrunk for week 18 resting" if "wk18" in s else "")
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
    """Why this week's win probability for the team is what it is."""
    r = res.projection.row(res.week, team)
    st = res.strength
    hfa = float((cfg or {}).get("model", {}).get("hfa", 1.6))
    parts = [f"{_pct(r['prob'])} comes from {_src(r['source'])}: {team} {r['spread']:+.1f} {_home_text(r)} vs {r['opp']}."]
    rt, ro = float(st.healthy[team]), float(st.healthy[r["opp"]])
    hf = 0.0 if r["neutral"] else (hfa if r["home"] else -hfa)
    parts.append(f"Ratings say {team} {rt:+.1f} vs {r['opp']} {ro:+.1f}" + (f", home field {hf:+.1f}" if hf else "") + f" (model spread {r['model_spread']:+.1f}).")
    if not np.isnan(r.get("line_spread", np.nan)) and abs(float(r["model_spread"]) - float(r["line_spread"])) >= 2.0:
        d = float(r["line_spread"]) - float(r["model_spread"])
        parts.append(f"The market is {abs(d):.1f} points {'higher' if d > 0 else 'lower'} on {team} than the ratings, so the price carries information the ratings do not (injuries, news, or the ratings lagging).")
    if r.get("qb_note"):
        parts.append(f"QB: {r['qb_note']}.")
    return " ".join(parts)


def explain_timing(res: PlanResult, team: str) -> str:
    """Why use the team now rather than later."""
    r = res.projection.row(res.week, team)
    later = best_later_spot(res, team)
    rank, n75, n70 = season_rank(res, res.week, team)
    parts = []
    if rank == 1:
        parts.append("This is the single best spot on the whole remaining schedule")
    elif rank <= 5:
        parts.append(f"This is the #{rank} spot on the whole remaining schedule")
    else:
        parts.append(f"This is the #{rank} spot on the remaining schedule")
    parts[-1] += f" ({n75} spots at 75%+ and {n70} at 70%+ all season)."
    if later is None:
        parts.append(f"{team} has no later game to save them for.")
    else:
        w, opp, pr = later
        gap = float(r["prob"]) - pr
        if gap >= 0.08:
            parts.append(f"{team}'s best later spot is week {w} vs {opp} at {_pct(pr)}, {_pct(gap)} worse, so nothing comparable is given up.")
        elif gap >= 0:
            parts.append(f"{team}'s best later spot is week {w} vs {opp} at {_pct(pr)}, nearly as good; the plan prefers the sure thing now because future spots are discounted heavily.")
        else:
            parts.append(f"{team} projects better in week {w} vs {opp} ({_pct(pr)}), but that projection is weeks away and discounted; the plan still spends them now.")
    return " ".join(parts)


def explain_not_used(res: PlanResult, entry, team: str) -> str:
    """For an entry picking `team`, say why bigger favourites on the board were not used."""
    p = res.projection
    wi0 = 0
    ti = TEAMS.index(team)
    pr = float(p.prob[wi0, ti])
    better = [(TEAMS[j], float(p.prob[wi0, j])) for j in range(32)
              if p.pickable[wi0, j] and entry.available[j] and p.prob[wi0, j] > pr + 0.02 and j != ti]
    if not better:
        return ""
    better.sort(key=lambda x: -x[1])
    path = {int(p.weeks[i]): TEAMS[t] for i, t in enumerate(entry.path.teams)} if entry.path else {}
    later_of = {v: k for k, v in path.items()}
    notes = []
    for t, q in better[:3]:
        if t in later_of:
            w = later_of[t]
            wi = p.weeks.index(w)
            notes.append(f"{t} ({_pct(q)} now) is saved for week {w} vs {TEAMS[int(p.opponent[wi, TEAMS.index(t)])]} ({_pct(p.prob[wi, TEAMS.index(t)])})")
        else:
            notes.append(f"{t} ({_pct(q)} now) is left to other entries so this one does not die with them")
    return "Bigger favourites passed over: " + "; ".join(notes) + "."


def explain_exposure(res: PlanResult) -> str:
    """Multi-entry: why the entries are spread the way they are this week."""
    tw = res.this_week()
    if tw.empty:
        return ""
    counts = tw.groupby("team").size().sort_values(ascending=False)
    p = res.projection
    board = p.table[(p.table["week"] == res.week) & (p.table["prob"] > 0)].sort_values("prob", ascending=False)
    probs = board["prob"].to_list()
    n_strong = sum(1 for x in probs if x >= 0.75)
    drop = (probs[1] - probs[2]) if len(probs) > 2 else 0.0
    top = counts.index[0]
    parts = [f"{len(tw)} live entries across {len(counts)} teams; {counts.iloc[0]} on {top}."]
    if n_strong >= 1:
        parts.append(f"The board has {n_strong} team{'s' if n_strong != 1 else ''} at 75% or better, then it drops to {_pct(probs[n_strong])}"
                     + (f" (a {_pct(drop)} step after the top two)" if len(probs) > 2 and n_strong == 2 else "") + ".")
    parts.append("The objective is that at least one entry survives the season, so entries only spread onto weaker teams when that costs little; "
                 "moving a dozen entries onto a 64% team to avoid sharing would lower that probability.")
    r = p.row(res.week, top)
    parts.append(f"The flip side is correlation: if {r['opp']} wins, the {counts.iloc[0]} {top} entries go out together.")
    return " ".join(parts)


def explain_summary(res: PlanResult, cfg: dict | None = None) -> str:
    s = res.state
    live = [e for e in res.entries if e.alive and e.path is not None]
    if not live:
        return "No entries alive."
    disc = float((cfg or {}).get("model", {}).get("future_discount", 16.0))
    n_weeks = len(res.projection.weeks)
    if s.mode == "strikes":
        e = live[0]
        probs = np.array(e.path.probs)
        p_sim = float(e.alive_mask.mean()) if e.alive_mask is not None else float("nan")
        return (f"{_pct(p_sim, 1)} is the simulated chance of at most {e.strikes_left} loss{'es' if e.strikes_left != 1 else ''} over the {n_weeks} remaining picks, "
                f"with calibrated uncertainty about how good each team really is. "
                f"The per-pick numbers in the plan (averaging {_pct(probs.mean())}) are the planning view, which discounts later weeks {disc:.0f}x and so pulls them toward 50%; "
                f"under that pessimistic view the chance is the plan score, {_pct(e.path.value, 2)}, a ranking device rather than a forecast.")
    sims = [float(e.alive_mask.mean()) for e in live if e.alive_mask is not None]
    per = float(np.mean(sims)) if sims else float("nan")
    indep = 1 - (1 - per) ** len(live)
    return (f"{_pct(res.summary['p_any'], 1)} is the simulated chance that at least one of {len(live)} entries wins all {n_weeks} remaining picks. "
            f"Simulated one at a time the entries survive about {_pct(per, 2)} each, which would give {_pct(indep, 1)} if they were independent; "
            f"they share teams and die together, so it lands at {_pct(res.summary['p_any'], 1)}. "
            f"Expected survivors {res.summary['expected_survivors']:.2f} is the same simulation counted per entry. "
            f"The per-pick numbers in the plans are the planning view, discounted {disc:.0f}x for later weeks, so the product of a plan's picks is far below its simulated survival.")


def explain_option(res: PlanResult, option, best) -> str:
    """Why an alternative for this week ranks where it does."""
    p = res.projection
    t0, b0 = TEAMS[option.teams[0]], TEAMS[best.teams[0]]
    now, bnow = float(option.detail.get("now_prob", 0)), float(best.detail.get("now_prob", 0))
    if option is best:
        return f"Best available: {_pct(now)} now and the strongest remaining schedule after it."
    parts = [f"{_pct(now)} now vs {_pct(bnow)} for {b0}."]
    diffs = []
    for i in range(1, min(len(option.teams), len(best.teams))):
        if option.teams[i] != best.teams[i]:
            w = int(p.weeks[i])
            diffs.append(f"w{w} {TEAMS[option.teams[i]]} {_pct(p.prob[i, option.teams[i]])} instead of {TEAMS[best.teams[i]]} {_pct(p.prob[i, best.teams[i]])}")
    if not diffs:
        parts.append("The rest of the plan is identical, so the whole gap is this week's game.")
    else:
        parts.append(f"Later weeks shift in {len(diffs)} place{'s' if len(diffs) != 1 else ''}: " + "; ".join(diffs[:3]) + ("; ..." if len(diffs) > 3 else "") + ".")
    sim, bsim = option.detail.get("sim"), best.detail.get("sim")
    if sim is not None and bsim is not None and sim > bsim:
        parts.append("Simulated P(season) is higher than the top row's; plans are chosen on the discounted score, which trusts this week's price over distant projections.")
    return " ".join(parts)


def explain_entry(res: PlanResult, entry) -> str:
    if not entry.alive or entry.path is None:
        return ""
    p = res.projection
    probs = np.array(entry.path.probs)
    i_min, i_max = int(np.argmin(probs)), int(np.argmax(probs))
    sim = float(entry.alive_mask.mean()) if entry.alive_mask is not None else float("nan")
    return (f"{len(probs)} picks averaging {_pct(probs.mean())}; weakest w{p.weeks[i_min]} {TEAMS[entry.path.teams[i_min]]} {_pct(probs[i_min])}, "
            f"strongest w{p.weeks[i_max]} {TEAMS[entry.path.teams[i_max]]} {_pct(probs[i_max])}. "
            f"P(season) {_pct(sim, 2)} is simulated with calibrated uncertainty; score {_pct(entry.path.value, 2)} discounts the future and is only for ranking.")


def notes(res: PlanResult, cfg: dict | None = None) -> list[str]:
    """Counterintuitive things worth knowing about this week's plan."""
    out: list[str] = []
    p, st = res.projection, res.strength
    m = (cfg or {}).get("model", {})
    chk = (st.detail or {}).get("inpredictable_check")
    if chk and chk.get("reason"):
        out.append(f"Ratings come from the market-implied fit: inpredictable's ratings were rejected because the {chk['reason']}.")
    elif st.source == "inpredictable":
        out.append(f"Ratings come from inpredictable (agreeing with this season's lines to {chk['rmse']:.1f} points rms)." if chk else "Ratings come from inpredictable.")
    else:
        out.append(f"Ratings are fitted from the {st.detail.get('n_lines')} lines posted so far this season (residual {st.detail.get('fit_resid_sd', 0):.2f} points), seeded by last season regressed toward average.")
    # market vs ratings disagreements on this week's board
    board = p.table[(p.table["week"] == res.week) & p.table["home"]]
    big = board[(board["line_spread"].notna()) & ((board["line_spread"] - board["model_spread"]).abs() >= 3.0)]
    for r in big.itertuples(index=False):
        d = float(r.line_spread) - float(r.model_spread)
        out.append(f"{r.team} vs {r.opp}: the line ({r.line_spread:+.1f}) is {abs(d):.1f} points {'more' if d > 0 else 'less'} favourable to {r.team} than the ratings ({r.model_spread:+.1f}); the line wins for this week, the ratings drive later weeks.")
    # picks that are not the biggest favourite: where did the top team go in those entries' plans?
    tw = res.this_week()
    if not tw.empty:
        board_all = p.table[(p.table["week"] == res.week) & (p.table["prob"] > 0)].sort_values("prob", ascending=False)
        top_team, top_p = board_all.iloc[0]["team"], float(board_all.iloc[0]["prob"])
        ent = {e.entry_id: e for e in res.entries}
        for team, g in tw.groupby("team"):
            if team == top_team:
                continue
            later_weeks = []
            n_none = 0
            for eid in g["entry"]:
                e = ent[str(eid)]
                weeks = [int(p.weeks[i]) for i, t in enumerate(e.path.teams) if i > 0 and TEAMS[t] == top_team] if e.path else []
                if weeks:
                    later_weeks.append(weeks[0])
                else:
                    n_none += 1
            bits = []
            if later_weeks:
                wk = sorted(set(later_weeks))
                bits.append(f"{len(later_weeks)} of them use{'s' if len(later_weeks) == 1 else ''} {top_team} later (week{'s' if len(wk) > 1 else ''} {', '.join(map(str, wk))})")
            if n_none:
                bits.append(f"{n_none} never use{'s' if n_none == 1 else ''} {top_team}, leaving them to other entries so the group does not all die together")
            out.append(f"{len(g)} entr{'ies' if len(g) != 1 else 'y'} on {team} ({_pct(g['p_win'].iloc[0])}) rather than {top_team} ({_pct(top_p)}): " + "; ".join(bits) + ".")
    # road favourites among picks
    if not tw.empty:
        road = sorted({r["team"] for r in tw.to_dict(orient="records") if str(r["opp"]).startswith("@")})
        if road:
            out.append(f"Road pick{'s' if len(road) != 1 else ''} {', '.join(road)}: the line already includes home field (about {float(m.get('hfa', 1.6)):.1f} points), so the probability is not optimistic about it.")
    # week 18 in plans
    w18 = sum(1 for e in res.entries if e.alive and e.path is not None and p.weeks and p.weeks[-1] == 18)
    if w18 and 18 in p.weeks:
        out.append(f"Week 18 spots are shrunk toward 50% ({int(100 * (1 - float(m.get('week18_shrink', 0.8))))}% flatter) and given extra variance because clinched teams rest starters.")
    # QB situations touching this week's picks
    if not tw.empty:
        for team in sorted(set(tw["team"])):
            r = p.row(res.week, team)
            if r.get("qb_note"):
                out.append(f"{team}: {r['qb_note']}.")
    out.append(f"Every percentage for a later week on this page is the planning view: the projection's uncertainty is inflated {float(m.get('future_discount', 16.0)):.0f}x (backtest-tuned), "
               f"which pulls distant games toward 50% so a sure thing now beats a slightly better projection months away. The survival percentages are simulated with calibrated uncertainty instead.")
    out.append("A tie counts as a loss, and later weeks of every plan are re-optimised on each run; only this week's pick is a recommendation.")
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

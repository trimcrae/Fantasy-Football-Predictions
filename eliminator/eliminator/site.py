"""Weekly recommendation snapshots and the static site built from them (GitHub Pages).

Two commands:

* ``eliminator snapshot`` runs the planner for every pool and writes one JSON file per
  (season, week, pool) to ``site/data/<season>/w<NN>-<pool>.json``. The file for the
  current week is overwritten on every run, so it always holds the latest recommendation;
  once the season moves on to the next week the file is frozen and becomes the record of
  what was recommended. Before planning, picks the pool file is missing for weeks that
  have already kicked off are filled in from the snapshot of that week, so an unattended
  pipeline keeps the entries alive without anyone running ``plan --commit``. A pick
  entered by hand (``record``) is never overwritten.
* ``eliminator site`` renders every snapshot into a self-contained set of HTML pages:
  a landing page with this week's picks for each format and a week-by-week table with
  results, plus one page per week with the full board, options and per-entry plans.
"""
from __future__ import annotations

import datetime as dt
import html
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from .config import ROOT
from .data.schedule import ET
from .plan import PlanResult
from .state import PoolState
from .teams import TEAMS

SCHEMA = 1
SITE_DIR = ROOT / "site"
DATA_DIR = SITE_DIR / "data"
BUILD_DIR = SITE_DIR / "build"

FORMAT_LABEL = {"multi": "single elimination", "strikes": "two strikes"}


# ---------------------------------------------------------------------------------
# snapshots
# ---------------------------------------------------------------------------------

def snapshot_path(season: int, week: int, pool: str, data_dir: Path = DATA_DIR) -> Path:
    return data_dir / str(season) / f"w{week:02d}-{pool}.json"


def load_snapshot(path: Path) -> dict | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError:
        return None


def _f(x) -> float | None:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return None if np.isnan(v) else round(v, 4)


def build_snapshot(res: PlanResult, pool: str, generated_at: dt.datetime, previous: dict | None = None,
                   ledger: list | None = None, cfg: dict | None = None) -> dict:
    """Serialise a plan result. ``previous`` is the snapshot this one replaces (same week)."""
    p = res.projection
    st = res.strength
    s = res.state
    status = {x.entry_id: x for x in res.statuses}

    board = []
    for r in p.table[p.table["week"] == res.week].sort_values("prob", ascending=False).itertuples(index=False):
        board.append({"team": r.team, "opp": r.opp, "home": bool(r.home), "neutral": bool(r.neutral),
                      "prob": _f(r.prob), "spread": _f(r.spread), "source": str(r.source),
                      "kickoff": r.kickoff.isoformat(), "locked": bool(r.locked), "played": bool(r.played),
                      "qb_note": str(r.qb_note or "")})

    options = []
    for o in res.options[:16]:
        options.append({"team": TEAMS[o.teams[0]], "p_now": _f(o.detail.get("now_prob", 0.0)), "score": _f(o.value),
                        "p_season": _f(o.detail.get("sim")), "plan": [TEAMS[t] for t in o.teams[1:]]})

    picks = []
    for r in res.this_week().to_dict(orient="records"):
        picks.append({"entry": str(r["entry"]), "team": r["team"], "opp": r["opp"], "p_win": _f(r["p_win"]),
                      "spread": _f(r["spread"]), "source": r["source"], "kickoff": r["kickoff"], "status": r["status"],
                      "on_file": status[str(r["entry"])].provisional_now or status[str(r["entry"])].locked_now,
                      "score": _f(r["p_season"]), "p_season": _f(r["p_season_sim"])})

    plans = []
    for e in res.entries:
        row = {"entry": str(e.entry_id), "alive": bool(e.alive), "strikes_left": int(e.strikes_left)}
        if e.alive and e.path is not None:
            row["score"] = _f(e.path.value)
            row["p_season"] = _f(e.alive_mask.mean()) if e.alive_mask is not None else None
            row["path"] = [{"week": int(w), "team": TEAMS[t], "p": _f(pr)} for w, t, pr in zip(p.weeks, e.path.teams, e.path.probs)]
        plans.append(row)

    statuses = [{"entry": x.entry_id, "alive": bool(x.alive), "losses": int(x.losses),
                 "picks": {int(w): t for w, t in sorted(x.picks.items())},
                 "results": {int(w): r for w, r in sorted(x.results.items())}} for x in res.statuses]

    summary = {k: (v if isinstance(v, (int, list)) else _f(v)) for k, v in res.summary.items()}
    summary["p_each"] = [_f(v) for v in summary.get("p_each", [])]

    qb_situations = []
    if ledger and st.p_out is not None:
        for sit in ledger:
            ti = TEAMS.index(sit.team)
            pw = {int(w): _f(st.p_out[w, ti]) for w in p.weeks if w < st.p_out.shape[0] and st.p_out[w, ti] > 0.005}
            qb_situations.append({"team": sit.team, "player": sit.player, "status": sit.status, "injury": sit.injury,
                                  "penalty": _f(sit.penalty), "injured_week": sit.injured_week, "return_week": sit.return_week,
                                  "source": "auto" if str(sit.note).startswith("auto") else "manual", "p_out": pw})

    from .explain import explain_all
    explain = explain_all(res, cfg)

    revisions = list((previous or {}).get("revisions") or [])
    if previous and previous.get("generated_at"):
        revisions.append({"generated_at": previous["generated_at"], "picks": _pick_summary(previous.get("picks", []))})
    revisions = revisions[-20:]

    n_lines = int(p.table[(p.table["source"].str.contains("moneyline|spread")) & p.table["home"]].shape[0])
    return {
        "schema": SCHEMA, "pool": pool, "name": s.name, "mode": s.mode, "entries": int(s.n_entries),
        "strikes": int(s.strikes), "season": int(res.season), "week": int(res.week),
        "planning_weeks": [int(w) for w in p.weeks], "generated_at": generated_at.isoformat(),
        "ratings_source": st.source, "n_lines_posted": st.detail.get("n_lines"), "n_priced_games": n_lines,
        "summary": summary, "picks": picks, "options": options, "board": board, "plans": plans,
        "statuses": statuses, "ratings": {k: _f(v) for k, v in res.strength.healthy.items()},
        "qb_situations": qb_situations, "explain": explain,
        "revisions": revisions,
    }


def _pick_summary(picks: list[dict]) -> str:
    if not picks:
        return ""
    counts: dict[str, int] = {}
    for r in picks:
        counts[r["team"]] = counts.get(r["team"], 0) + 1
    if len(picks) == 1:
        return picks[0]["team"]
    return ", ".join(f"{t} x{n}" if n > 1 else t for t, n in sorted(counts.items(), key=lambda kv: -kv[1]))


def write_snapshot(snap: dict, data_dir: Path = DATA_DIR) -> Path:
    path = snapshot_path(snap["season"], snap["week"], snap["pool"], data_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(snap, indent=1, default=str) + "\n")
    return path


def backfill_state(state: PoolState, games: pd.DataFrame, week: int, now: dt.datetime,
                   data_dir: Path = DATA_DIR) -> int:
    """Fill picks the pool file lacks for weeks whose game has kicked off, from the snapshots.

    Returns the number of picks added. Existing picks are never changed.
    """
    added = 0
    for w in range(1, week + 1):
        snap = load_snapshot(snapshot_path(int(state.season), w, state.path.stem if state.path else "pool", data_dir))
        if not snap:
            continue
        for r in snap.get("picks", []):
            eid, team = str(r["entry"]), r["team"]
            if state.picks.get(eid, {}).get(w):
                continue
            g = games[(games["week"] == w) & ((games["home"] == team) | (games["away"] == team))]
            if g.empty or g.iloc[0]["kickoff"] > now:
                continue
            state.picks.setdefault(eid, {})[w] = team
            added += 1
    return added


# ---------------------------------------------------------------------------------
# grading
# ---------------------------------------------------------------------------------

def grade(games: pd.DataFrame | None, season: int, week: int, team: str) -> str:
    """'win' | 'loss' | 'pending' | 'unknown' for a team in a week (a tie is a loss)."""
    if games is None:
        return "unknown"
    g = games[(games["season"] == season) & (games["week"] == week) & ((games["home"] == team) | (games["away"] == team))]
    if g.empty:
        return "unknown"
    g = g.iloc[0]
    if not g["played"]:
        return "pending"
    won = (g["result"] > 0 and g["home"] == team) or (g["result"] < 0 and g["away"] == team)
    return "win" if won else "loss"


def week_record(snaps: list[dict], games: pd.DataFrame | None) -> dict[str, dict]:
    """Per pool: cumulative losses per entry after each snapshotted week, and who is alive."""
    out: dict[str, dict] = {}
    for pool in sorted({s["pool"] for s in snaps}):
        rows = sorted([s for s in snaps if s["pool"] == pool], key=lambda s: s["week"])
        losses: dict[str, int] = {}
        strikes = rows[0]["strikes"] if rows else 0
        n_entries = rows[0]["entries"] if rows else 0
        by_week = {}
        for s in rows:
            graded = {}
            for r in s["picks"]:
                res = grade(games, s["season"], s["week"], r["team"])
                graded[r["entry"]] = res
                if res == "loss":
                    losses[r["entry"]] = losses.get(r["entry"], 0) + 1
            alive = [e for e in (str(i) for i in range(1, n_entries + 1)) if losses.get(e, 0) <= strikes]
            by_week[s["week"]] = {"graded": graded, "alive_after": len(alive), "losses": dict(losses)}
        out[pool] = {"by_week": by_week, "strikes": strikes, "entries": n_entries}
    return out


# ---------------------------------------------------------------------------------
# rendering
# ---------------------------------------------------------------------------------

CSS = """
:root{--bg:#f7f7f5;--fg:#1c1c1a;--muted:#6b6b66;--card:#ffffff;--line:#e2e2dd;--accent:#1f5fbf;--win:#1b7f3b;--loss:#b3261e;--pend:#8a6d00;--lock:#555}
@media (prefers-color-scheme: dark){:root:not([data-theme="light"]){--bg:#141412;--fg:#ecece8;--muted:#a0a099;--card:#1e1e1b;--line:#333330;--accent:#7fb0ff;--win:#5fcf7f;--loss:#ff7b72;--pend:#e0b84d;--lock:#aaa}}
:root[data-theme="dark"]{--bg:#141412;--fg:#ecece8;--muted:#a0a099;--card:#1e1e1b;--line:#333330;--accent:#7fb0ff;--win:#5fcf7f;--loss:#ff7b72;--pend:#e0b84d;--lock:#aaa}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);font:15px/1.45 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif}
main{max-width:1100px;margin:0 auto;padding:20px 16px 48px}a{color:var(--accent);text-decoration:none}a:hover{text-decoration:underline}
h1{font-size:26px;margin:0 0 4px}h2{font-size:20px;margin:28px 0 10px}h3{font-size:16px;margin:18px 0 8px}
.sub{color:var(--muted);font-size:13px}.nav{font-size:14px;margin:6px 0 18px;color:var(--muted)}.nav a{margin-right:12px}
.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(320px,1fr));gap:16px}
.card{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:16px 18px}
.card h2{margin:0 0 4px;font-size:18px}.big{font-size:30px;font-weight:700;letter-spacing:.5px;margin:8px 0 2px}
.stat{display:inline-block;margin:6px 18px 0 0;font-size:13px;color:var(--muted)}.stat b{color:var(--fg);font-size:15px}
.tw{overflow-x:auto;margin:8px 0}table{border-collapse:collapse;width:100%;font-size:13.5px;font-variant-numeric:tabular-nums}
th,td{padding:5px 8px;border-bottom:1px solid var(--line);text-align:left;white-space:nowrap;vertical-align:top}th{color:var(--muted);font-weight:600;font-size:12.5px}
td.n,th.n{text-align:right}.win{color:var(--win);font-weight:600}.loss{color:var(--loss);font-weight:600}.pending{color:var(--pend)}.lock{color:var(--lock)}
.tag{display:inline-block;font-size:11px;padding:1px 6px;border:1px solid var(--line);border-radius:10px;color:var(--muted);margin-left:6px}
.grid td{font-size:12px;padding:3px 5px}.grid td.now{font-weight:700}.muted{color:var(--muted)}details{margin:10px 0}summary{cursor:pointer;color:var(--accent)}
.path{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:12.5px;white-space:normal}
.why{font-size:13px;color:var(--muted);margin:6px 0 10px;line-height:1.45;white-space:normal;max-width:900px}
tr.whyrow td{padding:0 8px 8px;border-bottom:1px solid var(--line)}tr.whyrow .why{margin:0}
td.whycell{white-space:normal;font-size:12.5px;color:var(--muted);min-width:260px;max-width:460px}
ul.notes{font-size:13.5px;line-height:1.5;padding-left:20px;max-width:900px}ul.notes li{margin:4px 0}
"""


def _esc(x) -> str:
    return html.escape("" if x is None else str(x))


def _pct(x, digits: int = 1) -> str:
    return "" if x is None else f"{100 * float(x):.{digits}f}%"


def _spread(x) -> str:
    return "" if x is None else f"{float(x):+.1f}"


def _dt(iso: str) -> str:
    try:
        t = dt.datetime.fromisoformat(iso).astimezone(ET)
        return t.strftime("%a %b %d, %H:%M ET")
    except (TypeError, ValueError):
        return iso


def _kick(iso: str) -> str:
    try:
        return dt.datetime.fromisoformat(iso).astimezone(ET).strftime("%a %m/%d %H:%M")
    except (TypeError, ValueError):
        return iso


def _res_cls(res: str) -> str:
    return {"win": "win", "loss": "loss", "pending": "pending"}.get(res, "muted")


def _res_mark(res: str) -> str:
    return {"win": "W", "loss": "L", "pending": "…"}.get(res, "")


def _page(title: str, body: str, subtitle: str = "", nav: str = "") -> str:
    return (f"<!doctype html><html lang=\"en\"><head><meta charset=\"utf-8\"><meta name=\"viewport\" content=\"width=device-width,initial-scale=1\">"
            f"<title>{_esc(title)}</title><style>{CSS}</style></head><body><main>"
            f"<h1>{_esc(title)}</h1><div class=\"sub\">{subtitle}</div><div class=\"nav\">{nav}</div>{body}</main></body></html>")


def _opp(r: dict) -> str:
    return ("vs " if r.get("home") else "@ ") + r["opp"] if "home" in r else r["opp"]


def _week_page_name(season: int, week: int) -> str:
    return f"s{season}-w{week:02d}.html"


def _season_page_name(season: int) -> str:
    return f"s{season}.html"


def _picks_cell(snap: dict, graded: dict[str, str]) -> str:
    """Compact description of a week's picks with results, e.g. 'LAC x12 W, JAX x11 L'."""
    if snap["mode"] == "strikes":
        if not snap["picks"]:
            return "<span class=\"muted\">eliminated</span>"
        r = snap["picks"][0]
        res = graded.get(r["entry"], "unknown")
        return f"<b>{_esc(r['team'])}</b> <span class=\"muted\">{_esc(_opp(r))} {_pct(r['p_win'])}</span> <span class=\"{_res_cls(res)}\">{_res_mark(res)}</span>"
    if not snap["picks"]:
        return "<span class=\"muted\">no entries alive</span>"
    counts: dict[str, list] = {}
    for r in snap["picks"]:
        counts.setdefault(r["team"], [0, r, graded.get(r["entry"], "unknown")])[0] += 1
    parts = []
    for team, (n, r, res) in sorted(counts.items(), key=lambda kv: (-kv[1][0], -(kv[1][1]["p_win"] or 0))):
        parts.append(f"<b>{_esc(team)}</b>&thinsp;x{n} <span class=\"muted\">{_pct(r['p_win'])}</span> <span class=\"{_res_cls(res)}\">{_res_mark(res)}</span>")
    return "<br>".join(parts)


def render_season_index(season: int, snaps: list[dict], games: pd.DataFrame | None, seasons: list[int],
                        built_at: dt.datetime) -> str:
    pools = sorted({s["pool"] for s in snaps}, key=lambda p: (0 if any(x["pool"] == p and x["mode"] == "multi" for x in snaps) else 1, p))
    rec = week_record(snaps, games)
    latest_week = max(s["week"] for s in snaps)
    latest = {p: next(s for s in snaps if s["pool"] == p and s["week"] == max(x["week"] for x in snaps if x["pool"] == p)) for p in pools}

    cards = []
    for pool in pools:
        s = latest[pool]
        gen = _dt(s["generated_at"])
        head = f"<h2>{_esc(s['name'])}</h2><div class=\"sub\">week {s['week']} &middot; {FORMAT_LABEL.get(s['mode'], s['mode'])} &middot; updated {_esc(gen)}</div>"
        if s["mode"] == "strikes":
            if s["picks"]:
                r = s["picks"][0]
                body = (f"<div class=\"big\">{_esc(r['team'])} <span class=\"muted\" style=\"font-size:18px;font-weight:400\">{_esc(_opp(r))}</span></div>"
                        f"<div class=\"sub\">win probability {_pct(r['p_win'])} &middot; spread {_spread(r['spread'])} &middot; kickoff {_esc(r['kickoff'])}"
                        + (f" &middot; <span class=\"lock\">locked</span>" if r['status'] == 'locked' else "") + "</div>")
                sl = next((pl["strikes_left"] for pl in s["plans"] if pl["alive"]), None)
                ex = s.get("explain") or {}
                pk = (ex.get("picks") or {}).get(r["team"]) or {}
                if pk.get("probability"):
                    body += f"<div class=\"why\">{_esc(pk['probability'])} {_esc(pk.get('timing', ''))}</div>"
                body += (f"<div><span class=\"stat\">P(survive season) <b>{_pct(s['summary'].get('p_each', [None])[0] if s['summary'].get('p_each') else None)}</b></span>"
                         f"<span class=\"stat\">strikes left <b>{sl if sl is not None else ''} of {s['strikes']}</b></span></div>")
                if ex.get("summary"):
                    body += f"<div class=\"why\">{_esc(ex['summary'])}</div>"
            else:
                body = "<div class=\"big muted\">eliminated</div>"
        else:
            rows = []
            counts: dict[str, list] = {}
            for r in s["picks"]:
                c = counts.setdefault(r["team"], [0, r, 0])
                c[0] += 1
                c[2] += 1 if r["status"] == "locked" else 0
            for team, (n, r, nlock) in sorted(counts.items(), key=lambda kv: (-kv[1][0], -(kv[1][1]["p_win"] or 0))):
                lock = f" <span class=\"lock\">({nlock} locked)</span>" if nlock else ""
                rows.append(f"<tr><td><b>{_esc(team)}</b></td><td class=\"n\">{n}{lock}</td><td>{_esc(_opp(r))}</td><td class=\"n\">{_pct(r['p_win'])}</td>"
                            f"<td class=\"n\">{_spread(r['spread'])}</td><td>{_esc(r['kickoff'])}</td></tr>")
            n_alive = s["summary"].get("n_live", 0)
            body = (f"<div class=\"tw\"><table><tr><th>team</th><th class=\"n\">entries</th><th>opponent</th><th class=\"n\">win prob</th><th class=\"n\">spread</th><th>kickoff</th></tr>{''.join(rows)}</table></div>"
                    if rows else "<div class=\"big muted\">no entries alive</div>")
            ex = s.get("explain") or {}
            if ex.get("exposure"):
                body += f"<div class=\"why\">{_esc(ex['exposure'])}</div>"
            body += (f"<div><span class=\"stat\">entries alive <b>{n_alive} of {s['entries']}</b></span>"
                     f"<span class=\"stat\">P(at least one survives) <b>{_pct(s['summary'].get('p_any'))}</b></span>"
                     f"<span class=\"stat\">expected survivors <b>{s['summary'].get('expected_survivors', 0):.2f}</b></span></div>")
            if ex.get("summary"):
                body += f"<div class=\"why\">{_esc(ex['summary'])}</div>"
            changes = [r for r in s["picks"] if r["status"] == "change"]
            if changes:
                body += f"<div class=\"sub\" style=\"margin-top:6px\">changes vs picks on file: " + ", ".join(f"#{_esc(r['entry'])} {_esc(r['on_file'])}&rarr;{_esc(r['team'])}" for r in changes) + "</div>"
        body += f"<div class=\"sub\" style=\"margin-top:10px\"><a href=\"{_week_page_name(season, s['week'])}\">full week {s['week']} report &rarr;</a></div>"
        cards.append(f"<div class=\"card\">{head}{body}</div>")

    # week-by-week table
    weeks = sorted({s["week"] for s in snaps})
    ths = "".join(f"<th>{_esc(latest[p]['name'])}</th><th class=\"n\">alive</th>" for p in pools)
    trs = []
    for w in weeks:
        cells = []
        for p in pools:
            s = next((x for x in snaps if x["pool"] == p and x["week"] == w), None)
            if s is None:
                cells.append("<td class=\"muted\">&mdash;</td><td></td>")
                continue
            wr = rec[p]["by_week"].get(w, {})
            cells.append(f"<td>{_picks_cell(s, wr.get('graded', {}))}</td><td class=\"n\">{wr.get('alive_after', '')}</td>")
        tag = " <span class=\"tag\">current</span>" if w == latest_week else ""
        trs.append(f"<tr><td><a href=\"{_week_page_name(season, w)}\">week {w}</a>{tag}</td>{''.join(cells)}</tr>")
    table = f"<div class=\"tw\"><table><tr><th>week</th>{ths}</tr>{''.join(trs)}</table></div>"
    legend = "<div class=\"sub\">W / L grade the recommended pick against the final score (a tie counts as a loss); &hellip; means the game has not been played. \"alive\" is the number of entries still in after that week, following the recommendations.</div>"

    nav = " ".join(f"<a href=\"{_season_page_name(y)}\">{y}</a>" if y != season else f"<b>{y}</b>" for y in seasons)
    body = (f"<div class=\"cards\">{''.join(cards)}</div><h2>Week by week</h2>{legend}{table}"
            f"<h2>About</h2><div class=\"sub\">Picks come from the <code>eliminator</code> planner: Vegas prices where they exist, market-implied ratings "
            f"projected forward with home field, rest and QB availability everywhere else, and a season-long optimisation per format. "
            f"Later weeks in a plan are re-optimised on every run; only this week's pick is a recommendation. "
            f"Raw snapshots: <a href=\"data/\">data/</a>. Built {_esc(built_at.astimezone(ET).strftime('%a %b %d %Y, %H:%M ET'))}.</div>")
    return _page(f"Eliminator picks {season}", body, subtitle="NFL survivor pool recommendations, week to week, for each pool format",
                 nav=f"seasons: {nav}")


def render_week_page(season: int, week: int, snaps: list[dict], games: pd.DataFrame | None, all_weeks: list[int]) -> str:
    snaps = sorted(snaps, key=lambda s: 0 if s["mode"] == "multi" else 1)
    sections = []
    for s in snaps:
        ex = s.get("explain") or {}
        graded = {r["entry"]: grade(games, season, week, r["team"]) for r in s["picks"]}
        sec = [f"<h2>{_esc(s['name'])} <span class=\"tag\">{FORMAT_LABEL.get(s['mode'], s['mode'])}</span></h2>",
               f"<div class=\"sub\">generated {_esc(_dt(s['generated_at']))} &middot; ratings: {_esc(s['ratings_source'])} "
               f"({_esc(s.get('n_lines_posted'))} posted lines) &middot; priced games from here on: {_esc(s.get('n_priced_games'))}</div>"]
        if s["mode"] == "strikes":
            if s["picks"]:
                r = s["picks"][0]
                res = graded.get(r["entry"], "unknown")
                sl = next((pl["strikes_left"] for pl in s["plans"] if pl["alive"]), None)
                sec.append(f"<div class=\"big\">{_esc(r['team'])} <span class=\"muted\" style=\"font-size:18px;font-weight:400\">{_esc(_opp(r))}</span> "
                           f"<span class=\"{_res_cls(res)}\" style=\"font-size:20px\">{_res_mark(res)}</span></div>"
                           f"<div class=\"sub\">win probability {_pct(r['p_win'])} &middot; spread {_spread(r['spread'])} &middot; kickoff {_esc(r['kickoff'])} &middot; {_esc(r['status'])}</div>"
                           f"<div><span class=\"stat\">P(survive season) <b>{_pct(s['summary']['p_each'][0] if s['summary'].get('p_each') else None)}</b></span>"
                           f"<span class=\"stat\">plan score <b>{_pct(s['summary'].get('p_plugin_first'), 2)}</b></span>"
                           f"<span class=\"stat\">strikes left <b>{sl if sl is not None else ''} of {s['strikes']}</b></span></div>")
                pk = (ex.get("picks") or {}).get(r["team"]) or {}
                for key in ("probability", "timing", "not_used"):
                    if pk.get(key):
                        sec.append(f"<div class=\"why\">{_esc(pk[key])}</div>")
                if ex.get("summary"):
                    sec.append(f"<div class=\"why\">{_esc(ex['summary'])}</div>")
            else:
                sec.append("<div class=\"big muted\">eliminated</div>")
        else:
            sec.append(f"<div><span class=\"stat\">entries alive <b>{s['summary'].get('n_live', 0)} of {s['entries']}</b></span>"
                       f"<span class=\"stat\">P(at least one survives) <b>{_pct(s['summary'].get('p_any'))}</b></span>"
                       f"<span class=\"stat\">expected survivors <b>{s['summary'].get('expected_survivors', 0):.2f}</b></span></div>")
            if ex.get("summary"):
                sec.append(f"<div class=\"why\">{_esc(ex['summary'])}</div>")
            rows = []
            counts: dict[str, list] = {}
            for r in s["picks"]:
                c = counts.setdefault(r["team"], [[], r])
                c[0].append(r["entry"])
            for team, (ents, r) in sorted(counts.items(), key=lambda kv: (-len(kv[1][0]), -(kv[1][1]["p_win"] or 0))):
                res = graded.get(ents[0], "unknown")
                rows.append(f"<tr><td><b>{_esc(team)}</b></td><td class=\"n\">{len(ents)}</td><td>{_esc(_opp(r))}</td><td class=\"n\">{_pct(r['p_win'])}</td>"
                            f"<td class=\"n\">{_spread(r['spread'])}</td><td>{_esc(r['source'])}</td><td>{_esc(r['kickoff'])}</td>"
                            f"<td class=\"{_res_cls(res)}\">{_res_mark(res)}</td><td class=\"muted\">{_esc(' '.join('#' + e for e in ents))}</td></tr>")
                pk = (ex.get("picks") or {}).get(team) or {}
                why = " ".join(pk.get(k, "") for k in ("probability", "timing", "not_used") if pk.get(k))
                if why:
                    rows.append(f"<tr class=\"whyrow\"><td colspan=\"9\"><div class=\"why\">{_esc(why)}</div></td></tr>")
            sec.append("<h3>Picks by team</h3><div class=\"tw\"><table><tr><th>team</th><th class=\"n\">entries</th><th>opponent</th><th class=\"n\">win prob</th>"
                       f"<th class=\"n\">spread</th><th>source</th><th>kickoff</th><th>result</th><th>entries</th></tr>{''.join(rows)}</table></div>")
            if ex.get("exposure"):
                sec.append(f"<div class=\"why\">{_esc(ex['exposure'])}</div>")
            changes = [r for r in s["picks"] if r["status"] == "change"]
            locked = [r for r in s["picks"] if r["status"] == "locked"]
            notes = []
            if changes:
                notes.append("changes vs picks on file: " + ", ".join(f"#{_esc(r['entry'])} {_esc(r['on_file'])}&rarr;{_esc(r['team'])}" for r in changes))
            if locked:
                notes.append("locked (kicked off): " + ", ".join(f"#{_esc(r['entry'])} {_esc(r['team'])}" for r in locked))
            if notes:
                sec.append("<div class=\"sub\">" + "<br>".join(notes) + "</div>")

        # options
        if s.get("options"):
            whys = ex.get("options") or []
            rows = "".join(f"<tr><td><b>{_esc(o['team'])}</b></td><td class=\"n\">{_pct(o['p_now'])}</td><td class=\"n\">{_pct(o['score'], 2)}</td>"
                           f"<td class=\"n\">{_pct(o['p_season'], 2)}</td><td class=\"path\">{_esc(' '.join(o['plan']))}</td>"
                           f"<td class=\"whycell\">{_esc(whys[i]) if i < len(whys) else ''}</td></tr>" for i, o in enumerate(s["options"]))
            sec.append("<h3>This week's options</h3><div class=\"sub\">Use the team now and play the rest of the season optimally. "
                       "Score is the discounted number the plan is chosen on; P(season) is the simulated survival probability, the number to believe.</div>"
                       f"<div class=\"tw\"><table><tr><th>team</th><th class=\"n\">win prob now</th><th class=\"n\">score</th><th class=\"n\">P(season)</th><th>rest of the plan</th><th>why</th></tr>{rows}</table></div>")

        # board
        rows = []
        for r in s["board"]:
            res = grade(games, season, week, r["team"]) if r.get("played") else ("pending" if r.get("locked") else "")
            flag = " <span class=\"lock\">kicked off</span>" if r.get("locked") and not r.get("played") else ""
            rows.append(f"<tr><td><b>{_esc(r['team'])}</b></td><td>{_esc(_opp(r))}</td><td class=\"n\">{_pct(r['prob'])}</td><td class=\"n\">{_spread(r['spread'])}</td>"
                        f"<td>{_esc(r['source'])}{flag}</td><td>{_esc(_kick(r['kickoff']))}</td><td class=\"{_res_cls(res)}\">{_res_mark(res)}</td><td class=\"muted\">{_esc(r.get('qb_note', ''))}</td></tr>")
        sec.append(f"<details><summary>Week {week} board ({len(s['board'])} teams)</summary><div class=\"tw\"><table><tr><th>team</th><th>opponent</th><th class=\"n\">win prob</th>"
                   f"<th class=\"n\">spread</th><th>source</th><th>kickoff</th><th>result</th><th>QB note</th></tr>{''.join(rows)}</table></div></details>")

        # per-entry plans
        pw = s.get("planning_weeks", [])
        if pw and any(pl.get("path") for pl in s["plans"]):
            head = "".join(f"<th class=\"n\">w{w}</th>" for w in pw)
            trs = []
            for pl in s["plans"]:
                if not pl["alive"]:
                    trs.append(f"<tr><td>#{_esc(pl['entry'])}</td><td class=\"muted\" colspan=\"{len(pw) + 2}\">eliminated</td></tr>")
                    continue
                if not pl.get("path"):
                    trs.append(f"<tr><td>#{_esc(pl['entry'])}</td><td class=\"muted\" colspan=\"{len(pw) + 2}\">no feasible path</td></tr>")
                    continue
                cells = "".join(f"<td class=\"n{' now' if i == 0 else ''}\" title=\"{_pct(step['p'])}\">{_esc(step['team'])}</td>" for i, step in enumerate(pl["path"]))
                ewhy = (ex.get("entries") or {}).get(pl["entry"], "")
                trs.append(f"<tr title=\"{_esc(ewhy)}\"><td>#{_esc(pl['entry'])}</td><td class=\"n\">{_pct(pl.get('p_season'), 2)}</td><td class=\"n\">{_pct(pl.get('score'), 2)}</td>{cells}</tr>")
            sec.append(f"<details><summary>Per-entry season plans ({sum(1 for pl in s['plans'] if pl['alive'])} alive)</summary>"
                       "<div class=\"sub\">This week's pick first. Later weeks only justify this week's pick and are re-optimised on every run. Hover a cell for the projected win probability, a row for its weakest and strongest links.</div>"
                       f"<div class=\"tw\"><table class=\"grid\"><tr><th>entry</th><th class=\"n\">P(season)</th><th class=\"n\">score</th>{head}</tr>{''.join(trs)}</table></div></details>")

        # entry status history
        sts = [x for x in s.get("statuses", []) if x.get("picks")]
        if sts:
            hist_weeks = sorted({int(w) for x in sts for w in x["picks"]})
            head = "".join(f"<th>w{w}</th>" for w in hist_weeks)
            trs = []
            for x in sts:
                cells = []
                for w in hist_weeks:
                    t = x["picks"].get(str(w), x["picks"].get(w))
                    r = x["results"].get(str(w), x["results"].get(w, ""))
                    cells.append(f"<td class=\"{_res_cls(r)}\">{_esc(t or '')}{(' ' + _res_mark(r)) if r else ''}</td>")
                trs.append(f"<tr><td>#{_esc(x['entry'])}</td><td>{'alive' if x['alive'] else '<span class=loss>out</span>'}</td><td class=\"n\">{x['losses']}</td>{''.join(cells)}</tr>")
            sec.append(f"<details><summary>Picks on file and results</summary><div class=\"tw\"><table class=\"grid\"><tr><th>entry</th><th>status</th><th class=\"n\">losses</th>{head}</tr>{''.join(trs)}</table></div></details>")

        if ex.get("notes"):
            items = "".join(f"<li>{_esc(n)}</li>" for n in ex["notes"])
            sec.append(f"<h3>Why these numbers</h3><ul class=\"notes\">{items}</ul>")
        if s.get("revisions"):
            items = "".join(f"<li>{_esc(_dt(r['generated_at']))}: {_esc(r['picks'])}</li>" for r in s["revisions"])
            sec.append(f"<details><summary>Earlier recommendations this week ({len(s['revisions'])})</summary><ul class=\"sub\">{items}</ul></details>")
        sections.append("".join(sec))

    # QB situations (shared)
    qbs = snaps[0].get("qb_situations") or []
    if qbs:
        rows = "".join(f"<tr><td><b>{_esc(q['team'])}</b></td><td>{_esc(q['player'])}</td><td>{_esc(q['status'])}</td><td>{_esc(q['injury'])}</td>"
                       f"<td class=\"n\">{q['penalty']:.1f}</td><td class=\"n\">{_esc(q.get('injured_week') or '')}</td><td class=\"n\">{_esc(q.get('return_week') or '')}</td>"
                       f"<td>{_esc(q['source'])}</td><td class=\"path\">{_esc(' '.join(f'w{w}:{int(round(100 * v))}%' for w, v in sorted(q['p_out'].items(), key=lambda kv: int(kv[0]))))}</td></tr>" for q in qbs)
        sections.append(f"<details open><summary>Quarterback situations ({len(qbs)})</summary>"
                        "<div class=\"sub\">Penalty is points off the team's rating while the starter is out; P(out) by week drives the projection. "
                        "\"auto\" entries come from the nflverse injury report at the default tier; an entry in state/qb_status.yaml replaces them.</div>"
                        f"<div class=\"tw\"><table><tr><th>team</th><th>player</th><th>status</th><th>injury</th><th class=\"n\">penalty</th><th class=\"n\">since</th><th class=\"n\">return</th><th>source</th><th>P(out) by week</th></tr>{rows}</table></div></details>")

    # ratings (shared)
    ratings = snaps[0].get("ratings") or {}
    if ratings:
        rows = "".join(f"<tr><td>{_esc(t)}</td><td class=\"n\">{float(v):+.1f}</td></tr>" for t, v in sorted(ratings.items(), key=lambda kv: -(kv[1] or 0)))
        sections.append(f"<details><summary>Team ratings (points vs average, healthy starter, neutral field)</summary><div class=\"tw\" style=\"max-width:320px\"><table><tr><th>team</th><th class=\"n\">rating</th></tr>{rows}</table></div></details>")

    idx = all_weeks.index(week)
    prev_link = f"<a href=\"{_week_page_name(season, all_weeks[idx - 1])}\">&larr; week {all_weeks[idx - 1]}</a>" if idx > 0 else ""
    next_link = f"<a href=\"{_week_page_name(season, all_weeks[idx + 1])}\">week {all_weeks[idx + 1]} &rarr;</a>" if idx + 1 < len(all_weeks) else ""
    nav = f"<a href=\"{_season_page_name(season)}\">&uarr; season {season}</a> {prev_link} {next_link}"
    return _page(f"Week {week}, {season}", "".join(sections), subtitle="recommendations for each pool format", nav=nav)


def load_snapshots(data_dir: Path = DATA_DIR) -> list[dict]:
    out = []
    for p in sorted(data_dir.glob("*/w*-*.json")):
        s = load_snapshot(p)
        if s and s.get("schema") == SCHEMA:
            out.append(s)
    return out


def build_site(games: pd.DataFrame | None, data_dir: Path = DATA_DIR, out_dir: Path = BUILD_DIR,
               built_at: dt.datetime | None = None) -> list[Path]:
    """Render every snapshot under ``data_dir`` into ``out_dir``. Returns the written files."""
    built_at = built_at or dt.datetime.now(tz=ET)
    snaps = load_snapshots(data_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    (out_dir / ".nojekyll").write_text("")
    if not snaps:
        p = out_dir / "index.html"
        p.write_text(_page("Eliminator picks", "<p class=\"sub\">No snapshots yet. Run <code>python -m eliminator snapshot</code> and rebuild.</p>"))
        return [p]
    seasons = sorted({s["season"] for s in snaps}, reverse=True)
    for season in seasons:
        ss = [s for s in snaps if s["season"] == season]
        page = render_season_index(season, ss, games, seasons, built_at)
        p = out_dir / _season_page_name(season)
        p.write_text(page)
        written.append(p)
        if season == seasons[0]:
            (out_dir / "index.html").write_text(page)
            written.append(out_dir / "index.html")
        weeks = sorted({s["week"] for s in ss})
        for w in weeks:
            p = out_dir / _week_page_name(season, w)
            p.write_text(render_week_page(season, w, [s for s in ss if s["week"] == w], games, weeks))
            written.append(p)
    # raw data alongside the pages
    dd = out_dir / "data"
    dd.mkdir(exist_ok=True)
    links = []
    for s in snaps:
        src = snapshot_path(s["season"], s["week"], s["pool"], data_dir)
        dst = dd / f"{s['season']}-{src.name}"
        dst.write_text(src.read_text())
        links.append(f"<li><a href=\"{dst.name}\">{dst.name}</a></li>")
        written.append(dst)
    (dd / "index.html").write_text(_page("Snapshots", f"<ul>{''.join(links)}</ul>", nav="<a href=\"../index.html\">&larr; picks</a>"))
    written.append(dd / "index.html")
    return written


def pool_files(state_dir: Path) -> list[Path]:
    """Pool state files: every YAML in state/ with a ``mode`` key."""
    out = []
    for p in sorted(state_dir.glob("*.yaml")):
        txt = p.read_text()
        if re.search(r"^mode:\s*\w+", txt, flags=re.M):
            out.append(p)
    return out

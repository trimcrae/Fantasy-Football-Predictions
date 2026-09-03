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
    from .explain import pool_add_values
    pool_add = pool_add_values(res) if s.mode != "strikes" else {}
    for i, o in enumerate(res.options[:16]):
        options.append({"team": TEAMS[o.teams[0]], "p_now": _f(o.detail.get("now_prob", 0.0)), "score": _f(o.value),
                        "p_season": _f(o.detail.get("sim")), "p_pool_add": _f(pool_add.get(i)), "plan": [TEAMS[t] for t in o.teams[1:]]})

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
            row["p_season"] = _f(e.p_season())
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
        "qb_situations": qb_situations, "explain": explain, "allocation_view": res.allocation_view,
        "horizon": res.horizon,
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
:root{color-scheme:light;--bg:#f6f6f4;--card:#ffffff;--line:#e6e5e1;--ink:#0b0b0b;--ink2:#52514e;--ink3:#8a8985;
--accent:#2a78d6;--track:#cde2fb;--good:#0ca30c;--bad:#d03b3b;--good-bg:#e8f6e8;--bad-bg:#fbe9e9;--chip:#f1f1ee;--now:#eef4fc}
@media (prefers-color-scheme: dark){:root:not([data-theme="light"]){color-scheme:dark;--bg:#151514;--card:#1f1f1d;--line:#33332f;--ink:#f4f4f1;--ink2:#c3c2b7;--ink3:#8f8e88;
--accent:#3987e5;--track:#213a5c;--good:#3fbf3f;--bad:#e66767;--good-bg:#1c2e1c;--bad-bg:#3a2020;--chip:#2a2a27;--now:#1d2a3c}}
:root[data-theme="dark"]{color-scheme:dark;--bg:#151514;--card:#1f1f1d;--line:#33332f;--ink:#f4f4f1;--ink2:#c3c2b7;--ink3:#8f8e88;
--accent:#3987e5;--track:#213a5c;--good:#3fbf3f;--bad:#e66767;--good-bg:#1c2e1c;--bad-bg:#3a2020;--chip:#2a2a27;--now:#1d2a3c}
*{box-sizing:border-box}html{-webkit-text-size-adjust:100%}
body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.5 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif}
a{color:var(--accent);text-decoration:none}a:hover{text-decoration:underline}
main{max-width:1040px;margin:0 auto;padding:28px 20px 64px}
.top{display:flex;flex-wrap:wrap;align-items:baseline;gap:8px 18px;margin-bottom:22px}
.top h1{font-size:15px;font-weight:600;letter-spacing:.02em;text-transform:uppercase;color:var(--ink2);margin:0}
.top .hero{font-size:40px;font-weight:700;letter-spacing:-.02em;line-height:1.05;margin:0;width:100%}
.top .meta{color:var(--ink3);font-size:13px}.top nav a{margin-right:10px;font-size:13px}.top nav b{margin-right:10px;font-size:13px}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(340px,1fr));gap:18px}
.card{background:var(--card);border:1px solid var(--line);border-radius:14px;padding:20px 22px}
.card h2{font-size:17px;font-weight:650;margin:0 0 2px}.sub{color:var(--ink3);font-size:12.5px}.tag{display:inline-block;font-size:11px;padding:1px 8px;border:1px solid var(--line);border-radius:999px;color:var(--ink2);margin-left:6px;vertical-align:middle}
.pick{display:flex;align-items:center;gap:14px;margin:16px 0 8px}
.pick img{width:56px;height:56px;border-radius:10px;flex:none}
.pick .who{flex:1;min-width:0}.pick .team{font-size:30px;font-weight:700;letter-spacing:-.01em;line-height:1.05}.pick .vs{color:var(--ink2);font-size:14px;margin-top:2px}
.pick .num{font-size:44px;font-weight:700;letter-spacing:-.03em;line-height:1;text-align:right}.pick .num small{display:block;font-size:11.5px;font-weight:500;color:var(--ink3);letter-spacing:.02em;text-transform:uppercase;margin-top:4px}
.why{color:var(--ink2);font-size:13.5px;line-height:1.45;margin:6px 0 0}.why.more{display:none}.showall .why.more{display:inline}.showall .m.why.more{display:block;margin-top:4px}
.opts .row{padding:9px 0}.opts .plan{display:inline-block;margin-top:3px}.r.two{display:flex;flex-direction:column;align-items:flex-end;gap:0;line-height:1.25}.r.two b{font-size:15px}
.tiles{display:grid;grid-template-columns:repeat(auto-fit,minmax(120px,1fr));gap:10px;margin:16px 0 6px}
.tile{background:var(--chip);border-radius:10px;padding:10px 12px}.tile .l{font-size:11.5px;color:var(--ink3);text-transform:uppercase;letter-spacing:.03em}.tile .v{font-size:22px;font-weight:650;letter-spacing:-.01em;margin-top:2px}.tile .v small{font-size:13px;font-weight:500;color:var(--ink3)}
.chips{display:flex;flex-wrap:wrap;gap:8px;margin:12px 0 4px}
.chip{display:inline-flex;align-items:center;gap:7px;background:var(--chip);border-radius:999px;padding:5px 12px 5px 6px;font-size:13.5px;font-weight:600}
.chip img{width:22px;height:22px;border-radius:50%}.chip .c{font-weight:500;color:var(--ink2)}.chip .p{font-weight:500;color:var(--ink3)}
.badge{display:inline-flex;align-items:center;gap:4px;font-size:11.5px;font-weight:650;padding:1px 7px;border-radius:999px}
.badge.win{color:var(--good);background:var(--good-bg)}.badge.loss{color:var(--bad);background:var(--bad-bg)}.badge.pending{color:var(--ink3);background:var(--chip)}
.meter{display:inline-flex;align-items:center;gap:8px;min-width:150px;width:150px}.meter .bar{display:block;flex:1;height:6px;border-radius:3px;background:var(--track);overflow:hidden}.meter .fill{display:block;height:100%;background:var(--accent);border-radius:0 3px 3px 0}.meter .mv{display:block;font-variant-numeric:tabular-nums;width:40px;text-align:right;font-size:13px;font-weight:600}
details{margin:14px 0 0}summary{cursor:pointer;color:var(--accent);font-size:13.5px;font-weight:500;list-style:none;display:inline-block}summary::-webkit-details-marker{display:none}summary::before{content:"\\25B8";margin-right:6px;font-size:11px}details[open]>summary::before{content:"\\25BE"}
.tw{overflow-x:auto;margin:10px 0 0}table{border-collapse:collapse;width:100%;font-size:13.5px}
th,td{padding:8px 10px;border-bottom:1px solid var(--line);text-align:left;white-space:nowrap;vertical-align:middle}th{color:var(--ink3);font-weight:600;font-size:11.5px;text-transform:uppercase;letter-spacing:.03em}
td.n,th.n{text-align:right;font-variant-numeric:tabular-nums}tr.now td{background:var(--now)}tbody tr:last-child td{border-bottom:none}
.tm{display:inline-flex;align-items:center;gap:7px;font-weight:600;vertical-align:middle}.tm img{width:22px;height:22px;border-radius:5px}.tm.s img{width:18px;height:18px}.opp{display:inline-flex;align-items:center;gap:5px;color:var(--ink2)}
.plan{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:12px;color:var(--ink2);white-space:normal;letter-spacing:.01em}
.reason{display:none;white-space:normal;color:var(--ink2);font-size:12.5px;max-width:420px}.showall .reason{display:table-cell}
.grid-t td{padding:4px 6px;font-size:12px;text-align:center}.grid-t td.now{font-weight:700;background:var(--now)}.grid-t td:first-child{text-align:left}
.notes{margin:14px 0 0;padding:0 0 0 18px;color:var(--ink2);font-size:13.5px;line-height:1.5}.notes li{margin:5px 0}
section.pool{margin:26px 0 0}section.pool>.card{margin-bottom:12px}
.tog{font:inherit;font-size:12.5px;color:var(--accent);background:none;border:1px solid var(--line);border-radius:999px;padding:3px 11px;cursor:pointer;margin-left:10px}
h3{font-size:14px;font-weight:650;margin:22px 0 4px}.wk-nav{display:flex;gap:8px;margin:6px 0 0}.wk-nav a{border:1px solid var(--line);border-radius:999px;padding:3px 12px;font-size:13px;background:var(--card)}
.picks-list .row{display:grid;grid-template-columns:110px 1fr auto;gap:4px 14px;align-items:center;padding:12px 0;border-bottom:1px solid var(--line)}.picks-list .row:last-child{border-bottom:none}
.picks-list .row .m{grid-column:1/-1}.picks-list .row .r{text-align:right;font-size:15px}.picks-list .row .mid{display:flex;flex-direction:column;gap:2px}
footer{color:var(--ink3);font-size:12.5px;margin-top:36px;line-height:1.5}
.wbw .wrow{display:grid;grid-template-columns:96px 1fr 1fr;gap:6px 18px;padding:14px 0;border-bottom:1px solid var(--line);align-items:start}.wbw .wrow:last-child{border-bottom:none}
.wbw .wrow.now{background:var(--now);margin:0 -22px;padding:14px 22px;border-radius:10px}.wbw .wk{font-weight:650}.wbw .wk .sub{display:block;font-weight:400}
.wbw .fmt{font-size:11.5px;color:var(--ink3);text-transform:uppercase;letter-spacing:.03em;margin-bottom:2px}.wbw .alive{font-size:12.5px;color:var(--ink2);margin-top:2px}.wbw .chips{margin:4px 0 0}
@media (max-width:640px){main{padding:16px 12px 44px}.top .hero{font-size:30px}.top{margin-bottom:14px}.grid{gap:12px}.card{padding:16px 14px;border-radius:12px}
.pick{gap:10px}.pick img{width:48px;height:48px}.pick .team{font-size:24px}.pick .num{font-size:34px}
.tiles{grid-template-columns:repeat(3,1fr);gap:8px}.tile{padding:8px 9px}.tile .v{font-size:17px}.tile .l{font-size:10px}
.meter{width:120px;min-width:120px}.picks-list .row{grid-template-columns:88px 1fr auto;gap:2px 10px}
.mh{display:none}.showall .mh{display:table-cell}th,td{padding:8px 7px}
.wbw .wrow{grid-template-columns:1fr}.wbw .wrow.now{margin:0 -14px;padding:12px 14px}.wbw .wk{display:flex;gap:8px;align-items:baseline}.wbw .wk .sub{display:inline}
summary{padding:6px 0}.tog{padding:5px 13px}}
"""

LOGO_DIR = SITE_DIR / "logos"


def _esc(x) -> str:
    return html.escape("" if x is None else str(x))


def _team(code, size: int = 22, cls: str = "tm") -> str:
    """Team code with its logo (the code stays for accessibility and copy/paste)."""
    c = _esc(code)
    return f"<span class=\"{cls}\"><img src=\"logos/{c}.png\" alt=\"\" width=\"{size}\" height=\"{size}\" loading=\"lazy\">{c}</span>"


def _opp_text(r: dict) -> tuple[str, str]:
    """('vs'|'@', code) from either a board row (home flag) or a pick row ('@IND')."""
    if "home" in r:
        return ("vs" if r.get("home") else "@"), str(r["opp"])
    opp = str(r["opp"])
    return ("@", opp[1:]) if opp.startswith("@") else ("vs", opp)


def _opp_html(r: dict, size: int = 18) -> str:
    pre, code = _opp_text(r)
    return f"<span class=\"opp\">{pre} {_team(code, size, 'tm s')}</span>"


def _source_label(src) -> str:
    s = str(src)
    return {"moneyline": "odds", "spread": "spread", "posted-moneyline": "odds (posted)", "posted-spread": "spread (posted)",
            "model+wk18": "model, wk 18", "result": "final"}.get(s, s)


def _pct(x, digits: int = 1) -> str:
    return "" if x is None else f"{100 * float(x):.{digits}f}%"


def _surv(x) -> str:
    """Season survival to a precision the simulation supports: 0.9%, 1.2%, 17.1%."""
    return _pct(x, 1)


def _spread(x) -> str:
    return "" if x is None else f"{float(x):+.1f}"


def _dt(iso: str) -> str:
    try:
        return dt.datetime.fromisoformat(iso).astimezone(ET).strftime("%a %b %d, %H:%M ET")
    except (TypeError, ValueError):
        return iso


def _kick(iso: str) -> str:
    try:
        return dt.datetime.fromisoformat(iso).astimezone(ET).strftime("%a %m/%d %H:%M")
    except (TypeError, ValueError):
        return iso


def _badge(res: str) -> str:
    if res == "win":
        return "<span class=\"badge win\">✓ W</span>"
    if res == "loss":
        return "<span class=\"badge loss\">✗ L</span>"
    if res == "pending":
        return "<span class=\"badge pending\">…</span>"
    return ""


def _meter(p, digits: int = 0) -> str:
    v = 0.0 if p is None else max(0.0, min(1.0, float(p)))
    return f"<span class=\"meter\"><span class=\"bar\"><span class=\"fill\" style=\"width:{100 * v:.1f}%\"></span></span><span class=\"mv\">{_pct(p, digits)}</span></span>"


def _tile(label: str, value: str, small: str = "") -> str:
    return f"<div class=\"tile\"><div class=\"l\">{_esc(label)}</div><div class=\"v\">{value}{(' <small>' + _esc(small) + '</small>') if small else ''}</div></div>"


def _page(title: str, body: str, hero: str, meta: str = "", nav: str = "") -> str:
    return (f"<!doctype html><html lang=\"en\"><head><meta charset=\"utf-8\"><meta name=\"viewport\" content=\"width=device-width,initial-scale=1\">"
            f"<title>{_esc(title)}</title><style>{CSS}</style></head><body><main>"
            f"<header class=\"top\"><h1>Eliminator picks</h1><nav>{nav}</nav><p class=\"hero\">{hero}</p><span class=\"meta\">{meta}</span></header>"
            f"{body}</main></body></html>")


def _week_page_name(season: int, week: int) -> str:
    return f"s{season}-w{week:02d}.html"


def _season_page_name(season: int) -> str:
    return f"s{season}.html"


def _hero_pick(r: dict, res: str = "") -> str:
    pre, opp = _opp_text(r)
    lock = " <span class=\"tag\">locked</span>" if r.get("status") == "locked" else ""
    return (f"<div class=\"pick\"><img src=\"logos/{_esc(r['team'])}.png\" alt=\"\"><div class=\"who\"><div class=\"team\">{_esc(r['team'])}{lock} {_badge(res)}</div>"
            f"<div class=\"vs\">{pre} {_esc(opp)} &middot; {_esc(r['kickoff'])} &middot; {_spread(r['spread'])}</div></div>"
            f"<div class=\"num\">{_pct(r['p_win'], 0)}<small>to win</small></div></div>")


def _exposure_chips(picks: list[dict], graded: dict[str, str] | None = None) -> str:
    counts: dict[str, list] = {}
    for r in picks:
        c = counts.setdefault(r["team"], [0, r])
        c[0] += 1
    chips = []
    for team, (n, r) in sorted(counts.items(), key=lambda kv: (-kv[1][0], -(kv[1][1]["p_win"] or 0))):
        pre, opp = _opp_text(r)
        res = (graded or {}).get(r["entry"], "")
        chips.append(f"<span class=\"chip\"><img src=\"logos/{_esc(team)}.png\" alt=\"\">{_esc(team)} <span class=\"c\">&times;{n}</span> "
                     f"<span class=\"p\">{pre} {_esc(opp)} &middot; {_pct(r['p_win'], 0)}</span> {_badge(res)}</span>")
    return f"<div class=\"chips\">{''.join(chips)}</div>"


def _pool_summary_tiles(s: dict) -> str:
    if s["mode"] == "strikes":
        sl = next((pl["strikes_left"] for pl in s["plans"] if pl["alive"]), None)
        p = s["summary"]["p_each"][0] if s["summary"].get("p_each") else None
        return "<div class=\"tiles\">" + _tile("Season survival", _pct(p)) + _tile("Strikes left", str(sl if sl is not None else 0), "of %d" % s["strikes"]) + "</div>"
    exp = "%.2f" % float(s["summary"].get("expected_survivors", 0) or 0)
    return ("<div class=\"tiles\">" + _tile("Entries alive", str(s["summary"].get("n_live", 0)), "of %d" % s["entries"])
            + _tile("Any entry survives", _pct(s["summary"].get("p_any"))) + _tile("Expected survivors", exp) + "</div>")


def _pick_why(s: dict, team: str, full: bool = False) -> str:
    """Timing sentence always; the price and the passed-over favourites only in details mode."""
    pk = ((s.get("explain") or {}).get("picks") or {}).get(team) or {}
    base = _esc(" ".join(x for x in (pk.get("timing", ""), pk.get("hedge", "")) if x))
    if not full:
        return base
    more = " ".join(pk.get(k, "") for k in ("probability", "not_used") if pk.get(k))
    return base + (f" <span class=\"why more\">{_esc(more)}</span>" if more else "")


def render_season_index(season: int, snaps: list[dict], games: pd.DataFrame | None, seasons: list[int],
                        built_at: dt.datetime) -> str:
    pools = sorted({s["pool"] for s in snaps}, key=lambda p: (0 if any(x["pool"] == p and x["mode"] == "multi" for x in snaps) else 1, p))
    rec = week_record(snaps, games)
    latest_week = max(s["week"] for s in snaps)
    latest = {p: next(s for s in snaps if s["pool"] == p and s["week"] == max(x["week"] for x in snaps if x["pool"] == p)) for p in pools}
    updated = max(s["generated_at"] for s in latest.values())

    cards = []
    for pool in pools:
        s = latest[pool]
        ex = s.get("explain") or {}
        head = f"<h2>{_esc(s['name'])}<span class=\"tag\">{_esc(FORMAT_LABEL.get(s['mode'], s['mode']))}</span></h2><div class=\"sub\">week {s['week']}</div>"
        if not s["picks"]:
            body = "<div class=\"pick\"><div class=\"who\"><div class=\"team\">Eliminated</div></div></div>"
        elif s["mode"] == "strikes":
            r = s["picks"][0]
            body = _hero_pick(r) + f"<p class=\"why\">{_pick_why(s, r['team'])}</p>"
        else:
            top = max(s["picks"], key=lambda r: r["p_win"] or 0)
            body = _hero_pick(top) + f"<p class=\"why\">{_pick_why(s, top['team'])}</p>" + _exposure_chips(s["picks"])
            if ex.get("exposure"):
                body += f"<p class=\"why\">{_esc(ex['exposure'])}</p>"
        body += _pool_summary_tiles(s)
        if ex.get("summary"):
            body += f"<details><summary>What the survival number means</summary><p class=\"why\">{_esc(ex['summary'])}</p></details>"
        body += f"<p class=\"sub\" style=\"margin-top:14px\"><a href=\"{_week_page_name(season, s['week'])}\">Full week {s['week']} report &rarr;</a></p>"
        cards.append(f"<div class=\"card\">{head}{body}</div>")

    weeks = sorted({s["week"] for s in snaps})
    rows = []
    for w in weeks:
        cols = []
        for p in pools:
            s = next((x for x in snaps if x["pool"] == p and x["week"] == w), None)
            if s is None:
                cols.append(f"<div><div class=\"fmt\">{_esc(latest[p]['name'])}</div><span class=\"sub\">&mdash;</span></div>")
                continue
            wr = rec[p]["by_week"].get(w, {})
            chips = _exposure_chips(s["picks"], wr.get("graded", {})) if s["picks"] else "<span class=\"sub\">no entries alive</span>"
            alive = f"<div class=\"alive\">{wr.get('alive_after', '')} of {s['entries']} alive after</div>" if s["mode"] != "strikes" else \
                (f"<div class=\"alive\">{'alive' if wr.get('alive_after') else 'out'}</div>" if wr else "")
            cols.append(f"<div><div class=\"fmt\">{_esc(latest[p]['name'])}</div>{chips}{alive}</div>")
        tag = "<span class=\"sub\">this week</span>" if w == latest_week else ""
        rows.append(f"<div class=\"wrow{' now' if w == latest_week else ''}\"><div class=\"wk\"><a href=\"{_week_page_name(season, w)}\">Week {w}</a>{tag}</div>{''.join(cols)}</div>")
    table = f"<div class=\"card wbw\"><h2>Week by week</h2><div class=\"sub\">Recommended picks, graded against the final score. A tie counts as a loss.</div>{''.join(rows)}</div>"

    nav = " ".join(f"<a href=\"{_season_page_name(y)}\">{y}</a>" if y != season else f"<b>{y}</b>" for y in seasons)
    body = (f"<div class=\"grid\">{''.join(cards)}</div><div style=\"height:18px\"></div>{table}"
            f"<footer>Vegas lines where they exist, market-implied ratings only for games without a line, and a season-long optimisation per format. "
            f"Later weeks of every plan re-solve on each run; only this week's pick is a recommendation. "
            f"<a href=\"data/\">Raw snapshots</a> &middot; built {_esc(built_at.astimezone(ET).strftime('%a %b %d %Y, %H:%M ET'))}.</footer>")
    return _page(f"Eliminator picks {season}", body, hero=f"Week {latest_week}", meta=f"{season} season &middot; updated {_esc(_dt(updated))}", nav=nav)


def render_week_page(season: int, week: int, snaps: list[dict], games: pd.DataFrame | None, all_weeks: list[int]) -> str:
    snaps = sorted(snaps, key=lambda s: 0 if s["mode"] == "multi" else 1)
    sections = []
    for s in snaps:
        ex = s.get("explain") or {}
        graded = {r["entry"]: grade(games, season, week, r["team"]) for r in s["picks"]}
        sid = f"pool-{_esc(s['pool'])}"
        head = (f"<h2>{_esc(s['name'])}<span class=\"tag\">{_esc(FORMAT_LABEL.get(s['mode'], s['mode']))}</span>"
                f"<button class=\"tog\" onclick=\"document.getElementById('{sid}').classList.toggle('showall')\">Details</button></h2>"
                f"<div class=\"sub\">generated {_esc(_dt(s['generated_at']))} &middot; {_esc(s.get('n_priced_games'))} games priced by Vegas from here on</div>")
        card = [head]
        if not s["picks"]:
            card.append("<div class=\"pick\"><div class=\"who\"><div class=\"team\">Eliminated</div></div></div>")
        elif s["mode"] == "strikes":
            r = s["picks"][0]
            card.append(_hero_pick(r, graded.get(r["entry"], "")))
            card.append(f"<p class=\"why\">{_pick_why(s, r['team'], full=True)}</p>")
        else:
            counts: dict[str, list] = {}
            for r in s["picks"]:
                counts.setdefault(r["team"], [[], r])[0].append(r["entry"])
            rows = []
            for team, (ents, r) in sorted(counts.items(), key=lambda kv: (-len(kv[1][0]), -(kv[1][1]["p_win"] or 0))):
                pre, opp = _opp_text(r)
                res = graded.get(ents[0], "")
                lock = " <span class=\"tag\">locked</span>" if r["status"] == "locked" else ""
                rows.append(f"<div class=\"row\"><div>{_team(team, 30)}{lock}</div><div class=\"mid\">{_meter(r['p_win'])}<span class=\"sub\">{pre} {_esc(opp)} &middot; {_esc(r['kickoff'])}</span></div>"
                            f"<div class=\"r\"><b>&times;{len(ents)}</b> {_badge(res)}</div>"
                            f"<div class=\"m\"><span class=\"why\">{_pick_why(s, team, full=True)}</span></div></div>")
            card.append(f"<div class=\"picks-list\">{''.join(rows)}</div>")
            if ex.get("exposure"):
                card.append(f"<p class=\"why\">{_esc(ex['exposure'])}</p>")
        card.append(_pool_summary_tiles(s))
        if ex.get("summary"):
            card.append(f"<details><summary>What the survival number means</summary><p class=\"why\">{_esc(ex['summary'])}</p></details>")
        sec = [f"<div class=\"card\">{''.join(card)}</div>"]

        # options
        if s.get("options"):
            whys = ex.get("options") or []
            rows = []
            multi = s["mode"] != "strikes" and any(o.get("p_pool_add") is not None for o in s["options"])
            n_live = s["summary"].get("n_live", 0)
            for i, o in enumerate(s["options"]):
                why = whys[i] if i < len(whys) else ""
                right = (f"<div class=\"r two\"><span><b>{_surv(o['p_season'])}</b><span class=\"sub\"> alone</span></span>"
                         f"<span><b>{_pct(o.get('p_pool_add'), 2)}</b><span class=\"sub\"> adds</span></span></div>") if multi else \
                        f"<div class=\"r\"><b>{_surv(o['p_season'])}</b><span class=\"sub\"> season</span></div>"
                policy = s.get("allocation_view") == "policy"
                lead = "" if policy else f"<span class=\"sub\">score {_pct(o['score'], 2)}</span> &middot; "
                rows.append(f"<div class=\"row\"><div>{_team(o['team'], 26)}</div><div class=\"mid\">{_meter(o['p_now'])}</div>{right}"
                            f"<div class=\"m why more\">{lead}{_esc(why)}<br><span class=\"plan\">{_esc(' '.join(o['plan']))}</span></div></div>")
            policy = s.get("allocation_view") == "policy"
            if multi and policy:
                intro = (f"One entry uses the team now and from then on takes the best team still available each week, at the line at the time. <b>Alone</b> is that entry's chance of surviving the season. "
                         f"<b>Adds</b> is what the same team adds to the pool as your {n_live}th entry: it counts only in the seasons where the other {n_live - 1} are all dead, "
                         f"so a team that survives in the same seasons as the crowd adds little. The split is built on Adds. Details shows the reasoning and a sketch of the later weeks.")
            elif multi:
                view = ("on the planning view, which trusts later weeks much less, so these are smaller than the survival numbers"
                        if s.get("allocation_view", "planning") != "calibrated" else "on the same simulation as the survival numbers")
                intro = (f"One entry uses the team now, then plays the rest of the season optimally. <b>Alone</b> is that one path's chance of surviving the season. "
                         f"<b>Adds</b> is what the best path behind that team adds to the pool as your {n_live}th entry: it counts only in the seasons where the other {n_live - 1} are all dead, "
                         f"so a path that survives in the same seasons as the crowd adds little. The split is built on Adds, scored {view}. Details shows the ranking score, the reasoning and the rest of the plan.")
            elif policy:
                intro = "Use the team now and from then on take the best team still available each week, at the line at the time. The right-hand number is the chance of surviving the season; Details adds the reasoning and a sketch of the later weeks."
            else:
                intro = "Use the team now and play the rest of the season optimally. The right-hand number is the chance of surviving the season; Details adds the ranking score, the reasoning and the rest of the plan."
            sec.append(f"<div class=\"card\"><h2>This week's options</h2><div class=\"sub\">{intro}</div><div class=\"picks-list opts\">{''.join(rows)}</div></div>")

        more = []
        # board
        rows = []
        for r in s["board"]:
            res = grade(games, season, week, r["team"]) if r.get("played") else ("pending" if r.get("locked") else "")
            rows.append(f"<tr><td>{_team(r['team'])}</td><td>{_opp_html(r)}</td><td>{_meter(r['prob'])}</td><td class=\"n\">{_spread(r['spread'])}</td>"
                        f"<td class=\"sub mh\">{_esc(_source_label(r['source']))}</td><td class=\"sub mh\">{_esc(_kick(r['kickoff']))}</td><td>{_badge(res)}</td><td class=\"sub mh\">{_esc(r.get('qb_note', ''))}</td></tr>")
        more.append(f"<details><summary>Week {week} board, all {len(s['board'])} teams</summary><div class=\"tw\"><table><thead><tr><th>team</th><th>opponent</th><th>win prob</th>"
                    f"<th class=\"n\">spread</th><th class=\"mh\">source</th><th class=\"mh\">kickoff</th><th>result</th><th class=\"mh\">QB</th></tr></thead><tbody>{''.join(rows)}</tbody></table></div></details>")
        # per-entry plans
        pw = s.get("planning_weeks", [])
        if pw and any(pl.get("path") for pl in s["plans"]):
            head_cells = "".join(f"<th>w{w}</th>" for w in pw)
            trs = []
            for pl in s["plans"]:
                if not pl["alive"]:
                    trs.append(f"<tr><td>#{_esc(pl['entry'])}</td><td class=\"sub\" colspan=\"{len(pw) + 1}\">eliminated</td></tr>")
                    continue
                if not pl.get("path"):
                    trs.append(f"<tr><td>#{_esc(pl['entry'])}</td><td class=\"sub\" colspan=\"{len(pw) + 1}\">no feasible path</td></tr>")
                    continue
                cells = "".join(f"<td class=\"{'now' if i == 0 else ''}\" title=\"{_pct(step['p'])}\">{_esc(step['team'])}</td>" for i, step in enumerate(pl["path"]))
                ewhy = (ex.get("entries") or {}).get(pl["entry"], "")
                trs.append(f"<tr title=\"{_esc(ewhy)}\"><td>#{_esc(pl['entry'])}</td><td class=\"n\">{_surv(pl.get('p_season'))}</td>{cells}</tr>")
            caption = ("This week is the pick. Later weeks are a sketch on today's lines: the entry actually takes the best team still available each week, and P(season) assumes that. Hover a cell for its probability."
                       if s.get("allocation_view") == "policy" else "This week first; later weeks only justify it and re-solve every run. Hover a cell for its probability.")
            more.append(f"<details><summary>Per-entry season plans</summary><div class=\"sub\">{caption}</div>"
                        f"<div class=\"tw\"><table class=\"grid-t\"><thead><tr><th>entry</th><th class=\"n\">P(season)</th>{head_cells}</tr></thead><tbody>{''.join(trs)}</tbody></table></div></details>")
        # picks on file
        sts = [x for x in s.get("statuses", []) if x.get("picks")]
        if sts:
            hist_weeks = sorted({int(w) for x in sts for w in x["picks"]})
            head_cells = "".join(f"<th>w{w}</th>" for w in hist_weeks)
            trs = []
            for x in sts:
                cells = []
                for w in hist_weeks:
                    t = x["picks"].get(str(w), x["picks"].get(w))
                    r = x["results"].get(str(w), x["results"].get(w, ""))
                    cells.append(f"<td>{_esc(t or '')} {_badge(r)}</td>")
                trs.append(f"<tr><td>#{_esc(x['entry'])}</td><td>{'alive' if x['alive'] else '<span class=badge.loss>out</span>'}</td><td class=\"n\">{x['losses']}</td>{''.join(cells)}</tr>")
            more.append(f"<details><summary>Picks on file and results</summary><div class=\"tw\"><table class=\"grid-t\"><thead><tr><th>entry</th><th>status</th><th class=\"n\">losses</th>{head_cells}</tr></thead><tbody>{''.join(trs)}</tbody></table></div></details>")
        if s.get("revisions"):
            items = "".join(f"<li>{_esc(_dt(r['generated_at']))}: {_esc(r['picks'])}</li>" for r in s["revisions"])
            more.append(f"<details><summary>Earlier recommendations this week ({len(s['revisions'])})</summary><ul class=\"notes\">{items}</ul></details>")
        notes = ""
        if ex.get("notes"):
            notes = f"<h3>Good to know</h3><ul class=\"notes\">{''.join(f'<li>{_esc(n)}</li>' for n in ex['notes'])}</ul>"
        sec.append(f"<div class=\"card\">{notes}{''.join(more)}</div>")
        sections.append(f"<section class=\"pool\" id=\"{sid}\">{''.join(sec)}</section>")

    # shared: QB situations and ratings
    shared = []
    qbs = snaps[0].get("qb_situations") or []
    if qbs:
        rows = "".join(f"<tr><td>{_team(q['team'])}</td><td>{_esc(q['player'])}</td><td>{_esc(q['status'])}</td><td>{_esc(q['injury'])}</td>"
                       f"<td class=\"n\">{q['penalty']:.1f}</td><td class=\"n\">{_esc(q.get('injured_week') or '')}</td><td class=\"n\">{_esc(q.get('return_week') or '')}</td>"
                       f"<td class=\"sub\">{_esc(q['source'])}</td><td class=\"plan\">{_esc(' '.join(f'w{w}:{int(round(100 * v))}%' for w, v in sorted(q['p_out'].items(), key=lambda kv: int(kv[0]))))}</td></tr>" for q in qbs)
        shared.append(f"<details open><summary>Quarterback situations ({len(qbs)})</summary><div class=\"sub\">Penalty is points off the team while the starter is out. Automatic entries come from the injury report; an entry in qb_status.yaml replaces them.</div>"
                      f"<div class=\"tw\"><table><thead><tr><th>team</th><th>player</th><th>status</th><th>injury</th><th class=\"n\">penalty</th><th class=\"n\">since</th><th class=\"n\">return</th><th>source</th><th>P(out) by week</th></tr></thead><tbody>{rows}</tbody></table></div></details>")
    ratings = snaps[0].get("ratings") or {}
    if ratings:
        rows = "".join(f"<tr><td>{_team(t)}</td><td class=\"n\">{float(v):+.1f}</td></tr>" for t, v in sorted(ratings.items(), key=lambda kv: -(kv[1] or 0)))
        shared.append(f"<details><summary>Team ratings used for games without a line</summary><div class=\"tw\" style=\"max-width:320px\"><table><thead><tr><th>team</th><th class=\"n\">points vs avg</th></tr></thead><tbody>{rows}</tbody></table></div></details>")
    if shared:
        sections.append(f"<section class=\"pool\"><div class=\"card\">{''.join(shared)}</div></section>")

    idx = all_weeks.index(week)
    prev_link = f"<a href=\"{_week_page_name(season, all_weeks[idx - 1])}\">&larr; Week {all_weeks[idx - 1]}</a>" if idx > 0 else ""
    next_link = f"<a href=\"{_week_page_name(season, all_weeks[idx + 1])}\">Week {all_weeks[idx + 1]} &rarr;</a>" if idx + 1 < len(all_weeks) else ""
    nav = f"<a href=\"{_season_page_name(season)}\">&uarr; Season {season}</a>"
    body = f"<div class=\"wk-nav\">{prev_link}{next_link}</div>" + "".join(sections)
    return _page(f"Week {week}, {season}", body, hero=f"Week {week}", meta=f"{season} season", nav=nav)


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
    if LOGO_DIR.exists():
        (out_dir / "logos").mkdir(exist_ok=True)
        for f in LOGO_DIR.glob("*.png"):
            (out_dir / "logos" / f.name).write_bytes(f.read_bytes())
    if not snaps:
        p = out_dir / "index.html"
        p.write_text(_page("Eliminator picks", "<p class=\"sub\">No snapshots yet. Run <code>python -m eliminator snapshot</code> and rebuild.</p>", hero="No picks yet"))
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
    (dd / "index.html").write_text(_page("Snapshots", f"<ul class=\"notes\">{''.join(links)}</ul>", hero="Raw snapshots", nav="<a href=\"../index.html\">&larr; picks</a>"))
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

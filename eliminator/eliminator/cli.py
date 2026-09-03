"""Command line interface. Run ``python -m eliminator --help`` from the eliminator/ directory."""
from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from .config import ROOT, load_config
from .data import inpredictable as inp
from .data.injuries import load_injuries, qb_watch
from .data.schedule import ET, current_week, latest_season, load_games, regular_season
from .model.qb import ledger_summary, load_ledger
from .model.projection import load_overrides
from .plan import commit_picks, make_plan, render
from .state import PoolState, evaluate_entries
from .teams import TEAMS, normalize

STATE_DIR = ROOT / "state"


def _seasons(arg: str | None, default: list[int]) -> list[int]:
    if not arg:
        return default
    if "-" in arg:
        a, b = arg.split("-")
        return list(range(int(a), int(b) + 1))
    return [int(x) for x in arg.split(",")]


def _now(args) -> dt.datetime:
    if getattr(args, "now", None):
        t = pd.Timestamp(args.now)
        return t.tz_localize(ET).to_pydatetime() if t.tzinfo is None else t.tz_convert(ET).to_pydatetime()
    return dt.datetime.now(tz=ET)


def _ratings(args, cfg, season):
    if getattr(args, "no_inpredictable", False):
        return None
    return inp.load_ratings(from_file=getattr(args, "inpredictable_file", None), season=season,
                            refresh=not getattr(args, "offline", False),
                            max_age_hours=cfg["data"]["max_age_hours"] * 4)


def cmd_update(args):
    cfg = load_config()
    df = load_games(refresh=not args.offline)
    season = int(args.season or cfg.get("season") or latest_season(df))
    games = regular_season(df, season)
    now = _now(args)
    week = current_week(games, now)
    print(f"schedule: season {season}, {len(games)} games, current week {week} (now {now:%Y-%m-%d %H:%M %Z})")
    cov = games.groupby("week").apply(lambda g: pd.Series({"games": len(g), "moneyline": int(g["home_moneyline"].notna().sum()),
                                                            "spread": int(g["spread_line"].notna().sum()), "played": int(g["played"].sum())}))
    print(cov.T.to_string())
    r = _ratings(args, cfg, season)
    if r is not None:
        print(f"\ninpredictable ratings ({r.attrs.get('source')}, fetched {r.attrs.get('fetched_at')}):")
        print("  " + "  ".join(f"{t}:{g:+.1f}" for t, g in zip(r['team'], r['gpf'])))
    else:
        print("\ninpredictable ratings: unavailable (market-implied ratings will be used)")
    if not args.offline:
        load_injuries(season, refresh=True)
    watch = qb_watch(season, week)
    print(f"\nQB injury report, week {week} (nflverse):")
    print(watch.to_string(index=False) if not watch.empty else "  none published yet")
    ledger = load_ledger(STATE_DIR / "qb_status.yaml", cfg["qb"])
    print(f"\nQB ledger (state/qb_status.yaml): {len(ledger)} situations")
    if ledger:
        print(ledger_summary(ledger, week, int(cfg.get('season_weeks') or 18), cfg["qb"]))
    flagged = set(watch[watch["is_qb1"] & watch["status"].isin(["Out", "Doubtful", "Questionable"])]["team"]) if not watch.empty else set()
    missing = flagged - {s.team for s in ledger}
    if missing:
        print(f"\n!! starters with a game designation but no ledger entry: {sorted(missing)} - add them to state/qb_status.yaml")


def cmd_calibrate(args):
    from .calibration import calibrate, save_calibration
    df = load_games(refresh=not args.offline)
    seasons = _seasons(args.seasons, list(range(2011, latest_season(df))))
    cal = calibrate(df, seasons=seasons, verbose=False)
    p = save_calibration(cal)
    print(json.dumps(cal["model"], indent=1))
    print(f"written {p}")


def cmd_plan(args):
    cfg = load_config()
    if args.scenarios:
        cfg["simulation"]["scenarios"] = args.scenarios
    if args.discount is not None:
        cfg["model"]["future_discount"] = args.discount
    df = load_games(refresh=not args.offline, max_age_hours=cfg["data"]["max_age_hours"])
    state = PoolState.load(Path(args.pool))
    season = int(args.season or state.season or cfg.get("season") or latest_season(df))
    state.season = season
    now = _now(args)
    ledger = load_ledger(STATE_DIR / "qb_status.yaml", cfg["qb"])
    overrides = load_overrides(STATE_DIR / "overrides.yaml")
    if not args.offline:
        from .data.odds_api import live_overrides
        games = regular_season(df, season)
        wk = args.week or current_week(games, now)
        live = live_overrides(cfg, games, wk)
        if live:
            manual = overrides.get("lines") or []
            overrides["lines"] = manual + [r for r in live if not any(m.get("home") == r["home"] and int(m.get("week", 0)) == r["week"] for m in manual)]
    ratings = _ratings(args, cfg, season)
    res = make_plan(state, df, cfg, ledger, ratings, now=now, season=season, week=args.week,
                    source=args.source, objective=args.objective, overrides=overrides)
    print(render(res, show_paths=not args.no_paths))
    if args.json:
        payload = {"season": res.season, "week": res.week, "summary": res.summary,
                   "picks": res.this_week().to_dict(orient="records"),
                   "board": res.projection.table[res.projection.table["week"] == res.week].drop(columns=["kickoff"]).to_dict(orient="records"),
                   "ratings": res.strength.healthy.round(2).to_dict()}
        Path(args.json).write_text(json.dumps(payload, indent=1, default=str))
        print(f"written {args.json}")
    if args.commit:
        n = commit_picks(res)
        state.save()
        print(f"committed {n} picks for week {res.week} to {state.path}")


def cmd_status(args):
    cfg = load_config()
    df = load_games(refresh=False)
    state = PoolState.load(Path(args.pool))
    season = int(args.season or state.season or cfg.get("season") or latest_season(df))
    games = regular_season(df, season)
    now = _now(args)
    week = args.week or current_week(games, now)
    for s in evaluate_entries(state, games, week, now):
        hist = " ".join(f"w{w}:{s.picks.get(w, '--')}{'✓' if s.results.get(w) == 'win' else ('✗' if s.results.get(w) in ('loss', 'missed') else '?')}"
                        for w in range(1, week + 1))
        print(f"#{s.entry_id:<3} {'alive' if s.alive else 'OUT  '} losses={s.losses} {hist}")


def cmd_record(args):
    state = PoolState.load(Path(args.pool))
    for eid in (args.entry.split(",") if args.entry != "all" else state.entry_ids()):
        state.picks.setdefault(str(eid), {})[int(args.week)] = normalize(args.team)
    state.save()
    print(f"recorded week {args.week}: {normalize(args.team)} for entries {args.entry}")


def cmd_qb(args):
    cfg = load_config()
    df = load_games(refresh=False)
    season = int(args.season or cfg.get("season") or latest_season(df))
    week = args.week or current_week(regular_season(df, season), _now(args))
    print(qb_watch(season, week, refresh=not args.offline).to_string(index=False))


def cmd_backtest(args):
    from .backtest import discount_sweep, run_backtest
    cfg = load_config()
    if args.discount is not None:
        cfg["model"]["future_discount"] = args.discount
    df = load_games(refresh=False)
    last_complete = max(s for s in df["season"].unique() if regular_season(df, s)["played"].all())
    seasons = _seasons(args.seasons, list(range(2015, last_complete + 1)))
    modes = ["single", "strikes", "multi"] if args.mode == "all" else [args.mode]
    for mode in modes:
        strikes = args.strikes if mode == "strikes" else 0
        n_entries = args.entries if mode == "multi" else 1
        if args.sweep:
            print(f"\n=== discount sweep: {mode} ===")
            grid = [float(x) for x in args.discounts.split(",")] if args.discounts else [0.5, 1.0, 2.0, 4.0, 8.0]
            out = discount_sweep(df, seasons, cfg, mode, strikes, grid)
            print(out.to_string(index=False))
            continue
        print(f"\n=== backtest: {mode} (strikes={strikes}, entries={n_entries}) ===")
        bt = run_backtest(df, seasons, cfg, mode=mode, strikes=strikes, n_entries=n_entries,
                          scenarios=args.scenarios, verbose=args.verbose)
        bt["expected"] = np.exp(bt["mean_log_expected"])
        if mode == "multi":
            bt["alive_w4/8/12/16/end"] = bt["alive_by_week"].map(lambda d: "/".join(str(v) for v in d.values()))
            print(bt[["season", "n_survived", "any_survived", "first_elim_week", "last_elim_week", "alive_w4/8/12/16/end"]].to_string(index=False))
            print(f"seasons with a survivor: {int(bt['any_survived'].sum())}/{len(bt)}; mean survivors {bt['n_survived'].mean():.2f}")
        else:
            print(bt[["season", "n_survived", "any_survived", "first_elim_week", "expected", "picks"]].to_string(index=False))
            print(f"seasons survived: {int(bt['any_survived'].sum())}/{len(bt)}; geometric-mean season survival at closing prices: {np.exp(bt['mean_log_expected'].mean()):.4f}")
        if args.out:
            bt.to_csv(args.out, index=False)


def main(argv=None):
    ap = argparse.ArgumentParser(prog="eliminator", description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p):
        p.add_argument("--season", type=int)
        p.add_argument("--offline", action="store_true", help="do not hit the network")
        p.add_argument("--now", help="pretend it is this time (ET), e.g. '2026-09-13 12:00'")

    p = sub.add_parser("update", help="refresh schedule/lines, inpredictable ratings and injury reports"); common(p)
    p.add_argument("--inpredictable-file", help="saved inpredictable HTML page or CSV (team,gpf)")
    p.add_argument("--no-inpredictable", action="store_true")
    p.set_defaults(fn=cmd_update)

    p = sub.add_parser("calibrate", help="fit sigma, HFA, rest, horizon variance from history"); common(p)
    p.add_argument("--seasons", help="e.g. 2011-2025")
    p.set_defaults(fn=cmd_calibrate)

    p = sub.add_parser("plan", help="this week's picks and season plan for a pool"); common(p)
    p.add_argument("--pool", default=str(STATE_DIR / "multi25.yaml"))
    p.add_argument("--week", type=int)
    p.add_argument("--scenarios", type=int)
    p.add_argument("--source", choices=["auto", "inpredictable", "market", "blend"])
    p.add_argument("--objective", choices=["any", "expected"], default="any")
    p.add_argument("--discount", type=float, help="override model.future_discount")
    p.add_argument("--inpredictable-file")
    p.add_argument("--no-inpredictable", action="store_true")
    p.add_argument("--no-paths", action="store_true")
    p.add_argument("--json", help="write picks/board/ratings to this file")
    p.add_argument("--commit", action="store_true", help="write this week's picks into the pool file")
    p.set_defaults(fn=cmd_plan)

    p = sub.add_parser("status", help="results and survival of each entry"); common(p)
    p.add_argument("--pool", default=str(STATE_DIR / "multi25.yaml")); p.add_argument("--week", type=int)
    p.set_defaults(fn=cmd_status)

    p = sub.add_parser("record", help="record a pick by hand"); common(p)
    p.add_argument("--pool", required=True); p.add_argument("--entry", required=True, help="entry id, comma list, or all")
    p.add_argument("--week", type=int, required=True); p.add_argument("--team", required=True)
    p.set_defaults(fn=cmd_record)

    p = sub.add_parser("qb", help="QB injury report for a week"); common(p)
    p.add_argument("--week", type=int)
    p.set_defaults(fn=cmd_qb)

    p = sub.add_parser("backtest", help="replay past seasons with as-of information"); common(p)
    p.add_argument("--seasons", help="e.g. 2015-2025")
    p.add_argument("--mode", choices=["single", "strikes", "multi", "all"], default="single")
    p.add_argument("--strikes", type=int, default=2); p.add_argument("--entries", type=int, default=25)
    p.add_argument("--scenarios", type=int, default=400)
    p.add_argument("--discount", type=float); p.add_argument("--sweep", action="store_true")
    p.add_argument("--discounts", help="comma list for --sweep, e.g. 8,16,32")
    p.add_argument("--verbose", action="store_true"); p.add_argument("--out")
    p.set_defaults(fn=cmd_backtest)

    args = ap.parse_args(argv)
    args.fn(args)


if __name__ == "__main__":
    main()

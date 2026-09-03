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


def _posted_line_fit(cfg, season: int, games) -> dict | None:
    """Replace the default line-movement allowance with one fitted from archived lines, if enough."""
    from .data.lines_archive import load_archive, posted_line_variance
    fit = posted_line_variance(load_archive(season), games)
    cfg["model"]["posted_line_fit"] = fit
    if fit:
        cfg["model"]["posted_line_var_a"] = fit["posted_line_var_a"]
        cfg["model"]["posted_line_var_b"] = fit["posted_line_var_b"]
        print(f"[lines] posted-line movement fitted from {fit['n_obs']} archived lines: var = {fit['posted_line_var_a']} + {fit['posted_line_var_b']} * weeks")
    return fit


def _plan_inputs(args, cfg, state: PoolState, df):
    """Season, time, QB ledger and line overrides (manual + live odds) shared by plan and snapshot."""
    season = int(args.season or state.season or cfg.get("season") or latest_season(df))
    now = _now(args)
    _posted_line_fit(cfg, season, regular_season(df, season))
    ledger = load_ledger(STATE_DIR / "qb_status.yaml", cfg["qb"])
    overrides = load_overrides(STATE_DIR / "overrides.yaml")
    games = regular_season(df, season)
    wk = getattr(args, "week", None) or current_week(games, now)
    if cfg["qb"].get("auto_from_injuries", True):
        ledger = ledger + _auto_qb(cfg, season, wk, ledger, refresh=not args.offline)
    if not args.offline:
        from .data.odds_api import live_overrides
        live = live_overrides(cfg, games, wk)
        if live:
            manual = overrides.get("lines") or []
            overrides["lines"] = manual + [r for r in live if not any(m.get("home") == r["home"] and int(m.get("week", 0)) == r["week"] for m in manual)]
    return season, now, ledger, overrides


def _auto_qb(cfg, season: int, week: int, manual, refresh: bool):
    """Roll state/qb_auto.yaml forward from the nflverse injury report and return its situations."""
    from .data.injuries import report_weeks
    from .model.qb import load_auto, save_auto, update_auto
    path = STATE_DIR / "qb_auto.yaml"
    previous = load_auto(path, cfg["qb"])
    try:
        watch = qb_watch(season, week, refresh=refresh)
    except Exception as exc:  # feed trouble: keep what we had
        print(f"[qb-auto] injury report unavailable ({exc}); keeping {len(previous)} automatic situation(s)")
        return previous
    auto = update_auto(previous, watch, manual, week, cfg["qb"], report_published=week in report_weeks(season))
    before = {(s.team, s.player, s.status) for s in previous}
    after = {(s.team, s.player, s.status) for s in auto}
    if before != after or not path.exists():
        save_auto(auto, path)
    if auto:
        print("[qb-auto] " + ", ".join(f"{s.team} {s.player} {s.status} ({s.injury}, since w{s.injured_week})" for s in auto))
    return auto


def cmd_plan(args):
    cfg = load_config()
    if args.scenarios:
        cfg["simulation"]["scenarios"] = args.scenarios
    if args.discount is not None:
        cfg["model"]["future_discount"] = args.discount
    df = load_games(refresh=not args.offline, max_age_hours=cfg["data"]["max_age_hours"])
    state = PoolState.load(Path(args.pool))
    season, now, ledger, overrides = _plan_inputs(args, cfg, state, df)
    state.season = season
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


def cmd_snapshot(args):
    from .site import DATA_DIR, backfill_state, build_snapshot, load_snapshot, pool_files, snapshot_path, write_snapshot
    cfg = load_config()
    if args.scenarios:
        cfg["simulation"]["scenarios"] = args.scenarios
    df = load_games(refresh=not args.offline, max_age_hours=cfg["data"]["max_age_hours"])
    data_dir = Path(args.data_dir) if args.data_dir else DATA_DIR
    pools = [Path(p) for p in args.pool] if args.pool else pool_files(STATE_DIR)
    ratings = None
    ratings_loaded = False
    archived = False
    for pool in pools:
        state = PoolState.load(pool)
        season, now, ledger, overrides = _plan_inputs(args, cfg, state, df)
        state.season = season
        games = regular_season(df, season)
        week = args.week or current_week(games, now)
        if not archived and not args.no_archive:
            from .data.lines_archive import archive_lines
            n = archive_lines(games, season, now)
            print(f"[lines] archived {n} posted line(s) for {now:%Y-%m-%d}")
            archived = True
        if not args.no_backfill:
            n = backfill_state(state, games, week, now, data_dir)
            if n:
                state.save()
                print(f"[{pool.stem}] filled {n} missing pick(s) for kicked-off games from earlier snapshots")
        if not ratings_loaded:
            ratings = _ratings(args, cfg, season)
            ratings_loaded = True
        res = make_plan(state, df, cfg, ledger, ratings, now=now, season=season, week=week,
                        source=args.source, overrides=overrides, keep_wins=True)
        previous = load_snapshot(snapshot_path(res.season, res.week, pool.stem, data_dir))
        snap = build_snapshot(res, pool.stem, generated_at=now, previous=previous, ledger=ledger, cfg=cfg)
        path = write_snapshot(snap, data_dir)
        print(render(res, show_paths=False))
        print(f"written {path}\n")


def cmd_site(args):
    from .site import BUILD_DIR, DATA_DIR, build_site
    try:
        df = load_games(refresh=not args.offline)
    except Exception as exc:  # no cached feed and no network: pages still render, ungraded
        print(f"[site] schedule unavailable ({exc}); results will not be graded")
        df = None
    out = build_site(df, data_dir=Path(args.data_dir) if args.data_dir else DATA_DIR,
                     out_dir=Path(args.out) if args.out else BUILD_DIR, built_at=_now(args))
    print(f"wrote {len(out)} files to {out[0].parent}")


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
    from .backtest import discount_sweep, horizon_sweep, run_backtest
    cfg = load_config()
    if args.discount is not None:
        cfg["model"]["future_discount"] = args.discount
    if args.allocation:
        cfg["portfolio"]["allocation_view"] = args.allocation
    cfg.setdefault("planning", {})
    if args.planning:
        cfg["planning"]["mode"] = args.planning
    if args.horizon is not None:
        cfg["planning"]["horizon"] = args.horizon
    if args.spread_weights:
        cfg["planning"]["spread_weights"] = [float(x) for x in args.spread_weights.split(",")]
    df = load_games(refresh=False)
    last_complete = max(s for s in df["season"].unique() if regular_season(df, s)["played"].all())
    seasons = _seasons(args.seasons, list(range(2015, last_complete + 1)))
    modes = ["single", "strikes", "multi"] if args.mode == "all" else [args.mode]
    for mode in modes:
        strikes = args.strikes if mode == "strikes" else 0
        n_entries = args.entries if mode == "multi" else 1
        if args.horizons:
            print(f"\n=== horizon sweep: {mode} (strikes={strikes}, entries={n_entries}) ===")
            out = horizon_sweep(df, seasons, cfg, mode, strikes, [int(x) for x in args.horizons.split(",")], n_entries=n_entries, scenarios=args.scenarios)
            print(out.to_string(index=False))
            continue
        if args.sweep:
            cfg["planning"]["mode"] = "discount"
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

    p = sub.add_parser("snapshot", help="plan every pool and record this week's recommendations for the site"); common(p)
    p.add_argument("--pool", action="append", help="pool file (repeatable); default: every pool in state/")
    p.add_argument("--week", type=int)
    p.add_argument("--scenarios", type=int)
    p.add_argument("--source", choices=["auto", "inpredictable", "market", "blend"])
    p.add_argument("--inpredictable-file")
    p.add_argument("--no-inpredictable", action="store_true")
    p.add_argument("--data-dir", help="where snapshots live (default site/data)")
    p.add_argument("--no-backfill", action="store_true", help="do not fill missing kicked-off picks from earlier snapshots")
    p.add_argument("--no-archive", action="store_true", help="do not record today's posted lines")
    p.set_defaults(fn=cmd_snapshot)

    p = sub.add_parser("site", help="render the snapshots into static HTML (GitHub Pages)"); common(p)
    p.add_argument("--data-dir", help="where snapshots live (default site/data)")
    p.add_argument("--out", help="output directory (default site/build)")
    p.set_defaults(fn=cmd_site)

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
    p.add_argument("--scenarios", type=int, help="default 2000 for policy planning, 400 for discount planning")
    p.add_argument("--planning", choices=["policy", "discount"], help="override planning.mode")
    p.add_argument("--horizon", type=int, help="policy planning: weeks that are a commitment (default 1)")
    p.add_argument("--spread-weights", help="policy planning: how often an entry takes the best, 2nd, 3rd available team later, e.g. 0.6,0.3,0.1")
    p.add_argument("--horizons", help="comma list of horizons to compare, e.g. 1,2,4; also runs the discount planner and greedy")
    p.add_argument("--discount", type=float); p.add_argument("--sweep", action="store_true")
    p.add_argument("--allocation", choices=["planning", "calibrated"], help="which simulation scores the multi-entry split")
    p.add_argument("--discounts", help="comma list for --sweep, e.g. 8,16,32")
    p.add_argument("--verbose", action="store_true"); p.add_argument("--out")
    p.set_defaults(fn=cmd_backtest)

    args = ap.parse_args(argv)
    args.fn(args)


if __name__ == "__main__":
    main()

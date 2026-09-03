import datetime as dt
import json

from eliminator.data.schedule import ET, regular_season
from eliminator.plan import make_plan
from eliminator.site import (backfill_state, build_site, build_snapshot, grade, load_snapshots, pool_files,
                             snapshot_path, week_record, write_snapshot)
from eliminator.state import PoolState


def _pool(tmp_path, name, mode, entries, strikes):
    p = tmp_path / f"{name}.yaml"
    p.write_text(f"name: {name}\nmode: {mode}\nentries: {entries}\nstrikes: {strikes}\nseason: 2026\npicks: {{}}\n")
    return p


def test_snapshot_roundtrip_and_site(tmp_path, games_all, cfg, before_week1):
    data = tmp_path / "data"
    strikes = PoolState.load(_pool(tmp_path, "strikes2", "strikes", 1, 2))
    multi = PoolState.load(_pool(tmp_path, "multi25", "multi", 4, 0))
    snaps = []
    for st in (multi, strikes):
        res = make_plan(st, games_all, cfg, [], None, now=before_week1, season=2026, source="market")
        snap = build_snapshot(res, st.path.stem, generated_at=before_week1)
        path = write_snapshot(snap, data)
        assert path == snapshot_path(2026, 1, st.path.stem, data)
        back = json.loads(path.read_text())
        assert back["week"] == 1 and back["mode"] == st.mode and back["picks"]
        assert len(back["board"]) == 32 and back["options"] and back["planning_weeks"][0] == 1
        assert all(len(pl["path"]) == 18 for pl in back["plans"] if pl["alive"])
        snaps.append(back)
    assert len(snaps[0]["picks"]) == 4 and len(snaps[1]["picks"]) == 1

    # a second run in the same week keeps the earlier recommendation as a revision
    res = make_plan(strikes, games_all, cfg, [], None, now=before_week1, season=2026, source="market")
    snap2 = build_snapshot(res, "strikes2", generated_at=before_week1 + dt.timedelta(days=1), previous=snaps[1])
    write_snapshot(snap2, data)
    assert snap2["revisions"][-1]["generated_at"] == before_week1.isoformat()

    out = tmp_path / "build"
    files = build_site(games_all, data_dir=data, out_dir=out, built_at=before_week1)
    names = {f.name for f in files}
    assert {"index.html", "s2026.html", "s2026-w01.html"} <= names
    idx = (out / "index.html").read_text()
    assert "multi25" in idx and "strikes2" in idx and "week 1" in idx
    wk = (out / "s2026-w01.html").read_text()
    assert "This week's options" in wk and "Per-entry season plans" in wk
    assert (out / "data" / "2026-w01-strikes2.json").exists()
    assert len(load_snapshots(data)) == 2
    # team logos ship with the site and every pick shows one
    assert len(list((out / "logos").glob("*.png"))) == 32
    assert 'src="logos/' in idx and 'src="logos/' in wk


def test_backfill_fills_only_kicked_off_missing_picks(tmp_path, games_all, cfg, before_week1):
    data = tmp_path / "data"
    st = PoolState.load(_pool(tmp_path, "multi25", "multi", 3, 0))
    res = make_plan(st, games_all, cfg, [], None, now=before_week1, season=2026, source="market")
    write_snapshot(build_snapshot(res, "multi25", generated_at=before_week1), data)
    picks = {r["entry"]: r["team"] for r in res.this_week().to_dict(orient="records")}
    games = regular_season(games_all, 2026)
    # entry 2 was recorded by hand; nothing has kicked off yet -> nothing changes
    st.picks = {"2": {1: "KC"}}
    assert backfill_state(st, games, 1, before_week1, data) == 0
    assert st.picks == {"2": {1: "KC"}}
    # after the week's games kick off the missing picks come from the snapshot, the hand pick stays
    later = dt.datetime(2026, 9, 14, 12, 0, tzinfo=ET)
    n = backfill_state(st, games, 2, later, data)
    assert n == 2
    assert st.picks["2"][1] == "KC" and st.picks["1"][1] == picks["1"] and st.picks["3"][1] == picks["3"]


def test_grading_and_week_record(tmp_path, games_all):
    played = games_all[(games_all.season == 2025) & (games_all.week == 1)].iloc[0]
    home, away = played["home"], played["away"]
    winner, loser = (home, away) if played["result"] > 0 else (away, home)
    assert grade(games_all, 2025, 1, winner) == "win"
    assert grade(games_all, 2025, 1, loser) == "loss"
    assert grade(games_all, 2026, 1, home) == "pending"
    assert grade(None, 2026, 1, home) == "unknown"
    snaps = [{"pool": "p", "season": 2025, "week": 1, "strikes": 0, "entries": 2, "mode": "multi",
              "picks": [{"entry": "1", "team": winner}, {"entry": "2", "team": loser}]}]
    rec = week_record(snaps, games_all)
    assert rec["p"]["by_week"][1]["alive_after"] == 1
    assert rec["p"]["by_week"][1]["graded"] == {"1": "win", "2": "loss"}


def test_pool_files_skip_non_pool_yaml(tmp_path):
    _pool(tmp_path, "a", "multi", 2, 0)
    (tmp_path / "qb_status.yaml").write_text("[]\n")
    (tmp_path / "overrides.yaml").write_text("lines: []\n")
    assert [p.name for p in pool_files(tmp_path)] == ["a.yaml"]


def test_snapshot_carries_qb_situations_and_week_page_renders_them(tmp_path, games_all, cfg, before_week1):
    from eliminator.model.qb import QBSituation
    data = tmp_path / "data"
    st = PoolState.load(_pool(tmp_path, "strikes2", "strikes", 1, 2))
    ledger = [QBSituation(team="KC", player="Patrick Mahomes", penalty=7, status="out", injury="ankle", injured_week=1, note="auto: nflverse injury report")]
    res = make_plan(st, games_all, cfg, ledger, None, now=before_week1, season=2026, source="market")
    snap = build_snapshot(res, "strikes2", generated_at=before_week1, ledger=ledger)
    assert snap["qb_situations"][0]["team"] == "KC" and snap["qb_situations"][0]["source"] == "auto"
    assert snap["qb_situations"][0]["p_out"][1] == 1.0
    write_snapshot(snap, data)
    build_site(games_all, data_dir=data, out_dir=tmp_path / "build", built_at=before_week1)
    page = (tmp_path / "build" / "s2026-w01.html").read_text()
    assert "Quarterback situations (1)" in page and "Patrick Mahomes" in page

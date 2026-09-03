import datetime as dt

import numpy as np

from eliminator.backtest import as_of_view
from eliminator.data.schedule import ET, regular_season
from eliminator.model.projection import build_projection
from eliminator.model.qb import QBSituation
from eliminator.model.strength import assemble
from eliminator.optimize.simulate import simulate_wins
from eliminator.plan import commit_picks, make_plan, render
from eliminator.state import PoolState, evaluate_entries
from eliminator.teams import TEAMS


def test_projection_uses_lines_now_and_model_later(games_all, cfg, before_week1):
    st = assemble(games_all, 2026, 1, cfg, [], None, "market")
    proj = build_projection(regular_season(games_all, 2026), 2026, 1, st, cfg, now=before_week1)
    t = proj.table
    assert (t[t.week == 1].source == "moneyline").all()
    assert t[t.week == 2].source.str.startswith("blend").all()
    assert (t[t.week == 10].source == "model").all()
    assert (t[t.week == 18].source == "model+wk18").all()
    # probabilities of the two sides of a game sum to one, and every team has a game each week
    assert np.allclose(proj.prob.sum(axis=1), 16.0)
    assert proj.has_game.all()
    # horizon inflates uncertainty: same spread gives a probability closer to 0.5 further out
    assert t[t.week == 18].line_var.min() > t[t.week == 10].line_var.max()


def test_qb_ledger_moves_projection(games_all, cfg, before_week1):
    base = assemble(games_all, 2026, 1, cfg, [], None, "market")
    hurt = assemble(games_all, 2026, 1, cfg, [QBSituation(team="KC", penalty=7, status="out", injury="acl", injured_week=1)], None, "market")
    g = regular_season(games_all, 2026)
    p0 = build_projection(g, 2026, 1, base, cfg, now=before_week1)
    p1 = build_projection(g, 2026, 1, hurt, cfg, now=before_week1)
    kc = TEAMS.index("KC")
    assert p1.prob[5, kc] < p0.prob[5, kc]          # week 6: model only, KC weaker
    assert p1.prob[0, kc] == p0.prob[0, kc]         # week 1: the market line is the truth either way


def test_simulation_marginals_match_projection(games_all, cfg, before_week1):
    st = assemble(games_all, 2026, 1, cfg, [], None, "market")
    proj = build_projection(regular_season(games_all, 2026), 2026, 1, st, cfg, now=before_week1)
    wins = simulate_wins(proj, cfg, n=20000, seed=3)
    est = wins.mean(axis=0)
    assert np.abs(est - proj.prob).max() < 0.02


def test_make_plan_multi_and_commit(games_all, cfg, before_week1, tmp_path):
    state = PoolState(name="t", mode="multi", n_entries=6, strikes=0, season=2026, picks={}, path=tmp_path / "p.yaml")
    res = make_plan(state, games_all, cfg, [], None, now=before_week1, season=2026, source="market", scenarios=2000)
    assert len([e for e in res.entries if e.path]) == 6
    for e in res.entries:
        assert len(set(e.path.teams)) == 18
    assert 0 < res.summary["p_any"] <= 1
    n = commit_picks(res)
    assert n == 6
    state.save()
    reloaded = PoolState.load(tmp_path / "p.yaml")
    assert all(1 in reloaded.picks[str(i)] for i in range(1, 7))
    assert "week 1 picks by team" in render(res)


def test_make_plan_strikes_and_locked_pick(games_all, cfg, before_week1):
    g = regular_season(games_all, 2026)
    first = g[g.week == 1].iloc[0]
    locked_team = first["home"]
    state = PoolState(name="s", mode="strikes", n_entries=1, strikes=2, season=2026, picks={"1": {1: locked_team}})
    after_kick = first["kickoff"] + dt.timedelta(minutes=5)
    res = make_plan(state, games_all, cfg, [], None, now=after_kick, season=2026, source="market", scenarios=1000)
    e = res.entries[0]
    assert TEAMS[e.path.teams[0]] == locked_team
    assert res.this_week().iloc[0]["status"] == "locked"
    assert e.strikes_left == 2


def test_evaluate_entries_results(games_all, before_week1):
    g = regular_season(games_all, 2026).copy()
    # fake a played week 1: home teams all win by 3
    wk1 = g.week == 1
    g.loc[wk1, "result"] = 3.0; g.loc[wk1, "played"] = True; g.loc[wk1, "home_win"] = 1.0
    home, away = g[wk1].iloc[0]["home"], g[wk1].iloc[0]["away"]
    state = PoolState(name="e", mode="multi", n_entries=3, strikes=0, season=2026,
                      picks={"1": {1: home}, "2": {1: away}, "3": {}})
    st = evaluate_entries(state, g, 2, before_week1 + dt.timedelta(days=7))
    assert st[0].alive and st[0].results[1] == "win" and home in st[0].used
    assert not st[1].alive and st[1].results[1] == "loss"
    assert not st[2].alive and st[2].results[1] == "missed"


def test_as_of_view_hides_future(games_all):
    v = as_of_view(games_all, 2026, 2)
    cur = v[(v.season == 2026)]
    assert cur[cur.week > 2]["spread_line"].isna().all()
    assert cur[cur.week <= 2]["spread_line"].notna().all()
    assert not cur[cur.week >= 2]["played"].any()
    assert (v.season <= 2026).all()


def test_tie_is_a_loss_for_both_sides(games_all, cfg, before_week1):
    g = regular_season(games_all, 2026).copy()
    first = g.index[g.week == 1][0]
    g.loc[first, ["result", "played", "home_win"]] = [0.0, True, 0.5]
    st = assemble(games_all, 2026, 2, cfg, [], None, "market")
    proj = build_projection(g, 2026, 1, st, cfg, now=before_week1 + dt.timedelta(days=1))
    home, away = g.loc[first, "home"], g.loc[first, "away"]
    assert proj.prob[0, TEAMS.index(home)] == 0.0 and proj.prob[0, TEAMS.index(away)] == 0.0
    wins = simulate_wins(proj, cfg, n=50, seed=1)
    assert not wins[:, 0, TEAMS.index(home)].any() and not wins[:, 0, TEAMS.index(away)].any()

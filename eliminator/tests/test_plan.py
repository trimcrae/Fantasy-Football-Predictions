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
    assert t[t.week == 2].source.str.startswith("posted-").all()
    # a posted line is used as the spread, not blended with the ratings
    w2 = t[(t.week == 2) & t.home]
    assert (w2.spread - w2.line_spread).abs().max() < 1e-9
    assert (t[t.week == 10].source == "model").all()
    assert (t[t.week == 18].source == "model+wk18").all()
    # probabilities of the two sides of a game sum to one, and every team has a game each week
    assert np.allclose(proj.prob.sum(axis=1), 16.0)
    assert proj.has_game.all()
    # horizon inflates uncertainty: same spread gives a probability closer to 0.5 further out
    assert t[t.week == 18].line_var.min() > t[t.week == 10].line_var.max()


def test_week18_rest_is_settled_in_each_simulated_season(games_all, cfg, before_week1):
    from eliminator.optimize.simulate import simulate_season
    st = assemble(games_all, 2026, 1, cfg, [], None, "market")
    g = regular_season(games_all, 2026)
    proj = build_projection(g, 2026, 1, st, cfg, now=before_week1)
    before = proj.prob[-1].copy()
    sim = simulate_season(proj, cfg, n=3000, seed=5)
    assert proj.rest is not None and set(proj.rest) >= {"bye", "locked", "out", "expected_points"}
    # the strongest teams are the ones most often resting; nobody is flagged as settled 100% in week 1
    strong = np.argsort(-st.healthy.reindex(TEAMS).to_numpy())[:4]
    settled = proj.rest["bye"] + proj.rest["locked"]
    assert settled[strong].mean() > settled.mean() and settled.max() < 0.9
    assert proj.rest["out"].max() > 0.5
    # the strongest teams are docked the most, week-18 probabilities moved, and the scenario average
    # is what the projection now reports
    pts = proj.rest["expected_points"]
    assert pts[strong].mean() > pts.mean() and pts.max() > 0.5
    assert np.abs(proj.prob[-1] - before).max() > 0.005
    assert np.abs(sim.probs[:, -1, :].mean(axis=0) - proj.prob[-1]).max() < 1e-4      # float32 probabilities
    # a fixed-path simulation does not rewrite the projection
    snap = proj.prob[-1].copy()
    simulate_season(proj, cfg, n=500, seed=6, discount=16.0, update_projection=False)
    assert np.array_equal(snap, proj.prob[-1])


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
    # simulate with the same discount the projection used: marginals must agree
    wins = simulate_wins(proj, cfg, n=20000, seed=3, discount=cfg["model"]["future_discount"])
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
    assert e.strikes_left == 1                  # two strikes: one loss can be taken, the second is out
    assert res.strikes_left_of("1") == 2


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


def test_policy_mode_values_entries_by_best_available_later(games_all, cfg, before_week1):
    from eliminator.optimize.simulate import policy_alive
    state = PoolState(name="p", mode="multi", n_entries=5, strikes=0, season=2026, picks={})
    res = make_plan(state, games_all, cfg, [], None, now=before_week1, season=2026, source="market", scenarios=2000, keep_wins=True)
    assert res.horizon == 1 and res.allocation_view == "policy" and res.sim is not None
    # the projection is the calibrated one: no 16x variance on later weeks
    proj_disc = res.projection.line_var[5].max()
    assert proj_disc < 200
    # every entry's reported survival is its own policy mask, following its own later-week rank pattern
    for e in res.entries:
        assert abs(e.path.value - e.surv.mean()) < 1e-12 and e.surv.shape == (2000,)
        assert 0 < e.path.value < 1 and abs(e.surv.mean() - e.alive_mask.mean()) < 0.02
    from eliminator.optimize.simulate import rank_table
    R = rank_table(5, len(res.projection.weeks), tuple(res.planning["spread_weights"]), res.planning["seed"])
    for e in res.entries:
        m = policy_alive(res.sim, e.path.teams[:1], e.available, 0, ranks=R[e.slot].tolist())
        assert np.array_equal(m, e.alive_mask)
    assert sorted(e.slot for e in res.entries) == [0, 1, 2, 3, 4]
    # options are ranked by simulated season survival and carry a mask
    assert res.options and all(res.options[i].value >= res.options[i + 1].value for i in range(len(res.options) - 1))
    assert "mask" in res.options[0].detail


def test_discount_mode_still_available(games_all, cfg, before_week1):
    c = dict(cfg); c["planning"] = {"mode": "discount"}
    state = PoolState(name="p", mode="strikes", n_entries=1, strikes=2, season=2026, picks={})
    res = make_plan(state, games_all, c, [], None, now=before_week1, season=2026, source="market", scenarios=1000, keep_wins=True)
    assert res.horizon is None and res.allocation_view == "planning" and res.wins_policy is not None
    assert res.projection.line_var[5].max() > 100


def test_locked_loss_is_charged_once(games_all, cfg):
    """After a two-strike entry's pick has lost, its strike shows as used, and the season odds
    are P(at most one more loss over the remaining weeks), not the loss counted twice."""
    from eliminator.optimize.simulate import policy_run, survival_given_lines
    g = games_all.copy()
    first_idx = g.index[(g.season == 2026) & (g.week == 1)][0]
    g.loc[first_idx, ["result", "played", "home_win"]] = [-3.0, True, 0.0]     # the away team won
    home = g.loc[first_idx, "home"]
    state = PoolState(name="s", mode="strikes", n_entries=1, strikes=2, season=2026, picks={"1": {1: home}})
    after = g.loc[first_idx, "kickoff"] + dt.timedelta(hours=5)
    res = make_plan(state, g, cfg, [], None, now=after, season=2026, source="market", scenarios=1000, keep_wins=True)
    e = res.entries[0]
    assert e.alive and res.strikes_left_of("1") == 1 and res.decided_now("1")
    assert TEAMS[e.path.teams[0]] == home and e.path.probs[0] == 0.0
    _, pwin, _ = policy_run(res.sim, [TEAMS.index(home)], e.available)
    expected = survival_given_lines(pwin[:, 1:], strikes=0).mean()        # one strike left: no more losses allowed
    assert abs(e.path.value - expected) < 1e-9 and e.path.value > 0
    assert "strikes left: 1 of 2" in render(res)
    from eliminator.explain import explain_summary
    assert explain_summary(res).startswith("Chance of no losses in 17 picks")
    # a locked pick still pending is charged nothing yet
    pending = make_plan(state, games_all, cfg, [], None, now=after, season=2026, source="market", scenarios=1000, keep_wins=True)
    assert pending.entries[0].strikes_left == 1 and pending.strikes_left_of("1") == 2 and not pending.decided_now("1")
    assert explain_summary(pending).startswith("Chance of at most 1 loss in 18 picks")
    # a second loss puts the entry out
    state.picks["1"][2] = "KC"
    g2 = g.copy(); wk2 = (g2.season == 2026) & (g2.week == 2)
    g2.loc[wk2, "result"] = 3.0; g2.loc[wk2, "played"] = True; g2.loc[wk2, "home_win"] = 1.0
    kc_home = bool(g2[wk2 & (g2.home == "KC")].shape[0])
    if kc_home:                                          # make KC lose whichever side it is on
        g2.loc[wk2 & (g2.home == "KC"), ["result", "home_win"]] = [-3.0, 0.0]
    st = evaluate_entries(state, regular_season(g2, 2026), 3, dt.datetime(2026, 9, 30, 12, 0, tzinfo=ET))[0]
    assert st.losses == 2 and not st.alive

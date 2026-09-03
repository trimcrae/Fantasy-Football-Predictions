import itertools

import numpy as np

from eliminator.optimize.single import best_path, best_path_strikes, current_week_options, survival_prob


def brute_survival(ps, k):
    tot = 0.0
    for outs in itertools.product([0, 1], repeat=len(ps)):
        pr = np.prod([p if o else 1 - p for p, o in zip(ps, outs)])
        if sum(1 - o for o in outs) <= k:
            tot += pr
    return tot


def test_survival_prob_matches_brute_force():
    ps = [0.8, 0.7, 0.9, 0.55]
    for k in range(4):
        assert abs(survival_prob(ps, k) - brute_survival(ps, k)) < 1e-12


def test_best_path_uses_distinct_teams_and_respects_constraints():
    rng = np.random.default_rng(3)
    P = rng.uniform(0.4, 0.9, size=(8, 32))
    P[:, 0] = 0.0                       # team 0 never plays
    avail = np.ones(32, bool); avail[1] = False
    path = best_path(P, avail, fixed={2: 5})
    assert len(set(path.teams)) == 8
    assert 0 not in path.teams and 1 not in path.teams
    assert path.teams[2] == 5
    assert abs(path.value - np.prod(path.probs)) < 1e-12


def test_strike_path_at_least_as_good_as_product_path():
    rng = np.random.default_rng(4)
    P = rng.uniform(0.4, 0.9, size=(10, 32))
    avail = np.ones(32, bool)
    base = best_path(P, avail)
    two = best_path_strikes(P, avail, 2, rng=rng)
    assert two.value >= survival_prob(base.probs, 2) - 1e-12
    assert len(set(two.teams)) == 10


def test_current_week_options_sorted_and_forced():
    rng = np.random.default_rng(5)
    P = rng.uniform(0.4, 0.9, size=(5, 32))
    avail = np.ones(32, bool)
    pick_now = np.ones(32, bool); pick_now[7] = False
    opts = current_week_options(P, avail, pick_now, 0)
    assert all(opts[i].value >= opts[i + 1].value for i in range(len(opts) - 1))
    assert 7 not in [o.teams[0] for o in opts]
    assert opts[0].value >= best_path(P, avail).value - 1e-12


def _fake_sim(n=500, nW=6, seed=0):
    from eliminator.optimize.simulate import Sim
    rng = np.random.default_rng(seed)
    probs = rng.uniform(0.3, 0.9, size=(n, nW, 32)).astype(np.float32)
    probs[:, :, 0] = 0.0                                    # team 0 never has a game
    wins = rng.random((n, nW, 32)) < probs
    has_game = np.ones((nW, 32), bool); has_game[:, 0] = False
    return Sim(wins=wins, probs=probs, has_game=has_game, weeks=list(range(1, nW + 1)))


def test_policy_tail_never_reuses_a_team_and_takes_the_best_available():
    from eliminator.optimize.simulate import policy_alive
    sim = _fake_sim()
    avail = np.ones(32, bool); avail[5] = False
    alive, picks = policy_alive(sim, [7], avail, strikes=0, return_picks=True)
    assert picks.shape == (sim.n, 6) and (picks[:, 0] == 7).all()
    for s in range(sim.n):
        row = picks[s]
        assert len(set(row.tolist())) == 6 and 5 not in row and 0 not in row
        used = {7, 5}
        for wi in range(1, 6):
            pw = sim.probs[s, wi].copy(); pw[list(used)] = 0
            assert row[wi] == pw.argmax()
            used.add(int(row[wi]))
    expect = (sim.wins[np.arange(sim.n)[:, None], np.arange(6)[None, :], picks]).all(axis=1)
    assert np.array_equal(alive, expect)


def test_policy_ranks_and_rank_table():
    from eliminator.optimize.simulate import policy_alive, rank_table
    sim = _fake_sim()
    avail = np.ones(32, bool)
    _, p0 = policy_alive(sim, [7], avail, ranks=[0], return_picks=True)
    _, p1 = policy_alive(sim, [7], avail, ranks=[1], return_picks=True)
    assert (p0[:, 1] != p1[:, 1]).all()                     # second-best in the first later week
    for s in range(sim.n):
        pw = sim.probs[s, 1].copy(); pw[7] = 0
        assert p1[s, 1] == np.argsort(-pw)[1]
    R = rank_table(25, 17, (0.6, 0.3, 0.1), seed=5)
    assert R.shape == (25, 17) and not R[0].any()             # entry 0 is pure best-available
    assert len({r.tobytes() for r in R}) == 25                # no two entries play alike
    freq = np.bincount(R[1:].ravel(), minlength=3) / R[1:].size
    assert abs(freq[0] - 0.6) < 0.08 and abs(freq[2] - 0.1) < 0.06
    assert not rank_table(3, 5, (1.0,), seed=1).any()         # a single weight is pure greedy for everyone


def test_policy_options_value_is_simulated_survival_and_two_strikes_count():
    from eliminator.optimize.simulate import policy_alive
    from eliminator.optimize.single import policy_options
    sim = _fake_sim(n=300)
    P = sim.probs.mean(axis=0).astype(float)
    avail = np.ones(32, bool); pick_now = np.ones(32, bool); pick_now[3] = False
    opts = policy_options(P, sim, avail, pick_now, strikes=0, horizon=1)
    assert all(opts[i].value >= opts[i + 1].value for i in range(len(opts) - 1))
    assert 3 not in [o.teams[0] for o in opts] and 0 not in [o.teams[0] for o in opts]
    o = opts[0]
    from eliminator.optimize.simulate import policy_run, survival_given_lines
    _, pwin, _ = policy_run(sim, [o.teams[0]], avail)
    assert abs(o.value - survival_given_lines(pwin, 0).mean()) < 1e-12
    assert np.array_equal(o.detail["mask"], policy_alive(sim, [o.teams[0]], avail))
    assert len(set(o.teams)) == 6 and o.detail["plugin"] > 0
    two = policy_options(P, sim, avail, pick_now, strikes=2, horizon=2)
    assert two[0].value > opts[0].value                      # two lives beat none

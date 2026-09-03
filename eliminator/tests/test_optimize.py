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

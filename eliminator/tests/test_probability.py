import numpy as np

from eliminator.probability import american_to_prob, devig, moneyline_home_prob, prob_to_spread, spread_to_prob, game_home_prob


def test_american_to_prob():
    assert abs(american_to_prob(-110) - 110 / 210) < 1e-9
    assert abs(american_to_prob(150) - 100 / 250) < 1e-9


def test_devig_sums_to_one():
    a, b = devig(american_to_prob(-175), american_to_prob(145))
    assert abs(a + b - 1) < 1e-12 and a > b


def test_spread_prob_roundtrip():
    for s in (-7.5, -3, 0, 2.5, 10):
        p = spread_to_prob(s, 11.5)
        assert abs(prob_to_spread(p, 11.5) - s) < 1e-6
    assert spread_to_prob(0) == 0.5
    assert spread_to_prob(7) > spread_to_prob(3) > 0.5


def test_game_home_prob_prefers_moneyline():
    p_ml = moneyline_home_prob(-200, 170)
    assert game_home_prob(-200, 170, 3.0) == p_ml
    assert abs(game_home_prob(np.nan, np.nan, 0.0) - 0.5) < 1e-12
    assert game_home_prob(np.nan, np.nan, np.nan) is None

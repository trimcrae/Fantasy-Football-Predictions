import numpy as np

from eliminator.model.standings import Record, TEAM_INDEX, rest_penalty, seeds, week18_flags


def _rec(**wins):
    w = np.zeros((1, 32)); c = np.zeros((1, 32)); d = np.zeros((1, 32))
    for t, n in wins.items():
        w[0, TEAM_INDEX[t]] = n; c[0, TEAM_INDEX[t]] = min(n, 12); d[0, TEAM_INDEX[t]] = min(n, 6)
    return Record(w, c, d)


def test_division_winners_take_the_top_seeds():
    r = _rec(KC=13, LAC=12, BUF=11, MIA=10, BAL=9, HOU=8)     # LAC 12-5 and MIA 10-7 are wild cards behind four division winners
    s = seeds(r, np.random.default_rng(0))
    i = TEAM_INDEX
    assert s[0, i["KC"]] == 1 and s[0, i["BUF"]] == 2 and s[0, i["BAL"]] == 3 and s[0, i["HOU"]] == 4
    assert s[0, i["LAC"]] == 5 and s[0, i["MIA"]] == 6


def test_week18_flags_bye_locked_out():
    # KC 15-1: bye settled whatever happens. BUF 11-5 leads its division by two and cannot catch BAL or
    # HOU (13-3): seed 4 settled, no bye. NYJ 2-14 is out. LAC 12-4 and DEN 11-5 fight for a wild card.
    r = _rec(KC=15, LAC=12, DEN=11, BUF=11, MIA=9, NE=6, NYJ=2, BAL=13, PIT=10, HOU=13, IND=8, CIN=6, CLE=5, JAX=5, TEN=4, LV=3)
    pairs = [(TEAM_INDEX[h], TEAM_INDEX[a]) for h, a in (("KC", "LAC"), ("DEN", "LV"), ("BUF", "NYJ"), ("MIA", "NE"),
                                                          ("BAL", "PIT"), ("CIN", "CLE"), ("HOU", "IND"), ("JAX", "TEN"))]
    fl = week18_flags(r, pairs, 2026, draws=64, rng=np.random.default_rng(1))
    i = TEAM_INDEX
    assert fl["bye"][0, i["KC"]] and not fl["locked"][0, i["KC"]]
    assert fl["locked"][0, i["BUF"]] and not fl["bye"][0, i["BUF"]]
    assert fl["out"][0, i["NYJ"]] and fl["out"][0, i["LV"]]
    assert not (fl["bye"][0, i["LAC"]] or fl["locked"][0, i["LAC"]] or fl["out"][0, i["LAC"]])
    assert not (fl["bye"][0, i["DEN"]] or fl["locked"][0, i["DEN"]] or fl["out"][0, i["DEN"]])
    pen = rest_penalty(fl, {"bye": 9.6, "locked": 6.9, "out": 1.1})
    assert pen[0, i["KC"]] == 9.6 and pen[0, i["BUF"]] == 6.9 and pen[0, i["NYJ"]] == 1.1 and pen[0, i["LAC"]] == 0


def test_no_week18_games_means_no_flags():
    fl = week18_flags(_rec(KC=10), [], 2026)
    assert not fl["bye"].any() and not fl["locked"].any() and not fl["out"].any()

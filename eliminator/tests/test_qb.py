import numpy as np

from eliminator.config import DEFAULTS
from eliminator.model.qb import QBSituation, p_out_by_week, reflected_fraction

CFG = DEFAULTS["qb"]


def test_return_week_profile():
    s = QBSituation(team="BAL", penalty=7, status="out", injury="hamstring", injured_week=5, return_week=8)
    p = p_out_by_week(s, current_week=6, season_weeks=18, cfg_qb=CFG)
    assert p[5] == 1.0 and p[6] == 1.0 and p[7] == 1.0
    assert p[8] == CFG["return_week_setback"][1]
    assert p[9] == CFG["return_week_setback"][2]
    assert p[12] == 0.0


def test_duration_prior_decays_and_questionable_is_soft():
    s = QBSituation(team="SF", penalty=5, status="out", injury="concussion", injured_week=10)
    p = p_out_by_week(s, 10, 18, CFG)
    assert p[10] == 1.0 and p[11] < p[10] and p[13] < p[11]
    q = QBSituation(team="SF", penalty=5, status="questionable", injury="ankle", injured_week=10)
    pq = p_out_by_week(q, 10, 18, CFG)
    assert abs(pq[10] - 0.4) < 1e-9 and pq[11] < 0.4


def test_season_ending():
    s = QBSituation(team="NYJ", penalty=7, status="out", injury="achilles", injured_week=1)
    p = p_out_by_week(s, 1, 18, CFG)
    assert np.all(p[1:] == 1.0)


def test_reflected_fraction_grows_with_games_missed():
    s = QBSituation(team="BAL", penalty=7, injured_week=5)
    weeks = list(range(1, 19))
    assert reflected_fraction(s, 5, weeks, CFG) == 0.0
    assert 0 < reflected_fraction(s, 6, weeks, CFG) < reflected_fraction(s, 9, weeks, CFG) < 1
    s.reflected = 0.3
    assert reflected_fraction(s, 9, weeks, CFG) == 0.3

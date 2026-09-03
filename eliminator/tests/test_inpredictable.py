from pathlib import Path

import pytest

from eliminator.data.inpredictable import parse_ratings_html

FIXTURE = Path(__file__).parent / "fixtures" / "inpredictable_sample.html"


def test_parse_fixture():
    df = parse_ratings_html(FIXTURE.read_text())
    assert len(df) == 32
    assert abs(df["gpf"].mean()) < 1e-9
    assert set(df.columns) >= {"team", "gpf", "ogpf", "dgpf"}


def test_parse_rejects_pages_without_table():
    with pytest.raises(ValueError):
        parse_ratings_html("<html><body><table><tr><th>Team</th><th>Wins</th></tr><tr><td>KC</td><td>12</td></tr></table></body></html>")


LIVE = Path(__file__).parent / "fixtures" / "inpredictable_live_sample.html"


def test_parse_live_layout_with_hidden_cells():
    """The live rankings page has more data cells than header cells (sparklines, ranks)."""
    import re
    html = LIVE.read_text()
    df = parse_ratings_html(html)
    assert len(df) == 32 and abs(df["gpf"].mean()) < 1e-9
    # every parsed GPF is the visible one-decimal value, and o + d = GPF
    assert ((df["ogpf"] + df["dgpf"] - (df["gpf"] + df["gpf"].mean())).abs() < 0.2).all()
    raw = {}
    for m in re.finditer(r">([A-Z]{2,3})</a></td><td>\d+</td><td>[\d,]+</td><td>(-?\d+\.\d)</td>", html):
        raw[m.group(1)] = float(m.group(2))
    centred = {t: v - sum(raw.values()) / len(raw) for t, v in raw.items()}
    for t, g in zip(df["team"], df["gpf"]):
        assert abs(centred[t] - g) < 1e-6, t


def test_live_layout_records_give_games_played():
    df = parse_ratings_html(LIVE.read_text())
    assert df.attrs["games_played"] and df.attrs["games_played"] > 32 * 5


def test_auto_rejects_last_seasons_ratings_before_week_one(games_all, cfg, before_week1):
    import pandas as pd
    from eliminator.model.strength import assemble
    from eliminator.teams import TEAMS
    mk = assemble(games_all, 2026, 1, cfg, [], None, "market")
    # a page that agrees with the market but still carries last season's records -> rejected in preseason
    agree = pd.DataFrame({"team": TEAMS, "gpf": [mk.healthy[t] for t in TEAMS]}); agree.attrs["games_played"] = 32 * 17
    st = assemble(games_all, 2026, 1, cfg, [], agree, "auto")
    assert st.source == "market" and "last season" in st.detail["inpredictable_check"]["reason"]
    # same page with fresh records is accepted
    agree.attrs["games_played"] = 0
    assert assemble(games_all, 2026, 1, cfg, [], agree, "auto").source == "inpredictable"
    # a page that contradicts this season's lines is rejected whatever its records say
    wrong = agree.copy(); wrong["gpf"] = -wrong["gpf"]; wrong.attrs["games_played"] = 0
    st = assemble(games_all, 2026, 1, cfg, [], wrong, "auto")
    assert st.source == "market" and "disagree" in st.detail["inpredictable_check"]["reason"]
    # explicit choice is honoured
    assert assemble(games_all, 2026, 1, cfg, [], wrong, "inpredictable").source == "inpredictable"

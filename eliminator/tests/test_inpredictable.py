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

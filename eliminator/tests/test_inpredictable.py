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

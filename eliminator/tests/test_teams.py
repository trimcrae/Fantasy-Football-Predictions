import pytest

from eliminator.teams import normalize, resolve, TEAMS


def test_normalize_aliases():
    assert normalize("STL") == "LA" and normalize("SD") == "LAC" and normalize("OAK") == "LV"
    assert normalize("kc") == "KC"
    with pytest.raises(KeyError):
        normalize("XXX")


def test_resolve_free_text():
    assert resolve("Kansas City") == "KC"
    assert resolve("Los Angeles Chargers") == "LAC"
    assert resolve("Los Angeles Rams") == "LA"
    assert resolve("12 LAR") == "LA"
    assert resolve("NY Jets") == "NYJ"
    assert resolve("Washington Commanders") == "WAS"
    assert resolve("Niners") == "SF"
    assert resolve("garbage text here") is None
    assert len(TEAMS) == 32

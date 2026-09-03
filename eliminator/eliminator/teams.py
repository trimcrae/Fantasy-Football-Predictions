"""Team code normalisation.

nflverse uses the current franchise code from the season a move happened (STL -> LA in
2016, SD -> LAC in 2017, OAK -> LV in 2020). Ratings need franchise continuity across
seasons, so everything is mapped onto the 32 current codes.
"""
from __future__ import annotations

import re

TEAMS: list[str] = [
    "ARI", "ATL", "BAL", "BUF", "CAR", "CHI", "CIN", "CLE", "DAL", "DEN", "DET", "GB",
    "HOU", "IND", "JAX", "KC", "LA", "LAC", "LV", "MIA", "MIN", "NE", "NO", "NYG",
    "NYJ", "PHI", "PIT", "SEA", "SF", "TB", "TEN", "WAS",
]
TEAM_INDEX: dict[str, int] = {t: i for i, t in enumerate(TEAMS)}

# Alternate abbreviations seen in nflverse history, ESPN, PFR, inpredictable, Yahoo.
ALIASES: dict[str, str] = {
    "STL": "LA", "LAR": "LA", "RAM": "LA",
    "SD": "LAC", "SDG": "LAC",
    "OAK": "LV", "LVR": "LV", "RAI": "LV",
    "JAC": "JAX", "WSH": "WAS", "GNB": "GB", "KAN": "KC", "NWE": "NE", "NOR": "NO",
    "SFO": "SF", "TAM": "TB", "HST": "HOU", "BLT": "BAL", "CLV": "CLE", "ARZ": "ARI",
    "NYG": "NYG", "NYJ": "NYJ",
}

# City / nickname words -> code, used to resolve free-text team labels (inpredictable,
# manual files). Multi-word cities are matched on the full lower-cased string first.
TEAM_NAMES: dict[str, tuple[str, str]] = {
    "ARI": ("Arizona", "Cardinals"), "ATL": ("Atlanta", "Falcons"), "BAL": ("Baltimore", "Ravens"),
    "BUF": ("Buffalo", "Bills"), "CAR": ("Carolina", "Panthers"), "CHI": ("Chicago", "Bears"),
    "CIN": ("Cincinnati", "Bengals"), "CLE": ("Cleveland", "Browns"), "DAL": ("Dallas", "Cowboys"),
    "DEN": ("Denver", "Broncos"), "DET": ("Detroit", "Lions"), "GB": ("Green Bay", "Packers"),
    "HOU": ("Houston", "Texans"), "IND": ("Indianapolis", "Colts"), "JAX": ("Jacksonville", "Jaguars"),
    "KC": ("Kansas City", "Chiefs"), "LA": ("Los Angeles Rams", "Rams"), "LAC": ("Los Angeles Chargers", "Chargers"),
    "LV": ("Las Vegas", "Raiders"), "MIA": ("Miami", "Dolphins"), "MIN": ("Minnesota", "Vikings"),
    "NE": ("New England", "Patriots"), "NO": ("New Orleans", "Saints"), "NYG": ("New York Giants", "Giants"),
    "NYJ": ("New York Jets", "Jets"), "PHI": ("Philadelphia", "Eagles"), "PIT": ("Pittsburgh", "Steelers"),
    "SEA": ("Seattle", "Seahawks"), "SF": ("San Francisco", "49ers"), "TB": ("Tampa Bay", "Buccaneers"),
    "TEN": ("Tennessee", "Titans"), "WAS": ("Washington", "Commanders"),
}
_NICKNAME_ALIASES = {"redskins": "WAS", "football team": "WAS", "oilers": "TEN", "st. louis": "LA", "st louis": "LA",
                     "san diego": "LAC", "oakland": "LV", "la rams": "LA", "la chargers": "LAC", "ny giants": "NYG",
                     "ny jets": "NYJ", "niners": "SF", "bucs": "TB", "pats": "NE", "skins": "WAS"}


def normalize(code: str) -> str:
    """Map any historical / vendor abbreviation to the current franchise code."""
    c = str(code).strip().upper()
    c = ALIASES.get(c, c)
    if c not in TEAM_INDEX:
        raise KeyError(f"unknown team code: {code!r}")
    return c


def resolve(label: str) -> str | None:
    """Resolve free text such as 'Kansas City', 'Chiefs', '3 KC', 'LA Rams' to a code.

    Returns None when nothing matches, so callers can skip junk rows.
    """
    if label is None:
        return None
    text = str(label).strip()
    if not text:
        return None
    # Bare or rank-prefixed abbreviation, e.g. "KC", "12 LAR", "KC (7-3)".
    m = re.search(r"\b([A-Za-z]{2,3})\b", text)
    upper_tokens = [t.upper() for t in re.findall(r"[A-Za-z]{2,3}", text)]
    for tok in upper_tokens:
        tok = ALIASES.get(tok, tok)
        if tok in TEAM_INDEX and len(text) <= 12:
            return tok
    low = re.sub(r"[^a-z0-9. ]+", " ", text.lower())
    low = re.sub(r"\s+", " ", low).strip()
    for alias, code in _NICKNAME_ALIASES.items():
        if alias in low:
            return code
    # Full city names first (longest match wins so "Los Angeles Rams" beats "Los Angeles").
    best: tuple[int, str] | None = None
    for code, (city, nick) in TEAM_NAMES.items():
        for name in (city, nick):
            n = name.lower()
            if n in low and (best is None or len(n) > best[0]):
                best = (len(n), code)
    if best:
        return best[1]
    if m:
        tok = ALIASES.get(m.group(1).upper(), m.group(1).upper())
        if tok in TEAM_INDEX:
            return tok
    return None


def display_name(code: str) -> str:
    city, nick = TEAM_NAMES[normalize(code)]
    return f"{city} {nick}" if nick not in city else city

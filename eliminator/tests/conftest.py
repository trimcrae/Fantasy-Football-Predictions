import datetime as dt
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eliminator.data.schedule import ET, _normalise  # noqa: E402
from eliminator.teams import TEAMS  # noqa: E402


def synthetic_season(season: int, weeks: int = 18, seed: int = 0, lines_through: int = 3,
                     results_through: int = 0, strength: dict | None = None) -> pd.DataFrame:
    """A fake regular season: 32 teams, each week 16 games (no byes) with spreads/moneylines
    for the first `lines_through` weeks and results for the first `results_through` weeks."""
    rng = np.random.default_rng(seed)
    strength = strength or {t: float(rng.normal(0, 4)) for t in TEAMS}
    rows = []
    start = dt.date(season, 9, 10)
    for w in range(1, weeks + 1):
        order = list(rng.permutation(TEAMS))
        day = start + dt.timedelta(days=7 * (w - 1))
        for i in range(0, 32, 2):
            home, away = order[i], order[i + 1]
            spread = strength[home] - strength[away] + 1.6 + rng.normal(0, 1.0)
            has_line = w <= lines_through
            played = w <= results_through
            result = spread + rng.normal(0, 11.5) if played else np.nan
            p = 1 / (1 + np.exp(-spread / 7.0))
            ml_home = -round(100 * p / (1 - p)) if p >= 0.5 else round(100 * (1 - p) / p)
            ml_away = round(100 * p / (1 - p)) if p >= 0.5 else -round(100 * (1 - p) / p)
            rows.append({
                "game_id": f"{season}_{w:02d}_{away}_{home}", "season": season, "game_type": "REG", "week": w,
                "gameday": day.isoformat(), "weekday": "Sunday", "gametime": "13:00" if i < 24 else "20:20",
                "away_team": away, "home_team": home, "location": "Home", "result": round(result) if played else np.nan,
                "home_score": np.nan, "away_score": np.nan, "away_rest": 7, "home_rest": 7 if i else 14,
                "away_moneyline": ml_away if has_line else np.nan, "home_moneyline": ml_home if has_line else np.nan,
                "spread_line": round(spread * 2) / 2 if has_line else np.nan, "total_line": np.nan, "div_game": 0,
                "roof": "outdoors", "away_qb_id": "", "home_qb_id": "", "away_qb_name": "", "home_qb_name": "",
                "stadium": "",
            })
    return _normalise(pd.DataFrame(rows))


@pytest.fixture
def games_all():
    prev = synthetic_season(2025, lines_through=18, results_through=18, seed=1)
    cur = synthetic_season(2026, lines_through=3, results_through=0, seed=2)
    return pd.concat([prev, cur], ignore_index=True)


@pytest.fixture
def cfg():
    from eliminator.config import DEFAULTS, _deep_merge
    c = _deep_merge(DEFAULTS, {"simulation": {"scenarios": 2000, "seed": 1}, "portfolio": {"candidates_per_slot": 12, "improve_passes": 1}})
    return c


@pytest.fixture
def before_week1():
    return dt.datetime(2026, 9, 9, 12, 0, tzinfo=ET)

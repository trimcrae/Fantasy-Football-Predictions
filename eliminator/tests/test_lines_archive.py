import datetime as dt

import numpy as np
import pandas as pd

from eliminator.data.lines_archive import archive_lines, load_archive, posted_line_variance
from eliminator.data.schedule import ET, regular_season


def test_archive_appends_once_per_day_and_only_future_games(tmp_path, games_all):
    games = regular_season(games_all, 2026)
    now = dt.datetime(2026, 9, 9, 12, 0, tzinfo=ET)
    n = archive_lines(games, 2026, now, tmp_path)
    assert n == 16 * 3                     # lines are posted for weeks 1-3 in the fixture
    assert archive_lines(games, 2026, now, tmp_path) == 0
    later = dt.datetime(2026, 9, 14, 12, 0, tzinfo=ET)   # week 1 has kicked off
    assert archive_lines(games, 2026, later, tmp_path) == 16 * 2
    arch = load_archive(2026, tmp_path)
    assert len(arch) == 16 * 5 and set(arch.columns) >= {"as_of", "week", "home", "spread_line"}


def test_posted_line_variance_recovers_movement_by_horizon(games_all):
    # last season is fully played: pretend we saw each game's line h weeks early, moved by noise with var 1 + 0.5h
    games = regular_season(games_all, 2025)
    rng = np.random.default_rng(0)
    rows = []
    for g in games.itertuples(index=False):
        for h in range(0, 6):
            wk = int(g.week) - h
            if wk < 1:
                continue
            seen = games[games["week"] == wk]["kickoff"].min() - dt.timedelta(days=2)
            rows.append({"as_of": seen.date(), "week": int(g.week), "home": g.home, "away": g.away,
                         "spread_line": float(g.spread_line) + rng.normal(0, np.sqrt(1 + 0.5 * h)),
                         "home_moneyline": np.nan, "away_moneyline": np.nan})
    arch = pd.DataFrame(rows)
    fit = posted_line_variance(arch, games)
    assert fit and fit["n_obs"] > 1000 and fit["horizons"] >= 5
    assert abs(fit["posted_line_var_a"] - 1.0) < 0.3 and abs(fit["posted_line_var_b"] - 0.5) < 0.15


def test_posted_line_variance_needs_enough_data(games_all):
    games = regular_season(games_all, 2025)
    arch = pd.DataFrame([{"as_of": dt.date(2025, 9, 8), "week": 1, "home": games.iloc[0]["home"], "away": games.iloc[0]["away"],
                          "spread_line": 3.0, "home_moneyline": np.nan, "away_moneyline": np.nan}])
    assert posted_line_variance(arch, games) is None
    assert posted_line_variance(pd.DataFrame(), games) is None

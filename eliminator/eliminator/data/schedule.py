"""NFL schedule, market lines, rest days, starting QBs and results.

Source: Lee Sharpe / nflverse ``games.csv`` (1999-present, refreshed in season). It carries
the full upcoming schedule with whatever moneylines and spreads are posted, so it serves
both as the schedule and as the primary Vegas feed. Spread sign convention: positive means
the home team is favoured; ``result`` is home score minus away score.
"""
from __future__ import annotations

import datetime as dt
import os
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import requests

from ..teams import normalize

GAMES_URL = "https://raw.githubusercontent.com/nflverse/nfldata/master/data/games.csv"
ET = ZoneInfo("America/New_York")

_COLS = [
    "game_id", "season", "game_type", "week", "gameday", "weekday", "gametime", "away_team",
    "away_score", "home_team", "home_score", "location", "result", "away_rest", "home_rest",
    "away_moneyline", "home_moneyline", "spread_line", "total_line", "div_game", "roof",
    "away_qb_id", "home_qb_id", "away_qb_name", "home_qb_name", "stadium",
]


def cache_dir() -> Path:
    d = Path(os.environ.get("ELIMINATOR_CACHE", Path(__file__).resolve().parents[2] / "data" / "cache"))
    d.mkdir(parents=True, exist_ok=True)
    return d


def fetch_games(dest: Path | None = None, timeout: int = 60) -> Path:
    dest = dest or cache_dir() / "games.csv"
    r = requests.get(GAMES_URL, timeout=timeout)
    r.raise_for_status()
    tmp = dest.with_suffix(".tmp")
    tmp.write_bytes(r.content)
    tmp.replace(dest)
    return dest


def load_games(path: Path | None = None, refresh: bool = False, max_age_hours: float | None = None) -> pd.DataFrame:
    """Load and normalise games.csv. Refreshes from nflverse when asked or when stale."""
    path = path or cache_dir() / "games.csv"
    stale = False
    if path.exists() and max_age_hours is not None:
        age_h = (dt.datetime.now().timestamp() - path.stat().st_mtime) / 3600.0
        stale = age_h > max_age_hours
    if refresh or stale or not path.exists():
        try:
            fetch_games(path)
        except Exception as exc:  # network is optional once cached
            if not path.exists():
                raise
            print(f"[schedule] refresh failed ({exc}); using cached copy")
    df = pd.read_csv(path, usecols=lambda c: c in _COLS, low_memory=False)
    return _normalise(df)


def _normalise(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["home"] = df["home_team"].map(normalize)
    df["away"] = df["away_team"].map(normalize)
    df["week"] = df["week"].astype(int)
    df["season"] = df["season"].astype(int)
    df["neutral"] = df["location"].fillna("Home").str.lower().ne("home")
    for c in ["away_rest", "home_rest"]:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(7).astype(int)
    for c in ["away_moneyline", "home_moneyline", "spread_line", "total_line", "result", "home_score", "away_score"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["div_game"] = pd.to_numeric(df["div_game"], errors="coerce").fillna(0).astype(int)
    df["kickoff"] = [_kickoff(d, t) for d, t in zip(df["gameday"], df["gametime"])]
    df["played"] = df["result"].notna()
    df["home_win"] = np.where(df["played"], (df["result"] > 0).astype(float), np.nan)
    df.loc[df["played"] & (df["result"] == 0), "home_win"] = 0.5
    df["rest_diff"] = (df["home_rest"] - df["away_rest"]).clip(-7, 7)
    return df.sort_values(["season", "week", "kickoff"]).reset_index(drop=True)


def _kickoff(day, time) -> dt.datetime:
    d = pd.to_datetime(day).to_pydatetime()
    hh, mm = 13, 0
    if isinstance(time, str) and ":" in time:
        try:
            hh, mm = (int(x) for x in time.split(":")[:2])
        except ValueError:
            pass
    return dt.datetime(d.year, d.month, d.day, hh, mm, tzinfo=ET)


def regular_season(df: pd.DataFrame, season: int) -> pd.DataFrame:
    out = df[(df["season"] == season) & (df["game_type"] == "REG")].copy()
    return out.reset_index(drop=True)


def season_weeks(df: pd.DataFrame, season: int) -> int:
    return int(regular_season(df, season)["week"].max())


def current_week(games: pd.DataFrame, now: dt.datetime | None = None) -> int:
    """Earliest regular-season week with a game that has not kicked off yet.

    Once every game of a week has started the plan moves on to the next week even if
    results have not been published, so a Monday-night rerun already plans next week.
    """
    now = now or dt.datetime.now(tz=ET)
    pending = games[games["kickoff"] > now]
    if pending.empty:
        return int(games["week"].max())
    return int(pending["week"].min())


def latest_season(df: pd.DataFrame) -> int:
    return int(df["season"].max())


def team_games(games: pd.DataFrame, team: str) -> pd.DataFrame:
    return games[(games["home"] == team) | (games["away"] == team)]

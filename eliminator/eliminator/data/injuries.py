"""Quarterback availability from nflverse injury reports and depth charts.

Injury reports (Wed-Sat practice/game status) and depth charts (QB1 per team) are
published as release assets by nflverse. They are used to *suggest* entries for the
manual QB ledger (state/qb_status.yaml); the ledger is what the model actually uses,
because injury duration is a judgement call that no feed gets right automatically.
"""
from __future__ import annotations

import datetime as dt
from pathlib import Path

import pandas as pd
import requests

from ..teams import normalize
from .schedule import cache_dir

INJURIES_URL = "https://github.com/nflverse/nflverse-data/releases/download/injuries/injuries_{season}.csv"
DEPTH_URL = "https://github.com/nflverse/nflverse-data/releases/download/depth_charts/depth_charts_{season}.csv"

STATUS_OUT = {"out", "injured reserve", "ir", "pup", "nfi", "doubtful"}


def _fetch(url: str, dest: Path, refresh: bool, max_age_hours: float) -> Path | None:
    if dest.exists() and not refresh:
        age_h = (dt.datetime.now().timestamp() - dest.stat().st_mtime) / 3600.0
        if age_h <= max_age_hours:
            return dest
    try:
        r = requests.get(url, timeout=120)
        if r.status_code == 404:
            return dest if dest.exists() else None
        r.raise_for_status()
        dest.write_bytes(r.content)
    except Exception as exc:
        print(f"[injuries] fetch failed for {url}: {exc}")
        return dest if dest.exists() else None
    return dest


def load_injuries(season: int, refresh: bool = False, max_age_hours: float = 6.0) -> pd.DataFrame | None:
    p = _fetch(INJURIES_URL.format(season=season), cache_dir() / f"injuries_{season}.csv", refresh, max_age_hours)
    if p is None:
        return None
    df = pd.read_csv(p, low_memory=False)
    df["team"] = df["team"].map(normalize)
    return df


def load_depth_qb1(season: int, refresh: bool = False, max_age_hours: float = 24.0,
                   as_of: dt.datetime | None = None) -> pd.DataFrame | None:
    """QB1 per team from the most recent depth-chart snapshot (optionally as of a date)."""
    p = _fetch(DEPTH_URL.format(season=season), cache_dir() / f"depth_charts_{season}.csv", refresh, max_age_hours)
    if p is None:
        return None
    usecols = ["dt", "team", "player_name", "gsis_id", "pos_abb", "pos_rank"]
    df = pd.read_csv(p, usecols=lambda c: c in usecols, low_memory=False)
    df = df[df["pos_abb"] == "QB"].copy()
    df["dt"] = pd.to_datetime(df["dt"], utc=True)
    if as_of is not None:
        df = df[df["dt"] <= pd.Timestamp(as_of).tz_convert("UTC")]
    if df.empty:
        return None
    latest = df.groupby("team")["dt"].transform("max")
    df = df[df["dt"] == latest]
    df = df[df["pos_rank"] == 1].drop_duplicates("team")
    df["team"] = df["team"].map(normalize)
    return df[["team", "player_name", "gsis_id", "dt"]].rename(columns={"player_name": "qb1", "dt": "snapshot"})


def report_weeks(season: int) -> set[int]:
    """Weeks for which nflverse has published any injury report rows (cache only, no fetch)."""
    inj = load_injuries(season, refresh=False, max_age_hours=float("inf"))
    if inj is None or inj.empty:
        return set()
    return {int(w) for w in inj["week"].dropna().unique()}


def qb_watch(season: int, week: int, refresh: bool = False) -> pd.DataFrame:
    """QBs with a game-status designation this week, flagged when they are the team's QB1.

    Columns: team, player, status, injury, practice, is_qb1.
    """
    inj = load_injuries(season, refresh=refresh)
    if inj is None:
        return pd.DataFrame(columns=["team", "player", "status", "injury", "practice", "is_qb1"])
    q = inj[(inj["position"] == "QB") & (inj["week"] == week)].copy()
    q["status"] = q["report_status"].fillna("").str.strip()
    q["practice"] = q["practice_status"].fillna("").str.strip()
    q = q[(q["status"] != "") | q["practice"].str.startswith("Did Not")]
    qb1 = load_depth_qb1(season, refresh=refresh)
    starters = set(zip(qb1["team"], qb1["qb1"])) if qb1 is not None else set()
    q["is_qb1"] = [(t, n) in starters for t, n in zip(q["team"], q["full_name"])]
    out = q.rename(columns={"full_name": "player", "report_primary_injury": "injury"})
    return out[["team", "player", "status", "injury", "practice", "is_qb1"]].sort_values(["is_qb1", "team"], ascending=[False, True]).reset_index(drop=True)

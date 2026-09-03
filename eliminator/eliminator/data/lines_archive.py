"""Archive of posted lines, as seen on each run, and the line-movement fit built from it.

The schedule feed keeps one line per game (the latest), so it cannot say how far a line
posted weeks ahead tends to move before kickoff. Every ``snapshot`` run therefore appends
the lines it sees for games that have not kicked off to ``site/data/lines/<season>.csv``
(one row per day, game). Once enough of those games have closed, ``posted_line_variance``
fits ``var(h) = a + b * h`` (points squared) to the squared difference between the line
posted ``h`` weeks out and the closing line, which is the uncertainty the projection puts on
a posted line. Until then the configured defaults apply.
"""
from __future__ import annotations

import datetime as dt
from pathlib import Path

import numpy as np
import pandas as pd

from ..config import ROOT
from .schedule import ET, current_week

ARCHIVE_DIR = ROOT / "site" / "data" / "lines"
COLS = ["as_of", "week", "home", "away", "spread_line", "home_moneyline", "away_moneyline"]


def archive_path(season: int, archive_dir: Path = ARCHIVE_DIR) -> Path:
    return archive_dir / f"{season}.csv"


def load_archive(season: int, archive_dir: Path = ARCHIVE_DIR) -> pd.DataFrame:
    p = archive_path(season, archive_dir)
    if not p.exists():
        return pd.DataFrame(columns=COLS)
    df = pd.read_csv(p)
    df["as_of"] = pd.to_datetime(df["as_of"]).dt.date
    return df


def archive_lines(games: pd.DataFrame, season: int, now: dt.datetime, archive_dir: Path = ARCHIVE_DIR) -> int:
    """Append today's posted lines for games that have not kicked off. Returns rows added."""
    day = now.astimezone(ET).date()
    pending = games[(games["kickoff"] > now) & (games["spread_line"].notna() | games["home_moneyline"].notna())]
    if pending.empty:
        return 0
    rows = pd.DataFrame({"as_of": day, "week": pending["week"].astype(int), "home": pending["home"], "away": pending["away"],
                         "spread_line": pending["spread_line"], "home_moneyline": pending["home_moneyline"],
                         "away_moneyline": pending["away_moneyline"]})
    old = load_archive(season, archive_dir)
    if not old.empty:
        seen = set(zip(old["as_of"], old["week"], old["home"]))
        rows = rows[[(a, w, h) not in seen for a, w, h in zip(rows["as_of"], rows["week"], rows["home"])]]
    if rows.empty:
        return 0
    out = pd.concat([old, rows], ignore_index=True) if not old.empty else rows
    p = archive_path(season, archive_dir)
    p.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(p, index=False)
    return int(len(rows))


def posted_line_variance(archive: pd.DataFrame, games: pd.DataFrame, min_obs: int = 150,
                         min_horizons: int = 3) -> dict | None:
    """Fit var(h) = a + b*h from archived lines against the closing line of games since played.

    Uses each (day, game) observation once, with ``h`` the number of weeks between the
    week the line was seen in and the game's week. Returns None until there are at least
    ``min_obs`` observations spanning ``min_horizons`` distinct horizons.
    """
    if archive is None or archive.empty:
        return None
    played = games[games["played"] & games["spread_line"].notna()][["week", "home", "spread_line"]].rename(columns={"spread_line": "close"})
    df = archive.dropna(subset=["spread_line"]).merge(played, on=["week", "home"], how="inner")
    if df.empty:
        return None
    # horizon: week of the game minus the week in progress when the line was seen
    as_of_ts = pd.to_datetime(df["as_of"]).dt.tz_localize(ET) + pd.Timedelta(hours=12)
    seen_week = np.array([current_week(games, t.to_pydatetime()) for t in as_of_ts])
    df = df.assign(h=df["week"].astype(int) - seen_week)
    df = df[df["h"] >= 0]
    if len(df) < min_obs or df["h"].nunique() < min_horizons:
        return None
    sq = (df["spread_line"] - df["close"]) ** 2
    # method of moments per horizon, then a weighted least-squares line through the means
    g = pd.DataFrame({"h": df["h"], "sq": sq}).groupby("h")["sq"].agg(["mean", "count"]).reset_index()
    w = g["count"].to_numpy(float)
    X = np.column_stack([np.ones(len(g)), g["h"].to_numpy(float)])
    W = np.sqrt(w)[:, None]
    coef, *_ = np.linalg.lstsq(X * W, g["mean"].to_numpy(float) * W[:, 0], rcond=None)
    a, b = float(max(coef[0], 0.05)), float(max(coef[1], 0.0))
    return {"posted_line_var_a": round(a, 3), "posted_line_var_b": round(b, 3), "n_obs": int(len(df)),
            "horizons": int(df["h"].nunique()), "by_horizon": {int(r.h): (round(float(r["mean"]), 2), int(r["count"])) for _, r in g.iterrows()}}

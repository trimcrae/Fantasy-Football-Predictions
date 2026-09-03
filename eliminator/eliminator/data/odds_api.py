"""Optional live moneylines from The Odds API (https://the-odds-api.com, free tier is enough).

The nflverse feed refreshes about daily; on game day the books move. With an API key in
``config.yaml`` (``data.odds_api_key``) or ``ODDS_API_KEY`` in the environment, ``plan``
pulls the current NFL moneylines and turns them into line overrides for games that have not
kicked off. Prices are averaged across books after de-vigging (a consensus is sharper than
any single book). Untested against the live endpoint from the build environment; the request
and parsing follow the documented v4 schema and any failure just falls back to the feed.
"""
from __future__ import annotations

import datetime as dt
import os

import numpy as np
import pandas as pd
import requests

from ..probability import american_to_prob, devig, prob_to_spread
from ..teams import resolve

URL = "https://api.the-odds-api.com/v4/sports/americanfootball_nfl/odds"


def fetch_moneylines(api_key: str, region: str = "us", timeout: int = 30) -> list[dict]:
    r = requests.get(URL, params={"apiKey": api_key, "regions": region, "markets": "h2h", "oddsFormat": "american"},
                     timeout=timeout)
    r.raise_for_status()
    return r.json()


def consensus_overrides(events: list[dict], games: pd.DataFrame, week: int, sigma: float) -> list[dict]:
    """Map API events onto this week's games and return override records (week, home, away, spread)."""
    wk = games[games["week"] == week]
    out = []
    for ev in events:
        home = resolve(ev.get("home_team", "")); away = resolve(ev.get("away_team", ""))
        if home is None or away is None:
            continue
        g = wk[(wk["home"] == home) & (wk["away"] == away)]
        if g.empty:
            continue
        probs = []
        for book in ev.get("bookmakers", []):
            for mk in book.get("markets", []):
                if mk.get("key") != "h2h":
                    continue
                prices = {resolve(o.get("name", "")): o.get("price") for o in mk.get("outcomes", [])}
                if prices.get(home) is None or prices.get(away) is None:
                    continue
                ph, _ = devig(american_to_prob(prices[home]), american_to_prob(prices[away]))
                probs.append(ph)
        if not probs:
            continue
        p = float(np.mean(probs))
        out.append({"week": week, "home": home, "away": away, "spread": float(prob_to_spread(p, sigma)),
                    "books": len(probs), "p_home": p, "source": "odds-api"})
    return out


def live_overrides(cfg: dict, games: pd.DataFrame, week: int) -> list[dict]:
    key = cfg["data"].get("odds_api_key") or os.environ.get("ODDS_API_KEY")
    if not key:
        return []
    try:
        events = fetch_moneylines(key, cfg["data"].get("odds_api_region", "us"))
        recs = consensus_overrides(events, games, week, float(cfg["model"]["sigma"]))
        print(f"[odds-api] {len(recs)} games priced from live books at {dt.datetime.now():%H:%M}")
        return recs
    except Exception as exc:
        print(f"[odds-api] unavailable ({exc}); using the nflverse feed")
        return []

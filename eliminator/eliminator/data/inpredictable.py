"""Vegas-derived team strength from inpredictable.com.

``stats.inpredictable.com/nfl/ssnPower.php`` publishes power ratings backed out of the
betting market: GPF (generic points favoured) is how many points a team would be favoured
by against an average opponent on a neutral field. That is exactly the rating scale the
projection model uses, so it plugs straight in.

The parser is deliberately loose about layout: it scans every table for a header row that
mentions GPF (or Rating/Power) and a team column, and resolves team labels by name or code.
Use ``load_ratings(from_file=...)`` with a saved HTML page or a CSV (team,gpf[,ogpf,dgpf])
when the site is not reachable from where you run this.
"""
from __future__ import annotations

import datetime as dt
import json
import re
from html.parser import HTMLParser
from pathlib import Path

import pandas as pd
import requests

from ..teams import TEAMS, resolve
from .schedule import cache_dir

URL = "https://stats.inpredictable.com/nfl/ssnPower.php"
HEADERS = {"User-Agent": "Mozilla/5.0 (eliminator strategy tool; +https://github.com/trimcrae)"}


class _TableParser(HTMLParser):
    """Collects every <table> as a list of rows of cell texts."""

    def __init__(self) -> None:
        super().__init__()
        self.tables: list[list[list[str]]] = []
        self._row: list[str] | None = None
        self._cell: list[str] | None = None
        self._depth = 0

    def handle_starttag(self, tag, attrs):
        if tag == "table":
            self._depth += 1
            self.tables.append([])
        elif tag == "tr" and self._depth:
            self._row = []
        elif tag in ("td", "th") and self._row is not None:
            self._cell = []

    def handle_endtag(self, tag):
        if tag in ("td", "th") and self._cell is not None and self._row is not None:
            self._row.append(re.sub(r"\s+", " ", "".join(self._cell)).strip())
            self._cell = None
        elif tag == "tr" and self._row is not None and self._depth:
            if self._row:
                self.tables[-1].append(self._row)
            self._row = None
        elif tag == "table" and self._depth:
            self._depth -= 1

    def handle_data(self, data):
        if self._cell is not None:
            self._cell.append(data)


_RATING_PATTERNS = [r"^gpf$", r"^gpf\b", r"generic points", r"^rating$", r"^power$", r"^rtg$"]
_NUM = re.compile(r"[-+]?\d+(?:\.\d+)?")


def _find_col(header: list[str], patterns: list[str]) -> int | None:
    for pat in patterns:
        for i, h in enumerate(header):
            if re.search(pat, h.strip().lower()):
                return i
    return None


def parse_ratings_html(html: str) -> pd.DataFrame:
    """Return DataFrame[team, gpf, ogpf, dgpf] parsed from the inpredictable page."""
    parser = _TableParser()
    parser.feed(html)
    best: pd.DataFrame | None = None
    for table in parser.tables:
        for hi, header in enumerate(table[:5]):
            rating_col = _find_col(header, _RATING_PATTERNS)
            team_col = _find_col(header, [r"^team$", r"team"])
            if rating_col is None or team_col is None:
                continue
            off_col = _find_col(header, [r"^ogpf$", r"^off"])
            def_col = _find_col(header, [r"^dgpf$", r"^def"])
            rows = []
            for row in table[hi + 1:]:
                if len(row) <= max(rating_col, team_col):
                    continue
                team = resolve(row[team_col])
                m = _NUM.search(row[rating_col])
                if team is None or not m:
                    continue
                rec = {"team": team, "gpf": float(m.group())}
                for name, col in (("ogpf", off_col), ("dgpf", def_col)):
                    if col is not None and col < len(row):
                        mm = _NUM.search(row[col])
                        rec[name] = float(mm.group()) if mm else float("nan")
                rows.append(rec)
            df = pd.DataFrame(rows).drop_duplicates("team")
            if len(df) >= 28 and (best is None or len(df) > len(best)):
                best = df
    if best is None:
        raise ValueError("no power-rating table with a GPF column found in page")
    missing = sorted(set(TEAMS) - set(best["team"]))
    if missing:
        raise ValueError(f"inpredictable table missing teams: {missing}")
    best["gpf"] = best["gpf"] - best["gpf"].mean()  # centre: an average team is 0
    return best.sort_values("gpf", ascending=False).reset_index(drop=True)


def fetch_ratings(season: int | None = None, timeout: int = 30) -> pd.DataFrame:
    params = {"season": season} if season else None
    r = requests.get(URL, params=params, headers=HEADERS, timeout=timeout)
    r.raise_for_status()
    df = parse_ratings_html(r.text)
    df.attrs["fetched_at"] = dt.datetime.now(dt.timezone.utc).isoformat()
    return df


def save_ratings(df: pd.DataFrame, path: Path | None = None) -> Path:
    path = path or cache_dir() / "inpredictable.json"
    payload = {"fetched_at": df.attrs.get("fetched_at"), "ratings": df.to_dict(orient="records")}
    path.write_text(json.dumps(payload, indent=1))
    return path


def load_cached(path: Path | None = None, max_age_hours: float | None = None) -> pd.DataFrame | None:
    path = path or cache_dir() / "inpredictable.json"
    if not path.exists():
        return None
    payload = json.loads(path.read_text())
    df = pd.DataFrame(payload["ratings"])
    df.attrs["fetched_at"] = payload.get("fetched_at")
    if max_age_hours is not None and payload.get("fetched_at"):
        age = dt.datetime.now(dt.timezone.utc) - dt.datetime.fromisoformat(payload["fetched_at"])
        if age.total_seconds() > max_age_hours * 3600:
            df.attrs["stale"] = True
    return df


def load_ratings(from_file: str | Path | None = None, season: int | None = None,
                 refresh: bool = True, max_age_hours: float = 24.0) -> pd.DataFrame | None:
    """inpredictable ratings from a file (HTML page or CSV), the live site, or the cache.

    Returns None when nothing is available so the caller can fall back to the market fit.
    """
    if from_file:
        p = Path(from_file)
        text = p.read_text(errors="ignore")
        if p.suffix.lower() in (".htm", ".html") or "<table" in text.lower():
            df = parse_ratings_html(text)
        else:
            raw = pd.read_csv(p)
            raw.columns = [c.strip().lower() for c in raw.columns]
            tcol = next(c for c in raw.columns if "team" in c)
            rcol = next((c for c in raw.columns if c in ("gpf", "rating", "power")), raw.columns[1])
            df = pd.DataFrame({"team": raw[tcol].map(resolve), "gpf": pd.to_numeric(raw[rcol], errors="coerce")})
            df = df.dropna().drop_duplicates("team")
            df["gpf"] = df["gpf"] - df["gpf"].mean()
        df.attrs["fetched_at"] = dt.datetime.now(dt.timezone.utc).isoformat()
        df.attrs["source"] = str(p)
        save_ratings(df)
        return df
    if refresh:
        try:
            df = fetch_ratings(season=season)
            df.attrs["source"] = URL
            save_ratings(df)
            return df
        except Exception as exc:
            print(f"[inpredictable] fetch failed: {exc}")
    cached = load_cached(max_age_hours=max_age_hours)
    if cached is not None:
        cached.attrs["source"] = "cache"
    return cached

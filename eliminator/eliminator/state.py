"""Pool state: what each entry has picked, and what that implies given results so far.

state/<pool>.yaml::

    name: ESPN eliminator
    mode: multi            # multi (single elimination, many entries) | strikes (one entry)
    entries: 25            # multi only
    strikes: 2             # strikes that eliminate: out on the 2nd loss (0 or 1 = single elimination)
    season: 2026
    picks:                 # entry id -> {week: team}; entry ids are 1..N by default
      "1": {1: LAC, 2: BUF}
      "2": {1: JAX}

Picks are written by ``eliminator plan --commit`` and can be edited by hand. A pick whose
game has kicked off is locked; anything else is provisional and re-optimised on the next run.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd
import yaml

from .teams import normalize


@dataclass
class EntryStatus:
    entry_id: str
    picks: dict[int, str]
    used: set[str]                       # teams no longer available (past or locked picks)
    results: dict[int, str]              # week -> 'win' | 'loss' | 'pending' | 'missed'
    losses: int
    alive: bool
    locked_now: str | None               # current-week pick whose game already kicked off
    provisional_now: str | None          # current-week pick that can still be changed


@dataclass
class PoolState:
    name: str
    mode: str
    n_entries: int
    strikes: int
    season: int | None
    picks: dict[str, dict[int, str]] = field(default_factory=dict)
    path: Path | None = None

    @property
    def lives(self) -> int:
        """Losses an entry can take and stay alive. ``strikes`` counts the loss that puts it out
        (two strikes: the second loss eliminates), so a two-strike entry has one life."""
        return lives(self.strikes)

    @staticmethod
    def load(path: Path) -> "PoolState":
        raw = yaml.safe_load(path.read_text()) or {}
        picks = {}
        for eid, wk in (raw.get("picks") or {}).items():
            picks[str(eid)] = {int(w): normalize(t) for w, t in (wk or {}).items() if t}
        st = PoolState(name=str(raw.get("name", path.stem)), mode=str(raw.get("mode", "multi")),
                       n_entries=int(raw.get("entries", 1)), strikes=int(raw.get("strikes", 0)),
                       season=raw.get("season"), picks=picks, path=path)
        if st.mode == "strikes":
            st.n_entries = 1
        return st

    def save(self, path: Path | None = None) -> Path:
        path = path or self.path
        assert path is not None
        data = {"name": self.name, "mode": self.mode, "entries": self.n_entries, "strikes": self.strikes,
                "season": self.season,
                "picks": {eid: {int(w): t for w, t in sorted(p.items())} for eid, p in sorted(self.picks.items(), key=lambda kv: _idkey(kv[0]))}}
        path.write_text(yaml.safe_dump(data, sort_keys=False))
        return path

    def entry_ids(self) -> list[str]:
        ids = [str(i) for i in range(1, self.n_entries + 1)]
        for k in self.picks:
            if k not in ids:
                ids.append(k)
        return ids


def lives(strikes: int) -> int:
    """Losses allowed before elimination for a pool that eliminates on the ``strikes``-th loss."""
    return max(int(strikes) - 1, 0)


def _idkey(s: str):
    return (0, int(s)) if s.isdigit() else (1, s)


def evaluate_entries(state: PoolState, games: pd.DataFrame, current_week: int, now) -> list[EntryStatus]:
    """Apply results to every entry. A tie or a missing pick in a finished week is a loss."""
    out = []
    for eid in state.entry_ids():
        picks = state.picks.get(eid, {})
        used: set[str] = set(); results: dict[int, str] = {}; losses = 0
        locked_now = provisional_now = None
        for w in range(1, current_week + 1):
            team = picks.get(w)
            if w < current_week:
                if team is None:
                    results[w] = "missed"; losses += 1
                    continue
                used.add(team)
                g = games[(games["week"] == w) & ((games["home"] == team) | (games["away"] == team))]
                if g.empty:
                    results[w] = "missed"; losses += 1
                    continue
                g = g.iloc[0]
                if not g["played"]:
                    results[w] = "pending"
                else:
                    won = (g["result"] > 0 and g["home"] == team) or (g["result"] < 0 and g["away"] == team)
                    results[w] = "win" if won else "loss"
                    losses += 0 if won else 1
            else:  # current week
                if team is None:
                    continue
                g = games[(games["week"] == w) & ((games["home"] == team) | (games["away"] == team))]
                if not g.empty and g.iloc[0]["kickoff"] <= now:
                    locked_now = team; used.add(team)
                    if g.iloc[0]["played"]:
                        gg = g.iloc[0]
                        won = (gg["result"] > 0 and gg["home"] == team) or (gg["result"] < 0 and gg["away"] == team)
                        results[w] = "win" if won else "loss"; losses += 0 if won else 1
                    else:
                        results[w] = "pending"
                else:
                    provisional_now = team
        alive = losses <= state.lives
        out.append(EntryStatus(entry_id=eid, picks=picks, used=used, results=results, losses=losses, alive=alive,
                               locked_now=locked_now, provisional_now=provisional_now))
    return out

"""Playoff seeding after week 17, and who has nothing to play for in week 18.

Teams rest starters in the last week when their playoff seed can no longer change (a clinched
bye above all) and, less systematically, when they are eliminated. The simulation draws every
game of every scenario, so the standings after week 17 are known scenario by scenario; this
module turns them into per-team flags:

* ``bye``     the seed is settled and it is a first-round bye;
* ``locked``  the seed is settled, in the playoffs, and there is a wild-card game to play;
* ``out``     eliminated whatever happens in week 18.

"Settled" is tested by playing week 18 ``draws`` times at random and asking whether the seed
ever moves. Ties break on division and conference record (head-to-head, the first real
tiebreaker, is not tracked), and at random after that, so a team whose seed rides on a coin
flip counts as still playing for it, which is the conservative side. Fitted on 2011-2025
closing lines the flags are worth about 10 points (bye), 7 (settled seed) and 1 (eliminated);
see ``calibration.fit_week18_rest``. Divisions have been fixed since 2002; relocated franchises
keep their division under the current codes.
"""
from __future__ import annotations

import numpy as np

from ..teams import TEAMS, TEAM_INDEX

DIVISIONS: dict[str, tuple[str, ...]] = {
    "AFC East": ("BUF", "MIA", "NE", "NYJ"), "AFC North": ("BAL", "CIN", "CLE", "PIT"),
    "AFC South": ("HOU", "IND", "JAX", "TEN"), "AFC West": ("DEN", "KC", "LAC", "LV"),
    "NFC East": ("DAL", "NYG", "PHI", "WAS"), "NFC North": ("CHI", "DET", "GB", "MIN"),
    "NFC South": ("ATL", "CAR", "NO", "TB"), "NFC West": ("ARI", "LA", "SEA", "SF"),
}
NT = len(TEAMS)
DIV_OF = np.zeros(NT, int)          # team -> division 0..7
CONF_OF = np.zeros(NT, int)         # team -> conference 0 (AFC) / 1 (NFC)
for _d, (_name, _teams) in enumerate(DIVISIONS.items()):
    for _t in _teams:
        DIV_OF[TEAM_INDEX[_t]] = _d
        CONF_OF[TEAM_INDEX[_t]] = 0 if _name.startswith("AFC") else 1
_CONF_TEAMS = [np.where(CONF_OF == c)[0] for c in (0, 1)]          # 16 team indices each
_DIV_TEAMS = [np.where(DIV_OF == d)[0] for d in range(8)]


def playoff_format(season: int) -> tuple[int, int]:
    """(playoff teams per conference, first-round byes per conference)."""
    return (7, 1) if season >= 2020 else (6, 2)


class Record:
    """Per-scenario win counts [n, 32]: overall, in conference games and in division games."""

    def __init__(self, wins: np.ndarray, conf: np.ndarray | None = None, div: np.ndarray | None = None):
        self.wins = np.asarray(wins, np.int16)
        self.conf = np.zeros_like(self.wins) if conf is None else np.asarray(conf, np.int16)
        self.div = np.zeros_like(self.wins) if div is None else np.asarray(div, np.int16)

    @staticmethod
    def from_results(wins: np.ndarray, opponent: np.ndarray) -> "Record":
        """``wins`` [n, W, 32] bool and ``opponent`` [W, 32] (index, -1 = no game)."""
        w = wins.astype(np.int16)
        opp = np.where(opponent >= 0, opponent, 0)
        conf_game = (CONF_OF[opp] == CONF_OF[None, :]) & (opponent >= 0)
        div_game = (DIV_OF[opp] == DIV_OF[None, :]) & (opponent >= 0)
        return Record(w.sum(axis=1), (w * conf_game[None]).sum(axis=1), (w * div_game[None]).sum(axis=1))

    def plus(self, other: "Record") -> "Record":
        return Record(self.wins + other.wins, self.conf + other.conf, self.div + other.div)


def seeds(rec: Record, rng: np.random.Generator) -> np.ndarray:
    """Conference seed 1..16 for every team: division winners take seeds 1-4 by record, the rest
    follow by record. Ties break on division record then conference record inside a division,
    conference record then division record across the conference, and at random after that (the
    real tiebreakers start with head-to-head, which is not tracked)."""
    n = rec.wins.shape[0]
    noise = rng.random((n, NT)) * 1e-4
    key_div = rec.wins + 0.05 * rec.div + 0.002 * rec.conf + noise
    key_conf = rec.wins + 0.05 * rec.conf + 0.002 * rec.div + noise
    divwin = np.zeros((n, NT), bool)
    for members in _DIV_TEAMS:
        best = members[np.argmax(key_div[:, members], axis=1)]
        divwin[np.arange(n), best] = True
    order_key = key_conf + 100.0 * divwin
    out = np.zeros((n, NT), np.int8)
    for members in _CONF_TEAMS:
        rank = np.argsort(np.argsort(-order_key[:, members], axis=1), axis=1)     # 0 = top
        out[:, members] = rank + 1
    return out


def week18_flags(rec17: Record, pairs: list[tuple[int, int]], season: int, draws: int = 16,
                 rng: np.random.Generator | None = None) -> dict[str, np.ndarray]:
    """Flags [n, 32] from the records after week 17 and the week-18 pairings ``pairs`` (home
    index, away index). Week-18 games are division games, so a win counts for all three records."""
    rng = rng or np.random.default_rng(0)
    n = rec17.wins.shape[0]
    n_playoff, byes = playoff_format(season)
    if not pairs:
        z = np.zeros((n, NT), bool)
        return {"bye": z, "locked": z.copy(), "out": z.copy()}
    home = np.array([h for h, _ in pairs]); away = np.array([a for _, a in pairs])
    conf_game = CONF_OF[home] == CONF_OF[away]
    div_game = DIV_OF[home] == DIV_OF[away]
    first = None
    same = np.ones((n, NT), bool)
    in_playoffs = np.ones((n, NT), bool)
    in_byes = np.ones((n, NT), bool)
    eliminated = np.ones((n, NT), bool)
    for _ in range(max(int(draws), 2)):
        u = rng.random((n, len(pairs))) < 0.5
        w = rec17.wins.copy(); c = rec17.conf.copy(); d = rec17.div.copy()
        w[:, home] += u; w[:, away] += ~u
        c[:, home] += u & conf_game; c[:, away] += ~u & conf_game
        d[:, home] += u & div_game; d[:, away] += ~u & div_game
        s = seeds(Record(w, c, d), rng)
        if first is None:
            first = s
        else:
            same &= s == first
        in_playoffs &= s <= n_playoff
        in_byes &= s <= byes
        eliminated &= s > n_playoff
    return {"bye": same & in_byes, "locked": same & in_playoffs & ~in_byes, "out": eliminated}


def rest_penalty(flags: dict[str, np.ndarray], penalties: dict[str, float]) -> np.ndarray:
    """Points docked from each team in week 18, per scenario [n, 32]."""
    out = np.zeros(flags["bye"].shape, np.float64)
    for k, v in penalties.items():
        if k in flags:
            out += float(v) * flags[k]
    return out

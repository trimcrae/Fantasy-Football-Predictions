"""Monte Carlo seasons on common random numbers.

``simulate_season`` draws, for every scenario, the line each game closes at (today's line
plus per-team drift that grows with the horizon) and the outcome of every game, and keeps
both: the per-scenario closing probabilities are what a pool player would see at kickoff
that week. That lets a plan be valued the way it is actually played: this week's pick is a
commitment, later weeks are re-picked from whatever is best available at the time
(``policy_alive``). Spreads widen as the season goes on because the drift is calibrated to
how far closing lines move from an early projection, so the later menu is richer than
anything that can be named today.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import norm

from ..model.projection import Projection
from ..teams import TEAMS

NT = len(TEAMS)


@dataclass
class Sim:
    wins: np.ndarray        # [n, weeks, 32] bool: the team wins its game that week
    probs: np.ndarray       # [n, weeks, 32] float32: its win probability at that scenario's closing line (0 = no game)
    has_game: np.ndarray    # [weeks, 32]
    weeks: list[int]

    @property
    def n(self) -> int:
        return int(self.wins.shape[0])


def simulate_season(proj: Projection, cfg: dict, n: int | None = None, seed: int | None = None,
                    discount: float | None = None) -> Sim:
    """``discount`` multiplies the calibrated drift variance (1 = calibrated)."""
    sim = cfg["simulation"]
    n = int(n or sim["scenarios"])
    rng = np.random.default_rng(sim["seed"] if seed is None else seed)
    m = cfg["model"]
    disc = float(sim.get("discount", 1.0) if discount is None else discount)
    a, b, c = float(m["horizon_var_a"]), float(m["horizon_var_b"]), float(m.get("horizon_var_c", 0.0))
    la, lb = float(m.get("posted_line_var_a", 1.0)), float(m.get("posted_line_var_b", 1.0))
    k = proj.current_week
    v_est = 0.5 * disc * max(a + c / (1.0 + k), 0.0)       # per-team persistent estimation error
    v_step = 0.5 * disc * max(b, 0.0)                        # per-team per-week drift
    nW = len(proj.weeks)
    sigma = proj.sigma

    z_est = rng.standard_normal((n, NT)) * np.sqrt(v_est)
    steps = rng.standard_normal((n, nW, NT)) * np.sqrt(v_step)
    drift = z_est[:, None, :] + np.cumsum(steps, axis=1)      # drift at horizon h uses h steps
    drift[:, 0, :] = z_est                                   # h = 0: estimation error only

    wins = np.zeros((n, nW, NT), dtype=bool)
    probs = np.zeros((n, nW, NT), dtype=np.float32)
    tab = proj.table
    home_rows = tab[tab["home"]]
    plan_disc = float(m.get("future_discount", 1.0))
    for r in home_rows.itertuples(index=False):
        wi = proj.weeks.index(int(r.week)); h = wi
        ih, ia = TEAMS.index(r.team), TEAMS.index(r.opp)
        if r.played:
            hw = proj.prob[wi, ih] >= 0.5
            aw = proj.prob[wi, ia] >= 0.5                     # both False after a tie
            wins[:, wi, ih] = hw; wins[:, wi, ia] = aw
            probs[:, wi, ih] = float(hw); probs[:, wi, ia] = float(aw)
            continue
        if r.line_var <= 0:
            p = np.full(n, float(r.prob), dtype=np.float32)
        else:
            A = max(a + b * h + c / (1.0 + k), 0.0)                 # rating-projection variance at this horizon
            if str(r.source).startswith("posted"):
                # a posted line moves far less than a projection: shrink the team drift so this
                # game's variance is the posted-line variance, keeping the cross-week correlation
                L = max(la + lb * h, 0.0)
                scale = np.sqrt(L / A) if A > 0 else 0.0
                extra = max(float(r.line_var) - plan_disc * L, 0.0)      # week-18 noise
            else:
                scale = 1.0
                extra = max(float(r.line_var) - plan_disc * A, 0.0)      # week-18 noise
            game_noise = rng.standard_normal(n) * np.sqrt(extra) if extra > 1e-9 else 0.0
            line = float(r.spread) + scale * (drift[:, h, ih] - drift[:, h, ia]) + game_noise
            p = norm.cdf(line / sigma).astype(np.float32)
        u = rng.random(n)
        hw = u < p
        wins[:, wi, ih] = hw
        wins[:, wi, ia] = ~hw
        probs[:, wi, ih] = p
        probs[:, wi, ia] = 1.0 - p
    return Sim(wins=wins, probs=probs, has_game=proj.has_game.copy(), weeks=list(proj.weeks))


def simulate_wins(proj: Projection, cfg: dict, n: int | None = None, seed: int | None = None,
                  discount: float | None = None) -> np.ndarray:
    """Boolean array [n, n_weeks, 32]: does the team win its game that week in scenario s."""
    return simulate_season(proj, cfg, n=n, seed=seed, discount=discount).wins


def path_alive(wins: np.ndarray, teams: list[int], strikes: int = 0, prior_losses: int = 0) -> np.ndarray:
    """Per scenario: does an entry following `teams` (row order) finish the season alive."""
    rows = np.arange(len(teams))
    picked = wins[:, rows, teams]
    losses = (~picked).sum(axis=1) + prior_losses
    return losses <= strikes


def rank_table(n_entries: int, n_weeks: int, weights=(0.6, 0.3, 0.1), seed: int = 0) -> np.ndarray:
    """Which rank (0 = best available) each entry takes in each later week: [n_entries, n_weeks].

    A pool of entries is re-split every week, so its later weeks are modelled as a fixed
    pattern: entry 0 always takes the best team still available to it and every other entry
    takes rank 0, 1, 2, ... with the given frequencies, drawn once so that no two entries share
    a pattern. This is what charges concentrating many entries on one team now with the
    thinner shared menu they face later, without pretending they would all pick alike.
    """
    w = np.asarray(weights, float); w = w / w.sum()
    rng = np.random.default_rng(seed)
    rows = [np.zeros(n_weeks, np.int8)]
    seen = {rows[0].tobytes()}
    tries = 0
    while len(rows) < n_entries and tries < 100 * n_entries:
        tries += 1
        r = rng.choice(len(w), size=n_weeks, p=w).astype(np.int8)
        if r.tobytes() in seen:
            continue
        seen.add(r.tobytes()); rows.append(r)
    while len(rows) < n_entries:                      # degenerate weights (e.g. [1]): identical rows
        rows.append(np.zeros(n_weeks, np.int8))
    return np.vstack(rows)


def policy_run(sim: Sim, prefix: list[int], available: np.ndarray, ranks: list[int] | None = None):
    """Play ``prefix`` (one team per week from this week) and afterwards, every remaining week,
    the best team still available at that scenario's closing line (``ranks[i]``, 0 = best, is
    the rank taken in the i-th later week). Returns per scenario and week: the team picked,
    the probability it wins at that scenario's line, and whether it won in the common draw."""
    n, nW, _ = sim.wins.shape
    ar = np.arange(n)
    used = np.broadcast_to(~np.asarray(available, bool), (n, NT)).copy()
    picks = np.zeros((n, nW), dtype=np.int8)
    pwin = np.zeros((n, nW), dtype=np.float32)
    won = np.zeros((n, nW), dtype=bool)
    for wi, t in enumerate(prefix):
        picks[:, wi] = t; pwin[:, wi] = sim.probs[:, wi, t]; won[:, wi] = sim.wins[:, wi, t]
        used[:, t] = True
    ranks = list(ranks or [])
    for i, wi in enumerate(range(len(prefix), nW)):
        pw = sim.probs[:, wi, :] * ~used                     # 0 for used teams and teams without a game
        r = ranks[i] if i < len(ranks) else 0
        if r > 0:
            ch = np.argsort(-pw, axis=1)[:, r]
            top = pw.argmax(axis=1)
            ch = np.where(pw[ar, ch] > 0, ch, top)           # fewer than r+1 teams left: take the best
        else:
            ch = pw.argmax(axis=1)
        picks[:, wi] = ch; pwin[:, wi] = pw[ar, ch]; won[:, wi] = sim.wins[ar, wi, ch]
        used[ar, ch] = True
    return picks, pwin, won


def survival_given_lines(pwin: np.ndarray, strikes: int = 0, prior_losses: int = 0) -> np.ndarray:
    """Per scenario: P(at most ``strikes`` losses) given that scenario's lines, exactly.

    Games are independent once their lines are known, so this needs no coin flips: it is the
    product of the picks' probabilities (single elimination) or the tail of their loss count.
    Averaging it over scenarios estimates season survival with far less noise than counting
    simulated survivors, which is what lets close options be told apart."""
    if prior_losses > strikes:
        return np.zeros(pwin.shape[0])
    if strikes == 0:
        return pwin.astype(np.float64).prod(axis=1)
    n, nW = pwin.shape
    dist = np.zeros((n, strikes + 2)); dist[:, prior_losses] = 1.0
    for wi in range(nW):
        p = pwin[:, wi].astype(np.float64)[:, None]
        new = dist * p
        new[:, 1:] += dist[:, :-1] * (1.0 - p)
        new[:, -1] += dist[:, -1] * (1.0 - p[:, 0])          # already out stays out
        dist = new
    return dist[:, :strikes + 1].sum(axis=1)


def policy_alive(sim: Sim, prefix: list[int], available: np.ndarray, strikes: int = 0, prior_losses: int = 0,
                 ranks: list[int] | None = None, return_picks: bool = False):
    """Survival per scenario, in the common draw, of an entry that plays ``prefix`` and then the
    best available team every later week (see ``policy_run``)."""
    picks, _, won = policy_run(sim, prefix, available, ranks)
    alive = (~won).sum(axis=1) + prior_losses <= strikes
    return (alive, picks) if return_picks else alive

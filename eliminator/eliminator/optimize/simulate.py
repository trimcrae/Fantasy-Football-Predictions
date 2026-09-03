"""Monte Carlo seasons that respect *why* future picks are uncertain.

Each scenario draws a season-long random walk for every team's true strength, so a team
that turns out better (or loses its QB) is stronger or weaker in *all* of its future games.
The line for a future game is projection + drift; the outcome is then drawn around that
line with the game sigma. Averaged over scenarios this reproduces the projection's
probabilities exactly (a Gaussian shift convolved with the normal CDF), but the joint
outcomes across weeks are correlated, which is what the multi-entry objective needs.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import norm

from ..model.projection import Projection
from ..teams import TEAMS

NT = len(TEAMS)


def simulate_wins(proj: Projection, cfg: dict, n: int | None = None, seed: int | None = None,
                  discount: float | None = None) -> np.ndarray:
    """Boolean array [n, n_weeks, 32]: does the team win its game that week in scenario s.

    ``discount`` multiplies the calibrated drift variance. The planner selects plans on a
    heavily discounted view (model.future_discount, tuned by backtest) and reports survival
    odds on the calibrated view (simulation.discount = 1).
    """
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
    tab = proj.table
    home_rows = tab[tab["home"]]
    for r in home_rows.itertuples(index=False):
        wi = proj.weeks.index(int(r.week)); h = wi
        ih, ia = TEAMS.index(r.team), TEAMS.index(r.opp)
        if r.played:
            wins[:, wi, ih] = proj.prob[wi, ih] >= 0.5
            wins[:, wi, ia] = proj.prob[wi, ia] >= 0.5      # both False after a tie
            continue
        if r.line_var <= 0:
            p = np.full(n, float(r.prob))
        else:
            plan_disc = float(m.get("future_discount", 1.0))
            A = max(a + b * h + c / (1.0 + k), 0.0)                 # rating-projection variance at this horizon
            if str(r.source).startswith("posted"):
                # a posted line moves far less than a projection: shrink the team drift so this
                # game's variance is the posted-line variance, keeping the cross-week correlation
                L = max(la + lb * h, 0.0)
                scale = np.sqrt(L / A) if A > 0 else 0.0
                extra = 0.0
            else:
                scale = 1.0
                extra = max(float(r.line_var) - plan_disc * A, 0.0)      # week-18 noise
            game_noise = rng.standard_normal(n) * np.sqrt(extra) if extra > 1e-9 else 0.0
            line = float(r.spread) + scale * (drift[:, h, ih] - drift[:, h, ia]) + game_noise
            p = norm.cdf(line / sigma)
        u = rng.random(n)
        hw = u < p
        wins[:, wi, ih] = hw
        wins[:, wi, ia] = ~hw
    return wins


def path_alive(wins: np.ndarray, teams: list[int], strikes: int = 0, prior_losses: int = 0) -> np.ndarray:
    """Per scenario: does an entry following `teams` (row order) finish the season alive."""
    rows = np.arange(len(teams))
    picked = wins[:, rows, teams]
    losses = (~picked).sum(axis=1) + prior_losses
    return losses <= strikes

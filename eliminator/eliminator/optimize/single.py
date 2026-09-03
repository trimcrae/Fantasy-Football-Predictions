"""Season paths for one entry.

A path assigns a distinct team to every remaining week. With independent games the
survival probability of a single-elimination entry is the product of its picks' win
probabilities, so the best path is a minimum-cost assignment on -log p (Hungarian
algorithm, exact). For a k-strike entry the objective is P(at most k losses), which is
optimised by local search (replace / swap moves) from the max-product path with a few
perturbed restarts.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import linear_sum_assignment

BIG = 1e6


@dataclass
class Path:
    teams: list[int]                 # team index per planning week (row order of P)
    probs: list[float]
    value: float                     # objective: P(survive) under the plug-in model
    log_product: float = 0.0
    detail: dict = field(default_factory=dict)


def cost_matrix(P: np.ndarray) -> np.ndarray:
    with np.errstate(divide="ignore"):
        C = -np.log(np.clip(P, 1e-12, 1.0))
    C[P <= 0] = BIG
    return C


def survival_prob(ps, strikes: int) -> float:
    """P(at most `strikes` losses) for independent picks with win probabilities ps."""
    dist = np.zeros(strikes + 2)
    dist[0] = 1.0
    for p in ps:
        q = 1.0 - p
        new = dist * p
        new[1:] += dist[:-1] * q
        new[-1] += dist[-1] * q          # overflow bucket: already eliminated stays eliminated
        dist = new
    return float(dist[:strikes + 1].sum())


def _apply_constraints(C: np.ndarray, available: np.ndarray, fixed: dict[int, int] | None,
                       pickable_now: np.ndarray | None = None) -> np.ndarray:
    C = C.copy()
    C[:, ~available] = BIG
    if pickable_now is not None:
        C[0, ~pickable_now] = BIG          # games already kicked off cannot be a new pick
    for wi, ti in (fixed or {}).items():
        keep = C[wi, ti]
        C[wi, :] = BIG
        C[:, ti] = BIG
        C[wi, ti] = 0.0 if keep >= BIG else keep  # a locked pick stays even if it is a dog
    return C


def best_path(P: np.ndarray, available: np.ndarray, fixed: dict[int, int] | None = None,
              extra_cost: np.ndarray | None = None, pickable_now: np.ndarray | None = None) -> Path | None:
    """Exact max-product path (single elimination). Returns None if infeasible."""
    C = cost_matrix(P)
    if extra_cost is not None:
        C = C + extra_cost
    C = _apply_constraints(C, available, fixed, pickable_now)
    rows, cols = linear_sum_assignment(C)
    teams = [int(c) for _, c in sorted(zip(rows, cols))]
    if any(C[r, c] >= BIG for r, c in zip(rows, cols)):
        return None
    probs = [float(P[w, t]) for w, t in enumerate(teams)]
    lp = float(np.sum(np.log(np.clip(probs, 1e-12, 1))))
    return Path(teams=teams, probs=probs, value=float(np.exp(lp)), log_product=lp)


def _path_value(P: np.ndarray, teams: list[int], strikes: int) -> float:
    ps = [P[w, t] for w, t in enumerate(teams)]
    if strikes == 0:
        return float(np.prod(ps))
    return survival_prob(ps, strikes)


def _improve(P: np.ndarray, teams: list[int], available: np.ndarray, fixed: dict[int, int], strikes: int,
             pickable_now: np.ndarray | None = None) -> list[int]:
    nW, nT = P.shape
    teams = list(teams)
    best = _path_value(P, teams, strikes)
    improved = True
    while improved:
        improved = False
        used = set(teams)
        for wi in range(nW):
            if wi in fixed:
                continue
            for t in range(nT):
                if t in used or not available[t] or P[wi, t] <= 0:
                    continue
                if wi == 0 and pickable_now is not None and not pickable_now[t]:
                    continue
                cand = teams.copy(); cand[wi] = t
                v = _path_value(P, cand, strikes)
                if v > best + 1e-12:
                    best, teams, used, improved = v, cand, set(cand), True
        for wi in range(nW):
            for wj in range(wi + 1, nW):
                if wi in fixed or wj in fixed:
                    continue
                ti, tj = teams[wi], teams[wj]
                if P[wi, tj] <= 0 or P[wj, ti] <= 0:
                    continue
                if wi == 0 and pickable_now is not None and not pickable_now[tj]:
                    continue
                cand = teams.copy(); cand[wi], cand[wj] = tj, ti
                v = _path_value(P, cand, strikes)
                if v > best + 1e-12:
                    best, teams, improved = v, cand, True
    return teams


def best_path_strikes(P: np.ndarray, available: np.ndarray, strikes: int, fixed: dict[int, int] | None = None,
                      restarts: int = 6, rng: np.random.Generator | None = None,
                      pickable_now: np.ndarray | None = None) -> Path | None:
    """Path maximising P(at most `strikes` losses). Exact for strikes=0, local search otherwise."""
    fixed = fixed or {}
    base = best_path(P, available, fixed, pickable_now=pickable_now)
    if base is None:
        return None
    if strikes == 0:
        return base
    rng = rng or np.random.default_rng(0)
    best_teams = _improve(P, base.teams, available, fixed, strikes, pickable_now)
    best_v = _path_value(P, best_teams, strikes)
    for _ in range(restarts):
        noise = rng.gumbel(0.0, 0.15, size=P.shape)
        start = best_path(P, available, fixed, extra_cost=noise, pickable_now=pickable_now)
        if start is None:
            continue
        cand = _improve(P, start.teams, available, fixed, strikes, pickable_now)
        v = _path_value(P, cand, strikes)
        if v > best_v + 1e-12:
            best_v, best_teams = v, cand
    probs = [float(P[w, t]) for w, t in enumerate(best_teams)]
    lp = float(np.sum(np.log(np.clip(probs, 1e-12, 1))))
    return Path(teams=best_teams, probs=probs, value=best_v, log_product=lp)


def current_week_options(P: np.ndarray, available: np.ndarray, pickable_now: np.ndarray, strikes: int = 0,
                         fixed: dict[int, int] | None = None, min_prob: float = 0.0,
                         rng: np.random.Generator | None = None) -> list[Path]:
    """Value of the season if each candidate is used *this* week (row 0) and the rest is optimal."""
    fixed = dict(fixed or {})
    out = []
    if 0 in fixed:   # this week's pick is locked: only one option exists
        path = best_path_strikes(P, available, strikes, fixed, restarts=2, rng=rng, pickable_now=pickable_now) if strikes \
            else best_path(P, available, fixed, pickable_now=pickable_now)
        if path is not None:
            path.detail = {"now_prob": float(P[0, fixed[0]])}
            out.append(path)
        return out
    for t in range(P.shape[1]):
        if not (available[t] and pickable_now[t]) or P[0, t] <= max(min_prob, 0.0):
            continue
        f = dict(fixed); f[0] = t
        path = best_path_strikes(P, available, strikes, f, restarts=2, rng=rng, pickable_now=pickable_now) if strikes \
            else best_path(P, available, f, pickable_now=pickable_now)
        if path is None:
            continue
        path.detail = {"now_prob": float(P[0, t])}
        out.append(path)
    out.sort(key=lambda p: p.value, reverse=True)
    return out


def policy_options(P: np.ndarray, sim, available: np.ndarray, pickable_now: np.ndarray, strikes: int = 0,
                   fixed: dict[int, int] | None = None, horizon: int = 1, min_prob: float = 0.0,
                   rng: np.random.Generator | None = None) -> list[Path]:
    """Every candidate for this week, valued by simulation the way the season is played.

    The candidate is used now; the next ``horizon - 1`` picks are the max-product path's;
    every later week takes the best team still available at that scenario's closing line.
    ``value`` is that survival probability. ``teams`` keeps the full max-product path as a
    sketch of the later weeks (``detail['plugin']`` is its plug-in value)."""
    from .simulate import policy_run, survival_given_lines
    fixed = dict(fixed or {})
    horizon = max(int(horizon), 1)
    if 0 in fixed:
        cands = [int(fixed[0])]
    else:
        cands = [t for t in range(P.shape[1]) if available[t] and pickable_now[t] and P[0, t] > max(min_prob, 0.0)]
    out = []
    for t in cands:
        f = dict(fixed); f[0] = int(t)
        path = best_path_strikes(P, available, strikes, f, restarts=2, rng=rng, pickable_now=pickable_now) if strikes \
            else best_path(P, available, f, pickable_now=pickable_now)
        if path is None:
            continue
        picks, pwin, won = policy_run(sim, path.teams[:horizon], available)
        surv = survival_given_lines(pwin, strikes)
        mask = (~won).sum(axis=1) <= strikes
        path.detail = {"now_prob": float(P[0, t]), "plugin": float(path.value), "sim": float(surv.mean()),
                       "mask": mask, "surv": surv, "won": won, "picks": picks, "horizon": horizon}
        path.value = float(surv.mean())
        out.append(path)
    out.sort(key=lambda p: p.value, reverse=True)
    return out

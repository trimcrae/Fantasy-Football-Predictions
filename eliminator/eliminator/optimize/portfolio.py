"""Many entries, one pool: maximise P(at least one entry survives the season).

Entries that share picks live and die together, so the value of an extra entry is its
survival in the scenarios where every other entry is already dead. The portfolio is built
greedily on common random numbers: for each entry a pool of strong, deliberately diverse
candidate paths is generated (penalised assignments, forced current-week alternatives,
perturbed assignments) and the candidate with the largest marginal gain is kept; a couple
of coordinate-ascent passes then revisit every entry.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .simulate import path_alive
from .single import Path, best_path, best_path_strikes

NT = 32


@dataclass
class EntryPlan:
    entry_id: str
    available: np.ndarray                  # teams still usable
    fixed: dict[int, int] = field(default_factory=dict)   # locked picks (row -> team)
    strikes_left: int = 0
    alive: bool = True
    path: Path | None = None
    alive_mask: np.ndarray | None = None   # per scenario
    candidates: list[Path] = field(default_factory=list)


def _candidates(P: np.ndarray, pickable_now: np.ndarray, e: EntryPlan, usage: np.ndarray,
                n_target: int, rng: np.random.Generator) -> list[Path]:
    seen: set[tuple[int, ...]] = set()
    out: list[Path] = []

    def add(p: Path | None):
        if p is None:
            return
        key = tuple(p.teams)
        if key in seen:
            return
        seen.add(key); out.append(p)

    solve = (lambda fixed, extra=None: best_path(P, e.available, fixed, extra, pickable_now=pickable_now)) if e.strikes_left == 0 else \
            (lambda fixed, extra=None: best_path_strikes(P, e.available, e.strikes_left, fixed, restarts=1, rng=rng, pickable_now=pickable_now))
    add(solve(e.fixed))
    if not out:
        return out
    top = out[0].probs[0]
    # forced alternatives for this week
    if 0 not in e.fixed:
        for t in np.argsort(-P[0]):
            if P[0, t] <= 0 or not (e.available[t] and pickable_now[t]):
                continue
            if P[0, t] < max(0.5, top - 0.3):
                break
            f = dict(e.fixed); f[0] = int(t)
            add(solve(f))
    # penalised by current usage across entries, at several strengths
    if e.strikes_left == 0:
        for lam in (0.02, 0.05, 0.1, 0.2, 0.4):
            add(best_path(P, e.available, e.fixed, extra_cost=lam * usage, pickable_now=pickable_now))
        i = 0
        while len(out) < n_target and i < 4 * n_target:
            i += 1
            scale = rng.choice([0.03, 0.08, 0.15, 0.25])
            noise = rng.gumbel(0.0, scale, size=P.shape) + rng.choice([0.0, 0.05, 0.15]) * usage
            add(best_path(P, e.available, e.fixed, extra_cost=noise, pickable_now=pickable_now))
    return out


def build_portfolio(P: np.ndarray, pickable_now: np.ndarray, entries: list[EntryPlan], wins: np.ndarray,
                    cfg: dict, objective: str = "any", rng: np.random.Generator | None = None) -> list[EntryPlan]:
    rng = rng or np.random.default_rng(cfg["simulation"]["seed"] + 1)
    n_cand = int(cfg["portfolio"]["candidates_per_slot"])
    passes = int(cfg["portfolio"]["improve_passes"])
    live = [e for e in entries if e.alive]
    nS = wins.shape[0]
    any_alive = np.zeros(nS, bool)
    usage = np.zeros(P.shape)

    def gain(mask: np.ndarray, others: np.ndarray) -> float:
        if objective == "expected":
            return float(mask.mean())
        return float(np.mean(mask & ~others))

    # greedy construction
    for e in live:
        e.candidates = _candidates(P, pickable_now, e, usage, n_cand, rng)
        best, best_g, best_mask = None, -1.0, None
        for c in e.candidates:
            mask = path_alive(wins, c.teams, e.strikes_left)
            g = gain(mask, any_alive)
            if g > best_g:
                best, best_g, best_mask = c, g, mask
        if best is None:
            continue
        e.path, e.alive_mask = best, best_mask
        any_alive |= best_mask
        for w, t in enumerate(best.teams):
            usage[w, t] += 1.0
    # coordinate ascent: re-pick each entry given all the others
    for _ in range(passes):
        changed = 0
        for e in live:
            if e.path is None:
                continue
            others = np.zeros(nS, bool)
            for o in live:
                if o is not e and o.alive_mask is not None:
                    others |= o.alive_mask
            for w, t in enumerate(e.path.teams):
                usage[w, t] -= 1.0
            cands = e.candidates + _candidates(P, pickable_now, e, usage, n_cand // 2, rng)
            best, best_g, best_mask = e.path, gain(e.alive_mask, others), e.alive_mask
            for c in cands:
                mask = path_alive(wins, c.teams, e.strikes_left)
                g = gain(mask, others)
                if g > best_g + 1e-9:
                    best, best_g, best_mask = c, g, mask
            if best is not e.path:
                changed += 1
            e.path, e.alive_mask = best, best_mask
            for w, t in enumerate(best.teams):
                usage[w, t] += 1.0
        if not changed:
            break
    return entries


def portfolio_summary(entries: list[EntryPlan], wins: np.ndarray) -> dict:
    masks = [e.alive_mask for e in entries if e.alive and e.alive_mask is not None]
    if not masks:
        return {"p_any": 0.0, "expected_survivors": 0.0, "n_live": 0}
    M = np.vstack(masks)
    return {"p_any": float(M.any(axis=0).mean()), "expected_survivors": float(M.sum(axis=0).mean()),
            "n_live": len(masks), "p_each": [float(m.mean()) for m in masks]}

"""Many entries, one pool: maximise P(at least one entry survives the season).

Entries that share picks live and die together, so the value of an extra entry is its
survival in the scenarios where every other entry is already dead. The portfolio is built
greedily on common random numbers: for each entry a pool of strong, deliberately diverse
candidate paths is generated (penalised assignments, forced current-week alternatives,
perturbed assignments) and the candidate with the largest marginal gain is kept;
coordinate-ascent passes then revisit every entry until none moves (or the pass cap is hit),
so that no single entry can be moved onto any candidate for a gain.

With a planning horizon (the default) only the first ``horizon`` picks of a candidate are a
commitment; every later week is re-picked from whatever is best available at the time
(``simulate.policy_alive``). Because a pool is re-split every week, each entry follows its
own fixed rank pattern later on (``simulate.rank_table``): most weeks the best team still
available to it, sometimes the second or third. Concentrating many entries on one team now
is therefore charged for the thinner menu those entries share later, and no two entries are
ever assumed to play identical seasons.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .simulate import Sim, path_alive, policy_run, rank_table, survival_given_lines
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
    alive_mask: np.ndarray | None = None   # per scenario, in the common draw
    candidates: list[Path] = field(default_factory=list)
    slot: int = 0                          # row of the rank table this entry follows after the horizon
    picks: np.ndarray | None = None        # policy mode, [n, weeks]: team picked per scenario and week
    won: np.ndarray | None = None          # policy mode, [n, weeks]: that pick won in the common draw
    surv: np.ndarray | None = None         # policy mode, [n]: P(survive | that scenario's lines), exact

    def p_season(self) -> float | None:
        if self.surv is not None:
            return float(self.surv.mean())
        return float(self.alive_mask.mean()) if self.alive_mask is not None else None


def _candidates(P: np.ndarray, pickable_now: np.ndarray, e: EntryPlan, usage: np.ndarray,
                n_target: int, rng: np.random.Generator, horizon: int | None = None) -> list[Path]:
    seen: set[tuple[int, ...]] = set()
    out: list[Path] = []

    def add(p: Path | None):
        if p is None:
            return
        key = tuple(p.teams[:horizon]) if horizon else tuple(p.teams)
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
    if horizon == 1:
        return out                       # only this week is a commitment: the alternatives above are the whole menu
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


class PolicyScorer:
    """Plays candidate prefixes forward; each entry follows its own later-week rank pattern."""

    def __init__(self, sim: Sim, horizon: int, n_entries: int, weights=(0.6, 0.3, 0.1), seed: int = 0):
        self.sim, self.horizon = sim, max(int(horizon), 1)
        self.ranks = rank_table(max(int(n_entries), 1), len(sim.weeks), weights, seed)
        self.cache: dict = {}

    def run(self, e: EntryPlan, path: Path):
        """(picks, surv, won) for the entry following ``path`` up to the horizon, then its pattern."""
        prefix = [int(t) for t in path.teams[:self.horizon]]
        ranks = self.ranks[min(e.slot, len(self.ranks) - 1)]
        key = (tuple(prefix), ranks.tobytes(), e.available.tobytes(), e.strikes_left)
        r = self.cache.get(key)
        if r is None:
            picks, pwin, won = policy_run(self.sim, prefix, e.available, ranks=ranks.tolist())
            r = (picks, survival_given_lines(pwin, e.strikes_left), won)
            self.cache[key] = r
        return r

    def mask(self, e: EntryPlan, path: Path) -> np.ndarray:
        _, _, won = self.run(e, path)
        return (~won).sum(axis=1) <= e.strikes_left


def dead_given(others: list[EntryPlan], picks: np.ndarray) -> np.ndarray:
    """Per scenario: every entry in ``others`` is dead, given that the games ``picks`` chose were won.

    Once the lines are known games are independent, so conditioning on a candidate's picks
    winning only changes the games it shares with another entry: same team, same week."""
    n = picks.shape[0]
    dead = np.ones(n, bool)
    for o in others:
        if o.won is None or o.picks is None:
            continue
        losses = (~(o.won | (o.picks == picks))).sum(axis=1)
        dead &= losses > o.strikes_left
    return dead


def make_scorer(sim: Sim, planning: dict, n_entries: int, seed: int) -> PolicyScorer:
    return PolicyScorer(sim, int(planning.get("horizon", 1)), n_entries, tuple(planning.get("spread_weights", (0.6, 0.3, 0.1))), int(seed))


def build_portfolio(P: np.ndarray, pickable_now: np.ndarray, entries: list[EntryPlan], wins, cfg: dict,
                    objective: str = "any", rng: np.random.Generator | None = None,
                    horizon: int | None = None) -> list[EntryPlan]:
    """``wins`` is a boolean [n, weeks, 32] array (fixed paths, no horizon) or a ``Sim`` with ``horizon``."""
    rng = rng or np.random.default_rng(cfg["simulation"]["seed"] + 1)
    n_cand = int(cfg["portfolio"]["candidates_per_slot"])
    passes = int(cfg["portfolio"]["improve_passes"])
    live = [e for e in entries if e.alive]
    for i, e in enumerate(live):
        e.slot = i
    scorer = make_scorer(wins, cfg.get("planning") or {}, len(live), int(cfg["simulation"]["seed"]) + 2) if isinstance(wins, Sim) and horizon else None
    nS = wins.n if isinstance(wins, Sim) else wins.shape[0]
    any_alive = np.zeros(nS, bool)
    usage = np.zeros(P.shape)

    def score(e: EntryPlan, c: Path):
        """(gain against the entries currently in the pool, per-scenario survival, won, picks)."""
        if scorer is None:
            mask = path_alive(wins, c.teams, e.strikes_left)
            return mask, None, None, None
        picks, surv, won = scorer.run(e, c)
        return None, surv, won, picks

    def gain(e: EntryPlan, scored, others: list[EntryPlan], others_alive: np.ndarray | None) -> float:
        mask, surv, won, picks = scored
        if scorer is None:
            return float(mask.mean()) if objective == "expected" else float(np.mean(mask & ~others_alive))
        if objective == "expected":
            return float(surv.mean())
        if e.strikes_left == 0:
            # P(this survives and every other is dead) = E[ P(survive | lines) * P(others dead | it survived, lines) ]
            return float(np.mean(surv * dead_given(others, picks)))
        alive = (~won).sum(axis=1) <= e.strikes_left
        return float(np.mean(alive & ~others_alive))

    def adopt(e: EntryPlan, c: Path, scored) -> None:
        mask, surv, won, picks = scored
        e.path = c
        if scorer is None:
            e.alive_mask = mask
            return
        e.alive_mask = (~won).sum(axis=1) <= e.strikes_left
        e.surv, e.won, e.picks = surv, won, picks
        c.detail.setdefault("plugin", float(c.value))         # keep the plug-in value of the sketch
        c.value = float(surv.mean())

    def alive_of(members: list[EntryPlan]) -> np.ndarray:
        out = np.zeros(nS, bool)
        for o in members:
            if o.alive_mask is not None:
                out |= o.alive_mask
        return out

    # greedy construction
    chosen: list[EntryPlan] = []
    for e in live:
        e.candidates = _candidates(P, pickable_now, e, usage, n_cand, rng, horizon)
        others_alive = alive_of(chosen)
        best, best_g, best_s = None, -1.0, None
        for c in e.candidates:
            sc = score(e, c)
            g = gain(e, sc, chosen, others_alive)
            if g > best_g:
                best, best_g, best_s = c, g, sc
        if best is None:
            continue
        adopt(e, best, best_s)
        chosen.append(e)
        for w, t in enumerate(best.teams):
            usage[w, t] += 1.0
    # coordinate ascent: re-pick each entry given all the others
    for _ in range(passes):
        changed = 0
        for e in live:
            if e.path is None:
                continue
            others = [o for o in live if o is not e and o.path is not None]
            others_alive = alive_of(others)
            for w, t in enumerate(e.path.teams):
                usage[w, t] -= 1.0
            cands = e.candidates + _candidates(P, pickable_now, e, usage, n_cand // 2, rng, horizon)
            cur = e.path
            cur_s = (e.alive_mask, e.surv, e.won, e.picks)
            best, best_g, best_s = cur, gain(e, cur_s, others, others_alive), cur_s
            for c in cands:
                sc = score(e, c)
                g = gain(e, sc, others, others_alive)
                if g > best_g + 1e-9:
                    best, best_g, best_s = c, g, sc
            if best is not cur:
                changed += 1
                adopt(e, best, best_s)
            for w, t in enumerate(best.teams):
                usage[w, t] += 1.0
        if not changed:
            break
    return entries


def portfolio_summary(entries: list[EntryPlan], wins=None) -> dict:
    live = [e for e in entries if e.alive and e.alive_mask is not None]
    if not live:
        return {"p_any": 0.0, "expected_survivors": 0.0, "n_live": 0}
    if all(e.surv is not None and e.strikes_left == 0 for e in live):
        # P(any) = sum over entries of P(it survives and every earlier one is dead), each term exact given the lines
        p_any = 0.0
        for i, e in enumerate(live):
            p_any += float(np.mean(e.surv * dead_given(live[:i], e.picks)))
        p_each = [float(e.surv.mean()) for e in live]
        return {"p_any": min(p_any, 1.0), "expected_survivors": float(sum(p_each)), "n_live": len(live), "p_each": p_each}
    M = np.vstack([e.alive_mask for e in live])
    return {"p_any": float(M.any(axis=0).mean()), "expected_survivors": float(M.sum(axis=0).mean()),
            "n_live": len(live), "p_each": [e.p_season() for e in live]}

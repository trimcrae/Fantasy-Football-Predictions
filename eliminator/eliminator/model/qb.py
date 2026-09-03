"""Quarterback availability: penalties, duration priors and the manual ledger.

The ledger (state/qb_status.yaml) is a list of situations::

    - team: BAL
      player: Lamar Jackson
      tier: elite              # elite | good | average | replacement, or penalty: 7.0 (points)
      status: out              # out | doubtful | questionable | ir | season
      injury: hamstring        # keyword for the duration prior (see DURATION_PRIORS)
      injured_week: 5          # first week missed (or expected to be missed)
      return_week: 8           # optional: expected first week back; beats the prior
      reflected: null          # optional 0-1: share of the absence already inside the rating
      p_out_by_week: {5: 1.0, 6: 0.8}   # optional explicit override, week -> P(out)

For each team the model derives P(starter out) for every remaining week and a points
penalty, giving the expected QB effect ``-penalty * P(out)`` used by the projection.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import yaml

# Probability the starter is still out j games after the first missed game (j=0 is that game).
DURATION_PRIORS: dict[str, list[float]] = {
    "concussion": [1.0, 0.35, 0.12, 0.05],
    "illness": [1.0, 0.15],
    "ankle": [1.0, 0.6, 0.35, 0.15, 0.07],
    "high ankle": [1.0, 0.9, 0.75, 0.55, 0.35, 0.2, 0.1],
    "hamstring": [1.0, 0.75, 0.5, 0.3, 0.15, 0.07],
    "calf": [1.0, 0.7, 0.45, 0.25, 0.1],
    "groin": [1.0, 0.65, 0.4, 0.2, 0.1],
    "quad": [1.0, 0.7, 0.45, 0.25, 0.1],
    "knee": [1.0, 0.85, 0.7, 0.55, 0.4, 0.25, 0.15, 0.08],
    "mcl": [1.0, 0.9, 0.75, 0.55, 0.35, 0.2, 0.1],
    "acl": [1.0] * 30,
    "achilles": [1.0] * 30,
    "season": [1.0] * 30,
    "shoulder": [1.0, 0.8, 0.6, 0.45, 0.3, 0.2, 0.1],
    "elbow": [1.0, 0.75, 0.55, 0.4, 0.25, 0.15],
    "thumb": [1.0, 0.85, 0.65, 0.45, 0.3, 0.15, 0.08],
    "hand": [1.0, 0.85, 0.65, 0.45, 0.3, 0.15, 0.08],
    "finger": [1.0, 0.7, 0.45, 0.25, 0.12],
    "wrist": [1.0, 0.85, 0.65, 0.45, 0.3, 0.15, 0.08],
    "ribs": [1.0, 0.55, 0.3, 0.12],
    "rib": [1.0, 0.55, 0.3, 0.12],
    "oblique": [1.0, 0.7, 0.45, 0.25, 0.1],
    "back": [1.0, 0.65, 0.45, 0.3, 0.15],
    "neck": [1.0, 0.7, 0.5, 0.35, 0.2, 0.1],
    "foot": [1.0, 0.8, 0.6, 0.45, 0.3, 0.2, 0.12],
    "toe": [1.0, 0.8, 0.6, 0.45, 0.3, 0.2, 0.12],
    "lisfranc": [1.0] * 8 + [0.7, 0.5, 0.3],
    "jones": [1.0] * 6 + [0.7, 0.5, 0.3],
    "collarbone": [1.0] * 5 + [0.8, 0.5, 0.3, 0.15],
    "clavicle": [1.0] * 5 + [0.8, 0.5, 0.3, 0.15],
    "ir": [1.0] * 4 + [0.6, 0.45, 0.35, 0.25, 0.2, 0.15],
    "unknown": [1.0, 0.7, 0.5, 0.35, 0.25, 0.15, 0.1],
}
# Current-week probability of missing the game by designation, then the prior takes over.
STATUS_THIS_WEEK = {"out": 1.0, "ir": 1.0, "season": 1.0, "doubtful": 0.85, "questionable": 0.4,
                    "probable": 0.1, "healthy": 0.0, "active": 0.0}


@dataclass
class QBSituation:
    team: str
    player: str = ""
    penalty: float = 3.5
    status: str = "out"
    injury: str = "unknown"
    injured_week: int | None = None
    return_week: int | None = None
    reflected: float | None = None
    p_out_by_week: dict[int, float] = field(default_factory=dict)
    note: str = ""


def _prior_for(injury: str, status: str) -> list[float]:
    key = (injury or "").lower()
    if status in ("ir",) and "ir" not in key:
        return DURATION_PRIORS["ir"]
    if status in ("season",):
        return DURATION_PRIORS["season"]
    # longest matching keyword wins ("high ankle" before "ankle")
    best = None
    for k in DURATION_PRIORS:
        if k in key and (best is None or len(k) > len(best)):
            best = k
    return DURATION_PRIORS[best or "unknown"]


def load_ledger(path: Path, cfg_qb: dict) -> list[QBSituation]:
    if not path.exists():
        return []
    raw = yaml.safe_load(path.read_text()) or []
    tiers = cfg_qb["penalty_by_tier"]
    out = []
    for rec in raw:
        if not rec or "team" not in rec:
            continue
        from ..teams import normalize
        pen = rec.get("penalty")
        if pen is None:
            pen = tiers.get(str(rec.get("tier", cfg_qb["default_tier"])).lower(), tiers[cfg_qb["default_tier"]])
        p_by = {int(k): float(v) for k, v in (rec.get("p_out_by_week") or {}).items()}
        out.append(QBSituation(
            team=normalize(rec["team"]), player=str(rec.get("player", "")), penalty=float(pen),
            status=str(rec.get("status", "out")).lower(), injury=str(rec.get("injury", "unknown")),
            injured_week=rec.get("injured_week"), return_week=rec.get("return_week"),
            reflected=rec.get("reflected"), p_out_by_week=p_by, note=str(rec.get("note", "")),
        ))
    return out


AUTO_STATUSES = {"out": "out", "doubtful": "doubtful", "questionable": "questionable",
                 "injured reserve": "ir", "ir": "ir", "pup": "ir", "nfi": "ir"}


def load_auto(path: Path, cfg_qb: dict) -> list[QBSituation]:
    """Situations the pipeline added itself (state/qb_auto.yaml); same format as the ledger."""
    return load_ledger(path, cfg_qb)


def save_auto(sits: list[QBSituation], path: Path) -> Path:
    rows = [{"team": s.team, "player": s.player, "penalty": float(s.penalty), "status": s.status,
             "injury": s.injury, "injured_week": s.injured_week, "note": s.note} for s in sits]
    header = ("# Written by `eliminator snapshot` from the nflverse injury report: starters with a game\n"
              "# designation and no entry in qb_status.yaml. Do not edit; put your own view of a QB in\n"
              "# qb_status.yaml and it replaces the automatic entry for that team.\n")
    path.write_text(header + yaml.safe_dump(rows, sort_keys=False))
    return path


def update_auto(previous: list[QBSituation], watch, manual: list[QBSituation], current_week: int,
                cfg_qb: dict, report_published: bool) -> list[QBSituation]:
    """Roll the automatic situations forward one run.

    * a QB1 on this week's report with Out / Doubtful / Questionable and no manual entry for
      the team gets a new situation starting this week, at the default tier;
    * an automatic situation carries over while its player is still on the report with a
      designation (even after the depth chart promotes the backup), keeping its first missed
      week so the duration prior keeps counting;
    * it is dropped when the player is off the report (healthy, or a long absence the ratings
      have absorbed), unless no report for the week has been published yet;
    * a manual entry for the team always wins.
    """
    manual_teams = {s.team for s in manual}
    rows = [] if watch is None or watch.empty else watch.to_dict(orient="records")
    on_report = {}
    for r in rows:
        st = AUTO_STATUSES.get(str(r.get("status", "")).strip().lower())
        if st:
            on_report[(r["team"], str(r.get("player", "")))] = (st, str(r.get("injury") or "unknown").lower(), bool(r.get("is_qb1")))
    out: list[QBSituation] = []
    seen = set()
    for s in previous:
        if s.team in manual_teams:
            continue
        key = (s.team, s.player)
        if key in on_report:
            st, inj, _ = on_report[key]
            out.append(QBSituation(team=s.team, player=s.player, penalty=s.penalty, status=st, injury=inj or s.injury,
                                   injured_week=s.injured_week, note=s.note))
            seen.add(key)
        elif not report_published:
            out.append(s); seen.add(key)
    tier = cfg_qb["default_tier"]
    pen = float(cfg_qb["penalty_by_tier"][tier])
    for (team, player), (st, inj, is_qb1) in on_report.items():
        if (team, player) in seen or team in manual_teams or not is_qb1 or team in {o.team for o in out}:
            continue
        out.append(QBSituation(team=team, player=player, penalty=pen, status=st, injury=inj,
                               injured_week=current_week, note=f"auto: nflverse injury report, {tier} tier"))
    return sorted(out, key=lambda s: s.team)


def p_out_by_week(sit: QBSituation, current_week: int, season_weeks: int, cfg_qb: dict,
                  game_weeks: list[int] | None = None) -> np.ndarray:
    """P(starter unavailable) for weeks 1..season_weeks (index 0 unused)."""
    p = np.zeros(season_weeks + 1)
    start = int(sit.injured_week or current_week)
    if sit.p_out_by_week:
        for w, v in sit.p_out_by_week.items():
            if 1 <= w <= season_weeks:
                p[w] = v
        return p
    if sit.return_week is not None:
        setback = list(cfg_qb["return_week_setback"])
        rw = int(sit.return_week)
        for w in range(start, season_weeks + 1):
            if w < rw:
                p[w] = 1.0
            else:
                j = w - rw + 1  # 1 = return week
                p[w] = setback[j] if j < len(setback) else 0.0
        # A questionable/doubtful tag on the current week softens the first week.
        if start == current_week and sit.status in ("questionable", "doubtful", "probable"):
            p[start] = STATUS_THIS_WEEK[sit.status]
        return p
    prior = _prior_for(sit.injury, sit.status)
    this_week = STATUS_THIS_WEEK.get(sit.status, 1.0)
    weeks = [w for w in range(start, season_weeks + 1) if game_weeks is None or w in game_weeks or w == start]
    for j, w in enumerate(weeks):
        base = prior[j] if j < len(prior) else 0.0
        if j == 0:
            p[w] = this_week
        else:
            # conditional on missing the first game; if the first game was only a coin flip,
            # the tail is scaled down accordingly
            p[w] = base * (this_week if this_week < 1.0 else 1.0)
    # bye weeks in between do not count as missed games but the calendar still moves on;
    # game_weeks handles that by skipping them.
    if game_weeks is not None:
        for w in range(start, season_weeks + 1):
            if w not in game_weeks:
                p[w] = 0.0
    return p


def reflected_fraction(sit: QBSituation, current_week: int, game_weeks: list[int], cfg_qb: dict) -> float:
    """Share of the QB absence already priced into a blended rating such as inpredictable's."""
    if sit.reflected is not None:
        return float(np.clip(sit.reflected, 0.0, 1.0))
    if sit.injured_week is None:
        return 0.0
    missed = sum(1 for w in game_weeks if sit.injured_week <= w < current_week)
    hl = float(cfg_qb.get("reflect_half_life_games", 2.0))
    return float(1.0 - 0.5 ** (missed / hl)) if missed > 0 else 0.0


def ledger_summary(sits: list[QBSituation], current_week: int, season_weeks: int, cfg_qb: dict) -> str:
    lines = []
    for s in sits:
        p = p_out_by_week(s, current_week, season_weeks, cfg_qb)
        weeks = " ".join(f"w{w}:{p[w]:.2f}" for w in range(current_week, season_weeks + 1) if p[w] > 0.01)
        lines.append(f"{s.team:<4}{s.player:<20}{s.status:<12}pen {s.penalty:>4.1f}  {weeks or 'available'}")
    return "\n".join(lines)

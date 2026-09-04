"""Configuration: code defaults < calibration.json (written by `calibrate`) < config.yaml."""
from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]

DEFAULTS: dict[str, Any] = {
    "season": None,                 # None -> latest season in the schedule file
    "season_weeks": 18,
    "model": {
        "sigma": 13.45,             # sd of margin around the spread
        "hfa": 1.6,                 # home-field advantage, points (neutral sites get 0)
        "rest_per_day": 0.10,       # points per day of rest differential (clipped to +-7 days)
        "prior_regression": 0.55,   # preseason rating = this * last season's final rating
        "prior_weight": 6.0,        # pseudo-games of weight on the preseason prior
        "recency_half_life": 5.0,   # weeks; weight of a posted line decays with age
        "horizon_var_a": 4.0,       # projection error variance vs closing line at h=0 (points^2)
        "horizon_var_b": 2.0,       # additional variance per week of horizon
        "future_discount": 16.0,    # planning.mode = discount only: multiplier on horizon variance when *choosing* plans
        "posted_line_var_a": 1.0,   # variance (points^2) of a posted line vs its close, next week
        "posted_line_var_b": 1.0,   # additional variance per week of horizon: how far a line can move
        "week18_shrink": 0.6,       # week-18 probabilities shrunk toward 0.5 (starters rest)
        "week18_extra_var": 20.0,   # extra spread variance for week 18 projections
        "ratings_source": "auto",   # auto | inpredictable | market | blend
        "inpredictable_weight": 0.5,  # used when ratings_source == blend
        "inpredictable_max_rmse": 2.0,  # auto: reject inpredictable when it differs from the market fit by more (points rms)
        "min_pick_prob": 0.0,       # never pick below this probability (0 = no floor)
    },
    "qb": {
        "penalty_by_tier": {"elite": 7.0, "good": 5.0, "average": 3.5, "replacement": 1.5},
        "default_tier": "average",
        "return_week_setback": [1.0, 0.35, 0.15, 0.05],  # P(out) in weeks return-1, return, +1, +2
        "reflect_half_life_games": 2.0,  # how fast a rating absorbs a QB absence
        "auto_from_injuries": True,      # add starters with a game designation to the ledger automatically
    },
    "simulation": {"scenarios": 20000, "seed": 7, "discount": 1.0},  # discount: drift multiplier, 1 = calibrated
    "portfolio": {"candidates_per_slot": 60, "improve_passes": 6,   # coordinate ascent stops early once no entry moves
                  "allocation_view": "planning"},   # planning.mode = discount only: planning (discounted) | calibrated scores the split
    # How the future is valued when choosing this week's pick.
    #   policy   - this week (and the next horizon-1 weeks) are a commitment; every later week is
    #              re-picked from the best team available at that scenario's closing line. A pool of
    #              entries is re-split every week: spread_weights are the frequencies with which an
    #              entry takes the best, second-best, third-best available team in a later week.
    #   discount - fixed 18-week paths scored on a simulation with model.future_discount x the drift.
    "planning": {"mode": "policy", "horizon": 1, "spread_weights": [0.6, 0.3, 0.1]},
    "data": {"max_age_hours": 6.0, "odds_api_key": None, "odds_api_region": "us"},
}


def _deep_merge(base: dict, over: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in (over or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def load_config(root: Path | None = None) -> dict[str, Any]:
    root = root or ROOT
    cfg = copy.deepcopy(DEFAULTS)
    calib = root / "calibration.json"
    if calib.exists():
        cfg = _deep_merge(cfg, {"model": json.loads(calib.read_text()).get("model", {})})
    user = root / "config.yaml"
    if user.exists():
        cfg = _deep_merge(cfg, yaml.safe_load(user.read_text()) or {})
    return cfg

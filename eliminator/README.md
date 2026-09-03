# NFL eliminator / survivor strategy

Season-long pick optimisation for two pool formats:

| pool | file | objective |
|---|---|---|
| single elimination, 25 entries | `state/multi25.yaml` | maximise **P(at least one entry survives the whole season)** |
| one entry, two strikes | `state/strikes2.yaml` | maximise **P(the entry survives the whole season)** (out on the second loss) |

Everything is driven by the betting market. Where a Vegas price exists it is the truth;
where it does not, team strength is backed out of the market (inpredictable's GPF, or the
same thing fitted from posted spreads), projected forward with home field, rest and QB
availability, and discounted for the uncertainty of looking weeks ahead.

This directory is self-contained; nothing else in the repository is used.

## Quick start

```bash
cd eliminator
pip install -r requirements.txt

python -m eliminator update                       # schedule + lines, inpredictable, injury report
python -m eliminator plan --pool state/multi25.yaml
python -m eliminator plan --pool state/strikes2.yaml
python -m eliminator plan --pool state/multi25.yaml --commit   # write this week's picks to the pool file
python -m eliminator status --pool state/multi25.yaml         # results and who is still alive
```

From anywhere else use the launcher: `python eliminator/run.py plan --pool eliminator/state/multi25.yaml`.

Re-run `plan` as often as you like before kickoff. Picks whose game has started are locked
(and their teams burned); everything else is re-optimised against the latest lines. On a
Sunday morning after Thursday night football, entries that used the Thursday team keep it and
the rest are re-planned around what happened. `--now "2026-09-13 12:00"` pretends it is
another time (ET), which is handy for checking what locks.

Other commands: `calibrate` (refit all model parameters from history, writes
`calibration.json`), `backtest` (replay past seasons with as-of information),
`qb` (print the week's QB injury report), `record` (enter a pick by hand).

## Picks site (GitHub Pages)

`.github/workflows/eliminator-pages.yml` publishes the recommendations for every pool format
to GitHub Pages at `https://<user>.github.io/Fantasy-Football-Predictions/` (the page itself
lives under `/picks/`; the root forwards there). Pages serves the `master` branch, root
folder. The workflow runs every morning, Thursday evening and Sunday late morning during the
season (and on any push that touches `eliminator/`), and it can be started by hand from the
Actions tab.

Each run does three things:

1. `python -m eliminator snapshot` refreshes the feeds, plans every pool in `state/` and
   writes `site/data/<season>/w<NN>-<pool>.json`. The current week's file is overwritten on
   every run (earlier recommendations that week are kept inside it as revisions); once the
   season moves on, the file is frozen and becomes the record of what was recommended. That
   record is kept honest by the runs after kickoff: a pick whose game has been played stays in
   the file at the price it closed at (not at 100% or 0% from the score, which is graded
   separately), and an entry that lost this week keeps its pick on the week's record instead
   of vanishing with the eliminated entry, so the week-by-week table and its "alive after"
   counts are right even though the last run of a week happens on Monday morning.
   Before planning, any pick a pool file is missing for a game that has already kicked off is
   filled in from that week's snapshot, so the entries stay alive from week to week without
   anyone running `plan --commit`. A pick you enter yourself (`record`, or editing the YAML)
   is never touched, and if you enter it before kickoff the planner locks it exactly as it
   would your own.
2. `python -m eliminator site` renders the snapshots into `picks/` at the repository root: a
   landing page with this week's picks per format and a week-by-week table graded against
   final scores, plus one page per week with the board, the options table and every entry's
   season plan. Raw JSON is served under `picks/data/`.
3. The snapshot, the pool files and the rendered site are committed back to `master`;
   GitHub's Pages build of the branch publishes them. The workflow waits for that build and
   fails if the live page is not the picks site.

To view locally: `python -m eliminator site --offline` writes `site/build/`; open
`site/build/index.html`. Put a The Odds API key in the `ODDS_API_KEY` repository secret to
price game-day lines from live books.

## Data sources (all free, no keys)

* **Schedule, lines, rest days, results, starting QBs**: nflverse `games.csv`
  (1999-present, refreshed in season). Moneylines and spreads for the coming several weeks,
  plus scattered prime-time games, are posted well in advance.
* **inpredictable power ratings** (`stats.inpredictable.com/rankings/nfl.php`, "NFL Betting
  Market Rankings"): GPF (generic points favoured) is a rating on the point-spread scale
  derived from the betting market, exactly what the projection needs. Verified against the
  live page from GitHub Actions (the page's rows carry hidden sparkline cells, which the
  parser handles). If the fetch fails the tool says so and falls back to the market fit
  below; you can also feed a saved page or a CSV with `--inpredictable-file`. A page that is
  still showing last season is rejected in any week (its records carry far more team-games
  than the schedule has played), as is one that disagrees with this season's posted lines by
  more than `inpredictable_max_rmse` points per team. The Actions
  workflow **Eliminator source probe** (run it from the Actions tab) prints what the live
  page and The Odds API return, for when either changes shape.
* **Market-implied ratings** (built in): a recency-weighted ridge fit of team strengths to
  every posted spread this season, seeded by last season's rating regressed toward zero.
  With lines posted for weeks 1-7 this already reproduces a full preseason power rating.
  `ratings_source: auto | inpredictable | market | blend` in `config.yaml`.
* **Live moneylines (optional)**: with a free key from the-odds-api.com in `config.yaml`
  (`data.odds_api_key`) or `ODDS_API_KEY`, `plan` prices this week's games from a de-vigged
  consensus of US books, which beats the daily feed on game day. Also untested live here.
* **Injury reports and depth charts**: nflverse weekly injury reports (game status and
  injury type) and depth charts (QB1 per team). With `qb.auto_from_injuries: true` (the
  default) `plan` and `snapshot` turn them into ledger entries automatically, see below.

## How the model works

### 1. Win probability where a price exists
De-vigged moneyline (multiplicative), spread as a fallback via `P(home) = Phi(spread / sigma)`.
`sigma` is fitted by maximum likelihood on 1999-2025 results (11.6 for the win/loss link).
Calibration shows the moneyline and the spread agree to 1.5 percentage points on average
and have the same log-loss, so either is fine; the moneyline is the direct price and wins.

### 2. Projecting games that have no line yet
`spread = (R_home - R_away) + HFA * home + rest_per_day * (home_rest - away_rest) + QB_home - QB_away`

* `R` is the healthy-baseline rating (see QB section).
* `HFA` = 1.6 points, from a season-by-season fit of closing spreads (2.5 in 2011, falling
  to about 1.5 by 2025; the default is the mean of the last four seasons). Neutral-site
  games (the 8 international games in 2026) get none.
* `rest_per_day` = 0.14 points per day of rest differential, clipped to +/-7 days. A bye
  week is worth roughly a point; a short Thursday week costs about half a point.
* When a line is posted for a future game it is used as the spread outright, at any horizon:
  Vegas is the source of truth wherever it has spoken, and the ratings only fill in games
  that have no line yet. The remaining uncertainty on a posted line is how far it can still
  move before kickoff, `posted_line_var_a + posted_line_var_b * h` points squared (1 + 1h by
  default, a standard deviation of about 1.4 points one week out and 3 points eight weeks
  out), far smaller than the projection error of a rating-based spread. That allowance
  calibrates itself: every `snapshot` run records the lines it sees for games not yet
  kicked off in `site/data/lines/<season>.csv`, and once at least 150 of those observations
  have a closing line to compare with (across three or more horizons) the allowance is
  fitted from how far the lines actually moved and replaces the default on every run.
* A week-18 line posted early is shrunk like a projection (see week 18 below): a line set in
  September cannot know who will rest starters.

### 3. Valuing the future
The projection error of a rating-based spread against the eventual closing line grows with
horizon. Fitted on 2011-2025 (as-of week `k`, `h` weeks ahead):

`var(h, k) = 9.4 + 1.78 h - 2.5 / (1 + k)` points squared

A future win probability is `Phi(spread / sqrt(sigma^2 + var(h, k)))`: a projected 7-point
favourite is a 73% pick this week and about 70% ten weeks out. That is the honest number for
a *named* team in a far-off week, and it is not how the season is played. Only this week's
pick is a commitment; in week 10 the pick will be whoever is best on that week's board, and
that board is richer than anything visible today: spreads widen as the season goes on (the
average spread grows from about 4.3 points in week 1 to 6.6 in week 18, and the week's
biggest favourite from about 9 points to 13-16) because more is known about the teams and
injuries have happened.

So the planner does not score fixed 18-week paths. Each simulated season carries its own
closing lines (today's line plus per-team drift calibrated to the variance above, so the
spreads widen as they do in reality), and an entry is valued as: use this team now, then
every later week take the best team still available at that season's line. Survival given
the lines is computed exactly (a product of the picks' probabilities, or the one-loss tail of the two-strike pool),
so close options can be told apart without millions of coin flips. Using a strong team now is
charged for the thinner menu the entry faces later, and nothing else. A pool of entries is
re-split every week, so in the 25-entry format each entry follows its own pattern later
(`planning.spread_weights`: most weeks the best team available to it, sometimes the second or
third); an entry's value to the pool is how often it survives while every other entry is dead,
with its own games forced to win and the shared games handled exactly. Season survival at
closing prices, geometric mean over 2015-2025 (`BACKTEST.md`):

| planner | policy, horizon 1 | policy, horizon 2 | policy, horizon 4 | fixed paths, discount 16 (previous) | greedy (no lookahead) |
|---|---|---|---|---|---|
| single elimination | **1.96%** | 1.94% | 1.81% | 1.78% | 1.58% |
| two strikes (out on the 2nd loss) | **10.6%** | 10.6% | 10.1% | 9.9% | 9.0% |

The default is `planning: {mode: policy, horizon: 1}`. The previous planner (fixed paths
chosen on a simulation with the future variance inflated 16x, `planning.mode: discount`) is
kept for comparison. **P(season)** on the site is the number the plan is chosen on: the
chance the entry survives the season played this way, on the calibrated simulation.

The Monte Carlo layer goes further than the point estimate: each simulated season draws a
persistent estimation error plus a random walk in every team's strength, so a team that
loses its QB in week 8 is weaker in all its games after week 8 in that scenario. This is
what makes 25 entries sharing a team correlated, and it is what the multi-entry objective is
evaluated on. The expected win probability of every game still matches the projection.

### 4. Week 18
Teams that have clinched rest starters. Calibration finds week-18 closing lines about 12%
flatter than late-season projections and 13 points squared noisier, and projected 6-9 point
favourites close on average 2 points shorter than they normally would. Week-18 projections
are therefore shrunk (`week18_shrink`, set to 0.8 in `config.yaml`, a bit more cautious than
the fitted 0.88) and given extra variance. When week 18 itself arrives the posted lines,
which already know who is resting, are used directly. `state/overrides.yaml` lets you add a
rest risk per team (`week18_rest_risk: {KC: 0.7}`) once the playoff picture is clear.

### 5. Quarterbacks
`state/qb_status.yaml` is the source of truth: team, player, tier or explicit points penalty
(elite 7, good 5, average 3.5, replacement 1.5), status, injury type, first week missed and,
if known, expected return week. From that the model builds `P(starter out)` for every
remaining week:

* with a `return_week`: out until then, then a fading setback probability (35%, 15%, 5%);
* without one: a duration prior by injury type (concussion, hamstring, high ankle, MCL,
  thumb, ribs, IR designation, ...), scaled by the current designation (questionable 40%,
  doubtful 85%, out 100%).

The expected effect `-penalty * P(out)` enters every future projection. The two blind
spots you called out are handled explicitly:

* **rated too low because the QB is out now** - the market-implied fit residualises the QB
  effect out of the spreads it learns from, so its ratings describe the healthy team. For
  inpredictable's blended rating the share of the absence already priced in (growing with
  games missed) is added back before projecting the weeks after the QB returns.
* **rated too high because this week's opponent is missing its QB** - the same
  residualisation handles it in the market fit; for inpredictable a small correction
  removes the boost from this week's line.

`update` prints the nflverse injury report for the week and warns about any starter with a
game designation who is not in the ledger.

**Automatic entries.** Nobody has to watch the injury report: on every `plan` or `snapshot`,
each team's QB1 who carries Out / Doubtful / Questionable on this week's nflverse report and
has no entry in `qb_status.yaml` gets an automatic situation at the default tier
(`qb.default_tier`, average = 3.5 points), starting this week, with the injury type from the
report driving the duration prior. These live in `state/qb_auto.yaml` (written by the tool,
committed by the CI workflow) and roll forward week to week: the entry keeps its first missed
week while the player stays on the report with a designation, and it is dropped once he is
off a published report (back, or a long absence the ratings have absorbed). A manual entry
for the team in `qb_status.yaml` always replaces the automatic one, which is where to put a
tier (elite / good) or an expected return week when you know more than the report. Set
`qb.auto_from_injuries: false` in `config.yaml` to turn this off. The week pages on the site
list every situation in play with its source.

### 6. Optimisation
* **One entry, single elimination**: with independent games the season survival probability
  is the product of the picks' win probabilities, so the best plan is a minimum-cost
  assignment on `-log p` (one distinct team per week, Hungarian algorithm, exact).
* **One entry, k strikes**: the entry is out on its k-th loss (`strikes: 2` in the pool file
  means the second loss eliminates, so one loss can be taken), and the objective is
  `P(fewer than k losses)`, optimised by local search (replace / swap moves) from the
  max-product plan with perturbed restarts. Strikes already used are read from the pool
  file, so after a loss the plan is re-solved with one fewer.
* **25 entries**: entries that share picks live and die together, so the value of a new entry
  is its survival in the scenarios where every other entry is already dead. The portfolio
  is built greedily on common random numbers from a pool of strong, deliberately diverse
  candidate paths (usage-penalised assignments, forced current-week alternatives, perturbed
  assignments), followed by coordinate-ascent passes. The report shows how many entries land
  on each team this week and the simulated probability that at least one survives.

The plan for later weeks is only there to justify this week's pick; every run re-solves the
whole remaining season with the latest information.

The "this week's options" table is the most useful output when you want to overrule the
tool: it lists, for each team you could use now, the plan score and the simulated season
survival probability if you use it now and play the rest optimally.

Example (two-strike pool, 2026 week 1, market-implied ratings):

```
-- this week's options: use the team now, play the rest optimally --
  team  p_now   score    P(season)  plan
  LAC   0.812  0.0300   0.1668   SF DET MIN NE LA DEN DAL SEA IND KC JAX PHI CHI GB BAL BUF CIN
  JAX   0.766  0.0282   0.1502   LAC SF CHI NE LA DEN DAL SEA IND KC CIN PHI DET GB BAL BUF HOU
  DET   0.718  0.0267   0.1519   LAC SF MIN NE LA DEN DAL SEA IND KC JAX PHI CHI GB BAL BUF CIN

PICK week 1: LAC ARI p=0.812 (new, kickoff Sun 09/13 16:25)
```

## Backtest
`python -m eliminator backtest --mode single|strikes|multi --seasons 2015-2025` replays past
seasons with only the information that existed at the time (closing lines through the current
week, no future lines, no QB ledger), which is strictly less than the live tool has. Two
scores: realised survival, and the survival probability of the picks *at closing prices*
(a low-variance measure of pick quality). `--horizons 1,2,4` compares planning horizons with
the fixed-path discount planner and a no-lookahead greedy baseline; `--sweep` runs the
`future_discount` grid for the discount planner. Results are summarised in `BACKTEST.md`.

## Files
```
eliminator/
  config.yaml            user overrides (season, week-18 shrink, ratings source, ...)
  calibration.json       fitted parameters (regenerate with `calibrate`)
  state/multi25.yaml     25-entry pool picks
  state/strikes2.yaml    two-strike pool picks
  state/qb_status.yaml   QB availability ledger (yours)
  state/qb_auto.yaml     QB situations added automatically from the injury report (tool-written)
  state/overrides.yaml   manual line overrides and week-18 rest risks
  site/data/             weekly recommendation snapshots (written by `snapshot`, committed by CI)
  site/logos/            team logos shown on the site (nflverse squared logos)
  site/build/            rendered site (ignored by git)
  data/cache/            downloaded feeds (ignored by git)
  eliminator/            the package
    data/                schedule (nflverse), inpredictable scraper, injuries
    model/               ratings fit, strength assembly, QB layer, projection
    optimize/            single-entry paths, k-strike search, simulation, portfolio
    calibration.py, backtest.py, plan.py, state.py, cli.py
  tests/                 pytest suite (synthetic seasons, no network)
```

## Practical notes
* The nflverse feed updates lines roughly daily in season. For game-day moves, paste the
  current moneylines into `state/overrides.yaml`; they beat the feed.
* A tie counts as a loss (ESPN and Yahoo rules). A missing pick in a finished week is a loss.
* `python -m eliminator plan --json out.json` dumps picks, the week's board and the ratings.
* Ratings are on the points scale (an average team is 0). `update` prints them so you can
  sanity-check the market view before trusting the plan.

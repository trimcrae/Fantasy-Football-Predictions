# Backtest, 2015-2025

`python -m eliminator backtest --mode single|strikes|multi --seasons 2015-2025`

Each season is replayed week by week with only what was knowable before that week's first
kickoff: closing lines for the current and past weeks, results of past weeks, and market
ratings fitted from those lines. Future weeks have **no** lines (nflverse only keeps closing
lines historically) and there is **no** QB ledger. The live tool sees lookahead lines for
six or more weeks and whatever you put in the ledger, so it has strictly more information
than these replays.

Two numbers per season:

* **realised** - whether the entry (or any entry) actually survived, and when it went out;
* **expected** - the survival probability of the picks *at closing prices*, i.e. the product
  (or, for the two-strike pool, the probability of at most one loss) of the picked teams'
  de-vigged closing win probabilities over all 18 weeks.

The two-strike pool eliminates on the second loss (`strikes: 2`), so its entry can lose
once. Earlier versions of this file scored it as "at most two losses"; every two-strike
number below is for the real rule. This is the pick-quality measure: it does not depend on
  which upsets happened to land, only on how good the chosen spots were.

## Valuing the future: re-picking beats discounting

The planner now values a candidate as "use this team now, then every later week take the
best team still available at that simulated season's closing line" (`planning.mode: policy`,
see README section 3). Survival given a season's lines is computed exactly, so the ranking is
not at the mercy of coin flips. `--horizons 1,2,4` compares how many weeks are treated as a
commitment (1 = only this week) against the previous planner (fixed 18-week paths chosen on a
simulation with the future variance inflated 16x) and a greedy no-lookahead baseline.
Geometric mean of the expected season survival over 2015-2025:

| planner | policy h=1 | policy h=2 | policy h=4 | discount x16 (previous) | greedy |
|---|---|---|---|---|---|
| single elimination | **1.97%** | 1.94% | 1.81% | 1.78% | 1.58% |
| two strikes, out on the 2nd loss (seasons survived) | **10.7%** (1) | 10.6% (2) | 10.1% (2) | 9.9% (1) | 9.0% (1) |

The `policy h=1` column is with the week-18 rest model (README section 4: standings after
week 17 in every simulated season, settled seeds and eliminated teams docked the fitted
points); the other columns predate it. It moves the replay very little, since the replay's
own week-18 picks are made on closing lines either way; its job is the live tool's December
planning, which the replay does not reward.

Reading: committing only this week and treating everything after as "best available then"
is the best planner in both single-entry formats, 10% better than the discount planner for
single elimination and 8% better with two strikes (h=2 is level with it there, within
noise); the longer the commitment, the closer it gets to the fixed-path planner. The discount planner was a proxy for the same idea (squash
the far-off weeks so the optimiser stops hoarding teams for spots that do not materialise);
this does it directly and also lets the later menu widen as real spreads do. `horizon: 1` is
the default.

For the 25-entry pool the only score available is the realised one (a survivor in the
season or not), which is one rare event per season and therefore noisy: with a season-level
chance of roughly 20-25%, 2 and 4 out of 11 are one standard deviation apart.

| planner | policy h=1 | policy h=2 | policy h=4 | discount x16 (previous) | greedy |
|---|---|---|---|---|---|
| 25 entries, seasons with a survivor | 2 / 11 | 3 / 11 | 2 / 11 | 4 / 11 | 0 / 11 |
| mean survivors per season | 0.18 | 0.45 | 0.18 | 0.36 | 0.00 |

This does not confirm the improvement for the pool format, nor contradict it; the pool is
played with the same valuation the single-entry results validate, plus the weekly re-split
(`spread_weights`), which cannot be tuned on a metric this noisy. `horizon: 1` is used for
every format so that the site's season odds mean the same thing everywhere.

## The previous planner: choosing the future discount

`future_discount` multiplies the calibrated projection-error variance when the plan is
chosen. Geometric mean of the expected season survival over the 11 seasons, plus a greedy
"biggest favourite every week, no lookahead" baseline:

| future_discount | 0.5 | 1 (calibrated) | 2 | 4 | 8 | **16** | 32 | 64 | greedy |
|---|---|---|---|---|---|---|---|---|---|
| single elimination | 1.41% | 1.42% | 1.46% | 1.50% | 1.60% | **1.78%** | 1.76% | 1.81% | 1.58% |
| two strikes (out on the 2nd loss) | - | 8.3% | - | 8.7% | - | **9.9%** | - | 10.0% | 9.0% |

Reading: planning on the calibrated projection is *worse* than not planning at all. The
optimiser selects the future weeks whose projections happen to be most optimistic, so the
value of "the rest of the plan" is overstated (an optimiser's curse), and the entry burns
sure things now to protect spots that on average do not materialise. Discounting the future
by 16x keeps the ordering information (it beats greedy by 13% in single elimination and 7%
with two strikes) without over-committing to it. Beyond 16 the curve is flat within noise.
16 is the shipped default; the reported survival odds always use the calibrated uncertainty.

## Results with the previous planner (`future_discount = 16`)

### One entry, single elimination

| season | out in week | expected survival | picks |
|---|---|---|---|
| 2015 | 2 | 1.2% | DAL NO NE SEA KC NYJ ARI LA CIN PIT CAR GB CHI TB MIN DET DEN |
| 2016 | 8 | 1.0% | HOU CAR MIA WAS NE TEN CIN MIN DAL ARI KC BUF SEA DET ATL GB IND |
| 2017 | 5 | 2.8% | BUF LV GB SEA PIT ATL DAL PHI HOU LA KC NE LAC CIN NO BAL MIN |
| 2018 | 3 | 1.7% | BAL LA MIN GB TEN HOU IND PIT CHI KC NO LAC PHI BUF ATL NE SEA |
| 2019 | 10 | 4.4% | SEA BAL DAL LAC PHI NE BUF MIN SF NO LV CLE CAR GB KC DEN TEN |
| 2020 | 5 | 3.5% | NE TB CLE LA SF MIA BUF KC PIT GB LAC NO MIN SEA BAL CHI IND |
| 2021 | 9 | 2.4% | SF CLE DEN BUF MIN IND ARI KC DAL PIT TEN NE LA GB PHI LAC SEA TB |
| 2022 | 1 | 0.8% | CIN DEN MIN LAC BUF GB NE PHI KC SF BAL MIA CLE DAL NO TEN ATL JAX |
| 2023 | 12 | 1.1% | BAL BUF KC SF MIA LA SEA LAC CLE DAL DET NE PIT GB NO DEN PHI CIN |
| 2024 | 1 | 1.3% | CIN DET CLE NYJ SEA GB WAS DEN NO LAC MIA HOU KC PHI BAL BUF TB ATL |
| 2025 | 3 | 2.6% | DEN BAL GB BUF ARI IND KC NE LA SEA PIT DET LAC TB PHI HOU DAL JAX |

Survived 0 of 11 seasons; geometric-mean expected survival 1.8% per season (so about a 1 in 6
chance of at least one full-season survival across these 11 years). One entry surviving 18
weeks of single elimination is rare no matter how well it is played, which is exactly why the
25-entry pool is worth diversifying.

### One entry, two strikes (out on the second loss)

Previous planner (`future_discount = 16`) and, for comparison, the current policy planner
(`horizon: 1`), each replayed with one life:

| season | discount x16: out in week | expected | policy h=1: out in week | expected |
|---|---|---|---|---|
| 2015 | 5 | 7.1% | 8 | 9.1% |
| 2016 | 11 | 6.3% | 11 | 7.1% |
| 2017 | 6 | 14.3% | 5 | 11.3% |
| 2018 | 5 | 9.6% | 3 | 13.6% |
| 2019 | 13 | 19.7% | 10 | 17.5% |
| 2020 | survived | 16.6% | survived | 14.8% |
| 2021 | 10 | 12.6% | 9 | 13.3% |
| 2022 | 6 | 5.1% | 7 | 6.6% |
| 2023 | 13 | 6.8% | 12 | 8.5% |
| 2024 | 2 | 7.6% | 2 | 6.4% |
| 2025 | 5 | 13.3% | 14 | 16.3% |

Each survived 1 of 11 seasons (2020); geometric-mean expected survival 9.9% for the previous
planner and 10.7% for the current one, so the realised count is right where the closing
prices say it should be. With one life instead of two, a two-strike season is worth about a
third of what the old scoring said, and the lone-entry pool sits much closer to single
elimination than to a comfortable hedge.

### 25 entries, single elimination

| season | survivors | alive after week 4 / 8 / 12 / 16 | last entry out |
|---|---|---|---|
| 2015 | 0 | 4 / 1 / 0 / 0 | 11 |
| 2016 | 1 | 9 / 3 / 2 / 1 | - |
| 2017 | 0 | 10 / 2 / 1 / 0 | 14 |
| 2018 | 0 | 1 / 0 / 0 / 0 | 5 |
| 2019 | 1 | 14 / 6 / 2 / 1 | - |
| 2020 | 1 | 9 / 2 / 2 / 1 | - |
| 2021 | 0 | 13 / 7 / 0 / 0 | 12 |
| 2022 | 1 | 7 / 1 / 1 / 1 | - |
| 2023 | 0 | 12 / 5 / 3 / 0 | 13 |
| 2024 | 0 | 0 / 0 / 0 / 0 | 3 |
| 2025 | 0 | 15 / 2 / 1 / 0 | 14 |

At least one survivor in 4 of 11 seasons (mean 0.36 survivors). 2024 is the cautionary tale:
the Bengals, Lions and Browns all lost as favourites in weeks 1-3 and the whole pool was gone
by week 3. Note that this replay diversifies with 400 scenarios and no lookahead lines; the
live planner uses 20,000 scenarios and the posted lines for the coming weeks.

### Which simulation should score the 25-entry split?

The portfolio is built on the planning-view simulation (future discounted 16x), while
survival is reported on the calibrated one. Scoring the split on the calibrated simulation
instead (`portfolio.allocation_view: calibrated`) was replayed over 2015-2025 with the same
400 scenarios: a survivor in 3 of 11 seasons (mean 0.27) against 4 of 11 (mean 0.36) for the
planning view. Eleven realised seasons cannot separate the two, but there is no evidence for
switching, so the planning view stays the default. With that planner the site's "adds" column for this week's
options is scored on whichever view built the split, so the column and the split agree.

## Caveats

* 11 seasons is a small sample for realised outcomes; lean on the expected column.
* Closing lines are used as "the current week's price". You can act on them until kickoff,
  so this is fair, but a Tuesday line is a little worse than a Sunday-morning one.
* No QB information is used in the replay. The ledger is the main way the live tool should
  beat these numbers, especially for the weeks-ahead projections that a backup QB distorts.
* Realised 25-entry results depend on the simulation seed; expect a survivor or two to move
  between seasons on a different seed.

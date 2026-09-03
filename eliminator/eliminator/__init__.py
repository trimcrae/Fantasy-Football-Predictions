"""NFL eliminator / survivor pool strategy engine.

Two pool formats are supported:

* ``multi``   - single elimination, many entries (e.g. 25). Objective: maximise the
                probability that at least one entry survives the whole season.
* ``strikes`` - one entry that is eliminated on its k-th loss (e.g. two strikes).
                Objective: maximise the probability the entry survives the season.

Vegas prices are the source of truth wherever a line exists. Weeks without a line
are projected from market-implied team strength (inpredictable GPF when reachable,
otherwise a rating fit to the posted spreads), home field, rest and QB availability,
with certainty discounted the further out a game is.
"""

__version__ = "0.1.0"

from eliminator.data.odds_api import consensus_overrides
from eliminator.data.schedule import regular_season


def test_consensus_overrides_maps_events(games_all):
    g = regular_season(games_all, 2026)
    first = g[g.week == 1].iloc[0]
    from eliminator.teams import TEAM_NAMES
    ev = {"home_team": " ".join(TEAM_NAMES[first["home"]]), "away_team": " ".join(TEAM_NAMES[first["away"]]),
          "bookmakers": [{"markets": [{"key": "h2h", "outcomes": [
              {"name": " ".join(TEAM_NAMES[first["home"]]), "price": -200},
              {"name": " ".join(TEAM_NAMES[first["away"]]), "price": 170}]}]},
                         {"markets": [{"key": "h2h", "outcomes": [
              {"name": " ".join(TEAM_NAMES[first["home"]]), "price": -190},
              {"name": " ".join(TEAM_NAMES[first["away"]]), "price": 160}]}]}]}
    recs = consensus_overrides([ev], g, 1, 11.5)
    assert len(recs) == 1 and recs[0]["books"] == 2 and recs[0]["spread"] > 0 and recs[0]["home"] == first["home"]

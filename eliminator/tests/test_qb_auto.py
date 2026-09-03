import pandas as pd

from eliminator.config import DEFAULTS
from eliminator.model.qb import QBSituation, load_auto, save_auto, update_auto

CFG = DEFAULTS["qb"]


def _watch(rows):
    return pd.DataFrame(rows, columns=["team", "player", "status", "injury", "practice", "is_qb1"])


def test_new_starter_with_designation_is_added_at_default_tier():
    w = _watch([("BAL", "Lamar Jackson", "Questionable", "Hamstring", "Limited", True),
                ("BAL", "Backup Guy", "Out", "Knee", "Did Not Participate", False),
                ("KC", "Patrick Mahomes", "", "", "Full", True)])
    auto = update_auto([], w, [], 5, CFG, report_published=True)
    assert [(s.team, s.player, s.status, s.injury, s.injured_week) for s in auto] == [("BAL", "Lamar Jackson", "questionable", "hamstring", 5)]
    assert auto[0].penalty == CFG["penalty_by_tier"][CFG["default_tier"]]


def test_carries_over_and_keeps_first_week_even_after_depth_chart_moves():
    prev = [QBSituation(team="BAL", player="Lamar Jackson", status="out", injury="hamstring", injured_week=5, note="auto")]
    w = _watch([("BAL", "Lamar Jackson", "Doubtful", "Hamstring", "Limited", False)])   # backup now QB1
    auto = update_auto(prev, w, [], 7, CFG, report_published=True)
    assert len(auto) == 1 and auto[0].status == "doubtful" and auto[0].injured_week == 5


def test_dropped_when_off_a_published_report_but_kept_when_no_report_yet():
    prev = [QBSituation(team="BAL", player="Lamar Jackson", status="out", injury="hamstring", injured_week=5, note="auto")]
    assert update_auto(prev, _watch([]), [], 8, CFG, report_published=True) == []
    assert len(update_auto(prev, _watch([]), [], 8, CFG, report_published=False)) == 1


def test_manual_entry_wins_over_auto():
    prev = [QBSituation(team="BAL", player="Lamar Jackson", status="out", injury="hamstring", injured_week=5, note="auto")]
    manual = [QBSituation(team="BAL", player="Lamar Jackson", penalty=7, status="out", injury="hamstring", injured_week=5, return_week=9)]
    w = _watch([("BAL", "Lamar Jackson", "Out", "Hamstring", "DNP", True)])
    assert update_auto(prev, w, manual, 6, CFG, report_published=True) == []


def test_auto_file_roundtrip(tmp_path):
    sits = [QBSituation(team="BAL", player="Lamar Jackson", penalty=3.5, status="out", injury="hamstring", injured_week=5, note="auto: x")]
    p = save_auto(sits, tmp_path / "qb_auto.yaml")
    back = load_auto(p, CFG)
    assert len(back) == 1 and back[0].team == "BAL" and back[0].injured_week == 5 and back[0].penalty == 3.5
    assert p.read_text().startswith("# Written by")

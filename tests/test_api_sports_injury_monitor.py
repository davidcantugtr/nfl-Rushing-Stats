from datetime import date, time

from src.api_sports_injury_monitor import build_game_day_plan, diff_injuries, normalize_injury


def test_sunday_plan_stays_inside_free_tier_and_has_priority_sweeps():
    plan = build_game_day_plan(date(2026, 9, 20))
    assert len(plan) == 94
    assert {call.priority for call in plan} == {"STANDARD", "NOON_PREKICK", "LATE_PREKICK"}
    assert sum(call.priority == "NOON_PREKICK" for call in plan) == 2
    assert sum(call.priority == "LATE_PREKICK" for call in plan) == 2

    cycle_times = {call.local_time[11:16] for call in plan}
    assert len([value for value in cycle_times if time.fromisoformat(value) <= time(14, 45)]) == 14
    assert len([value for value in cycle_times if time(14, 55) <= time.fromisoformat(value) <= time(18, 45)]) == 17
    assert len([value for value in cycle_times if time.fromisoformat(value) >= time(19, 0)]) == 16


def test_diff_is_append_only_and_flags_disappearance_for_confirmation():
    first = normalize_injury(
        {"player": {"id": 10, "name": "Defender One"}, "team": {"id": 2, "name": "ABC"}, "status": "Questionable"}
    )
    changed = normalize_injury(
        {"player": {"id": 10, "name": "Defender One"}, "team": {"id": 2, "name": "ABC"}, "status": "Out"}
    )
    added = normalize_injury(
        {"player": {"id": 11, "name": "Defender Two"}, "team": {"id": 2, "name": "ABC"}, "status": "Questionable"}
    )

    changes = diff_injuries([first], [changed, added])
    assert [row["event_type"] for row in changes] == ["STATUS_CHANGED", "ADDED_TO_CURRENT_FEED"]

    removals = diff_injuries([changed], [])
    assert removals[0]["event_type"] == "REMOVED_FROM_CURRENT_FEED"
    assert removals[0]["verification_state"] == "REQUIRES_CONFIRMATION"

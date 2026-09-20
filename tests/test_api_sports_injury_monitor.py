from datetime import date, time
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from src.api_sports_injury_monitor import (
    build_game_day_plan,
    change_fingerprint,
    classify_severity,
    diff_injuries,
    format_slack_alert,
    is_defensive_or_unclassified,
    notify_slack_changes,
    normalize_injury,
)


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


def test_slack_alert_classifies_and_formats_critical_defensive_change():
    previous = normalize_injury(
        {
            "player": {"id": 10, "name": "Defender One", "position": "CB"},
            "team": {"id": 2, "name": "ABC"},
            "status": "Questionable",
        }
    )
    current = normalize_injury(
        {
            "player": {"id": 10, "name": "Defender One", "position": "CB"},
            "team": {"id": 2, "name": "ABC"},
            "status": "Out",
        }
    )
    change = diff_injuries([previous], [current])[0]

    assert classify_severity(change) == "CRITICAL"
    assert is_defensive_or_unclassified(change)
    assert "Defender One" in format_slack_alert(change)["text"]
    assert len(change_fingerprint(change)) == 24


def test_offensive_position_is_suppressed_but_missing_position_is_retained():
    offensive = normalize_injury(
        {"player": {"id": 20, "name": "Receiver", "position": "WR"}, "status": "Questionable"}
    )
    unknown = normalize_injury({"player": {"id": 21, "name": "Unknown"}, "status": "Questionable"})

    assert not is_defensive_or_unclassified(diff_injuries([], [offensive])[0])
    assert is_defensive_or_unclassified(diff_injuries([], [unknown])[0])


def test_slack_delivery_is_deduplicated_across_retries():
    injury = normalize_injury(
        {"player": {"id": 30, "name": "Safety", "position": "S"}, "status": "Questionable"}
    )
    change = diff_injuries([], [injury])[0]
    with TemporaryDirectory() as directory, patch(
        "src.api_sports_injury_monitor.post_slack_payload"
    ) as post:
        ledger = Path(directory) / "slack.jsonl"
        first = notify_slack_changes([change], webhook_url="https://example.invalid", notification_ledger=ledger)
        second = notify_slack_changes([change], webhook_url="https://example.invalid", notification_ledger=ledger)

    assert first == {"sent": 1, "skipped": 0, "failed": 0}
    assert second == {"sent": 0, "skipped": 1, "failed": 0}
    assert post.call_count == 1

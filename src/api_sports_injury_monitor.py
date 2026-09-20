#!/usr/bin/env python3
"""Low-cost API-Sports NFL injury monitor.

The API-Sports injuries endpoint exposes current state rather than history. This
module snapshots every response and creates an append-only change ledger so a
later status change or disappearance is never silently lost.

Authentication is read from API_SPORTS_KEY. Never commit the key.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
import time
from dataclasses import asdict, dataclass
from datetime import date, datetime, time as dt_time, timedelta
from pathlib import Path
from typing import Any, Iterable
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen
from zoneinfo import ZoneInfo


DEFAULT_BASE_URL = "https://v1.american-football.api-sports.io"
DEFAULT_TIMEZONE = "America/Chicago"
DEFAULT_DAILY_CAP = 100
DEFAULT_RESERVE = 6
DEFAULT_SLACK_TIMEOUT = 10

DEFENSIVE_POSITIONS = {
    "CB",
    "DB",
    "DE",
    "DEFENSIVE BACK",
    "DEFENSIVE END",
    "DEFENSIVE LINE",
    "DEFENSIVE TACKLE",
    "DL",
    "DT",
    "FS",
    "ILB",
    "LB",
    "LINEBACKER",
    "NT",
    "OLB",
    "S",
    "SAFETY",
    "SS",
}


@dataclass(frozen=True)
class PlannedCall:
    local_time: str
    utc_time: str
    endpoint: str
    priority: str
    purpose: str


def build_game_day_plan(
    game_date: date,
    *,
    timezone: str = DEFAULT_TIMEZONE,
    start: dt_time = dt_time(11, 45),
    end: dt_time = dt_time(22, 45),
    interval_minutes: int = 15,
    priority_times: Iterable[dt_time] = (dt_time(11, 55), dt_time(14, 55)),
) -> list[PlannedCall]:
    """Build the 94-request Sunday pilot plan.

    Each polling cycle calls the global injuries endpoint and the date-level
    games endpoint. Priority sweeps are inserted five minutes before the main
    kickoff windows.
    """

    tz = ZoneInfo(timezone)
    cursor = datetime.combine(game_date, start, tzinfo=tz)
    end_dt = datetime.combine(game_date, end, tzinfo=tz)
    cycle_times: set[datetime] = set()
    while cursor <= end_dt:
        cycle_times.add(cursor)
        cursor += timedelta(minutes=interval_minutes)
    for priority_time in priority_times:
        cycle_times.add(datetime.combine(game_date, priority_time, tzinfo=tz))

    priority_labels = {
        dt_time(11, 55): "NOON_PREKICK",
        dt_time(14, 55): "LATE_PREKICK",
    }
    calls: list[PlannedCall] = []
    for local_dt in sorted(cycle_times):
        priority = priority_labels.get(local_dt.time(), "STANDARD")
        for endpoint, purpose in (
            ("injuries", "Detect additions, status changes, and removals"),
            ("games", "Track game state and clock for injury context"),
        ):
            calls.append(
                PlannedCall(
                    local_time=local_dt.isoformat(),
                    utc_time=local_dt.astimezone(ZoneInfo("UTC")).isoformat(),
                    endpoint=endpoint,
                    priority=priority,
                    purpose=purpose,
                )
            )
    return calls


def _stable_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def _first(mapping: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        value = mapping.get(key)
        if value not in (None, ""):
            return value
    return None


def normalize_injury(item: dict[str, Any]) -> dict[str, Any]:
    """Normalize API variations without discarding the raw record."""

    player = item.get("player") if isinstance(item.get("player"), dict) else {}
    team = item.get("team") if isinstance(item.get("team"), dict) else {}
    game = item.get("game") if isinstance(item.get("game"), dict) else {}
    normalized = {
        "player_id": _first(player, "id") or _first(item, "player_id", "player"),
        "player_name": _first(player, "name") or _first(item, "player_name", "name"),
        "position": _first(player, "position", "pos") or _first(item, "position", "pos"),
        "team_id": _first(team, "id") or _first(item, "team_id"),
        "team_name": _first(team, "name") or _first(item, "team_name", "team"),
        "game_id": _first(game, "id") or _first(item, "game_id"),
        "status": _first(item, "status", "type", "designation"),
        "reason": _first(item, "reason", "description", "injury"),
        "date": _first(item, "date", "updated"),
    }
    normalized["identity"] = "|".join(
        str(normalized.get(key) or "")
        for key in ("player_id", "player_name", "team_id", "game_id")
    )
    normalized["state_hash"] = hashlib.sha256(
        _stable_json({k: v for k, v in normalized.items() if k not in {"identity", "state_hash"}}).encode()
    ).hexdigest()[:16]
    normalized["raw"] = item
    return normalized


def _status_text(change: dict[str, Any]) -> str:
    current = change.get("current") or {}
    previous = change.get("previous") or {}
    return " ".join(
        str(value or "")
        for value in (
            current.get("status"),
            current.get("reason"),
            previous.get("status"),
            previous.get("reason"),
        )
    ).lower()


def classify_severity(change: dict[str, Any]) -> str:
    """Classify an injury-feed change for mobile notification priority."""

    event_type = change.get("event_type")
    text = _status_text(change)
    if event_type == "REMOVED_FROM_CURRENT_FEED":
        return "IMPORTANT"
    if any(term in text for term in ("out", "inactive", "ruled out", "cart", "did not return")):
        return "CRITICAL"
    if any(term in text for term in ("doubtful", "questionable", "limited", "did not practice", "dnp")):
        return "IMPORTANT"
    return "INFORMATIONAL"


def is_defensive_or_unclassified(change: dict[str, Any]) -> bool:
    """Retain defensive changes and unknown positions during the validation pilot."""

    row = change.get("current") or change.get("previous") or {}
    position = str(row.get("position") or "").strip().upper()
    return not position or position in DEFENSIVE_POSITIONS


def change_fingerprint(change: dict[str, Any]) -> str:
    payload = {
        "event_type": change.get("event_type"),
        "identity": change.get("identity"),
        "previous_hash": (change.get("previous") or {}).get("state_hash"),
        "current_hash": (change.get("current") or {}).get("state_hash"),
    }
    return hashlib.sha256(_stable_json(payload).encode()).hexdigest()[:24]


def format_slack_alert(change: dict[str, Any]) -> dict[str, Any]:
    row = change.get("current") or change.get("previous") or {}
    previous = change.get("previous") or {}
    current = change.get("current") or {}
    severity = classify_severity(change)
    icon = {"CRITICAL": "\U0001f6a8", "IMPORTANT": "\u26a0\ufe0f", "INFORMATIONAL": "\u2139\ufe0f"}[severity]
    team = row.get("team_name") or "Unknown team"
    player = row.get("player_name") or "Unknown player"
    position = row.get("position") or "position unclassified"
    old_status = previous.get("status") or "not previously listed"
    new_status = current.get("status") or "removed from current feed"
    observed = change.get("observed_at_utc") or datetime.now(ZoneInfo("UTC")).isoformat()
    verification = change.get("verification_state") or "PROVISIONAL"
    fallback = f"{icon} {severity}: {team} {position} {player} — {old_status} -> {new_status}"
    return {
        "text": fallback,
        "blocks": [
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": f"{icon} *{severity} — NFL Injury Update*\n*{team} | {position} | {player}*",
                },
            },
            {
                "type": "section",
                "fields": [
                    {"type": "mrkdwn", "text": f"*Previous*\n{old_status}"},
                    {"type": "mrkdwn", "text": f"*Current*\n{new_status}"},
                    {"type": "mrkdwn", "text": f"*Verification*\n{verification}"},
                    {"type": "mrkdwn", "text": f"*Observed (UTC)*\n{observed}"},
                ],
            },
        ],
    }


def post_slack_payload(webhook_url: str, payload: dict[str, Any], *, timeout: int = DEFAULT_SLACK_TIMEOUT) -> None:
    body = json.dumps(payload).encode("utf-8")
    request = Request(webhook_url, data=body, headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urlopen(request, timeout=timeout) as response:
            response_body = response.read().decode("utf-8", errors="replace")
            if response.status != 200 or response_body.strip() != "ok":
                raise RuntimeError(f"Slack webhook rejected notification: HTTP {response.status}")
    except HTTPError as exc:
        raise RuntimeError(f"Slack webhook HTTP {exc.code}") from exc
    except (URLError, TimeoutError) as exc:
        raise RuntimeError(f"Slack webhook connection failed: {exc}") from exc


def notify_slack_changes(
    changes: Iterable[dict[str, Any]],
    *,
    webhook_url: str,
    notification_ledger: Path,
) -> dict[str, int]:
    notified = {
        row.get("fingerprint")
        for row in _read_jsonl(notification_ledger)
        if row.get("delivery_state") == "SENT"
    }
    sent = skipped = failed = 0
    for change in changes:
        if not is_defensive_or_unclassified(change):
            skipped += 1
            continue
        fingerprint = change_fingerprint(change)
        if fingerprint in notified:
            skipped += 1
            continue
        try:
            post_slack_payload(webhook_url, format_slack_alert(change))
        except RuntimeError as exc:
            failed += 1
            _append_jsonl(
                notification_ledger,
                [{"fingerprint": fingerprint, "delivery_state": "FAILED", "error": str(exc)}],
            )
            continue
        sent += 1
        notified.add(fingerprint)
        _append_jsonl(
            notification_ledger,
            [{"fingerprint": fingerprint, "delivery_state": "SENT", "event": change}],
        )
    return {"sent": sent, "skipped": skipped, "failed": failed}


def diff_injuries(previous: Iterable[dict[str, Any]], current: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return append-only changes between two current-state injury lists."""

    old = {row["identity"]: row for row in previous}
    new = {row["identity"]: row for row in current}
    changes: list[dict[str, Any]] = []
    observed_at = datetime.now(ZoneInfo("UTC")).isoformat()
    for identity, row in sorted(new.items()):
        prior = old.get(identity)
        if prior is None:
            event_type = "ADDED_TO_CURRENT_FEED"
        elif prior.get("state_hash") != row.get("state_hash"):
            event_type = "STATUS_CHANGED"
        else:
            continue
        changes.append(
            {
                "observed_at_utc": observed_at,
                "event_type": event_type,
                "verification_state": "PROVISIONAL",
                "identity": identity,
                "previous": prior,
                "current": row,
            }
        )
    for identity, row in sorted(old.items()):
        if identity not in new:
            changes.append(
                {
                    "observed_at_utc": observed_at,
                    "event_type": "REMOVED_FROM_CURRENT_FEED",
                    "verification_state": "REQUIRES_CONFIRMATION",
                    "identity": identity,
                    "previous": row,
                    "current": None,
                }
            )
    return changes


class ApiSportsClient:
    def __init__(self, api_key: str, *, base_url: str = DEFAULT_BASE_URL, timeout: int = 20):
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def get(self, endpoint: str, params: dict[str, Any]) -> tuple[dict[str, Any], dict[str, str]]:
        url = f"{self.base_url}/{endpoint}?{urlencode(params)}"
        request = Request(url, headers={"x-apisports-key": self.api_key, "Accept": "application/json"})
        try:
            with urlopen(request, timeout=self.timeout) as response:
                payload = json.loads(response.read().decode("utf-8"))
                headers = {key.lower(): value for key, value in response.headers.items()}
        except HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"API-Sports HTTP {exc.code}: {body[:500]}") from exc
        except (URLError, TimeoutError) as exc:
            raise RuntimeError(f"API-Sports connection failed: {exc}") from exc
        errors = payload.get("errors")
        if errors:
            raise RuntimeError(f"API-Sports returned errors: {errors}")
        return payload, headers


def _read_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    return json.loads(path.read_text())


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def _append_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(_stable_json(row) + "\n")


def run_poll(
    client: ApiSportsClient,
    *,
    game_date: date,
    output_dir: Path,
    league_id: int,
    season: int,
    timezone: str,
    daily_cap: int,
    reserve: int,
) -> dict[str, Any]:
    ledger_path = output_dir / "request_ledger.jsonl"
    used_total = 0
    utc_day = datetime.now(ZoneInfo("UTC")).date().isoformat()
    if ledger_path.exists():
        used_total = sum(1 for line in ledger_path.read_text().splitlines() if line)
    if used_total + 2 > daily_cap - reserve:
        raise RuntimeError(
            f"Hard pilot-budget guard stopped poll: {used_total} requests already used; "
            f"{reserve} of {daily_cap} reserved across the full game-day window."
        )

    common = {"league": league_id, "season": season}
    injury_payload, injury_headers = client.get("injuries", common)
    games_payload, games_headers = client.get(
        "games", {**common, "date": game_date.isoformat(), "timezone": timezone}
    )
    observed_at = datetime.now(ZoneInfo("UTC"))
    stamp = observed_at.strftime("%Y%m%dT%H%M%SZ")
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / f"injuries_{stamp}.json").write_text(json.dumps(injury_payload, indent=2, sort_keys=True))
    (output_dir / f"games_{stamp}.json").write_text(json.dumps(games_payload, indent=2, sort_keys=True))

    current = [normalize_injury(row) for row in injury_payload.get("response", [])]
    latest_path = output_dir / "injuries_latest_normalized.json"
    is_baseline = not latest_path.exists()
    previous = _read_json(latest_path, [])
    changes = [] if is_baseline else diff_injuries(previous, current)
    latest_path.write_text(json.dumps(current, indent=2, sort_keys=True))
    _append_jsonl(output_dir / "injury_change_ledger.jsonl", changes)

    slack_result = {"sent": 0, "skipped": 0, "failed": 0}
    slack_webhook = os.getenv("SLACK_INJURY_WEBHOOK")
    if changes and slack_webhook:
        slack_result = notify_slack_changes(
            changes,
            webhook_url=slack_webhook,
            notification_ledger=output_dir / "slack_notification_ledger.jsonl",
        )

    for endpoint, headers in (("injuries", injury_headers), ("games", games_headers)):
        _append_jsonl(
            ledger_path,
            [
                {
                    "observed_at_utc": observed_at.isoformat(),
                    "utc_day": utc_day,
                    "endpoint": endpoint,
                    "remaining": headers.get("x-ratelimit-requests-remaining"),
                    "limit": headers.get("x-ratelimit-requests-limit"),
                }
            ],
        )
    return {
        "observed_at_utc": observed_at.isoformat(),
        "injuries_seen": len(current),
        "changes_detected": len(changes),
        "slack_notifications": slack_result,
        "baseline_created": is_baseline,
        "games_seen": len(games_payload.get("response", [])),
        "output_dir": str(output_dir),
    }


def write_plan_csv(plan: Iterable[PlannedCall], destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    rows = [asdict(call) for call in plan]
    with destination.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(PlannedCall.__dataclass_fields__))
        writer.writeheader()
        writer.writerows(rows)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", default=date.today().isoformat(), help="Game date, YYYY-MM-DD")
    parser.add_argument("--timezone", default=DEFAULT_TIMEZONE)
    parser.add_argument("--league-id", type=int, default=int(os.getenv("API_SPORTS_NFL_LEAGUE_ID", "1")))
    parser.add_argument("--season", type=int, default=2026)
    parser.add_argument("--output-dir", type=Path, default=Path("data/live/api_sports"))
    parser.add_argument("--plan-csv", type=Path, default=Path("config/api_sports_request_plan.csv"))
    parser.add_argument("--dry-run", action="store_true", help="Write and print the request plan without API calls")
    parser.add_argument("--once", action="store_true", help="Run one injuries+games polling cycle")
    parser.add_argument("--run-plan", action="store_true", help="Stay alive and execute the remaining scheduled pilot cycles")
    parser.add_argument("--window-start", help="Only run cycles at/after this local HH:MM time")
    parser.add_argument("--window-end", help="Only run cycles at/before this local HH:MM time")
    parser.add_argument("--slack-test", action="store_true", help="Send one safe deployment-test notification")
    return parser.parse_args(argv)


def _parse_local_time(value: str) -> dt_time:
    try:
        return dt_time.fromisoformat(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"Invalid local time {value!r}; expected HH:MM") from exc


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    game_date = date.fromisoformat(args.date)
    plan = build_game_day_plan(game_date, timezone=args.timezone)
    write_plan_csv(plan, args.plan_csv)
    if len(plan) > DEFAULT_DAILY_CAP - DEFAULT_RESERVE:
        raise RuntimeError(f"Plan has {len(plan)} calls; maximum allowed is {DEFAULT_DAILY_CAP - DEFAULT_RESERVE}")
    print(json.dumps({"planned_requests": len(plan), "reserve": DEFAULT_DAILY_CAP - len(plan), "plan": str(args.plan_csv)}))
    if args.slack_test:
        webhook_url = os.getenv("SLACK_INJURY_WEBHOOK")
        if not webhook_url:
            print("SLACK_INJURY_WEBHOOK is required for the Slack test.", file=sys.stderr)
            return 2
        post_slack_payload(
            webhook_url,
            {
                "text": "NFL Injury Monitor connected successfully.",
                "blocks": [
                    {
                        "type": "section",
                        "text": {
                            "type": "mrkdwn",
                            "text": "\u2705 *NFL Injury Monitor connected*\nTest notification successful. Live alerts will post only when a new injury-feed change is detected.",
                        },
                    }
                ],
            },
        )
        print(json.dumps({"slack_test": "SENT"}))
        return 0
    if args.dry_run and not args.once and not args.run_plan:
        return 0
    api_key = os.getenv("API_SPORTS_KEY")
    if not api_key:
        print("API_SPORTS_KEY is required for live polling.", file=sys.stderr)
        return 2
    client = ApiSportsClient(api_key, base_url=os.getenv("API_SPORTS_BASE_URL", DEFAULT_BASE_URL))
    poll_kwargs = {
        "game_date": game_date,
        "output_dir": args.output_dir,
        "league_id": args.league_id,
        "season": args.season,
        "timezone": args.timezone,
        "daily_cap": DEFAULT_DAILY_CAP,
        "reserve": DEFAULT_RESERVE,
    }
    if args.run_plan:
        tz = ZoneInfo(args.timezone)
        cycle_times = sorted({datetime.fromisoformat(call.local_time) for call in plan})
        if args.window_start:
            window_start = _parse_local_time(args.window_start)
            cycle_times = [scheduled for scheduled in cycle_times if scheduled.time() >= window_start]
        if args.window_end:
            window_end = _parse_local_time(args.window_end)
            cycle_times = [scheduled for scheduled in cycle_times if scheduled.time() <= window_end]
        for scheduled in cycle_times:
            now = datetime.now(tz)
            if scheduled < now - timedelta(minutes=2):
                print(json.dumps({"scheduled": scheduled.isoformat(), "status": "SKIPPED_PAST"}), flush=True)
                continue
            wait_seconds = max(0.0, (scheduled - now).total_seconds())
            if wait_seconds:
                print(
                    json.dumps({"scheduled": scheduled.isoformat(), "status": "WAITING", "seconds": round(wait_seconds)}),
                    flush=True,
                )
                time.sleep(wait_seconds)
            result = run_poll(client, **poll_kwargs)
            result["scheduled_local"] = scheduled.isoformat()
            print(json.dumps(result), flush=True)
    else:
        result = run_poll(client, **poll_kwargs)
        print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

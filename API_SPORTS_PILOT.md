# API-Sports NFL injury pilot — September 20, 2026

## Objective

Evaluate whether the API-Sports free plan updates defensive injury state quickly
enough to supplement official NFL/team reporting. API-Sports is a structured
signal source, not the final authority. New or changed statuses remain
`PROVISIONAL` until verified.

## Architecture

1. Poll the global `injuries` endpoint for current injury state.
2. Poll the date-level `games` endpoint for game status and clock context.
3. Save timestamped raw snapshots.
4. Compare the current injury list with the preceding snapshot.
5. Append additions, changes, and removals to `injury_change_ledger.jsonl`.
6. Treat disappearance from the current feed as `REQUIRES_CONFIRMATION`, never
   as an automatic return-to-play event.
7. After verification, map the event to both workbooks' `Defensive Injury Live`
   tabs. Spreadsheet delivery requires either a Google service account or an
   Apps Script webhook and is intentionally kept separate from the API key.

## Request budget

The pilot uses two calls per polling cycle. Standard cycles run every 15 minutes
from 11:45 AM through 10:45 PM America/Chicago. Extra pre-kickoff cycles run at
11:55 AM and 2:55 PM.

| Window (CT) | Cycles | Requests | Purpose |
|---|---:|---:|---|
| 11:45–11:55 | 2 | 4 | Baseline plus noon inactive sweep |
| 12:00–2:45 | 12 | 24 | Noon games |
| 2:55 | 1 | 2 | Late-window inactive sweep |
| 3:00–6:45 | 16 | 32 | Late games |
| 7:00–10:45 | 16 | 32 | Sunday-night game |
| **Total** | **47** | **94** | **Six-request reserve** |

The provider resets its daily quota at 00:00 UTC, which is 7:00 PM Central on
this date. The monitor nevertheless enforces the stricter 94-of-100 combined
local-day plan and keeps six calls unused for troubleshooting.

## Files

- `src/api_sports_injury_monitor.py` — planner, client, snapshotter and change detector
- `config/api_sports_request_plan_2026-09-20.csv` — exact call-level schedule
- `config/api_sports_pilot.env.example` — required environment names, without credentials
- `tests/test_api_sports_injury_monitor.py` — budget and change-detection tests

## Runbook

Create an API-Sports free-plan key and expose it as an environment variable. Do
not paste it into source code or the workbooks.

```bash
export API_SPORTS_KEY='your-key'
python3 src/api_sports_injury_monitor.py --date 2026-09-20 --once
```

After the first call confirms the league ID and response shapes, run the
remaining coordinated plan from a machine that will stay online:

```bash
python3 src/api_sports_injury_monitor.py --date 2026-09-20 --run-plan
```

Raw and normalized outputs are written beneath `data/live/api_sports/`. The
quota guard refuses a polling cycle when it would consume the six-request
reserve.

## GitHub Actions pilot

The workflow `.github/workflows/api-sports-injury-pilot.yml` divides the day
into noon, late, and night segments so each hosted job stays below GitHub's
maximum runtime. It starts each runner ten minutes early, restores the latest
pilot state from the Actions cache, and uploads the raw snapshots and ledgers
as a 14-day artifact.

Before the first segment, add `API_SPORTS_KEY` under **Settings → Secrets and
variables → Actions**. Scheduled executions are date-gated to September 20,
2026; future Sundays no-op. A manual dispatch remains available for a selected
segment. The cumulative ledger enforces 94 total calls even across the
provider's 7:00 PM Central quota reset.

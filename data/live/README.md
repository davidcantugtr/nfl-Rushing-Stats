# LIVE Google Sheets feeds

These files are the stable public feed layer for the permanent Google Sheets dashboard.

- `model_status.csv` — current week, refresh timestamp, scoring version, freshness state.
- `prime_rb_dashboard.csv` — ranked weekly Prime RB shortlist used by the LIVE dashboard.
- `defensive_freshness.csv` — Sharp defensive rank, verification date, current public defensive evidence, and model-source integrity notes.

The Thursday 3:00 AM America/Chicago automation should overwrite these LIVE files after validating the upcoming NFL slate, completed Q1 rushing data, injuries, depth-chart roles, and defensive strength. Preserve historical weekly outputs elsewhere in the repository; these files always represent the latest published state.

Integrity rule: never invent an inaccessible Sharp value. Keep the last verified Sharp rank, mark it stale, and separately identify any current public substitute used by the model.

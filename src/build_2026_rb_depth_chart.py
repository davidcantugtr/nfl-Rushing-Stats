from pathlib import Path

import pandas as pd
import nflreadpy as nfl

SEASON = 2026
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "processed"
OUT.mkdir(parents=True, exist_ok=True)


def to_pandas(frame):
    return frame.to_pandas() if hasattr(frame, "to_pandas") else pd.DataFrame(frame)


def main():
    dc = to_pandas(nfl.load_depth_charts(SEASON))
    required = {"dt", "team", "player_name", "gsis_id", "pos_abb", "pos_slot", "pos_rank"}
    missing = required.difference(dc.columns)
    if missing:
        raise KeyError(f"Missing expected depth-chart columns: {sorted(missing)}; columns={list(dc.columns)}")

    dc["dt"] = pd.to_datetime(dc["dt"], errors="coerce", utc=True)
    latest_dt = dc["dt"].max()
    latest = dc.loc[dc["dt"].eq(latest_dt)].copy()

    # Preserve all listed running backs from the latest league-wide snapshot.
    rb = latest.loc[latest["pos_abb"].astype(str).str.upper().eq("RB")].copy()
    rb["pos_slot"] = pd.to_numeric(rb["pos_slot"], errors="coerce")
    rb["pos_rank"] = pd.to_numeric(rb["pos_rank"], errors="coerce")
    rb = rb.sort_values(["team", "pos_slot", "pos_rank", "player_name"])
    rb["depth_snapshot_utc"] = latest_dt.isoformat() if pd.notna(latest_dt) else ""
    rb["depth_source"] = "nflverse depth charts via nflreadpy"

    cols = ["team", "player_name", "gsis_id", "pos_slot", "pos_rank", "depth_snapshot_utc", "depth_source"]
    rb[cols].to_csv(OUT / "rb_depth_chart_2026_latest.csv", index=False)

    # Also write a compact top-two view by team using depth-chart slot/rank order.
    compact = rb[cols].copy()
    compact["team_rb_order"] = compact.groupby("team").cumcount() + 1
    compact = compact.loc[compact["team_rb_order"] <= 3]
    compact.to_csv(OUT / "rb_depth_chart_2026_top3.csv", index=False)

    teams = rb["team"].nunique()
    if teams < 30:
        raise AssertionError(f"Expected broad league RB depth-chart coverage, found only {teams} teams")

    print(f"PASS: latest depth snapshot {latest_dt}; {len(rb)} RB rows across {teams} teams")


if __name__ == "__main__":
    main()

from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "processed"


def main():
    team = pd.read_csv(OUT / "team_rb_model_input_2025.csv")
    team_cols = [
        "team", "q1_rush_yds_mean", "q1_rush_yds_median", "q1_rush_yds_std",
        "q1_rush_yds_p25", "q1_rush_yds_p75", "q1_rush_yds_max",
        "q1_rush_yds_hit_20", "q1_rush_yds_hit_25", "q1_rush_yds_hit_30",
        "q1_rush_att_mean", "rb_carry_capture", "rb_yard_capture",
        "opp_q1_rush_yds_allowed_mean", "opp_q1_rush_yds_allowed_median",
        "opp_q1_rush_yds_allowed_p75", "opp_q1_rush_yds_allowed_hit_30",
        "opp_q1_rush_att_allowed_mean",
    ]
    team[team_cols].to_csv(OUT / "excel_team_inputs.csv", index=False)

    rb = pd.read_csv(OUT / "rb_player_model_input_2026.csv")
    # Prefer the current-team historical row when a player has multiple 2025 team rows; then the larger historical sample.
    rb["same_team_num"] = rb["same_team_2025_2026"].fillna(False).astype(int)
    rb["hist_games_num"] = pd.to_numeric(rb["q1_rush_yds_games"], errors="coerce").fillna(-1)
    rb = rb.sort_values(["team_depth", "team_rb_order", "same_team_num", "hist_games_num"], ascending=[True, True, False, False])
    rb = rb.drop_duplicates(["team_depth", "player_id", "team_rb_order"], keep="first")
    rb_cols = [
        "team_depth", "depth_player_name", "player_id", "team_rb_order",
        "q1_rush_yds_games", "q1_rush_yds_mean", "q1_rush_yds_median", "q1_rush_yds_p25",
        "q1_rush_yds_p75", "q1_rush_yds_max", "q1_rush_yds_hit_15", "q1_rush_yds_hit_20",
        "q1_rush_yds_hit_25", "q1_rush_att_mean", "rb_q1_carry_share_mean", "rb_q1_yard_share_mean",
        "same_team_2025_2026", "depth_default_room_share", "role_confidence", "depth_snapshot_utc",
    ]
    rb[rb_cols].to_csv(OUT / "excel_rb_inputs.csv", index=False)
    print(f"PASS: Excel exports {len(team)} team rows / {len(rb)} RB rows")


if __name__ == "__main__":
    main()

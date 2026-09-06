from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "processed"


def main():
    team_log = pd.read_csv(OUT / "team_q1_game_log_2025.csv")
    team_sum = pd.read_csv(OUT / "team_q1_summary_2025.csv")
    def_sum = pd.read_csv(OUT / "team_q1_defense_summary_2025.csv")
    rb_log = pd.read_csv(OUT / "rb_q1_game_log_2025.csv")
    rb_sum = pd.read_csv(OUT / "rb_q1_summary_2025.csv")
    depth = pd.read_csv(OUT / "rb_depth_chart_2026_top3.csv")

    team_tot = team_log.groupby("team", as_index=False).agg(
        team_q1_att=("q1_rush_att", "sum"),
        team_q1_yds=("q1_rush_yds", "sum"),
    )
    rb_tot = rb_log.groupby("team", as_index=False).agg(
        rb_q1_att=("q1_rush_att", "sum"),
        rb_q1_yds=("q1_rush_yds", "sum"),
    )
    capture = team_tot.merge(rb_tot, on="team", how="left").fillna({"rb_q1_att": 0, "rb_q1_yds": 0})
    capture["rb_carry_capture"] = capture["rb_q1_att"] / capture["team_q1_att"].where(capture["team_q1_att"].ne(0))
    capture["rb_yard_capture"] = capture["rb_q1_yds"] / capture["team_q1_yds"].where(capture["team_q1_yds"].ne(0))

    team_model = team_sum.merge(capture, on="team", how="left")
    def_cols = {c: f"opp_{c}" for c in def_sum.columns if c != "defense"}
    def_side = def_sum.rename(columns={"defense": "team", **def_cols})
    team_model = team_model.merge(def_side, on="team", how="left")
    team_model.to_csv(OUT / "team_rb_model_input_2025.csv", index=False)

    depth = depth.rename(columns={"gsis_id": "player_id", "player_name": "depth_player_name"})
    player_model = depth.merge(rb_sum, on="player_id", how="left", suffixes=("_depth", "_hist"))
    player_model["hist_data_available"] = player_model["q1_rush_yds_games"].notna()
    player_model["same_team_2025_2026"] = player_model["team_depth"].eq(player_model["team_hist"])
    player_model["depth_default_room_share"] = player_model["team_rb_order"].map({1: 0.68, 2: 0.25, 3: 0.07}).fillna(0.0)
    player_model["role_confidence"] = 0.55
    player_model.loc[player_model["hist_data_available"], "role_confidence"] = 0.72
    player_model.loc[player_model["same_team_2025_2026"] & player_model["q1_rush_yds_games"].fillna(0).ge(10), "role_confidence"] = 0.90
    player_model.loc[player_model["same_team_2025_2026"] & player_model["q1_rush_yds_games"].fillna(0).ge(15), "role_confidence"] = 0.97
    player_model.to_csv(OUT / "rb_player_model_input_2026.csv", index=False)

    print(f"PASS: wrote team model inputs for {len(team_model)} teams")
    print(f"PASS: wrote current depth/history model inputs for {len(player_model)} players")


if __name__ == "__main__":
    main()

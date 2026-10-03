from __future__ import annotations

from math import erf, sqrt
from pathlib import Path
import re

import numpy as np
import pandas as pd


SEASON = 2026
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "processed"
LIVE = ROOT / "data" / "live"
PBP_URL = f"https://github.com/nflverse/nflverse-data/releases/download/pbp/play_by_play_{SEASON}.parquet"
ROSTER_URL = f"https://github.com/nflverse/nflverse-data/releases/download/rosters/roster_{SEASON}.csv"

WINDOWS = {
    "Q1": {"quarters": {1}, "benchmark": 15.0},
    "1H": {"quarters": {1, 2}, "benchmark": 35.0},
    "Full Game": {"quarters": {1, 2, 3, 4, 5}, "benchmark": 60.0},
}


def flag(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").fillna(0).eq(1)


def clean_name(value: object) -> str:
    text = re.sub(r"[^a-z0-9 ]", "", str(value).lower())
    return re.sub(r"\b(jr|sr|ii|iii|iv)\b", "", text).strip()


def normal_over(mean: float, std: float, line: float) -> float:
    sigma = max(float(std) if pd.notna(std) else 0.0, max(6.0, abs(float(mean)) * 0.30))
    z = (float(line) - float(mean)) / sigma
    return float(0.5 * (1.0 - erf(z / sqrt(2.0))))


def summarize(group: pd.DataFrame, name: str) -> dict:
    yards = pd.to_numeric(group["rush_yds"], errors="coerce").fillna(0.0)
    attempts = pd.to_numeric(group["rush_att"], errors="coerce").fillna(0.0)
    return {
        f"{name}_games": int(len(group)),
        f"{name}_mean": float(yards.mean()),
        f"{name}_median": float(yards.median()),
        f"{name}_std": float(yards.std(ddof=0)),
        f"{name}_min": float(yards.min()),
        f"{name}_p25": float(yards.quantile(0.25)),
        f"{name}_p75": float(yards.quantile(0.75)),
        f"{name}_max": float(yards.max()),
        "rush_att_mean": float(attempts.mean()),
        "rush_att_median": float(attempts.median()),
    }


def load_current_model() -> tuple[pd.DataFrame, int]:
    current = pd.read_csv(LIVE / "current_week_model.csv")
    target_week = int(pd.to_numeric(current["week"], errors="raise").max())
    if current["week"].nunique() != 1:
        raise AssertionError("current_week_model must contain exactly one target week")
    current["player_key"] = current["player"].map(clean_name)
    return current, target_week


def build_current_history(target_week: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, int]:
    pbp = pd.read_parquet(PBP_URL)
    rosters = pd.read_csv(ROSTER_URL, low_memory=False)
    reg = pbp.loc[pbp["season_type"].astype(str).eq("REG")].copy()
    if reg.empty:
        raise AssertionError(f"No {SEASON} regular-season play-by-play was available")

    games = reg[["game_id", "week", "home_team", "away_team"]].drop_duplicates("game_id")
    completed_week = int(pd.to_numeric(games["week"], errors="coerce").max())
    home = games.assign(team=games.home_team, opponent=games.away_team, site="H")
    away = games.assign(team=games.away_team, opponent=games.home_team, site="A")
    skeleton = pd.concat([home, away], ignore_index=True)[["game_id", "week", "team", "opponent", "site"]]

    rush = reg.loc[flag(reg["rush_attempt"])].copy()
    if "no_play" in rush:
        rush = rush.loc[~flag(rush["no_play"])].copy()
    if "two_point_attempt" in rush:
        rush = rush.loc[~flag(rush["two_point_attempt"])].copy()
    rush["rushing_yards"] = pd.to_numeric(rush["rushing_yards"], errors="coerce").fillna(0.0)
    rush["rusher_player_id"] = rush["rusher_player_id"].astype(str)

    id_col = "gsis_id" if "gsis_id" in rosters.columns else "player_id"
    name_col = next(c for c in ["full_name", "player_name", "football_name"] if c in rosters.columns)
    roster_week = pd.to_numeric(rosters.get("week", 0), errors="coerce").fillna(0)
    positions = rosters.assign(_week=roster_week).sort_values("_week")
    positions = positions[[id_col, name_col, "position"]].dropna(subset=[id_col]).copy()
    positions[id_col] = positions[id_col].astype(str)
    positions = positions.drop_duplicates(id_col, keep="last")

    rusher_games = rush[[
        "game_id", "week", "posteam", "defteam", "rusher_player_id", "rusher_player_name"
    ]].drop_duplicates(["game_id", "rusher_player_id"])
    rusher_games = rusher_games.merge(positions, left_on="rusher_player_id", right_on=id_col, how="left")
    rb_games = rusher_games.loc[
        rusher_games["position"].fillna("UNKNOWN").str.upper().isin(["RB", "FB"])
    ].rename(columns={
        "posteam": "team", "defteam": "opponent", "rusher_player_id": "player_id",
        "rusher_player_name": "pbp_player_name", name_col: "player_name",
    })
    rb_games["player_name"] = rb_games["player_name"].fillna(rb_games["pbp_player_name"])

    team_logs: list[pd.DataFrame] = []
    rb_logs: list[pd.DataFrame] = []
    for window, config in WINDOWS.items():
        wrush = rush.loc[pd.to_numeric(rush["qtr"], errors="coerce").fillna(0).isin(config["quarters"])].copy()
        team_actual = wrush.groupby(["game_id", "posteam"], as_index=False).agg(
            rush_att=("rush_attempt", "size"), rush_yds=("rushing_yards", "sum")
        ).rename(columns={"posteam": "team"})
        team_log = skeleton.merge(team_actual, on=["game_id", "team"], how="left")
        team_log[["rush_att", "rush_yds"]] = team_log[["rush_att", "rush_yds"]].fillna(0)
        team_log["rush_att"] = team_log["rush_att"].astype(int)
        team_log.insert(0, "season", SEASON)
        team_log.insert(4, "window", window)
        team_logs.append(team_log)

        rb_actual = wrush.groupby(["game_id", "rusher_player_id"], as_index=False).agg(
            rush_att=("rush_attempt", "size"), rush_yds=("rushing_yards", "sum")
        ).rename(columns={"rusher_player_id": "player_id"})
        rb_log = rb_games.merge(rb_actual, on=["game_id", "player_id"], how="left")
        rb_log[["rush_att", "rush_yds"]] = rb_log[["rush_att", "rush_yds"]].fillna(0)
        rb_log["rush_att"] = rb_log["rush_att"].astype(int)
        keys = team_log[["game_id", "team", "site", "rush_att", "rush_yds"]].rename(columns={
            "rush_att": "team_rush_att", "rush_yds": "team_rush_yds"
        })
        rb_log = rb_log.merge(keys, on=["game_id", "team"], how="left")
        rb_log["carry_share"] = np.where(
            rb_log["team_rush_att"].gt(0), rb_log["rush_att"] / rb_log["team_rush_att"], 0.0
        )
        rb_log["yard_share"] = np.where(
            rb_log["team_rush_yds"].ne(0), rb_log["rush_yds"] / rb_log["team_rush_yds"], 0.0
        )
        rb_log.insert(0, "season", SEASON)
        rb_log.insert(4, "window", window)
        rb_logs.append(rb_log)

    team_log_all = pd.concat(team_logs, ignore_index=True)
    rb_log_all = pd.concat(rb_logs, ignore_index=True)
    model_team = team_log_all.loc[team_log_all["week"].lt(target_week)].copy()
    model_rb = rb_log_all.loc[rb_log_all["week"].lt(target_week)].copy()
    if model_team.empty or model_rb.empty:
        raise AssertionError(f"No completed games exist before target Week {target_week}")

    team_summary_rows = []
    defense_summary_rows = []
    for (team, window), group in model_team.groupby(["team", "window"]):
        team_summary_rows.append({"team": team, "window": window, **summarize(group, "rush_yds")})
    allowed = model_team.rename(columns={"team": "offense", "opponent": "defense"})
    for (defense, window), group in allowed.groupby(["defense", "window"]):
        values = summarize(group, "rush_yds_allowed")
        defense_summary_rows.append({"defense": defense, "window": window, **values})

    rb_summary_rows = []
    for (player_id, player_name, team, window), group in model_rb.groupby([
        "player_id", "player_name", "team", "window"
    ]):
        row = {
            "player_id": player_id, "player_name": player_name, "player_key": clean_name(player_name),
            "team": team, "window": window, **summarize(group, "rush_yds"),
            "carry_share_mean": float(group["carry_share"].mean()),
            "yard_share_mean": float(group["yard_share"].mean()),
        }
        rb_summary_rows.append(row)

    team_summary = pd.DataFrame(team_summary_rows)
    defense_summary = pd.DataFrame(defense_summary_rows)
    rb_summary = pd.DataFrame(rb_summary_rows)
    team_log_all.to_csv(OUT / "team_rushing_windows_game_log_2026.csv", index=False)
    rb_log_all.to_csv(OUT / "rb_rushing_windows_game_log_2026.csv", index=False)
    team_summary.to_csv(OUT / f"team_rushing_windows_summary_2026_through_week_{target_week - 1}.csv", index=False)
    defense_summary.to_csv(OUT / f"team_rushing_windows_defense_summary_2026_through_week_{target_week - 1}.csv", index=False)
    rb_summary.to_csv(OUT / f"rb_rushing_windows_summary_2026_through_week_{target_week - 1}.csv", index=False)
    return team_log_all, rb_log_all, team_summary, defense_summary, rb_summary, completed_week


def one_row(frame: pd.DataFrame, **conditions: object) -> pd.Series | None:
    subset = frame
    for column, value in conditions.items():
        subset = subset.loc[subset[column].eq(value)]
    return None if subset.empty else subset.iloc[0]


def build_live_board(
    current: pd.DataFrame,
    team_summary: pd.DataFrame,
    defense_summary: pd.DataFrame,
    rb_summary: pd.DataFrame,
    target_week: int,
) -> pd.DataFrame:
    rows = []
    for _, player in current.iterrows():
        rank = int(player.get("rank", 99))
        sharp_rank = int(player.get("sharp_rush_eff_rank", player.get("sharp_def_rank", 16)))
        for window, config in WINDOWS.items():
            player_hist = one_row(
                rb_summary, player_key=player["player_key"], team=player["team"], window=window
            )
            if player_hist is None:
                any_team = rb_summary.loc[
                    rb_summary["player_key"].eq(player["player_key"]) & rb_summary["window"].eq(window)
                ]
                player_hist = None if any_team.empty else any_team.iloc[0]
            team_hist = one_row(team_summary, team=player["team"], window=window)
            defense_hist = one_row(defense_summary, defense=player["opponent"], window=window)

            hist_mean = float(player_hist["rush_yds_mean"]) if player_hist is not None else np.nan
            hist_std = float(player_hist["rush_yds_std"]) if player_hist is not None else np.nan
            hist_carries = float(player_hist["rush_att_mean"]) if player_hist is not None else np.nan
            carry_share = float(player_hist["carry_share_mean"]) if player_hist is not None else np.nan

            if window == "Q1":
                baseline = float(player.get("neutral_q1_yds_baseline", hist_mean))
                carries = hist_carries if pd.notna(hist_carries) else float(player.get("espn_w4_carries", 0)) / 4.0
            elif window == "1H":
                baseline = hist_mean if pd.notna(hist_mean) else float(player.get("neutral_q1_yds_baseline", 0)) * 2.0
                carries = hist_carries if pd.notna(hist_carries) else float(player.get("espn_w4_carries", 0)) / 2.0
            else:
                external = float(player.get("espn_w4_rush_yds", np.nan))
                baseline = 0.65 * hist_mean + 0.35 * external if pd.notna(hist_mean) and pd.notna(external) else (
                    hist_mean if pd.notna(hist_mean) else external
                )
                carries = hist_carries if pd.notna(hist_carries) else float(player.get("espn_w4_carries", 0))

            offense_mean = float(team_hist["rush_yds_mean"]) if team_hist is not None else np.nan
            defense_mean = float(defense_hist["rush_yds_allowed_mean"]) if defense_hist is not None else np.nan
            team_yards = np.nanmean([offense_mean, defense_mean])
            if pd.isna(team_yards):
                team_yards = baseline / max(carry_share if pd.notna(carry_share) else 0.55, 0.25)
            if pd.isna(carry_share):
                carry_share = min(0.90, max(0.25, baseline / max(team_yards, 1.0)))

            trailing = baseline * {"Q1": 0.94, "1H": 0.88, "Full Game": 0.78}[window]
            leading = baseline * {"Q1": 1.06, "1H": 1.10, "Full Game": 1.18}[window]
            adjusted = 0.30 * trailing + 0.40 * baseline + 0.30 * leading
            sigma = max(hist_std if pd.notna(hist_std) else 0.0, max(6.0, adjusted * 0.32))
            benchmark = float(config["benchmark"])
            probability = normal_over(adjusted, sigma, benchmark)
            score = max(0.0, min(100.0, 94.0 - 2.0 * (rank - 1)))
            tier = "ELITE" if score >= 86 else "PRIME" if score >= 78 else "STRONG" if score >= 68 else "WATCH"
            rows.append({
                "week": target_week, "team": player["team"], "opponent": player["opponent"],
                "site": player.get("site", ""), "player": player["player"],
                "depth_role": player.get("depth_role", "Tracked RB"),
                "backfield_type": player.get("backfield_type", "Current role"), "window": window,
                "sharp_rush_def_rank": sharp_rank, "carry_share": carry_share,
                "base_team_rush_yds": team_yards, "base_player_carries": carries,
                "base_player_rush_yds": baseline, "team_spread": np.nan, "game_script": "UNKNOWN",
                "trailing_probability": 0.30, "neutral_probability": 0.40, "leading_probability": 0.30,
                "trailing_rush_yds": trailing, "neutral_rush_yds": baseline,
                "leading_rush_yds": leading, "script_adjusted_rush_yds": adjusted,
                "floor": max(0.0, adjusted - 0.75 * sigma), "ceiling": adjusted + 0.75 * sigma,
                "model_benchmark_yards": benchmark, "model_benchmark_probability": probability,
                "probability_source": f"2026 play-by-play through Week {target_week - 1}",
                "sportsbook_market": "", "consensus_line": np.nan,
                "sportsbook_over_probability": np.nan, "model_over_probability": np.nan,
                "probability_edge": np.nan, "sportsbook_count": 0,
                "odds_status": "MODEL ONLY — WINDOW MARKET UNAVAILABLE" if window != "Full Game" else "AWAITING FULL-GAME LINE",
                "opportunity_score": score, "tier": tier,
                "history_games": int(player_hist["rush_yds_games"]) if player_hist is not None else 0,
                "history_source": f"Exact {SEASON} play-by-play before Week {target_week}",
                "availability": player.get("availability", "IN"),
            })
    board = pd.DataFrame(rows).sort_values(["window", "opportunity_score"], ascending=[True, False])
    board.to_csv(LIVE / "rb_window_opportunity_board.csv", index=False, float_format="%.6f")
    return board


def write_prime_dashboard(current: pd.DataFrame, board: pd.DataFrame) -> None:
    q1 = board.loc[board["window"].eq("Q1"), [
        "player", "script_adjusted_rush_yds", "floor", "ceiling", "history_games", "tier"
    ]]
    dashboard = current.merge(q1, on="player", how="left")
    dashboard["model_note"] = (
        dashboard["tier"].fillna("WATCH") + "; exact 2026 Q1 PBP through prior completed week"
    )
    keep = [
        "week", "rank", "player", "team", "opponent", "sharp_rush_eff_rank",
        "season_carries", "season_rush_yds", "espn_w4_carries", "espn_w4_rush_yds",
        "neutral_q1_yds_baseline", "script_adjusted_rush_yds", "floor", "ceiling",
        "history_games", "status", "availability", "model_note",
    ]
    dashboard[[c for c in keep if c in dashboard.columns]].to_csv(LIVE / "prime_rb_dashboard.csv", index=False)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    LIVE.mkdir(parents=True, exist_ok=True)
    current, target_week = load_current_model()
    _, _, team_summary, defense_summary, rb_summary, feed_week = build_current_history(target_week)
    board = build_live_board(current, team_summary, defense_summary, rb_summary, target_week)
    write_prime_dashboard(current, board)
    if not board["week"].eq(target_week).all():
        raise AssertionError("Live board contains a stale target week")
    q1 = board.loc[board["window"].eq("Q1")]
    if q1["history_source"].str.contains(str(SEASON)).all() is False:
        raise AssertionError("Q1 rows are not sourced from current-season play-by-play")
    print(f"PASS: exact {SEASON} play-by-play ingested through provider Week {feed_week}")
    print(f"PASS: projections use completed Weeks 1-{target_week - 1}; partial Week {target_week} excluded")
    print(f"PASS: {len(board)} live player-window rows for target Week {target_week}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

from __future__ import annotations

from math import erf, exp, sqrt
from pathlib import Path
import re

import numpy as np
import pandas as pd


SEASON = 2025
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "processed"
LIVE = ROOT / "data" / "live"
OUT.mkdir(parents=True, exist_ok=True)
LIVE.mkdir(parents=True, exist_ok=True)

PROPS_URL = (
    "https://raw.githubusercontent.com/davidcantugtr/"
    "nfl-player-prop-opportunity/main/data/latest/player_props.csv"
)
LINES_URL = (
    "https://raw.githubusercontent.com/davidcantugtr/"
    "nfl-player-prop-opportunity/main/data/latest/game_lines.csv"
)
PBP_URL = "https://github.com/nflverse/nflverse-data/releases/download/pbp/play_by_play_2025.parquet"
ROSTER_URL = "https://github.com/nflverse/nflverse-data/releases/download/rosters/roster_2025.csv"

WINDOWS = {
    "Q1": {"quarters": {1}, "team_thresholds": [20, 25, 30, 35, 40], "rb_thresholds": [15, 20, 25]},
    "1H": {"quarters": {1, 2}, "team_thresholds": [35, 45, 55, 65, 75], "rb_thresholds": [25, 35, 45, 55]},
    "Full Game": {"quarters": {1, 2, 3, 4, 5}, "team_thresholds": [50, 60, 70, 80, 100, 120], "rb_thresholds": [40, 50, 60, 70, 80, 100]},
}

# Must remain identical to the validated Q1 pipeline.
OFFICIAL_STAT_CORRECTIONS = {
    ("2025_07_IND_LAC", 641.0): 3.0,
}

TEAM_NAMES = {
    "ARI": "Arizona Cardinals", "ATL": "Atlanta Falcons", "BAL": "Baltimore Ravens",
    "BUF": "Buffalo Bills", "CAR": "Carolina Panthers", "CHI": "Chicago Bears",
    "CIN": "Cincinnati Bengals", "CLE": "Cleveland Browns", "DAL": "Dallas Cowboys",
    "DEN": "Denver Broncos", "DET": "Detroit Lions", "GB": "Green Bay Packers",
    "HOU": "Houston Texans", "IND": "Indianapolis Colts", "JAX": "Jacksonville Jaguars",
    "KC": "Kansas City Chiefs", "LA": "Los Angeles Rams", "LAC": "Los Angeles Chargers",
    "LV": "Las Vegas Raiders", "MIA": "Miami Dolphins", "MIN": "Minnesota Vikings",
    "NE": "New England Patriots", "NO": "New Orleans Saints", "NYG": "New York Giants",
    "NYJ": "New York Jets", "PHI": "Philadelphia Eagles", "PIT": "Pittsburgh Steelers",
    "SEA": "Seattle Seahawks", "SF": "San Francisco 49ers", "TB": "Tampa Bay Buccaneers",
    "TEN": "Tennessee Titans", "WAS": "Washington Commanders",
}


def flag(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").fillna(0).eq(1)


def clean_name(value: str) -> str:
    text = re.sub(r"[^a-z0-9 ]", "", str(value).lower())
    return re.sub(r"\b(jr|sr|ii|iii|iv)\b", "", text).strip()


def summary(series: pd.Series, prefix: str, thresholds: list[int]) -> dict:
    s = pd.to_numeric(series, errors="coerce").fillna(0.0).astype(float)
    result = {
        f"{prefix}_games": int(len(s)),
        f"{prefix}_mean": float(s.mean()),
        f"{prefix}_median": float(s.median()),
        f"{prefix}_std": float(s.std(ddof=0)),
        f"{prefix}_min": float(s.min()),
        f"{prefix}_p25": float(s.quantile(0.25)),
        f"{prefix}_p75": float(s.quantile(0.75)),
        f"{prefix}_max": float(s.max()),
    }
    for threshold in thresholds:
        result[f"{prefix}_hit_{threshold}"] = float((s >= threshold).mean())
    return result


def no_vig_probability_from_history(mean: float, std: float, line: float) -> float:
    if pd.isna(mean) or pd.isna(line):
        return np.nan
    sigma = max(float(std) if pd.notna(std) else 0.0, max(8.0, abs(float(mean)) * 0.28))
    z = (float(line) - float(mean)) / sigma
    return float(0.5 * (1.0 - erf(z / sqrt(2.0))))


def script_label(spread: float) -> str:
    if pd.isna(spread):
        return "UNKNOWN"
    if spread <= -6:
        return "STRONG FAVORITE"
    if spread <= -2.5:
        return "FAVORITE"
    if spread < 2.5:
        return "NEUTRAL"
    if spread < 6:
        return "UNDERDOG"
    return "LARGE UNDERDOG"


def script_probabilities(spread: float, window: str) -> tuple[float, float, float]:
    if pd.isna(spread):
        return 0.33, 0.34, 0.33
    scale = {"Q1": 11.0, "1H": 8.5, "Full Game": 6.5}[window]
    lead = 1.0 / (1.0 + exp(float(spread) / scale))
    neutral = {"Q1": 0.34, "1H": 0.25, "Full Game": 0.16}[window] * exp(-abs(float(spread)) / 7.0)
    remain = 1.0 - neutral
    lead *= remain
    trail = remain - lead
    return trail, neutral, lead


def load_consensus() -> tuple[pd.DataFrame, pd.DataFrame]:
    try:
        props = pd.read_csv(PROPS_URL)
        props = props.loc[
            props["Position"].eq("RB")
            & props["Market Key"].eq("player_rush_yds")
            & props["Market Window"].eq("Full Game")
            & props["Side"].eq("Over")
        ].copy()
        props["player_key"] = props["Player"].map(clean_name)
        consensus = props.groupby("player_key", as_index=False).agg(
            sportsbook_player=("Player", "first"),
            consensus_line=("Line", "mean"),
            consensus_over_probability=("No-Vig Implied", "mean"),
            sportsbook_count=("Book Key", "nunique"),
            odds_last_update=("Last Update", "max"),
        )
    except Exception as exc:
        print(f"WARNING: sportsbook player props unavailable: {exc}")
        consensus = pd.DataFrame(columns=[
            "player_key", "sportsbook_player", "consensus_line",
            "consensus_over_probability", "sportsbook_count", "odds_last_update",
        ])

    try:
        lines = pd.read_csv(LINES_URL)
        spread_rows = lines.loc[lines["Market"].eq("spreads")].copy()
        spread_rows["team_name"] = spread_rows["Team / Side"].astype(str)
        game_consensus = spread_rows.groupby(["Week", "team_name"], as_index=False).agg(
            consensus_team_spread=("Line", "mean"),
            game_line_books=("Bookmaker", "nunique"),
            game_line_last_update=("Last Update", "max"),
        )
    except Exception as exc:
        print(f"WARNING: game lines unavailable: {exc}")
        game_consensus = pd.DataFrame(columns=[
            "Week", "team_name", "consensus_team_spread", "game_line_books", "game_line_last_update"
        ])
    return consensus, game_consensus


def build_history() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    # Read the same nflverse release artifacts directly. This keeps the window
    # extension independent of a particular nflreadpy/polars runtime while
    # preserving the current pipeline's source of truth.
    pbp = pd.read_parquet(PBP_URL)
    rosters = pd.read_csv(ROSTER_URL, low_memory=False)
    reg = pbp.loc[pbp["season_type"].astype(str).eq("REG")].copy()
    games = reg[["game_id", "week", "home_team", "away_team"]].drop_duplicates("game_id")
    if len(games) != 272:
        raise AssertionError(f"Expected 272 regular-season games, got {len(games)}")
    home = games.assign(team=games.home_team, opponent=games.away_team, site="H")
    away = games.assign(team=games.away_team, opponent=games.home_team, site="A")
    skeleton = pd.concat([home, away], ignore_index=True)[["game_id", "week", "team", "opponent", "site"]]

    rush = reg.loc[flag(reg["rush_attempt"])].copy()
    if "no_play" in rush:
        rush = rush.loc[~flag(rush["no_play"])].copy()
    if "two_point_attempt" in rush:
        rush = rush.loc[~flag(rush["two_point_attempt"])].copy()
    rush["rushing_yards"] = pd.to_numeric(rush["rushing_yards"], errors="coerce").fillna(0.0)
    for (game_id, play_id), corrected_yards in OFFICIAL_STAT_CORRECTIONS.items():
        mask = rush["game_id"].eq(game_id) & pd.to_numeric(rush["play_id"], errors="coerce").eq(play_id)
        if mask.sum() != 1:
            raise AssertionError(f"Expected one correction target for {game_id}/{play_id}, found {mask.sum()}")
        rush.loc[mask, "rushing_yards"] = corrected_yards

    id_col = "gsis_id" if "gsis_id" in rosters.columns else "player_id"
    name_col = next(c for c in ["full_name", "player_name", "football_name"] if c in rosters.columns)
    positions = rosters[[id_col, name_col, "position"]].dropna(subset=[id_col]).copy()
    positions[id_col] = positions[id_col].astype(str)
    positions = positions.drop_duplicates(id_col, keep="last")
    rush["rusher_player_id"] = rush["rusher_player_id"].astype(str)
    rusher_games = rush[["game_id", "week", "posteam", "defteam", "rusher_player_id", "rusher_player_name"]].drop_duplicates(["game_id", "rusher_player_id"])
    rusher_games = rusher_games.merge(positions, left_on="rusher_player_id", right_on=id_col, how="left")
    rb_games = rusher_games.loc[rusher_games["position"].fillna("UNKNOWN").str.upper().isin(["RB", "FB"])].rename(columns={
        "posteam": "team", "defteam": "opponent", "rusher_player_id": "player_id",
        "rusher_player_name": "pbp_player_name", name_col: "player_name",
    })

    team_logs = []
    rb_logs = []
    team_summaries = []
    defense_summaries = []
    rb_summaries = []
    for window, cfg in WINDOWS.items():
        key = {"Q1": "q1", "1H": "h1", "Full Game": "full_game"}[window]
        wrush = rush.loc[pd.to_numeric(rush["qtr"], errors="coerce").fillna(0).isin(cfg["quarters"])].copy()
        team_actual = wrush.groupby(["game_id", "posteam"], as_index=False).agg(
            rush_att=("rush_attempt", "size"), rush_yds=("rushing_yards", "sum")
        ).rename(columns={"posteam": "team"})
        team_log = skeleton.merge(team_actual, on=["game_id", "team"], how="left")
        team_log[["rush_att", "rush_yds"]] = team_log[["rush_att", "rush_yds"]].fillna(0)
        team_log["rush_att"] = team_log["rush_att"].astype(int)
        team_log["ypc"] = np.where(team_log["rush_att"].gt(0), team_log["rush_yds"] / team_log["rush_att"], 0.0)
        team_log.insert(3, "window", window)
        team_logs.append(team_log)

        for team, group in team_log.groupby("team"):
            team_summaries.append({
                "team": team, "window": window,
                **summary(group["rush_yds"], "rush_yds", cfg["team_thresholds"]),
                "rush_att_mean": group["rush_att"].mean(),
                "rush_att_median": group["rush_att"].median(),
                "ypc_mean": group["rush_yds"].sum() / max(1, group["rush_att"].sum()),
            })
        allowed = team_log.rename(columns={"team": "offense", "opponent": "defense"})
        for defense, group in allowed.groupby("defense"):
            defense_summaries.append({
                "defense": defense, "window": window,
                **summary(group["rush_yds"], "rush_yds_allowed", cfg["team_thresholds"]),
                "rush_att_allowed_mean": group["rush_att"].mean(),
                "ypc_allowed": group["rush_yds"].sum() / max(1, group["rush_att"].sum()),
            })

        rb_actual = wrush.groupby(["game_id", "rusher_player_id"], as_index=False).agg(
            rush_att=("rush_attempt", "size"), rush_yds=("rushing_yards", "sum")
        ).rename(columns={"rusher_player_id": "player_id"})
        rb_log = rb_games.merge(rb_actual, on=["game_id", "player_id"], how="left")
        rb_log[["rush_att", "rush_yds"]] = rb_log[["rush_att", "rush_yds"]].fillna(0)
        rb_log["rush_att"] = rb_log["rush_att"].astype(int)
        team_keys = team_log[["game_id", "team", "site", "rush_att", "rush_yds"]].rename(columns={
            "rush_att": "team_rush_att", "rush_yds": "team_rush_yds"
        })
        rb_log = rb_log.merge(team_keys, on=["game_id", "team"], how="left")
        rb_log["carry_share"] = np.where(rb_log["team_rush_att"].gt(0), rb_log["rush_att"] / rb_log["team_rush_att"], 0.0)
        rb_log["yard_share"] = np.where(rb_log["team_rush_yds"].ne(0), rb_log["rush_yds"] / rb_log["team_rush_yds"], 0.0)
        rb_log.insert(3, "window", window)
        rb_logs.append(rb_log)
        for (player_id, player_name, team), group in rb_log.groupby(["player_id", "player_name", "team"]):
            rb_summaries.append({
                "player_id": player_id, "player_name": player_name, "team": team, "window": window,
                **summary(group["rush_yds"], "rush_yds", cfg["rb_thresholds"]),
                "rush_att_mean": group["rush_att"].mean(),
                "rush_att_median": group["rush_att"].median(),
                "carry_share_mean": group["carry_share"].mean(),
                "yard_share_mean": group["yard_share"].mean(),
            })

    team_log_all = pd.concat(team_logs, ignore_index=True)
    rb_log_all = pd.concat(rb_logs, ignore_index=True)
    team_summary = pd.DataFrame(team_summaries)
    defense_summary = pd.DataFrame(defense_summaries)
    rb_summary = pd.DataFrame(rb_summaries)
    team_log_all.to_csv(OUT / "team_rushing_windows_game_log_2025.csv", index=False)
    rb_log_all.to_csv(OUT / "rb_rushing_windows_game_log_2025.csv", index=False)
    team_summary.to_csv(OUT / "team_rushing_windows_summary_2025.csv", index=False)
    defense_summary.to_csv(OUT / "team_rushing_windows_defense_summary_2025.csv", index=False)
    rb_summary.to_csv(OUT / "rb_rushing_windows_summary_2025.csv", index=False)
    return team_summary, defense_summary, rb_summary


def build_live_board(team_summary: pd.DataFrame, defense_summary: pd.DataFrame, rb_summary: pd.DataFrame) -> pd.DataFrame:
    current = pd.read_csv(LIVE / "current_week_model.csv")
    props, lines = load_consensus()
    current["player_key"] = current["player"].map(clean_name)
    current["team_name"] = current["team"].map(TEAM_NAMES)
    current = current.merge(lines, left_on=["week", "team_name"], right_on=["Week", "team_name"], how="left")
    rows = []
    script_mult = {
        "Q1": {"Trailing": 0.94, "Neutral": 1.00, "Leading": 1.06},
        "1H": {"Trailing": 0.88, "Neutral": 1.00, "Leading": 1.10},
        "Full Game": {"Trailing": 0.78, "Neutral": 1.00, "Leading": 1.18},
    }
    for _, player in current.iterrows():
        for window in WINDOWS:
            team_hist = team_summary.loc[(team_summary["team"].eq(player["team"])) & team_summary["window"].eq(window)]
            opp_hist = defense_summary.loc[(defense_summary["defense"].eq(player["opponent"])) & defense_summary["window"].eq(window)]
            rb_hist = rb_summary.loc[
                rb_summary["player_name"].map(clean_name).eq(player["player_key"])
                & rb_summary["window"].eq(window)
            ]
            if window == "Q1":
                base_team_yards = float(player["proj_team_q1_yds"])
                base_player_yards = float(player["exp_player_q1_yds"])
                base_carries = float(player["exp_q1_carries"])
                hist_std = float(rb_hist["rush_yds_std"].iloc[0]) if len(rb_hist) else max(8.0, base_player_yards * 0.35)
            else:
                offense_yards = float(team_hist["rush_yds_mean"].iloc[0]) if len(team_hist) else np.nan
                defense_yards = float(opp_hist["rush_yds_allowed_mean"].iloc[0]) if len(opp_hist) else np.nan
                base_team_yards = 0.55 * offense_yards + 0.45 * defense_yards
                sharp_adjust = 1.0 + max(-0.10, min(0.10, (float(player["sharp_def_rank"]) - 16.5) / 155.0))
                base_team_yards *= sharp_adjust
                hist_carry_share = float(rb_hist["carry_share_mean"].iloc[0]) if len(rb_hist) else float(player["final_carry_share"])
                hist_yard_share = float(rb_hist["yard_share_mean"].iloc[0]) if len(rb_hist) else float(player["final_carry_share"])
                carry_share = 0.70 * float(player["final_carry_share"]) + 0.30 * hist_carry_share
                yard_share = 0.70 * float(player["final_carry_share"]) + 0.30 * hist_yard_share
                team_carries = 0.55 * float(team_hist["rush_att_mean"].iloc[0]) + 0.45 * float(opp_hist["rush_att_allowed_mean"].iloc[0])
                base_carries = team_carries * carry_share
                base_player_yards = base_team_yards * yard_share
                hist_std = float(rb_hist["rush_yds_std"].iloc[0]) if len(rb_hist) else max(10.0, base_player_yards * 0.38)

            spread = player.get("consensus_team_spread", np.nan)
            trail_p, neutral_p, lead_p = script_probabilities(spread, window)
            trailing = base_player_yards * script_mult[window]["Trailing"]
            neutral = base_player_yards
            leading = base_player_yards * script_mult[window]["Leading"]
            script_adjusted = trail_p * trailing + neutral_p * neutral + lead_p * leading
            floor = max(0.0, script_adjusted - 0.75 * hist_std)
            ceiling = script_adjusted + 0.75 * hist_std

            prop = props.loc[props["player_key"].eq(player["player_key"])] if window == "Full Game" else props.iloc[0:0]
            line = float(prop["consensus_line"].iloc[0]) if len(prop) else np.nan
            book_prob = float(prop["consensus_over_probability"].iloc[0]) if len(prop) else np.nan
            books = int(prop["sportsbook_count"].iloc[0]) if len(prop) else 0
            model_prob = no_vig_probability_from_history(script_adjusted, hist_std, line) if len(prop) else np.nan
            edge = model_prob - book_prob if pd.notna(model_prob) and pd.notna(book_prob) else np.nan
            status = "LIVE CONSENSUS" if len(prop) else (
                "MODEL ONLY — WINDOW MARKET UNAVAILABLE" if window in ["Q1", "1H"] else "AWAITING FULL-GAME LINE"
            )
            base_score = float(player["v05_candidate_score"])
            script_bonus = (lead_p - trail_p) * {"Q1": 4.0, "1H": 7.0, "Full Game": 10.0}[window]
            edge_bonus = max(-10.0, min(10.0, (edge if pd.notna(edge) else 0.0) * 100.0))
            score = max(0.0, min(100.0, base_score + script_bonus + edge_bonus))
            tier = "ELITE" if score >= 82 else "PRIME" if score >= 74 else "STRONG" if score >= 66 else "WATCH" if score >= 58 else "PASS"
            rows.append({
                "week": int(player["week"]), "team": player["team"], "opponent": player["opponent"],
                "site": player["site"], "player": player["player"], "depth_role": player["depth_role"],
                "backfield_type": player["backfield_type"], "window": window,
                "sharp_rush_def_rank": int(player["sharp_def_rank"]),
                "carry_share": float(player["final_carry_share"]),
                "base_team_rush_yds": base_team_yards, "base_player_carries": base_carries,
                "base_player_rush_yds": base_player_yards,
                "team_spread": spread, "game_script": script_label(spread),
                "trailing_probability": trail_p, "neutral_probability": neutral_p,
                "leading_probability": lead_p, "trailing_rush_yds": trailing,
                "neutral_rush_yds": neutral, "leading_rush_yds": leading,
                "script_adjusted_rush_yds": script_adjusted, "floor": floor, "ceiling": ceiling,
                "sportsbook_market": "player_rush_yds" if window == "Full Game" else "",
                "consensus_line": line, "sportsbook_over_probability": book_prob,
                "model_over_probability": model_prob, "probability_edge": edge,
                "sportsbook_count": books, "odds_status": status,
                "opportunity_score": score, "tier": tier,
            })
    board = pd.DataFrame(rows).sort_values(["window", "opportunity_score"], ascending=[True, False])
    board.to_csv(LIVE / "rb_window_opportunity_board.csv", index=False, float_format="%.6f")
    return board


def main() -> int:
    team_summary, defense_summary, rb_summary = build_history()
    board = build_live_board(team_summary, defense_summary, rb_summary)
    q1 = board.loc[board["window"].eq("Q1")]
    current = pd.read_csv(LIVE / "current_week_model.csv")
    merged = q1.merge(current[["player", "exp_player_q1_yds"]], on="player", how="left")
    if not np.allclose(merged["base_player_rush_yds"], merged["exp_player_q1_yds"], atol=1e-9):
        raise AssertionError("Q1 preservation gate failed")
    print(f"PASS: {len(board)} player-window rows ({board['player'].nunique()} RBs x 3 windows)")
    print("PASS: Q1 projections preserved exactly")
    print(f"INFO: full-game sportsbook lines matched {board.query('window == \"Full Game\"')['consensus_line'].notna().sum()} RBs")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

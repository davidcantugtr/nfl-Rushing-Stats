from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd
import nflreadpy as nfl

SEASON = 2025
ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
CONFIG = ROOT / "config"
PROCESSED.mkdir(parents=True, exist_ok=True)


def _to_pandas(frame):
    if hasattr(frame, "to_pandas"):
        return frame.to_pandas()
    return pd.DataFrame(frame)


def _boolish(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series.fillna(False)
    return series.fillna(0).astype(float).eq(1)


def _find_col(df: pd.DataFrame, *names: str) -> str:
    for name in names:
        if name in df.columns:
            return name
    raise KeyError(f"None of the expected columns exist: {names}")


def _pct_hit(values: pd.Series, threshold: int) -> float:
    if len(values) == 0:
        return np.nan
    return float((values >= threshold).mean())


def _summary(group: pd.DataFrame, value_col: str, prefix: str) -> pd.Series:
    s = group[value_col].astype(float)
    return pd.Series({
        f"{prefix}_games": int(len(s)),
        f"{prefix}_mean": float(s.mean()),
        f"{prefix}_median": float(s.median()),
        f"{prefix}_std": float(s.std(ddof=0)),
        f"{prefix}_min": float(s.min()),
        f"{prefix}_p25": float(s.quantile(0.25)),
        f"{prefix}_p75": float(s.quantile(0.75)),
        f"{prefix}_max": float(s.max()),
        f"{prefix}_hit_20": _pct_hit(s, 20),
        f"{prefix}_hit_25": _pct_hit(s, 25),
        f"{prefix}_hit_30": _pct_hit(s, 30),
        f"{prefix}_hit_35": _pct_hit(s, 35),
        f"{prefix}_hit_40": _pct_hit(s, 40),
    })


def main() -> int:
    pbp = _to_pandas(nfl.load_pbp(SEASON))
    rosters = _to_pandas(nfl.load_rosters(SEASON))

    season_type_col = _find_col(pbp, "season_type")
    week_col = _find_col(pbp, "week")
    game_id_col = _find_col(pbp, "game_id")
    qtr_col = _find_col(pbp, "qtr")
    posteam_col = _find_col(pbp, "posteam")
    defteam_col = _find_col(pbp, "defteam")
    rush_attempt_col = _find_col(pbp, "rush_attempt")
    rush_yards_col = _find_col(pbp, "rushing_yards")
    rusher_id_col = _find_col(pbp, "rusher_player_id")
    rusher_name_col = _find_col(pbp, "rusher_player_name")
    home_col = _find_col(pbp, "home_team")
    away_col = _find_col(pbp, "away_team")

    reg = pbp.loc[pbp[season_type_col].astype(str).eq("REG")].copy()

    # Full team-game skeleton so zero Q1 rushing games remain real zeroes.
    games = (
        reg[[game_id_col, week_col, home_col, away_col]]
        .drop_duplicates(game_id_col)
        .rename(columns={game_id_col: "game_id", week_col: "week", home_col: "home_team", away_col: "away_team"})
        .sort_values(["week", "game_id"])
    )
    if len(games) != 272:
        raise AssertionError(f"Expected 272 regular-season games, found {len(games)}")

    home_rows = games.assign(team=games["home_team"], opponent=games["away_team"], site="H")
    away_rows = games.assign(team=games["away_team"], opponent=games["home_team"], site="A")
    team_skeleton = pd.concat([home_rows, away_rows], ignore_index=True)[["game_id", "week", "team", "opponent", "site"]]
    if len(team_skeleton) != 544:
        raise AssertionError(f"Expected 544 team-game rows, found {len(team_skeleton)}")

    rush_mask = _boolish(reg[rush_attempt_col])
    q1 = reg.loc[reg[qtr_col].fillna(0).astype(float).eq(1) & rush_mask].copy()

    if "no_play" in q1.columns:
        q1 = q1.loc[~_boolish(q1["no_play"])].copy()

    # Keep official team rush attempts. Kneels and scrambles are official team rushing unless explicitly excluded by no_play.
    q1[rush_yards_col] = pd.to_numeric(q1[rush_yards_col], errors="coerce").fillna(0)

    team_actual = (
        q1.groupby([game_id_col, posteam_col], dropna=False)
        .agg(q1_rush_att=(rush_attempt_col, "size"), q1_rush_yds=(rush_yards_col, "sum"))
        .reset_index()
        .rename(columns={game_id_col: "game_id", posteam_col: "team"})
    )
    team_actual["q1_ypc"] = np.where(team_actual["q1_rush_att"] > 0, team_actual["q1_rush_yds"] / team_actual["q1_rush_att"], 0.0)

    team_log = team_skeleton.merge(team_actual, on=["game_id", "team"], how="left")
    for col in ["q1_rush_att", "q1_rush_yds", "q1_ypc"]:
        team_log[col] = team_log[col].fillna(0)
    team_log["q1_rush_att"] = team_log["q1_rush_att"].astype(int)
    team_log["q1_rush_yds"] = team_log["q1_rush_yds"].astype(float)

    for threshold in [20, 25, 30, 35, 40]:
        team_log[f"hit_{threshold}"] = (team_log["q1_rush_yds"] >= threshold).astype(int)

    # Team offensive distributions.
    team_summary = (
        team_log.groupby("team", group_keys=False)
        .apply(lambda g: _summary(g, "q1_rush_yds", "q1_rush_yds"), include_groups=False)
        .reset_index()
    )
    att_summary = team_log.groupby("team", as_index=False).agg(
        q1_rush_att_mean=("q1_rush_att", "mean"),
        q1_rush_att_median=("q1_rush_att", "median"),
        q1_rush_att_max=("q1_rush_att", "max"),
    )
    team_summary = team_summary.merge(att_summary, on="team", how="left")

    # Defense summaries are the same team-game outcomes viewed through the opponent.
    allowed = team_log.rename(columns={"team": "offense", "opponent": "defense", "q1_rush_yds": "q1_rush_yds_allowed", "q1_rush_att": "q1_rush_att_allowed"})
    defense_summary = (
        allowed.groupby("defense", group_keys=False)
        .apply(lambda g: _summary(g, "q1_rush_yds_allowed", "q1_rush_yds_allowed"), include_groups=False)
        .reset_index()
    )
    def_att_summary = allowed.groupby("defense", as_index=False).agg(
        q1_rush_att_allowed_mean=("q1_rush_att_allowed", "mean"),
        q1_rush_att_allowed_median=("q1_rush_att_allowed", "median"),
        q1_rush_att_allowed_max=("q1_rush_att_allowed", "max"),
    )
    defense_summary = defense_summary.merge(def_att_summary, on="defense", how="left")

    # Roster lookup. Use gsis_id when available; normalize position to a compact RB/FB classification.
    roster_id_col = _find_col(rosters, "gsis_id", "player_id")
    roster_name_col = _find_col(rosters, "full_name", "player_name", "football_name")
    roster_pos_col = _find_col(rosters, "position")
    roster_lookup = rosters[[roster_id_col, roster_name_col, roster_pos_col]].dropna(subset=[roster_id_col]).copy()
    roster_lookup[roster_id_col] = roster_lookup[roster_id_col].astype(str)
    roster_lookup = roster_lookup.drop_duplicates(roster_id_col, keep="last")

    # Build an all-game rusher list so RBs who rushed later in the game but had zero Q1 attempts are retained as zero-Q1 rows.
    all_rush = reg.loc[rush_mask, [game_id_col, week_col, posteam_col, defteam_col, rusher_id_col, rusher_name_col]].copy()
    if "no_play" in reg.columns:
        all_rush = reg.loc[rush_mask & ~_boolish(reg["no_play"]), [game_id_col, week_col, posteam_col, defteam_col, rusher_id_col, rusher_name_col]].copy()
    all_rush[rusher_id_col] = all_rush[rusher_id_col].astype(str)
    rusher_games = all_rush.dropna(subset=[rusher_id_col]).drop_duplicates([game_id_col, rusher_id_col])
    rusher_games = rusher_games.merge(roster_lookup, left_on=rusher_id_col, right_on=roster_id_col, how="left")
    rusher_games[roster_pos_col] = rusher_games[roster_pos_col].fillna("UNKNOWN").astype(str).str.upper()

    unknown = rusher_games.loc[rusher_games[roster_pos_col].eq("UNKNOWN"), [rusher_id_col, rusher_name_col, posteam_col]].drop_duplicates()
    unknown.to_csv(PROCESSED / "unknown_rusher_positions_2025.csv", index=False)

    rb_games = rusher_games.loc[rusher_games[roster_pos_col].isin(["RB", "FB"])].copy()
    rb_games = rb_games.rename(columns={
        game_id_col: "game_id", week_col: "week", posteam_col: "team", defteam_col: "opponent",
        rusher_id_col: "player_id", rusher_name_col: "pbp_player_name", roster_name_col: "player_name",
        roster_pos_col: "position",
    })

    rb_q1_actual = (
        q1.assign(**{rusher_id_col: q1[rusher_id_col].astype(str)})
        .groupby([game_id_col, rusher_id_col], dropna=False)
        .agg(q1_rush_att=(rush_attempt_col, "size"), q1_rush_yds=(rush_yards_col, "sum"))
        .reset_index()
        .rename(columns={game_id_col: "game_id", rusher_id_col: "player_id"})
    )
    rb_log = rb_games.merge(rb_q1_actual, on=["game_id", "player_id"], how="left")
    rb_log["q1_rush_att"] = rb_log["q1_rush_att"].fillna(0).astype(int)
    rb_log["q1_rush_yds"] = rb_log["q1_rush_yds"].fillna(0).astype(float)
    rb_log["q1_ypc"] = np.where(rb_log["q1_rush_att"] > 0, rb_log["q1_rush_yds"] / rb_log["q1_rush_att"], 0.0)

    team_keys = team_log[["game_id", "team", "site", "q1_rush_att", "q1_rush_yds"]].rename(columns={"q1_rush_att": "team_q1_rush_att", "q1_rush_yds": "team_q1_rush_yds"})
    rb_log = rb_log.merge(team_keys, on=["game_id", "team"], how="left")
    rb_log["rb_q1_carry_share"] = np.where(rb_log["team_q1_rush_att"] > 0, rb_log["q1_rush_att"] / rb_log["team_q1_rush_att"], 0.0)
    rb_log["rb_q1_yard_share"] = np.where(rb_log["team_q1_rush_yds"] != 0, rb_log["q1_rush_yds"] / rb_log["team_q1_rush_yds"], 0.0)
    for threshold in [15, 20, 25]:
        rb_log[f"hit_{threshold}"] = (rb_log["q1_rush_yds"] >= threshold).astype(int)

    rb_summary = (
        rb_log.groupby(["player_id", "player_name", "team"], group_keys=False)
        .apply(lambda g: _summary(g, "q1_rush_yds", "q1_rush_yds"), include_groups=False)
        .reset_index()
    )
    rb_usage = rb_log.groupby(["player_id", "player_name", "team"], as_index=False).agg(
        q1_rush_att_mean=("q1_rush_att", "mean"),
        q1_rush_att_median=("q1_rush_att", "median"),
        q1_rush_att_max=("q1_rush_att", "max"),
        rb_q1_carry_share_mean=("rb_q1_carry_share", "mean"),
        rb_q1_yard_share_mean=("rb_q1_yard_share", "mean"),
    )
    rb_summary = rb_summary.merge(rb_usage, on=["player_id", "player_name", "team"], how="left")

    # Exact reconciliation against the published Q1 team aggregate targets.
    expected = pd.read_csv(CONFIG / "expected_team_q1_2025.csv")
    actual = team_log.groupby("team", as_index=False).agg(
        actual_q1_rush_att=("q1_rush_att", "sum"),
        actual_q1_rush_yds=("q1_rush_yds", "sum"),
    )
    recon = expected.merge(actual, on="team", how="outer")
    recon["att_diff"] = recon["actual_q1_rush_att"] - recon["expected_q1_rush_att"]
    recon["yds_diff"] = recon["actual_q1_rush_yds"] - recon["expected_q1_rush_yds"]
    recon["pass"] = recon["att_diff"].eq(0) & recon["yds_diff"].eq(0)

    # Save outputs before asserting, so failed CI exposes the discrepancy artifact.
    team_log.sort_values(["week", "game_id", "team"]).to_csv(PROCESSED / "team_q1_game_log_2025.csv", index=False)
    team_summary.sort_values("q1_rush_yds_mean", ascending=False).to_csv(PROCESSED / "team_q1_summary_2025.csv", index=False)
    defense_summary.sort_values("q1_rush_yds_allowed_mean", ascending=False).to_csv(PROCESSED / "team_q1_defense_summary_2025.csv", index=False)
    rb_log.sort_values(["week", "game_id", "team", "q1_rush_yds"], ascending=[True, True, True, False]).to_csv(PROCESSED / "rb_q1_game_log_2025.csv", index=False)
    rb_summary.sort_values("q1_rush_yds_mean", ascending=False).to_csv(PROCESSED / "rb_q1_summary_2025.csv", index=False)
    recon.sort_values("team").to_csv(PROCESSED / "reconciliation_2025.csv", index=False)

    if not recon["pass"].all():
        failed = recon.loc[~recon["pass"], ["team", "expected_q1_rush_att", "actual_q1_rush_att", "att_diff", "expected_q1_rush_yds", "actual_q1_rush_yds", "yds_diff"]]
        print("Reconciliation failed:\n", failed.to_string(index=False), file=sys.stderr)
        return 2

    if len(team_log) != 544:
        raise AssertionError(f"Team log row count changed unexpectedly: {len(team_log)}")

    print(f"Validated {len(games)} games, {len(team_log)} team-game rows, {len(rb_log)} RB/FB game rows.")
    print(f"League Q1 rushing yards: {team_log['q1_rush_yds'].sum():.0f}")
    print(f"Unknown rusher position records: {len(unknown)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

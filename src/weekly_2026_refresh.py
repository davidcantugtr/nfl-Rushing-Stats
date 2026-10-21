from __future__ import annotations

import argparse
import json
import math
import re
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from openpyxl import load_workbook


ROOT = Path(__file__).resolve().parents[1]
LIVE = ROOT / "data" / "live"
PROCESSED = ROOT / "data" / "processed"
ARCHIVE = ROOT / "data" / "archive"
LIVE.mkdir(parents=True, exist_ok=True)
PROCESSED.mkdir(parents=True, exist_ok=True)
ARCHIVE.mkdir(parents=True, exist_ok=True)

SEASON = 2026
TARGET_WEEK = 7
COMPLETED_WEEK = 6
REFRESH_DATE = "2026-10-21"
REFRESH_CT = "2026-10-21 03:00 CT"
OUTGOING_PUBLISHED = "2026-10-14"
DEPTH_URL = "https://github.com/nflverse/nflverse-data/releases/download/depth_charts/depth_charts_2026.csv"
PBP_URL = "https://github.com/nflverse/nflverse-data/releases/download/pbp/play_by_play_2026.csv.gz"
ROSTER_URL = "https://github.com/nflverse/nflverse-data/releases/download/rosters/roster_2026.csv"
SNAP_URL = "https://github.com/nflverse/nflverse-data/releases/download/snap_counts/snap_counts_2026.csv"
SCHEDULE_URL = "https://raw.githubusercontent.com/nflverse/nfldata/master/data/games.csv"

WEIGHTS = {
    "team_env": 0.15,
    "share": 0.30,
    "defense": 0.10,
    "history": 0.30,
    "consistency": 0.10,
    "confidence": 0.05,
}

INJURY = {
    "kyle monangai": ("DNP Monday/Tuesday — thumb/toe; Week 7 game status not yet assigned", 0.55),
    "treveyon henderson": ("FULL Tuesday — official Week 7 practice report", 1.0),
    "saquon barkley": ("Hamstring — missed Weeks 5-6; Week 7 status unresolved at early refresh", 0.70),
    "jeremiyah love": ("Ankle — missed Week 6; Week 7 status unresolved at early refresh", 0.70),
    "josh jacobs": ("Eligible to return; Week 7 role/game status to be determined", 0.70),
}


def norm_name(value: object) -> str:
    text = re.sub(r"[^a-z0-9 ]", "", str(value).lower())
    text = re.sub(r"\b(jr|sr|ii|iii|iv)\b", "", text)
    return re.sub(r"\s+", " ", text).strip()


def clip(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, float(value)))


def flag(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").fillna(0).eq(1)


def normal_over(mean: float, sd: float, threshold: float) -> float:
    sigma = max(5.0, float(sd))
    z = (float(threshold) - float(mean)) / sigma
    return 0.5 * (1.0 - math.erf(z / math.sqrt(2.0)))


def extract_table(ws, header_row: int, max_col: int, key_col: int = 1) -> pd.DataFrame:
    headers = [ws.cell(header_row, c).value for c in range(1, max_col + 1)]
    headers = [str(x).strip() if x is not None else f"_blank_{i}" for i, x in enumerate(headers)]
    rows = []
    for r in range(header_row + 1, ws.max_row + 1):
        if ws.cell(r, key_col).value in (None, ""):
            continue
        rows.append([ws.cell(r, c).value for c in range(1, max_col + 1)])
    return pd.DataFrame(rows, columns=headers)


def clean_pbp(pbp: pd.DataFrame) -> pd.DataFrame:
    reg = pbp.loc[pbp["season_type"].astype(str).eq("REG") & pd.to_numeric(pbp["week"], errors="coerce").le(COMPLETED_WEEK)].copy()
    if "play_type" in reg:
        reg = reg.loc[~reg["play_type"].astype(str).eq("no_play")].copy()
    if "two_point_attempt" in reg:
        reg = reg.loc[~flag(reg["two_point_attempt"])].copy()
    return reg


def build_window_logs(reg: pd.DataFrame, roster: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    games = reg[["game_id", "week", "home_team", "away_team"]].drop_duplicates("game_id")
    counts = games.groupby("week")["game_id"].nunique().to_dict()
    expected = {1: 16, 2: 16, 3: 16, 4: 16, 5: 15, 6: 14}
    if counts != expected:
        raise AssertionError(f"Completed-game coverage mismatch for Weeks 1-{COMPLETED_WEEK}: {counts}")
    home = games.assign(team=games.home_team, opponent=games.away_team, site="H")
    away = games.assign(team=games.away_team, opponent=games.home_team, site="A")
    skeleton = pd.concat([home, away], ignore_index=True)[["game_id", "week", "team", "opponent", "site"]]
    expected_team_games = 2 * sum(expected.values())
    if len(skeleton) != expected_team_games:
        raise AssertionError(f"Expected {expected_team_games} completed team-game rows, got {len(skeleton)}")

    rush = reg.loc[flag(reg["rush_attempt"])].copy()
    rush["rushing_yards"] = pd.to_numeric(rush["rushing_yards"], errors="coerce").fillna(0.0)
    rush["rusher_player_id"] = rush["rusher_player_id"].astype(str)

    id_col = "gsis_id" if "gsis_id" in roster.columns else "player_id"
    name_col = next(c for c in ["full_name", "player_name", "football_name"] if c in roster.columns)
    pos = roster[[id_col, name_col, "position"]].dropna(subset=[id_col]).copy()
    pos[id_col] = pos[id_col].astype(str)
    pos = pos.drop_duplicates(id_col, keep="last")

    rusher_games = rush[["game_id", "week", "posteam", "defteam", "rusher_player_id", "rusher_player_name"]].drop_duplicates(["game_id", "rusher_player_id"])
    rusher_games = rusher_games.merge(pos, left_on="rusher_player_id", right_on=id_col, how="left")
    rusher_games["position"] = rusher_games["position"].fillna("UNKNOWN").astype(str).str.upper()
    rb_games = rusher_games.loc[rusher_games["position"].isin(["RB", "FB"])].rename(columns={
        "posteam": "team", "defteam": "opponent", "rusher_player_id": "player_id",
        "rusher_player_name": "pbp_player_name", name_col: "player_name",
    })

    team_rows = []
    rb_rows = []
    windows = {"Q1": {1}, "1H": {1, 2}, "Full Game": {1, 2, 3, 4, 5}}
    for window, quarters in windows.items():
        wr = rush.loc[pd.to_numeric(rush["qtr"], errors="coerce").fillna(0).isin(quarters)].copy()
        ta = wr.groupby(["game_id", "posteam"], as_index=False).agg(rush_att=("rush_attempt", "size"), rush_yds=("rushing_yards", "sum")).rename(columns={"posteam": "team"})
        tl = skeleton.merge(ta, on=["game_id", "team"], how="left")
        tl[["rush_att", "rush_yds"]] = tl[["rush_att", "rush_yds"]].fillna(0)
        tl["rush_att"] = tl["rush_att"].astype(int)
        tl.insert(3, "window", window)
        team_rows.append(tl)

        pa = wr.groupby(["game_id", "rusher_player_id"], as_index=False).agg(rush_att=("rush_attempt", "size"), rush_yds=("rushing_yards", "sum")).rename(columns={"rusher_player_id": "player_id"})
        rl = rb_games.merge(pa, on=["game_id", "player_id"], how="left")
        rl[["rush_att", "rush_yds"]] = rl[["rush_att", "rush_yds"]].fillna(0)
        rl["rush_att"] = rl["rush_att"].astype(int)
        keys = tl[["game_id", "team", "site", "rush_att", "rush_yds"]].rename(columns={"rush_att": "team_rush_att", "rush_yds": "team_rush_yds"})
        rl = rl.merge(keys, on=["game_id", "team"], how="left")
        rl["carry_share"] = np.where(rl.team_rush_att.gt(0), rl.rush_att / rl.team_rush_att, 0.0)
        rl["yard_share"] = np.where(rl.team_rush_yds.ne(0), rl.rush_yds / rl.team_rush_yds, 0.0)
        rl.insert(3, "window", window)
        rb_rows.append(rl)

    team_log = pd.concat(team_rows, ignore_index=True)
    rb_log = pd.concat(rb_rows, ignore_index=True)
    out = rb_log.copy()
    out["gsis_id"] = out["player_id"]
    out = out[["game_id", "week", "team", "window", "opponent", "player_id", "pbp_player_name", "gsis_id", "player_name", "position", "rush_att", "rush_yds", "site", "team_rush_att", "team_rush_yds", "carry_share", "yard_share"]]
    out = out.sort_values(["week", "game_id", "team", "window", "rush_yds"], ascending=[True, True, True, True, False])
    if out.duplicated(["game_id", "team", "window", "player_id"]).any():
        raise AssertionError("Duplicate RB/FB player-game-window rows detected")
    return team_log, rb_log, out


def build_pace_and_usage(reg: pd.DataFrame, snaps: pd.DataFrame, roster: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    snaps = snaps.loc[(snaps["season"].eq(SEASON)) & snaps["week"].le(COMPLETED_WEEK)].copy()
    snaps["player_key"] = snaps["player"].map(norm_name)
    snaps["team_snaps_est"] = np.where(pd.to_numeric(snaps["offense_pct"], errors="coerce").gt(0.05), pd.to_numeric(snaps["offense_snaps"], errors="coerce") / pd.to_numeric(snaps["offense_pct"], errors="coerce"), np.nan)
    team_snaps = snaps.groupby(["game_id", "week", "team", "opponent"], as_index=False).agg(team_off_snaps=("team_snaps_est", "median"))
    team_snaps["team_off_snaps"] = team_snaps["team_off_snaps"].round().astype("Int64")

    valid = reg.loc[reg["posteam"].notna()].copy()
    pass_att = flag(valid["pass_attempt"])
    rush_att = flag(valid["rush_attempt"])
    sack = flag(valid["sack"])
    valid_snap = valid.loc[pass_att | rush_att | sack].copy()
    team_counts = valid_snap.groupby(["game_id", "week", "posteam", "defteam"], as_index=False).agg(
        pass_attempts=("pass_attempt", lambda x: int(flag(x).sum())),
        rush_attempts=("rush_attempt", lambda x: int(flag(x).sum())),
    ).rename(columns={"posteam": "team", "defteam": "opponent"})
    targets = valid.loc[valid["receiver_player_id"].notna()].copy()
    target_counts = targets.groupby(["game_id", "week", "posteam"], as_index=False).size().rename(columns={"posteam": "team", "size": "team_targets"})

    drive = valid.loc[valid["fixed_drive"].notna() & valid["drive_time_of_possession"].notna(), ["game_id", "week", "posteam", "fixed_drive", "drive_time_of_possession"]].drop_duplicates(["game_id", "fixed_drive"])
    def to_sec(v):
        try:
            m, s = str(v).split(":")
            return int(m) * 60 + int(s)
        except Exception:
            return 0
    drive["top_seconds"] = drive["drive_time_of_possession"].map(to_sec)
    drive_sum = drive.groupby(["game_id", "week", "posteam"], as_index=False).agg(drives=("fixed_drive", "nunique"), top_seconds=("top_seconds", "sum")).rename(columns={"posteam": "team"})

    pace = team_counts.merge(team_snaps, on=["game_id", "week", "team", "opponent"], how="left").merge(target_counts, on=["game_id", "week", "team"], how="left").merge(drive_sum, on=["game_id", "week", "team"], how="left")
    pace["team_targets"] = pace["team_targets"].fillna(0).astype(int)
    pace["offensive_snaps"] = pace["team_off_snaps"].fillna(pace.pass_attempts + pace.rush_attempts)
    opp = pace[["game_id", "team", "offensive_snaps", "top_seconds"]].rename(columns={"team": "opponent", "offensive_snaps": "opponent_off_snaps", "top_seconds": "opponent_top_seconds"})
    pace = pace.merge(opp, on=["game_id", "opponent"], how="left")
    pace["plays_per_drive"] = pace.offensive_snaps / pace.drives.replace(0, np.nan)
    pace["seconds_per_play"] = pace.top_seconds / pace.offensive_snaps.replace(0, np.nan)
    sites = reg[["game_id", "home_team", "away_team"]].drop_duplicates("game_id")
    pace = pace.merge(sites, on="game_id", how="left")
    pace["site"] = np.where(pace.team.eq(pace.home_team), "Home", "Away")
    pace = pace.sort_values(["team", "week"])
    pace["prior_seconds_per_play"] = pace.groupby("team")["seconds_per_play"].shift(1)
    delta = pd.to_numeric(pace.seconds_per_play, errors="coerce").astype(float) - pd.to_numeric(pace.prior_seconds_per_play, errors="coerce").astype(float)
    pace["pace_trend"] = np.select([delta.isna(), delta.le(-2), delta.ge(2)], ["BASELINE", "FASTER", "SLOWER"], default="STABLE")
    def mmss(sec):
        sec = int(round(float(sec))) if pd.notna(sec) else 0
        return f"{sec // 60}:{sec % 60:02d}"
    pace["time_of_possession"] = pace.top_seconds.map(mmss)
    pace["opponent_top"] = pace.opponent_top_seconds.map(mmss)
    edge = pace.top_seconds - pace.opponent_top_seconds
    pace["possession_edge"] = edge.map(lambda x: ("+" if x >= 0 else "-") + mmss(abs(x)))
    pace_out = pace.assign(
        Season=SEASON, Week=pace.week, **{"Game ID": pace.game_id, "Team": pace.team, "Opponent": pace.opponent, "Site": pace.site, "Game State": "Final", "Offensive Snaps": pace.offensive_snaps, "Opponent Off Snaps": pace.opponent_off_snaps, "Time of Possession": pace.time_of_possession, "Opponent TOP": pace.opponent_top, "Drives": pace.drives, "Plays / Drive": pace.plays_per_drive, "Seconds / Play": pace.seconds_per_play, "Pass Attempts": pace.pass_attempts, "Rush Attempts": pace.rush_attempts, "Team Targets": pace.team_targets, "Pace Trend": pace.pace_trend, "Possession Edge": pace.possession_edge, "Source Updated": REFRESH_CT, "Source": PBP_URL, "Data State": "FINAL"}
    )[["Season", "Week", "Game ID", "Team", "Opponent", "Site", "Game State", "Offensive Snaps", "Opponent Off Snaps", "Time of Possession", "Opponent TOP", "Drives", "Plays / Drive", "Seconds / Play", "Pass Attempts", "Rush Attempts", "Team Targets", "Pace Trend", "Possession Edge", "Source Updated", "Source", "Data State"]]

    id_col = "gsis_id" if "gsis_id" in roster.columns else "player_id"
    name_col = next(c for c in ["full_name", "player_name", "football_name"] if c in roster.columns)
    rb_names = roster.loc[roster["position"].astype(str).str.upper().isin(["RB", "FB"]), [id_col, name_col]].drop_duplicates(id_col)
    rb_names["player_key"] = rb_names[name_col].map(norm_name)
    rb_keys = set(rb_names.player_key)
    rs = snaps.loc[snaps.player_key.isin(rb_keys) | snaps.position.astype(str).str.upper().isin(["RB", "FB"])].copy()
    target_player = targets.groupby(["game_id", "week", "posteam", "receiver_player_id"], as_index=False).size().rename(columns={"posteam": "team", "receiver_player_id": "gsis_id", "size": "targets"})
    rush_player = valid.loc[flag(valid["rush_attempt"]) & valid["rusher_player_id"].notna()].groupby(["game_id", "week", "posteam", "rusher_player_id"], as_index=False).size().rename(columns={"posteam": "team", "rusher_player_id": "gsis_id", "size": "rush_attempts"})
    id_lookup = rb_names.copy(); id_lookup[id_col] = id_lookup[id_col].astype(str)
    id_lookup = id_lookup.rename(columns={id_col: "gsis_id"})[["gsis_id", "player_key"]]
    target_player["gsis_id"] = target_player.gsis_id.astype(str); rush_player["gsis_id"] = rush_player.gsis_id.astype(str)
    tp = target_player.merge(id_lookup, on="gsis_id", how="left")
    rp = rush_player.merge(id_lookup, on="gsis_id", how="left")
    usage = rs.merge(team_snaps[["game_id", "week", "team", "team_off_snaps"]], on=["game_id", "week", "team"], how="left")
    usage = usage.merge(tp[["game_id", "week", "team", "player_key", "targets"]], on=["game_id", "week", "team", "player_key"], how="left")
    usage = usage.merge(rp[["game_id", "week", "team", "player_key", "rush_attempts"]], on=["game_id", "week", "team", "player_key"], how="left")
    usage = usage.merge(target_counts[["game_id", "week", "team", "team_targets"]], on=["game_id", "week", "team"], how="left")
    usage[["targets", "rush_attempts", "team_targets"]] = usage[["targets", "rush_attempts", "team_targets"]].fillna(0)
    usage["snap_share"] = usage.offense_snaps / usage.team_off_snaps.replace(0, np.nan)
    usage["target_share"] = usage.targets / usage.team_targets.replace(0, np.nan)
    usage["total_opportunities"] = usage.targets + usage.rush_attempts
    team_rush = team_counts[["game_id", "week", "team", "rush_attempts"]].rename(columns={"rush_attempts": "team_rush_attempts"})
    usage = usage.merge(team_rush, on=["game_id", "week", "team"], how="left")
    usage["opportunity_share"] = usage.total_opportunities / (usage.team_targets + usage.team_rush_attempts).replace(0, np.nan)
    usage = usage.sort_values(["player_key", "week"])
    usage["prior_snap_share"] = usage.groupby("player_key")["snap_share"].shift(1)
    usage["prior_target_share"] = usage.groupby("player_key")["target_share"].shift(1)
    usage["snap_trend"] = usage.snap_share - usage.prior_snap_share
    usage["target_trend"] = usage.target_share - usage.prior_target_share
    usage["team_usage_rank"] = usage.groupby(["week", "team"])["opportunity_share"].rank(method="first", ascending=False).astype(int)
    usage_out = pd.DataFrame({
        "Season": SEASON, "Week": usage.week, "Team": usage.team, "Opponent": usage.opponent,
        "Player": usage.player, "Role": usage.team_usage_rank.map(lambda x: "RB1" if x == 1 else "RB2" if x == 2 else "DEPTH"),
        "Offensive Snaps": usage.offense_snaps, "Team Off Snaps": usage.team_off_snaps,
        "Snap Share": usage.snap_share, "Targets": usage.targets, "Team Targets": usage.team_targets,
        "Target Share": usage.target_share, "Rush Attempts": usage.rush_attempts, "Routes": np.nan,
        "Total Opportunities": usage.total_opportunities, "Opportunity Share": usage.opportunity_share,
        "Prior Snap Share": usage.prior_snap_share, "Snap Trend": usage.snap_trend,
        "Prior Target Share": usage.prior_target_share, "Target Trend": usage.target_trend,
        "Team Usage Rank": usage.team_usage_rank, "Sportsbook Trend": "", "Market Note": "",
        "Source Updated": REFRESH_CT, "Data State": "FINAL",
    }).sort_values(["Week", "Team", "Team Usage Rank", "Player"])
    return pace_out, usage_out


def build_model(depth: pd.DataFrame, team_log: pd.DataFrame, rb_log: pd.DataFrame, schedule: pd.DataFrame, sheet_wk3: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    depth["dt"] = pd.to_datetime(depth["dt"], errors="coerce", utc=True)
    latest = depth.loc[depth.dt.eq(depth.dt.max()) & depth.pos_abb.astype(str).str.upper().eq("RB")].copy()
    # NFL transaction wire: New York released Najee Harris before Week 7; the
    # upstream depth-chart snapshot still retained him, so exclude that stale row.
    latest = latest.loc[~(
        latest.team.eq("NYG") & latest.player_name.map(norm_name).eq("najee harris")
    )].copy()
    latest["pos_rank"] = pd.to_numeric(latest["pos_rank"], errors="coerce")
    latest = latest.sort_values(["team", "pos_rank", "player_name"])
    top2 = latest.groupby("team", as_index=False).head(2).copy()
    top2["team_rb_order"] = top2.groupby("team").cumcount() + 1
    if len(top2) != 64 or top2.team.nunique() != 32:
        raise AssertionError(f"Expected 64 current RB rows across 32 teams, got {len(top2)} / {top2.team.nunique()}")

    sched_home = schedule.assign(team=schedule.home_team, opponent=schedule.away_team, site="H")
    sched_away = schedule.assign(team=schedule.away_team, opponent=schedule.home_team, site="A")
    team_sched = pd.concat([sched_home, sched_away], ignore_index=True)[["game_id", "team", "opponent", "site"]]
    active_teams = set(team_sched.team)
    all_teams = set(latest.team)
    bye_teams = sorted(all_teams - active_teams)
    expected_active = 2 * len(schedule)
    if len(team_sched) != expected_active or len(active_teams) != expected_active or len(bye_teams) != 32 - expected_active:
        raise AssertionError(f"Week {TARGET_WEEK} schedule/bye coverage failed: {len(team_sched)} rows, byes={bye_teams}")
    recip = team_sched.merge(team_sched[["team", "opponent"]], left_on=["team", "opponent"], right_on=["opponent", "team"], how="inner")
    if len(recip) != expected_active:
        raise AssertionError(f"Week {TARGET_WEEK} reciprocal matchup validation failed")
    team_sched = pd.concat([
        team_sched,
        pd.DataFrame({"game_id": [f"2026_{TARGET_WEEK:02d}_BYE_{t}" for t in bye_teams], "team": bye_teams, "opponent": "BYE", "site": "B"}),
    ], ignore_index=True)

    q1_team = team_log.loc[team_log.window.eq("Q1")].copy()
    q1_rb = rb_log.loc[rb_log.window.eq("Q1")].copy()
    fg_team = team_log.loc[team_log.window.eq("Full Game")].copy()
    fg_rb = rb_log.loc[rb_log.window.eq("Full Game")].copy()
    public = q1_team.groupby("opponent", as_index=False).agg(q1_allowed=("rush_yds", "mean"), q1_att_allowed=("rush_att", "mean")).rename(columns={"opponent": "team"})
    public["public_def_rank"] = public.q1_allowed.rank(method="first", ascending=True).astype(int)
    off = q1_team.groupby("team", as_index=False).agg(off_q1_yds=("rush_yds", "mean"), off_q1_att=("rush_att", "mean"))
    off25 = pd.read_csv(PROCESSED / "team_q1_summary_2025.csv")[["team", "q1_rush_yds_mean", "q1_rush_att_mean"]]
    def25 = pd.read_csv(PROCESSED / "team_q1_defense_summary_2025.csv")[["defense", "q1_rush_yds_allowed_mean", "q1_rush_att_allowed_mean"]].rename(columns={"defense": "team"})
    rb25 = pd.read_csv(PROCESSED / "rb_q1_summary_2025.csv")
    rb25["player_key"] = rb25.player_name.map(norm_name)

    sharp = pd.read_csv(LIVE / "sharp_week4_rush_efficiency_def.csv")
    sharp_map = dict(zip(sharp.team, pd.to_numeric(sharp.rush_eff_def_opponent_rank, errors="coerce")))
    if "LAR" in sharp_map:
        sharp_map["LA"] = sharp_map["LAR"]

    top2 = top2.merge(team_sched, on="team", how="left").merge(off, on="team", how="left")
    top2 = top2.merge(public.rename(columns={"team": "opponent", "q1_allowed": "opp_q1_allowed", "q1_att_allowed": "opp_q1_att_allowed", "public_def_rank": "public_def_rank"}), on="opponent", how="left")
    top2 = top2.merge(off25, on="team", how="left").merge(def25.rename(columns={"team": "opponent", "q1_rush_yds_allowed_mean": "opp25_yds", "q1_rush_att_allowed_mean": "opp25_att"}), on="opponent", how="left")
    top2["player_key"] = top2.player_name.map(norm_name)
    top2["injury_status"] = top2.player_key.map(lambda k: INJURY.get(k, ("No RB limitation listed in the early official Week 7 report; most Sunday reports incomplete", 1.0))[0])
    top2["availability"] = top2.player_key.map(lambda k: INJURY.get(k, ("", 1.0))[1])

    q1p = q1_rb.copy(); q1p["player_key"] = q1p.player_name.map(norm_name)
    fgp = fg_rb.copy(); fgp["player_key"] = fgp.player_name.map(norm_name)
    q1stats = q1p.groupby(["team", "player_key"], as_index=False).agg(q1_att=("rush_att", "sum"), q1_yds=("rush_yds", "sum"), q1_mean=("rush_yds", "mean"), q1_std=("rush_yds", lambda x: float(np.std(x, ddof=0))), q1_hit20=("rush_yds", lambda x: float((x >= 20).mean())))
    fgstats = fgp.groupby(["team", "player_key"], as_index=False).agg(fg_att=("rush_att", "sum"), fg_yds=("rush_yds", "sum"))
    fg_avg = fgp.groupby(["team", "player_key"], as_index=False).agg(avg_rush_att_per_game=("rush_att", "mean"))
    opp_ypc = fg_team.groupby("opponent", as_index=False).agg(allowed_rush_att=("rush_att", "sum"), allowed_rush_yds=("rush_yds", "sum")).rename(columns={"opponent": "team"})
    opp_ypc["opp_avg_ypc_allowed"] = opp_ypc.allowed_rush_yds / opp_ypc.allowed_rush_att.replace(0, np.nan)
    teamq = q1_team.groupby("team", as_index=False).agg(team_q1_att=("rush_att", "sum"), team_q1_yds=("rush_yds", "sum"))
    teamfg = fg_team.groupby("team", as_index=False).agg(team_fg_att=("rush_att", "sum"))
    top2 = top2.merge(q1stats, on=["team", "player_key"], how="left").merge(fgstats, on=["team", "player_key"], how="left").merge(fg_avg, on=["team", "player_key"], how="left").merge(teamq, on="team", how="left").merge(teamfg, on="team", how="left")
    top2 = top2.merge(opp_ypc[["team", "opp_avg_ypc_allowed"]].rename(columns={"team": "opponent"}), on="opponent", how="left")
    for c in ["q1_att", "q1_yds", "q1_mean", "q1_std", "q1_hit20", "fg_att", "fg_yds"]:
        top2[c] = pd.to_numeric(top2[c], errors="coerce").fillna(0.0)
    top2["q1_team_share"] = top2.q1_att / top2.team_q1_att.replace(0, np.nan)
    top2["fg_team_share"] = top2.fg_att / top2.team_fg_att.replace(0, np.nan)
    top2[["q1_team_share", "fg_team_share"]] = top2[["q1_team_share", "fg_team_share"]].fillna(0)

    rows = []
    for team, group in top2.groupby("team", sort=True):
        g = group.copy().sort_values("team_rb_order")
        room_capture = clip(max(0.68, g.q1_team_share.sum() * 0.65 + g.fg_team_share.sum() * 0.35), 0.68, 0.94)
        priors = np.array([0.72, 0.28])
        q = g.q1_team_share.to_numpy(float); f = g.fg_team_share.to_numpy(float)
        qn = q / q.sum() if q.sum() else priors
        fn = f / f.sum() if f.sum() else priors
        raw = (0.55 * qn + 0.30 * fn + 0.15 * priors) * g.availability.to_numpy(float)
        raw = raw / raw.sum() if raw.sum() else priors
        shares = room_capture * raw
        yard_eff = np.where(g.q1_att.to_numpy(float) > 0, g.q1_yds.to_numpy(float) / np.maximum(1, g.q1_att.to_numpy(float)), 4.0)
        eff_adj = np.clip(yard_eff / max(1.0, np.average(yard_eff, weights=np.maximum(raw, 0.05))), 0.80, 1.20)
        yard_raw = raw * eff_adj
        yard_shares = room_capture * yard_raw / yard_raw.sum()
        for idx, (_, r) in enumerate(g.iterrows()):
            if r.opponent == "BYE":
                rows.append({
                    "week": TARGET_WEEK, "team": team, "opponent": "BYE", "site": "B",
                    "player": r.player_name, "player_id": r.gsis_id, "depth_role": f"RB{int(r.team_rb_order)}",
                    "backfield_type": "BYE", "sharp_def_rank": np.nan,
                    "final_carry_share": shares[idx], "exp_q1_carries": 0.0,
                    "proj_team_q1_yds": 0.0, "exp_player_q1_yds": 0.0,
                    "floor": 0.0, "ceiling": 0.0, "prob20_proxy": 0.0, "prob25_proxy": 0.0,
                    "v05_candidate_score": 0.0, "prime_label": "PASS", "public_def_rank": np.nan,
                    "def_input_used": "BYE — no matchup", "def_source_status": "Sharp Week 4 snapshot preserved — STALE for Week 7",
                    "injury_status": "BYE",
                    "avg_rush_att_per_game": float(r.avg_rush_att_per_game) if pd.notna(r.avg_rush_att_per_game) else 0.0,
                    "opp_avg_ypc_allowed": np.nan, "volume_x_ypc_proj_rush_yds": 0.0,
                })
                continue
            team_proj = (0.50 * float(r.off_q1_yds) + 0.35 * float(r.opp_q1_allowed) + 0.15 * float(r.q1_rush_yds_mean))
            rank_factor = clip(1.0 + (float(r.public_def_rank) - 16.5) * 0.008, 0.88, 1.12)
            team_proj *= rank_factor
            team_att = 0.50 * float(r.off_q1_att) + 0.35 * float(r.opp_q1_att_allowed) + 0.15 * float(r.q1_rush_att_mean)
            exp_carries = team_att * shares[idx]
            exp_yards = team_proj * yard_shares[idx]
            h25 = rb25.loc[rb25.player_key.eq(r.player_key)]
            hist25_mean = float(h25.q1_rush_yds_mean.iloc[0]) if len(h25) else exp_yards
            hist25_std = float(h25.q1_rush_yds_std.iloc[0]) if len(h25) else max(6.0, exp_yards * 0.45)
            hist25_hit = float(h25.q1_rush_yds_hit_20.iloc[0]) if len(h25) else normal_over(exp_yards, hist25_std, 20)
            hist_mean = 0.70 * float(r.q1_mean) + 0.30 * hist25_mean
            hist_std = max(6.0, 0.70 * float(r.q1_std) + 0.30 * hist25_std)
            hit20 = 0.70 * float(r.q1_hit20) + 0.30 * hist25_hit
            components = {
                "team_env": clip((team_proj - 18.0) / 25.0 * 100.0, 0, 100),
                "share": clip(shares[idx] / 0.70 * 100.0, 0, 100),
                "defense": clip(float(r.public_def_rank) / 32.0 * 100.0, 0, 100),
                "history": clip(hist_mean / 30.0 * 100.0, 0, 100),
                "consistency": clip(hit20 * 100.0, 0, 100),
                "confidence": 100.0,
            }
            score = sum(WEIGHTS[k] * components[k] for k in WEIGHTS)
            if r.availability < 1:
                score *= 0.88 + 0.12 * float(r.availability)
            tier = "ELITE" if score >= 82 else "PRIME" if score >= 74 else "STRONG" if score >= 66 else "WATCH" if score >= 58 else "PASS"
            qbdil = room_capture < 0.80
            if shares[0] >= 0.68 and shares[1] <= 0.25:
                bft = "Bellcow"
            elif shares[0] >= 0.58:
                bft = "Strong RB1"
            else:
                bft = "RB1A/RB1B"
            if qbdil:
                bft += " / QB diluted"
            rows.append({
                "week": TARGET_WEEK, "team": team, "opponent": r.opponent, "site": r.site,
                "player": r.player_name, "player_id": r.gsis_id, "depth_role": f"RB{int(r.team_rb_order)}",
                "backfield_type": bft, "sharp_def_rank": sharp_map.get(r.opponent, np.nan),
                "final_carry_share": shares[idx], "exp_q1_carries": exp_carries,
                "proj_team_q1_yds": team_proj, "exp_player_q1_yds": exp_yards,
                "floor": max(0, exp_yards - 0.75 * hist_std), "ceiling": exp_yards + 0.75 * hist_std,
                "prob20_proxy": normal_over(exp_yards, hist_std, 20), "prob25_proxy": normal_over(exp_yards, hist_std, 25),
                "v05_candidate_score": score, "prime_label": tier, "public_def_rank": int(r.public_def_rank),
                "def_input_used": "2026 public Q1 rush allowed rank", "def_source_status": "Sharp Week 4 snapshot preserved separately — STALE for Week 7",
                "injury_status": r.injury_status,
                "avg_rush_att_per_game": float(r.avg_rush_att_per_game) if pd.notna(r.avg_rush_att_per_game) else 0.0,
                "opp_avg_ypc_allowed": float(r.opp_avg_ypc_allowed),
                "volume_x_ypc_proj_rush_yds": (float(r.avg_rush_att_per_game) if pd.notna(r.avg_rush_att_per_game) else 0.0) * float(r.opp_avg_ypc_allowed),
            })
    model = pd.DataFrame(rows).sort_values(["v05_candidate_score", "team", "player"], ascending=[False, True, True]).reset_index(drop=True)
    if len(model) != 64 or model.duplicated(["team", "player"]).any():
        raise AssertionError("Current Week Model row-count/duplicate validation failed")
    sums = model.groupby("team").final_carry_share.sum()
    valid_share = sums.between(0.679999, 0.940001)
    if not valid_share.all():
        raise AssertionError(f"Role-share sanity failed: {sums[~valid_share].to_dict()}")

    dash = model.copy()
    dash.insert(1, "rank", range(1, len(dash) + 1))
    dash["v04_score"] = dash.v05_candidate_score
    dash["v05_delta"] = 0.0
    dash = dash[["week", "rank", "player", "player_id", "team", "opponent", "site", "depth_role", "backfield_type", "sharp_def_rank", "public_def_rank", "def_input_used", "final_carry_share", "exp_q1_carries", "proj_team_q1_yds", "exp_player_q1_yds", "floor", "ceiling", "prob20_proxy", "prob25_proxy", "v04_score", "v05_candidate_score", "v05_delta", "prime_label", "injury_status"]]

    defense = public.copy()
    defense["sharp_rush_def_rank"] = defense.team.map(sharp_map).astype("Int64")
    defense["sharp_last_verified"] = "2026-10-03"
    defense["sharp_status"] = "STALE — last verified Week 4 snapshot"
    defense["current_q1_rush_yds_allowed_per_game"] = defense.q1_allowed
    defense["current_public_rush_def_rank"] = defense.public_def_rank
    defense["model_def_input"] = defense.public_def_rank
    defense["model_source_used"] = "nflverse 2026 Weeks 1–6 Q1 rushing allowed"
    defense["integrity_note"] = "Sharp proprietary Week 4 value preserved but not guessed; completed Weeks 1-6 public rank used for Week 7"
    defense = defense[["team", "sharp_rush_def_rank", "sharp_last_verified", "sharp_status", "current_q1_rush_yds_allowed_per_game", "current_public_rush_def_rank", "model_def_input", "model_source_used", "integrity_note"]].sort_values("team")
    return model, dash, defense


def build_board(model: pd.DataFrame, team_log: pd.DataFrame, rb_log: pd.DataFrame) -> pd.DataFrame:
    rows = []
    thresholds = {"Q1": 20.0, "1H": 40.0, "Full Game": 60.0}
    for _, p in model.iterrows():
        for window in ["Q1", "1H", "Full Game"]:
            t = team_log.loc[team_log.window.eq(window)]
            r = rb_log.loc[rb_log.window.eq(window)].copy(); r["player_key"] = r.player_name.map(norm_name)
            off = t.loc[t.team.eq(p.team), "rush_yds"].mean()
            opp = t.loc[t.opponent.eq(p.opponent), "rush_yds"].mean()
            tatt = t.loc[t.team.eq(p.team), "rush_att"].mean()
            ratt = r.loc[(r.team.eq(p.team)) & r.player_key.eq(norm_name(p.player)), "rush_att"].mean()
            ryd = r.loc[(r.team.eq(p.team)) & r.player_key.eq(norm_name(p.player)), "rush_yds"]
            if window == "Q1":
                base_team = p.proj_team_q1_yds; base_carries = p.exp_q1_carries; base_player = p.exp_player_q1_yds
            else:
                base_team = 0.58 * off + 0.42 * opp
                hist_share = (ratt / tatt) if pd.notna(ratt) and pd.notna(tatt) and tatt else p.final_carry_share
                carry_share = 0.65 * p.final_carry_share + 0.35 * hist_share
                base_carries = tatt * carry_share
                yard_share = r.loc[(r.team.eq(p.team)) & r.player_key.eq(norm_name(p.player)), "yard_share"].mean()
                if pd.isna(yard_share): yard_share = carry_share
                base_player = base_team * (0.65 * p.final_carry_share + 0.35 * yard_share)
            sd = max(7.0 if window == "Q1" else 12.0 if window == "1H" else 18.0, float(ryd.std(ddof=0)) if len(ryd) else base_player * 0.45)
            benchmark = thresholds[window]
            prob = normal_over(base_player, sd, benchmark)
            score = p.v05_candidate_score
            tier = "ELITE" if score >= 82 else "PRIME" if score >= 74 else "STRONG" if score >= 66 else "WATCH" if score >= 58 else "PASS"
            rows.append({
                "week": TARGET_WEEK, "team": p.team, "opponent": p.opponent, "site": p.site, "player": p.player,
                "player_id": p.player_id, "depth_role": p.depth_role, "backfield_type": p.backfield_type, "window": window,
                "sharp_rush_def_rank": p.sharp_def_rank, "public_rush_def_rank": p.public_def_rank,
                "def_input_used": p.def_input_used, "carry_share": p.final_carry_share,
                "base_team_rush_yds": base_team, "base_player_carries": base_carries, "base_player_rush_yds": base_player,
                "team_spread": np.nan, "game_script": "UNKNOWN — no verified market line", "trailing_probability": 0.33,
                "neutral_probability": 0.34, "leading_probability": 0.33, "trailing_rush_yds": base_player * 0.90,
                "neutral_rush_yds": base_player, "leading_rush_yds": base_player * 1.10, "script_adjusted_rush_yds": base_player,
                "floor": max(0, base_player - 0.75 * sd), "ceiling": base_player + 0.75 * sd,
                "model_benchmark_yards": benchmark, "model_benchmark_probability": prob,
                "probability_source": "2025 baseline + completed 2026 Weeks 1–6 window distribution",
                "sportsbook_market": "", "consensus_line": np.nan, "sportsbook_over_probability": np.nan,
                "model_over_probability": np.nan, "probability_edge": np.nan, "sportsbook_count": 0,
                "odds_status": "MODEL ONLY — NO VERIFIED LINE", "opportunity_score": score, "tier": tier,
                "injury_status": p.injury_status,
            })
    board = pd.DataFrame(rows).sort_values(["window", "opportunity_score"], ascending=[True, False])
    if len(board) != 192 or board.groupby("window").size().to_dict() != {"1H": 64, "Full Game": 64, "Q1": 64}:
        raise AssertionError("RB Window Board validation failed")
    if board.duplicated(["week", "team", "player", "window"]).any():
        raise AssertionError("Duplicate Week 4 player-window rows detected")
    return board


def grade_archive(archive: pd.DataFrame, outgoing: pd.DataFrame, rb_log: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    current_week = int(pd.to_numeric(outgoing.week, errors="coerce").dropna().iloc[0])
    base = archive.loc[pd.to_numeric(archive.week, errors="coerce").lt(current_week)].copy()
    current = outgoing.copy()
    current["season"] = SEASON
    current["published"] = OUTGOING_PUBLISHED
    if "neutral_q1_yds_baseline" in current.columns:
        current["exp_player_q1_yds"] = pd.to_numeric(current["neutral_q1_yds_baseline"], errors="coerce")
    if "sharp_rush_eff_rank" in current.columns:
        current["sharp_def_rank"] = pd.to_numeric(current["sharp_rush_eff_rank"], errors="coerce")
    current["_published_rank"] = pd.to_numeric(current.get("rank"), errors="coerce")
    keep = ["season", "week", "published", "team", "opponent", "site", "player", "depth_role", "backfield_type", "sharp_def_rank", "final_carry_share", "exp_q1_carries", "proj_team_q1_yds", "exp_player_q1_yds", "floor", "ceiling", "prob20_proxy", "prob25_proxy", "v05_candidate_score", "prime_label", "avg_rush_att_per_game", "opp_avg_ypc_allowed", "volume_x_ypc_proj_rush_yds"]
    for c in keep:
        if c not in current.columns: current[c] = np.nan
    if "_published_rank" not in base.columns:
        base["_published_rank"] = np.nan
    combined = pd.concat([base[keep + ["_published_rank"]], current[keep + ["_published_rank"]]], ignore_index=True)
    combined = combined.drop_duplicates(["season", "week", "team", "player"], keep="last").sort_values(["week", "_published_rank", "v05_candidate_score"], ascending=[True, True, False], kind="stable")
    counts = combined.groupby("week").size().to_dict()
    if counts.get(current_week) != len(current) or len(combined) != len(base.drop_duplicates(["season", "week", "team", "player"])) + len(current):
        raise AssertionError(f"Weekly Archive preservation failed: {counts}")

    actual = rb_log.loc[rb_log.window.eq("Q1"), ["game_id", "week", "team", "opponent", "player_id", "player_name", "rush_att", "rush_yds", "team_rush_att", "team_rush_yds", "carry_share", "yard_share"]].copy()
    actual["player_key"] = actual.player_name.map(norm_name)
    pred = combined.copy(); pred["player_key"] = pred.player.map(norm_name)
    joined = pred.merge(actual, on=["week", "team", "player_key"], how="left", suffixes=("_pred", "_act"))
    for c in ["rush_att", "rush_yds", "team_rush_att", "team_rush_yds", "carry_share", "yard_share"]:
        joined[c] = pd.to_numeric(joined[c], errors="coerce").fillna(0)
    score_rank = joined.groupby("week")["v05_candidate_score"].rank(method="first", ascending=False)
    fallback_rank = joined.groupby("week").cumcount() + 1
    joined["rank"] = pd.to_numeric(joined["_published_rank"], errors="coerce").fillna(score_rank).fillna(fallback_rank).astype(int)
    joined["hit20"] = (joined.rush_yds >= 20).astype(int); joined["hit25"] = (joined.rush_yds >= 25).astype(int)
    joined["abs_error"] = (pd.to_numeric(joined.exp_player_q1_yds) - joined.rush_yds).abs()
    results = pd.DataFrame({
        "Season": SEASON, "Week": joined.week, "Rank": joined["rank"], "Game ID": joined.game_id,
        "Team": joined.team, "Opponent": joined.opponent_pred, "Site": joined.site, "Player": joined.player,
        "Player ID": joined.player_id, "Role": joined.depth_role, "Backfield Flag": joined.backfield_type,
        "Exp Carries": joined.exp_q1_carries, "Exp Q1 Yds": joined.exp_player_q1_yds,
        "Actual Q1 Carries": joined.rush_att, "Actual Q1 Yds": joined.rush_yds,
        "Team Q1 Carries": joined.team_rush_att, "Team Q1 Yds": joined.team_rush_yds,
        "Actual Carry Share": joined.carry_share, "Actual Yard Share": joined.yard_share,
        "Hit 20+": joined.hit20, "Hit 25+": joined.hit25, "Absolute Error": joined.abs_error,
        "Model Score": joined.v05_candidate_score, "Tier": joined.prime_label, "Refreshed": REFRESH_DATE,
    }).sort_values(["Week", "Rank"])

    accuracy_rows = []
    for week, g in joined.groupby("week"):
        g = g.sort_values("rank", ascending=True)
        top10 = g.head(10); top5 = g.head(5)
        actual_top10 = set(g.sort_values("rush_yds", ascending=False).head(10).player_key)
        overlap = sum(x in actual_top10 for x in top10.player_key) / 10
        spearman = (-g["rank"]).corr(g.rush_yds.rank())
        accuracy_rows.append({
            "Week": int(week), "Slate Date": f"Week {int(week)} slate", "Model Version": "v0.5 production candidate",
            "Elite Picks": int((g.prime_label == "ELITE").sum()), "Prime Picks": int((g.prime_label == "PRIME").sum()),
            "Top-10 Overlap": overlap, "Top-5 Avg Actual Q1 Yds": top5.rush_yds.mean(),
            "20+ Hit Rate": (top10.rush_yds >= 20).mean(), "25+ Hit Rate": (top10.rush_yds >= 25).mean(),
            "MAE": g.abs_error.mean(), "Spearman": spearman,
            "Notes": f"GRADED {REFRESH_DATE} from nflverse Q1 play-by-play; top-10 metrics use frozen published Week {int(week)} ranking",
        })
    accuracy = pd.DataFrame(accuracy_rows)
    return combined.drop(columns=["_published_rank"]), results, accuracy


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sheet-xlsx", required=True)
    args = ap.parse_args()
    wb = load_workbook(args.sheet_xlsx, data_only=True, read_only=False)
    outgoing = extract_table(wb["Current Week Model"], 4, 26)
    archive = extract_table(wb["Weekly Archive"], 6, 23)
    outgoing["week"] = pd.to_numeric(outgoing.week, errors="coerce")
    if len(outgoing) != 64 or set(outgoing.week.dropna().astype(int)) != {6}:
        raise AssertionError(f"Expected frozen 64-row published Week 6 Sheet snapshot, got {len(outgoing)} rows / {set(outgoing.week)}")

    pbp = pd.read_csv(PBP_URL, compression="gzip", low_memory=False)
    roster = pd.read_csv(ROSTER_URL, low_memory=False)
    depth = pd.read_csv(DEPTH_URL, low_memory=False)
    snaps = pd.read_csv(SNAP_URL, low_memory=False)
    games = pd.read_csv(SCHEDULE_URL, low_memory=False)
    schedule = games.loc[(games.season.eq(SEASON)) & (games.week.eq(TARGET_WEEK)) & games.game_type.eq("REG")].copy()
    if len(schedule) != 14:
        raise AssertionError(f"Expected 14 Week {TARGET_WEEK} games plus four byes, got {len(schedule)}")
    reg = clean_pbp(pbp)
    team_log, rb_log, game_log = build_window_logs(reg, roster)
    pace, usage = build_pace_and_usage(reg, snaps, roster)
    model, dashboard, defense = build_model(depth, team_log, rb_log, schedule, outgoing)
    board = build_board(model, team_log, rb_log)
    weekly_archive, results, accuracy = grade_archive(archive, outgoing, rb_log)

    roles = depth.copy(); roles["dt"] = pd.to_datetime(roles.dt, utc=True)
    roles = roles.loc[roles.dt.eq(roles.dt.max()) & roles.pos_abb.astype(str).str.upper().eq("RB")].sort_values(["team", "pos_rank"])
    roles = roles.loc[~(
        roles.team.eq("NYG") & roles.player_name.map(norm_name).eq("najee harris")
    )].copy()
    roles["team_rb_order"] = roles.groupby("team").cumcount() + 1
    roles = roles.loc[roles.team_rb_order.le(3), ["team", "player_name", "gsis_id", "team_rb_order", "pos_slot", "pos_rank", "dt"]]
    roles["injury_status"] = roles.player_name.map(lambda x: INJURY.get(norm_name(x), ("No RB limitation listed in the early official Week 7 report; most Sunday reports incomplete", 1))[0])
    roles["depth_source"] = DEPTH_URL
    roles["data_state"] = "CURRENT DEPTH / EARLY WEEK 7 INJURY REPORT"

    archive_dir = ARCHIVE / "2026_week_06"
    archive_dir.mkdir(parents=True, exist_ok=True)
    outgoing.to_csv(archive_dir / "current_week_model.csv", index=False)
    model.to_csv(LIVE / "current_week_model.csv", index=False, float_format="%.6f")
    dashboard.to_csv(LIVE / "prime_rb_dashboard.csv", index=False, float_format="%.6f")
    defense.to_csv(LIVE / "defensive_freshness.csv", index=False, float_format="%.6f")
    board.to_csv(LIVE / "rb_window_opportunity_board.csv", index=False, float_format="%.6f")
    game_log.to_csv(LIVE / "rb_window_game_log_2026.csv", index=False, float_format="%.6f")
    weekly_archive.to_csv(LIVE / "weekly_archive_2026.csv", index=False, float_format="%.6f")
    results.to_csv(LIVE / "weekly_results_2026.csv", index=False, float_format="%.6f")
    accuracy.to_csv(LIVE / "model_accuracy_2026.csv", index=False, float_format="%.6f")
    usage.to_csv(LIVE / "rb_usage_leaders_2026.csv", index=False, float_format="%.6f")
    pace.to_csv(LIVE / "matchup_pace_2026.csv", index=False, float_format="%.6f")
    roles.to_csv(LIVE / "rb_roles_2026.csv", index=False)
    schedule[["game_id", "gameday", "gametime", "away_team", "home_team"]].to_csv(LIVE / "schedule_week_7.csv", index=False)

    status = pd.DataFrame([
        ("season", SEASON), ("current_target_week", TARGET_WEEK), ("latest_completed_week", COMPLETED_WEEK),
        ("last_model_refresh", REFRESH_DATE), ("next_scheduled_refresh", "2026-10-28 03:00 CT"),
        ("timezone", "America/Chicago"), ("scoring_version", "v0.5 production candidate + 2026 live calibration"),
        ("source_repo", "davidcantugtr/nfl-Rushing-Stats"),
        ("sharp_status", "STALE — last verified 32/32 Week 4 snapshot on 2026-10-03; not guessed"),
        ("public_defense_status", "2026 Weeks 1-6 complete; public Q1 rush-allowed rank used for Week 7"),
        ("validation_status", "PASS — 14 games + 4 byes / 28 reciprocal active teams / 64 RBs / 192 windows / no duplicates / role-share sanity"),
    ], columns=["key", "value"])
    status.to_csv(LIVE / "model_status.csv", index=False)

    top10 = dashboard.head(10)[["rank", "player", "team", "opponent", "exp_player_q1_yds", "v05_candidate_score", "prime_label", "injury_status"]].to_dict("records")
    latest_accuracy = accuracy.loc[accuracy.Week.eq(COMPLETED_WEEK)].iloc[0].to_dict()
    summary = {
        "target_week": TARGET_WEEK, "completed_week": COMPLETED_WEEK,
        "row_counts": {"current_week_model": len(model), "rb_window_board": len(board), "rb_window_game_log": len(game_log), "weekly_archive": len(weekly_archive), "weekly_results": len(results), "usage_leaders": len(usage), "matchup_pace": len(pace), "rb_roles": len(roles), "defense_monitor": len(defense)},
        "top10": top10, "latest_accuracy": latest_accuracy,
        "public_defense_worst": defense.sort_values("current_public_rush_def_rank", ascending=False).head(5)[["team", "current_q1_rush_yds_allowed_per_game", "current_public_rush_def_rank"]].to_dict("records"),
        "validation": {"schedule_games": len(schedule), "schedule_team_rows": 2 * len(schedule), "bye_teams": sorted(model.loc[model.opponent.eq("BYE"), "team"].unique().tolist()), "reciprocal_alignment": True, "model_unique": not model.duplicated(["team", "player"]).any(), "board_unique": not board.duplicated(["team", "player", "window"]).any(), "sharp_stale": True, "injury_reports_incomplete_for_sunday": True},
    }
    (LIVE / "weekly_refresh_summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    print(json.dumps(summary, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

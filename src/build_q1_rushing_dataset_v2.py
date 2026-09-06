from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd
import nflreadpy as nfl

SEASON = 2025
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "processed"
OUT.mkdir(parents=True, exist_ok=True)
EXPECTED = ROOT / "config" / "expected_team_q1_2025.csv"

# nflverse's current PBP encodes this aborted snap/fumble recovery as 0 rushing yards,
# while the official NFL/FootballDB first-quarter rushing split credits Daniel Jones
# with 3 net rushing yards on the play. The +3 correction reconciles both Jones
# (69 official Q1 rush yards) and Indianapolis (517 official Q1 rush yards).
OFFICIAL_STAT_CORRECTIONS = {
    ("2025_07_IND_LAC", 641.0): 3.0,
}


def pdframe(x):
    return x.to_pandas() if hasattr(x, "to_pandas") else pd.DataFrame(x)


def flag(s):
    return pd.to_numeric(s, errors="coerce").fillna(0).eq(1)


def summarize(s: pd.Series, prefix: str, thresholds: list[int]) -> dict:
    s = pd.to_numeric(s, errors="coerce").fillna(0).astype(float)
    out = {
        f"{prefix}_games": int(len(s)),
        f"{prefix}_mean": float(s.mean()),
        f"{prefix}_median": float(s.median()),
        f"{prefix}_std": float(s.std(ddof=0)),
        f"{prefix}_min": float(s.min()),
        f"{prefix}_p25": float(s.quantile(.25)),
        f"{prefix}_p75": float(s.quantile(.75)),
        f"{prefix}_max": float(s.max()),
    }
    for t in thresholds:
        out[f"{prefix}_hit_{t}"] = float((s >= t).mean())
    return out


def main() -> int:
    pbp = pdframe(nfl.load_pbp(SEASON))
    rosters = pdframe(nfl.load_rosters(SEASON))
    reg = pbp.loc[pbp["season_type"].astype(str).eq("REG")].copy()

    games = reg[["game_id", "week", "home_team", "away_team"]].drop_duplicates("game_id")
    if len(games) != 272:
        raise AssertionError(f"Expected 272 regular-season games, got {len(games)}")

    home = games.assign(team=games.home_team, opponent=games.away_team, site="H")
    away = games.assign(team=games.away_team, opponent=games.home_team, site="A")
    skeleton = pd.concat([home, away], ignore_index=True)[["game_id", "week", "team", "opponent", "site"]]
    if len(skeleton) != 544:
        raise AssertionError(f"Expected 544 team-game rows, got {len(skeleton)}")

    rush = flag(reg["rush_attempt"])
    q1 = reg.loc[reg["qtr"].fillna(0).eq(1) & rush].copy()
    if "no_play" in q1:
        q1 = q1.loc[~flag(q1["no_play"])].copy()
    # Two-point conversions are scoring attempts but are not official rushing attempts/yards.
    if "two_point_attempt" in q1:
        q1 = q1.loc[~flag(q1["two_point_attempt"])].copy()

    q1["rushing_yards"] = pd.to_numeric(q1["rushing_yards"], errors="coerce").fillna(0.0)
    for (game_id, play_id), corrected_yds in OFFICIAL_STAT_CORRECTIONS.items():
        m = q1["game_id"].eq(game_id) & pd.to_numeric(q1["play_id"], errors="coerce").eq(play_id)
        if m.sum() != 1:
            raise AssertionError(f"Expected one correction target for {game_id}/{play_id}, found {m.sum()}")
        q1.loc[m, "rushing_yards"] = corrected_yds

    actual = q1.groupby(["game_id", "posteam"], as_index=False).agg(
        q1_rush_att=("rush_attempt", "size"), q1_rush_yds=("rushing_yards", "sum")
    ).rename(columns={"posteam": "team"})

    team_log = skeleton.merge(actual, on=["game_id", "team"], how="left")
    team_log[["q1_rush_att", "q1_rush_yds"]] = team_log[["q1_rush_att", "q1_rush_yds"]].fillna(0)
    team_log["q1_rush_att"] = team_log.q1_rush_att.astype(int)
    team_log["q1_ypc"] = np.where(team_log.q1_rush_att > 0, team_log.q1_rush_yds / team_log.q1_rush_att, 0.0)
    for t in [20, 25, 30, 35, 40]:
        team_log[f"hit_{t}"] = (team_log.q1_rush_yds >= t).astype(int)

    team_summary = pd.DataFrame([
        {"team": tm, **summarize(g.q1_rush_yds, "q1_rush_yds", [20,25,30,35,40]),
         "q1_rush_att_mean": g.q1_rush_att.mean(), "q1_rush_att_median": g.q1_rush_att.median(),
         "q1_rush_att_max": g.q1_rush_att.max()}
        for tm, g in team_log.groupby("team")
    ])

    allowed = team_log.rename(columns={"team":"offense", "opponent":"defense", "q1_rush_yds":"q1_rush_yds_allowed", "q1_rush_att":"q1_rush_att_allowed"})
    defense_summary = pd.DataFrame([
        {"defense": tm, **summarize(g.q1_rush_yds_allowed, "q1_rush_yds_allowed", [20,25,30,35,40]),
         "q1_rush_att_allowed_mean": g.q1_rush_att_allowed.mean(),
         "q1_rush_att_allowed_median": g.q1_rush_att_allowed.median(),
         "q1_rush_att_allowed_max": g.q1_rush_att_allowed.max()}
        for tm, g in allowed.groupby("defense")
    ])

    # Position lookup for RB/FB layer.
    id_col = "gsis_id" if "gsis_id" in rosters.columns else "player_id"
    name_col = next(c for c in ["full_name","player_name","football_name"] if c in rosters.columns)
    pos = rosters[[id_col, name_col, "position"]].dropna(subset=[id_col]).copy()
    pos[id_col] = pos[id_col].astype(str)
    pos = pos.drop_duplicates(id_col, keep="last")

    all_rush = reg.loc[rush].copy()
    if "no_play" in all_rush:
        all_rush = all_rush.loc[~flag(all_rush["no_play"])].copy()
    if "two_point_attempt" in all_rush:
        all_rush = all_rush.loc[~flag(all_rush["two_point_attempt"])].copy()
    all_rush["rusher_player_id"] = all_rush.rusher_player_id.astype(str)
    rusher_games = all_rush[["game_id","week","posteam","defteam","rusher_player_id","rusher_player_name"]].drop_duplicates(["game_id","rusher_player_id"])
    rusher_games = rusher_games.merge(pos, left_on="rusher_player_id", right_on=id_col, how="left")
    rusher_games["position"] = rusher_games.position.fillna("UNKNOWN").str.upper()
    unknown = rusher_games.loc[rusher_games.position.eq("UNKNOWN")].copy()

    rb_games = rusher_games.loc[rusher_games.position.isin(["RB","FB"])].rename(columns={
        "posteam":"team", "defteam":"opponent", "rusher_player_id":"player_id",
        "rusher_player_name":"pbp_player_name", name_col:"player_name"
    })
    q1_ids = q1.copy()
    q1_ids["rusher_player_id"] = q1_ids.rusher_player_id.astype(str)
    rb_actual = q1_ids.groupby(["game_id","rusher_player_id"], as_index=False).agg(
        q1_rush_att=("rush_attempt","size"), q1_rush_yds=("rushing_yards","sum")
    ).rename(columns={"rusher_player_id":"player_id"})
    rb_log = rb_games.merge(rb_actual, on=["game_id","player_id"], how="left")
    rb_log[["q1_rush_att","q1_rush_yds"]] = rb_log[["q1_rush_att","q1_rush_yds"]].fillna(0)
    rb_log["q1_rush_att"] = rb_log.q1_rush_att.astype(int)
    rb_log["q1_ypc"] = np.where(rb_log.q1_rush_att > 0, rb_log.q1_rush_yds / rb_log.q1_rush_att, 0.0)
    team_keys = team_log[["game_id","team","site","q1_rush_att","q1_rush_yds"]].rename(columns={"q1_rush_att":"team_q1_rush_att","q1_rush_yds":"team_q1_rush_yds"})
    rb_log = rb_log.merge(team_keys, on=["game_id","team"], how="left")
    rb_log["rb_q1_carry_share"] = np.where(rb_log.team_q1_rush_att > 0, rb_log.q1_rush_att / rb_log.team_q1_rush_att, 0.0)
    rb_log["rb_q1_yard_share"] = np.where(rb_log.team_q1_rush_yds != 0, rb_log.q1_rush_yds / rb_log.team_q1_rush_yds, 0.0)
    for t in [15,20,25]:
        rb_log[f"hit_{t}"] = (rb_log.q1_rush_yds >= t).astype(int)

    rb_summary = pd.DataFrame([
        {"player_id": pid, "player_name": name, "team": tm,
         **summarize(g.q1_rush_yds, "q1_rush_yds", [15,20,25]),
         "q1_rush_att_mean": g.q1_rush_att.mean(), "q1_rush_att_median": g.q1_rush_att.median(),
         "q1_rush_att_max": g.q1_rush_att.max(), "rb_q1_carry_share_mean": g.rb_q1_carry_share.mean(),
         "rb_q1_yard_share_mean": g.rb_q1_yard_share.mean()}
        for (pid,name,tm), g in rb_log.groupby(["player_id","player_name","team"])
    ])

    expected = pd.read_csv(EXPECTED)
    totals = team_log.groupby("team", as_index=False).agg(actual_q1_rush_att=("q1_rush_att","sum"), actual_q1_rush_yds=("q1_rush_yds","sum"))
    recon = expected.merge(totals, on="team", how="outer")
    recon["att_diff"] = recon.actual_q1_rush_att - recon.expected_q1_rush_att
    recon["yds_diff"] = recon.actual_q1_rush_yds - recon.expected_q1_rush_yds
    recon["pass"] = recon.att_diff.eq(0) & recon.yds_diff.eq(0)

    team_log.sort_values(["week","game_id","team"]).to_csv(OUT/"team_q1_game_log_2025.csv", index=False)
    team_summary.sort_values("q1_rush_yds_mean", ascending=False).to_csv(OUT/"team_q1_summary_2025.csv", index=False)
    defense_summary.sort_values("q1_rush_yds_allowed_mean", ascending=False).to_csv(OUT/"team_q1_defense_summary_2025.csv", index=False)
    rb_log.sort_values(["week","game_id","team","q1_rush_yds"], ascending=[True,True,True,False]).to_csv(OUT/"rb_q1_game_log_2025.csv", index=False)
    rb_summary.sort_values("q1_rush_yds_mean", ascending=False).to_csv(OUT/"rb_q1_summary_2025.csv", index=False)
    recon.sort_values("team").to_csv(OUT/"reconciliation_2025.csv", index=False)
    unknown.to_csv(OUT/"unknown_rusher_positions_2025.csv", index=False)

    if not recon["pass"].all():
        print(recon.loc[~recon["pass"]].to_string(index=False), file=sys.stderr)
        return 2

    print(f"PASS: 272 games / {len(team_log)} team-game rows / {len(rb_log)} RB-FB rows")
    print(f"PASS: league Q1 rushing yards = {team_log.q1_rush_yds.sum():.0f}")
    print(f"PASS: all 32 team aggregate attempts and yards reconcile to FootballDB")
    print(f"INFO: unknown rusher-position game records = {len(unknown)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

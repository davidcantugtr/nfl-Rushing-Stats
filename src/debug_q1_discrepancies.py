import pandas as pd
import nflreadpy as nfl

TEAMS = {"CIN", "IND", "NYJ", "WAS"}

pbp = nfl.load_pbp(2025).to_pandas()
reg = pbp.loc[pbp["season_type"].astype(str).eq("REG")].copy()
rush = reg["rush_attempt"].fillna(0).astype(float).eq(1)
q1 = reg.loc[reg["qtr"].fillna(0).astype(float).eq(1) & rush & reg["posteam"].isin(TEAMS)].copy()
if "no_play" in q1.columns:
    q1 = q1.loc[~q1["no_play"].fillna(0).astype(float).eq(1)].copy()
q1["rushing_yards"] = pd.to_numeric(q1["rushing_yards"], errors="coerce").fillna(0)

print("=== TEAM TOTALS ===")
print(q1.groupby("posteam").agg(att=("rush_attempt","size"), yds=("rushing_yards","sum")).to_string())

print("\n=== RUSHER TOTALS ===")
print(q1.groupby(["posteam","rusher_player_name"], dropna=False).agg(att=("rush_attempt","size"), yds=("rushing_yards","sum")).to_string())

cols = [c for c in ["game_id","week","posteam","defteam","play_id","rusher_player_name","rushing_yards","qb_kneel","qb_scramble","play_type","desc"] if c in q1.columns]

print("\n=== ZERO-YARD RUSHES: CIN / NYJ / WAS ===")
print(q1.loc[q1["posteam"].isin({"CIN","NYJ","WAS"}) & q1["rushing_yards"].eq(0), cols].to_string(index=False))

print("\n=== INDIANAPOLIS Q1 RUSHES BY GAME ===")
ind = q1.loc[q1["posteam"].eq("IND")].copy()
print(ind.groupby(["week","game_id","defteam"]).agg(att=("rush_attempt","size"), yds=("rushing_yards","sum")).to_string())
print("\n=== INDIANAPOLIS NEGATIVE/ZERO Q1 RUSHES ===")
print(ind.loc[ind["rushing_yards"].le(0), cols].to_string(index=False))

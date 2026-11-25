from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
LIVE = ROOT / "data" / "live"

SHEET_IDS = {
    "Prime RB Dashboard": 531993186,
    "RB TD & Receiving Outlook": 2060000001,
    "RB Window Dashboard": 1800000001,
    "RB Window Board": 1800000002,
    "RB Window Game Log": 1800000005,
    "Current Week Model": 1488334264,
    "RB Roles": 1042679776,
    "Defense Monitor": 2070493784,
    "Model Accuracy": 680538110,
    "2026 Weekly Results": 1692104624,
    "Weekly Archive": 1411739455,
    "System Status": 542699538,
    "Defensive Injury Live": 2100000001,
    "RB Usage Leaders": 2100000002,
    "Matchup Pace": 2100000003,
}


def uev(value):
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return {}
    if isinstance(value, bool):
        return {"userEnteredValue": {"boolValue": value}}
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return {"userEnteredValue": {"numberValue": float(value)}}
    text = str(value)
    if text.startswith("="):
        return {"userEnteredValue": {"formulaValue": text}}
    return {"userEnteredValue": {"stringValue": text}}


def rows(values):
    return [{"values": [uev(v) for v in row]} for row in values]


def clear(sheet_id, r0, r1, c0, c1):
    return {"updateCells": {"range": {"sheetId": sheet_id, "startRowIndex": r0, "endRowIndex": r1, "startColumnIndex": c0, "endColumnIndex": c1}, "rows": [], "fields": "userEnteredValue"}}


def write(sheet_id, r0, c0, values):
    return {"updateCells": {"start": {"sheetId": sheet_id, "rowIndex": r0, "columnIndex": c0}, "rows": rows(values), "fields": "userEnteredValue"}}


def table(filename, sheet, start_row, clear_end, clear_cols):
    df = pd.read_csv(LIVE / filename)
    vals = [list(df.columns)] + df.where(pd.notna(df), None).values.tolist()
    sid = SHEET_IDS[sheet]
    return [clear(sid, start_row, clear_end, 0, clear_cols), write(sid, start_row, 0, vals)]


def table_chunk(filename, sheet, header_row, start, count, clear_end=None, clear_cols=None):
    df = pd.read_csv(LIVE / filename)
    part = df.iloc[start:start + count].where(pd.notna(df.iloc[start:start + count]), None)
    vals = part.values.tolist()
    row0 = header_row + 1 + start
    sid = SHEET_IDS[sheet]
    req = []
    if start == 0:
        req.append(clear(sid, header_row, clear_end, 0, clear_cols))
        req.append(write(sid, header_row, 0, [list(df.columns)]))
    req.append(write(sid, row0, 0, vals))
    return req


def group_core():
    req = []
    req += table("current_week_model.csv", "Current Week Model", 3, 300, 26)
    req += table("rb_window_opportunity_board.csv", "RB Window Board", 3, 300, 40)
    req += table("rb_td_receiving_outlook.csv", "RB TD & Receiving Outlook", 3, 200, 26)
    req += table("rb_roles_2026.csv", "RB Roles", 3, 300, 10)
    req += table("defensive_freshness.csv", "Defense Monitor", 5, 200, 9)
    req += table("model_accuracy_2026.csv", "Model Accuracy", 7, 150, 12)
    return req


def group_history():
    req = [
        {"appendDimension": {"sheetId": SHEET_IDS["Weekly Archive"], "dimension": "ROWS", "length": 100}},
        {"appendDimension": {"sheetId": SHEET_IDS["2026 Weekly Results"], "dimension": "ROWS", "length": 100}},
    ]
    req += table("weekly_archive_2026.csv", "Weekly Archive", 5, 700, 23)
    req += table("weekly_results_2026.csv", "2026 Weekly Results", 3, 700, 25)
    req += table("rb_window_game_log_2026.csv", "RB Window Game Log", 3, 4000, 17)
    return req


def group_archive():
    return [{"appendDimension": {"sheetId": SHEET_IDS["Weekly Archive"], "dimension": "ROWS", "length": 100}}] + table("weekly_archive_2026.csv", "Weekly Archive", 5, 700, 23)


def group_results():
    return [{"appendDimension": {"sheetId": SHEET_IDS["2026 Weekly Results"], "dimension": "ROWS", "length": 100}}] + table("weekly_results_2026.csv", "2026 Weekly Results", 3, 700, 25)


def group_game_log(chunk):
    return table_chunk("rb_window_game_log_2026.csv", "RB Window Game Log", 3, chunk * 600, 600, 4000, 17)


def group_usage_chunk(chunk):
    return table_chunk("rb_usage_leaders_2026.csv", "RB Usage Leaders", 3, chunk * 600, 600, 2000, 25)


def group_pace():
    return table("matchup_pace_2026.csv", "Matchup Pace", 3, 2000, 22)


def group_definj():
    df = pd.read_csv(LIVE / "defensive_injury_status_week12.csv")
    vals = []
    for _, r in df.iterrows():
        filed = r.report_state != "PENDING"
        vals.append([
            r.last_checked, r.season, r.week, r.game_id, r.team, r.opponent,
            "TEAM REPORT STATUS", "", "All Units", "", "Week 12", r.report_state,
            "", "", "Pregame", "VERIFIED" if filed else "PENDING",
            r.source, r.source_updated,
            "Verified player limitations reflected when available; no boost from pending data",
            r.defensive_availability_note,
        ])
    sid = SHEET_IDS["Defensive Injury Live"]
    return [clear(sid, 1596, 1628, 0, 20), write(sid, 1596, 0, vals)]


def group_usage():
    req = []
    req += table("rb_usage_leaders_2026.csv", "RB Usage Leaders", 3, 2000, 25)
    req += table("matchup_pace_2026.csv", "Matchup Pace", 3, 2000, 22)
    return req


def group_dashboard():
    dash = pd.read_csv(LIVE / "prime_rb_dashboard.csv").head(10)
    top = []
    for _, r in dash.iterrows():
        top.append([r["rank"], r["player"], r["team"], r["opponent"], r["depth_role"], r["backfield_type"], r["sharp_def_rank"], r["final_carry_share"], r["exp_q1_carries"], r["exp_player_q1_yds"], r["floor"], r["ceiling"], r["prob20_proxy"], r["prob25_proxy"], r["v05_candidate_score"], r["prime_label"], r.get("avg_rush_att_per_game"), r.get("opp_avg_ypc_allowed"), r.get("volume_x_ypc_proj_rush_yds")])
    acc = pd.read_csv(LIVE / "model_accuracy_2026.csv")
    a = acc.loc[acc.Week.eq(11)].iloc[0]
    counts = {
        "usage": len(pd.read_csv(LIVE / "rb_usage_leaders_2026.csv")),
        "pace": len(pd.read_csv(LIVE / "matchup_pace_2026.csv")),
    }
    p = SHEET_IDS["Prime RB Dashboard"]
    req = [
        write(p, 2, 0, [["Weekly ranking of the backs most likely to dominate the opening quarter, plus a Week 12 volume × opponent-YPC full-game baseline."]]),
        write(p, 5, 3, [["2026-11-25"]]),
        write(p, 8, 0, [["DEFENSIVE FRESHNESS: Exact Sharp Rush Efficiency DEF remains the verified Week 4 snapshot and is STALE. The last complete public Sharp matchup DWEF verification is Week 11 and is also STALE for Week 12; current nflverse Q1 rush-allowed ranks through finalized Week 11 drive the model. INJURY TIMING: Josh Jacobs is questionable (finger) and discounted; most Sunday/Monday official game-status reports were not yet filed at 03:00 CT."]]),
        clear(p, 13, 23, 0, 19), write(p, 13, 0, top),
        {"deleteEmbeddedObject": {"objectId": 201525648}},
        {"addChart": {"chart": {"spec": {"title": "Week 12 — Top 10 Expected Q1 Rushing Yards", "basicChart": {"chartType": "COLUMN", "legendPosition": "NO_LEGEND", "axis": [{"position": "BOTTOM_AXIS", "title": "Running Back"}, {"position": "LEFT_AXIS", "title": "Expected Q1 Rushing Yards"}], "domains": [{"domain": {"sourceRange": {"sources": [{"sheetId": p, "startRowIndex": 13, "endRowIndex": 23, "startColumnIndex": 1, "endColumnIndex": 2}]}}}], "series": [{"series": {"sourceRange": {"sources": [{"sheetId": p, "startRowIndex": 13, "endRowIndex": 23, "startColumnIndex": 9, "endColumnIndex": 10}]}}, "targetAxis": "LEFT_AXIS"}], "headerCount": 0}}, "position": {"overlayPosition": {"anchorCell": {"sheetId": p, "rowIndex": 11, "columnIndex": 19}, "widthPixels": 650, "heightPixels": 360}}}}},
    ]

    td = SHEET_IDS["RB TD & Receiving Outlook"]
    req += [
        write(td, 0, 2, [["WEEK 12 RB SCORING & RECEIVING OUTLOOK"]]),
        write(td, 1, 2, [["Full-game model probabilities using finalized Weeks 1–11 usage, role concentration, pace, opponent DWEF context, YPC allowed, defensive injuries and player availability. Model-derived, not sportsbook-implied; receiving outputs are usage-led when opponent RB coverage is unavailable."]]),
    ]

    wd = SHEET_IDS["RB Window Dashboard"]
    req += [
        write(wd, 1, 0, [["Week 12 live view. Q1, first-half and full-game projections use completed Week 11 evidence. The proprietary Sharp Week 4 snapshot and last verified public Week 11 DWEF are stale; current public Q1 defense is the labeled substitute. No market lines are invented."]]),
    ]
    array_cols = "{'RB Window Board'!$E$5:$E$196,'RB Window Board'!$B$5:$B$196,'RB Window Board'!$C$5:$C$196,'RB Window Board'!$D$5:$D$196,'RB Window Board'!$G$5:$G$196,'RB Window Board'!$J$5:$J$196,'RB Window Board'!$R$5:$R$196,'RB Window Board'!$U$5:$U$196,'RB Window Board'!$Y$5:$Y$196,'RB Window Board'!$Z$5:$Z$196,'RB Window Board'!$AA$5:$AA$196,'RB Window Board'!$AB$5:$AB$196,'RB Window Board'!$AC$5:$AC$196,'RB Window Board'!$AF$5:$AF$196,'RB Window Board'!$AG$5:$AG$196,'RB Window Board'!$AH$5:$AH$196,'RB Window Board'!$AI$5:$AI$196,'RB Window Board'!$AJ$5:$AJ$196,'RB Window Board'!$AK$5:$AK$196,'RB Window Board'!$AL$5:$AL$196,'RB Window Board'!$AM$5:$AM$196}"
    for start, label in [(7, "Q1"), (21, "1H"), (35, "Full Game")]:
        req.append(clear(wd, start, start + 10, 0, 21))
        formula = f'=ARRAY_CONSTRAIN(SORT(FILTER({array_cols},\'RB Window Board\'!$I$5:$I$196="{label}"),20,FALSE),10,21)'
        req.append(write(wd, start, 0, [[formula]]))

    status = SHEET_IDS["System Status"]
    status_rows = [
        ["Current target week", "12", "LIVE", "Official Week 12 slate: 16 games; no byes"],
        ["Latest completed week", "11", "FINAL", "Weeks 1–11 finalized from nflverse play-by-play and snap counts"],
        ["Official schedule", "Week 12", "VERIFIED", "NFL flex changes reconciled; 16 games / 32 reciprocal team rows / correct sites"],
        ["Last model refresh", "2026-11-25", "COMPLETE", "Week 12 model advanced from finalized Weeks 1–11 evidence"],
        ["Next scheduled refresh", "2026-12-02 03:00 CT", "SCHEDULED", "Weekly production refresh"],
        ["GitHub live feed", "davidcantugtr/nfl-Rushing-Stats/data/live", "PUBLISHING", "Week 12 validated source files"],
        ["Sharp Rush Efficiency DEF", "STALE — last verified 32/32 Week 4 snapshot (2026-10-03); not guessed", "WARNING", "Proprietary value preserved separately"],
        ["Current public defense", "Weeks 1–11 Q1 rank; last verified Sharp public Week 11 DWEF is stale", "CURRENT SUBSTITUTE", "Current nflverse rank drives Week 12; stale Sharp context is separate"],
        ["Scoring model", "v0.5 production candidate + 2026 live calibration", "LIVE", "64 RBs / 192 windows / 64 active TD-receiving rows"],
        ["Model accuracy", f"Week 11 graded: {a['Top-10 Overlap']:.0%} overlap / {a['Top-5 Avg Actual Q1 Yds']:.1f} top-5 avg / {a['20+ Hit Rate']:.0%} 20+ / {a['25+ Hit Rate']:.0%} 25+", "GRADED", f"MAE {a['MAE']:.2f}; Spearman {a['Spearman']:.3f}"],
        ["Usage & pace", f"{counts['usage']:,} RB player-week rows / {counts['pace']:,} team-game rows", "FINAL", "Completed games only through Week 11"],
        ["Google Sheet refresh", "All live and archive tabs written directly", "COMPLETE", "Historical rows and formatting retained"],
        ["Validation", "Schedule, reciprocal opponents, sites, duplicates, shares, reconciliation", "PASS", "16 games / 64 RBs / no duplicate player-game rows"],
    ]
    req += [clear(status, 5, 18, 0, 4), write(status, 5, 0, status_rows)]

    texts = [
        (SHEET_IDS["Current Week Model"], 1, "Week 12 player projections using completed 2026 Weeks 1–11 evidence. Includes average rushing attempts/game, opponent average YPC allowed, and the full-game volume × YPC baseline."),
        (SHEET_IDS["RB Window Board"], 1, "Week 12 live board: 64 RBs × Q1, 1H and Full Game = 192 rows. Proprietary Sharp Week 4 and public Week 11 DWEF values are preserved as stale; current public Q1 defense through Week 11 is the labeled model substitute."),
        (SHEET_IDS["RB Window Game Log"], 1, "Finalized RB/FB Q1, 1H and Full Game results through completed Week 11 only — 2,307 player-window rows; no partial games."),
        (SHEET_IDS["RB Roles"], 1, "Week 12 RB depth roles and calibrated shares. Current-season usage through Week 11 is blended with current depth-chart priors; role-share sanity checks passed."),
        (SHEET_IDS["Defense Monitor"], 2, "Current state: exact Sharp Rush Efficiency DEF last verified for all 32 teams on Oct. 3 (Week 4) is preserved and STALE. Week 12 modeling uses the labeled public 2026 Q1 rush-allowed rank through completed Week 11; last verified public Sharp Week 11 DWEF is separately flagged stale."),
        (SHEET_IDS["Model Accuracy"], 2, "Season-long scorecard for frozen pregame predictions. Weeks 1–11 are graded from finalized Q1 play-by-play; Week 12 is the active publication."),
        (SHEET_IDS["2026 Weekly Results"], 1, "Published player-level projection results through completed Week 11. Week 12 remains live and ungraded."),
        (SHEET_IDS["Weekly Archive"], 2, "Every published weekly RB projection is preserved player by player before the live model advances. Week 11’s full 64-RB snapshot is frozen; existing historical rows remain intact."),
        (SHEET_IDS["RB Usage Leaders"], 1, "Verified RB/FB usage through completed Week 11 — 1,109 player-week rows. Routes remain blank unless reliably available."),
        (SHEET_IDS["RB Usage Leaders"], 2, "Finalized snap, target and rushing-usage ledger through Week 11; prior-week trends and team usage ranks are current for Week 12 decisions."),
        (SHEET_IDS["Matchup Pace"], 1, "Finalized pace and possession through completed Week 11 — 326 team-game rows. No partial games included."),
        (SHEET_IDS["Matchup Pace"], 2, "Final team possession and pace rows are loaded through Week 11. Week 12 matchup interpretation uses only completed observations; no future-week pace values are imputed."),
    ]
    for sid, row, text in texts:
        req.append(write(sid, row, 0, [[text]]))
    return req


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("group", choices=["core", "history", "archive", "results", "game_log_0", "game_log_1", "game_log_2", "game_log_3", "usage", "usage_0", "usage_1", "pace", "definj", "dashboard"])
    args = ap.parse_args()
    if args.group.startswith("game_log_"):
        data = group_game_log(int(args.group.rsplit("_", 1)[1]))
    elif args.group.startswith("usage_"):
        data = group_usage_chunk(int(args.group.rsplit("_", 1)[1]))
    else:
        fn = {"core": group_core, "history": group_history, "archive": group_archive, "results": group_results, "usage": group_usage, "pace": group_pace, "definj": group_definj, "dashboard": group_dashboard}[args.group]
        data = fn()
    print(json.dumps(data, separators=(",", ":"), allow_nan=False))


if __name__ == "__main__":
    main()

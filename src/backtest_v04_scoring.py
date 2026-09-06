from __future__ import annotations

import csv
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed"
RB_LOG = DATA / "rb_q1_game_log_2025.csv"
TEAM_LOG = DATA / "team_q1_game_log_2025.csv"

BASELINE_WEIGHTS = {
    "team_env": 0.30,
    "share": 0.25,
    "defense": 0.15,
    "history": 0.15,
    "consistency": 0.10,
    "confidence": 0.05,
}
COMPONENTS = list(BASELINE_WEIGHTS)
TRAIN_WEEKS = set(range(5, 13))
TEST_WEEKS = set(range(13, 19))
BACKTEST_WEEKS = set(range(5, 19))
RNG = np.random.default_rng(2306)


def read_csv(path: Path):
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def fnum(v, default=0.0):
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def mean(xs, default=0.0):
    return sum(xs) / len(xs) if xs else default


def clip(v, lo, hi):
    return max(lo, min(hi, v))


def average_ranks(values):
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i + 1
        while j < len(order) and values[order[j]] == values[order[i]]:
            j += 1
        avg = (i + 1 + j) / 2.0
        for k in range(i, j):
            ranks[order[k]] = avg
        i = j
    return ranks


def spearman(xs, ys):
    if len(xs) < 2:
        return 0.0
    rx = np.asarray(average_ranks(xs), dtype=float)
    ry = np.asarray(average_ranks(ys), dtype=float)
    if np.std(rx) == 0 or np.std(ry) == 0:
        return 0.0
    return float(np.corrcoef(rx, ry)[0, 1])


def percentile_rank_map(d: dict[str, float]):
    teams = sorted(d)
    vals = [d[t] for t in teams]
    ranks = average_ranks(vals)
    n = max(1, len(teams))
    return {t: (r / n) * 32.0 for t, r in zip(teams, ranks)}


def score(row, weights):
    return sum(weights[c] * row[c] for c in COMPONENTS)


def metrics(rows, weights, weeks):
    by_week = defaultdict(list)
    for r in rows:
        if r["week"] in weeks:
            rr = dict(r)
            rr["score"] = score(rr, weights)
            by_week[rr["week"]].append(rr)

    week_stats = []
    for wk in sorted(by_week):
        rr = by_week[wk]
        scores = [x["score"] for x in rr]
        yards = [x["actual_yards"] for x in rr]
        corr = spearman(scores, yards)
        pred = sorted(rr, key=lambda x: x["score"], reverse=True)
        actual = sorted(rr, key=lambda x: x["actual_yards"], reverse=True)
        top5 = pred[:5]
        top10 = pred[:10]
        true5 = actual[:5]
        true10 = actual[:10]
        denom5 = sum(x["actual_yards"] for x in true5) or 1.0
        denom10 = sum(x["actual_yards"] for x in true10) or 1.0
        set5 = {x["key"] for x in true5}
        set10 = {x["key"] for x in true10}
        week_stats.append({
            "week": wk,
            "spearman": corr,
            "top5_capture": sum(x["actual_yards"] for x in top5) / denom5,
            "top10_capture": sum(x["actual_yards"] for x in top10) / denom10,
            "top5_overlap": sum(x["key"] in set5 for x in top5) / 5.0,
            "top10_overlap": sum(x["key"] in set10 for x in top10) / 10.0,
            "top5_hit20": mean([x["actual_yards"] >= 20 for x in top5]),
            "top10_hit20": mean([x["actual_yards"] >= 20 for x in top10]),
            "top5_avg_yards": mean([x["actual_yards"] for x in top5]),
            "top10_avg_yards": mean([x["actual_yards"] for x in top10]),
            "top1_yards": top5[0]["actual_yards"] if top5 else 0.0,
        })
    if not week_stats:
        return {}, []
    agg = {k: mean([x[k] for x in week_stats]) for k in week_stats[0] if k != "week"}
    agg["weeks"] = len(week_stats)
    agg["objective"] = (
        0.45 * ((agg["spearman"] + 1.0) / 2.0)
        + 0.25 * agg["top5_capture"]
        + 0.15 * agg["top5_hit20"]
        + 0.15 * agg["top5_overlap"]
    )
    return agg, week_stats


def build_rows():
    team = read_csv(TEAM_LOG)
    rb = read_csv(RB_LOG)
    for r in team:
        r["week"] = int(r["week"])
        r["q1_rush_yds"] = fnum(r["q1_rush_yds"])
    for r in rb:
        r["week"] = int(r["week"])
        for col in ["q1_rush_yds", "rb_q1_carry_share", "rb_q1_yard_share"]:
            r[col] = fnum(r[col])

    teams = sorted({r["team"] for r in team})
    rows = []
    for wk in sorted(BACKTEST_WEEKS):
        prior_team = [r for r in team if r["week"] < wk]
        prior_rb = [r for r in rb if r["week"] < wk]

        off = defaultdict(list)
        allowed = defaultdict(list)
        for r in prior_team:
            off[r["team"]].append(r["q1_rush_yds"])
            allowed[r["opponent"]].append(r["q1_rush_yds"])
        league_off = mean([r["q1_rush_yds"] for r in prior_team], 28.5)
        off_mean = {t: mean(off[t], league_off) for t in teams}
        allowed_mean = {t: mean(allowed[t], league_off) for t in teams}
        def_rank = percentile_rank_map(allowed_mean)

        pstats = defaultdict(lambda: {"yds": [], "carry": [], "yardshare": [], "hit20": []})
        for r in prior_rb:
            k = r["player_id"]
            pstats[k]["yds"].append(r["q1_rush_yds"])
            pstats[k]["carry"].append(r["rb_q1_carry_share"])
            pstats[k]["yardshare"].append(r["rb_q1_yard_share"])
            pstats[k]["hit20"].append(r["q1_rush_yds"] >= 20)

        for r in rb:
            if r["week"] != wk:
                continue
            ps = pstats[r["player_id"]]
            games = len(ps["yds"])
            carry = mean(ps["carry"], 0.18)
            yardshare = mean(ps["yardshare"], carry)
            hist_mean = mean(ps["yds"], 0.0)
            hit20 = mean(ps["hit20"], 0.0)
            team_mean = off_mean[r["team"]]
            opp_allow = allowed_mean[r["opponent"]]
            dr = def_rank[r["opponent"]]
            factor = clip(1 + (dr - 16.5) * 0.009, 0.86, 1.14)
            team_proj = (team_mean * 0.55 + opp_allow * 0.45) * factor
            exp_yards = team_proj * yardshare

            env_score = clip((team_proj - 18.0) / 25.0 * 100.0, 0, 100)
            share_score = clip(carry / 0.70 * 100.0, 0, 100)
            defense_score = clip(dr / 32.0 * 100.0, 0, 100)
            hist_score = clip(hist_mean / 30.0 * 100.0, 0, 100) if games else 50.0
            consistency = hit20 * 100.0 if games else clip(exp_yards / 30.0 * 100.0, 0, 75)
            confidence = clip(games / 6.0 * 100.0, 0, 100)
            rows.append({
                "key": f'{r["game_id"]}|{r["player_id"]}',
                "game_id": r["game_id"],
                "week": wk,
                "team": r["team"],
                "opponent": r["opponent"],
                "player_id": r["player_id"],
                "player_name": r["player_name"],
                "prior_games": games,
                "actual_yards": r["q1_rush_yds"],
                "actual_hit20": int(r["q1_rush_yds"] >= 20),
                "expected_yards": exp_yards,
                "team_proj": team_proj,
                "carry_share": carry,
                "yard_share": yardshare,
                "def_rank_proxy": dr,
                "team_env": env_score,
                "share": share_score,
                "defense": defense_score,
                "history": hist_score,
                "consistency": consistency,
                "confidence": confidence,
            })
    return rows


def optimize(rows):
    base_train, _ = metrics(rows, BASELINE_WEIGHTS, TRAIN_WEEKS)
    best_w = dict(BASELINE_WEIGHTS)
    best_obj = base_train["objective"]
    # Keep all six signals positive and avoid pathological single-factor fits.
    for _ in range(12000):
        w = RNG.dirichlet(np.array([3.0, 3.0, 2.0, 2.0, 1.5, 1.0]))
        if max(w) > 0.45 or w[5] > 0.15:
            continue
        cand = dict(zip(COMPONENTS, w))
        m, _ = metrics(rows, cand, TRAIN_WEEKS)
        if m["objective"] > best_obj:
            best_obj = m["objective"]
            best_w = cand
    return best_w


def write_csv(path, fieldnames, rows):
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


def main():
    rows = build_rows()
    opt = optimize(rows)

    summary_rows = []
    weekly_rows = []
    for model_name, weights in [("v0.4 baseline", BASELINE_WEIGHTS), ("walk-forward optimized", opt)]:
        for split_name, weeks in [("train_wk5_12", TRAIN_WEEKS), ("test_wk13_18", TEST_WEEKS), ("all_wk5_18", BACKTEST_WEEKS)]:
            agg, weekly = metrics(rows, weights, weeks)
            summary_rows.append({"model": model_name, "split": split_name, **agg})
            for w in weekly:
                weekly_rows.append({"model": model_name, "split": split_name, **w})

    weight_rows = []
    for c in COMPONENTS:
        weight_rows.append({
            "component": c,
            "v04_weight": BASELINE_WEIGHTS[c],
            "optimized_weight": opt[c],
            "change": opt[c] - BASELINE_WEIGHTS[c],
        })

    # Out-of-sample prediction detail with both score versions.
    pred_rows = []
    for r in rows:
        if r["week"] not in TEST_WEEKS:
            continue
        pred_rows.append({
            "week": r["week"], "game_id": r["game_id"], "team": r["team"], "opponent": r["opponent"],
            "player_name": r["player_name"], "prior_games": r["prior_games"], "expected_yards": r["expected_yards"],
            "actual_yards": r["actual_yards"], "actual_hit20": r["actual_hit20"],
            "v04_score": score(r, BASELINE_WEIGHTS), "optimized_score": score(r, opt),
            "team_env": r["team_env"], "share": r["share"], "defense": r["defense"],
            "history": r["history"], "consistency": r["consistency"], "confidence": r["confidence"],
        })
    pred_rows.sort(key=lambda x: (x["week"], -x["optimized_score"]))

    # Score bands on out-of-sample test only.
    bucket_rows = []
    for name, weights in [("v0.4 baseline", BASELINE_WEIGHTS), ("walk-forward optimized", opt)]:
        test = [dict(r, score=score(r, weights)) for r in rows if r["week"] in TEST_WEEKS]
        bands = [(0, 40), (40, 50), (50, 60), (60, 70), (70, 80), (80, 101)]
        for lo, hi in bands:
            b = [r for r in test if lo <= r["score"] < hi]
            if not b:
                continue
            bucket_rows.append({
                "model": name, "score_band": f"{lo}-{hi-1}", "n": len(b),
                "avg_actual_yards": mean([r["actual_yards"] for r in b]),
                "median_actual_yards": float(np.median([r["actual_yards"] for r in b])),
                "hit20_rate": mean([r["actual_yards"] >= 20 for r in b]),
                "hit25_rate": mean([r["actual_yards"] >= 25 for r in b]),
            })

    write_csv(DATA / "backtest_v04_summary_2025.csv", list(summary_rows[0]), summary_rows)
    write_csv(DATA / "backtest_v04_weights_2025.csv", list(weight_rows[0]), weight_rows)
    write_csv(DATA / "backtest_v04_weekly_2025.csv", list(weekly_rows[0]), weekly_rows)
    write_csv(DATA / "backtest_v04_test_predictions_2025.csv", list(pred_rows[0]), pred_rows)
    write_csv(DATA / "backtest_v04_score_buckets_2025.csv", list(bucket_rows[0]), bucket_rows)

    base_test, _ = metrics(rows, BASELINE_WEIGHTS, TEST_WEEKS)
    opt_test, _ = metrics(rows, opt, TEST_WEEKS)
    print(f"Backtest rows: {len(rows)}; test rows: {len(pred_rows)}")
    print("Baseline test:", {k: round(v, 4) if isinstance(v, float) else v for k, v in base_test.items()})
    print("Optimized test:", {k: round(v, 4) if isinstance(v, float) else v for k, v in opt_test.items()})
    print("Optimized weights:", {k: round(v, 4) for k, v in opt.items()})


if __name__ == "__main__":
    main()
